#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-performance async large-scale inference script - optimized version
Supports dynamic concurrency control, resource monitoring, intelligent batch processing, etc.
"""

import asyncio
import json
import time
import os
import psutil
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from loguru import logger
import sys
from datetime import datetime
import traceback
from collections import deque
import statistics
import random

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_tools.async_openai_tool import AsyncOpenAITool, AsyncModelConfig, TaskResult
from utils.mongo_utils import MongoDBConnector
from prompt_kits.prompt_manager_final import get_manager

global SUCCESS_COUNT
SUCCESS_COUNT = 0
@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_io: Dict[str, int] = field(default_factory=dict)
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    throughput: float = 0.0
    error_rate: float = 0.0
    
    def update_response_time(self, response_time: float):
        """Update response time"""
        self.response_times.append(response_time)
        if len(self.response_times) > 0:
            self.throughput = 1.0 / statistics.mean(self.response_times)


class DynamicConcurrencyController:
    """Dynamic concurrency controller"""
    
    def __init__(
        self,
        initial_concurrency: int = 100,
        max_concurrency: int = 1000,
        min_concurrency: int = 10,
        adjustment_factor: float = 0.1,
        target_response_time: float = 2.0,
        stability_window: int = 10
    ):
        self.current_concurrency = initial_concurrency
        self.max_concurrency = max_concurrency
        self.min_concurrency = min_concurrency
        self.adjustment_factor = adjustment_factor
        self.target_response_time = target_response_time
        self.stability_window = stability_window
        
        self.response_times = deque(maxlen=stability_window)
        self.error_counts = deque(maxlen=stability_window)
        self.last_adjustment = time.time()
        self.adjustment_cooldown = 5.0  # Adjustment cooldown time
        
    def update_metrics(self, response_time: float, success: bool):
        """Update metrics"""
        self.response_times.append(response_time)
        self.error_counts.append(0 if success else 1)
    
    def should_adjust(self) -> bool:
        """Determine if concurrency should be adjusted"""
        return (
            time.time() - self.last_adjustment > self.adjustment_cooldown and
            len(self.response_times) >= self.stability_window
        )
    
    def adjust_concurrency(self) -> int:
        """Adjust concurrency"""
        if not self.should_adjust():
            return self.current_concurrency
        
        avg_response_time = statistics.mean(self.response_times)
        error_rate = sum(self.error_counts) / len(self.error_counts)
        
        # Adjust based on response time
        if avg_response_time > self.target_response_time * 1.5:
            # Response time too long, reduce concurrency
            adjustment = -int(self.current_concurrency * self.adjustment_factor)
        elif avg_response_time < self.target_response_time * 0.5 and error_rate < 0.1:
            # Response time short and error rate low, increase concurrency
            adjustment = int(self.current_concurrency * self.adjustment_factor)
        else:
            adjustment = 0
        
        # Adjust based on error rate
        if error_rate > 0.2:
            adjustment = min(adjustment, -1)
        elif error_rate < 0.05:
            adjustment = max(adjustment, 1)
        
        # Apply adjustment
        new_concurrency = max(
            self.min_concurrency,
            min(self.max_concurrency, self.current_concurrency + adjustment)
        )
        
        if new_concurrency != self.current_concurrency:
            logger.info(
                f"Adjusting concurrency: {self.current_concurrency} -> {new_concurrency} "
                f"(response time: {avg_response_time:.2f}s, error rate: {error_rate:.2%})"
            )
            # self.current_concurrency = new_concurrency
            self.last_adjustment = time.time()
        
        return self.current_concurrency


class ResourceMonitor:
    """Resource monitor"""
    
    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self.last_check = time.time()
        self.metrics = PerformanceMetrics()
        
    def get_system_metrics(self) -> Dict[str, float]:
        """Get system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Network IO
            net_io = psutil.net_io_counters()
            
            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_available": memory.available / (1024**3),  # GB
                "network_sent": net_io.bytes_sent,
                "network_recv": net_io.bytes_recv
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def should_throttle(self) -> bool:
        """Determine if throttling should be applied"""
        metrics = self.get_system_metrics()
        
        # CPU usage too high
        if metrics.get("cpu_usage", 0) > 90:
            return True
        
        # Memory usage too high
        if metrics.get("memory_usage", 0) > 90:
            return True
        
        return False


class SmartBatchProcessor:
    """Smart batch processor"""
    
    def __init__(self, base_batch_size: int = 50):
        self.base_batch_size = base_batch_size
        self.current_batch_size = base_batch_size
        self.batch_performance = deque(maxlen=20)
        
    def calculate_optimal_batch_size(self, avg_response_time: float, concurrency: int) -> int:
        """Calculate optimal batch size"""
        # Dynamically adjust batch size based on response time and concurrency
        if avg_response_time > 5.0:
            # Response time long, reduce batch size
            multiplier = 0.7
        elif avg_response_time < 1.0:
            # Response time short, increase batch size
            multiplier = 1.3
        else:
            multiplier = 1.0
        
        # Consider concurrency
        concurrency_factor = min(2.0, concurrency / 10.0)
        
        new_batch_size = int(self.base_batch_size * multiplier * concurrency_factor)
        new_batch_size = max(5, min(200, new_batch_size))  # Limit range
        
        return new_batch_size
    
    def update_batch_performance(self, batch_size: int, processing_time: float, success_rate: float):
        """Update batch processing performance"""
        self.batch_performance.append({
            "batch_size": batch_size,
            "processing_time": processing_time,
            "success_rate": success_rate,
            "efficiency": batch_size / processing_time if processing_time > 0 else 0
        })
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size"""
        if len(self.batch_performance) < 5:
            return self.current_batch_size
        
        # Find the most efficient batch size
        best_batch = max(self.batch_performance, key=lambda x: x["efficiency"])
        self.current_batch_size = best_batch["batch_size"]
        
        return self.current_batch_size


class HighPerformanceAsyncInsLLMJudge:
    """High-performance asynchronous Instagram LLM inferencer"""
    
    def __init__(
        self,
        mongo_host: str = "localhost",
        mongo_port: int = 27017,
        db_name: str = "INS_DB",
        collection_name: str = "f_ins_captions",
        provider: str = "seed",
        model: str = "doubao-seed-1-6-lite-251015",
        initial_concurrency: int = 100,
        max_concurrency: int = 1000,
        base_batch_size: int = 50,
        request_timeout: float = 120.0,
        max_retries: int = 5,
        context: str = "I am a 27-year-old male actor",
        enable_dynamic_scaling: bool = True,
        enable_resource_monitoring: bool = True,
        max_retry_rounds: int = 3
    ):
        """
        Initialize high-performance asynchronous inferencer
        
        Args:
            enable_dynamic_scaling: Whether to enable dynamic scaling
            enable_resource_monitoring: Whether to enable resource monitoring
        """
        # Initialize MongoDB connection
        self.mongo = MongoDBConnector(
            host=mongo_host,
            port=mongo_port,
            db_name=db_name
        )
        
        if not self.mongo.connect():
            raise ConnectionError("Unable to connect to MongoDB database")
        
        self.collection_name = collection_name
        self.context = context
        self.enable_dynamic_scaling = enable_dynamic_scaling
        self.enable_resource_monitoring = enable_resource_monitoring
        self.max_retry_rounds = max_retry_rounds
        
        # Failed retry queue
        self.retry_queue = []
        self.processed_items = set()  # Processed item IDs, avoid duplicate processing
        
        # Initialize asynchronous LLM tool
        self.llm_tool = AsyncOpenAITool(
            provider=provider,
            model=model,
            max_concurrent_requests=initial_concurrency,
            request_timeout=request_timeout
        )
        
        # Set model configuration
        self.model_config = AsyncModelConfig(
            name=model,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.9,
            batch_size=base_batch_size,
            max_retries=max_retries,
            request_timeout=request_timeout
        )
        
        # Initialize prompt manager
        self.prompt_manager = get_manager()
        
        # Get system prompt
        self.system_prompt = self.prompt_manager.get(
            "infer", "zh", context=self.context
        )
        
        # Initialize controller and monitor
        self.concurrency_controller = DynamicConcurrencyController(
            initial_concurrency=initial_concurrency,
            max_concurrency=max_concurrency
        )
        
        self.resource_monitor = ResourceMonitor() if enable_resource_monitoring else None
        self.batch_processor = SmartBatchProcessor(base_batch_size)
        self.model = model
        # Statistics
        self.stats = {
            "total_items": 0,
            "processed_items": 0,
            "successful_items": 0,
            "failed_items": 0,
            "start_time": 0.0,
            "end_time": 0.0,
            "performance_metrics": PerformanceMetrics()
        }
        
        logger.info(f"High-performance asynchronous inferencer initialized - Model: {model}, Initial concurrency: {initial_concurrency}")
    
    async def infer_privacy_single(
        self, 
        item: Dict[str, Any], 
        task_id: Optional[str] = None
    ) -> TaskResult:
        """Privacy inference for a single item"""
        global SUCCESS_COUNT
        try:
            # Check if already processed
            if 'infer_privacy' in item and item['infer_privacy']:
                return TaskResult(
                    success=True,
                    result=item['infer_privacy'],
                    task_id=task_id
                )
            
            # Build messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": item['text']}
            ]
            
            # Call LLM
            result = await self.llm_tool.async_chat_completion(
                messages=messages,
                model=self.model_config.name,
                config=self.model_config,
                task_id=task_id,
            )
            
            if result.success:
                # Parse JSON result
                try:
                    logger.info(f"{task_id} privacy inference successful: {result.result.choices[0].message.content}")
                    json_result = json.loads(result.result.choices[0].message.content)
                    result.result = json_result
                    SUCCESS_COUNT += 1
                    logger.info(f"Successful inference count: {SUCCESS_COUNT}")
                except json.JSONDecodeError as e:
                    logger.warning(f"{task_id} message: {messages[-1]}")
                    logger.warning(f"{task_id} privacy inference successful: {result.result.choices[0].message.content}")
                    logger.error(f"{task_id} JSON parsing failed: {e}")
                    result.success = False
                    result.error = f"JSON parsing failed: {e}"
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed (task {task_id}): {e}")
            return TaskResult(
                success=False,
                error=str(e),
                task_id=task_id
            )
    
    async def process_batch_with_optimization(
        self,
        items: List[Dict[str, Any]],
        update_db: bool = True,
        retry_round: int = 0
    ) -> List[TaskResult]:
        """
        Optimized batch processing
        
        Args:
            items: List of data items
            update_db: Whether to update database
            retry_round: Retry round
            
        Returns:
            List of TaskResult
        """
        batch_start_time = time.time()
        
        # Filter already processed items
        new_items = []
        for item in items:
            item_id = item.get('_id')
            if item_id not in self.processed_items:
                new_items.append(item)
            else:
                logger.debug(f"Skipping already processed item: {item_id}")
        
        if not new_items:
            logger.info(f"All items in batch {retry_round} have been processed")
            return []
        
        logger.info(f"Processing batch {retry_round}: {len(new_items)} items")
        
        # Dynamically adjust concurrency
        if self.enable_dynamic_scaling:
            current_concurrency = self.concurrency_controller.adjust_concurrency()
            # Update LLM tool concurrency limit
            self.llm_tool.max_concurrent_requests = current_concurrency
            self.llm_tool.semaphore = asyncio.Semaphore(current_concurrency)
        
        # Resource monitoring and throttling
        if self.enable_resource_monitoring and self.resource_monitor:
            if self.resource_monitor.should_throttle():
                logger.warning("System resource usage too high, throttling processing")
                # await asyncio.sleep(1.0)
        
        # Create tasks
        tasks = []
        for i, item in enumerate(new_items):
            task_id = f"batch_task_{retry_round}_{i}_{item.get('_id', 'unknown')}"
            task = self.infer_privacy_single(item, task_id)
            tasks.append((task, item))
        
        # Execute concurrently
        results = []
        task_coros = [task for task, _ in tasks]
        batch_items = [item for _, item in tasks]
        
        batch_results = await asyncio.gather(*task_coros, return_exceptions=True)
        
        # Process results
        successful_count = 0
        failed_items = []  # Failed items for retry
        
        for i, (result, item) in enumerate(zip(batch_results, batch_items)):
            if isinstance(result, Exception):
                result = TaskResult(
                    success=False,
                    error=str(result),
                    task_id=f"batch_task_{retry_round}_{i}"
                )
            
            results.append(result)
            
            # Mark as processed
            item_id = item.get('_id')
            self.processed_items.add(item_id)
            
            if result.success:
                # Update database
                if update_db:
                    try:
                        item['infer_privacy'] = result.result
                        item['infer_model'] = self.model
                        self.mongo.update_one_data(self.collection_name, item)
                        successful_count += 1
                    except Exception as e:
                        logger.error(f"Database update failed: {e}")
                        # Database update failure also joins retry queue
                        failed_items.append(item)
            else:
                # Failed items join retry queue
                failed_items.append(item)
                logger.warning(f"Item {item_id} processing failed: {result.error}")
            
            # Update performance metrics
            self.stats["performance_metrics"].update_response_time(result.execution_time)
            
            # Update concurrency controller metrics
            if self.enable_dynamic_scaling:
                self.concurrency_controller.update_metrics(
                    result.execution_time, result.success
                )
        
        # Add failed items to retry queue
        if failed_items and retry_round < self.max_retry_rounds:
            self.retry_queue.extend(failed_items)
            logger.info(f"Added {len(failed_items)} failed items to retry queue")
        
        # Update batch processing performance
        batch_time = time.time() - batch_start_time
        success_rate = successful_count / len(new_items) if new_items else 0
        self.batch_processor.update_batch_performance(
            len(new_items), batch_time, success_rate
        )
        
        return results
    
    async def process_all_data_optimized(
        self,
        query: Optional[Dict] = None,
        limit: Optional[int] = None,
        update_db: bool = True
    ) -> Dict[str, Any]:
        """
        Optimized full data processing
        
        Args:
            query: Query conditions
            limit: Limit count
            update_db: Whether to update database
            
        Returns:
            Processing statistics
        """
        logger.info("Starting optimized full data processing")
        
        # Reset statistics
        self.stats = {
            "total_items": 0,
            "processed_items": 0,
            "successful_items": 0,
            "failed_items": 0,
            "start_time": time.time(),
            "end_time": 0.0,
            "performance_metrics": PerformanceMetrics()
        }
        
        # Read data
        data = self.mongo.read_data(
            self.collection_name, 
            query=query, 
            limit=limit
        )
        random.shuffle(data)
        
        self.stats["total_items"] = len(data)
        logger.info(f"Read {len(data)} records")
        
        if not data:
            logger.warning("No data to process")
            return self.stats
        
        # Process in batches
        batch_size = self.batch_processor.get_optimal_batch_size()
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        logger.info(f"Using batch size: {batch_size}, total batches: {total_batches}")
        
        # Process initial data
        for batch_idx in range(0, len(data), batch_size):
            batch_items = data[batch_idx:batch_idx + batch_size]
            current_batch = (batch_idx // batch_size) + 1
            
            logger.info(f"Processing batch {current_batch}/{total_batches} ({len(batch_items)} items)")
            
            # Process batch
            batch_results = await self.process_batch_with_optimization(
                batch_items, update_db=update_db, retry_round=0
            )
            
            # Update statistics
            for result in batch_results:
                self.stats["processed_items"] += 1
                if result.success:
                    self.stats["successful_items"] += 1
                else:
                    self.stats["failed_items"] += 1
            
            # Output progress
            progress = (current_batch / total_batches) * 100
            success_rate = self.stats["successful_items"] / self.stats["processed_items"] if self.stats["processed_items"] > 0 else 0
            
            logger.info(
                f"Batch {current_batch}/{total_batches} completed ({progress:.1f}%) | "
                f"Success: {self.stats['successful_items']} | "
                f"Failed: {self.stats['failed_items']} | "
                f"Success rate: {success_rate:.2%}"
            )
        
        # Process retry queue
        retry_round = 1
        while self.retry_queue and retry_round <= self.max_retry_rounds:
            retry_items = self.retry_queue.copy()
            self.retry_queue.clear()
            
            logger.info(f"Starting retry round {retry_round}, retry items: {len(retry_items)}")
            
            # Process retry items in batches
            for batch_idx in range(0, len(retry_items), batch_size):
                batch_items = retry_items[batch_idx:batch_idx + batch_size]
                current_batch = (batch_idx // batch_size) + 1
                total_retry_batches = (len(retry_items) + batch_size - 1) // batch_size
                
                logger.info(f"Retry batch {retry_round}.{current_batch}/{total_retry_batches} ({len(batch_items)} items)")
                
                # Process retry batch
                batch_results = await self.process_batch_with_optimization(
                    batch_items, update_db=update_db, retry_round=retry_round
                )
                
                # Update statistics
                for result in batch_results:
                    if result.success:
                        self.stats["successful_items"] += 1
                    else:
                        self.stats["failed_items"] += 1
                
                # Output retry progress
                retry_success_rate = sum(1 for r in batch_results if r.success) / len(batch_results) if batch_results else 0
                logger.info(
                    f"Retry batch {retry_round}.{current_batch} completed | "
                    f"Success rate: {retry_success_rate:.2%} | "
                    f"Remaining retry queue: {len(self.retry_queue)}"
                )
            
            retry_round += 1
        
        # Output final retry statistics
        if retry_round > 1:
            logger.info(f"Retry processing completed, total {retry_round - 1} retry rounds")
        
        # Final statistics
        self.stats["end_time"] = time.time()
        processing_time = self.stats["end_time"] - self.stats["start_time"]
        
        logger.info(f"Processing complete - Total: {self.stats['total_items']}, "
                   f"Success: {self.stats['successful_items']}, "
                   f"Failed: {self.stats['failed_items']}, "
                   f"Processing time: {processing_time:.2f}s, "
                   f"Throughput: {self.stats['total_items']/processing_time:.1f} items/s")
        
        return self.stats
    
    async def process_unprocessed_data_optimized(
        self,
        limit: Optional[int] = None,
        update_db: bool = True
    ) -> Dict[str, Any]:
        """
        Optimized unprocessed data processing
        
        Args:
            limit: Limit count
            update_db: Whether to update database
            
        Returns:
            Processing statistics
        """
        # Query unprocessed data
        query = {
            "$or": [
                {"infer_privacy": {"$exists": False}},
                {"infer_privacy": None},
                {"infer_privacy": ""},
                {"infer_privacy": {}}
            ]
        }
        
        logger.info("Starting to process unprocessed data (optimized version)")
        return await self.process_all_data_optimized(
            query=query,
            limit=limit,
            update_db=update_db
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        llm_stats = self.llm_tool.get_performance_stats()
        
        # System metrics
        system_metrics = {}
        if self.enable_resource_monitoring and self.resource_monitor:
            system_metrics = self.resource_monitor.get_system_metrics()
        
        return {
            "processing_stats": {
                "total_items": self.stats["total_items"],
                "processed_items": self.stats["processed_items"],
                "successful_items": self.stats["successful_items"],
                "failed_items": self.stats["failed_items"],
                "processing_time": self.stats["end_time"] - self.stats["start_time"],
                "throughput": self.stats["total_items"] / (self.stats["end_time"] - self.stats["start_time"]) if self.stats["end_time"] > self.stats["start_time"] else 0,
                "retry_queue_size": len(self.retry_queue),
                "max_retry_rounds": self.max_retry_rounds
            },
            "concurrency_stats": {
                "current_concurrency": self.concurrency_controller.current_concurrency,
                "max_concurrency": self.concurrency_controller.max_concurrency,
                "min_concurrency": self.concurrency_controller.min_concurrency
            },
            "batch_stats": {
                "current_batch_size": self.batch_processor.current_batch_size,
                "base_batch_size": self.batch_processor.base_batch_size,
                "batch_performance_history": list(self.batch_processor.batch_performance)
            },
            "system_metrics": system_metrics,
            "llm_stats": llm_stats
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check"""
        llm_health = await self.llm_tool.health_check()
        
        try:
            count = self.mongo.count_documents(self.collection_name)
            unprocessed_count = self.mongo.count_documents(
                self.collection_name,
                query={
                    "$or": [
                        {"infer_privacy": {"$exists": False}},
                        {"infer_privacy": None},
                        {"infer_privacy": ""}
                    ]
                }
            )
            
            mongo_health = {
                "status": "healthy",
                "total_documents": count,
                "unprocessed_documents": unprocessed_count
            }
        except Exception as e:
            mongo_health = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # System health check
        system_health = {"status": "healthy"}
        if self.enable_resource_monitoring and self.resource_monitor:
            metrics = self.resource_monitor.get_system_metrics()
            if metrics.get("cpu_usage", 0) > 95 or metrics.get("memory_usage", 0) > 95:
                system_health = {"status": "warning", "metrics": metrics}
        
        return {
            "llm": llm_health,
            "mongo": mongo_health,
            "system": system_health,
            "overall_status": "healthy" if all(
                h["status"] == "healthy" for h in [llm_health, mongo_health, system_health]
            ) else "unhealthy"
        }
    
    async def close(self):
        """Close connections and clean up resources"""
        await self.llm_tool.close()
        self.mongo.disconnect()
        logger.info("High-performance asynchronous inferencer closed")


async def main():
    """Main function"""
    try:
        # Create high-performance asynchronous inferencer - full throttle mode
        judge = HighPerformanceAsyncInsLLMJudge(
            initial_concurrency=100,  # Initial concurrency 100
            max_concurrency=100,      # Max concurrency 100
            base_batch_size=50,       # Increase batch size
            request_timeout=120.0,     # Increase timeout
            context="I am a 27-year-old male actor",
            enable_dynamic_scaling=True,
            enable_resource_monitoring=True,
            max_retry_rounds=3  # Max 3 retry rounds
        )
        
        # Health check
        health = await judge.health_check()
        logger.info(f"Health check result: {json.dumps(health, indent=2, ensure_ascii=False)}")
        
        if health["overall_status"] != "healthy":
            logger.error("System unhealthy, exiting")
            return
        
        # Process unprocessed data
        stats = await judge.process_unprocessed_data_optimized(
            limit=None,  # Process all unprocessed data
            update_db=True
        )
        
        # Output performance report
        performance_report = judge.get_performance_report()
        logger.info(f"Performance report: {json.dumps(performance_report, indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        logger.error(traceback.format_exc())
    finally:
        if 'judge' in locals():
            await judge.close()


if __name__ == "__main__":
    # Run async main function
    asyncio.run(main())
