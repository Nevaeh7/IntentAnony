#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async OpenAI tool class for high-performance concurrent LLM calls
Supports batch processing, concurrency control, task queue management, etc.
"""

import os
import json
import time
import asyncio
import logging
from typing import List, Dict, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from collections import defaultdict, deque
import statistics
from .utils import ENV_VARS, set_env_key
from .openai_tool import ModelConfig, ModelProvider
from loguru import logger
import traceback

set_env_key()


@dataclass
class AsyncModelConfig(ModelConfig):
    """Async model configuration class"""
    max_concurrent_requests: int = 100
    request_timeout: float = 1000.0
    batch_size: int = 20
    retry_delay: float = 1.0
    max_retries: int = 3
    max_tokens: int = 8192
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    top_p: float = 1.0
    temperature: float = 0.7
    api_type: str = "chat"



@dataclass
class TaskResult:
    """Task result class"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    tokens_used: int = 0
    task_id: Optional[str] = None


@dataclass
class PerformanceStats:
    """Performance statistics class"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_execution_time: float = 0.0
    avg_response_time: float = 0.0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    response_times: List[float] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class AsyncOpenAITool:
    """Async OpenAI tool class providing high-performance concurrent LLM API interface"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        max_concurrent_requests: int = 10,
        request_timeout: float = 30.0,
        batch_size: int = 5,
        temperature: float = 0.1,
        top_p: float = 1.0,
        api_type: str = "chat",
    ):
        """
        Initialize async OpenAI tool

        Args:
            api_key: API key
            base_url: API base URL
            provider: Model provider
            model: Default model name
            max_concurrent_requests: Maximum concurrent requests
            request_timeout: Request timeout
            batch_size: Batch size
        """
        self.provider = provider
        self.provider_config = ModelProvider.get_provider_config(provider)

        # Get API key
        if not api_key:
            env_var = ENV_VARS.get(provider, "OPENAI_API_KEY")
            api_key = os.getenv(env_var)

        if not api_key:
            raise ValueError(
                f"Please provide API key or set environment variable {ENV_VARS.get(provider, 'OPENAI_API_KEY')}"
            )

        self.api_key = api_key

        # Determine base_url
        if not base_url:
            if 'speciale' in model:
                base_url = self.provider_config["base_url_speciale"]
                logger.warning(f"Using speciale model, base_url: {base_url}")
                model = model.replace('-speciale','')
            else:
                base_url = self.provider_config["base_url"]

        self.base_url = base_url

        # Create asynchronous client
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)


        # Default model configuration
        if model:
            self.default_config = AsyncModelConfig(name=model, api_type=api_type)
        else:
            default_models = {
                "openai": "gpt-5",
                "deepseek": "deepseek-chat",
                "qwen": "qwen-flash",
                "glm": "glm-4.5-flash",
                "claude": "claude-4-sonnet",
                "custom": "GLM-4.5-Flash",
                "dmx": "xxxx",
            }
            default_model = default_models.get(provider, "gpt-3.5-turbo")
            self.default_config = AsyncModelConfig(name=default_model, api_type=api_type)

        self.default_config.temperature = temperature
        self.default_config.top_p = top_p

        # Setup concurrency control
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout = request_timeout
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Setup logging
        self.logger = self._setup_logger()

        # Performance statistics
        self.stats = PerformanceStats()

        # Task queue
        self.task_queue = asyncio.Queue()
        self.results_queue = asyncio.Queue()

        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        logger.info(
            f"Async tool initialization complete - Provider: {provider}, Model: {self.default_config.name} Max concurrent: {max_concurrent_requests}, Timeout: {request_timeout}s"
        )

    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger("AsyncOpenAITool")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _build_api_params_and_func(
        self,
        model: str,
        messages: List[Dict[str, str]],
        config: AsyncModelConfig,
        stream: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Callable]:
        """
        Build API parameters and select corresponding call function
        
        Args:
            model: Model name
            messages: Message list
            config: Model configuration
            stream: Whether to use streaming output
            **kwargs: Other parameters
            
        Returns:
            Tuple of (parameter dict, call function)
        """
        is_gpt_model = 'gpt' in model.lower()
        
        if is_gpt_model:
            # GPT models use different API format
            if config.api_type == "responses":
                params = {
                    "model": model,
                    "input": messages,
                    "max_output_tokens": config.max_tokens,
                    "stream": stream,
                    **kwargs,
                }
                call_func = self.async_client.responses.create
                logger.warning(f"Using responses.create API")
            else:
                params = {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": config.max_tokens,
                    "frequency_penalty": config.frequency_penalty,
                    "presence_penalty": config.presence_penalty,
                    "stream": stream,
                    **kwargs,
                }
                call_func = self.async_client.chat.completions.create
        else:
            # Standard OpenAI format
            params = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": config.max_tokens,
                "top_p": config.top_p,
                "temperature": config.temperature,
                "frequency_penalty": config.frequency_penalty,
                "presence_penalty": config.presence_penalty,
                "stream": stream,
                **kwargs,
            }
            call_func = self.async_client.chat.completions.create
        
        return params, call_func

    def _extract_response_content(self, response: Any, is_gpt_model: bool = False) -> str:
        """
        Extract text content from response object, supports different API response formats
        
        Args:
            response: API response object
            is_gpt_model: Whether it's a GPT model (using responses.create API)
            
        Returns:
            Extracted text content
        """
        if is_gpt_model:
            # GPT models use responses.create, response format may differ
            # Try multiple possible attribute paths
            if self.default_config.api_type == "responses":
                return response.output_text
            else:
                return response.choices[0].message.content
            
        else:
            # Standard OpenAI format
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message'):
                    return response.choices[0].message.content
                elif hasattr(response.choices[0], 'text'):
                    return response.choices[0].text
        
        # If all attempts fail, try direct access
        raise AttributeError(
            f"Unable to extract content from response object. Response type: {type(response)}, "
            f"Available attributes: {dir(response)}"
        )

    def _extract_stream_chunk_content(self, chunk: Any, is_gpt_model: bool = False) -> Optional[str]:
        """
        Extract content from streaming response chunk
        
        Args:
            chunk: Streaming response chunk object
            is_gpt_model: Whether it's a GPT model
            
        Returns:
            Extracted content, returns None if not found
        """
        if is_gpt_model:
            # GPT model streaming response format
            if hasattr(chunk, 'output'):
                return chunk.output
            elif hasattr(chunk, 'text'):
                return chunk.text
            elif hasattr(chunk, 'content'):
                return chunk.content
            elif hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
                return chunk.delta.content
            elif hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    return chunk.choices[0].delta.content
        else:
            # Standard OpenAI streaming format
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    return content if content is not None else None
        
        return None

    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        config: Optional[AsyncModelConfig] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> TaskResult:
        """
        Async chat completion interface

        Args:
            messages: List of messages
            model: Model name
            config: Model configuration
            task_id: Task ID
            **kwargs: Other parameters

        Returns:
            TaskResult object
        """
        config = config or self.default_config
        model = model or config.name

        start_time = time.time()

        async with self.semaphore:
            try:
                params, call_func = self._build_api_params_and_func(
                    model, messages, config, stream=False, **kwargs
                )
                response = await asyncio.wait_for(call_func(**params), timeout=config.request_timeout)

                execution_time = time.time() - start_time
                tokens_used = response.usage.total_tokens if response.usage else 0

                # Update statistics
                self._update_stats(True, execution_time, tokens_used)

                return TaskResult(
                    success=True,
                    result=response,
                    execution_time=execution_time,
                    tokens_used=tokens_used,
                    task_id=task_id,
                )

            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = str(e)
                traceback_str = traceback.format_exc()
                logger.error(f"Exception in async_chat_completion: {error_msg}\n{traceback_str}")

                # Update statistics
                self._update_stats(False, execution_time, 0, error_msg)

                return TaskResult(
                    success=False,
                    error=error_msg,
                    execution_time=execution_time,
                    task_id=task_id,
                )

    async def async_simple_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[AsyncModelConfig] = None,
        task_id: Optional[str] = None,
    ) -> TaskResult:
        """
        Async simple chat interface

        Args:
            prompt: User input
            system_prompt: System prompt
            model: Model name
            config: Model configuration
            task_id: Task ID

        Returns:
            TaskResult object
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        result = await self.async_chat_completion(messages, model, config, task_id)

        if result.success:
            # Determine model name: prioritize passed model, then config.name, finally default_config.name
            model_name = model
            if not model_name and config:
                model_name = config.name
            if not model_name:
                model_name = self.default_config.name
            
            is_gpt_model = 'gpt' in (model_name or '').lower()
            result.result = self._extract_response_content(result.result, is_gpt_model)

        return result

    async def batch_chat_completion(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[AsyncModelConfig] = None,
        progress_callback: Optional[Callable] = None,
    ) -> List[TaskResult]:
        """
        Batch async chat completion

        Args:
            prompts: List of prompts
            system_prompt: System prompt
            model: Model name
            config: Model configuration
            progress_callback: Progress callback function

        Returns:
            List of TaskResult
        """
        config = config or self.default_config
        batch_size = config.batch_size

        # Create tasks
        tasks = []
        for i, prompt in enumerate(prompts):
            task_id = f"batch_task_{i}"
            task = self.async_simple_chat(
                prompt, system_prompt, model, config, task_id
            )
            tasks.append(task)

        # Process in batches
        results = []
        total_batches = (len(tasks) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(tasks), batch_size):
            batch_tasks = tasks[batch_idx:batch_idx + batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle exception results
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    result = TaskResult(
                        success=False,
                        error=str(result),
                        task_id=f"batch_task_{batch_idx + i}",
                    )
                results.append(result)

            # Call progress callback
            if progress_callback:
                current_batch = (batch_idx // batch_size) + 1
                progress_callback(current_batch, total_batches, len(results))

        return results

    async def concurrent_chat_completion(
        self,
        chat_requests: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> List[TaskResult]:
        """
        Concurrent chat completion

        Args:
            chat_requests: List of chat requests, each request contains messages, model, config, etc.
            max_concurrent: Maximum concurrency
            progress_callback: Progress callback function

        Returns:
            List of TaskResult
        """
        if max_concurrent:
            semaphore = asyncio.Semaphore(max_concurrent)
        else:
            semaphore = self.semaphore

        async def process_request(request_data: Dict[str, Any], task_id: str) -> TaskResult:
            async with semaphore:
                messages = request_data.get("messages", [])
                model = request_data.get("model")
                config = request_data.get("config")
                kwargs = request_data.get("kwargs", {})

                return await self.async_chat_completion(
                    messages, model, config, task_id, **kwargs
                )

        # Create all tasks
        tasks = []
        for i, request_data in enumerate(chat_requests):
            task_id = f"concurrent_task_{i}"
            task = process_request(request_data, task_id)
            tasks.append(task)

        # Execute concurrently
        results = []
        completed = 0
        total = len(tasks)

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1

            if progress_callback:
                progress_callback(completed, total, result)

        return results

    async def stream_batch_chat(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[AsyncModelConfig] = None,
        stream_callback: Optional[Callable] = None,
    ) -> List[str]:
        """
        Streaming batch chat

        Args:
            prompts: List of prompts
            system_prompt: System prompt
            model: Model name
            config: Model configuration
            stream_callback: Streaming callback function

        Returns:
            List of complete response texts
        """
        config = config or self.default_config
        model = model or config.name

        async def stream_single_chat(prompt: str, task_id: str) -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            params, call_func = self._build_api_params_and_func(
                model, messages, config, stream=True
            )

            try:
                stream = await call_func(**params)
                full_response = ""
                is_gpt_model = 'gpt' in model.lower()

                async for chunk in stream:
                    content = self._extract_stream_chunk_content(chunk, is_gpt_model)
                    if content is not None:
                        full_response += content

                        if stream_callback:
                            stream_callback(task_id, content)

                return full_response

            except Exception as e:
                self.logger.error(f"Streaming chat error (task {task_id}): {e}")
                return f"Error: {str(e)}"

        # Execute streaming chat concurrently
        tasks = []
        for i, prompt in enumerate(prompts):
            task_id = f"stream_task_{i}"
            task = stream_single_chat(prompt, task_id)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exception results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(f"Error: {str(result)}")
            else:
                processed_results.append(result)

        return processed_results

    async def async_embedding(
        self,
        texts: Union[str, List[str]],
        model: str = "text-embedding-ada-002",
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Asynchronous text embedding

        Args:
            texts: Text or list of texts
            model: Embedding model name
            batch_size: Batch size

        Returns:
            List of embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or self.batch_size
        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            try:
                response = await asyncio.wait_for(
                    self.async_client.embeddings.create(
                        model=model, input=batch_texts
                    ),
                    timeout=self.request_timeout
                )

                batch_embeddings = [item.embedding for item in response.data]
                results.extend(batch_embeddings)

            except Exception as e:
                self.logger.error(f"Embedding processing error: {e}")
                # Add empty vectors as placeholders
                results.extend([[] for _ in batch_texts])

        return results

    def _update_stats(
        self,
        success: bool,
        execution_time: float,
        tokens_used: int = 0,
        error_msg: Optional[str] = None,
    ):
        """Update performance statistics"""
        self.stats.total_requests += 1
        self.stats.total_execution_time += execution_time
        self.stats.total_tokens += tokens_used
        self.stats.response_times.append(execution_time)

        if success:
            self.stats.successful_requests += 1
        else:
            self.stats.failed_requests += 1
            if error_msg:
                self.stats.error_counts[error_msg] += 1

        # Calculate average
        if self.stats.total_requests > 0:
            self.stats.avg_response_time = (
                self.stats.total_execution_time / self.stats.total_requests
            )
            self.stats.error_rate = (
                self.stats.failed_requests / self.stats.total_requests
            )

        # Calculate requests per second (based on last 100 requests)
        if len(self.stats.response_times) > 100:
            recent_times = self.stats.response_times[-100:]
            avg_time = statistics.mean(recent_times)
            self.stats.requests_per_second = 1.0 / avg_time if avg_time > 0 else 0

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "total_tokens": self.stats.total_tokens,
            "total_execution_time": self.stats.total_execution_time,
            "avg_response_time": self.stats.avg_response_time,
            "requests_per_second": self.stats.requests_per_second,
            "error_rate": self.stats.error_rate,
            "error_counts": dict(self.stats.error_counts),
            "recent_response_times": self.stats.response_times[-10:],  # Last 10 response times
        }

    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = PerformanceStats()

    async def health_check(self) -> Dict[str, Any]:
        """Health check"""
        try:
            start_time = time.time()
            result = await self.async_simple_chat("Hello", task_id="health_check")
            response_time = time.time() - start_time

            return {
                "status": "healthy" if result.success else "unhealthy",
                "response_time": response_time,
                "error": result.error if not result.success else None,
                "provider": self.provider,
                "model": self.default_config.name,
                "base_url": self.base_url,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider": self.provider,
                "model": self.default_config.name,
                "base_url": self.base_url,
            }

    async def close(self):
        """Close client and clean up resources"""
        if hasattr(self, 'async_client'):
            await self.async_client.close()
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        self.logger.info("Async tool closed")

    def __del__(self):
        """Destructor"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)


# Convenience functions
def create_async_any_tool(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: str = "glm",
    model: Optional[str] = 'glm-4.5-flash',
    max_concurrent_requests: int = 50,
    temperature: float = 0.1,
    top_p: float = 1.0,
    api_type: str = "chat",
) -> AsyncOpenAITool:
    """Convenience function to create an async tool for any provider"""

    return AsyncOpenAITool(api_key=api_key, base_url=base_url, provider=provider, model=model, max_concurrent_requests=max_concurrent_requests, temperature=temperature, top_p=top_p, api_type=api_type)

def create_async_openai_tool(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: str = "openai",
    model: Optional[str] = 'gpt-5',
    max_concurrent_requests: int = 10,
) -> AsyncOpenAITool:
    """Convenience function to create an async OpenAI tool"""
    return AsyncOpenAITool(
        api_key, base_url, provider, model, max_concurrent_requests
    )


def create_async_deepseek_tool(
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
    max_concurrent_requests: int = 10,
) -> AsyncOpenAITool:
    """Convenience function to create an async DeepSeek tool"""
    return AsyncOpenAITool(
        api_key=api_key,
        provider="deepseek",
        model=model,
        max_concurrent_requests=max_concurrent_requests,
    )


def create_async_claude_tool(
    api_key: Optional[str] = None,
    model: str = "claude-4-sonnet",
    max_concurrent_requests: int = 10,
) -> AsyncOpenAITool:
    """Convenience function to create an async Claude tool"""
    return AsyncOpenAITool(
        api_key=api_key,
        provider="claude",
        model=model,
        max_concurrent_requests=max_concurrent_requests,
    )


async def quick_async_chat(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    provider: str = "openai",
) -> str:
    """Quick async chat function"""
    tool = create_async_openai_tool(api_key, provider=provider, model=model)
    try:
        result = await tool.async_simple_chat(prompt)
        return result.result if result.success else f"Error: {result.error}"
    finally:
        await tool.close()


if __name__ == "__main__":
    print("Async OpenAI tool class created!")
    print("Supports high-performance concurrent processing, batch operations, streaming output, etc.")
    print("\nUsage examples:")
    print("1. Basic async usage:")
    print("   async def main():")
    print("       tool = AsyncOpenAITool()")
    print("       result = await tool.async_simple_chat('Hello')")
    print("       print(result.result)")
    print("       await tool.close()")
    print("\n2. Batch processing:")
    print("   async def batch_example():")
    print("       tool = AsyncOpenAITool()")
    print("       prompts = ['Hello', 'Goodbye', 'Thank you']")
    print("       results = await tool.batch_chat_completion(prompts)")
    print("       for result in results:")
    print("           print(result.result)")
    print("       await tool.close()")
    print("\n3. Concurrent processing:")
    print("       async def concurrent_example():")
    print("       tool = AsyncOpenAITool()")
    print("       requests = [{'messages': [{'role': 'user', 'content': 'Hello'}]}]")
    print("       results = await tool.concurrent_chat_completion(requests)")
    print("       await tool.close()")
    print("\n4. Performance monitoring:")
    print("   stats = tool.get_performance_stats()")
    print("   print(f'Total requests: {stats[\"total_requests\"]}')")
    print("   print(f'Success rate: {1 - stats[\"error_rate\"]:.2%}')")
    print("   print(f'Average response time: {stats[\"avg_response_time\"]:.2f}s')")
