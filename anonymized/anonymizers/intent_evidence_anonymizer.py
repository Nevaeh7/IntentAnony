#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Asynchronous anonymization processing script - Reddit data
Uses asynchronous tools for efficient LLM calls and data processing
"""

import json
import os
import asyncio
from typing import Dict, Any, List, Optional, Generator, Tuple
from loguru import logger
from tqdm import tqdm

from prompt_kits.policy_manager import get_policy_manager, reload_policies
from prompt_kits.prompt_manager_final import get_manager, PromptManager
from llm_tools.async_openai_tool import create_async_any_tool, TaskResult, AsyncModelConfig
from utils.mongo_utils import MongoDBConnector
from utils.x_utils import save_jsonl, parse_json_response, write_add_jsonl, strip_surrogates
from utils.dataset_utils import *
from pathlib import Path
import sys

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class AsyncPIECAnonymizer:
    """Asynchronous anonymization processor"""
    
    def __init__(
        self,
        cfg: Any = None,
        mongo_host: str = "localhost",
        mongo_port: int = 27017,
        db_name: str = "INS_DB",
        collection_name: str = "personal_reddit",
        provider: str = "deepseek",
        model: str = "deepseek-reasoner",
        anony_llm_model: Any = None,
        anony_prompt_manager: Any = None,
        anony_policy: Any = None,
        adversary_attack_model: Any = None,
        adversary_attack_model_cfg: Any = None,
        piec_model: Any = None,
        piec_model_cfg: Any = None,
        max_concurrent_requests: int = 100,
        request_timeout: float = 6000.0,
        batch_size: int = 100,
        policy_version: str = "4.0",
        policy_language: str = "zh",
        prompt_template: str = "intent_anonymizers_v2",
        max_retry_rounds: int = 6,
        temperature: float = 0.7,
        top_p: float = 1.0,
        anonymized_which_parts: str = None,
        update_db: bool = False,
        p_results: Dict[str, Any] = None,
        is_pre_iiv: bool = False,
    ):
        """
        Initialize asynchronous anonymization processor
        
        Args:
            mongo_host: MongoDB host address
            mongo_port: MongoDB port
            db_name: Database name
            collection_name: Collection name
            provider: Model provider
            model: Model name
            max_concurrent_requests: Maximum concurrent requests
            request_timeout: Request timeout (seconds)
            batch_size: Batch size
            policy_version: Policy version
            policy_language: Policy language
            prompt_template: Prompt template name
            max_retry_rounds: Maximum retry rounds
        """
        # Initialize MongoDB connection
        self.cfg = cfg
        self.mongo = MongoDBConnector(
            host=mongo_host,
            port=mongo_port,
            db_name=db_name
        )
        if not self.mongo.connect():
            raise ConnectionError("Unable to connect to MongoDB database")
        
        self.collection_name = collection_name
        self.model_config = AsyncModelConfig(
            max_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            batch_size=batch_size,
            max_retries=max_retry_rounds,
            request_timeout=request_timeout
        )
        
        
        # Initialize asynchronous LLM tool
        if anony_llm_model is None:
            self.llm_tool = create_async_any_tool(
                provider=provider,
                model=model,
                max_concurrent_requests=max_concurrent_requests,
                temperature=temperature,
                top_p=top_p,
            )
            self.model_config.name = self.llm_tool.default_config.name
        else:
            self.llm_tool = anony_llm_model
            logger.warning(f"Using provided model: {self.llm_tool.default_config.name}")
            self.model_config.name = self.llm_tool.default_config.name

        self.adversary_attack_model = adversary_attack_model
        self.adversary_attack_model_cfg = adversary_attack_model_cfg
        self.p_results = p_results

        logger.warning(f"Using intent adversarial attack model: {self.adversary_attack_model.default_config.name}")
        self.piec_model = piec_model
        self.piec_model_cfg = piec_model_cfg

        logger.warning(f"Using privacy inference evidence chain model: {self.piec_model.default_config.name}")

        self.update_key = self.cfg.task_config.anonymizer.anon_model_name
        self.update_db = update_db
        self.is_pre_iiv = is_pre_iiv
        
        # Initialize prompt manager
        if anony_prompt_manager is None:
            self.prompt_manager = get_manager(auto_reload=True)
        else:
            self.prompt_manager = anony_prompt_manager

        if anony_policy is None:
            reload_policies(version=policy_version, language=policy_language)
            manager = get_policy_manager(auto_reload=True)
            manager.reload()
            self.anony_policy = manager.get(version=policy_version, language=policy_language)
        else:
            self.anony_policy = anony_policy
        if anonymized_which_parts is not None:
            self.anonymized_which_parts = anonymized_which_parts
        else:
            self.anonymized_which_parts = self.cfg.task_config.anonymizer.anonymized_which_parts
        if cfg is not None:
            self.prompt_category = self.cfg.task_config.anon_model.prompt_category
            self.policy_language = self.cfg.task_config.anon_model.prompt_language
        else:
            self.prompt_category = prompt_template
            self.policy_language = policy_language


        



        
        self.model = self.model_config.name
        self.batch_size = batch_size
        self.request_timeout = request_timeout
        self.max_retry_rounds = max_retry_rounds
        
        # Retry queue: stores (item, retry_count) tuples
        self.retry_queue: List[tuple] = []
        
        # Current processing queue size (for logging display)
        self.current_queue_size: int = 0
        
        # File write lock: protects file writes in async high-concurrency scenarios
        self._file_write_lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            "total_items": 0,
            "processed_items": 0,
            "successful_items": 0,
            "failed_items": 0,
            "retry_count": 0,
        }
        
        logger.info(
            f"Async anonymization processor initialization complete - "
            f"Model: {self.model_config.name}, Concurrency: {max_concurrent_requests}, Batch size: {batch_size}"
        )
    
    @staticmethod
    def read_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Read JSONL file and return generator
        
        Args:
            path: File path
            
        Yields:
            Parsed JSON object
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File does not exist: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        yield json.loads(line, strict=False)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num} JSON parsing failed: {e}, skipping this line")
                        continue
    
    def _extract_content_from_result(self, result: TaskResult) -> str:
        """
        Extract text content from TaskResult
        
        Args:
            result: TaskResult object
            
        Returns:
            Extracted text content
        """
        api_type = self.llm_tool.default_config.api_type
        if api_type == 'responses':
            return result.result.output_text
        else:
            return result.result.choices[0].message.content
    
    async def _safe_write_jsonl(self, data: Dict[str, Any], path: str):
        """
        Asynchronously and safely write JSONL file (thread-safe)
        
        Uses asyncio.Lock to ensure file writing safety in asynchronous high-concurrency scenarios,
        avoiding data corruption or loss due to multiple coroutines writing simultaneously.
        
        Args:
            data: Data dictionary to write
            path: File path
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Use lock to protect file write operation
        async with self._file_write_lock:
            try:
                # Clean surrogate characters in data to avoid UTF-8 encoding errors
                def clean_data_for_write(obj):
                    """Recursively clean surrogate characters from all strings in an object"""
                    if isinstance(obj, str):
                        # Directly check if it contains surrogate characters
                        if any(0xD800 <= ord(c) <= 0xDFFF for c in obj):
                            return strip_surrogates(obj)
                        return obj
                    elif isinstance(obj, dict):
                        return {k: clean_data_for_write(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [clean_data_for_write(item) for item in obj]
                    else:
                        return obj
                
                # Clean data
                cleaned_data = clean_data_for_write(data)
                
                # Execute file write within the lock, ensuring atomicity
                with open(path, "a", encoding="utf-8") as json_file:
                    # json_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                    json_file.write(json.dumps(cleaned_data, ensure_ascii=False) + "\n")

                    
                    json_file.flush()
            except Exception as e:
                logger.error(f"File write failed {path}: {e}")
                raise

    async def _infer_intent_attack(
        self,
        eval_attributes: List[str],
        user_context: str,
        task_id: str,
        max_retries: int = 2,
    ) -> Tuple[TaskResult, Dict[str, Any]]:
        """
        Infer intent attack (with retry mechanism)
        
        Args:
            eval_attributes: List of attributes to evaluate
            user_context: User context text
            task_id: Task ID
            max_retries: Maximum retry attempts (default 2, total up to 3 attempts)
            
        Returns:
            (TaskResult, json_result) tuple
        """
        last_result = None
        last_json_result = {}
        user_context = strip_surrogates(user_context)
        
        for attempt in range(max_retries + 1):  # Total max_retries + 1 attempts
            try:
                infer_intent_attack_messages = self.prompt_manager.get_messages(
                    category=self.adversary_attack_model_cfg.prompt_category,
                    language=self.adversary_attack_model_cfg.prompt_language,
                    inference_attributes_types=eval_attributes,
                    user_context=user_context
                )
                infer_intent_attack_result = await self.adversary_attack_model.async_chat_completion(
                    messages=infer_intent_attack_messages,
                    task_id=task_id,
                )
                
                if not infer_intent_attack_result.success:
                    last_result = infer_intent_attack_result
                    if attempt < max_retries:
                        logger.warning(
                            f"Intent attack inference failed (attempt {attempt + 1}/{max_retries + 1}): {infer_intent_attack_result.error}"
                        )
                        continue
                    return infer_intent_attack_result, {}
                
                content = self._extract_content_from_result(infer_intent_attack_result)
                infer_intent_attack_json_result = parse_json_response(content)
                # infer_intent_attack_json_result = content
                infer_intent_attack_result.result = infer_intent_attack_json_result
                # logger.success(infer_intent_attack_json_result)
                
                return infer_intent_attack_result, infer_intent_attack_json_result
                
            except (ValueError, json.JSONDecodeError, AttributeError) as e:
                last_result = TaskResult(
                    success=False,
                    error=f"JSON parsing failed: {e}",
                    task_id=task_id,
                )
                if attempt < max_retries:
                    logger.warning(
                        f"Intent attack JSON parsing failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
                    continue
                logger.error(f"Intent attack inference failed, maximum retries reached: {e}")
                return last_result, {}
        
        return last_result or TaskResult(success=False, error="Unknown error", task_id=task_id), {}

    async def _infer_evidence_chain(
        self,
        attribute_inference_results: Dict[str, Any],
        user_context: str,
        task_id: str,
        max_retries: int = 2,
    ) -> Tuple[TaskResult, Dict[str, Any]]:
        """
        Infer evidence chain (with retry mechanism)
        
        Args:
            attribute_inference_results: Attribute inference results
            user_context: User context text
            task_id: Task ID
            max_retries: Maximum retry attempts (default 2, total up to 3 attempts)
            
        Returns:
            (TaskResult, json_result) tuple
        """
        last_result = None
        last_json_result = {}
        user_context = strip_surrogates(user_context)
        attribute_inference_results = strip_surrogates(json.dumps(attribute_inference_results))

        
        for attempt in range(max_retries + 1):  # Total max_retries + 1 attempts
            try:
                if self.cfg.intent_conf_thres > 0:
                    intent_conf_thres = self.cfg.intent_conf_thres
                    infer_evidence_chain_messages = self.prompt_manager.get_messages(
                    category=self.piec_model_cfg.prompt_category,
                    language=self.piec_model_cfg.prompt_language,
                    attribute_inference_results=attribute_inference_results,
                    user_context=user_context,
                    intent_conf_thres=intent_conf_thres
                )
                else:
                    infer_evidence_chain_messages = self.prompt_manager.get_messages(
                    category=self.piec_model_cfg.prompt_category,
                    language=self.piec_model_cfg.prompt_language,
                    attribute_inference_results=attribute_inference_results,
                    user_context=user_context,
                )

                infer_evidence_chain_result = await self.piec_model.async_chat_completion(
                    messages=infer_evidence_chain_messages,
                    task_id=task_id,
                )
                
                if not infer_evidence_chain_result.success:
                    last_result = infer_evidence_chain_result
                    if attempt < max_retries:
                        logger.warning(
                            f"Evidence chain inference failed (attempt {attempt + 1}/{max_retries + 1}): {infer_evidence_chain_result.error}"
                        )
                        continue
                    return infer_evidence_chain_result, {}
                
                content = self._extract_content_from_result(infer_evidence_chain_result)
                infer_evidence_chain_json_result = parse_json_response(content)
                # infer_evidence_chain_json_result = content
                infer_evidence_chain_result.result = infer_evidence_chain_json_result
                # logger.success(infer_evidence_chain_json_result)
                
                return infer_evidence_chain_result, infer_evidence_chain_json_result
                
            except (ValueError, json.JSONDecodeError, AttributeError) as e:
                last_result = TaskResult(
                    success=False,
                    error=f"JSON parsing failed: {e}",
                    task_id=task_id,
                )
                if attempt < max_retries:
                    logger.warning(
                        f"Evidence chain JSON parsing failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
                    continue
                logger.error(f"Evidence chain inference failed, maximum retries reached: {e}")
                return last_result, {}
        
        return last_result or TaskResult(success=False, error="Unknown error", task_id=task_id), {}

    async def _infer_intent_vector(
        self,
        user_context: str,
        task_id: str,
        max_retries: int = 2,
    ) -> Tuple[TaskResult, Dict[str, Any]]:
        """
        Infer intent vector (with retry mechanism)
        
        Args:
            user_context: User context text
            task_id: Task ID
            max_retries: Maximum retry attempts (default 2, total up to 3 attempts)
            
        Returns:
            (TaskResult, json_result) tuple
        """
        last_result = None
        last_json_result = {}
        user_context = strip_surrogates(user_context)

        
        for attempt in range(max_retries + 1):  # Total max_retries + 1 attempts
            try:
                infer_intent_vector_messages = self.prompt_manager.get_messages(
                    category="intentv2",
                    language="en",
                    user_context=user_context,
                )

                infer_intent_vector_result = await self.llm_tool.async_chat_completion(
                    messages=infer_intent_vector_messages,
                    task_id=task_id,
                )
                
                if not infer_intent_vector_result.success:
                    last_result = infer_intent_vector_result
                    if attempt < max_retries:
                        logger.warning(
                            f"Intent vector inference failed (attempt {attempt + 1}/{max_retries + 1}): {infer_intent_vector_result.error}"
                        )
                        continue
                    return infer_intent_vector_result, {}
                
                content = self._extract_content_from_result(infer_intent_vector_result)
                infer_intent_vector_json_result = parse_json_response(content)
                infer_intent_vector_result.result = infer_intent_vector_json_result
                
                return infer_intent_vector_result, infer_intent_vector_json_result
                
            except (ValueError, json.JSONDecodeError, AttributeError) as e:
                last_result = TaskResult(
                    success=False,
                    error=f"JSON parsing failed: {e}",
                    task_id=task_id,
                )
                if attempt < max_retries:
                    logger.warning(
                        f"Intent vector JSON parsing failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
                    continue
                logger.error(f"Intent vector inference failed, maximum retries reached: {e}")
                return last_result, {}
        
        return last_result or TaskResult(success=False, error="Unknown error", task_id=task_id), {}

    async def _anonymize_with_intent_evidence(
        self,
        attribute_inference_results: Dict[str, Any],
        privacy_inference_evidence_chain: Dict[str, Any],
        user_context: str,
        task_id: str,
        max_retries: int = 5,
        intent_vector: Dict[str, float] = None,
    ) -> Tuple[TaskResult, Dict[str, Any]]:
        """
        Anonymize using intent and evidence chain (with retry mechanism)
        
        Args:
            attribute_inference_results: Attribute inference results
            privacy_inference_evidence_chain: Privacy inference evidence chain
            user_context: User context text
            task_id: Task ID
            max_retries: Maximum retry attempts (default 5, total up to 6 attempts)
            
        Returns:
            (TaskResult, json_result) tuple
        """
        last_result = None
        last_json_result = {}

        # Safely clean surrogate characters to avoid UTF-8 encoding errors (before calling LLM API)
        def safe_clean_surrogates(obj):
            """Recursively clean surrogate characters from all strings in an object"""
            if isinstance(obj, str):
                # Directly check if it contains surrogate characters, avoiding unnecessary encoding attempts
                if any(0xD800 <= ord(c) <= 0xDFFF for c in obj):
                    return strip_surrogates(obj)
                return obj
            elif isinstance(obj, dict):
                return {k: safe_clean_surrogates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [safe_clean_surrogates(item) for item in obj]
            else:
                return obj
        
        # Clean all input parameters
        user_context = safe_clean_surrogates(user_context)
        attribute_inference_results = safe_clean_surrogates(attribute_inference_results)
        privacy_inference_evidence_chain = safe_clean_surrogates(privacy_inference_evidence_chain)
        
        for attempt in range(max_retries + 1):  # Total max_retries + 1 attempts
            try:
                intent_evidence_anonymization_messages = self.prompt_manager.get_messages(
                    category=self.cfg.task_config.anon_model.prompt_category,
                    language=self.cfg.task_config.anon_model.prompt_language,
                    policy_config=self.anony_policy,
                    attribute_inference_results=attribute_inference_results,
                    privacy_inference_evidence_chain=privacy_inference_evidence_chain,
                    user_context=user_context,
                    intent_vector=intent_vector
                )

                result = await self.llm_tool.async_chat_completion(
                    messages=intent_evidence_anonymization_messages,
                    task_id=task_id,
                )
                
                if not result.success:
                    last_result = result
                    if attempt < max_retries:
                        logger.warning(
                            f"Anonymization failed (attempt {attempt + 1}/{max_retries + 1}): {result.error}"
                        )
                        continue
                    return result, {}
                
                content = self._extract_content_from_result(result)
                if not content:
                    if attempt < max_retries:
                        logger.warning(
                            f"Anonymization failed (attempt {attempt + 1}/{max_retries + 1}): {result.error}"
                        )
                        continue

                # if 'gemini' in self.cfg.task_config.anon_model.name:
                #     json_result = {
                #         'anonymized_text': content
                #     }
                # else:
                json_result = parse_json_response(content)
                result.result = json_result

                logger.success(json_result)
                return result, json_result
                
            except (ValueError, json.JSONDecodeError, AttributeError) as e:
                last_result = TaskResult(
                    success=False,
                    error=f"JSON parsing failed: {e}",
                    task_id=task_id,
                )
                if attempt < max_retries:
                    logger.warning(
                        f"Anonymization JSON parsing failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
                    continue
                logger.error(f"Anonymization failed, maximum retries reached: {e}")
                return last_result, {}
        
        return last_result or TaskResult(success=False, error="Unknown error", task_id=task_id), {}

    
    async def anonymize_single_item(
        self,
        item: Dict[str, Any],
        index: int,
        task_id: Optional[str] = None,
        retry_count: int = 0,
    ) -> Tuple[TaskResult, bool]:
        """
        Anonymize a single data item
        
        Args:
            item: Data item
            index: Data item index
            task_id: Task ID
            retry_count: Current retry count
            
        Returns:
            (TaskResult, should_retry) tuple, where should_retry indicates whether to retry
        """
        item.setdefault('anonymized_results', {})
        if task_id is None:
            task_id = f"anonymize_{index}"
        
        try:
            # Check if already processed
            if self.update_key in item['anonymized_results'] and item['anonymized_results'].get(self.update_key, None):
                logger.debug(f"Task {task_id} already processed, skipping")
                return (
                    TaskResult(
                        success=True,
                        result=item[self.update_key],
                        task_id=task_id,
                    ),
                    False
                )
            
            # Set ID
            # if '_id' not in item:
                # item['_id'] = f"{self.cfg.dataset_name}_{index}"
            
            user_prompt = item.get('user_text', '')
            anonymized_max_iter=self.cfg.task_config.anonymizer.anonymized_max_iter
            iter_index=0
            while iter_index < anonymized_max_iter:
                # Step 1: Infer intent attack
                eval_attributes = item.get('intent_privacy_labled', {}).get('protected_attributes', [])
                infer_intent_attack_result, infer_intent_attack_json_result = await self._infer_intent_attack(
                    eval_attributes=eval_attributes,
                    user_context=user_prompt,
                    task_id=task_id,
                )
                if not infer_intent_attack_result.success:
                    continue

                # Step 2: Infer evidence chain
                infer_evidence_chain_result, infer_evidence_chain_json_result = await self._infer_evidence_chain(
                    attribute_inference_results=infer_intent_attack_json_result,
                    user_context=user_prompt,
                    task_id=task_id,
                )
                if not infer_evidence_chain_result.success:
                    continue

                # Step 3: Anonymize

                if self.is_pre_iiv:
                    # Infer vector before anonymization
                    infer_intent_vector_result, infer_intent_vector_json_result = await self._infer_intent_vector(
                        user_context=user_prompt,
                        task_id=task_id,
                    )
                    if not infer_intent_vector_result.success:
                        continue
                else:
                    # Infer vector together during anonymization
                    infer_intent_vector_json_result = None
                   


                result, json_result = await self._anonymize_with_intent_evidence(
                    attribute_inference_results=infer_intent_attack_json_result,
                    privacy_inference_evidence_chain=infer_evidence_chain_json_result,
                    user_context=user_prompt,
                    task_id=task_id,
                    intent_vector=infer_intent_vector_json_result,
                )
                if not result.success:
                    continue
                
                user_prompt = json_result.get('anonymized_text', '')
                iter_index += 1
                logger.success(f" Anonymization iteration ({iter_index}/{anonymized_max_iter}) successful - task_id: {task_id}")

            # logger.success(result)
            if result.success:
                try:
                    logger.info(
                        f" Task {task_id} anonymization successful | "
                        f"Queue remaining: {self.current_queue_size} items"
                    )
                    item['anonymized_results'][self.update_key] = result.result
                    if self.update_db:
                        self.mongo.update_one_data(self.collection_name, item)
                    # Use asynchronous safe write method to ensure file write safety in high-concurrency scenarios
                    if not result.result:
                        logger.error(f" Task {task_id} anonymization result is empty")
                        return result, True
                    await self._safe_write_jsonl(item, self.cfg.task_config.outpath + "/each_anonymized_results.jsonl")
                    return result, False
                except (ValueError, json.JSONDecodeError, AttributeError) as e:
                    logger.error(f" Task {task_id} JSON parsing failed (retry count: {retry_count}): {e}")
                    result.success = False
                    result.error = f"JSON parsing failed: {e}"
                    # Check if should retry
                    should_retry = retry_count < self.max_retry_rounds
                    if should_retry:
                        logger.info(f" Task {task_id} will be added to retry queue (retry {retry_count + 1})")
                    else:
                        logger.warning(f" Task {task_id} reached maximum retry count ({self.max_retry_rounds}), giving up")
                    return result, should_retry
            else:
                # LLM call failed, extract error message
                error_msg = result.error if result.error else "LLM call failed (no error details)"
                logger.error(f" Task {task_id} LLM call failed: {error_msg}")
                # Ensure error message is set
                if not result.error:
                    result.error = error_msg
                # LLM call failure also considers retry
                should_retry = retry_count < self.max_retry_rounds
                if should_retry:
                    logger.info(f" Task {task_id} will be added to retry queue (retry {retry_count + 1})")
                return result, should_retry
            
        except Exception as e:
            logger.error(f" Task {task_id} processing exception: {e}")
            should_retry = retry_count < self.max_retry_rounds
            return (
                TaskResult(
                    success=False,
                    error=str(e),
                    task_id=task_id,
                ),
                should_retry
            )
    
    async def process_batch(
        self,
        items: List[Dict[str, Any]],
        start_index: int = 0,
        retry_round: int = 0,
    ) -> List[TaskResult]:
        """
        Process data items in batches
        
        Args:
            items: List of data items, each element can be an item or a (item, retry_count) tuple
            start_index: Starting index
            retry_round: Retry round
            
        Returns:
            List of TaskResult
        """
        # Parse items, handle retry queue format
        processed_items = []
        updated_profiles = []
        for item_data in items:
            if isinstance(item_data, tuple):
                item, retry_count = item_data
            else:
                item = item_data
                retry_count = 0
            processed_items.append((item, retry_count))
        
        # Create tasks
        tasks = []
        for i, (item, retry_count) in enumerate(processed_items):
            task_id = f"batch_{start_index + i}_r{retry_round}"
            task = self.anonymize_single_item(item, start_index + i, task_id, retry_count)
            tasks.append((task, item, retry_count))
        
        # Execute concurrently
        task_coros = [task for task, _, _ in tasks]
        batch_items = [(item, retry_count) for _, item, retry_count in tasks]
        
        batch_results = await asyncio.gather(*task_coros, return_exceptions=True)
        
        # Process results (ensure results correspond one-to-one with inputs)
        results = []
        if len(batch_results) != len(batch_items):
            logger.error(
                f" process_batch result count mismatch: "
                f"batch_results={len(batch_results)}, batch_items={len(batch_items)}"
            )
        
        for i, (result_data, (item, retry_count)) in enumerate(zip(batch_results, batch_items)):
            # Handle exception results
            if isinstance(result_data, Exception):
                result = TaskResult(
                    success=False,
                    error=str(result_data) if result_data else "Processing exception (no error details)",
                    task_id=f"batch_{start_index + i}_r{retry_round}",
                )
            else:
                result, _ = result_data  # Ignore should_retry, determined in process_all_data
                # Ensure error message exists
                if not result.success and not result.error:
                    result.error = "Processing failed (no error details)"
            
            results.append(result)
            
            # Update database (only on success)
            if result.success:
                if self.update_db:
                    try:
                        item['anonymized_results'][self.update_key] = result.result
                        self.mongo.update_one_data(self.collection_name, item)
                        logger.debug(f" Database update successful: {item.get('_id')}")
                    except Exception as e:
                        logger.error(f" Database update failed: {e}")
                        # When database update fails, mark as failed, but don't handle retry here
                        # Retry logic is handled uniformly in process_all_data
                        result.success = False
                        result.error = f"Database update failed: {e}"
                else:
                    item['anonymized_results'][self.update_key] = result.result
                    updated_profiles.append(item)

        # add_save_jsonl(updated_profiles, self.cfg.task_config.outpath + "/anonymized_results.jsonl")
            
        
        # Ensure returned result count matches input
        if len(results) != len(batch_items):
            logger.error(
                f" process_batch returned result count mismatch: "
                f"results={len(results)}, batch_items={len(batch_items)}"
            )
            # If result count is insufficient, fill with failure results
            while len(results) < len(batch_items):
                results.append(TaskResult(
                    success=False,
                    error="Processing result missing",
                    task_id=f"batch_missing_{len(results)}",
                ))
        
        return results
    
    async def process_all_data(
        self,
        data_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        synthetic_dataset: Optional[List[Dict[str, Any]]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process all data - unified queue processing mode
        Data is added to the queue from the beginning, processed in batches, and failed items are automatically added to the retry queue
        
        Args:
            data_path: Data file path
            collection_name: Collection name
            limit: Limit on the number of items to process (None means process all)
            
        Returns:
            Processing statistics
        """
        logger.info("Starting to process all data (unified queue mode)")
        
        # Reset statistics and queue
        self.stats = {
            "total_items": 0,
            "processed_items": 0,
            "successful_items": 0,
            "failed_items": 0,
            "retry_count": 0,
        }
        self.retry_queue = []
        
        # Read data and initialize queue (format: [(item, retry_count), ...])
        if self.cfg.mode=='new':
            if data_path:
                synthetic_dataset = list(self.read_jsonl(data_path))
            else:
                query = {f'anonymized_results.{self.update_key}': {"$exists": False}}
                synthetic_dataset = self.mongo.read_data(collection_name, query=query)

            
                
            synthetic_dataset = prepare_datasets(synthetic_dataset, self.cfg.dataset_name)
            synthetic_dataset = filter_dataset_by_privacy_count(synthetic_dataset, threshold=1)
        else:
            synthetic_dataset = self.p_results["unfinished_profiles"]
        if limit:
            synthetic_dataset = synthetic_dataset[:limit]
        
        self.stats["total_items"] = len(synthetic_dataset)
        logger.info(f"Successfully read {len(synthetic_dataset)} records")
        
        if not synthetic_dataset:
            logger.warning("No data to process")
            return self.stats
        
        # Initialize data as queue format: (item, retry_count=0)
        processing_queue = [(item, 0) for item in synthetic_dataset]
        batch_number = 0
        updated_profiles = []
        progress_bar = tqdm(
            total=self.stats["total_items"],
            desc="Anonymizing profiles",
            unit="item",
            leave=True,
            ascii=True,
            file=sys.stdout,
            dynamic_ncols=True,
        )
        progress_bar.display()
        progress_bar.update(0)
        progress_bar.refresh()
        
        # Unified processing queue: loop until queue is empty
        try:
            while processing_queue:
                batch_number += 1
                
                # Take a batch of data from the queue
                batch_items = processing_queue[:self.batch_size]
                processing_queue = processing_queue[self.batch_size:]
                
                # Record current queue state
                queue_size = len(processing_queue)
                self.current_queue_size = queue_size  # Update instance variable for logging
                max_retry_in_batch = max(retry_count for _, retry_count in batch_items) if batch_items else 0
                
                logger.info(
                    f"Processing batch {batch_number} | "
                    f"Current batch: {len(batch_items)} items | "
                    f"Queue remaining: {queue_size} items | "
                    f"Max retry count in batch: {max_retry_in_batch}"
                )
                
                # Process batch (returned batch_items need to be consistent with process_batch internals)
                batch_results = await self.process_batch(
                    batch_items,
                    start_index=batch_number * self.batch_size,  # Index no longer important since using queue
                    retry_round=max_retry_in_batch,
                )
                
                # Ensure batch_items and batch_results have consistent length
                if len(batch_items) != len(batch_results):
                    logger.error(
                        f" Batch {batch_number} result count mismatch: "
                        f"items={len(batch_items)}, results={len(batch_results)}"
                    )
                    # Truncate to shorter length
                    min_len = min(len(batch_items), len(batch_results))
                    batch_items = batch_items[:min_len]
                    batch_results = batch_results[:min_len]
                
                # Process results: successful ones are done, failed ones that can retry are put back in queue
                for (item, retry_count), result in zip(batch_items, batch_results):
                    self.stats["processed_items"] += 1
                    
                    if result.success:
                        # Success: update statistics
                        self.stats["successful_items"] += 1
                        item['anonymized_results'][self.update_key] = result.result
                        updated_profiles.append(item)
                        progress_bar.update(1)
                        progress_bar.refresh()

                        # Update queue size (because items may be put back in queue, need to recalculate)
                        current_queue = len(processing_queue)
                        self.current_queue_size = current_queue
                        logger.debug(f" Item {item.get('_id', 'unknown')} processed successfully | Queue remaining: {current_queue}")
                    else:
                        # Failure: check if retry is needed
                        if retry_count < self.max_retry_rounds:
                            # Can retry: put back in queue
                            processing_queue.append((item, retry_count + 1))
                            self.stats["retry_count"] += 1
                            error_display = (result.error or "No error details")[:100]
                            logger.info(
                                f" Item {item.get('_id', 'unknown')} added back to queue "
                                f"(retry count: {retry_count + 1}/{self.max_retry_rounds}, error: {error_display})"
                            )
                        else:
                            # Exceeded maximum retry count: mark as final failure
                            self.stats["failed_items"] += 1
                            error_display = (result.error or "No error details")[:100]
                            logger.warning(
                                f" Item {item.get('_id', 'unknown')} reached maximum retry count ({self.max_retry_rounds}), giving up. Error: {error_display}"
                            )
                            progress_bar.update(1)
                            progress_bar.refresh()
                
                # Output progress
                success_rate = (
                    self.stats["successful_items"] / self.stats["processed_items"]
                    if self.stats["processed_items"] > 0
                    else 0
                )
                
                completed_rate = (
                    (self.stats["successful_items"] + self.stats["failed_items"]) / self.stats["total_items"] * 100
                    if self.stats["total_items"] > 0
                    else 0
                )
                
                logger.info(
                    f"Batch {batch_number} completed | "
                    f"Success: {self.stats['successful_items']} | "
                    f"Failed: {self.stats['failed_items']} | "
                    f"Queue remaining: {queue_size} | "
                    f"Success rate: {success_rate:.2%} | "
                    f"Completion: {completed_rate:.1f}%"
                )
        finally:
            progress_bar.close()

        
        if self.cfg.mode=='new':
            save_jsonl(updated_profiles, self.cfg.task_config.outpath + "/anonymized_results.jsonl")
        else:
           finished_profiles = self.p_results["finished_profiles"]
           finished_profiles.extend(updated_profiles)
           save_jsonl(finished_profiles, self.cfg.task_config.outpath + "/anonymized_results.jsonl")
        
        # Output final statistics
        logger.info(
            f"Processing complete - Total: {self.stats['total_items']}, "
            f"Successful: {self.stats['successful_items']}, "
            f"Failed: {self.stats['failed_items']}, "
            f"Total retries: {self.stats['retry_count']}, "
            f"Total batches: {batch_number}"
        )
        
        return self.stats
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        llm_stats = self.llm_tool.get_performance_stats()
        
        return {
            "processing_stats": self.stats,
            "llm_stats": llm_stats,
        }
    
    async def close(self):
        """Close connections and clean up resources"""
        await self.llm_tool.close()
        self.mongo.disconnect()
        logger.info("Async anonymization processor closed")
    


async def main():
    """Main function"""
    processor = None
    try:
        # Create async processor
        processor = AsyncPIECAnonymizer(
            provider='deepseek',
            model='deepseek-reasoner',
            max_concurrent_requests=100,
            batch_size=100,
            request_timeout=1000.0,
            update_key="anonymized_labled_en_v2",
            policy_version="4.0",
            policy_language="en",
            prompt_template="intent_anonymizers_v2",
            max_retry_rounds=6,
            anonymized_which_parts="User Response",
        )
        
        # Process data
        data_path = "./dataset/anonymization/personalreddit/Reddit_synthetic/synthetic_dataset.jsonl"
        stats = await processor.process_all_data(
            collection_name="personal_reddit",
            limit=None,  # Process all data, can set limit like limit=10
            update_db=False,
        )
        
        # Output performance report
        performance_report = processor.get_performance_stats()
        logger.info(f"Performance report: {json.dumps(performance_report, indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        if processor:
            await processor.close()


if __name__ == "__main__":
    # Run async main function
    asyncio.run(main())
