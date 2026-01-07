from typing import Dict, Any, Optional, AsyncGenerator, List
import asyncio
import json
import os

from loguru import logger

# Set HuggingFace mirror address
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from prompt_kits.policy_manager import get_policy_manager, reload_policies
from prompt_kits.prompt_manager_final import get_manager
from utils.string_utils import compute_bleu, compute_rouge
from utils.mongo_utils import MongoDBConnector
from utils.x_utils import parse_json_response, calculate_stats, save_jsonl
from llm_tools.async_openai_tool import create_async_any_tool, TaskResult, AsyncModelConfig
from llm_tools.openai_tool import create_any_tool
from tqdm import tqdm
from utils.mongo_utils import MongoDBConnector
async def llm_eval_utility(
    profile: Dict[str, Any] = None,
    original_string: str = None,
    latest_string: str = None,
    prompt_manager: Any = None,
    llm_model: Any = None,
    anon_model_name: str = 'unknown'
) -> AsyncGenerator[Any, None]:
    """
    Use LLM to evaluate text utility.
    
    Args:
        profile: Configuration dictionary containing original and anonymized text
        prompt_manager: Prompt manager
        llm_model: LLM model instance
        
    Yields:
        LLM evaluation response result (TaskResult)
    """
    try:
        if profile is not None:
            original_string = profile.get('user_text', '')
            anonymized_results = profile.get("anonymized_results", {})
            anonymized = anonymized_results.get(anon_model_name, {})

            if isinstance(anonymized, dict):
                latest_string = anonymized.get("anonymized_text", "")
            elif isinstance(anonymized, str):
                latest_string = anonymized
            elif anonymized:
                latest_string = str(anonymized)
        elif original_string is None or latest_string is None:
            raise ValueError("Missing required fields: original_string or latest_string")
        messages = prompt_manager.get_messages(
            original_string=original_string, 
            latest_string=latest_string
        )
        
        # Use async API for evaluation
        response = await llm_model.async_chat_completion(messages)
        logger.debug(f"LLM evaluation response received: success={response.success}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in llm_eval_utility: {e}")
        raise


async def llm_judge_utility(
    profile: Dict[str, Any] = None,
    original_string: str = None,
    latest_string: str = None,
    prompt_manager: Any = None,
    llm_model: Any = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    anon_model_name: str = 'unknown'
) -> Optional[Dict[str, Any]]:
    """
    Use LLM to evaluate text utility with retry mechanism.
    
    Args:
        profile: Configuration dictionary containing original and anonymized text (optional)
        original_string: Original text (optional, extracted from profile if profile provided)
        latest_string: Anonymized text (optional, extracted from profile if profile provided)
        prompt_manager: Prompt manager
        llm_model: LLM model instance
        max_retries: Maximum retry count (default 3)
        retry_delay: Initial retry delay (seconds, default 1.0)
        
    Returns:
        LLM evaluation result dictionary, returns None on failure
    """
    retry_count = 0
    success = False
    
    while retry_count <= max_retries and not success:
        try:
            response = await llm_eval_utility(
                profile=profile,
                original_string=original_string,
                latest_string=latest_string,
                prompt_manager=prompt_manager,
                llm_model=llm_model,
                anon_model_name=anon_model_name
            )
            
            if response and response.success:
                # Check response format

                # Parse LLM response
                if llm_model.default_config.api_type == "responses":
                    content = response.result.output_text.strip()
                else:
                    if not hasattr(response.result, 'choices') or not response.result.choices:
                        raise ValueError(f"Invalid response format: {type(response.result)}")
                    content = response.result.choices[0].message.content.strip()
                
                if not content:
                    raise ValueError("Empty response content")
                
                try:
                    # Try to parse JSON format response
                    llm_eval_result = await asyncio.to_thread(parse_json_response, content)
                    # logger.success(f"LLM evaluation completed successfully (attempt {retry_count + 1}): {llm_eval_result}")
                    # logger.info(f"LLM evaluation completed successfully (attempt {retry_count + 1})")
                    return llm_eval_result
                except (json.JSONDecodeError, ValueError) as parse_error:
                    # JSON parsing failed, but if there's content, save raw text
                    if content and content.strip():
                        logger.warning(
                            f"LLM evaluation completed but JSON parse failed (attempt {retry_count + 1}): {parse_error}. "
                            f"Saved raw response."
                        )
                        raise ValueError(f"Empty or invalid content: {parse_error}")
                    else:
                        # Empty content, need to retry
                        raise ValueError(f"Empty or invalid content: {parse_error}")
            else:
                # API call failed
                error_msg = response.error if hasattr(response, 'error') and response.error else "Unknown error"
                raise Exception(f"LLM API call failed: {error_msg}")
                
        except Exception as e:
            retry_count += 1
            
            if retry_count <= max_retries:
                # Calculate exponential backoff delay
                delay = retry_delay * (2 ** (retry_count - 1))
                logger.warning(
                    f"LLM evaluation failed (attempt {retry_count}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                # await asyncio.sleep(delay)
            else:
                # Reached maximum retry count
                logger.error(
                    f"LLM evaluation failed after {max_retries + 1} attempts: {e}"
                )
                return None
    
    return None

async def score_utility_each(
    profile: Dict[str, Any] = None,
    original_string: str = None,
    latest_string: str = None,
    prompt_manager: Any = None,
    llm_model: Any = None,
    is_llm_judge: bool = True,
    llm_retry: int = 3,
    anon_model_name: str = 'unknown',
    update_db: bool = False,
    collection_name: str = 'unknown',
    mongo: MongoDBConnector = None,
) -> Dict[str, Any]:
    """
    Evaluate utility score for a single configuration, including BLEU, ROUGE and optional LLM evaluation.
    
    Args:
        profile: Configuration dictionary containing original and anonymized text
        prompt_manager: Prompt manager
        llm_model: LLM model instance
        is_llm_judge: Whether to use LLM for evaluation
        llm_retry: Maximum retry count when LLM evaluation fails (default 3)
        
    Returns:
        Updated profile dictionary containing evaluation results
    """
    try:
        # Extract text
        if profile is not None:
            original_string = profile.get('user_text', '')
            anonymized_results = profile.get("anonymized_results", {})
            anonymized = anonymized_results.get(anon_model_name, {})

            if isinstance(anonymized, dict):
                latest_string = anonymized.get("anonymized_text", "")
            elif isinstance(anonymized, str):
                latest_string = anonymized
            elif anonymized:
                latest_string = str(anonymized)
        elif original_string is None or latest_string is None:
            raise ValueError("Missing required fields: original_string or latest_string")
        
        # Calculate BLEU/ROUGE scores (moved to thread pool to avoid blocking event loop)
        bleu_score, rouge_scores = await asyncio.gather(
            asyncio.to_thread(compute_bleu, original_string, latest_string),
            asyncio.to_thread(compute_rouge, original_string, [latest_string]),
        )
        
        # Extract ROUGE metrics
        rouge_metrics = {}
        if rouge_scores and len(rouge_scores) > 0:
            rouge_dict = rouge_scores[0]
            rouge_metrics = {
                'rouge1': {
                    'precision': rouge_dict['rouge1'].precision,
                    'recall': rouge_dict['rouge1'].recall,
                    'fmeasure': rouge_dict['rouge1'].fmeasure
                },
                'rougeL': {
                    'precision': rouge_dict['rougeL'].precision,
                    'recall': rouge_dict['rougeL'].recall,
                    'fmeasure': rouge_dict['rougeL'].fmeasure
                },
                'rougeLsum': {
                    'precision': rouge_dict['rougeLsum'].precision,
                    'recall': rouge_dict['rougeLsum'].recall,
                    'fmeasure': rouge_dict['rougeLsum'].fmeasure
                }
            }
        
        # Initialize evaluation result dictionary
        eval_item = profile.setdefault('eval', {}).setdefault(anon_model_name, {})
        eval_utility = eval_item.setdefault('eval_utility', {})
        eval_utility['bleu'] = bleu_score
        eval_utility['rouge'] = rouge_metrics
        eval_utility['llm_judge'] = eval_utility.get('llm_judge', {})  # Preserve existing structure for subsequent writes
        # LLM evaluation (if enabled)
        if is_llm_judge:
            eval_item['eval_utility']['llm_judge'][llm_model.default_config.name] = await llm_judge_utility(
                original_string=original_string,
                latest_string=latest_string,
                prompt_manager=prompt_manager,
                llm_model=llm_model,
                max_retries=llm_retry,
                anon_model_name=anon_model_name
            )
        
        # Update profile
        profile['eval'][anon_model_name] = eval_item
        if update_db:
            mongo.update_one_data(collection_name, profile)

        return profile
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error in score_utility_each: {e}")
        raise

async def calculate_utility_all_score(
    cfg: Any = None,
    anon_model_name: str = 'unknown',
    profiles: List[Dict[str, Any]] = None,
    utility_llm_model: Any = None,
    utility_prompt_manager: Any = None,
    is_llm_judge: bool = True,
    llm_retry: int = 3,
    update_db: bool = False,
    collection_name: str = None
) -> Dict[str, Any]:
    """
    Calculates aggregated utility scores (mean, total counts, etc. statistics) for all configurations.

    Args:
        profiles: List of configuration dictionaries containing original and anonymized text
        prompt_manager: Prompt manager
        llm_model: LLM model instance
        is_llm_judge: Whether to use LLM for evaluation
        llm_retry: Maximum retry attempts for LLM evaluation failure (default 3)
        evaluate_all: Whether to evaluate all profiles (if False, only aggregates existing evaluation results)
        
    Returns:
        Dict[str, Any]: Dictionary containing aggregated utility scores for all configurations, including:
            - total_count: Total count
            - valid_count: Valid evaluation count
            - bleu: BLEU score statistics (mean, std, min, max)
            - rouge: ROUGE score statistics (rouge1, rougeL, rougeLsum mean, std, min, max)
            - llm_judge: LLM evaluation statistics (readability, meaning, hallucinations mean, std, min, max)
    """
    mongo=None
    if update_db:
        mongo = MongoDBConnector()
        mongo.connect()
        collection_name = cfg.collection_name

    if not profiles:
        logger.warning("Empty profiles list")
        return {
            'total_count': 0,
            'valid_count': 0,
            'bleu': None,
            'rouge': None,
            'llm_judge': None
        }
    
    # If all profiles need to be evaluated
    utility_config = cfg.task_config.utility_model
    if utility_prompt_manager is None:
        utility_prompt_manager = get_manager(
            default_category=utility_config.prompt_category,
            default_language=utility_config.prompt_language
        )
    if utility_llm_model is None:
        utility_llm_model = create_async_any_tool(
            model=utility_config.name,
            provider=utility_config.provider,
            max_concurrent_requests=cfg.task_config.anonymizer.max_workers
        )
    logger.info(f"Evaluating {len(profiles)} profiles...")

    max_concurrent = (
        cfg.task_config.anonymizer.max_workers
        or 50
    )
    logger.info(f"max_concurrent: {max_concurrent}")
    semaphore = asyncio.Semaphore(max_concurrent)

    progress_bar = tqdm(
        total=len(profiles),
        desc="Evaluating utility scores",
        unit="profile",
        leave=False,
    )

    async def run_profile(idx: int, profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            async with semaphore:
                return await score_utility_each(
                    profile=profile,
                    anon_model_name=anon_model_name,
                    prompt_manager=utility_prompt_manager,
                    llm_model=utility_llm_model,
                    is_llm_judge=is_llm_judge,
                    llm_retry=llm_retry,
                    update_db=update_db,
                    collection_name=collection_name,
                    mongo=mongo,
                )
        finally:
            progress_bar.update(1)

    tasks = [
        asyncio.create_task(run_profile(idx, profile))
        for idx, profile in enumerate(profiles)
    ]

    try:
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        progress_bar.close()

    processed_profiles: List[Dict[str, Any]] = []
    for idx, result in enumerate(gather_results):
        if isinstance(result, Exception):
            logger.error(f"Utility evaluation failed for profile index {idx}: {result}")
            continue
        if result is not None:
            processed_profiles.append(result)

    profiles = processed_profiles
    
    # Collect all valid evaluation results
    bleu_scores = []
    rouge_scores = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeLsum': {'precision': [], 'recall': [], 'fmeasure': []}
    }
    llm_judge_scores = {
        'readability': [],
        'meaning': [],
        'hallucinations': []
    }
    
    valid_count = 0
    
    # Define constants
    ROUGE_TYPES = ['rouge1', 'rougeL', 'rougeLsum']
    ROUGE_METRICS = ['precision', 'recall', 'fmeasure']
    LLM_JUDGE_METRICS = ['readability', 'meaning', 'hallucinations']
    
    for profile in tqdm(profiles,desc='Calculating utility scores'):
        eval_utility = profile.get('eval', {}).get(anon_model_name, {}).get('eval_utility', {})
        
        if not eval_utility:
            continue
        
        valid_count += 1
        
        # Collect BLEU scores
        bleu_score = eval_utility.get('bleu')
        if bleu_score is not None:
            bleu_scores.append(bleu_score)
        
        # Collect ROUGE scores
        rouge_data = eval_utility.get('rouge', {})
        if rouge_data:
            for rouge_type in ROUGE_TYPES:
                rouge_type_data = rouge_data.get(rouge_type, {})
                if rouge_type_data:
                    for metric in ROUGE_METRICS:
                        metric_value = rouge_type_data.get(metric)
                        if metric_value is not None:
                            rouge_scores[rouge_type][metric].append(metric_value)
        
        # Collect LLM evaluation scores
        llm_judge_data = eval_utility.get('llm_judge',{}).get(utility_llm_model.default_config.name,{})
        if isinstance(llm_judge_data, dict) and 'raw_response' not in llm_judge_data:
            for metric in LLM_JUDGE_METRICS:
                metric_data = llm_judge_data.get(metric)
                if isinstance(metric_data, dict):
                    score_value = metric_data.get('score')
                    if score_value is not None:
                        llm_judge_scores[metric].append(score_value)
    

    # Aggregate BLEU statistics
    bleu_stats = calculate_stats(bleu_scores)
    
    # Aggregate ROUGE statistics
    rouge_stats = {
        rouge_type: {
            metric: calculate_stats(rouge_scores[rouge_type][metric])
            for metric in ROUGE_METRICS
            if rouge_scores[rouge_type][metric]
        }
        for rouge_type in ROUGE_TYPES
        if any(rouge_scores[rouge_type][metric] for metric in ROUGE_METRICS)
    }
    
    # Aggregate LLM evaluation statistics
    llm_judge_stats = {
        metric: calculate_stats(llm_judge_scores[metric])
        for metric in LLM_JUDGE_METRICS
        if llm_judge_scores[metric]
    }
    llm_utility = {
        'readability': (llm_judge_stats['readability']['mean'] - 1) / 9,
        'meaning': (llm_judge_stats['meaning']['mean'] - 1) / 9,
        'hallucinations': llm_judge_stats['hallucinations']['mean']
    }
    llm_utility['mean'] =  sum(llm_utility.values()) / len(llm_utility)
    score_utility = {
        'bleu': bleu_stats['mean'],
        'rouge': rouge_stats['rouge1']['fmeasure']['mean'],
        'llm_judge': llm_utility['mean'],
        'mean': (bleu_stats['mean'] + rouge_stats['rouge1']['fmeasure']['mean'] + llm_utility['mean']) / 3
    }
    
    result = {
        'total_count': len(profiles),
        'valid_count': valid_count,
        # 'bleu': bleu_stats,
        # 'rouge': rouge_stats if rouge_stats else None,
        # 'llm_judge': llm_judge_stats if llm_judge_stats else None,
        'llm_utility': llm_utility if llm_utility else None,
        'score_utility': score_utility if score_utility else None,
        
    }
    # Format log output
    logger.success(json.dumps(result,indent=4))
    save_jsonl(profiles, f"{cfg.task_config.outpath}/eval_utility_profiles.jsonl")
    if update_db:
        mongo.batch_update_db_items(collection_name, profiles, batch_size=1000)
    # logger.success(result['llm_utility'])
    # logger.success(result['score_utility'])
    
    
    return profiles, result





async def calculate_utility_all_score_v0(
    cfg: Any = None,
    profiles: List[Dict[str, Any]] = None,
    prompt_manager: Any = None,
    llm_model: Any = None,
    is_llm_judge: bool = True,
    llm_retry: int = 3,
    evaluate_all: bool = False
) -> Dict[str, Any]:
    """
    Calculates aggregated utility scores (mean, total counts, etc. statistics) for all configurations.

    Args:
        profiles: List of configuration dictionaries containing original and anonymized text
        prompt_manager: Prompt manager
        llm_model: LLM model instance
        is_llm_judge: Whether to use LLM for evaluation
        llm_retry: Maximum retry attempts for LLM evaluation failure (default 3)
        evaluate_all: Whether to evaluate all profiles (if False, only aggregates existing evaluation results)
        
    Returns:
        Dict[str, Any]: Dictionary containing aggregated utility scores for all configurations, including:
            - total_count: Total count
            - valid_count: Valid evaluation count
            - bleu: BLEU score statistics (mean, std, min, max)
            - rouge: ROUGE score statistics (rouge1, rougeL, rougeLsum mean, std, min, max)
            - llm_judge: LLM evaluation statistics (readability, meaning, hallucinations mean, std, min, max)
    """
    if not profiles:
        logger.warning("Empty profiles list")
        return {
            'total_count': 0,
            'valid_count': 0,
            'bleu': None,
            'rouge': None,
            'llm_judge': None
        }
    
    # If all profiles need to be evaluated
    if evaluate_all:
        logger.info(f"Evaluating {len(profiles)} profiles...")
        tasks = [
            score_utility_each(
                profile=profile,
                prompt_manager=prompt_manager,
                llm_model=llm_model,
                is_llm_judge=is_llm_judge,
                llm_retry=llm_retry
            )
            for profile in profiles
        ]
        profiles = await asyncio.gather(*tasks, return_exceptions=True)
        # Filter out exception results
        profiles = [p for p in profiles if not isinstance(p, Exception)]
    
    # Collect all valid evaluation results
    bleu_scores = []
    rouge_scores = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeLsum': {'precision': [], 'recall': [], 'fmeasure': []}
    }
    llm_judge_scores = {
        'readability': [],
        'meaning': [],
        'hallucinations': []
    }
    
    valid_count = 0
    
    # Define constants
    ROUGE_TYPES = ['rouge1', 'rougeL', 'rougeLsum']
    ROUGE_METRICS = ['precision', 'recall', 'fmeasure']
    LLM_JUDGE_METRICS = ['readability', 'meaning', 'hallucinations']
    
    for profile in tqdm(profiles,desc='Calculating utility scores'):
        eval_utility = profile.get('eval', {}).get('eval_utility', {})
        
        if not eval_utility:
            continue
        
        valid_count += 1
        
        # Collect BLEU scores
        bleu_score = eval_utility.get('bleu')
        if bleu_score is not None:
            bleu_scores.append(bleu_score)
        
        # Collect ROUGE scores
        rouge_data = eval_utility.get('rouge', {})
        if rouge_data:
            for rouge_type in ROUGE_TYPES:
                rouge_type_data = rouge_data.get(rouge_type, {})
                if rouge_type_data:
                    for metric in ROUGE_METRICS:
                        metric_value = rouge_type_data.get(metric)
                        if metric_value is not None:
                            rouge_scores[rouge_type][metric].append(metric_value)
        
        # Collect LLM evaluation scores
        llm_judge_data = eval_utility.get('llm_judge',{}).get(llm_model.default_config.name,{})
        if isinstance(llm_judge_data, dict) and 'raw_response' not in llm_judge_data:
            for metric in LLM_JUDGE_METRICS:
                metric_data = llm_judge_data.get(metric)
                if isinstance(metric_data, dict):
                    score_value = metric_data.get('score')
                    if score_value is not None:
                        llm_judge_scores[metric].append(score_value)
    

    # Aggregate BLEU statistics
    bleu_stats = calculate_stats(bleu_scores)
    
    # Aggregate ROUGE statistics
    rouge_stats = {
        rouge_type: {
            metric: calculate_stats(rouge_scores[rouge_type][metric])
            for metric in ROUGE_METRICS
            if rouge_scores[rouge_type][metric]
        }
        for rouge_type in ROUGE_TYPES
        if any(rouge_scores[rouge_type][metric] for metric in ROUGE_METRICS)
    }
    
    # Aggregate LLM evaluation statistics
    llm_judge_stats = {
        metric: calculate_stats(llm_judge_scores[metric])
        for metric in LLM_JUDGE_METRICS
        if llm_judge_scores[metric]
    }
    llm_utility = {
        'readability': (llm_judge_stats['readability']['mean'] - 1) / 9,
        'meaning': (llm_judge_stats['meaning']['mean'] - 1) / 9,
        'hallucinations': llm_judge_stats['hallucinations']['mean']
    }
    llm_utility['mean'] =  sum(llm_utility.values()) / len(llm_utility)
    score_utility = {
        'bleu': bleu_stats['mean'],
        'rouge': rouge_stats['rouge1']['fmeasure']['mean'],
        'llm_judge': llm_utility['mean'],
        'mean': (bleu_stats['mean'] + rouge_stats['rouge1']['fmeasure']['mean'] + llm_utility['mean']) / 3
    }
    
    result = {
        'total_count': len(profiles),
        'valid_count': valid_count,
        # 'bleu': bleu_stats,
        # 'rouge': rouge_stats if rouge_stats else None,
        # 'llm_judge': llm_judge_stats if llm_judge_stats else None,
        'llm_utility': llm_utility if llm_utility else None,
        'score_utility': score_utility if score_utility else None,
        
    }
    
    # Format log output
    bleu_mean = f"{bleu_stats['mean']:.4f}" if bleu_stats else "N/A"
    rouge1_fmean = (
        f"{rouge_stats['rouge1']['fmeasure']['mean']:.4f}"
        if rouge_stats.get('rouge1', {}).get('fmeasure') else "N/A"
    )

    logger.success(json.dumps(result,indent=4))
    # logger.success(result['llm_utility'])
    # logger.success(result['score_utility'])
    
    
    return result

