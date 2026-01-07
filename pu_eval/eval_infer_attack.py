"""Inference attack evaluation module: Use LLM to infer user privacy attributes from anonymized text."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from loguru import logger

# Prefer sklearn metrics calculation, fallback to manual implementation if unavailable
try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except ImportError:  # pragma: no cover - Fallback logic when sklearn is missing in environment
    accuracy_score = None  # type: ignore[assignment]
    precision_recall_fscore_support = None  # type: ignore[assignment]

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_tools.async_openai_tool import TaskResult, create_async_any_tool
from prompt_kits.prompt_manager_final import PromptManager
from pu_eval.eval_privacy import check_infer_correctness
from utils.mongo_utils import MongoDBConnector
from utils.x_utils import save_jsonl
from tqdm import tqdm
from infer_attack.llm_attack import LLMPrivacyAttacker, InferenceConfig
from utils.dataset_utils import *
# Python 3.10+ supports anext, otherwise use manual implementation
if sys.version_info >= (3, 10):
    from builtins import anext
else:
    # Python < 3.10 compatibility
    async def anext(aiterator: AsyncGenerator) -> Any:
        """Async iterator next for Python < 3.10 compatibility."""
        return await aiterator.__anext__()


async def evaluate_each_profile_privacy(
    profile: Dict[str, Any],
    judge_llm_model: Any,
    infer_llm_model: Any,
    infer_prompt_manager: Any,
    judge_prompt_manager: Any,
    anony_models: List[str],
    decider: str = 'model',
    need_infer_attack: bool = False,
    privacy_attacker: Optional[LLMPrivacyAttacker] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluate privacy inference accuracy for a single profile and store results in profile.
    
    Args:
        profile: Single profile (will be modified, adding evaluation results)
        judge_llm_model: LLM model instance for judgment
        infer_llm_model: LLM model instance for inference
        infer_prompt_manager: Inference prompt manager instance
        judge_prompt_manager: Judgment prompt manager instance
        anony_models: List of anonymization models
        decider: Decision method, defaults to 'model'
        need_infer_attack: Whether to execute inference attack
        
    Returns:
        (profile, profile_items): Updated profile and evaluation result items list
    """
    profile_items: List[Dict[str, Any]] = []
    
    # If inference attack is needed or profile has no inference results
    # Determine if inference attack needs to be executed
    profile.setdefault('infers', {})

    if need_infer_attack or profile['infers'].get(anony_models[0],None) is None:
        logger.debug(f"Profile {profile.get('_id', 'unknown')} does not have infer_attributes for {anony_models[0]}, performing inference...")
            
        answer = await privacy_attacker.llm_infer_attributes(profile=profile)
        profile['infers'][anony_models[0]] = answer

    gt_items = get_gt_pred_item(profile, anony_models)
    profile_items.extend(gt_items)

    # Concurrently evaluate all PII types
    tasks = []
    for item in profile_items:
        pii_type = item['pii_type']
        task = check_infer_correctness(
            item,
            pii_type,
            judge_llm_model,
            judge_prompt_manager,
            decider=decider
        )
        tasks.append((item, task))
    
    # Concurrently execute all judgment tasks
    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
    
    # Update results
    for i, ((item, _), is_correct) in enumerate(zip(tasks, results)):
        if isinstance(is_correct, Exception):
            import traceback
            logger.error(f"Error checking correctness for {item.get('pii_type')}: {is_correct}")
            item['is_correct'] = False
        else:
            item['is_correct'] = is_correct
        profile_items[i] = item

    eval_ans = {
        'items': profile_items,
        'decider': decider,
        'anony_model': anony_models[0],
        'total_items': len(profile_items),
        'correct_count': sum(1 for item in profile_items if item.get('is_correct', False)),
    }
    # Store evaluation results in profile
    profile.setdefault('eval_privacy', {})
    profile.setdefault('eval', {}).setdefault(anony_models[0], {}).setdefault('eval_privacy', {})
    profile['eval'][anony_models[0]]['eval_privacy'] = eval_ans


    eval_key = f"{anony_models[0]}_{decider}"
    profile['eval_privacy'][eval_key] = eval_ans
    # Calculate accuracy
    if profile_items:
        profile['eval_privacy'][eval_key]['accuracy'] = (
            profile['eval_privacy'][eval_key]['correct_count'] / len(profile_items)
        )

    return profile, profile_items

async def evaluate_profiles_privacy(
    profiles: List[Dict[str, Any]],
    judge_llm_model: Any,
    infer_llm_model: Any,
    prompt_manager: Any,
    anony_models: List[str],
    decider: str = 'model',
    need_infer_attack: bool = False,
) -> List[Dict[str, Any]]:
    """
    Serial evaluation of privacy inference accuracy for profiles (compatible with old interface).
    
    Args:
        profiles: List of profiles
        judge_llm_model: LLM model instance for judgment
        infer_llm_model: LLM model instance for inference
        prompt_manager: Prompt manager instance (used for both inference and judgment)
        anony_models: List of anonymization models
        decider: Decision method, defaults to 'model'
        need_infer_attack: Whether to execute inference attack
        
    Returns:
        List of items containing evaluation results
    """
    _, all_items = await async_evaluate_profiles_privacy(
        profiles=profiles,
        judge_llm_model=judge_llm_model,
        infer_llm_model=infer_llm_model,
        infer_prompt_manager=prompt_manager,
        judge_prompt_manager=prompt_manager,
        anony_models=anony_models,
        decider=decider,
        need_infer_attack=need_infer_attack,
        max_concurrent=1,  # Serial processing
    )
    return all_items

async def async_evaluate_profiles_privacy(
    cfg: Any = None,
    profiles: List[Dict[str, Any]] = None,
    judge_llm_model: Any = None,
    infer_llm_model: Any = None,
    infer_prompt_manager: Any = None,
    judge_prompt_manager: Any = None,
    anony_models: List[str] = None,
    decider: str = 'model',
    need_infer_attack: bool = False,
    max_concurrent: int = 10,
    batch_size: Optional[int] = None,
    update_db: bool = False,
    mongo: Any =None,
    collection_name: Any =None,
    db_batch_size: int = 50,
    infer_save_path: Optional[str] = None,
    judge_privacy_save_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Efficient async batch evaluation of privacy inference accuracy for profiles and store results in profiles.
    
    Uses concurrency control and batch processing mechanisms to significantly improve processing speed.
    
    Args:
        profiles: List of profiles (will be modified, adding evaluation results)
        judge_llm_model: LLM model instance for judgment
        infer_llm_model: LLM model instance for inference
        infer_prompt_manager: Inference prompt manager instance
        judge_prompt_manager: Judgment prompt manager instance
        anony_models: List of anonymization models
        decider: Decision method, defaults to 'model'
        need_infer_attack: Whether to execute inference attack
        max_concurrent: Maximum concurrency, controls number of profiles processed simultaneously (default 10)
        batch_size: Batch size, if None then use max_concurrent
        mongo: MongoDB connector instance (optional)
        collection_name: MongoDB collection name (optional)
        update_db: Whether to update to MongoDB (default False)
        db_batch_size: Database batch update size (default 50)
        
    Returns:
        (updated_profiles, all_items): Updated profiles list and all evaluation items list
    """
    if not profiles:
        logger.warning("No profiles to evaluate")
        return [], []
    
    total_profiles = len(profiles)
    logger.info(f"Starting batch evaluation: Total={total_profiles}, Max concurrent={max_concurrent}")
    
    if batch_size is None:
        batch_size = max_concurrent
        
    if judge_privacy_save_path is None:
        judge_privacy_save_path = cfg.task_config.outpath
    
    if infer_save_path is None:
        infer_save_path = cfg.task_config.outpath

    if update_db:
        mongo = MongoDBConnector()
        mongo.connect()
        collection_name = cfg.collection_name
    else:
        mongo = None
        collection_name = None

    
    # Use semaphore to control concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    privacy_attacker = LLMPrivacyAttacker(
        cfg=cfg,
        infer_llm_model=infer_llm_model,
        infer_prompt_manager=infer_prompt_manager,
        config=InferenceConfig(
            anony_models=anony_models,
            need_infer_attack=need_infer_attack,
            max_concurrent=max_concurrent,
            batch_size=batch_size,
            update_db=update_db,
            db_batch_size=db_batch_size,
            save_path=infer_save_path,
        ),
        mongo=mongo,
        collection_name=collection_name,
    )
    
    async def process_single_profile(profile: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Process single profile with concurrency control"""
        async with semaphore:
            try:
                updated_profile, items = await evaluate_each_profile_privacy(
                    profile=profile,
                    judge_llm_model=judge_llm_model,
                    infer_llm_model=infer_llm_model,
                    infer_prompt_manager=infer_prompt_manager,
                    judge_prompt_manager=judge_prompt_manager,
                    anony_models=anony_models,
                    decider=decider,
                    need_infer_attack=need_infer_attack,
                    privacy_attacker=privacy_attacker,
                )
                return updated_profile, items
            except Exception as e:
                import traceback
                logger.error(traceback.format_exc())
                profile_id = profile.get('_id', 'unknown')
                logger.error(f"Error processing profile {profile_id}: {e}")
                return profile, []
            finally:
                profile_bar.update(1)
    
    # Process in batches to avoid creating too many tasks at once
    all_items: List[Dict[str, Any]] = []
    updated_profiles: List[Dict[str, Any]] = []
    total_batches = (total_profiles + batch_size - 1) // batch_size
    
    profile_bar = tqdm(total=total_profiles, desc="Profiles processed", leave=False)

    try:
        for batch_num, batch_idx in enumerate(
            tqdm(
                range(0, total_profiles, batch_size),
                total=total_batches,
                desc="Evaluation batches",
                leave=False,
            ),
            start=1,
        ):
            batch_end = min(batch_idx + batch_size, total_profiles)
            batch_profiles = profiles[batch_idx:batch_end]

            logger.debug(f"Processing batch {batch_num}/{total_batches}: {len(batch_profiles)} profiles")

            # Create batch tasks
            tasks = [process_single_profile(profile) for profile in batch_profiles]

            # Execute batch tasks concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            batch_updated_profiles = []
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    profile_id = batch_profiles[i].get('_id', 'unknown')
                    logger.error(f"Batch {batch_num} profile {profile_id} processing exception: {result}")
                    batch_updated_profiles.append(batch_profiles[i])
                else:
                    updated_profile, items = result
                    batch_updated_profiles.append(updated_profile)
                    all_items.extend(items)

            updated_profiles.extend(batch_updated_profiles)

            # If database update is enabled, batch save
            if update_db and mongo and collection_name and batch_updated_profiles:
                try:
                    mongo.batch_update_db_items(
                        collection_name=collection_name,
                        items=batch_updated_profiles,
                        batch_size=db_batch_size
                    )
                    logger.debug(f"Batch {batch_num} saved to MongoDB")
                except Exception as e:
                    logger.error(f"Batch {batch_num} failed to save to MongoDB: {e}")

            # Progress is updated one by one in process_single_profile, no need for additional update
    finally:
        profile_bar.close()
    
    logger.success(
        f"Batch evaluation complete: Processed {total_profiles} profiles, "
        f"generated {len(all_items)} evaluation items"
    )
    
    # If database update is enabled but there are unsaved profiles, save once more
    if update_db and mongo and collection_name and updated_profiles:
        try:
            mongo.batch_update_db_items(
                collection_name=collection_name,
                items=updated_profiles,
                batch_size=db_batch_size
            )
            logger.success("All profiles saved to MongoDB")
        except Exception as e:
            logger.error(f"Final save to MongoDB failed: {e}")

    if judge_privacy_save_path:
        save_jsonl(updated_profiles, f"{judge_privacy_save_path}/judge_privacy_profiles.jsonl")
        save_jsonl(all_items, f"{judge_privacy_save_path}/judge_privacy_all_infer_items.jsonl")
    
    return updated_profiles, all_items
    
def calculate_privacy_score(all_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate privacy score (accuracy) for each PII type and overall
    """
    if not all_items:
        return {
            "overall_accuracy": 0.0,
            "per_pii_type_accuracy": {},
        }

    per_type_scores: Dict[str, List[float]] = {}

    for item in all_items:
        pii_type = item.get("pii_type", "unknown")
        raw_score = item.get("is_correct", [0])
        try:
            if isinstance(raw_score, list):
                raw_score = raw_score[0]
            score = int(raw_score)
        except (TypeError, ValueError):
            score = 0
            # score = 1.0 if raw_score else 0.0
        score = max(0.0, min(1.0, score))
        # Treat partial credit (<1.0) as incorrect
        score = 1 if score >= 1.0 else 0.0

        scores = per_type_scores.setdefault(pii_type, [])
        scores.append(score)


    def compute_metrics(scores: List[float]) -> Dict[str, float]:
        total = len(scores)
        if total == 0:
            return {
                "acc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
            }

        tp = sum(scores)
        fp = total - tp
        fn = total - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        acc = tp / total if total else 0.0

        return {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": total,
        }

    per_type_metrics = {
        pii_type: compute_metrics(scores)
        for pii_type, scores in per_type_scores.items()
    }
    all_scores = [score for scores in per_type_scores.values() for score in scores]
    overall_metrics = compute_metrics(all_scores)

    result = {
        "overall_metrics": overall_metrics,
        "per_pii_type_metrics": per_type_metrics,
        "overall_accuracy": overall_metrics["acc"],
        "per_pii_type_accuracy": {
            pii_type: metrics["acc"] for pii_type, metrics in per_type_metrics.items()
        },
    }
    logger.success(json.dumps(result,indent=4))
    return result

if __name__ == '__main__':
    mongo = MongoDBConnector()
    mongo.connect()
    query={ 'eval_privacy.intent_anonymization_model': { '$exists': False } }

    datas = mongo.read_data('personal_reddit', query=query)

    infer_llm_model = create_async_any_tool(
        model='doubao-seed-1-6-lite-251015',
        provider='seed',
    )

    judge_llm_model = create_async_any_tool(
        model='deepseek-chat',
        provider='deepseek',
    )

    prompt_manager = PromptManager(
        default_category='infer_v3',
        default_language='en'
    )
    judge_prompt_manager = PromptManager(
        default_category='eval_attributes',
        default_language='en'
    )

    # profile = datas[0]
    # answer = asyncio.run(
    #     llm_infer_attributes(
    #         profile,
    #         prompt_manager=prompt_manager,
    #         llm_model=infer_llm_model
    #     )
    # )

    # item.setdefault('infers', {})['intent_anonymization'] = answer
    # profiles = [item]
    anony_models = ['intent_anonymization']
    updated_profiles, all_items = asyncio.run(
        async_evaluate_profiles_privacy(
            profiles=datas,
            judge_llm_model=judge_llm_model,
            infer_llm_model=infer_llm_model,
            infer_prompt_manager=prompt_manager,
            judge_prompt_manager=judge_prompt_manager,
            anony_models=anony_models,
            max_concurrent=50,  # Can adjust concurrency based on actual situation
            need_infer_attack=False,  # If inference results already exist, set to False
            mongo=mongo,  # MongoDB connector
            collection_name='personal_reddit',  # Collection name
            update_db=True,  # Whether to save to database
            db_batch_size=50,  # Database batch update size
        )
    )
    logger.success(
        f"Evaluation complete: processed {len(updated_profiles)} profiles, "
        f"total {len(all_items)} evaluation items"
    )

