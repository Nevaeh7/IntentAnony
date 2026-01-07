import sys
import os
import asyncio
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prompt_kits.policy_manager import get_policy_manager, reload_policies

from anonymized.anonymizers.intent_evidence_anonymizer import AsyncPIECAnonymizer
from anonymized.anonymizers.intent_anonymizer import AsyncIntentAnonymizer
from loguru import logger
from utils.mongo_utils import MongoDBConnector
from utils.x_utils import load_jsonl,save_json
from utils.dataset_utils import *
from prompt_kits.prompt_manager_final import PromptManager
from llm_tools.async_openai_tool import create_async_any_tool
from infer_attack.llm_attack import LLMPrivacyAttacker
from pu_eval.async_eval_utility import calculate_utility_all_score
from pu_eval.eval_infer_attack import async_evaluate_profiles_privacy,calculate_privacy_score

mongo = MongoDBConnector()
mongo.connect()


async def run_anon_infer_eval(cfg, p_results):
    """Main function"""
    processor = None
    try:

        anony_config = cfg.task_config.anon_model
        adversary_attack_config = cfg.task_config.adversary_attack_model
        piec_config = cfg.task_config.piec_model
        infer_config = cfg.task_config.inference_model
        judge_config = cfg.task_config.judge_infer_model
        utility_config = cfg.task_config.utility_model
        temperature = cfg.task_config.anon_model.args['temperature']


        anony_policy = get_policy_manager().get(
            version=anony_config.prompt_policy_version,
            language=anony_config.prompt_language,
        )


        anony_llm_model = create_async_any_tool(
            model=anony_config.name,
            provider=anony_config.provider,
            max_concurrent_requests=cfg.task_config.anonymizer.max_workers,
            temperature=temperature,
            api_type='chat'
        )
        judge_llm_model = create_async_any_tool(
            model=judge_config.name,
            provider=judge_config.provider,
            max_concurrent_requests=cfg.task_config.anonymizer.max_workers,
            temperature=temperature

        )
        infer_llm_model = create_async_any_tool(
            model=infer_config.name,
            provider=infer_config.provider,
            max_concurrent_requests=cfg.task_config.anonymizer.max_workers,
            temperature=temperature
        )
        utility_llm_model = create_async_any_tool(
            model=utility_config.name,
            provider=utility_config.provider,
            max_concurrent_requests=cfg.task_config.anonymizer.max_workers,
            temperature=temperature
        )

        anony_prompt_manager = PromptManager(
            default_category=anony_config.prompt_category,
            default_language=anony_config.prompt_language,
        )
        infer_prompt_manager = PromptManager(
            default_category=infer_config.prompt_category,
            default_language=infer_config.prompt_language,
        )
        judge_prompt_manager = PromptManager(
            default_category=judge_config.prompt_category,
            default_language=judge_config.prompt_language,
        )
        utility_prompt_manager = PromptManager(
            default_category=utility_config.prompt_category,
            default_language=utility_config.prompt_language
        )
        anony_models = [cfg.task_config.anonymizer.anon_model_name]

        # Create async anonymization processor for async anonymization of profiles

        adversary_attack_llm_model = create_async_any_tool(
            model=adversary_attack_config.name,
            provider=adversary_attack_config.provider,
            max_concurrent_requests=cfg.task_config.anonymizer.max_workers,
            temperature=adversary_attack_config.args['temperature']
        )
        piec_llm_model = create_async_any_tool(
            model=piec_config.name,
            provider=piec_config.provider,
            max_concurrent_requests=cfg.task_config.anonymizer.max_workers,
            temperature=piec_config.args['temperature']
        )

        processor = AsyncPIECAnonymizer(
            cfg=cfg,
            max_concurrent_requests=cfg.task_config.anonymizer.max_workers,
            batch_size=cfg.task_config.anonymizer.batch_size,
            anony_policy=anony_policy,
            anony_prompt_manager=anony_prompt_manager,
            anony_llm_model=anony_llm_model,
            adversary_attack_model_cfg=adversary_attack_config,
            piec_model_cfg=piec_config,
            adversary_attack_model=adversary_attack_llm_model,
            piec_model=piec_llm_model,
            max_retry_rounds=6,
            collection_name=cfg.collection_name,
            update_db=cfg.task_config.update_db,
            p_results=p_results,
            is_pre_iiv=cfg.task_config.anonymizer.is_pre_iiv,
        )
        
        logger.info("Starting anonymization process...")
        if cfg.mode != 'only_evaluate':
            stats = await processor.process_all_data(
                data_path=cfg.task_config.profile_path,
            )
            performance_report = processor.get_performance_stats()
            logger.info(
                f"Performance report: {json.dumps(performance_report, indent=4, ensure_ascii=False)}"
            )

            # Attack model performs privacy attacks on anonymized text for subsequent privacy security evaluation
        updated_profiles = load_jsonl(cfg.task_config.outpath + "/anonymized_results.jsonl")
        updated_profiles = prepare_datasets(updated_profiles, cfg.dataset_name, cfg.task_config.anonymizer.anon_model_name)

        privacy_attacker = LLMPrivacyAttacker(
        cfg=cfg,
        infer_llm_model=infer_llm_model,
        infer_prompt_manager=infer_prompt_manager,
        )
        updated_profiles, all_items = await privacy_attacker.infer_profiles(updated_profiles)

            # updated_profiles = load_jsonl(cfg.task_config.outpath + "/infer_attack_results.jsonl")
            
            
            ### Evaluate model privacy security capability
        updated_profiles, all_items = await async_evaluate_profiles_privacy(
            cfg=cfg,
            profiles=updated_profiles,
            judge_llm_model=judge_llm_model,
            infer_llm_model=infer_llm_model,
            infer_prompt_manager=infer_prompt_manager,
            judge_prompt_manager=judge_prompt_manager,
            anony_models=anony_models,
            max_concurrent=cfg.task_config.anonymizer.max_workers,  # Adjust concurrency based on actual situation
            update_db=cfg.task_config.update_db,
            need_infer_attack=False,  # Set to False if inference results already exist
            
        )
        # all_items = load_jsonl(cfg.task_config.outpath + "/judge_privacy_all_infer_items.jsonl")
        privacy_score_result = calculate_privacy_score(all_items)
        # else:
            # updated_profiles = load_jsonl(cfg.task_config.outpath + "/anonymized_results.jsonl")
            # updated_profiles = load_jsonl(cfg.task_config.outpath + "/judge_privacy_profiles.jsonl")

        ### Evaluate utility of anonymized text
        updated_profiles, utility_score_result = await calculate_utility_all_score(
            cfg=cfg,
            profiles=updated_profiles,
            utility_llm_model=utility_llm_model,
            utility_prompt_manager=utility_prompt_manager,
            anon_model_name=anony_models[0],
            update_db=cfg.task_config.update_db,
        )
        ### Persist privacy security and text utility evaluation results
        eval_final_result = {
            'anon_model_name':anony_models[0],
            'privacy_scores': privacy_score_result,
            'utility_scores':utility_score_result
        }
        save_json(eval_final_result, f"{cfg.task_config.outpath}/eval_final_result.json")


    except Exception as e:
        logger.error(f"Error occurred during processing: {e}")
        import traceback

        logger.error(traceback.format_exc())
    finally:
        if processor:
            await processor.close()

