"""Tool class for async privacy inference attacks."""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_tools.async_openai_tool import TaskResult, create_async_any_tool
from prompt_kits.prompt_manager_final import PromptManager
from utils.mongo_utils import MongoDBConnector
from utils.x_utils import parse_json_response, save_jsonl


if sys.version_info >= (3, 10):
    from builtins import anext  # type: ignore  # pragma: no cover - Platform differences
else:  # pragma: no cover - Python < 3.10 compatibility branch

    async def anext(aiterator: AsyncGenerator[TaskResult, None]) -> TaskResult:
        """Polyfill for anext on Python < 3.10."""

        return await aiterator.__anext__()


@dataclass
class InferenceConfig:
    """Batch inference configuration."""

    anony_models: List[str]
    need_infer_attack: bool = False
    max_concurrent: int = 50
    batch_size: Optional[int] = None
    update_db: bool = False
    db_batch_size: int = 50
    save_path: Optional[str] = None


class LLMPrivacyAttacker:
    """Manages LLM privacy inference attack workflow."""

    def __init__(
        self,
        cfg: Any = None,
        infer_llm_model: Any = None,
        infer_prompt_manager: PromptManager = None,
        config: InferenceConfig = None,
        mongo: Optional[MongoDBConnector] = None,
        collection_name: Optional[str] = None,
        anon_model_name: str = None,
    ) -> None:

        self.cfg = cfg
        if infer_llm_model is None:
            self.infer_llm_model = create_async_any_tool(
                model=self.cfg.task_config.inference_model.name,
                provider=self.cfg.task_config.inference_model.provider,
            )
        else:
            self.infer_llm_model = infer_llm_model
        if infer_prompt_manager is None:
            self.infer_prompt_manager = PromptManager(
                default_category=self.cfg.task_config.inference_model.prompt_category,
                default_language=self.cfg.task_config.inference_model.prompt_language,
            )
        else:
            self.infer_prompt_manager = infer_prompt_manager
        if config is None:
            self.config = InferenceConfig(
                anony_models=[self.cfg.task_config.anonymizer.anon_model_name],
                need_infer_attack=True,
                max_concurrent=self.cfg.task_config.anonymizer.max_workers,
                batch_size=self.cfg.task_config.anonymizer.batch_size,
                update_db=self.cfg.task_config.update_db,
                db_batch_size=self.cfg.task_config.anonymizer.batch_size,
                save_path=self.cfg.task_config.attack_outpath,
            )
        else:
            self.config = config
        self.mongo = mongo
        self.collection_name = collection_name
        self.anony_model_name = self.cfg.task_config.anonymizer.anon_model_name



    # ------------------------------------------------------------------
    # Low-level calls
    # ------------------------------------------------------------------
    async def _llm_attack_attributes(
        self,
        profile: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
        max_retries: int = 3,
    ) -> AsyncGenerator[TaskResult, None]:
        """Call LLM to perform privacy attribute inference on anonymized text."""

        if profile is not None and not context:
            # logger.success(f'profile: {profile}')
            
            anonymized_results = profile.get("anonymized_results", {})
            anonymized = anonymized_results.get(self.anony_model_name, {})

            if isinstance(anonymized, dict):
                context = anonymized.get("anonymized_text", "")
            elif isinstance(anonymized, str):
                context = anonymized
            elif anonymized:
                context = str(anonymized)

        if not context or not context.strip():
            raise ValueError(
                "Missing required field: context. "
                "Either pass `context` explicitly or provide a profile that contains "
                f"'anonymized_results.{self.anony_model_name}.anonymized_text'."
            )

        backoff_seconds = 1.0
        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                messages = self.infer_prompt_manager.get_messages(context=context.strip())
                if not messages:
                    raise ValueError("Failed to generate messages from prompt manager")

                response = await self.infer_llm_model.async_chat_completion(messages)
                if response and response.success:
                    logger.debug(f"LLM attack response received (attempt {attempt}/{max_retries})")
                    yield response
                    return

                logger.warning(
                    f"LLM attack failed (attempt {attempt}/{max_retries}) with success status "
                    f"{response.success if response else False}"
                )

            except Exception as exc:  # noqa: BLE001 - Catch all exceptions and retry
                last_error = exc
                logger.error(
                    f"Error in llm_attack_attributes (attempt {attempt}/{max_retries}) with error {exc}"
                )

            if attempt < max_retries:
                # await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 8.0)

        if last_error:
            raise last_error
        raise RuntimeError("LLM attack failed after all retries")

    async def llm_infer_attributes(
        self,
        profile: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
        max_retries: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """Execute privacy attribute inference and parse LLM response."""

        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                async_gen = self._llm_attack_attributes(
                    profile=profile,
                    context=context,
                    max_retries=max_retries,
                )
                response = await anext(async_gen)
            except StopAsyncIteration:
                response = None
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.error(
                    f"Error in llm_infer_attributes (attempt {attempt}/{max_retries}) with error {exc}"
                )
                # await asyncio.sleep(min(2 ** attempt, 8))
                continue

            if not response or not getattr(response, "success", False) or not getattr(response, "result", None):
                logger.warning(f"LLM inference returned empty response (attempt {attempt}/{max_retries})")
                # await asyncio.sleep(min(2 ** attempt, 8))
                continue

            try:
                if self.infer_llm_model.default_config.api_type == "responses":
                    content = response.result.output_text.strip()
                else:
                    content = response.result.choices[0].message.content.strip()
                result = parse_json_response(content)
                if result and result.get("instructions"):
                    logger.success(f"Successfully inferred {len(result['instructions'])} attributes")
                    return result
            except Exception as exc:  # noqa: BLE001 - Continue retrying on parse failure
                last_error = exc
                logger.error(
                    f"Failed to parse LLM response (attempt {attempt}/{max_retries}) with error {exc}"
                )
                # await asyncio.sleep(min(2 ** attempt, 8))
                continue

            logger.warning(
                f"Response missing 'instructions' field (attempt {attempt}/{max_retries})"
            )
            # await asyncio.sleep(min(2 ** attempt, 8))

        if last_error:
            logger.error(f"LLM inference failed after all retries: {last_error}")
        return None

    # ------------------------------------------------------------------
    # Profile level
    # ------------------------------------------------------------------
    @staticmethod
    def _should_infer(profile: Dict[str, Any], model_name: str, force: bool) -> bool:
        existing = profile.get("infers", {}).get(model_name)
        return force or not existing

    @staticmethod
    def _store_infer_result(profile: Dict[str, Any], model_name: str, result: Optional[Dict[str, Any]]) -> None:
        profile.setdefault("infers", {})[model_name] = result

    @staticmethod
    def _build_infer_item(
        profile: Dict[str, Any],
        model_name: str,
        result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "profile_id": profile.get("_id"),
            "anony_model": model_name,
            "instructions": (result or {}).get("instructions", []),
            "raw_result": result,
        }

    async def infer_profile(self, profile: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Execute privacy inference attack for a single profile."""

        profile.setdefault("infers", {})
        infer_items: List[Dict[str, Any]] = []

        for model_name in self.config.anony_models:
            if not self._should_infer(profile, model_name, self.config.need_infer_attack):
                infer_items.append(self._build_infer_item(profile, model_name, profile["infers"].get(model_name)))
                continue

            result = await self.llm_infer_attributes(profile=profile)
            self._store_infer_result(profile, model_name, result)
            infer_items.append(self._build_infer_item(profile, model_name, result))

        return profile, infer_items

    # ------------------------------------------------------------------
    # Batch workflow
    # ------------------------------------------------------------------
    async def infer_profiles(self, profiles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Batch execute privacy inference attacks."""

        if not profiles:
            logger.warning("No profiles to process")
            return [], []

        total_profiles = len(profiles)
        batch_size = self.config.batch_size or max(1, self.config.max_concurrent)

        logger.info(
            f"Start async inference: profiles={total_profiles}, max_concurrent={self.config.max_concurrent}, "
            f"batch_size={batch_size}"
        )

        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def process_single(profile: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
            async with semaphore:
                try:
                    return await self.infer_profile(profile)
                except Exception as exc:  # noqa: BLE001
                    profile_id = profile.get("_id", "unknown")
                    logger.error(f"Failed to infer profile {profile_id}: {exc}")
                    return profile, []
                finally:
                    profiles_bar.update(1)

        updated_profiles: List[Dict[str, Any]] = []
        infer_items: List[Dict[str, Any]] = []
        total_batches = (total_profiles + batch_size - 1) // batch_size

        profiles_bar = tqdm(total=total_profiles, desc="Profiles processed", leave=False)

        try:
            for batch_index, batch_start in enumerate(
                tqdm(
                    range(0, total_profiles, batch_size),
                    total=total_batches,
                    desc="Inference batches",
                    leave=False,
                ),
                start=1,
            ):
                batch_end = min(batch_start + batch_size, total_profiles)
                batch_profiles = profiles[batch_start:batch_end]

                logger.debug(f"Processing batch {batch_index}/{total_batches} ({len(batch_profiles)} profiles)")

                tasks = [process_single(profile) for profile in batch_profiles]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                batch_updated: List[Dict[str, Any]] = []
                for profile, result in zip(batch_profiles, batch_results):
                    if isinstance(result, Exception):
                        profile_id = profile.get("_id", "unknown")
                        logger.error(f"Batch {batch_index} profile {profile_id} raised exception: {result}")
                        batch_updated.append(profile)
                        continue

                    updated_profile, items = result
                    batch_updated.append(updated_profile)
                    infer_items.extend(items)

                updated_profiles.extend(batch_updated)

                if self.config.update_db and self.mongo and self.collection_name and batch_updated:
                    try:
                        self.mongo.batch_update_db_items(
                            self.collection_name,
                            batch_updated,
                            batch_size=self.config.db_batch_size,
                        )
                        logger.debug(f"Batch {batch_index} persisted to MongoDB")
                    except Exception as exc:  # noqa: BLE001
                        logger.error(f"Failed to persist batch {batch_index}: {exc}")

                # Progress is updated one by one in process_single, no need to add here
        finally:
            profiles_bar.close()

        if self.config.update_db and self.mongo and self.collection_name and updated_profiles:
            try:
                self.mongo.batch_update_db_items(
                    self.collection_name,
                    updated_profiles,
                    batch_size=self.config.db_batch_size,
                )
                logger.success("All profiles persisted to MongoDB")
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to persist profiles to MongoDB: {exc}")

        if self.config.save_path:
            output_dir = Path(self.config.save_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_jsonl(updated_profiles, str(output_dir / "infer_attack_results.jsonl"))

        logger.success(
            f"Async inference finished: profiles={len(updated_profiles)}, inference_items={len(infer_items)}"
        )

        return updated_profiles, infer_items


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _build_default_query(anony_models: List[str], force: bool) -> Dict[str, Any]:
    if force or not anony_models:
        return {}
    return {f"infers.{anony_models[0]}": {"$exists": False}}


def _parse_query(query: Optional[str]) -> Optional[Dict[str, Any]]:
    if not query:
        return None
    try:
        return json.loads(query)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise ValueError(f"Invalid JSON for --query: {exc}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run async privacy inference attacks")
    parser.add_argument("--collection", default="personal_reddit", help="MongoDB collection name")
    parser.add_argument("--mongo-host", default="localhost")
    parser.add_argument("--mongo-port", type=int, default=27017)
    parser.add_argument("--mongo-username")
    parser.add_argument("--mongo-password")
    parser.add_argument("--db-name", default="INS_DB")
    parser.add_argument("--limit", type=int, help="Limit number of profiles loaded from MongoDB")
    parser.add_argument("--query", help="Custom MongoDB query in JSON format")
    parser.add_argument("--anony-models", nargs="+", default=["intent_anonymization"], help="Anonymization model keys")
    parser.add_argument("--need-infer-attack", action="store_true", help="Force re-run inference even if results exist")
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--update-db", action="store_true", help="Persist results back to MongoDB")
    parser.add_argument("--db-batch-size", type=int, default=50)
    parser.add_argument("--save-path", help="Directory to dump inference results as JSONL")
    parser.add_argument("--infer-model", default="gpt-5")
    parser.add_argument("--infer-provider", default="openai")
    parser.add_argument("--prompt-category", default="infer_v3")
    parser.add_argument("--prompt-language", default="en")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    query_override = _parse_query(args.query)
    default_query = _build_default_query(args.anony_models, args.need_infer_attack)
    mongo_query = query_override if query_override is not None else default_query

    mongo = MongoDBConnector(
        host=args.mongo_host,
        port=args.mongo_port,
        username=args.mongo_username,
        password=args.mongo_password,
        db_name=args.db_name,
    )
    mongo.connect()

    logger.info(f"Loading profiles from MongoDB collection '{args.collection}'")
    profiles = mongo.read_data(args.collection, query=mongo_query, limit=args.limit)
    if not profiles:
        logger.warning("No profiles matched the query; exiting")
        return

    infer_llm_model = create_async_any_tool(
        model=args.infer_model,
        provider=args.infer_provider,
    )
    infer_prompt_manager = PromptManager(
        default_category=args.prompt_category,
        default_language=args.prompt_language,
    )

    config = InferenceConfig(
        anony_models=args.anony_models,
        need_infer_attack=args.need_infer_attack,
        max_concurrent=args.max_concurrent,
        batch_size=args.batch_size,
        update_db=args.update_db,
        db_batch_size=args.db_batch_size,
        save_path=args.save_path,
    )

    attacker = LLMPrivacyAttacker(
        infer_llm_model=infer_llm_model,
        infer_prompt_manager=infer_prompt_manager,
        config=config,
        mongo=mongo if args.update_db else None,
        collection_name=args.collection if args.update_db else None,
    )

    updated_profiles, infer_items = asyncio.run(attacker.infer_profiles(profiles))

    logger.success(
        f"Inference pipeline finished: profiles={len(updated_profiles)}, infer_items={len(infer_items)}"
    )


if __name__ == "__main__":
    main()
