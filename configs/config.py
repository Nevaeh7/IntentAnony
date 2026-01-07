from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel as PBM
from pydantic import Extra, Field
from pathlib import Path
import sys

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.string_utils import string_hash


class Task(Enum):
    REDDIT = "REDDIT"  # Reddit
    ANONYMIZED = "ANONYMIZED"  # Anonymization of reddit comments (potentially in multiple rounds)
    EVALUATE = "EVALUATE"  # Evaluate the anonymization results
    INTENT = "INTENT"  # Intent recognition
    CHANGE_INTENT = "CHANGE_INTENT"  # Change intent

class ModelConfig(PBM):
    name: str = Field(description="Name of the model")
    tokenizer_name: Optional[str] = Field(
        None, description="Name of the tokenizer to use"
    )
    provider: str = Field(description="Provider of the model")
    dtype: str = Field(
        "float16", description="Data type of the model (only used for local models)"
    )
    device: str = Field(
        "auto", description="Device to use for the model (only used for local models)"
    )
    max_workers: int = Field(
        1, description="Number of workers (Batch-size) to use for parallel generation"
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the model upon generation",
    )
    model_template: str = Field(
        default="{prompt}",
        description="Template to use for the model (only used for local models)",
    )
    prompt_category: str = Field(
        default="intent_anonymizers_v2", description="Category of the prompt"
    )
    prompt_policy_version: str = Field(
        default="4.0", description="Version of the policy"
    )
    prompt_language: str = Field(
        default="en", description="Language of the prompt"
    )
    prompt_template: Dict[str, Any] = Field(  # TODO could be a BasePromptConfig
        default_factory=dict, description="Arguments to pass to the prompt"
    )
    submodels: List["ModelConfig"] = Field(
        default_factory=list, description="Submodels to use"
    )
    multi_selector: str = Field(
        default="majority", description="How to select the final answer"
    )


    def get_name(self) -> str:
        if self.name == "multi":
            return "multi" + "_".join(
                [submodel.get_name() for submodel in self.submodels]
            )
        if self.name == "chain":
            return "chain_" + "_".join(
                [submodel.get_name() for submodel in self.submodels]
            )
        if self.provider == "hf":
            return self.name.split("/")[-1]
        else:
            return self.name


class BasePromptConfig(PBM):
    # Contains prompt attributes which pertain to the header and footer of the prompt
    # These attributes are not used in the intermediate text (i.e. the meat of the prompt)
    modifies: List[str] = Field(
        default_factory=list,
        description="Whether this prompt config is used to modify existing prompts",
    )
    num_answers: int = Field(3, description="Number of answer given by the model")
    num_shots: int = Field(
        0, description="Number of shots to be presented to the model"
    )
    cot: bool = Field(False, description="Whether to use COT prompting")
    use_qa: bool = Field(
        False, description="Whether to present answer options to the model"
    )
    header: Optional[str] = Field(
        default=None,
        description="In case we want to set a specific header for the prompt.",
    )
    footer: Optional[str] = Field(
        default=None,
        description="In case we want to set a specific footer for the prompt.",
    )

    # Workaround to use this as a pure modifier as well
    def __init__(self, **data):
        super().__init__(**data)
        self.modifies = list(data.keys())

    def get_filename(self) -> str:
        file_path = ""
        for attr in vars(self):
            if attr in ["dryrun", "save_prompts", "header", "footer", "modifies"]:
                continue

            if "_" in attr:
                attr_short = attr.split("_")[1][:4]
            else:
                attr_short = attr[:4]

            file_path += f"{attr_short}={getattr(self, attr)}_"
        return file_path[:-1] + ".txt"


class REDDITConfig(PBM):
    path: str = Field(
        ...,
        description="Path to the file",
    )
    paths: List[str] = Field(
        default_factory=list,
        description="Paths to the files for merging",
    )
    outpath: str = Field(
        ...,
        description="Path to write to for comment scoring",
    )
    eval: bool = Field(
        default="False",
        description="Whether to only evaluate the corresponding profiles.",
    )
    eval_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Settings for evaluation.",
    )
    decider: str = Field(
        default="model", description="Decider to use in case there's no match."
    )
    profile_filter: Dict[str, int] = Field(
        default_factory=dict, description="Filter profiles based on comment statistics."
    )
    max_prompts: Optional[int] = Field(
        default=None, description="Maximum number of prompts asked (int total)"
    )
    header: Optional[str] = Field(default=None, description="Prompt header to use")
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt to use"
    )
    individual_prompts: bool = Field(
        False,
        description="Whether we want one prompt per attribute inferred or one for all.",
    )

    def get_filename(self) -> str:
        file_path = ""
        for attr in vars(self):
            if attr in ["path", "outpath"]:
                continue
            if attr == "profile_filter":
                file_path += (
                    str([f"{k}:{v}" for k, v in getattr(self, attr).items()]) + "_"
                )
            else:
                file_path += f"{attr}={getattr(self, attr)}_"
        return file_path[:-1] + ".txt"

    class Config:
        extra = Extra.forbid


class AnonymizerConfig(PBM):
    anon_type: str = Field(
        default="span",
        description="Type of anonymizer to use",
        choices=["span", "azure", "llm", "span_llm"],
    )
    anon_model_name: str = Field(
        default="intent_anonymization",
        description="Name of the anonymizer model",
    )
    batch_size: int = Field(
        default=50,
        description="Batch size to use for parallel generation",
    )
    anonymized_which_parts: str = Field(
        default=None, description="Which parts of the text to anonymize"
    )
    max_workers: int = Field(
        default=1,
        description="Number of workers (Batch-size) to use for parallel generation",
    )
    target_mode: str = Field(
        default="single",
    )
    prompt_level: int = Field(
        default=3,
        description="Level of prompt to use for the anonymizer",
    )
    ### For span anonymizer
    model_dir: str = Field(
        default="models/anonymizer",
        description="Path to the model directory if needed",
    )
    replacement_type: str = Field(
        default="*",
        description="Type of replacement to use",
        choices=["model", "entity", "*"],
    )
    ######
    azure_threshhold: float = Field(
        default=0.4,
        description="Confidence threshhold for the Azure Anonymizer",
    )
    azure_replacement: str = Field(
        default="*",
        description="Replacement string for the Azure Anonymizer",
        options=["*", "entity"],
    )
    llm_assume_inference: bool = Field(
        default=True,
        description="Whether to assume inference exists for the llm anonymizer",
    )

    anonymized_max_iter: int = Field(
        default=1,
        description="Maximum number of iterations to run the anonymization for",
    )

    is_pre_iiv: bool = Field(
        default=False,
        description="Whether to use the pre-IIV anonymizer",
    )



class AnonymizationConfig(PBM):
    # Prompt Loading
    profile_path: str = Field(
        default=None,
        description="Path to a file containing profiles including comments",
    )
    offset: int = Field(
        default=0,
        description="Offset to start from in the profile file",
    )
    num_profiles: int = Field(
        default=1000,
        description="Number of profiles to use from the profile file",
    )
    outpath: str = Field(
        default=None,
        description="Path to a file to write the anonymized profiles to",
    )
    update_db: bool = Field(
        default=False,
        description="Whether to update the database",
    )
    attack_outpath: str = Field(
        default=None,
        description="Path to a file to write the attack results to",
    )
    judge_infer_outpath: str = Field(
        default=None,
        description="Path to a file to write the judge inference results to",
    )
    utility_outpath: str = Field(
        default=None,
        description="Path to a file to write the utility results to",
    )
    anon_model: Optional[ModelConfig] = Field(
        default=None,
        description="Model to use for anonymization, otherwise same as the generation model",
    )
    intent_model: Optional[ModelConfig] = Field(
        default=None,
        description="Model to use for intent recognition",
    )
    adversary_attack_model: Optional[ModelConfig] = Field(
        default=None,
        description="Model to use for anonymization, otherwise same as the generation model",
    )
    piec_model: Optional[ModelConfig] = Field(
        default=None,
        description="Model to use for privacy inference evidence chain",
    )
    utility_model: Optional[ModelConfig] = Field(
        default=None, description="Model to use for utility"
    )
    inference_model: Optional[ModelConfig] = Field(
        default=None, description="Model to use for inference"
    )
    judge_infer_model: Optional[ModelConfig] = Field(
        default=None, description="Model to use for judge inference"
    )
    run_eval_inference: bool = Field(
        default=False, description="Whether to run evaluation inference"
    )
    run_utility_scoring: bool = Field(
        default=False, description="Whether to run utility scoring"
    )
    eval_inference_model: Optional[ModelConfig] = Field(
        default=None, description="Model to use for evaluation inference"
    )
    anonymizer: AnonymizerConfig = Field(
        default_factory=AnonymizerConfig,
        description="Config for the anonymizer",
    )

    max_num_iterations: int = Field(
        default=1,
        description="Maximum number of iterations to run the anonymization for. We might stop earlier if we can no longer find a correct answer",
    )
    use_ner: bool = Field(
        default=False,
        description="Whether to use NER to find entities in the text",
    )
    profile_filter: Dict[str, int] = Field(
        default_factory=dict, description="Filter profiles based on comment statistics."
    )
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt to use"
    )
    header: Optional[str] = Field(default=None, description="Prompt header to use")
    footer: Optional[str] = Field(default=None, description="Prompt footer to use")

    def get_filename(self) -> str:
        file_path = "anon_"
        for attr in vars(self):
            if attr in ["path", "outpath"]:
                continue
            if attr == "profile_filter":
                file_path += (
                    str([f"{k}:{v}" for k, v in getattr(self, attr).items()]) + "_"
                )
            else:
                file_path += f"{attr}={getattr(self, attr)}_"
        return file_path[:-1] + ".txt"

    class Config:
        extra = Extra.forbid


class Config(PBM):
    # This is the outermost config containing subconfigs for each benchmark as well as
    # IO and logging configs. The default values are set to None so that they can be
    # overridden by the user
    output_dir: str = Field(
        default=None, description="Directory to store the results in"
    )
    result: Dict[str, Any] = Field(
        default={}, description="Result of the workflow"
    )
    seed: int = Field(default=42, description="Seed to use for reproducibility")
    task: Task = Field(
        default=None, description="Task to run", choices=list(Task.__members__.values())
    )
    mode: str = Field(
        default="continue", description="Mode to run the workflow in", choices=["new", "continue"]
    )
    intent_conf_thres: float = Field(
        default=-1, description="Threshold for the intent strength, if -1, use the default threshold"
    )
    config_name: str = Field(
        default=None, description="Name of the model"
    )
    dataset_name: str = Field(
        default=None, description="Name of the dataset to store the results in"
    )
    collection_name: str = Field(
        default=None, description="Name of the collection to store the results in"
    )
    iter_num: int = Field(
        default=3, description="Number of iterations to run the anonymization for"
    )
    task_config: AnonymizationConfig | REDDITConfig = Field(
        default=None, description="Config for the task"
    )
    gen_model: ModelConfig = Field(
        default=None, description="Model to use for generation, ignored for CHAT task"
    )
    store: bool = Field(
        default=True, description="Whether to store the results in a file"
    )
    save_prompts: bool = Field(
        False, description="Whether to ouput the prompts in JSON format"
    )
    dryrun: bool = Field(
        False, description="Whether to just output the queries and not predict"
    )
    timeout: int = Field(
        0.5, description="Timeout in seconds between requests for API restrictions"
    )
    max_workers: int = Field(
        100, description="Number of workers (Batch-size) to use for parallel generation"
    )
    check_type: str = Field(
        default="anonymization", description="Type of the check", choices=["anonymization", "intent", "aupi_metrics", "ip_aupi_metrics", "ip_change_intent"]
    )
    def get_out_path(self, file_name) -> str:
        path_prefix = "results" if self.output_dir is None else self.output_dir

        if self.task.value == "CHAT":
            investigator_bot_name = self.task_config.investigator_bot.get_name()
            user_bot_name = self.task_config.user_bot.get_name()
            file_path = f"{path_prefix}/{self.task.value}/{investigator_bot_name}-{user_bot_name}/{self.seed}/{self.task_config.guess_feature}/{file_name}"
        elif self.task.value == "CHAT_EVAL":
            file_path = (
                "/".join((self.task_config.chat_path_prefix).split("/")[:-1])
                + "/"
                + file_name
            )
        else:
            model_name = self.gen_model.get_name()
            file_path = (
                f"{path_prefix}/{self.task.value}/{model_name}/{self.seed}/{file_name}"
            )

        return file_path
