import os
import random
import sys
from typing import TextIO, Tuple, no_type_check, Optional, List, Dict
from pathlib import Path
from typing import Any

# Add project root directory to Python path

import numpy as np
import openai
import yaml
from pydantic import ValidationError


from configs import Config
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.string_utils import string_hash
from loguru import logger
from utils.x_utils import load_jsonl
from utils.dataset_utils import prepare_datasets
from datetime import datetime
from typing import Tuple, Dict, Any

class SafeOpen:
    def __init__(self, path: str, mode: str = "a", ask: bool = False):
        self.path = path

        if ask and os.path.exists(path):
            if input("File already exists. Overwrite? (y/n)") != "y":
                raise Exception("File already exists")
        self.mode = "a+" if os.path.exists(path) else "w+"
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        self.file = None
        self.lines = []

    def __enter__(self):
        self.file = open(self.path, self.mode)
        self.file.seek(0)  # move the cursor to the beginning of the file
        self.lines = self.file.readlines()
        # Remove last lines if empty
        while len(self.lines) > 0 and self.lines[-1] == "":
            self.lines.pop()
        return self

    def flush(self):
        self.file.flush()

    def write(self, content):
        self.file.write(content)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()


def read_config_from_yaml(path) -> Config:
    with open(path, "r", encoding='utf-8') as stream:
        try:
            yaml_obj = yaml.safe_load(stream)
            print(yaml_obj)
            cfg = Config(**yaml_obj)
            cfg.max_workers = cfg.task_config.anon_model.max_workers
            logger.success(f"set max_workers: {cfg.max_workers}")
            return cfg
        except (yaml.YAMLError, ValidationError) as exc:
            print(exc)
            raise exc


def seed_everything(seed: int) -> None:
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.set_num_threads(1)





@no_type_check
def get_out_file(cfg: Config) -> Tuple[TextIO, str]:
    file_path = cfg.get_out_path(cfg.task_config.get_filename())

    if not cfg.store:
        return sys.stdout, ""

    if len(file_path) > 255:
        file_path = file_path.split("/")
        file_name = file_path[-1]
        file_name_hash = string_hash(file_name)
        file_path = "/".join(file_path[:-1]) + "/hash_" + str(file_name_hash) + ".txt"

    ctr = 1
    while os.path.exists(file_path):
        with open(file_path, "r", encoding='utf-8') as fp:
            num_lines = len(fp.readlines())

        if num_lines >= 20:
            file_path = file_path.split("/")
            file_name = file_path[-1]

            ext = file_name.split(".")[-1]
            v_counter = file_name.split("_")[-1].split(".")[0]
            if v_counter.isdigit():
                ext_len = len(ext) + len(v_counter) + 2
                file_name = file_name[:-ext_len]
            else:
                file_name = file_name[: -(len(ext) + 1)]

            file_path = (
                "/".join(file_path[:-1]) + "/" + file_name + "_" + str(ctr) + ".txt"
            )
            ctr += 1
        else:
            break

    if len(file_path) > 255:
        file_path = file_path.split("/")
        prefix = "/".join(file_path[:4])
        file_name = "/".join(file_path[4:])
        file_name_hash = string_hash(file_name)
        file_path = prefix + "/hash_" + str(file_name_hash) + ".txt"

    if cfg.store:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        f = open(file_path, "w")
        sys.stdout = f

    return f, file_path

def _get_anonymized_text(profile: Dict[str, Any]) -> Optional[str]:
    """
    Extract anonymized text from profile
    
    Args:
        profile: User profile dictionary
        
    Returns:
        Anonymized text, returns None if not exists
    """
    anonymized_text = profile.get('anonymized_results', {})
    intent_anonymization = anonymized_text.get('intent_anonymization', {})
    text = intent_anonymization.get('anonymized_text', "")
    return text if text!="" and text is not None else None

def _get_intent_recognition_text(profile: Dict[str, Any]) -> Optional[str]:
    """
    Extract intent recognition text from profile
    
    Args:
        profile: User profile dictionary
        
    Returns:
        Intent recognition text, if not exists or empty, return None
    """
    intent_recognition_results = profile.get('intent_recognition_results', {})
    return intent_recognition_results if intent_recognition_results!="" and intent_recognition_results is not None else None


def _get_change_intent_recognition_text(profile: Dict[str, Any]) -> Optional[str]:
    """
    Extract change intent recognition text from profile
    
    Args:
        profile: User profile dictionary
        
    Returns:
        Change intent recognition text, if not exists or empty, return None
    """
    change_intent_recognition_results = profile.get('change_intent_recognition_results', {})
    return change_intent_recognition_results if change_intent_recognition_results!="" and change_intent_recognition_results is not None else None
def _get_aupi_evaluation_results(profile: Dict[str, Any]) -> Optional[str]:
    """
    Extract AUPI evaluation results from profile
    
    Args:
        profile: User profile dictionary
        
    Returns:
        AUPI evaluation results, returns None if not exists
    """
    aupi_evaluation_results = profile.get('aupi_evaluation_results', {})
    return aupi_evaluation_results if aupi_evaluation_results!="" and aupi_evaluation_results is not None else None

def _is_valid_finished_profile(profile: Dict[str, Any], check_type: str = 'anonymization') -> bool:
    """
    Check if profile is completed and valid
    
    Args:
        profile: User profile dictionary
        
    Returns:
        Returns True if profile is completed and valid
    """
    if check_type == 'anonymization':
        return _get_anonymized_text(profile) is not None
    elif check_type == 'intent':
        return _get_intent_recognition_text(profile) is not None
    elif check_type == 'ip_change_intent' or check_type == 'change_intent':
        return _get_change_intent_recognition_text(profile) is not None
    elif 'aupi_metrics' in check_type:
        return _get_aupi_evaluation_results(profile) is not None
    else:
        raise ValueError(f"Invalid check type: {check_type}")   


def filter_finished_profiles(
    profiles: List[Dict[str, Any]], 
    already_finished_profiles: List[Dict[str, Any]],
    check_type: str = 'anonymization'
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter finished and unfinished profiles
    
    Args:
        profiles: List of all profiles to process
        already_finished_profiles: List of already finished profiles
        
    Returns:
        Tuple of (finished_profiles, unfinished_profiles)
    """
    # Filter out valid finished profiles (with valid anonymized text)
    valid_finished_profiles = [
        p for p in already_finished_profiles 
        if _is_valid_finished_profile(p, check_type)
    ]
    finished_profiles = valid_finished_profiles
    
    # Use set to improve lookup performance
    finished_profile_ids = {profile['_id'] for profile in valid_finished_profiles}
    
    # Separate finished and unfinished profiles
    unfinished_profiles = []
    
    for profile in profiles:
        profile_id = profile.get('_id')
        if profile_id not in finished_profile_ids:
            unfinished_profiles.append(profile)
    
    return finished_profiles, unfinished_profiles


def _find_results_file(folder_path: str, possible_filenames: List[str]) -> Optional[str]:
    """
    Find results file in specified folder
    
    Args:
        folder_path: Folder path
        possible_filenames: List of possible filenames (sorted by priority)
        
    Returns:
        Found file path, returns None if none exist
    """
    for filename in possible_filenames:
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):
            logger.debug(f"Found results file: {file_path}")
            return file_path
        logger.debug(f"Results file does not exist, trying next: {file_path}")
    return None


def _build_result_dict(
    finished_profiles: List[Dict[str, Any]],
    unfinished_profiles: List[Dict[str, Any]],
    check_newest_folder: Optional[str],
    new_path: Optional[str] = None,
    cfg: Optional[Config] = None
) -> Dict[str, Any]:
    """
    Build unified result dictionary
    
    Args:
        finished_profiles: List of finished profiles
        unfinished_profiles: List of unfinished profiles
        check_newest_folder: Newest folder name
        new_path: New path (optional)
        
    Returns:
        Result dictionary
    """

    if cfg is not None:
        cfg.task_config.outpath = new_path
        logger.success(f"set outpath: {cfg.task_config.outpath}")
    if len(finished_profiles) == 0 and cfg.check_type != 'ip_change_intent' and cfg.check_type != 'ip_aupi_metrics':
        cfg.mode = "new"
    else:
        cfg.mode = "continue"
    result = {
        "finished_profiles": finished_profiles,
        "unfinished_profiles": unfinished_profiles,
        "check_newest_folder": check_newest_folder,
        "is_finished": len(unfinished_profiles) == 0,
        "new_path": new_path,
        "cfg": cfg
    }
    return result

def check_finished_or_create_path(cfg: Config, config_file: str, check_type='anonymization') -> Dict[str, Any]:
    """
    Check if task is completed, or create new output path
    
    Args:
        cfg: Configuration object
        config_file: Configuration file path
        
    Returns:
        Dictionary containing finished/unfinished profiles information, with the following keys:
        - finished_profiles: List of finished profiles
        - unfinished_profiles: List of unfinished profiles
        - check_newest_folder: Newest folder name
        - is_finished: Whether all are completed
        - new_path: New path (if exists)
        
    Raises:
        ValueError: When output directory is empty or cannot find newest folder
    """
    cur_path = cfg.task_config.outpath
    os.makedirs(cur_path, exist_ok=True)
    
    # Check if output directory exists
    if not os.path.exists(cur_path):
        os.makedirs(cur_path, exist_ok=True)
        logger.warning(f"Output directory does not exist, created: {cur_path}")
        return _build_result_dict([], [], None, None, cfg)
    
    # Get all folders and filter those matching date format
    try:
        folders = [
            f for f in os.listdir(cur_path) 
            if os.path.isdir(os.path.join(cur_path, f)) 
            and len(f) == 15  # YYYYMMDD_HHMMSS format
        ]
        
        if not folders:
            logger.warning(f"No valid folders found in output directory: {cur_path}")
            return _build_result_dict([], [], None, None, cfg)
        
        # Find newest folder
        check_newest_folder = max(
            folders, 
            key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S")
        )
        
        # Load original profiles
        if check_type == 'aupi_metrics' or check_type=='change_intent':
            p_path = os.path.dirname(cfg.task_config.outpath).replace('change_intent/', '').replace('aupi_metrics/', '')
            basename = os.path.basename(cfg.task_config.profile_path).replace('.jsonl', '')
            cfg.task_config.profile_path = os.path.join(p_path, f'{basename}_iter{cfg.iter_num}.jsonl')
            # raw_profiles = load_jsonl(cfg.task_config.profile_path)
        elif check_type == 'ip_aupi_metrics' or check_type=='ip_change_intent':
            cfg.task_config.profile_path = os.path.join(cur_path, check_newest_folder,'anonymized_results.jsonl')
        raw_profiles = load_jsonl(cfg.task_config.profile_path)

        profiles = prepare_datasets(raw_profiles, cfg.dataset_name)
        
        # Build full path of newest folder
        newest_folder_path = os.path.join(cur_path, check_newest_folder)
        
        # Try to find completed results file (in priority order)
        if check_type == 'anonymization':
            possible_result_filenames = [
                "each_anonymized_results.jsonl",
                "anonymized_results.jsonl"
            ]
        elif check_type == 'intent' or check_type == 'ip_change_intent' or check_type == 'change_intent':
            possible_result_filenames = [
                "each_intent_recognition_results.jsonl",
                "intent_recognition_results.jsonl"
            ]
        elif check_type =='aupi_metrics' or check_type=='ip_aupi_metrics':
            possible_result_filenames = [
                "each_aupi_evaluation_results.jsonl",
                "aupi_evaluation_results.jsonl"
            ]

        finished_results_path = _find_results_file(
            newest_folder_path, 
            possible_result_filenames
        )
        
        # If results file not found, return unfinished status
        if finished_results_path is None:
            logger.warning(
                f"Completed results file not found, tried filenames: {possible_result_filenames}, "
                f"folder: {newest_folder_path}"
            )
            return _build_result_dict(
                [], 
                profiles, 
                check_newest_folder,
                newest_folder_path,
                cfg
            )
        
        # Load and process finished profiles
        already_finished_profiles = load_jsonl(finished_results_path)
        finished_profiles, unfinished_profiles = filter_finished_profiles(
            profiles, 
            already_finished_profiles,
            check_type
        )
        logger.warning(f"finished_profiles: {len(finished_profiles)}, unfinished_profiles: {len(unfinished_profiles)}")
        
        return _build_result_dict(
            finished_profiles,
            unfinished_profiles,
            check_newest_folder,
            newest_folder_path,
            cfg
        )
        
    except (ValueError, KeyError) as e:
        logger.error(f"Error processing folder: {e}")
        raise ValueError(f"Unable to parse folder date format or missing required fields: {e}") from e


