

import asyncio
import json
import sys
import random
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy.orm.base import instance_str
from tqdm import tqdm

# Prefer sklearn metrics calculation, fallback to manual implementation if unavailable
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))



# Constant definitions
GT_ATTRIBUTE_MAP = {
    'age': 'age',
    'gender': 'sex',
    'married': 'relationship_status',
    # 'relationship_status': 'relationship_status',
    'pobp': 'birth_city_country',
    'location': 'city_country',
    'education': 'education',
    'income': 'income_level',
    'occupation': 'occupation',
}
REVERSE_GT_ATTRIBUTE_MAP = {v: k for k, v in GT_ATTRIBUTE_MAP.items()}

INFER_ATTRIBUTE_MAP = {
    'age': 'age',
    'gender': 'gender',
    'relationship_status': 'married',
    'married': 'married',
    'pobp': 'pobp',
    'location': 'location',
    'education': 'education',
    'income': 'income',
    'occupation': 'occupation',
}

DEFAULT_EVAL_PII_TYPES = [
    'age', 'gender', 'married', 'pobp', 'location',
    'education', 'income', 'occupation'
]
PII_TYPE_MAP = {
    'age': 'age',
    'sex': 'gender',
    'mar': 'married',
    'pob': 'pobp',
    'loc': 'location',
    'edu': 'education',
    'inc': 'income',
    'occ': 'occupation',
}
REVERSE_PII_TYPE_MAP = {v: k for k, v in PII_TYPE_MAP.items()}


def prepare_raw_user_text(profiles: Dict[str, Any], dataset_name: str, anonymizer_name: str=None) -> str:
    """
    Prepare raw user text.
    
    Args:
        profile: User profile
        
    Returns:
        str: Raw user text
    """
    for i, profile in tqdm(enumerate(profiles), total=len(profiles), desc="Preparing raw user text"):
        if  'synthpai' in dataset_name:
            # user_prompt = item.get('text', '')
            comments = profile.get('comments', [])
            if not comments:
                logger.error(f"No comments found for profile {profile['username']}")
                continue
            comments = comments[0].get('comments', [])
            comment_string = "\n".join([comment.get('text', '') for comment in comments if comment.get('text', '').strip()])
            user_text = f"{comment_string}"
            user_prompt = f"User Comments:\n{comment_string}\n"
            profile['response'] = comment_string
        elif 'reddit' in dataset_name:
            response = profile.get('response', '')
            question = profile.get('question_asked', '')
            comments = response.split('\n')
            comments = [c for c in comments if c.strip()]
            user_text = "\n".join(comments)
            user_prompt = f"Question: {question}\nUser Response: {user_text}"
            age = profile.get('personality', {}).get('age', '')
            sex = profile.get('personality', {}).get('sex', '')
            user_name = f"{age}{sex}"
            profiles[i]['username'] = user_name
        elif 'med' in dataset_name or 'law' in dataset_name:
            user_text = profile.get('user_text', '')
            user_prompt = f"User Comments:\n{user_text}\n"
            profile['response'] = user_text

        if anonymizer_name == 'original':
            profiles[i].setdefault('anonymized_results', {})
            profiles[i]['anonymized_results'].setdefault(anonymizer_name, {})
            profiles[i]['anonymized_results'][anonymizer_name]['anonymized_text'] = user_text
        if '_id' not in profiles[i]:
            if 'reddit' in dataset_name or 'synthetic' in dataset_name:
                profiles[i]['_id'] = f"personal_reddit_{i}"
            elif 'synthpai' in dataset_name:
                profiles[i]['_id'] = f"synthpai_{i}"
            elif 'synthetic' in dataset_name:
                profiles[i]['_id'] = f"personal_reddit_{i}"

        profiles[i]['user_text'] = user_text
        profiles[i]['user_prompt'] = user_prompt
    return profiles

def prepare_personality_gt(profiles: Dict[str, Any], dataset_name: str, anonymizer_name: str=None) -> str:
    for i, profile in tqdm(enumerate(profiles), total=len(profiles), desc="Preparing personality GT"):
        if 'synthpai' in dataset_name:
            attrs= []
            personality = profiles[i].setdefault('gt_personality', {})
            personality_reddit = {}
            human_evaluated = profiles[i].get('reviews', {}).get('human_evaluated', {})
            # logger.warning(f"Human evaluated: {human_evaluated}")
            intent_privacy_labled = profiles[i].setdefault('intent_privacy_labled',{})
            if intent_privacy_labled:
                protected_attributes = intent_privacy_labled.get('protected_attributes',[])
                protected_attributes = [t.lower() for t in protected_attributes]
            else:
                logger.error(f"No intent_privacy_labled found for profile {profile['username']}")
                protected_attributes = []
            if isinstance(human_evaluated['age']['estimate'], int):
                human_evaluated['age']['estimate'] = int(human_evaluated['age']['estimate'])
            elif human_evaluated['age']['estimate'].strip().isdigit():
                human_evaluated['age']['estimate'] = int(human_evaluated['age']['estimate'])
            else:
                human_evaluated['age']['estimate'] = 0
            for k in DEFAULT_EVAL_PII_TYPES:
                if k not in human_evaluated:
                    human_evaluated[k] = {'estimate': '', 'hardness': 0}
                value = human_evaluated[k]['estimate']

                if value == '':
                    if REVERSE_PII_TYPE_MAP[k] in protected_attributes:
                        protected_attributes.remove(REVERSE_PII_TYPE_MAP[k])
                    human_evaluated[k]['estimate'] = value
                else:
                    attrs.append(k)
                personality_reddit[GT_ATTRIBUTE_MAP[k]] = value
            protected_attributes = [t.upper() for t in protected_attributes]
            profiles[i]['intent_privacy_labled']['protected_attributes'] = protected_attributes
            profiles[i]['gt_personality'] = human_evaluated
            profiles[i]['personality'] = personality_reddit
            profiles[i]['feature'] = GT_ATTRIBUTE_MAP[random.choice(attrs)]
            profiles[i]['label'] =  personality_reddit['occupation']
            profiles[i]['eval_attributes'] = protected_attributes

        elif 'reddit' in dataset_name or 'synthetic' in dataset_name:
            raw_personality = profiles[i].get('personality', {})
            human_evaluated = {}
            profiles[i].setdefault('intent_privacy_labled',{})
            protected_attributes = profiles[i]['intent_privacy_labled'].setdefault('protected_attributes', [])
            for k, v in raw_personality.items():
                if k not in REVERSE_GT_ATTRIBUTE_MAP:
                    continue
                human_evaluated[REVERSE_GT_ATTRIBUTE_MAP[k]] = {'estimate': v, 'hardness': profile.get('hardness', 0)}
            profiles[i]['gt_personality'] = human_evaluated
            profiles[i]['eval_attributes'] = protected_attributes

        elif 'law' in dataset_name or 'med' in dataset_name:
            raw_personality = profiles[i].get('personality', {})
            human_evaluated = {}
            profiles[i].setdefault('intent_privacy_labled',{})
            protected_attributes = profiles[i]['intent_privacy_labled'].setdefault('protected_attributes', [])
            new_removed_attributes = []
            for pii_type in protected_attributes:
                if dataset_name == 'law':
                    if pii_type not in ['AGE', 'SEX', 'LOC'] :
                        protected_attributes.remove(pii_type)
                elif dataset_name == 'med':
                    if pii_type not in ['AGE', 'SEX', 'OCC', 'INC'] :
                        new_removed_attributes.append(pii_type)
            for new_removed_attribute in new_removed_attributes:
                protected_attributes.remove(new_removed_attribute)
            profiles[i]['intent_privacy_labled']['protected_attributes'] = protected_attributes
            for k, v in raw_personality.items():
                if k not in REVERSE_GT_ATTRIBUTE_MAP:
                    continue
                human_evaluated[REVERSE_GT_ATTRIBUTE_MAP[k]] = {'estimate': v, 'hardness': profile.get('hardness', 0)}
            profiles[i]['gt_personality'] = human_evaluated
            profiles[i]['eval_attributes'] = protected_attributes
    return profiles

def prepare_datasets(profiles: Dict[str, Any], dataset_name: str, anonymizer_name: str=None) -> Dict[str, Any]:
    profiles = prepare_raw_user_text(profiles, dataset_name, anonymizer_name)
    profiles = prepare_personality_gt(profiles, dataset_name, anonymizer_name)
    return profiles


def prese_infer_answer(answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Preprocess inference answers, normalize key names and value formats.
    
    Args:
        answers: List of inference answers, each answer is a dictionary
        
    Returns:
        Processed inference answer list
    """
    for i, infer in enumerate(answers):
        new_infer = {}
        for k, v in infer.items():
            try:
                k = k.lower().strip()
                new_infer[k] = v
                if k == 'type':
                    new_infer[k] = INFER_ATTRIBUTE_MAP.get(v.lower().strip(), v)
                if k == 'guess' and isinstance(v, str):
                    new_infer[k] = [g.lower().strip() for g in v.split(';')]
            except Exception as e:
                logger.error(f"Error in prese_infer_answer: {e}")
                new_infer[k] = v
        answers[i] = new_infer
    return answers


def reprocess_infer_attributes(
    profile: Dict[str, Any],
    anony_models: List[str]
) -> Dict[str, Any]:
    """
    Reprocess inference attributes, organize inference results into dictionary format.
    
    Args:
        profile: User profile
        anony_models: List of anonymization models
        
    Returns:
        Processed profile
    """
    profile.setdefault('infer_attributes', {})
    for anony_model in anony_models:
        infer_attack = {}
        infers = profile['infers'][anony_model]['instructions']
        infers = prese_infer_answer(infers)
        for infer in infers:
            infer_attack[infer['type']] = infer
        profile['infer_attributes'][anony_model] = infer_attack
    return profile

def get_eval_pii_types(profile: Dict[str, Any]) -> List[str]:
    eval_pii_types = []
    pii_types = profile.get('intent_privacy_labled',{}).get('protected_attributes',[])    
    pii_types = [t.lower() for t in pii_types]
    for pii_type in pii_types:
        if pii_type in PII_TYPE_MAP:
            eval_pii_types.append(PII_TYPE_MAP[pii_type])
    return eval_pii_types

    

def get_gt_pred_item(
    profile: Dict[str, Any],
    anony_models: List[str],
    eval_pii_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Get ground truth and prediction comparison items.
    
    Args:
        profile: User profile
        anony_models: List of anonymization models
        eval_pii_types: List of PII types to evaluate, defaults to all types
        
    Returns:
        List of items containing ground truth and predictions
    """
    # if eval_pii_types is None:
    #     eval_pii_types = DEFAULT_EVAL_PII_TYPES
    eval_pii_types = get_eval_pii_types(profile)
    
    gt_items = []
    profile = reprocess_infer_attributes(profile, anony_models)
    
    for pii_type in eval_pii_types:
        gt_item = {
            "id": profile['_id'],
            "pii_type": pii_type,
            "gt": str(profile['gt_personality'][pii_type]['estimate']).strip().lower(),
            "gt_hardness": profile['gt_personality'][pii_type]['hardness'],
            "infer_attacks": {}
        }
        for anony_model in anony_models:
            pii_infer_attack = profile['infer_attributes'][anony_model][pii_type]
            gt_item['infer_attacks'][anony_model] = pii_infer_attack
        gt_items.append(gt_item)
    
    return gt_items

def filter_dataset_by_privacy_count(dataset: List[Dict[str, Any]], threshold: int = 1) -> List[Dict[str, Any]]:
    items = []
    for item in dataset:
        if len(item.get('intent_privacy_labled',{}).get('protected_attributes',[])) >= threshold:
            items.append(item)
    logger.success(f"Dataset count before filtering: {len(dataset)}, Dataset count after filtering: {len(items)}")
    return items
