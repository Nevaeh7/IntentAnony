import re
from loguru import logger
from typing import List, Dict, Any, Optional
import os
import json
from datetime import datetime
import shutil
from utils.json_utils import extract_json_like_blocks, load_first_json_like
# Precompile regex patterns (module level, avoid repeated compilation)
import re

_UNICODE_RANGES = {
    # CJK unified ideographs
    'CJK_UNIFIED': r'[\u4e00-\u9fff]',
    'CJK_EXT_A': r'[\u3400-\u4dbf]',
    'CJK_EXT_B': r'[\u20000-\u2a6df]',
    'CJK_EXT_C': r'[\u2a700-\u2b73f]',
    'CJK_EXT_D': r'[\u2b740-\u2b81f]',
    'CJK_EXT_E': r'[\u2b820-\u2ceaf]',
    
    # Japanese
    'HIRAGANA': r'[\u3040-\u309f]',
    'KATAKANA': r'[\u30a0-\u30ff]',
    'KATAKANA_EXT': r'[\u31f0-\u31ff]',
    
    # Korean
    'HANGUL_SYLLABLES': r'[\uac00-\ud7af]',
    'HANGUL_JAMO': r'[\u1100-\u11ff]',
    'HANGUL_COMPAT': r'[\u3130-\u318f]',
    
    # Southeast Asian languages
    'THAI': r'[\u0e00-\u0e7f]',
    'LAO': r'[\u0e80-\u0eff]',
    'KHMER': r'[\u1780-\u17ff]',
    'MYANMAR': r'[\u1000-\u109f]',
    
    # Arabic script
    'ARABIC': r'[\u0600-\u06ff]',
    'ARABIC_EXT_A': r'[\u0750-\u077f]',
    'ARABIC_EXT_B': r'[\u08a0-\u08ff]',
    'PERSIAN': r'[\u06a0-\u06ff]',
    
    # Cyrillic
    'CYRILLIC': r'[\u0400-\u04ff]',
    'CYRILLIC_EXT_A': r'[\u0500-\u052f]',
    'CYRILLIC_EXT_B': r'[\u2de0-\u2dff]',
    
    # Other European languages
    'GREEK': r'[\u0370-\u03ff]',
    'GREEK_EXT': r'[\u1f00-\u1fff]',
    'ARMENIAN': r'[\u0530-\u058f]',
    'HEBREW': r'[\u0590-\u05ff]',
    'GEORGIAN': r'[\u10a0-\u10ff]',
    
    # Indic scripts
    'DEVANAGARI': r'[\u0900-\u097f]',
    'BENGALI': r'[\u0980-\u09ff]',
    'TAMIL': r'[\u0b80-\u0bff]',
    'TELUGU': r'[\u0c00-\u0c7f]',
    'GUJARATI': r'[\u0a80-\u0aff]',
    'KANNADA': r'[\u0c80-\u0cff]',
    'MALAYALAM': r'[\u0d00-\u0d7f]',
    
    # Other languages
    'TIBETAN': r'[\u0f00-\u0fff]',
    'ETHIOPIC': r'[\u1200-\u137f]',
    'CHEROKEE': r'[\u13a0-\u13ff]',
    'RUNIC': r'[\u16a0-\u16ff]',
}

# Precompile regex
_CJK_PATTERN = re.compile('|'.join([
    _UNICODE_RANGES['CJK_UNIFIED'],
    _UNICODE_RANGES['CJK_EXT_A'],
    _UNICODE_RANGES['CJK_EXT_B'],
    _UNICODE_RANGES['CJK_EXT_C'],
    _UNICODE_RANGES['CJK_EXT_D'],
    _UNICODE_RANGES['CJK_EXT_E'],
    _UNICODE_RANGES['HIRAGANA'],
    _UNICODE_RANGES['KATAKANA'],
    _UNICODE_RANGES['KATAKANA_EXT'],
    _UNICODE_RANGES['HANGUL_SYLLABLES'],
    _UNICODE_RANGES['HANGUL_JAMO'],
    _UNICODE_RANGES['HANGUL_COMPAT'],
]))

_OTHER_SCRIPTS_PATTERN = re.compile('|'.join([
    _UNICODE_RANGES['THAI'],
    _UNICODE_RANGES['LAO'],
    _UNICODE_RANGES['KHMER'],
    _UNICODE_RANGES['MYANMAR'],
    _UNICODE_RANGES['ARABIC'],
    _UNICODE_RANGES['ARABIC_EXT_A'],
    _UNICODE_RANGES['ARABIC_EXT_B'],
    _UNICODE_RANGES['PERSIAN'],
    _UNICODE_RANGES['CYRILLIC'],
    _UNICODE_RANGES['CYRILLIC_EXT_A'],
    _UNICODE_RANGES['CYRILLIC_EXT_B'],
    _UNICODE_RANGES['GREEK'],
    _UNICODE_RANGES['GREEK_EXT'],
    _UNICODE_RANGES['ARMENIAN'],
    _UNICODE_RANGES['HEBREW'],
    _UNICODE_RANGES['GEORGIAN'],
    _UNICODE_RANGES['DEVANAGARI'],
    _UNICODE_RANGES['BENGALI'],
    _UNICODE_RANGES['TAMIL'],
    _UNICODE_RANGES['TELUGU'],
    _UNICODE_RANGES['GUJARATI'],
    _UNICODE_RANGES['KANNADA'],
    _UNICODE_RANGES['MALAYALAM'],
    _UNICODE_RANGES['TIBETAN'],
    _UNICODE_RANGES['ETHIOPIC'],
    _UNICODE_RANGES['CHEROKEE'],
    _UNICODE_RANGES['RUNIC'],
]))

# Precompile special content patterns
_SPECIAL_PATTERNS = [
    re.compile(r'https?://[^\s]+'),  # URL
    re.compile(r'www\.[^\s]+'),      # www links
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # Email
    re.compile(r'\b\d+\.?\d*\b'),    # Numbers
    re.compile(r'[^\w\s]'),          # Punctuation and special characters
]

# Precompile emoji patterns
_EMOJI_PATTERN = re.compile(r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\U0001F1E0-\U0001F1FF]|[\U00002600-\U000026FF]|[\U00002700-\U000027BF]')

# Precompile other script continuous character patterns
_OTHER_SCRIPTS_CONTINUOUS = re.compile(f'({_OTHER_SCRIPTS_PATTERN.pattern})+')


def count_multilingual_words(text: str) -> int:
    """
    Count words/characters in multilingual text, supports major global languages.
    
    Language processing strategy:
    - CJK languages (Chinese, Japanese, Korean): Each character counts as 1 word
    - Latin scripts (English, etc.): Split by spaces and punctuation
    - Other scripts: Count by continuous character sequences
    - Special handling: URLs, emails, numbers, emojis, etc.
    
    Args:
        text: Text to count
        
    Returns:
        int: Word/character count result
        
    Raises:
        TypeError: Input is not string type
        ValueError: Input is empty or invalid
    """
    # Input validation
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text).__name__}")
    
    if not text.strip():
        return 0
    
    # Preprocessing: clean and normalize
    text = text.strip()
    
    try:
        # 1. Count CJK characters (each character counts as 1 word)
        cjk_matches = _CJK_PATTERN.findall(text)
        cjk_count = len(cjk_matches)
        
        # 2. Remove CJK characters, process other languages
        text_no_cjk = _CJK_PATTERN.sub(' ', text)
        
        # 3. Remove special content (using precompiled patterns)
        for pattern in _SPECIAL_PATTERNS:
            text_no_cjk = pattern.sub(' ', text_no_cjk)
        
        # 4. Count continuous characters of other scripts
        other_script_matches = _OTHER_SCRIPTS_CONTINUOUS.findall(text_no_cjk)
        other_script_count = len(other_script_matches)
        
        # 5. Count Latin language words (split by spaces)
        latin_text = _OTHER_SCRIPTS_PATTERN.sub(' ', text_no_cjk)
        latin_words = [word for word in latin_text.split() if word.strip()]
        latin_count = len(latin_words)
        
        # 6. Count emojis and special Unicode characters
        emoji_matches = _EMOJI_PATTERN.findall(text)
        emoji_count = len(emoji_matches)
        
        # 7. Calculate total word count
        total_count = cjk_count + other_script_count + latin_count + emoji_count
        
        # 8. Debug information (optional)
        logger.debug(
            "Word count details: CJK={}, Other scripts={}, Latin={}, Emoji={}, Total={}",
            cjk_count,
            other_script_count,
            latin_count,
            emoji_count,
            total_count,
        )
        
        return total_count
        
    except Exception as e:
        logger.error("Word count failed: {}", e)
        # Fallback: simple split by spaces
        return len([word for word in text.split() if word.strip()])


def filter_tags(caption_content):
    """
    Separate tags and plain text content from text
    
    Args:
        caption_content: Original text containing tags
        
    Returns:
        dict: Dictionary containing 'text' and 'tags'
    """
    import re
    
    if not caption_content:
        return {'text': '', 'tags': []}
    
    # Use regex to match all tags starting with #
    # Pattern: # followed by one or more non-whitespace characters
    tag_pattern = r'#([^\s#]+)'
    tags = re.findall(tag_pattern, caption_content)
    
    # Remove all tags, keep plain text
    text_without_tags = re.sub(tag_pattern, '', caption_content).strip()


    text_without_tags = re.sub(r'^[\s\.,;:!?！？。、《》〈〉【】\[\]\(\)\-—…“”‘’"\'、，:：;；！!？?\|\\/·~`$%^&*_=+<>]+', '', text_without_tags)
    

    return {
        'text': text_without_tags,
        'tags': tags
    }

def read_jsonl(path):
    """Read JSONL file and return generator"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line,strict=False)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num} JSON parsing failed: {e}, skipping this line")
                    continue

def parse_json_response(result):
    """
    Call LLM interface and gracefully parse returned JSON content.
    Automatically remove Markdown code block wrapping and safely load JSON.
    """
    if isinstance(result, str):
        content = result.strip()
        raw_content = result
    else:
        content = result.choices[0].message.content.strip()
        raw_content = result.choices[0].message.content


    try:
        parsed = load_first_json_like(content)
        if parsed is None:
            raise ValueError(f"Model returned content is not valid JSON format v0. {raw_content}")
    except json.JSONDecodeError as e:
        logger.error(f" JSON parsing failed: {e}\nOriginal content:\n{raw_content}")
        raise ValueError("Model returned content is not valid JSON format.") from e
    return parsed


def load_jsonl(path) -> List[Dict[str, Any]]:
    extension = path.split(".")[-1]

    assert extension == "jsonl"

    with open(path, "r", encoding="utf-8") as json_file:
        json_list = json_file.readlines()

    return [json.loads(line) for line in json_list]

def clean_data_for_json(obj):
    """Recursively clean surrogate characters from all strings in object to ensure safe UTF-8 encoding"""
    if isinstance(obj, str):
        # Directly check if contains surrogate characters
        if any(0xD800 <= ord(c) <= 0xDFFF for c in obj):
            return strip_surrogates(obj)
        return obj
    elif isinstance(obj, dict):
        return {k: clean_data_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_data_for_json(item) for item in obj]
    else:
        return obj

def save_jsonl(data: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as json_file:
        for item in data:
        try:
            # First attempt to write directly, no cleaning needed if successful
            json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
        except UnicodeEncodeError:
            # If encoding error occurs, clean surrogate characters and retry
            cleaned_item = clean_data_for_json(item)
            json_file.write(json.dumps(cleaned_item, ensure_ascii=False) + "\n")
        json_file.flush()
    logger.success(f"Data saved to {path}")
    
def save_json(data: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        # First try direct write, if successful no need to clean
        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
    except UnicodeEncodeError:
        # If encoding error occurs, clean surrogate characters and retry
        cleaned_data = clean_data_for_json(data)
        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(cleaned_data, json_file, ensure_ascii=False, indent=4)
    logger.success(f"Data saved to {path}")

def add_save_jsonl(data: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as json_file:
        for item in data:
            try:
                # First try direct write, if successful no need to clean
                json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            except UnicodeEncodeError:
                # If encoding error occurs, clean surrogate characters and retry
                cleaned_item = clean_data_for_json(item)
                json_file.write(json.dumps(cleaned_item, ensure_ascii=False) + "\n")
            json_file.flush()
    logger.success(f"Data saved to {path}")

def calculate_stats(values: List[float]) -> Optional[Dict[str, float]]:
    """Calculate statistics: mean, standard deviation, minimum, maximum"""
    if not values:
        return None
    import statistics
    return {
        'mean': statistics.mean(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }

def copy_config(config_file: str, new_path: str):
    shutil.copy(config_file, new_path)
    logger.success(f"Config copied to {new_path}")





def get_new_out_puth(cfg, config_file: str):
    cur_path = cfg.task_config.outpath
    new_dirname = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path = os.path.join(cur_path, new_dirname)
    os.makedirs(new_path, exist_ok=True)
    cfg.task_config.outpath = new_path
    logger.success(f"New output path: {cfg.task_config.outpath}")
    config_name = os.path.basename(config_file)
    copy_config(config_file, os.path.join(new_path, config_name))
    return cfg

def write_add_jsonl(data: Dict[str, Any], path: str):
    try:
        # First try direct write, if successful no need to clean
        with open(path, "a", encoding="utf-8") as json_file:
            json_file.write(json.dumps(data, ensure_ascii=False) + "\n")
            json_file.flush()
    except UnicodeEncodeError:
        # If encoding error occurs, clean surrogate characters and retry
        cleaned_data = clean_data_for_json(data)
        with open(path, "a", encoding="utf-8") as json_file:
            json_file.write(json.dumps(cleaned_data, ensure_ascii=False) + "\n")
            json_file.flush()
    # logger.success(f"Data saved to {path}")


_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')

def strip_surrogates(s: str) -> str:
    if not isinstance(s, str):
        return s
    return _SURROGATE_RE.sub('', s)

