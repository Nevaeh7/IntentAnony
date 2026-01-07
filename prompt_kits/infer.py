from loguru import logger
from .prompt_manager import get_infer_prompts, get_manager

# Backward compatibility: preserve original dictionary format
infer_prompt = get_infer_prompts()

# Get manager instance
manager = get_manager()

# Backward compatibility: preserve original string variables
zh_prompt = infer_prompt['zh']
en_prompt = infer_prompt['en']

# Log output
# logger.info(f"Chinese prompt loaded, length: {len(zh_prompt)}")
# logger.info(f"English prompt loaded, length: {len(en_prompt)}")

# Usage examples:
# Method 1: Use manager (recommended)
# manager = get_manager()
# prompt = manager.get("infer", "zh")
#
# Method 2: Direct dictionary usage (backward compatible)
# prompt = infer_prompt['zh']