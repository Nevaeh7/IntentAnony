"""
Prompt management module

Provides elegant LLM prompt management functionality
"""

# from .prompt_manager import (
#     PromptManager,
#     PromptTemplate,
#     get_manager,
#     get_infer_prompts,
#     INFER_PROMPTS
# )


__all__ = [
    # Core classes
    "PromptManager",
    "PromptTemplate",
    
    # Functions
    "get_manager",
    "get_infer_prompts",
    
    # Data
    "INFER_PROMPTS",
    "infer_prompt",
    "zh_prompt",
    "en_prompt",
]

# Version information
__version__ = "0.1.0"
