
# from .azure_anonymizer import AzureAnonymizer
from .llm_anonymizers import LLMFullAnonymizer, LLMBaselineAnonymizer
from .anonymizer import Anonymizer
import sys
from pathlib import Path
from typing import Any

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs import AnonymizationConfig

def get_anonymizer(cfg: AnonymizationConfig) -> Anonymizer:
    
    if cfg.anonymizer.anon_type == "azure":
        # return AzureAnonymizer(cfg.anonymizer)
        pass
    elif cfg.anonymizer.anon_type == "llm":
        # return LLMFullAnonymizer(cfg.anonymizer, model)
        pass
    elif cfg.anonymizer.anon_type == "llm_base":
        pass
        # return LLMBaselineAnonymizer(cfg.anonymizer, model)
    else:
        raise ValueError(f"Unknown anonymizer type {cfg.anonymizer.anon_type}")