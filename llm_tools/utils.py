import os
import json
from loguru import logger
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "qwen": "QWEN_API_KEY",
    "seed": "SEED_API_KEY",
    "glm": "GLM_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "dmx": "DMX_API_KEY",
    "google": "GEMINI_API_KEY",
}


def set_env_key():
    """
    Read keys from keys.json file and set them as environment variables
    """
    with open("keys.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        for k, v in data.items():
            os.environ[ENV_VARS[k]] = v
    logger.success("Environment variables configured")


def get_env_key(provider: str) -> str:
    """
    Get API key from environment variables
    """
    return os.environ.get(ENV_VARS[provider])


def get_env_key_by_provider(provider: str) -> str:
    """
    Get API key from environment variables by provider
    """
    return os.environ.get(ENV_VARS[provider])
