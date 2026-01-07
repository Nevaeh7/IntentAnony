#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI tool class for convenient LLM API calls
Supports multiple models and features, includes error handling and retry mechanisms
"""

import os
import json
from this import d
import time
import logging
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from openai import OpenAI, AsyncOpenAI
import asyncio
from .utils import ENV_VARS,set_env_key
from loguru import logger
import json5
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
set_env_key()


@dataclass
class ModelConfig:
    """Model configuration class"""

    name: str = "gpt-5"
    max_tokens: int = 2000
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    api_type: Optional[str] = 'chat'  # "responses" or "chat", None means auto-select


class ModelProvider:
    """Model provider configuration class"""
    # Predefined model provider configurations
    PROVIDERS = json5.load(open("provider.json", "r", encoding='utf-8'))
    @classmethod
    def get_provider_config(cls, provider_name: str) -> dict:
        """Get provider configuration"""
        return cls.PROVIDERS.get(provider_name, cls.PROVIDERS["custom"])

    @classmethod
    def list_providers(cls) -> list:
        """List all available providers"""
        return list(cls.PROVIDERS.keys())

    @classmethod
    def list_models(cls, provider_name: str) -> list:
        """List all models for specified provider"""
        config = cls.get_provider_config(provider_name)
        return list(config["models"].keys())

    @classmethod
    def get_model_name(cls, provider_name: str, model_key: str) -> str:
        """Get actual model name"""
        config = cls.get_provider_config(provider_name)
        return config["models"].get(model_key, model_key)


class OpenAITool:
    """OpenAI tool class providing convenient LLM API interface"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        api_type: Optional[str] = None,
    ):
        """
        Initialize OpenAI tool

        Args:
            api_key: API key, if not provided will be retrieved from environment variables
            base_url: API base URL for custom API endpoints
            provider: Model provider (openai, deepseek, claude, custom)
            model: Default model name
        """
        self.provider = provider
        self.provider_config = ModelProvider.get_provider_config(provider)

        # Get API key
        if not api_key:
            # Try different environment variables based on provider

            env_var = ENV_VARS.get(provider, "OPENAI_API_KEY")
            api_key = os.getenv(env_var)

        if not api_key:
            raise ValueError(
                f"Please provide API key or set environment variable {ENV_VARS.get(provider, 'OPENAI_API_KEY')}"
            )

        self.api_key = api_key

        # Determine base_url
        if not base_url:
            base_url = self.provider_config["base_url"]

        if not base_url:
            raise ValueError(f"Provider {provider} requires base_url to be specified")

        self.base_url = base_url

        # Create client
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

        # Async client
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)

        # Default model configuration
        if model:
            self.default_config = ModelConfig(name=model, api_type=api_type)
        else:
            # Use provider's default model
            default_models = {
                "openai": "gpt-5",
                "deepseek": "deepseek-chat",
                "qwen": "qwen-flash",
                "claude": "claude-4-sonnet",
                "glm": "glm-4-flash",
                "custom": "gpt-5",
            }
            default_model = default_models.get(provider, "gpt-3.5-turbo")
            self.default_config = ModelConfig(name=default_model, api_type=api_type)

        # Setup logging
        self.logger = self._setup_logger()

        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1

        self.logger.info(
            f"Initialization complete - Provider: {provider}, Model: {self.default_config.name}, URL: {base_url}"
        )

    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger("OpenAITool")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _build_api_params_and_func(
        self,
        model: str,
        messages: List[Dict[str, str]],
        config: ModelConfig,
        stream: bool = False,
        use_async: bool = False,
        **kwargs,
    ) -> tuple:
        """
        Build API parameters and select corresponding call function
        
        Args:
            model: Model name
            messages: Message list
            config: Model configuration
            stream: Whether to use streaming output
            use_async: Whether to use async client
            **kwargs: Other parameters
            
        Returns:
            Tuple of (parameter dict, call function)
        """
        is_gpt_model = 'gpt' in model.lower()
        
        if is_gpt_model:
            # GPT models use different API format
            if config.api_type == "responses":
                params = {
                    "model": model,
                    "input": messages,
                    "max_output_tokens": config.max_tokens,
                    "stream": stream,
                    **kwargs,
                }
                client = self.async_client if use_async else self.client
                call_func = client.responses.create
            else:
                params = {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": config.max_tokens,
                    "frequency_penalty": config.frequency_penalty,
                    "presence_penalty": config.presence_penalty,
                    "stream": stream,
                    **kwargs,
                }
                client = self.async_client if use_async else self.client
                call_func = client.chat.completions.create
        else:
            # Standard OpenAI format
            params = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": config.max_tokens,
                "top_p": config.top_p,
                "temperature": config.temperature,
                "frequency_penalty": config.frequency_penalty,
                "presence_penalty": config.presence_penalty,
                "stream": stream,
                **kwargs,
            }
            client = self.async_client if use_async else self.client
            call_func = client.chat.completions.create
        
        return params, call_func

    def _extract_response_content(self, response: Any, is_gpt_model: bool = False) -> str:
        """
        Extract text content from response object, supports different API response formats
        
        Args:
            response: API response object
            is_gpt_model: Whether it's a GPT model (using responses.create API)
            
        Returns:
            Extracted text content
        """
        if is_gpt_model:
            # GPT models use responses.create, response format may differ
            # Try multiple possible attribute paths
            if self.default_config.api_type == "responses":
                return response.output_text
            else:
                return response.choices[0].message.content
        else:
            # Standard OpenAI format
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message'):
                    return response.choices[0].message.content
                elif hasattr(response.choices[0], 'text'):
                    return response.choices[0].text
        
        # If all attempts fail, try direct access
        raise AttributeError(
            f"Unable to extract content from response object. Response type: {type(response)}, "
            f"Available attributes: {dir(response)}"
        )

    def _extract_stream_chunk_content(self, chunk: Any, is_gpt_model: bool = False) -> Optional[str]:
        """
        Extract content from streaming response chunk
        
        Args:
            chunk: Streaming response chunk object
            is_gpt_model: Whether it's a GPT model
            
        Returns:
            Extracted content, or None if not found
        """
        if is_gpt_model:
            # GPT model streaming response format
            if hasattr(chunk, 'output'):
                return chunk.output
            elif hasattr(chunk, 'text'):
                return chunk.text
            elif hasattr(chunk, 'content'):
                return chunk.content
            elif hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
                return chunk.delta.content
            elif hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    return chunk.choices[0].delta.content
        else:
            # Standard OpenAI streaming format
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    return content if content is not None else None
        
        return None

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Chat completion interface

        Args:
            messages: Message list, format: [{"role": "user", "content": "..."}]
            model: Model name, defaults to configured model
            config: Model configuration, defaults to default config
            **kwargs: Other parameters

        Returns:
            Dictionary containing response content
        """
        config = config or self.default_config
        model = model or config.name

        params, call_func = self._build_api_params_and_func(
            model, messages, config, stream=False, use_async=False, **kwargs
        )
        logger.info(json.dumps(params, indent=4))
        return self._make_request_with_retry(call_func, **params)


    def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        **kwargs,
    ):
        """
        Streaming chat completion interface

        Args:
            messages: Message list, format: [{"role": "user", "content": "..."}]
            model: Model name, defaults to configured model
            config: Model configuration, defaults to default config
            **kwargs: Other parameters

        Returns:
            Streaming response object
        """
        config = config or self.default_config
        model = model or config.name

        params, call_func = self._build_api_params_and_func(
            model, messages, config, stream=True, use_async=False, **kwargs
        )
        return self._make_request_with_retry(call_func, **params)

    def simple_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        stream: bool = False,
    ) -> Union[str, Any]:
        """
        Simple chat interface

        Args:
            prompt: User input
            system_prompt: System prompt
            model: Model name
            config: Model configuration
            stream: Whether to use streaming output

        Returns:
            Model response text content or streaming response object
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        if stream:
            return self.chat_completion_stream(messages, model, config)
        else:
            response = self.chat_completion(messages, model, config)
            is_gpt_model = 'gpt' in (model or config.name).lower()
            return self._extract_response_content(response, is_gpt_model)

    def text_completion(
        self,
        prompt: str,
        model: str = "text-davinci-003",
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Text completion interface (for legacy models)

        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum number of tokens
            temperature: Temperature parameter
            **kwargs: Other parameters

        Returns:
            Completed text
        """
        params = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        response = self._make_request_with_retry(
            self.client.completions.create, **params
        )

        return response.choices[0].text

    def embedding(
        self, text: Union[str, List[str]], model: str = "text-embedding-ada-002"
    ) -> List[List[float]]:
        """
        Text embedding interface

        Args:
            text: Text or text list to embed
            model: Embedding model name

        Returns:
            List of embedding vectors
        """
        if isinstance(text, str):
            text = [text]

        response = self._make_request_with_retry(
            self.client.embeddings.create, model=model, input=text
        )

        return [item.embedding for item in response.data]

    def image_generation(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
    ) -> List[str]:
        """
        Image generation interface

        Args:
            prompt: Image description prompt
            n: Number of images to generate
            size: Image size
            quality: Image quality
            style: Image style

        Returns:
            List of image URLs
        """
        response = self._make_request_with_retry(
            self.client.images.generate,
            model="dall-e-3",
            prompt=prompt,
            n=n,
            size=size,
            quality=quality,
            style=style,
        )

        return [item.url for item in response.data]

    def audio_transcription(
        self,
        audio_file_path: str,
        model: str = "whisper-1",
        language: Optional[str] = None,
    ) -> str:
        """
        Audio transcription interface

        Args:
            audio_file_path: Audio file path
            model: Transcription model
            language: Language code (optional)

        Returns:
            Transcribed text
        """
        with open(audio_file_path, "rb") as audio_file:
            params = {"model": model, "file": audio_file}

            if language:
                params["language"] = language

            response = self._make_request_with_retry(
                self.client.audio.transcriptions.create, **params
            )

        return response.text

    def audio_translation(self, audio_file_path: str, model: str = "whisper-1") -> str:
        """
        Audio translation interface

        Args:
            audio_file_path: Audio file path
            model: Translation model

        Returns:
            Translated text
        """
        with open(audio_file_path, "rb") as audio_file:
            response = self._make_request_with_retry(
                self.client.audio.translations.create, model=model, file=audio_file
            )

        return response.text

    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Async chat completion interface

        Args:
            messages: Message list
            model: Model name
            config: Model configuration
            **kwargs: Other parameters

        Returns:
            Dictionary containing response content
        """
        config = config or self.default_config
        model = model or config.name

        params, call_func = self._build_api_params_and_func(
            model, messages, config, stream=False, use_async=True, **kwargs
        )
        return await self._make_async_request_with_retry(call_func, **params)

    async def async_chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        **kwargs,
    ):
        """
        Async streaming chat completion interface

        Args:
            messages: Message list
            model: Model name
            config: Model configuration
            **kwargs: Other parameters

        Returns:
            Async streaming response object
        """
        config = config or self.default_config
        model = model or config.name

        params, call_func = self._build_api_params_and_func(
            model, messages, config, stream=True, use_async=True, **kwargs
        )
        return await self._make_async_request_with_retry(call_func, **params)

    async def async_stream_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        print_stream: bool = True,
    ) -> str:
        """
        Async streaming chat interface

        Args:
            prompt: User input
            system_prompt: System prompt
            model: Model name
            config: Model configuration
            print_stream: Whether to print streaming output in real-time

        Returns:
            Complete response text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        config = config or self.default_config
        model = model or config.name
        stream = await self.async_chat_completion_stream(messages, model, config)
        full_response = ""
        is_gpt_model = 'gpt' in model.lower()

        try:
            async for chunk in stream:
                content = self._extract_stream_chunk_content(chunk, is_gpt_model)
                if content is not None:
                    full_response += content
                    if print_stream:
                        print(content, end="", flush=True)

            if print_stream:
                print()  # New line

            return full_response

        except Exception as e:
            self.logger.error(f"Async streaming output error: {e}")
            raise

    def stream_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        print_stream: bool = True,
        fallback_to_normal: bool = True,
    ) -> str:
        """
        Streaming chat interface with real-time output

        Args:
            prompt: User input
            system_prompt: System prompt
            model: Model name
            config: Model configuration
            print_stream: Whether to print streaming output in real-time
            fallback_to_normal: Whether to fallback to normal chat if streaming fails

        Returns:
            Complete response text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            config = config or self.default_config
            model = model or config.name
            stream = self.chat_completion_stream(messages, model, config)
            full_response = ""
            is_gpt_model = 'gpt' in model.lower()

            for chunk in stream:
                content = self._extract_stream_chunk_content(chunk, is_gpt_model)
                if content is not None:
                    full_response += content
                    if print_stream:
                        print(content, end="", flush=True)

            if print_stream:
                print()  # New line

            return full_response

        except Exception as e:
            self.logger.error(f"Streaming output error: {e}")

            # Check if it's an organization verification issue
            if "organization must be verified" in str(
                e
            ).lower() or "unsupported_value" in str(e):
                self.logger.warning("Organization verification issue detected, streaming unavailable")
                if fallback_to_normal:
                    self.logger.info("Falling back to normal chat mode")
                    return self._fallback_to_normal_chat(
                        messages, model, config, print_stream
                    )
                else:
                    raise Exception(f"Streaming unavailable, organization verification required: {e}")
            else:
                raise

    def process_stream_response(self, stream, callback=None):
        """
        General method for processing streaming responses

        Args:
            stream: Streaming response object
            callback: Callback function that receives each chunk's content

        Returns:
            Complete response text
        """
        full_response = ""

        try:
            # Need to know if it's a GPT model, but no model info here, try general method
            for chunk in stream:
                # Try standard format first
                content = self._extract_stream_chunk_content(chunk, is_gpt_model=False)
                if content is None:
                    # If standard format fails, try GPT format
                    content = self._extract_stream_chunk_content(chunk, is_gpt_model=True)
                
                if content is not None:
                    full_response += content

                    if callback:
                        callback(content)

            return full_response

        except Exception as e:
            self.logger.error(f"Error processing streaming response: {e}")
            raise

    def _fallback_to_normal_chat(self, messages, model, config, print_stream):
        """Fallback to normal chat mode"""
        try:
            config = config or self.default_config
            model = model or config.name
            response = self.chat_completion(messages, model, config)
            is_gpt_model = 'gpt' in model.lower()
            content = self._extract_response_content(response, is_gpt_model)

            if print_stream:
                print("Streaming output unavailable, using normal mode:")
                print("-" * 50)
                print(content)
                print()

            return content
        except Exception as e:
            self.logger.error(f"Fallback to normal chat also failed: {e}")
            raise

    def check_stream_support(self) -> bool:
        """Check if streaming is supported"""
        try:
            # Try a simple streaming request
            test_messages = [{"role": "user", "content": "test"}]
            stream = self.chat_completion_stream(test_messages)

            # Try to read the first chunk
            for chunk in stream:
                break  # If we can read a chunk, streaming is supported

            return True
        except Exception as e:
            if "organization must be verified" in str(
                e
            ).lower() or "unsupported_value" in str(e):
                return False
            else:
                # Other errors, might support but failed this time
                return True

    def _make_request_with_retry(self, func, **kwargs):
        """Request with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                self.logger.info(
                    f"Sending request to OpenAI API (attempt {attempt + 1}/{self.max_retries})"
                )
                response = func(**kwargs)
                self.logger.info("Request completed successfully")
                return response
            except Exception as e:
                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}"
                )

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))  # Exponential backoff
                else:
                    self.logger.error(f"All retries failed: {str(e)}")
                    raise

    async def _make_async_request_with_retry(self, func, **kwargs):
        """Async request with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                self.logger.info(
                    f"Sending async request to OpenAI API (attempt {attempt + 1}/{self.max_retries})"
                )
                response = await func(**kwargs)
                self.logger.info("Async request completed successfully")
                return response
            except Exception as e:
                self.logger.warning(
                    f"Async request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}"
                )

                if attempt < self.max_retries - 1:
                    # await asyncio.sleep(self.retry_delay * (2**attempt))
                    pass
                else:
                    self.logger.error(f"All async retries failed: {str(e)}")
                    raise

    def set_model_config(self, **kwargs):
        """Set default model configuration"""
        for key, value in kwargs.items():
            if hasattr(self.default_config, key):
                setattr(self.default_config, key, value)
            else:
                self.logger.warning(f"Unknown configuration parameter: {key}")

    def get_usage_info(self, response) -> Dict[str, Any]:
        """Get usage information"""
        if hasattr(response, "usage") and response.usage:
            return {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return {}

    def save_config(self, file_path: str):
        """Save configuration to file"""
        config_dict = {
            "model_name": self.default_config.name,
            "max_completion_tokens": self.default_config.max_tokens,
            "top_p": self.default_config.top_p,
            "frequency_penalty": self.default_config.frequency_penalty,
            "presence_penalty": self.default_config.presence_penalty,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Configuration saved to: {file_path}")

    def load_config(self, file_path: str):
        """Load configuration from file"""
        with open(file_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        self.set_model_config(**config_dict)
        self.max_retries = config_dict.get("max_retries", self.max_retries)
        self.retry_delay = config_dict.get("retry_delay", self.retry_delay)

        self.logger.info(f"Configuration loaded from file: {file_path}")

    def switch_provider(
        self, provider: str, api_key: Optional[str] = None, model: Optional[str] = None
    ):
        """Switch model provider"""
        old_provider = self.provider
        self.provider = provider
        self.provider_config = ModelProvider.get_provider_config(provider)

        # Update API key
        if api_key:
            self.api_key = api_key
        else:
            env_vars = ENV_VARS
            new_api_key = os.getenv(env_vars[provider])
            if new_api_key:
                self.api_key = new_api_key
            else:
                raise ValueError(f"Please provide API key or set environment variable {ENV_VARS[provider]}")

        # Update base_url
        self.base_url = self.provider_config["base_url"]

        # Recreate client
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        # Update default model
        if model:
            self.default_config.name = model
        else:
            default_models = {
                "openai": "gpt-3.5-turbo",
                "deepseek": "deepseek-chat",
                "claude": "claude-3-sonnet",
                "custom": "GLM-4.5-Flash",
            }
            self.default_config.name = default_models.get(provider, "gpt-3.5-turbo")

        self.logger.info(
            f"Switched provider: {old_provider} -> {provider}, Model: {self.default_config.name}"
        )

    def list_available_models(self) -> list:
        """List all available models for current provider"""
        return ModelProvider.list_models(self.provider)

    def set_model(self, model_name: str):
        """Set currently used model"""
        available_models = self.list_available_models()
        if available_models and model_name not in available_models:
            self.logger.warning(f"Model {model_name} not in available list: {available_models}")

        self.default_config.name = model_name
        self.logger.info(f"Switched to model: {model_name}")

    def get_current_info(self) -> dict:
        """Get current configuration information"""
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.default_config.name,
            "available_models": self.list_available_models(),
            "max_tokens": self.default_config.max_tokens,
            "temperature": self.default_config.temperature,
        }

    def predict_multi(
        self,
        batch_messages,        
        model: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        max_workers: Optional[int] = 10,
        max_retries_per_task: Optional[int] = None,
        timeout: Optional[float] = 1200,
        **kwargs,
        ):
        """
        Multi-threaded batch prediction interface
        
        Args:
            batch_messages: List of message lists
            model: Model name
            config: Model configuration
            max_workers: Maximum number of worker threads
            max_retries_per_task: Maximum retry count per task, defaults to class max_retries
            timeout: Timeout for single task (seconds)
            **kwargs: Other parameters
            
        Yields:
            Tuple of (id, messages, response), where id is index, messages is message list, response is response object
        """
        max_retries_per_task = max_retries_per_task or self.max_retries
        total_tasks = len(batch_messages)
        
        def process_single_task(task_id: int, retry_count: int = 0):
            """
            Process single task with retry mechanism
            """
            messages = batch_messages[task_id]  # Get messages at function start to ensure available in exception handling
            try:
                response = self.chat_completion(messages, model, config, **kwargs)
                return task_id, messages, response, None  # Success return
            except Exception as e:
                # If last retry, return error
                if retry_count >= max_retries_per_task:
                    self.logger.error(
                        f"Task {task_id} still failed after {retry_count + 1} retries: {str(e)}"
                    )
                    return task_id, messages, None, e
                
                # Calculate backoff time
                delay = self.retry_delay * (2 ** retry_count)
                self.logger.warning(
                    f"Task {task_id} failed (attempt {retry_count + 1}/{max_retries_per_task}): {str(e)}, "
                    f"retrying after {delay} seconds"
                )
                time.sleep(delay)
                
                # Recursive retry
                return process_single_task(task_id, retry_count + 1)
        
        # Use thread pool to execute tasks
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task_id in range(total_tasks):
                future = executor.submit(process_single_task, task_id, 0)
                future_to_task[future] = task_id
            
            # Process completed tasks
            completed_tasks = set()
            failed_tasks = []
            
            # Use as_completed to get completed tasks
            try:
                for future in as_completed(future_to_task, timeout=timeout * total_tasks if timeout else None):
                    task_id = future_to_task[future]
                    try:
                        # Get result with timeout check
                        result = future.result(timeout=timeout if timeout else None)
                        task_id_result, messages, response, error = result
                        
                        if error is None:
                            # Success
                            completed_tasks.add(task_id_result)
                            yield task_id_result, messages, response
                        else:
                            # Failed (retried but still failed)
                            failed_tasks.append((task_id_result, messages, error))
                            self.logger.error(
                                f"Task {task_id_result} ultimately failed: {str(error)}"
                            )
                    except FutureTimeoutError:
                        self.logger.error(f"Task {task_id} timed out ({timeout} seconds)")
                        failed_tasks.append((task_id, batch_messages[task_id], TimeoutError(f"Task timeout: {timeout} seconds")))
                    except Exception as e:
                        self.logger.error(f"Task {task_id} execution exception: {str(e)}")
                        failed_tasks.append((task_id, batch_messages[task_id], e))
            
            except Exception as e:
                self.logger.error(f"Error occurred during batch processing: {str(e)}")
                # Ensure all tasks are processed
                for future, task_id in future_to_task.items():
                    if task_id not in completed_tasks:
                        if not future.done():
                            future.cancel()
                            failed_tasks.append((task_id, batch_messages[task_id], Exception("Task cancelled")))
                        elif future.done() and task_id not in [t[0] for t in failed_tasks]:
                            # If task completed but not in completed list, try to get result
                            try:
                                result = future.result(timeout=1)
                                task_id_result, messages, response, error = result
                                if error is None:
                                    completed_tasks.add(task_id_result)
                                    yield task_id_result, messages, response
                                else:
                                    failed_tasks.append((task_id_result, messages, error))
                            except Exception as ex:
                                failed_tasks.append((task_id, batch_messages[task_id], ex))
            
            # If there are failed tasks, log them
            if failed_tasks:
                self.logger.warning(
                    f"Batch processing complete: Success {len(completed_tasks)}/{total_tasks}, "
                    f"Failed {len(failed_tasks)}/{total_tasks}"
                )
                # Can choose to raise exception or continue
                # Here we choose to continue but log failed tasks
                for task_id, messages, error in failed_tasks:
                    self.logger.error(f"Failed task {task_id}: {type(error).__name__}: {str(error)}")


# Convenience functions
def create_openai_tool(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: str = "openai",
    model: Optional[str] = 'gpt-5',
) -> OpenAITool:
    """Convenience function to create OpenAI tool"""
    return OpenAITool(api_key, base_url, provider, model)

def create_any_tool(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: str = "glm",
    model: Optional[str] = 'glm-4.5-flash',
    api_type: Optional[str] = None,
) -> OpenAITool:
    """Convenience function to create any tool"""
    return OpenAITool(api_key=api_key, base_url=base_url, provider=provider, model=model, api_type=api_type)


def create_deepseek_tool(
    api_key: Optional[str] = None, model: str = "deepseek-chat"
) -> OpenAITool:
    """Convenience function to create DeepSeek tool"""
    return OpenAITool(api_key=api_key, provider="deepseek", model=model)


def create_claude_tool(
    api_key: Optional[str] = None, model: str = "claude-4-sonnet"
) -> OpenAITool:
    """Convenience function to create Claude tool"""
    return OpenAITool(api_key=api_key, provider="claude", model=model)


def create_qwen_tool(
    api_key: Optional[str] = None, model: str = "qwen3-max"
) -> OpenAITool:
    """Convenience function to create Qwen tool"""
    return OpenAITool(api_key=api_key, provider="qwen", model=model)


def quick_chat(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    provider: str = "openai",
) -> str:
    """Quick chat function"""
    tool = create_openai_tool(api_key, provider=provider, model=model)
    return tool.simple_chat(prompt)


def list_all_providers() -> list:
    """List all available providers"""
    return ModelProvider.list_providers()


def list_provider_models(provider: str) -> list:
    """List all models for specified provider"""
    return ModelProvider.list_models(provider)


if __name__ == "__main__":
    # Usage examples
    print("OpenAI tool class created successfully!")
    print("Supports multiple model providers: OpenAI, DeepSeek, Claude, Custom")
    print("\nUsage examples:")
    print("1. Basic usage:")
    print("   tool = OpenAITool()  # Default OpenAI")
    print("   response = tool.simple_chat('Hello, please introduce yourself')")
    print("\n2. Using DeepSeek:")
    print("   tool = OpenAITool(provider='deepseek', api_key='your-deepseek-key')")
    print("   response = tool.simple_chat('Hello')")
    print("\n3. Using Claude:")
    print("   tool = OpenAITool(provider='claude', api_key='your-claude-key')")
    print("   response = tool.simple_chat('Hello')")
    print("\n4. Custom base_url:")
    print(
        "   tool = OpenAITool(base_url='https://your-api.com/v1', api_key='your-key')"
    )
    print("\n5. Switch model:")
    print("   tool.set_model('deepseek-coder')")
    print("   tool.switch_provider('deepseek')")
    print("\n6. Streaming output:")
    print("   response = tool.stream_chat('Please write a story')")
    print("\n7. Convenience functions:")
    print("   from openai_tool import create_deepseek_tool, quick_chat")
    print("   tool = create_deepseek_tool()")
    print("   response = quick_chat('Hello', provider='deepseek')")
    print("\n8. View available models:")
    print("   print(tool.list_available_models())")
    print("   print(list_all_providers())")
