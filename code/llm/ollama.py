# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
OpenAI wrapper for LLM functionality.

WARNING: This code is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

import os
import json
import re
import logging
import asyncio
from typing import Dict, Any, List, Optional

from ollama import AsyncClient
from config.config import CONFIG
import threading
from utils.logging_config_helper import get_configured_logger
from utils.logger import LogLevel


from llm.llm_provider import LLMProvider

from utils.logging_config_helper import get_configured_logger, LogLevel
logger = get_configured_logger("llm")


class ConfigurationError(RuntimeError):
    """
    Raised when configuration is missing or invalid.
    """
    pass



class OllamaProvider(LLMProvider):
    """Implementation of LLMProvider for Ollama API."""
    
    _client_lock = threading.Lock()
    _client = None

    @classmethod
    def get_api_key(cls) -> str:
        """
        Return dummy key.
        """
        return None

    @classmethod
    def get_client(cls) -> AsyncClient:
        """
        Configure and return an asynchronous Ollama client.
        """
        with cls._client_lock:  # Thread-safe client initialization
            if cls._client is None:
                cls._client = AsyncClient()
        return cls._client

    @classmethod
    def _build_messages(cls, prompt: str, schema: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Construct the system and user message sequence enforcing a JSON schema.
        """
        return [
            {
                "role": "system",
                "content": (
                    f"Provide a valid JSON response matching this schema: "
                    f"{json.dumps(schema)}"
                )
            },
            {"role": "user", "content": prompt}
        ]

    @classmethod
    def clean_response(cls, content: str) -> Dict[str, Any]:
        """
        Strip markdown fences and extract the first JSON object.
        """
        cleaned = re.sub(r"```(?:json)?\s*", "", content).strip()
        match = re.search(r"(\{.*\})", cleaned, re.S)
        if not match:
            logger.error("Failed to parse JSON from content: %r", content)
            return {}
        return json.loads(match.group(1))

    async def get_completion(
        self,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 30.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send an async chat completion request and return parsed JSON output.
        """
        # If model not provided, get it from config
        if model is None:
            provider_config = CONFIG.llm_endpoints["openai"]
            # Use the 'high' model for completions by default
            model = provider_config.models.high
        
        client = self.get_client()
        messages = self._build_messages(prompt, schema)
        try:
            response = await asyncio.wait_for(
                client.chat(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout
            )
        except asyncio.TimeoutError:
            logger.error("Completion request timed out after %s seconds", timeout)
            return {}

        try:
            return self.clean_response(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error processing OpenAI response: {e}")
            return {}



# Create a singleton instance
provider = OpenAIProvider()
