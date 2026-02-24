"""LLM client abstraction for AgentHER.

Provides a unified interface for calling language models with structured output,
supporting any OpenAI-compatible API endpoint.
"""

from __future__ import annotations

import json
import logging
from typing import TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_RETRY_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Synchronous LLM client with structured output parsing and retry logic."""

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        reraise=True,
    )
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> str:
        """Send a chat completion request and return the raw text response."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature or self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty content")
        return content

    def call_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[T],
        temperature: float | None = None,
    ) -> T:
        """Call the LLM and parse the response into a Pydantic model.

        The schema is injected into the system prompt as a JSON schema directive.
        The response is parsed with fallback strategies for robustness.
        """
        schema_json = json.dumps(output_schema.model_json_schema(), indent=2)
        augmented_system = (
            f"{system_prompt}\n\n"
            f"You MUST respond with valid JSON matching this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Do NOT include any text outside the JSON object."
        )

        raw = self.call(augmented_system, user_prompt, temperature=temperature)
        return _parse_structured_response(raw, output_schema)


class AsyncLLMClient:
    """Async LLM client for high-throughput batch processing."""

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=30),
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        reraise=True,
    )
    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature or self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty content")
        return content

    async def call_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[T],
        temperature: float | None = None,
    ) -> T:
        schema_json = json.dumps(output_schema.model_json_schema(), indent=2)
        augmented_system = (
            f"{system_prompt}\n\n"
            f"You MUST respond with valid JSON matching this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Do NOT include any text outside the JSON object."
        )
        raw = await self.call(augmented_system, user_prompt, temperature=temperature)
        return _parse_structured_response(raw, output_schema)


def _parse_structured_response(raw: str, schema: type[T]) -> T:
    """Parse LLM text output into a Pydantic model with fallback extraction."""
    text = raw.strip()
    last_error: str | None = None

    # Try direct parse first
    try:
        return schema.model_validate_json(text)
    except (ValueError, ValidationError) as e:
        last_error = str(e)

    # Extract JSON from markdown code fences
    if "```" in text:
        for block in text.split("```"):
            candidate = block.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                try:
                    return schema.model_validate_json(candidate)
                except (ValueError, ValidationError) as e:
                    last_error = str(e)
                    continue

    # Last resort: find the first {...} substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return schema.model_validate_json(text[start : end + 1])
        except (ValueError, ValidationError) as e:
            last_error = str(e)

    err_detail = f"Last parse error: {last_error}" if last_error else ""
    raise ValueError(
        f"Failed to parse LLM response into {schema.__name__}. {err_detail} Raw (first 500 chars):\n{text[:500]}"
    )
