"""
LLMClient — single interface for all LLM calls in the pipeline.

All provider-specific code lives here. The rest of the codebase calls only:
    client.complete(messages, schema=None, fast=False) → dict | str
"""
from __future__ import annotations
import json
import re
from typing import Any, Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMClient:
    """Provider-agnostic LLM interface."""

    def __init__(self, settings=None):
        if settings is None:
            from app.config import settings as _s
            settings = _s
        self.settings = settings
        self._client = None
        self._init_client()

    def _init_client(self):
        provider = self.settings.llm_provider
        if provider == "anthropic":
            import anthropic
            self.provider = "anthropic"
            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
            self.scoring_model = self.settings.claude_scoring_model
            self.fast_model = self.settings.claude_fast_model
        elif provider == "openrouter":
            import httpx
            self.provider = "openrouter"
            self._base_url = "https://openrouter.ai/api/v1/chat/completions"
            self._headers = {
                "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                "Content-Type": "application/json",
            }
            self.scoring_model = self.settings.openrouter_model
            self.fast_model = self.settings.openrouter_model
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def complete(
        self,
        messages: list[dict[str, str]],
        schema: Optional[dict] = None,
        fast: bool = False,
        system: Optional[str] = None,
    ) -> Any:
        """
        Call the LLM and return the response.
        If schema is provided, attempt to parse JSON from the response.
        fast=True uses the cheaper/faster model.
        """
        model = self.fast_model if fast else self.scoring_model

        if self.provider == "anthropic":
            return self._complete_anthropic(messages, model, schema, system)
        elif self.provider == "openrouter":
            return self._complete_openrouter(messages, model, schema, system)

    def _complete_anthropic(self, messages, model, schema, system):
        import anthropic
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=4096,
            temperature=self.settings.llm_temperature,
            messages=messages,
        )
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        text = response.content[0].text
        logger.debug(f"[LLM] model={model} tokens_in={response.usage.input_tokens} tokens_out={response.usage.output_tokens}")

        if schema is not None:
            return self._parse_json(text)
        return text

    def _complete_openrouter(self, messages, model, schema, system):
        import httpx
        payload = {
            "model": model,
            "messages": messages if not system else [{"role": "system", "content": system}] + messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": 4096,
        }
        r = httpx.post(self._base_url, json=payload, headers=self._headers, timeout=60)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        if schema is not None:
            return self._parse_json(text)
        return text

    @staticmethod
    def _parse_json(text: str) -> Any:
        """Extract JSON from LLM output that may contain markdown fences."""
        # Try raw parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Strip markdown fences
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Last resort: find first { ... }
        match = re.search(r"\{[\s\S]+\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse JSON from LLM output:\n{text[:500]}")


# Module-level singleton
_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
