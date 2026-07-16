from __future__ import annotations

from functools import lru_cache

from curarag.config import LLMProvider, Settings, get_settings

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Provider-swappable chat client. The API key is read from settings for the
    active provider only; nothing here is ever hardcoded."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if not self.settings.active_api_key:
            raise LLMError(
                f"No API key set for provider '{self.settings.llm_provider.value}'. "
                "Set the matching *_API_KEY in .env."
            )
        self._client = self._build_client()

    def _build_client(self):
        provider = self.settings.llm_provider
        if provider in (LLMProvider.deepseek, LLMProvider.openai):
            from openai import OpenAI

            base_url = _DEEPSEEK_BASE_URL if provider == LLMProvider.deepseek else None
            return OpenAI(api_key=self.settings.active_api_key, base_url=base_url)
        if provider == LLMProvider.anthropic:
            from anthropic import Anthropic

            return Anthropic(api_key=self.settings.active_api_key)
        raise LLMError(f"Unsupported provider: {provider}")

    def complete(self, system: str, user: str, temperature: float | None = None) -> str:
        temperature = self.settings.llm_temperature if temperature is None else temperature
        provider = self.settings.llm_provider
        try:
            if provider == LLMProvider.anthropic:
                resp = self._client.messages.create(
                    model=self.settings.llm_model,
                    max_tokens=1500,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return resp.content[0].text
            resp = self._client.chat.completions.create(
                model=self.settings.llm_model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - surface provider failures uniformly
            raise LLMError(f"LLM request failed: {exc}") from exc


@lru_cache
def get_llm() -> LLMClient:
    return LLMClient()
