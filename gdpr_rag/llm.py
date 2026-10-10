"""Pluggable LLM backend for answer generation.

Two interchangeable backends are supported and selected via ``LLM_BACKEND``:

* ``openai``  - any OpenAI-compatible Chat Completions API (OpenAI, Groq,
  OpenRouter, a local server, ...). Configure the key, base URL and model name
  in ``.env``.
* ``ollama``  - a local model served by Ollama (fully offline, no API key).

Both are driven at temperature 0 for deterministic, grounded answers.
"""

from __future__ import annotations

from typing import Protocol

from .config import Settings

_TEMPERATURE = 0.0


class LLMClient(Protocol):
    """Minimal contract the answer layer depends on."""

    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class OpenAICompatibleClient:
    """Chat completions against any OpenAI-compatible endpoint."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()


class OllamaClient:
    """Chat completions against a local Ollama server."""

    def __init__(self, model: str, host: str | None = None):
        import ollama

        self._client = ollama.Client(host=host) if host else ollama.Client()
        self._model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat(
            model=self._model,
            options={"temperature": _TEMPERATURE},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"].strip()


def build_llm_client(settings: Settings) -> LLMClient:
    """Instantiate the configured LLM backend."""
    if settings.llm_backend == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "LLM_BACKEND=openai requires OPENAI_API_KEY (set it in .env)."
            )
        return OpenAICompatibleClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
        )
    if settings.llm_backend == "ollama":
        return OllamaClient(model=settings.ollama_model, host=settings.ollama_host)
    raise ValueError(f"Unknown LLM backend {settings.llm_backend!r}.")
