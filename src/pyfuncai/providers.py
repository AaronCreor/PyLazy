"""Provider abstraction for local and remote model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pyfuncai.exceptions import ConfigurationError, ProviderError


class BaseProvider(ABC):
    """Abstract base class for providers that can generate Python source text."""

    provider_name: str

    def __init__(self, *, model: str, timeout: float = 60.0) -> None:
        """Store the common model selection and request timeout settings."""

        self.model = model
        self.timeout = timeout

    @abstractmethod
    def generate_text(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        """Generate a text completion for a prompt."""

    @abstractmethod
    def cache_identity(self) -> dict[str, Any]:
        """Return provider settings that affect cache reuse."""


class OllamaProvider(BaseProvider):
    """Generate code using a locally or remotely hosted Ollama instance."""

    provider_name = "ollama"

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        """Configure an Ollama endpoint and model name."""

        super().__init__(model=model, timeout=timeout)
        self.base_url = base_url.rstrip("/")

    def generate_text(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        """Call Ollama's `/api/generate` endpoint and return the response text."""

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_output_tokens is not None:
            options["num_predict"] = max_output_tokens
        if options:
            payload["options"] = options

        response = _post_json(
            f"{self.base_url}/api/generate",
            payload=payload,
            headers=None,
            timeout=self.timeout,
        )
        text = response.get("response")
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("Ollama returned an empty `response` field.")
        return text

    def cache_identity(self) -> dict[str, Any]:
        """Return cache-sensitive Ollama settings."""

        return {
            "provider": self.provider_name,
            "model": self.model,
            "base_url": self.base_url,
        }


class OpenAIProvider(BaseProvider):
    """Generate code using the OpenAI Responses API."""

    provider_name = "openai"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ) -> None:
        """Configure an OpenAI API client using an explicit or env-based key."""

        super().__init__(model=model, timeout=timeout)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        if not self.api_key:
            raise ConfigurationError(
                "OpenAIProvider requires `api_key=` or OPENAI_API_KEY."
            )

    def generate_text(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        """Call the OpenAI Responses API and normalize the text output."""

        content = []
        if system_prompt:
            content.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                }
            )
        content.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "input": content,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens

        response = _post_json(
            f"{self.base_url}/responses",
            payload=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )

        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        output = response.get("output", [])
        for item in output:
            for content_item in item.get("content", []):
                text = content_item.get("text")
                if isinstance(text, str) and text.strip():
                    return text

        raise ProviderError("OpenAI did not return any text content.")

    def cache_identity(self) -> dict[str, Any]:
        """Return cache-sensitive OpenAI settings."""

        return {
            "provider": self.provider_name,
            "model": self.model,
            "base_url": self.base_url,
        }


class GeminiProvider(BaseProvider):
    """Generate code using the Gemini REST API."""

    provider_name = "gemini"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 60.0,
    ) -> None:
        """Configure a Gemini API client using an explicit or env-based key."""

        super().__init__(model=model, timeout=timeout)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.base_url = base_url.rstrip("/")
        if not self.api_key:
            raise ConfigurationError(
                "GeminiProvider requires `api_key=` or GEMINI_API_KEY."
            )

    def generate_text(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        """Call Gemini's `generateContent` endpoint and return the first text part."""

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ]
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_output_tokens is not None:
            generation_config["maxOutputTokens"] = max_output_tokens
        if generation_config:
            payload["generationConfig"] = generation_config

        response = _post_json(
            f"{self.base_url}/models/{self.model}:generateContent",
            payload=payload,
            headers=None,
            timeout=self.timeout,
            params={"key": self.api_key},
        )
        candidates = response.get("candidates", [])
        for candidate in candidates:
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return text

        raise ProviderError("Gemini did not return any text content.")

    def cache_identity(self) -> dict[str, Any]:
        """Return cache-sensitive Gemini settings."""

        return {
            "provider": self.provider_name,
            "model": self.model,
            "base_url": self.base_url,
        }


def create_provider(provider: str, **config: Any) -> BaseProvider:
    """Instantiate a provider class from a user-facing provider name."""

    normalized = provider.strip().lower()
    if normalized == "ollama":
        return OllamaProvider(**config)
    if normalized == "openai":
        return OpenAIProvider(**config)
    if normalized == "gemini":
        return GeminiProvider(**config)

    raise ConfigurationError(
        f"Unsupported provider '{provider}'. Expected one of: ollama, openai, gemini."
    )


def _post_json(
    url: str,
    *,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None,
    timeout: float,
    params: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Issue a JSON POST request and return the decoded response payload."""

    target_url = url
    if params:
        target_url = f"{url}?{urlencode(dict(params))}"

    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)

    try:
        request = Request(
            target_url,
            data=json.dumps(dict(payload)).encode("utf-8"),
            headers=request_headers,
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        message = error.read().decode("utf-8", errors="replace").strip()
        raise ProviderError(
            f"Provider request failed with HTTP {error.code}: {message}"
        ) from error
    except URLError as error:
        raise ProviderError(f"Provider request failed: {error}") from error

    if not isinstance(data, dict):
        raise ProviderError("Provider response was not a JSON object.")
    return data
