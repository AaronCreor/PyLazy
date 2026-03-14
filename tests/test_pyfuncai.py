"""Unit tests for the core PyFuncAI generation pipeline."""

from __future__ import annotations

from typing import Any

import pytest

from pyfuncai import (
    ConfigurationError,
    ValidationError,
    connect,
    createFunction,
    create_function,
)
from pyfuncai.cache import DiskCache
from pyfuncai.core import GeneratedFunction
from pyfuncai.providers import BaseProvider


class FakeProvider(BaseProvider):
    """Deterministic provider used to test generation behavior without network calls."""

    provider_name = "fake"

    def __init__(self, response_text: str) -> None:
        """Store the exact provider response that tests should receive."""

        super().__init__(model="fake-model", timeout=1.0)
        self.response_text = response_text
        self.calls = 0

    def generate_text(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        """Return the preconfigured provider response and count calls."""

        self.calls += 1
        return self.response_text

    def cache_identity(self) -> dict[str, Any]:
        """Return cache identity data that stays stable across test calls."""

        return {
            "provider": self.provider_name,
            "model": self.model,
        }


def test_lazy_function_builds_once_and_uses_cache(tmp_path: Any) -> None:
    """A lazy function should call the provider once and reuse the disk cache later."""

    provider = FakeProvider("""
def slugify(text: str) -> str:
    \"\"\"Convert text into a lowercase slug joined by hyphens.\"\"\"
    import re
    normalized = re.sub(r"[^a-z0-9]+", "-", text.lower())
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-")
""".strip())

    generated = create_function(
        "Convert text into a slug.",
        signature="(text: str) -> str",
        function_name="slugify",
        provider=provider,
        cache_dir=str(tmp_path),
    )

    assert generated("Hello, World!") == "hello-world"
    assert provider.calls == 1

    second_provider = FakeProvider("def slugify(text: str) -> str:\n    return 'wrong'")
    second = create_function(
        "Convert text into a slug.",
        signature="(text: str) -> str",
        function_name="slugify",
        provider=second_provider,
        cache_dir=str(tmp_path),
    )

    assert second("Hello, World!") == "hello-world"
    assert second_provider.calls == 0


def test_eager_mode_builds_immediately(tmp_path: Any) -> None:
    """Eager mode should compile the function during construction."""

    provider = FakeProvider("""
def greet(name: str) -> str:
    \"\"\"Return a simple greeting for a name.\"\"\"
    return f"Hello, {name}!"
""".strip())

    generated = create_function(
        "Greet a user.",
        signature="(name: str) -> str",
        function_name="greet",
        provider=provider,
        cache_dir=str(tmp_path),
        mode="eager",
    )

    assert generated.is_built is True
    assert provider.calls == 1
    assert generated("Alice") == "Hello, Alice!"


def test_validation_rejects_blocked_imports(tmp_path: Any) -> None:
    """Unsafe imports should be rejected before compilation happens."""

    provider = FakeProvider("""
import os

def unsafe(name: str) -> str:
    \"\"\"Try to touch the filesystem, which should be blocked.\"\"\"
    return os.getcwd()
""".strip())

    generated = create_function(
        "Return the current directory.",
        signature="(name: str) -> str",
        function_name="unsafe",
        provider=provider,
        cache_dir=str(tmp_path),
    )

    with pytest.raises(ValidationError):
        generated.build()


def test_signature_mismatch_is_rejected(tmp_path: Any) -> None:
    """Generated code must match the exact requested signature."""

    provider = FakeProvider("""
def mismatch(value: str, extra: str) -> str:
    \"\"\"Return a value with an unexpected extra argument.\"\"\"
    return value + extra
""".strip())

    generated = create_function(
        "Echo a string.",
        signature="(value: str) -> str",
        function_name="mismatch",
        provider=provider,
        cache_dir=str(tmp_path),
    )

    with pytest.raises(ValidationError):
        generated.build()


def test_camel_case_alias_returns_generated_function(tmp_path: Any) -> None:
    """The compatibility alias should map to the same implementation."""

    provider = FakeProvider("""
def add_one(value: int) -> int:
    \"\"\"Increment the provided integer by one.\"\"\"
    return value + 1
""".strip())

    generated = createFunction(
        "Add one to an integer.",
        signature="(value: int) -> int",
        function_name="add_one",
        provider=provider,
        cache_dir=str(tmp_path),
    )

    assert isinstance(generated, GeneratedFunction)
    assert generated(3) == 4


def test_connect_requires_supported_provider() -> None:
    """The public connect helper should reject unknown provider names."""

    with pytest.raises(ConfigurationError):
        connect("unknown-provider", model="unused")


def test_cache_store_round_trip(tmp_path: Any) -> None:
    """Disk cache entries should round-trip with the stored source intact."""

    cache = DiskCache(tmp_path)
    entry = cache.set(
        cache_key="abc123",
        function_name="cached_fn",
        signature="() -> str",
        source="def cached_fn() -> str:\n    return 'ok'\n",
        provider={"provider": "fake", "model": "fake-model"},
        prompt="Return ok.",
        mode="lazy",
        allow_modules=["math"],
    )

    loaded = cache.get(entry.cache_key)
    assert loaded is not None
    assert loaded.source == entry.source
