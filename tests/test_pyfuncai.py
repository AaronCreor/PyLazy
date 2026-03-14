"""Unit tests for the core PyFuncAI generation pipeline."""

from __future__ import annotations

from pathlib import Path
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
from pyfuncai.cli import main as cli_main
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


def test_cache_entries_delete_and_clear(tmp_path: Path) -> None:
    """Cache helpers should list, delete, and clear entries on disk."""

    cache = DiskCache(tmp_path)
    first = cache.set(
        cache_key="first",
        function_name="first_fn",
        signature="() -> str",
        source="def first_fn() -> str:\n    return 'first'\n",
        provider={"provider": "fake", "model": "fake-model"},
        prompt="Return first.",
        mode="lazy",
        allow_modules=["math"],
    )
    cache.set(
        cache_key="second",
        function_name="second_fn",
        signature="() -> str",
        source="def second_fn() -> str:\n    return 'second'\n",
        provider={"provider": "fake", "model": "fake-model"},
        prompt="Return second.",
        mode="eager",
        allow_modules=["math"],
    )

    entries = cache.entries()
    assert len(entries) == 2
    assert {entry.cache_key for entry in entries} == {"first", "second"}

    assert cache.delete(first.cache_key) is True
    assert cache.delete(first.cache_key) is False
    assert cache.clear() == 1
    assert cache.entries() == []


def test_cli_build_materializes_generated_functions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The build command should execute a script and prebuild created functions."""

    cache_dir = tmp_path / "cache"
    script_path = tmp_path / "build_target.py"
    script_path.write_text(
        "\n".join(
            [
                "from typing import Any",
                "",
                "from pyfuncai import create_function",
                "from pyfuncai.providers import BaseProvider",
                "",
                "",
                "class ScriptProvider(BaseProvider):",
                '    """Return a deterministic function for CLI tests."""',
                "",
                '    provider_name = "script-fake"',
                "",
                "    def __init__(self) -> None:",
                '        super().__init__(model="script-model", timeout=1.0)',
                "",
                "    def generate_text(",
                "        self,",
                "        *,",
                "        prompt: str,",
                "        system_prompt: str | None = None,",
                "        temperature: float | None = None,",
                "        max_output_tokens: int | None = None,",
                "    ) -> str:",
                '        return "def greet(name: str) -> str:\\n    return f\\"Hello, {name}!\\""',
                "",
                "    def cache_identity(self) -> dict[str, Any]:",
                '        return {"provider": self.provider_name, "model": self.model}',
                "",
                "",
                "create_function(",
                '    "Return a short greeting for the provided name.",',
                '    signature="(name: str) -> str",',
                '    function_name="greet",',
                "    provider=ScriptProvider(),",
                f"    cache_dir={str(cache_dir)!r},",
                ")",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli_main(["build", str(script_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Built greet" in captured.out
    assert len(list(cache_dir.glob("*.json"))) == 1


def test_cli_cache_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The cache subcommands should report, list, remove, and clear entries."""

    cache = DiskCache(tmp_path)
    entry = cache.set(
        cache_key="cache-entry",
        function_name="cached_fn",
        signature="() -> str",
        source="def cached_fn() -> str:\n    return 'ok'\n",
        provider={"provider": "fake", "model": "fake-model"},
        prompt="Return ok.",
        mode="lazy",
        allow_modules=["math"],
    )

    assert cli_main(["cache", "--cache-dir", str(tmp_path), "path"]) == 0
    path_output = capsys.readouterr()
    assert str(tmp_path) in path_output.out

    assert cli_main(["cache", "--cache-dir", str(tmp_path), "list"]) == 0
    list_output = capsys.readouterr()
    assert entry.cache_key in list_output.out
    assert entry.function_name in list_output.out

    assert (
        cli_main(["cache", "--cache-dir", str(tmp_path), "remove", entry.cache_key])
        == 0
    )
    remove_output = capsys.readouterr()
    assert "Removed cache entry" in remove_output.out

    assert cli_main(["cache", "--cache-dir", str(tmp_path), "clear"]) == 0
    clear_output = capsys.readouterr()
    assert "Removed 0 cache entries." in clear_output.out
