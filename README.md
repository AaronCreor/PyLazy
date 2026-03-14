# PyFuncAI

[![CI](https://github.com/AaronCreor/PyFuncAI/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AaronCreor/PyFuncAI/actions/workflows/ci.yml)
[![GitHub Release](https://img.shields.io/github/v/release/AaronCreor/PyFuncAI)](https://github.com/AaronCreor/PyFuncAI/releases)
[![PyPI](https://img.shields.io/pypi/v/pyfuncai?cacheSeconds=300)](https://pypi.org/project/PyFuncAI/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyfuncai?cacheSeconds=300)](https://pypi.org/project/PyFuncAI/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PyFuncAI is a Python library for defining functions with prompts and generating their implementations with an LLM at runtime.

The library is built around a small API:

- connect to a model provider
- describe a function in natural language
- request lazy or eager generation
- optionally cache generated source for reuse across runs

PyFuncAI currently supports local Ollama models and remote OpenAI and Gemini providers.

## Installation

Install from PyPI:

```bash
pip install pyfuncai
```

For local development:

```bash
pip install -e .[dev]
```

## Quick Start

### Ollama

```python
from pyfuncai import connect, create_function

connect(
    "ollama",
    model="qwen3.5:latest",
    base_url="http://localhost:11434",
    timeout=180,
)

greet = create_function(
    "Return a short greeting for the provided name.",
    signature="(name: str) -> str",
    function_name="greet",
    mode="eager",
    cache=True,
)

print(greet("Alice"))
```

### OpenAI

```python
from pyfuncai import connect, create_function

connect(
    "openai",
    model="gpt-5-mini",
    api_key="YOUR_OPENAI_API_KEY",
)

slugify = create_function(
    "Convert text into a URL-friendly slug.",
    signature="(text: str) -> str",
    function_name="slugify",
)
```

### Gemini

```python
from pyfuncai import connect, create_function

connect(
    "gemini",
    model="gemini-2.5-flash",
    api_key="YOUR_GEMINI_API_KEY",
)

summarize = create_function(
    "Summarize a short paragraph in one sentence.",
    signature="(text: str) -> str",
    function_name="summarize",
)
```

## Core Concepts

### `connect(provider, **config)`

Registers the default provider used by subsequent `create_function()` calls.

Supported providers:

- `ollama`
- `openai`
- `gemini`

Common configuration keys:

- `model`
- `timeout`
- `cache_dir`

Provider-specific keys:

- Ollama: `base_url`
- OpenAI: `api_key`, `base_url`
- Gemini: `api_key`, `base_url`

### `create_function(...)`

Creates a prompt-defined callable.

Common arguments:

- `prompt`
- `signature`
- `function_name`
- `mode="lazy"` or `mode="eager"`
- `cache=True`
- `provider`
- `cache_dir`
- `system_prompt`
- `temperature`
- `max_output_tokens`
- `allow_modules`

The return value is a `GeneratedFunction`. It behaves like a normal callable and also exposes:

- `.build(force=False)`
- `.source`
- `.cache_key`
- `.is_built`

`createFunction(...)` is available as a compatibility alias.

## Generation Modes

- `lazy`: generation happens on first invocation
- `eager`: generation happens during `create_function()`

## Caching

Generated source can be cached on disk and reused across runs. Cache keys include:

- prompt
- signature
- function name
- provider identity
- mode
- allowed modules

The default cache location is OS-specific. You can override it globally with `connect(..., cache_dir=...)` or per function with `create_function(..., cache_dir=...)`.

## CLI

PyFuncAI ships with a small command-line interface for prebuilding functions and
inspecting cached source.

Prebuild every prompt-defined function created by a script:

```bash
pyfuncai build path/to/script.py
```

Force regeneration even when cached source already exists:

```bash
pyfuncai build --force path/to/script.py
```

Inspect the cache directory:

```bash
pyfuncai cache path
pyfuncai cache list
pyfuncai cache clear
```

You can point cache commands at a different directory with `--cache-dir`.

## Safety
PyFuncAI validates generated code before execution. The current implementation includes:

- AST validation
- exact signature matching
- restricted imports
- restricted builtins at execution time

This is still a code-generation library, not a hardened sandbox. Generated code should be treated as untrusted.

## Development

The repository uses three long-lived branches:

- `main`: stable development and PyPI releases
- `preview`: prerelease validation and release-candidate testing
- `dev`: active integration branch

Automation lives under `.github/workflows/`:

- `ci.yml`: formatting, tests, and package build checks on `main`, `preview`, `dev`, and pull requests
- `release.yml`: stable releases from `main` tags such as `v0.1.3`
- `prerelease.yml`: GitHub prereleases from `preview` tags such as `v0.1.4rc1`

Run the test suite locally:

```bash
pytest
```

Run the optional live Ollama test:

```bash
PYFUNCAI_RUN_OLLAMA_TESTS=1 pytest tests/test_ollama_live.py
```

PowerShell:

```powershell
$env:PYFUNCAI_RUN_OLLAMA_TESTS = "1"
pytest tests/test_ollama_live.py
```

## Releasing

Stable releases are published to PyPI from `main`.

Typical stable release flow:

1. Update `version` in `pyproject.toml`.
2. Commit and push to `main`.
3. Create and push a matching tag.

Example:

```bash
git checkout main
git pull
git tag v0.1.3
git push origin main
git push origin v0.1.3
```

Preview releases are created from `preview` and published as GitHub prereleases only.

Example:

```bash
git checkout preview
git pull
git tag v0.1.4rc1
git push origin preview
git push origin v0.1.4rc1
```

## Contributing

Contributions are welcome. A good contribution usually includes:

- a clear problem statement or use case
- tests for behavior changes
- documentation updates when the public API changes

Before opening a pull request:

```bash
python -m black src tests example.py
python -m pytest
python -m build
```

## License

Apache-2.0
