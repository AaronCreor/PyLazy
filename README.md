# PyFuncAI

[![CI](https://github.com/AaronCreor/PyFuncAI/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AaronCreor/PyFuncAI/actions/workflows/ci.yml)
[![Release](https://github.com/AaronCreor/PyFuncAI/actions/workflows/release.yml/badge.svg)](https://github.com/AaronCreor/PyFuncAI/actions/workflows/release.yml)
[![codecov](https://codecov.io/gh/AaronCreor/PyFuncAI/branch/main/graph/badge.svg)](https://codecov.io/gh/AaronCreor/PyFuncAI)
[![PyPI](https://img.shields.io/pypi/v/pyfuncai.svg)](https://pypi.org/project/pyfuncai/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyfuncai.svg)](https://pypi.org/project/pyfuncai/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PyFuncAI is a Python library for defining functions with prompts and letting an LLM generate the implementation on demand.

It is built as a normal PyPI-style package, exposes a small public API, supports local Ollama models and remote OpenAI/Gemini providers, and can cache generated source on disk for reuse across runs.

## Branch Strategy

The repository is set up around three long-lived branches:

- `main`: production and release branch, including PyPI deployment
- `preview`: pre-release validation branch
- `dev`: active development branch

The GitHub Actions workflows are wired to run CI on all three branches, while PyPI publishing is restricted to version tags that point to commits on `main`.

## What It Does

PyFuncAI turns a prompt plus a Python signature into a callable object:

- `mode="lazy"` delays generation until first use.
- `mode="eager"` builds immediately.
- `cache=True` stores generated source on disk.
- `.build()` forces materialization ahead of time.
- `.source` exposes the generated Python source once available.

The generated code is validated before execution:

- exactly one top-level function must be returned
- the generated signature must exactly match the requested signature
- only approved standard-library imports are allowed
- obvious dangerous builtins such as `open`, `eval`, and `exec` are blocked

This is still experimental. Validation is intentionally conservative, but LLM-generated code should still be treated as untrusted.

## Installation

Once published:

```bash
pip install pyfuncai
```

For local development from this repo:

```bash
pip install -e .[dev]
```

To run the minimal example directly from the repository checkout:

```bash
python example.py
```

## Automation

GitHub Actions workflows live in `.github/workflows/`:

- `ci.yml`: formatting, tests, coverage, and package builds on `main`, `preview`, `dev`, and pull requests
- `release.yml`: verifies the tag commit is on `main`, builds distributions, creates a GitHub release, and publishes to PyPI

The coverage badge expects Codecov to be enabled for the repository. The CI workflow already uploads `coverage.xml`; once Codecov is connected, the badge will begin reporting real data.

## Quick Start

### Ollama

This was smoke-tested locally on March 13, 2026 with:

- Ollama at `http://localhost:11434`
- model `qwen3.5:latest`

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
print(greet.source)
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

`OPENAI_API_KEY` is also supported.

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

`GEMINI_API_KEY` is also supported.

## Public API

### `connect(provider, **config)`

Registers the default provider used by later `create_function()` calls.

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

Main arguments:

- `prompt`
- `signature`
- `cache=True`
- `mode="lazy"` or `mode="eager"`
- `function_name`
- `provider`
- `cache_dir`
- `system_prompt`
- `temperature`
- `max_output_tokens`
- `allow_modules`

It returns a `GeneratedFunction`, which is callable and also exposes:

- `.build(force=False)`
- `.source`
- `.cache_key`
- `.is_built`

`createFunction(...)` is available as a compatibility alias.

## Cache Behavior

Cached source is keyed by:

- prompt
- requested signature
- function name
- provider identity
- mode
- allowed module list

By default PyFuncAI uses an OS-appropriate user cache directory. You can override that globally in `connect(..., cache_dir=...)` or per function with `create_function(..., cache_dir=...)`.

## Live Testing

The unit test suite is offline and deterministic:

```bash
pytest
```

An optional live Ollama integration test is included but skipped by default:

```bash
PYFUNCAI_RUN_OLLAMA_TESTS=1 pytest tests/test_ollama_live.py
```

PowerShell:

```powershell
$env:PYFUNCAI_RUN_OLLAMA_TESTS = "1"
pytest tests/test_ollama_live.py
```

Optional environment variables for the live test:

- `PYFUNCAI_OLLAMA_MODEL`
- `PYFUNCAI_OLLAMA_BASE_URL`
- `PYFUNCAI_OLLAMA_TIMEOUT`

## Project Layout

```text
.github/workflows/
  ci.yml
  release.yml
src/pyfuncai/
  __init__.py
  cache.py
  compiler.py
  core.py
  exceptions.py
  prompts.py
  providers.py
  validation.py
tests/
example.py
```

The package uses a `src/` layout and `pyproject.toml`, which keeps it ready for normal wheel and sdist publishing.

## Safety Notes

PyFuncAI validates generated code, but it does not provide hard sandboxing. The current implementation focuses on:

- AST checks
- restricted imports
- restricted builtins during execution

That helps, but it is not the same thing as secure isolation. Do not use generated code against secrets, production systems, or privileged environments without stronger containment.

## Status

Current state of the repository:

- distributable package layout via `pyproject.toml`
- Ollama/OpenAI/Gemini provider adapters
- lazy and eager generation
- disk cache
- validation and restricted execution
- unit tests and optional live Ollama test
- CI, release automation, and PyPI publishing workflow

## Publishing To PyPI

PyFuncAI is configured for PyPI trusted publishing with GitHub Actions. The release workflow file is:

- `.github/workflows/release.yml`

### 1. Configure GitHub

In the GitHub repository:

1. Create or confirm the long-lived branches: `main`, `preview`, and `dev`.
2. In `Settings -> Environments`, create an environment named `pypi`.
3. Optionally add protection rules so only approved maintainers can publish.
4. Push the workflow files to GitHub.

Recommended:

- protect `main`
- create tags like `v0.1.0` only from `main`

### 2. Configure PyPI Trusted Publishing

In PyPI, add a new trusted publisher using the GitHub tab with these values:

- PyPI Project Name: `pyfuncai`
- Owner: `AaronCreor`
- Repository name: `PyFuncAI`
- Workflow name: `release.yml`
- Environment name: `pypi`

This matches the current repository and workflow layout.

### 3. Publish A Release

From a clean `main` branch state:

1. Update `version` in `pyproject.toml`.
2. Commit and push to `main`.
3. Create and push a version tag:

```bash
git checkout main
git pull
git tag v0.1.0
git push origin main
git push origin v0.1.0
```

That tag triggers `release.yml`, which will:

- verify the tag commit is reachable from `main`
- build the wheel and sdist
- create a GitHub Release with the built artifacts
- publish the package to PyPI through trusted publishing

### 4. Create The Extra Branches On GitHub

Local branches now exist, but GitHub will not see them until you push them:

```bash
git push -u origin preview
git push -u origin dev
```

### References

- PyPI trusted publishing docs: https://docs.pypi.org/trusted-publishers/
- PyPA publishing guide: https://packaging.python.org/en/latest/tutorials/packaging-projects/
- GitHub Actions badges: https://docs.github.com/actions/monitoring-and-troubleshooting-workflows/monitoring-workflows/adding-a-workflow-status-badge

## License

Apache-2.0
