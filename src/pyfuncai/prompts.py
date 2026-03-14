"""Prompt construction and provider response parsing helpers."""

from __future__ import annotations

import ast
import json
import re

from pyfuncai.exceptions import GenerationError


def build_generation_prompt(
    *,
    prompt: str,
    function_name: str,
    signature: str,
    allowed_modules: tuple[str, ...],
) -> str:
    """Construct the user-facing generation prompt sent to the model provider."""

    allowed_module_text = ", ".join(allowed_modules)
    return f"""
Write Python source code for exactly one top-level function.

Requirements:
- Return only valid Python source code. Do not wrap it in Markdown fences.
- The function name must be `{function_name}`.
- The function signature must be exactly `{function_name}{signature}`.
- Include a concise function docstring that explains behavior, arguments, and return value.
- Use only Python's standard library.
- If you need imports, they must come only from: {allowed_module_text}.
- Do not read or write files, access the network, execute shell commands, use eval/exec,
  or import unsafe modules.
- Keep the implementation deterministic and focused on correctness.

Behavior specification:
{prompt}
""".strip()


def extract_python_source(response_text: str) -> str:
    """Extract raw Python source code from a model response."""

    stripped = response_text.strip()
    if not stripped:
        raise GenerationError("The provider response was empty.")

    json_candidate = _extract_json_candidate(stripped)
    if json_candidate is not None:
        return json_candidate

    fenced_match = re.search(r"```(?:python)?\s*(.*?)```", stripped, re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    if _looks_like_python_module(stripped):
        return stripped

    def_index = stripped.find("def ")
    if def_index != -1:
        return stripped[def_index:].strip()

    return stripped


def _extract_json_candidate(response_text: str) -> str | None:
    """Pull Python source from common JSON response formats if present."""

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        for key in ("source", "function_source", "code"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def _looks_like_python_module(candidate: str) -> bool:
    """Check whether text already parses as a Python module with a function definition."""

    try:
        module = ast.parse(candidate)
    except SyntaxError:
        return False

    return any(isinstance(node, ast.FunctionDef) for node in module.body)
