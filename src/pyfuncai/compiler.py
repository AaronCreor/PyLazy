"""Compilation helpers for validated generated Python source."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Iterable
from types import CodeType
from typing import Any

from pyfuncai.exceptions import ValidationError

SAFE_BUILTIN_NAMES = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "Exception",
    "filter",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}


def _build_safe_builtins(allowed_modules: Iterable[str]) -> dict[str, Any]:
    """Create a restricted builtins mapping for executing generated code."""

    allowed_roots = {module.split(".")[0] for module in allowed_modules}
    safe_builtins = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}

    def restricted_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        """Allow imports only from explicitly approved standard-library modules."""

        if level != 0:
            raise ValidationError("Relative imports are not allowed in generated code.")

        root_name = name.split(".")[0]
        if root_name not in allowed_roots:
            raise ValidationError(
                f"Import '{name}' is not allowed. Approved modules: {sorted(allowed_roots)}"
            )

        return builtins.__import__(name, globals_, locals_, fromlist, level)

    safe_builtins["__import__"] = restricted_import
    return safe_builtins


def compile_function(
    source: str,
    *,
    function_name: str,
    allowed_modules: Iterable[str],
) -> Callable[..., Any]:
    """Compile validated source code and return the generated function object."""

    code: CodeType = compile(
        source, filename=f"<pyfuncai:{function_name}>", mode="exec"
    )
    namespace: dict[str, Any] = {
        "__builtins__": _build_safe_builtins(allowed_modules),
        "__name__": "pyfuncai.generated",
    }
    exec(code, namespace, namespace)

    function = namespace.get(function_name)
    if not callable(function):
        raise ValidationError(
            f"Generated source did not define a callable named '{function_name}'."
        )

    return function
