"""AST-based validation for generated source code."""

from __future__ import annotations

import ast

from pyfuncai.exceptions import ValidationError

DEFAULT_ALLOWED_MODULES = (
    "collections",
    "csv",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "json",
    "math",
    "operator",
    "re",
    "statistics",
    "string",
)

BLOCKED_CALL_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
}


def validate_function_source(
    source: str,
    *,
    function_name: str,
    signature: str,
    allowed_modules: tuple[str, ...],
) -> None:
    """Validate generated source against a conservative safety and shape policy."""

    try:
        module = ast.parse(source)
    except SyntaxError as error:
        raise ValidationError(
            f"Generated source is not valid Python: {error}"
        ) from error

    function_nodes = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if len(function_nodes) != 1:
        raise ValidationError(
            "Generated source must define exactly one top-level function."
        )

    function_node = function_nodes[0]
    if function_node.name != function_name:
        raise ValidationError(
            f"Generated function name '{function_node.name}' does not match '{function_name}'."
        )

    expected_signature = _parse_expected_signature(function_name, signature)
    if ast.dump(function_node.args) != ast.dump(expected_signature.args):
        raise ValidationError(
            "Generated function parameters do not match the requested signature."
        )
    if ast.dump(function_node.returns) != ast.dump(expected_signature.returns):
        raise ValidationError(
            "Generated function return annotation does not match the request."
        )

    _validate_top_level_statements(module.body, allowed_modules)
    _validate_ast_nodes(module, allowed_modules)


def _parse_expected_signature(function_name: str, signature: str) -> ast.FunctionDef:
    """Parse a requested signature into an AST function node for comparison."""

    try:
        expected_module = ast.parse(f"def {function_name}{signature}:\n    pass\n")
    except SyntaxError as error:
        raise ValidationError(
            f"Invalid requested signature '{signature}': {error}"
        ) from error

    return expected_module.body[0]


def _validate_top_level_statements(
    body: list[ast.stmt], allowed_modules: tuple[str, ...]
) -> None:
    """Restrict top-level statements to imports, a single function, and an optional docstring."""

    allowed_roots = {module.split(".")[0] for module in allowed_modules}
    function_seen = False

    for index, node in enumerate(body):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str) and index == 0:
                continue

        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed_roots:
                    raise ValidationError(
                        f"Import '{alias.name}' is not allowed. Approved modules: {sorted(allowed_roots)}"
                    )
            continue

        if isinstance(node, ast.ImportFrom):
            if node.level != 0:
                raise ValidationError("Relative imports are not allowed.")
            module_name = (node.module or "").split(".")[0]
            if module_name not in allowed_roots:
                raise ValidationError(
                    f"Import '{node.module}' is not allowed. Approved modules: {sorted(allowed_roots)}"
                )
            continue

        if isinstance(node, ast.FunctionDef) and not function_seen:
            function_seen = True
            continue

        raise ValidationError(
            "Generated source contains unsupported top-level statements."
        )


def _validate_ast_nodes(module: ast.AST, allowed_modules: tuple[str, ...]) -> None:
    """Walk the AST and reject blocked operations and suspicious dunder access."""

    allowed_roots = {module_name.split(".")[0] for module_name in allowed_modules}

    for node in ast.walk(module):
        if isinstance(node, ast.AsyncFunctionDef):
            raise ValidationError("Async functions are not currently supported.")

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_CALL_NAMES:
                raise ValidationError(
                    f"Generated source uses blocked builtin '{node.func.id}'."
                )

        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValidationError(
                "Dunder attribute access is not allowed in generated code."
            )

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed_roots:
                    raise ValidationError(f"Import '{alias.name}' is not allowed.")

        if isinstance(node, ast.ImportFrom):
            module_name = (node.module or "").split(".")[0]
            if module_name not in allowed_roots:
                raise ValidationError(f"Import '{node.module}' is not allowed.")
