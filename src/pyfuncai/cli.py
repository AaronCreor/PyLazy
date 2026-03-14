"""Command-line entry points for PyFuncAI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import runpy
import sys

from pyfuncai.cache import DiskCache
from pyfuncai.core import _clear_registered_functions, _registered_functions


def main(argv: Sequence[str] | None = None) -> int:
    """Run the PyFuncAI command-line interface."""

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


def _build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser for the CLI."""

    parser = argparse.ArgumentParser(
        prog="pyfuncai",
        description="Build prompt-defined functions and inspect the local cache.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="Execute a script and materialize every created GeneratedFunction.",
    )
    build_parser.add_argument(
        "target",
        help="Path to a Python script that creates functions through PyFuncAI.",
    )
    build_parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate functions even if their source is already cached.",
    )
    build_parser.set_defaults(handler=_handle_build)

    cache_parser = subparsers.add_parser(
        "cache",
        help="Inspect or clear the on-disk source cache.",
    )
    cache_parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override the cache directory instead of using the default location.",
    )
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command", required=True)

    cache_path_parser = cache_subparsers.add_parser(
        "path",
        help="Print the active cache directory.",
    )
    cache_path_parser.set_defaults(handler=_handle_cache_path)

    cache_list_parser = cache_subparsers.add_parser(
        "list",
        help="List cache entries stored on disk.",
    )
    cache_list_parser.set_defaults(handler=_handle_cache_list)

    cache_remove_parser = cache_subparsers.add_parser(
        "remove",
        help="Delete one cache entry by key.",
    )
    cache_remove_parser.add_argument(
        "cache_key",
        help="Full cache key to delete.",
    )
    cache_remove_parser.set_defaults(handler=_handle_cache_remove)

    cache_clear_parser = cache_subparsers.add_parser(
        "clear",
        help="Delete every cache entry in the active cache directory.",
    )
    cache_clear_parser.set_defaults(handler=_handle_cache_clear)

    return parser


def _handle_build(args: argparse.Namespace) -> int:
    """Execute a script and prebuild every tracked generated function."""

    script_path = Path(args.target).resolve()
    if not script_path.exists():
        raise SystemExit(f"Target script does not exist: {script_path}")

    _clear_registered_functions()

    original_sys_path = list(sys.path)
    try:
        sys.path.insert(0, str(script_path.parent))
        runpy.run_path(str(script_path), run_name="__main__")
        functions = _unique_functions(_registered_functions())
        if not functions:
            print(f"No PyFuncAI functions were created by {script_path.name}.")
            return 0

        for generated in functions:
            generated.build(force=args.force)
            print(
                f"Built {generated.function_name} "
                f"[{generated.mode}] {generated.cache_key[:12]}"
            )

        print(f"Built {len(functions)} function(s) from {script_path.name}.")
        return 0
    finally:
        sys.path[:] = original_sys_path
        _clear_registered_functions()


def _handle_cache_path(args: argparse.Namespace) -> int:
    """Print the path to the active cache directory."""

    cache = DiskCache(cache_dir=args.cache_dir)
    print(str(cache.cache_dir))
    return 0


def _handle_cache_list(args: argparse.Namespace) -> int:
    """List cache entries in a human-readable format."""

    cache = DiskCache(cache_dir=args.cache_dir)
    entries = cache.entries()
    if not entries:
        print("Cache is empty.")
        return 0

    for entry in entries:
        print(
            f"{entry.function_name}\t{entry.mode}\t{entry.created_at}\t"
            f"{entry.cache_key}"
        )
    return 0


def _handle_cache_remove(args: argparse.Namespace) -> int:
    """Remove one cache entry and print the outcome."""

    cache = DiskCache(cache_dir=args.cache_dir)
    removed = cache.delete(args.cache_key)
    if removed:
        print(f"Removed cache entry {args.cache_key}.")
        return 0

    print(f"Cache entry {args.cache_key} was not found.")
    return 1


def _handle_cache_clear(args: argparse.Namespace) -> int:
    """Remove all cache entries and print how many files were deleted."""

    cache = DiskCache(cache_dir=args.cache_dir)
    removed = cache.clear()
    print(f"Removed {removed} cache entr{'y' if removed == 1 else 'ies'}.")
    return 0


def _unique_functions(functions: Sequence[object]) -> list[object]:
    """Return functions in creation order while dropping duplicate objects."""

    seen: set[int] = set()
    unique: list[object] = []
    for function in functions:
        marker = id(function)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(function)
    return unique


if __name__ == "__main__":
    raise SystemExit(main())
