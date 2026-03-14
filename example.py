from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pyfuncai import connect, create_function


def main() -> None:
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
        cache=False,
    )

    print(greet("Alice"))


if __name__ == "__main__":
    main()
