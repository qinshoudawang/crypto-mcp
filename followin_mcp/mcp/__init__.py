from __future__ import annotations

from typing import Any


def main() -> None:
    from .server import main as server_main

    server_main()


def __getattr__(name: str) -> Any:
    if name == "mcp":
        from .server import mcp

        return mcp
    raise AttributeError(name)


__all__ = ["main", "mcp"]
