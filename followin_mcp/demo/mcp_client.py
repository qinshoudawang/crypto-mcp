from __future__ import annotations

import asyncio
import json
import os
import threading
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Callable, Dict, List

import httpx
from mcp import ClientSession
try:
    from mcp.client.streamable_http import streamable_http_client
    _MCP_STREAMABLE_HTTP_USES_HTTP_CLIENT = True
except ImportError:  # mcp<=1.22
    from mcp.client.streamable_http import streamablehttp_client as streamable_http_client
    _MCP_STREAMABLE_HTTP_USES_HTTP_CLIENT = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger("followin_mcp.demo.mcp_client")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[mcp-client] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(getattr(logging, os.getenv("FOLLOWIN_MCP_CLIENT_LOG_LEVEL", "INFO").upper(), logging.INFO))
logger.propagate = False


class FollowinMCPClient:
    def __init__(
        self,
        server_url: str | None = None,
        headers: Dict[str, str] | None = None,
        event_callback: Callable[[Dict[str, Any]], None] | None = None,
    ) -> None:
        self.server_url = server_url or self._default_server_url()
        self.headers = headers or {}
        self.event_callback = event_callback
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._startup_error: Exception | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._session: ClientSession | None = None
        self._session_lock: asyncio.Lock | None = None

    def _default_server_url(self) -> str:
        return os.getenv(
            "FOLLOWIN_MCP_SERVER_URL",
            "http://127.0.0.1:8001/mcp",
        )

    def _ensure_runtime(self) -> None:
        if self._thread and self._thread.is_alive():
            if self._startup_error is not None:
                raise RuntimeError("Failed to initialize MCP client runtime.") from self._startup_error
            return

        self._ready.clear()
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._runtime_thread,
            name="followin-mcp-client",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError("Failed to initialize MCP client runtime.") from self._startup_error

    def _runtime_thread(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._startup_runtime())
            self._ready.set()
            loop.run_forever()
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()
        finally:
            try:
                loop.run_until_complete(self._shutdown_runtime())
            except Exception:
                pass
            loop.close()
            self._loop = None

    async def _startup_runtime(self) -> None:
        self._exit_stack = AsyncExitStack()
        if _MCP_STREAMABLE_HTTP_USES_HTTP_CLIENT:
            self._http_client = httpx.AsyncClient(headers=self.headers)
            await self._exit_stack.enter_async_context(self._http_client)
            read_stream, write_stream, get_session_id = await self._exit_stack.enter_async_context(
                streamable_http_client(
                    self.server_url,
                    http_client=self._http_client,
                )
            )
        else:
            read_stream, write_stream, get_session_id = await self._exit_stack.enter_async_context(
                streamable_http_client(
                    self.server_url,
                    headers=self.headers,
                )
            )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        self._session_lock = asyncio.Lock()
        await self._session.initialize()

    async def _shutdown_runtime(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._exit_stack = None
        self._http_client = None
        self._session = None
        self._session_lock = None

    async def _call_tool_async(self, name: str, arguments: Dict[str, Any] | None = None) -> Any:
        call_args = arguments or {}
        if self._session is None or self._session_lock is None:
            raise RuntimeError("MCP client session is not initialized.")
        logger.info("[mcp-client] call start: tool=%s", name)

        async def on_progress(progress: float, total: float | None, message: str | None) -> None:
            self._emit_event(
                {
                    "type": "tool_progress",
                    "tool_name": name,
                    "progress": progress,
                    "total": total,
                    "message": message or "",
                }
            )

        async with self._session_lock:
            result = await self._session.call_tool(
                name,
                call_args,
                progress_callback=on_progress,
            )

        if result.isError:
            error_message = self._extract_error(result)
            self._emit_event(
                {
                    "type": "tool_error",
                    "tool_name": name,
                    "message": error_message,
                }
            )
            raise RuntimeError(error_message)
        if result.structuredContent is not None:
            decoded = result.structuredContent
        else:
            decoded = self._decode_content(result.content)
        decoded = self._unwrap_result_payload(decoded)

        self._emit_event(
            {
                "type": "tool_result",
                "tool_name": name,
                "preview": self._summarize_payload(decoded),
            }
        )
        self._emit_event(
            {
                "type": "tool_complete",
                "tool_name": name,
            }
        )
        logger.info("[mcp-client] call done: tool=%s", name)
        return decoded

    async def _list_tool_names_async(self) -> List[str]:
        if self._session is None or self._session_lock is None:
            raise RuntimeError("MCP client session is not initialized.")
        async with self._session_lock:
            tools = await self._session.list_tools()
        return sorted(tool.name for tool in tools.tools)

    def list_tool_names(self) -> List[str]:
        self._ensure_runtime()
        if self._loop is None:
            raise RuntimeError("MCP client event loop is not available.")
        future = asyncio.run_coroutine_threadsafe(self._list_tool_names_async(), self._loop)
        return future.result()

    def call_tool(self, name: str, arguments: Dict[str, Any] | None = None) -> Any:
        self._ensure_runtime()
        self._emit_event(
            {
                "type": "tool_start",
                "tool_name": name,
                "arguments": arguments or {},
                "server_url": self.server_url,
                "transport": "streamable-http",
            }
        )
        if self._loop is None:
            raise RuntimeError("MCP client event loop is not available.")
        future = asyncio.run_coroutine_threadsafe(self._call_tool_async(name, arguments), self._loop)
        return future.result()

    def _emit_event(self, payload: Dict[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(payload)

    @staticmethod
    def _decode_content(content: List[Any]) -> Any:
        if len(content) == 1 and getattr(content[0], "type", None) == "text":
            text = str(getattr(content[0], "text", ""))
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text

        decoded: List[Any] = []
        for item in content:
            if getattr(item, "type", None) == "text":
                text = str(getattr(item, "text", ""))
                try:
                    decoded.append(json.loads(text))
                except json.JSONDecodeError:
                    decoded.append(text)
            else:
                decoded.append(item.model_dump())
        return decoded

    @staticmethod
    def _extract_error(result: Any) -> str:
        parts: List[str] = []
        for item in result.content:
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
        return " ".join(parts) if parts else "Unknown MCP tool error."

    @staticmethod
    def _unwrap_result_payload(payload: Any) -> Any:
        current = payload
        while (
            isinstance(current, dict)
            and set(current.keys()) == {"result"}
            and current.get("result") is not None
        ):
            current = current["result"]
        return current

    @staticmethod
    def _summarize_payload(payload: Any) -> str:
        if isinstance(payload, dict):
            keys = ", ".join(sorted(payload.keys())[:6])
            return f"object keys: {keys}" if keys else "object"
        if isinstance(payload, list):
            if not payload:
                return "0 items"
            first = payload[0]
            if isinstance(first, dict):
                label = (
                    first.get("title")
                    or first.get("name")
                    or first.get("topic_name")
                    or first.get("topic")
                    or "item"
                )
                return f"{len(payload)} items, first: {str(label)[:80]}"
            return f"{len(payload)} items"
        if isinstance(payload, str):
            return payload[:120]
        return type(payload).__name__

    def close(self) -> None:
        if self._loop is None or self._thread is None:
            return None
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)
        self._thread = None
        self._ready.clear()
        return None
