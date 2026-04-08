from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, List

from dotenv import load_dotenv

from ..core.models import UserProfile
from ..demo.agent import FollowinChatAgent
from ..demo.mcp_client import FollowinMCPClient


logger = logging.getLogger("followin_mcp.acp.server")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[acp] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(getattr(logging, os.getenv("FOLLOWIN_ACP_LOG_LEVEL", "INFO").upper(), logging.INFO))
logger.propagate = False

try:
    from acp import (
        Agent,
        InitializeResponse,
        NewSessionResponse,
        PromptResponse,
        SessionNotification,
        stdio_streams,
    )
    from acp.schema import (
        AgentCapabilities,
        AgentMessageChunk,
        CloseSessionResponse,
        Implementation,
        TextContentBlock,
        ToolCallProgress,
        ToolCallStart,
    )
    from acp.stdio import AgentSideConnection
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(
        "agent-client-protocol is required to run the ACP server. "
        "Install project dependencies first."
    ) from exc


def _default_profile() -> UserProfile:
    profile_json = os.getenv("FOLLOWIN_ACP_PROFILE_JSON", "").strip()
    if profile_json:
        try:
            data = json.loads(profile_json)
            if isinstance(data, dict):
                return UserProfile(**data)
        except Exception:
            pass

    return UserProfile(
        user_id=os.getenv("FOLLOWIN_ACP_USER_ID", "acp_user"),
        interests=["Bitcoin", "Ethereum"],
        muted_topics=[],
        preferred_languages=["zh"],
        risk_preference="medium",
        followed_projects=[],
        followed_kols=[],
    )


def _prompt_to_text(prompt_blocks: List[Any]) -> str:
    parts: List[str] = []
    for block in prompt_blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text)
            continue
        if isinstance(block, dict):
            value = block.get("text")
            if isinstance(value, str) and value.strip():
                parts.append(value)
    return "\n".join(parts).strip()


class FollowinACPAgent(Agent):
    def __init__(self, conn: Any) -> None:
        super().__init__()
        self._conn = conn
        self._sessions: Dict[str, FollowinChatAgent] = {}
        self._tool_calls_by_session: Dict[str, Dict[str, str]] = {}
        self._loop = asyncio.get_running_loop()
        logger.info("agent instance created")

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Any = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        logger.info("initialize: protocol_version=%s", protocol_version)
        response = InitializeResponse(protocolVersion=protocol_version)
        response.agent_info = Implementation(
            name="followin",
            title="Followin Crypto Research Agent",
            version="0.1.0",
        )
        response.agent_capabilities = AgentCapabilities()
        response.field_meta = {
            "description": (
                "A crypto research and recommendation agent built on Followin data. "
                "Useful for latest headlines, project news, KOL opinions, trending narratives, "
                "and personalized crypto event feeds."
            ),
            "recommendedPrompts": [
                "给我推荐今天值得看的 crypto 内容",
                "总结一下 ETH 最近的重要资讯",
                "看看 SOL 最近有哪些 KOL 观点",
                "最近市场在讨论哪些热门叙事",
            ],
            "domains": ["crypto", "news", "research", "recommendation"],
        }
        logger.info("initialize complete")
        return response

    async def new_session(self, cwd: str, mcp_servers: Any = None, **kwargs: Any) -> Any:
        logger.info("new_session: cwd=%s", cwd)
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = self._build_chat_agent(session_id)
        self._tool_calls_by_session[session_id] = {}
        logger.info("session created: %s", session_id)
        response = NewSessionResponse(sessionId=session_id)
        return response

    async def prompt(
        self,
        prompt: List[Any],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        logger.info("prompt start: session_id=%s message_id=%s", session_id, message_id)
        agent = self._sessions.get(session_id)
        if agent is None:
            agent = self._build_chat_agent(session_id)
            self._sessions[session_id] = agent
            self._tool_calls_by_session[session_id] = {}

        user_message = _prompt_to_text(prompt)
        if not user_message:
            user_message = "继续"
        logger.info("prompt message: %s", user_message[:200])

        for payload in agent.chat_stream(user_message):
            if payload.get("type") != "assistant_chunk":
                continue
            delta = payload.get("delta", "")
            if not isinstance(delta, str) or not delta:
                continue
            await self._conn.sessionUpdate(
                SessionNotification(
                    sessionId=session_id,
                    update=AgentMessageChunk(
                        sessionUpdate="agent_message_chunk",
                        content=TextContentBlock(type="text", text=delta),
                    ),
                )
            )

        logger.info("prompt complete: session_id=%s", session_id)
        return PromptResponse(stopReason="end_turn", userMessageId=message_id)

    async def cancel(self, params: Any) -> None:
        return None

    async def close_session(self, session_id: str, **kwargs: Any) -> Any:  # pragma: no cover - ACP lifecycle callback
        logger.info("close_session: %s", session_id)
        agent = self._sessions.pop(session_id, None)
        self._tool_calls_by_session.pop(session_id, None)
        if agent is not None:
            agent.close()
        return CloseSessionResponse()

    def _build_chat_agent(self, session_id: str) -> FollowinChatAgent:
        return FollowinChatAgent(
            mcp_client=FollowinMCPClient(
                event_callback=self._build_tool_event_callback(session_id),
            ),
            user=_default_profile(),
            model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    def _build_tool_event_callback(self, session_id: str):
        def emit(payload: Dict[str, Any]) -> None:
            asyncio.run_coroutine_threadsafe(
                self._handle_tool_event(session_id, payload),
                self._loop,
            )

        return emit

    async def _handle_tool_event(self, session_id: str, payload: Dict[str, Any]) -> None:
        event_type = payload.get("type")
        tool_name = str(payload.get("tool_name", "tool"))
        logger.info("tool event: session_id=%s type=%s tool=%s", session_id, event_type, tool_name)
        session_tool_calls = self._tool_calls_by_session.setdefault(session_id, {})

        if event_type == "tool_start":
            tool_call_id = uuid.uuid4().hex
            session_tool_calls[tool_name] = tool_call_id
            update = ToolCallStart(
                session_update="tool_call",
                tool_call_id=tool_call_id,
                title=tool_name,
                kind="fetch",
                status="in_progress",
                raw_input=payload.get("arguments", {}),
            )
        else:
            tool_call_id = session_tool_calls.get(tool_name)
            if not tool_call_id:
                return
            if event_type == "tool_progress":
                progress = payload.get("progress")
                total = payload.get("total")
                message = payload.get("message", "")
                update = ToolCallProgress(
                    session_update="tool_call_update",
                    tool_call_id=tool_call_id,
                    status="in_progress",
                    title=tool_name,
                    raw_output={
                        "progress": progress,
                        "total": total,
                        "message": message,
                    },
                )
            elif event_type == "tool_result":
                update = ToolCallProgress(
                    session_update="tool_call_update",
                    tool_call_id=tool_call_id,
                    status="in_progress",
                    title=tool_name,
                    raw_output={"preview": payload.get("preview", "")},
                )
            elif event_type == "tool_error":
                update = ToolCallProgress(
                    session_update="tool_call_update",
                    tool_call_id=tool_call_id,
                    status="failed",
                    title=tool_name,
                    raw_output={"message": payload.get("message", "tool error")},
                )
                session_tool_calls.pop(tool_name, None)
            elif event_type == "tool_complete":
                update = ToolCallProgress(
                    session_update="tool_call_update",
                    tool_call_id=tool_call_id,
                    status="completed",
                    title=tool_name,
                )
                session_tool_calls.pop(tool_name, None)
            else:
                return

        await self._conn.sessionUpdate(
            SessionNotification(
                sessionId=session_id,
                update=update,
            )
        )


async def _run() -> None:
    load_dotenv()
    logger.info("starting ACP stdio server")
    reader, writer = await stdio_streams()
    AgentSideConnection(lambda conn: FollowinACPAgent(conn), writer, reader)
    logger.info("ACP stdio server ready")
    await asyncio.Event().wait()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
