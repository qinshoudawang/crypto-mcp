from __future__ import annotations

import json
import os
import random
import uuid
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..core.models import UserProfile
from .agent import FollowinChatAgent
from .mcp_client import FollowinMCPClient


class ProfilePayload(BaseModel):
    user_id: str
    interests: List[str] = Field(default_factory=list)
    muted_topics: List[str] = Field(default_factory=list)
    preferred_languages: List[str] = Field(default_factory=lambda: ["zh"])
    risk_preference: str = "medium"
    followed_projects: List[str] = Field(default_factory=list)
    followed_kols: List[str] = Field(default_factory=list)

    def to_user_profile(self) -> UserProfile:
        return UserProfile(**self.model_dump())


class ChatRequest(BaseModel):
    session_id: str
    profile: ProfilePayload
    user_message: str


class SessionRequest(BaseModel):
    profile: ProfilePayload


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = PROJECT_ROOT / "web"

app = FastAPI(title="Followin MCP Agent Demo")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
CHAT_SESSIONS: Dict[str, FollowinChatAgent] = {}
SESSION_EVENT_QUEUES: Dict[str, Queue[Dict[str, Any]]] = {}


PROFILE_PRESETS = [
    {
        "user_id": "solana_degen",
        "interests": ["Solana", "DeFi", "AI Agents", "Base"],
        "muted_topics": ["NFT"],
        "followed_projects": ["Jupiter", "Base"],
        "risk_preference": "high",
    },
    {
        "user_id": "btc_macro",
        "interests": ["Bitcoin", "ETF", "Macro", "Institutional Adoption"],
        "muted_topics": ["Meme"],
        "followed_projects": ["Bitcoin", "Ethereum"],
        "risk_preference": "medium",
    },
    {
        "user_id": "airdrop_hunter",
        "interests": ["Airdrop", "Layer2", "Base", "Governance"],
        "muted_topics": ["Macro"],
        "followed_projects": ["Arbitrum", "Base"],
        "risk_preference": "high",
    },
    {
        "user_id": "safety_first",
        "interests": ["Security", "Exploit", "Ethereum", "Governance"],
        "muted_topics": ["Meme"],
        "followed_projects": ["Ethereum", "Safe"],
        "risk_preference": "low",
    },
]


def _build_session_event_callback(session_id: str):
    def emit(payload: Dict[str, Any]) -> None:
        queue = SESSION_EVENT_QUEUES.get(session_id)
        if queue is None:
            return
        queue.put({"session_id": session_id, **payload})

    return emit


def _emit_session_event(session_id: str, payload: Dict[str, Any]) -> None:
    queue = SESSION_EVENT_QUEUES.get(session_id)
    if queue is None:
        return
    queue.put({"session_id": session_id, **payload})


def _build_agent(user: UserProfile, session_id: str) -> FollowinChatAgent:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to start the LangChain web demo.")

    event_callback = _build_session_event_callback(session_id)
    return FollowinChatAgent(
        mcp_client=FollowinMCPClient(event_callback=event_callback),
        user=user,
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def _ensure_agent_profile(agent: FollowinChatAgent, profile: UserProfile, session_id: str) -> None:
    if not agent.update_user(profile):
        return
    _emit_session_event(
        session_id,
        {
            "type": "profile_update",
            "profile": profile.__dict__,
        },
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/profile/random")
def random_profile(exclude_user_id: str | None = Query(default=None)) -> Dict[str, Any]:
    candidates = PROFILE_PRESETS
    if exclude_user_id:
        filtered = [preset for preset in PROFILE_PRESETS if preset.get("user_id") != exclude_user_id]
        if filtered:
            candidates = filtered

    preset = random.choice(candidates)
    return ProfilePayload(**preset).model_dump()


@app.post("/api/session")
def create_session(request: SessionRequest) -> Dict[str, Any]:
    user = request.profile.to_user_profile()
    session_id = uuid.uuid4().hex
    SESSION_EVENT_QUEUES[session_id] = Queue()
    CHAT_SESSIONS[session_id] = _build_agent(user, session_id)
    return {"session_id": session_id, "profile": user.__dict__}


@app.post("/api/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    user = request.profile.to_user_profile()
    agent = CHAT_SESSIONS.get(request.session_id)
    if agent is None:
        SESSION_EVENT_QUEUES[request.session_id] = Queue()
        agent = _build_agent(user, request.session_id)
        CHAT_SESSIONS[request.session_id] = agent
    else:
        _ensure_agent_profile(agent, user, request.session_id)

    result = agent.chat(request.user_message)
    return {
        "session_id": request.session_id,
        "assistant_message": result["assistant_message"],
        "tool_runs": result["tool_runs"],
        "history_size": result["history_size"],
        "profile": agent.user.__dict__,
    }


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    user = request.profile.to_user_profile()
    agent = CHAT_SESSIONS.get(request.session_id)
    if agent is None:
        SESSION_EVENT_QUEUES[request.session_id] = Queue()
        agent = _build_agent(user, request.session_id)
        CHAT_SESSIONS[request.session_id] = agent
    else:
        _ensure_agent_profile(agent, user, request.session_id)

    def event_stream():
        yield "event: start\ndata: {}\n\n"
        try:
            for payload in agent.chat_stream(request.user_message):
                event_type = str(payload.get("type", "message"))
                yield f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            error_payload = {"type": "error", "message": str(exc)}
            yield f"event: error\ndata: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/session/{session_id}/events")
def stream_session_events(session_id: str) -> StreamingResponse:
    queue = SESSION_EVENT_QUEUES.get(session_id)
    if queue is None:
        queue = Queue()
        SESSION_EVENT_QUEUES[session_id] = queue

    def event_stream():
        yield "event: ready\ndata: {}\n\n"
        while True:
            try:
                payload = queue.get(timeout=15)
                yield f"event: tool\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except Empty:
                yield "event: ping\ndata: {}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.on_event("shutdown")
def shutdown_sessions() -> None:
    for agent in CHAT_SESSIONS.values():
        agent.close()
    CHAT_SESSIONS.clear()
    SESSION_EVENT_QUEUES.clear()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "followin_mcp.demo.webapp:app",
        host=os.getenv("FOLLOWIN_WEB_HOST", "127.0.0.1"),
        port=int(os.getenv("FOLLOWIN_WEB_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
