from __future__ import annotations

import os
import random
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..core.adapters import FollowinAPIAdapter
from ..core.models import UserProfile
from ..core.service import FollowinMCPService
from .agent import FollowinChatAgent


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


@lru_cache(maxsize=1)
def _build_service() -> FollowinMCPService:
    load_dotenv()
    api_key = os.getenv("FOLLOWIN_API_KEY")
    if not api_key:
        raise RuntimeError("FOLLOWIN_API_KEY is required to start the web demo.")

    adapter = FollowinAPIAdapter(
        api_key=api_key,
        lang=os.getenv("FOLLOWIN_LANG", "zh-Hans"),
        timeout=int(os.getenv("FOLLOWIN_TIMEOUT", "15")),
    )
    return FollowinMCPService(adapter)


def _build_agent(user: UserProfile) -> FollowinChatAgent:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to start the LangChain web demo.")

    return FollowinChatAgent(
        service=_build_service(),
        user=user,
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/profile/random")
def random_profile() -> Dict[str, Any]:
    preset = random.choice(PROFILE_PRESETS)
    return ProfilePayload(**preset).model_dump()


@app.post("/api/session")
def create_session(request: SessionRequest) -> Dict[str, Any]:
    user = request.profile.to_user_profile()
    session_id = uuid.uuid4().hex
    CHAT_SESSIONS[session_id] = _build_agent(user)
    return {"session_id": session_id}


@app.post("/api/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    agent = CHAT_SESSIONS.get(request.session_id)
    if agent is None:
        user = request.profile.to_user_profile()
        agent = _build_agent(user)
        CHAT_SESSIONS[request.session_id] = agent

    result = agent.chat(request.user_message)
    return {
        "session_id": request.session_id,
        "assistant_message": result["assistant_message"],
        "tool_runs": result["tool_runs"],
        "history_size": result["history_size"],
    }


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
