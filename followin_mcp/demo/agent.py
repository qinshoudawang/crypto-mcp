from __future__ import annotations

import json
import os
from dataclasses import asdict
from importlib.util import find_spec
from typing import Any, Dict, List

from langchain.agents import create_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..core.models import ContentItem, UserProfile
from ..core.service import FollowinMCPService


def serialize_item(item: ContentItem) -> Dict[str, Any]:
    data = asdict(item)
    data["published_at"] = item.published_at.isoformat()
    return data


def summarize_profile(user: UserProfile) -> str:
    interests = ", ".join(user.interests) or "general market"
    followed = ", ".join(user.followed_projects) or "none"
    muted = ", ".join(user.muted_topics) or "none"
    return (
        f"user_id={user.user_id}; "
        f"interests={interests}; "
        f"followed_projects={followed}; "
        f"muted_topics={muted}; "
        f"risk_preference={user.risk_preference}; "
        f"preferred_languages={', '.join(user.preferred_languages) or 'zh'}"
    )


def compact_item(item: ContentItem) -> Dict[str, Any]:
    return {
        "id": item.id,
        "title": item.title,
        "summary": item.summary,
        "source_name": item.source_name,
        "published_at": item.published_at.isoformat(),
        "url": item.url,
        "projects": item.entities.projects[:3],
        "topics": item.entities.topics[:3],
        "event_type": item.event_type,
    }


class HeadlinesInput(BaseModel):
    limit: int = Field(default=8, ge=1, le=20, description="Number of latest headlines to fetch.")


class TrendingFeedsInput(BaseModel):
    feed_type: str = Field(default="hot_news", description="Feed type, usually hot_news.")
    limit: int = Field(default=8, ge=1, le=20, description="Number of trending items to fetch.")


class ProjectFeedInput(BaseModel):
    symbol: str = Field(description="Project or token symbol, such as BTC, ETH, JUP, or BASE.")
    feed_type: str = Field(
        default="tag_information_feed",
        description="Feed type for the project feed.",
    )
    limit: int = Field(default=8, ge=1, le=20, description="Number of project feed items to fetch.")


class ProjectOpinionsInput(BaseModel):
    symbol: str = Field(description="Project or token symbol, such as BTC, ETH, or JUP.")
    limit: int = Field(default=8, ge=1, le=20, description="Number of opinion items to fetch.")


class SearchInput(BaseModel):
    query: str = Field(description="Search query for Followin content.")
    limit: int = Field(default=8, ge=1, le=20, description="Number of search results to fetch.")


class TopicsInput(BaseModel):
    limit: int = Field(default=8, ge=1, le=20, description="Number of trending topics to fetch.")


class PersonalDigestInput(BaseModel):
    max_items: int = Field(default=5, ge=1, le=10, description="Maximum number of items in the digest.")


class FollowinChatAgent:
    def __init__(
        self,
        service: FollowinMCPService,
        user: UserProfile,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.service = service
        self.user = user
        self.history = InMemoryChatMessageHistory()
        self.tool_runs: List[Dict[str, Any]] = []

        self._normalize_proxy_env()

        llm_kwargs: Dict[str, Any] = {
            "model": model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            "api_key": api_key or os.getenv("OPENAI_API_KEY"),
        }
        if base_url or os.getenv("OPENAI_BASE_URL"):
            llm_kwargs["base_url"] = base_url or os.getenv("OPENAI_BASE_URL")

        self.llm = ChatOpenAI(**llm_kwargs)
        self.tools = self._build_tools()
        self.agent = self._build_agent()

    def _normalize_proxy_env(self) -> None:
        has_socks_proxy = any(
            str(os.getenv(name, "")).startswith("socks")
            for name in ("ALL_PROXY", "all_proxy")
        )
        if not has_socks_proxy:
            return

        if find_spec("socksio") is not None:
            return

        # Fall back to HTTP(S) proxy settings when SOCKS support is unavailable.
        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("all_proxy", None)

    def _build_agent(self):
        system_prompt = (
            "You are Followin, a crypto research assistant inside a web chat. "
            "Hold a natural multi-turn conversation, remember prior turns, and use tools whenever "
            "you need fresh market data or project-specific evidence.\n"
            f"Current user profile:\n{summarize_profile(self.user)}\n"
            "When you use tools, synthesize their outputs into a direct answer instead of dumping raw JSON."
        )
        return create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            debug=False,
            name="followin_chat_agent",
        )

    def _record_tool_run(
        self,
        tool_name: str,
        tool_arguments: Dict[str, Any],
        items: List[Dict[str, Any]],
    ) -> str:
        payload = {
            "tool_name": tool_name,
            "tool_arguments": tool_arguments,
            "items": items,
        }
        self.tool_runs.append(payload)
        return json.dumps({"items": items}, ensure_ascii=False)

    def _build_tools(self) -> List[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self._tool_get_latest_headlines,
                name="get_latest_headlines",
                description="Fetch the latest crypto headlines from Followin.",
                args_schema=HeadlinesInput,
            ),
            StructuredTool.from_function(
                func=self._tool_get_trending_feeds,
                name="get_trending_feeds",
                description="Fetch currently trending crypto feeds or hot news.",
                args_schema=TrendingFeedsInput,
            ),
            StructuredTool.from_function(
                func=self._tool_get_project_feed,
                name="get_project_feed",
                description="Fetch recent news for a specific project or token symbol.",
                args_schema=ProjectFeedInput,
            ),
            StructuredTool.from_function(
                func=self._tool_get_project_opinions,
                name="get_project_opinions",
                description="Fetch opinions or sentiment posts for a specific project or token symbol.",
                args_schema=ProjectOpinionsInput,
            ),
            StructuredTool.from_function(
                func=self._tool_search_content,
                name="search_content",
                description="Search Followin content for a topic, project, narrative, or phrase.",
                args_schema=SearchInput,
            ),
            StructuredTool.from_function(
                func=self._tool_get_trending_topics,
                name="get_trending_topics",
                description="Fetch the most active crypto topics or narratives right now.",
                args_schema=TopicsInput,
            ),
            StructuredTool.from_function(
                func=self._tool_get_personal_digest,
                name="get_personal_digest",
                description=(
                    "Build a personalized digest for the current user profile when the user asks "
                    "what is worth reading or wants recommendations."
                ),
                args_schema=PersonalDigestInput,
            ),
        ]

    def _tool_get_latest_headlines(self, limit: int = 8) -> str:
        items = self.service.get_latest_headlines(limit=limit)
        compact = [compact_item(item) for item in items[:limit]]
        return self._record_tool_run("get_latest_headlines", {"limit": limit}, compact)

    def _tool_get_trending_feeds(self, feed_type: str = "hot_news", limit: int = 8) -> str:
        items = self.service.get_trending_feeds(feed_type=feed_type, limit=limit)
        compact = [compact_item(item) for item in items[:limit]]
        return self._record_tool_run(
            "get_trending_feeds",
            {"feed_type": feed_type, "limit": limit},
            compact,
        )

    def _tool_get_project_feed(
        self,
        symbol: str,
        feed_type: str = "tag_information_feed",
        limit: int = 8,
    ) -> str:
        items = self.service.get_project_feed(symbol=symbol, feed_type=feed_type, limit=limit)
        compact = [compact_item(item) for item in items[:limit]]
        return self._record_tool_run(
            "get_project_feed",
            {"symbol": symbol, "feed_type": feed_type, "limit": limit},
            compact,
        )

    def _tool_get_project_opinions(self, symbol: str, limit: int = 8) -> str:
        items = self.service.get_project_opinions(symbol=symbol, limit=limit)
        compact = [compact_item(item) for item in items[:limit]]
        return self._record_tool_run(
            "get_project_opinions",
            {"symbol": symbol, "limit": limit},
            compact,
        )

    def _tool_search_content(self, query: str, limit: int = 8) -> str:
        items = self.service.search_content(query=query, limit=limit)
        compact = [compact_item(item) for item in items[:limit]]
        return self._record_tool_run("search_content", {"query": query, "limit": limit}, compact)

    def _tool_get_trending_topics(self, limit: int = 8) -> str:
        topics = self.service.get_trending_topics(limit=limit)
        compact = topics[:limit]
        return self._record_tool_run("get_trending_topics", {"limit": limit}, compact)

    def _tool_get_personal_digest(self, max_items: int = 5) -> str:
        candidates = self.service.recall_candidates_for_user(self.user)
        digest = self.service.build_personal_digest(self.user, candidates, max_items=max_items)
        top_items = [compact_item(item) for item in candidates[:max_items]]
        payload = {
            "digest": digest,
            "items": top_items,
        }
        self.tool_runs.append(
            {
                "tool_name": "get_personal_digest",
                "tool_arguments": {"max_items": max_items},
                "items": top_items,
            }
        )
        return json.dumps(payload, ensure_ascii=False)

    def chat(self, user_message: str) -> Dict[str, Any]:
        self.tool_runs = []
        input_messages = [*self.history.messages, HumanMessage(content=user_message)]
        result = self.agent.invoke({"messages": input_messages})
        messages = result["messages"]

        assistant_message = ""
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                assistant_message = message.text
                break

        self.history.add_user_message(user_message)
        self.history.add_ai_message(assistant_message)

        return {
            "assistant_message": assistant_message,
            "tool_runs": self.tool_runs,
            "history_size": len(self.history.messages),
        }
