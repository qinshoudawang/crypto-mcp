from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from importlib.util import find_spec
from typing import Any, Dict, Iterator, List

from langchain.agents import create_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..core.models import ContentItem, UserProfile
from .mcp_client import FollowinMCPClient


logger = logging.getLogger("followin_mcp.demo.agent")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[agent] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(getattr(logging, os.getenv("FOLLOWIN_AGENT_LOG_LEVEL", "INFO").upper(), logging.INFO))
logger.propagate = False


def serialize_item(item: ContentItem) -> Dict[str, Any]:
    data = asdict(item)
    data["published_at"] = item.published_at.isoformat()
    return data


def summarize_profile(user: UserProfile) -> str:
    parts: List[str] = []
    if user.interests:
        parts.append(f"interests: {', '.join(user.interests[:4])}")
    if user.followed_projects:
        parts.append(f"follows: {', '.join(user.followed_projects[:4])}")
    if user.muted_topics:
        parts.append(f"muted: {', '.join(user.muted_topics[:3])}")
    if user.risk_preference:
        parts.append(f"risk: {user.risk_preference}")
    if user.preferred_languages:
        parts.append(f"lang: {', '.join(user.preferred_languages[:2])}")
    return "; ".join(parts) or "general crypto reader"


def compact_item(item: Dict[str, Any]) -> Dict[str, Any]:
    entities = item.get("entities", {}) if isinstance(item, dict) else {}
    title = (
        item.get("title")
        or item.get("name")
        or item.get("topic_name")
        or item.get("topic")
        or item.get("content")
        or item.get("text")
        or item.get("body")
        or ""
    )
    summary = (
        item.get("summary")
        or item.get("content")
        or item.get("text")
        or item.get("body")
        or ""
    )
    return {
        "id": item.get("id", ""),
        "title": title,
        "summary": summary,
        "source_name": item.get("source_name", ""),
        "published_at": item.get("published_at", ""),
        "url": item.get("url", ""),
        "projects": list(entities.get("projects", []))[:3],
        "topics": list(entities.get("topics", []))[:3],
        "event_type": item.get("event_type", "unknown"),
    }


class HeadlinesInput(BaseModel):
    limit: int = Field(default=8, ge=1, le=20, description="Number of latest headlines to fetch.")
    last_cursor: str = Field(
        description="Opaque pagination cursor returned by the previous get_latest_headlines call. Pass it back unchanged. Use an empty string for the first page."
    )
    no_tag: bool = Field(default=False, description="Whether to request the no_tag variant.")
    only_important: bool = Field(default=False, description="Whether to request only important headlines.")


class TrendingFeedsInput(BaseModel):
    feed_type: str = Field(default="hot_news", description="Feed type, usually hot_news.")
    limit: int = Field(default=8, ge=1, le=20, description="Number of trending items to fetch.")
    cursor: str = Field(
        description="Opaque pagination cursor returned by the previous get_trending_feeds call. Pass it back unchanged. Use an empty string for the first page."
    )


class ProjectFeedInput(BaseModel):
    symbol: str = Field(description="Project or token symbol, such as BTC, ETH, JUP, or BASE.")
    feed_type: str = Field(
        default="tag_information_feed",
        description="Feed type for the project feed.",
    )
    limit: int = Field(default=8, ge=1, le=20, description="Number of project feed items to fetch.")
    cursor: str = Field(
        description="Opaque pagination cursor returned by the previous get_project_feed call. Pass it back unchanged. Use an empty string for the first page."
    )


class ProjectOpinionsInput(BaseModel):
    symbol: str = Field(description="Project or token symbol, such as BTC, ETH, or JUP.")
    limit: int = Field(default=8, ge=1, le=20, description="Number of opinion items to fetch.")
    cursor: str = Field(
        description="Opaque pagination cursor returned by the previous get_project_opinions call. Pass it back unchanged. Use an empty string for the first page."
    )


class SearchInput(BaseModel):
    query: str = Field(description="Search query for Followin content.")
    limit: int = Field(default=8, ge=1, le=20, description="Number of search results to fetch.")
    cursor: str = Field(
        description="Opaque pagination cursor returned by the previous search_content call. Pass it back unchanged. Use an empty string for the first page."
    )


class TopicsInput(BaseModel):
    limit: int = Field(default=8, ge=1, le=20, description="Number of trending topics to fetch.")
    cursor: str = Field(
        description="Opaque pagination cursor returned by the previous get_trending_topics call. Pass it back unchanged. Use an empty string for the first page."
    )


class PersonalDigestInput(BaseModel):
    max_items: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of ranked items to fetch.",
    )
    cursor: str = Field(
        description="Feed session cursor returned by the previous get_personal_feed call. Pass it back unchanged as a string. Use an empty string for the first page."
    )


class FollowinChatAgent:
    def __init__(
        self,
        mcp_client: FollowinMCPClient,
        user: UserProfile,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.mcp_client = mcp_client
        self.user = user
        self.history = InMemoryChatMessageHistory()
        self.tool_runs: List[Dict[str, Any]] = []
        self.recent_tool_state: List[Dict[str, Any]] = []
        self._current_user_message = ""

        self._normalize_proxy_env()

        llm_kwargs: Dict[str, Any] = {
            "model": model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            "api_key": api_key or os.getenv("OPENAI_API_KEY"),
        }
        if base_url or os.getenv("OPENAI_BASE_URL"):
            llm_kwargs["base_url"] = base_url or os.getenv("OPENAI_BASE_URL")
        llm_kwargs["streaming"] = True

        self.llm = ChatOpenAI(**llm_kwargs)
        self.tools = self._build_tools()
        self.agent = self._build_agent()

    def update_user(self, user: UserProfile) -> bool:
        if user == self.user:
            return False
        self.user = user
        self.recent_tool_state = []
        self.agent = self._build_agent()
        return True

    def _normalize_proxy_env(self) -> None:
        has_socks_proxy = any(
            str(os.getenv(name, "")).startswith("socks")
            for name in ("ALL_PROXY", "all_proxy")
        )
        if not has_socks_proxy:
            return

        if find_spec("socksio") is not None:
            return

        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("all_proxy", None)

    def _build_agent(self):
        system_prompt = (
            "You are Followin, a crypto research assistant inside a web chat. "
            "Hold a natural multi-turn conversation, remember prior turns, and use tools whenever "
            "you need fresh market data or project-specific evidence.\n"
            f"User context for tool choice and ranking: {summarize_profile(self.user)}.\n"
            "When you use tools, synthesize their outputs into a direct answer instead of dumping raw JSON.\n"
            "Use the user context for ranking, but keep the wording natural.\n"
            "For recommendation answers, avoid repetitive personalized recommendation templates unless the user explicitly asks for personalized reasoning.\n"
            "When the user asks for more, the next page, or another batch of the same results, reuse the most recent relevant tool and pass along its cursor if one was returned.\n"
            "Use the prior conversation and prior tool outputs deliberately in follow-up turns. "
            "When the user asks to expand on something already shown, prefer grounding on the previously returned results before calling tools again."
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
        extra: Dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "tool_name": tool_name,
            "tool_arguments": tool_arguments,
            "items": items,
        }
        if extra:
            payload.update(extra)
        self.tool_runs.append(payload)
        self._remember_tool_state(tool_name, tool_arguments, payload)
        response_payload: Dict[str, Any] = {"items": items}
        if extra:
            response_payload.update(extra)
        return json.dumps(response_payload, ensure_ascii=False)

    def _remember_tool_state(
        self,
        tool_name: str,
        tool_arguments: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> None:
        state_entry = {
            "tool_name": tool_name,
            "tool_arguments": dict(tool_arguments),
        }
        for key in ("cursor", "next_cursor", "last_cursor", "has_more", "has_next"):
            if key in payload:
                state_entry[key] = payload.get(key)

        self.recent_tool_state.append(state_entry)
        self.recent_tool_state = self.recent_tool_state[-6:]

    def _tool_state_context(self) -> str:
        if not self.recent_tool_state:
            return ""

        lines = ["Available next cursors:"]
        for state in self.recent_tool_state[-4:]:
            next_cursor = state.get("next_cursor")
            if next_cursor:
                lines.append(
                    f"- {state.get('tool_name', '')}: {json.dumps(next_cursor, ensure_ascii=False)}"
                )
        return "\n".join(lines)

    def _call_items_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        result = self.mcp_client.call_tool(tool_name, arguments)
        if isinstance(result, dict):
            result = result.get("items", [])
        if not isinstance(result, list):
            raise RuntimeError(
                f"MCP tool {tool_name} returned unexpected payload type: {type(result).__name__}"
            )
        return [item for item in result if isinstance(item, dict)]

    def _build_tools(self) -> List[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self._tool_get_latest_headlines,
                name="get_latest_headlines",
                description="Fetch the latest crypto headlines and breaking news in chronological order. For the next page, pass back the returned last_cursor unchanged.",
                args_schema=HeadlinesInput,
            ),
            StructuredTool.from_function(
                func=self._tool_get_trending_feeds,
                name="get_trending_feeds",
                description="Fetch currently trending or hot crypto news when the user asks what is popular right now. For the next page, pass back the returned cursor unchanged.",
                args_schema=TrendingFeedsInput,
            ),
            StructuredTool.from_function(
                func=self._tool_get_project_feed,
                name="get_project_feed",
                description="Fetch recent factual news updates for a specific project or token symbol. For the next page, pass back the returned cursor unchanged.",
                args_schema=ProjectFeedInput,
            ),
            StructuredTool.from_function(
                func=self._tool_get_project_opinions,
                name="get_project_opinions",
                description="Fetch opinion, commentary, or sentiment posts for a specific project or token symbol. For the next page, pass back the returned cursor unchanged.",
                args_schema=ProjectOpinionsInput,
            ),
            StructuredTool.from_function(
                func=self._tool_search_content,
                name="search_content",
                description="Search Followin content for a specific topic, project, narrative, phrase, or question term. For the next page, pass back the returned cursor unchanged.",
                args_schema=SearchInput,
            ),
            StructuredTool.from_function(
                func=self._tool_get_trending_topics,
                name="get_trending_topics",
                description="Fetch the most active crypto topics or narratives when the user asks about hot themes or narratives. For the next page, pass back the returned cursor unchanged.",
                args_schema=TopicsInput,
            ),
            StructuredTool.from_function(
                func=self._tool_get_personal_feed,
                name="get_personal_feed",
                description=(
                    "Fetch ranked personalized crypto event clusters when the user asks "
                    "for recommendations, what is worth reading, or what to watch today. "
                    "Prefer this over combining several raw retrieval tools for recommendation-style requests. "
                    "Treat ranked_clusters as the primary result and use items only as supporting raw content. "
                    "For the next batch, pass back the returned feed session cursor unchanged as a string."
                ),
                args_schema=PersonalDigestInput,
            ),
        ]

    def _tool_get_latest_headlines(
        self,
        limit: int = 8,
        last_cursor: str | None = None,
        no_tag: bool = False,
        only_important: bool = False,
    ) -> str:
        result = self.mcp_client.call_tool(
            "get_latest_headlines",
            {
                "limit": limit,
                "last_cursor": last_cursor or "",
                "no_tag": no_tag,
                "only_important": only_important,
            },
        )
        if isinstance(result, dict):
            items = result.get("items", [])
            extra = {
                key: result.get(key)
                for key in ("last_cursor", "next_cursor", "cursor", "has_more", "has_next")
                if key in result
            }
        else:
            items = result
            extra = {}
        if not isinstance(items, list):
            raise RuntimeError("MCP tool get_latest_headlines returned unexpected payload.")
        compact = [compact_item(item) for item in items[:limit]]
        return self._record_tool_run(
            "get_latest_headlines",
            {
                "limit": limit,
                "last_cursor": last_cursor or "",
                "no_tag": no_tag,
                "only_important": only_important,
            },
            compact,
            extra=extra,
        )

    def _tool_get_trending_feeds(
        self,
        feed_type: str = "hot_news",
        limit: int = 8,
        cursor: str | None = None,
    ) -> str:
        result = self.mcp_client.call_tool(
            "get_trending_feeds",
            {"feed_type": feed_type, "limit": limit, "cursor": cursor or ""},
        )
        if isinstance(result, dict):
            items = result.get("items", [])
            extra = {
                key: result.get(key)
                for key in ("last_cursor", "next_cursor", "cursor", "has_more", "has_next")
                if key in result
            }
        else:
            items = result
            extra = {}
        if not isinstance(items, list):
            raise RuntimeError("MCP tool get_trending_feeds returned unexpected payload.")
        compact = [compact_item(item) for item in items[:limit]]
        return self._record_tool_run(
            "get_trending_feeds",
            {"feed_type": feed_type, "limit": limit, "cursor": cursor or ""},
            compact,
            extra=extra,
        )

    def _tool_get_project_feed(
        self,
        symbol: str,
        feed_type: str = "tag_information_feed",
        limit: int = 8,
        cursor: str | None = None,
    ) -> str:
        result = self.mcp_client.call_tool(
            "get_project_feed",
            {"symbol": symbol, "feed_type": feed_type, "limit": limit, "cursor": cursor or ""},
        )
        if isinstance(result, dict):
            items = result.get("items", [])
            extra = {
                key: result.get(key)
                for key in ("last_cursor", "next_cursor", "cursor", "has_more", "has_next")
                if key in result
            }
        else:
            items = result
            extra = {}
        if not isinstance(items, list):
            raise RuntimeError("MCP tool get_project_feed returned unexpected payload.")
        compact = [compact_item(item) for item in items[:limit]]
        return self._record_tool_run(
            "get_project_feed",
            {"symbol": symbol, "feed_type": feed_type, "limit": limit, "cursor": cursor or ""},
            compact,
            extra=extra,
        )

    def _tool_get_project_opinions(
        self,
        symbol: str,
        limit: int = 8,
        cursor: str | None = None,
    ) -> str:
        result = self.mcp_client.call_tool(
            "get_project_opinions",
            {"symbol": symbol, "limit": limit, "cursor": cursor or ""},
        )
        if isinstance(result, dict):
            items = result.get("items", [])
            extra = {
                key: result.get(key)
                for key in ("last_cursor", "next_cursor", "cursor", "has_more", "has_next")
                if key in result
            }
        else:
            items = result
            extra = {}
        if not isinstance(items, list):
            raise RuntimeError("MCP tool get_project_opinions returned unexpected payload.")
        compact = [compact_item(item) for item in items[:limit]]
        return self._record_tool_run(
            "get_project_opinions",
            {"symbol": symbol, "limit": limit, "cursor": cursor or ""},
            compact,
            extra=extra,
        )

    def _tool_search_content(
        self,
        query: str,
        limit: int = 8,
        cursor: str | None = None,
    ) -> str:
        result = self.mcp_client.call_tool(
            "search_content",
            {"query": query, "limit": limit, "cursor": cursor or ""},
        )
        if isinstance(result, dict):
            items = result.get("items", [])
            extra = {
                key: result.get(key)
                for key in ("last_cursor", "next_cursor", "cursor", "has_more", "has_next")
                if key in result
            }
        else:
            items = result
            extra = {}
        if not isinstance(items, list):
            raise RuntimeError("MCP tool search_content returned unexpected payload.")
        compact = [compact_item(item) for item in items[:limit]]
        return self._record_tool_run(
            "search_content",
            {"query": query, "limit": limit, "cursor": cursor or ""},
            compact,
            extra=extra,
        )

    def _tool_get_trending_topics(self, limit: int = 8, cursor: str | None = None) -> str:
        topics = self.mcp_client.call_tool(
            "get_trending_topics",
            {"limit": limit, "cursor": cursor or ""},
        )
        if isinstance(topics, dict):
            extra = {
                key: topics.get(key)
                for key in ("last_cursor", "next_cursor", "cursor", "has_more", "has_next")
                if key in topics
            }
            topics = topics.get("items", [])
        else:
            extra = {}
        compact = topics[:limit] if isinstance(topics, list) else []
        return self._record_tool_run(
            "get_trending_topics",
            {"limit": limit, "cursor": cursor or ""},
            compact,
            extra=extra,
        )

    def _tool_get_personal_feed(
        self,
        max_items: int = 5,
        cursor: str | None = None,
    ) -> str:
        logger.info("[agent] get_personal_feed start: max_items=%s cursor=%s", max_items, bool(cursor))
        result = self.mcp_client.call_tool(
            "get_personal_feed",
            {
                "user": asdict(self.user),
                "max_items": max_items,
                "user_message": self._current_user_message,
                "cursor": cursor or "",
            },
        )
        logger.info("[agent] get_personal_feed result received")
        if not isinstance(result, dict):
            raise RuntimeError("MCP tool get_personal_feed returned non-dict payload.")

        top_items = [compact_item(item) for item in result.get("items", []) if isinstance(item, dict)]
        ranked_clusters = [
            cluster
            for cluster in result.get("ranked_clusters", [])
            if isinstance(cluster, dict)
        ][:max_items]
        payload = {
            "ranked_clusters": ranked_clusters,
            "items": top_items,
        }
        extra = {
            key: result.get(key)
            for key in ("last_cursor", "next_cursor", "cursor", "has_more", "has_next")
            if key in result
        }
        payload.update(extra)
        tool_payload = {
            "tool_name": "get_personal_feed",
            "tool_arguments": {
                "max_items": max_items,
                "user_message": self._current_user_message,
                "cursor": cursor or "",
            },
            "items": top_items,
            "ranked_clusters": ranked_clusters,
            **extra,
        }
        self.tool_runs.append(tool_payload)
        self._remember_tool_state(
            "get_personal_feed",
            {
                "max_items": max_items,
                "user_message": self._current_user_message,
                "cursor": cursor or "",
            },
            tool_payload,
        )
        return json.dumps(payload, ensure_ascii=False)

    def chat(self, user_message: str) -> Dict[str, Any]:
        self.tool_runs = []
        self._current_user_message = user_message
        input_messages = list(self.history.messages)
        tool_state_context = self._tool_state_context()
        if tool_state_context:
            input_messages.append(SystemMessage(content=tool_state_context))
        input_messages.append(HumanMessage(content=user_message))
        result = self.agent.invoke({"messages": input_messages})
        messages = result["messages"]

        assistant_message = ""
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                assistant_message = message.text
                break

        self.history.add_user_message(user_message)
        self.history.add_ai_message(assistant_message)
        self._current_user_message = ""

        return {
            "assistant_message": assistant_message,
            "tool_runs": self.tool_runs,
            "history_size": len(self.history.messages),
        }

    def _looks_like_structured_payload(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if stripped.startswith(("```json", '{"', "{", "[", '"items"', '"ranked_clusters"', '"tool_name"')):
            return True
        markers = (
            '"items"',
            '"ranked_clusters"',
            '"tool_name"',
            '"tool_arguments"',
            '"projects"',
            '"topics"',
            '"event_type"',
            '{"id":',
            '{"title":',
        )
        return any(marker in stripped for marker in markers)

    def _extract_chunk_text(self, message: Any, metadata: Dict[str, Any] | None = None) -> str:
        if metadata and metadata.get("langgraph_node") not in (None, "model"):
            return ""
        if not isinstance(message, AIMessage):
            return ""
        if getattr(message, "tool_calls", None) or getattr(message, "tool_call_chunks", None):
            return ""

        content = getattr(message, "content", "")
        if isinstance(content, str):
            if self._looks_like_structured_payload(content):
                return ""
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            text = "".join(parts)
            if self._looks_like_structured_payload(text):
                return ""
            return text
        return ""

    def _extract_final_assistant_message(self, messages: List[Any]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                return message.text
        return ""

    def chat_stream(self, user_message: str) -> Iterator[Dict[str, Any]]:
        self.tool_runs = []
        self._current_user_message = user_message
        input_messages = list(self.history.messages)
        tool_state_context = self._tool_state_context()
        if tool_state_context:
            input_messages.append(SystemMessage(content=tool_state_context))
        input_messages.append(HumanMessage(content=user_message))
        final_state: Dict[str, Any] | None = None
        streamed_parts: List[str] = []
        pending_parts: List[str] = []
        saw_tool_call = False
        emitted_final_phase = False
        tool_runs_seen = 0

        for mode, data in self.agent.stream(
            {"messages": input_messages},
            stream_mode=["messages", "values"],
        ):
            current_tool_runs = len(self.tool_runs)
            if current_tool_runs > tool_runs_seen:
                tool_runs_seen = current_tool_runs
                pending_parts = []
                emitted_final_phase = True

            if mode == "messages":
                message, metadata = data
                if getattr(message, "tool_calls", None) or getattr(message, "tool_call_chunks", None):
                    saw_tool_call = True
                    pending_parts = []
                    continue

                delta = self._extract_chunk_text(message, metadata)
                if delta:
                    if saw_tool_call and not emitted_final_phase:
                        pending_parts.append(delta)
                        continue

                    streamed_parts.append(delta)
                    yield {"type": "assistant_chunk", "delta": delta}
            elif mode == "values" and isinstance(data, dict):
                final_state = data

        assistant_message = ""
        if final_state is not None:
            assistant_message = self._extract_final_assistant_message(final_state.get("messages", []))
        if not assistant_message:
            assistant_message = "".join(streamed_parts).strip()

        self.history.add_user_message(user_message)
        self.history.add_ai_message(assistant_message)
        self._current_user_message = ""

        yield {
            "type": "done",
            "assistant_message": assistant_message,
            "tool_runs": self.tool_runs,
            "history_size": len(self.history.messages),
        }

    def close(self) -> None:
        self.mcp_client.close()
