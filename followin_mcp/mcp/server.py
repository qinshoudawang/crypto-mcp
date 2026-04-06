from __future__ import annotations

import os
import logging
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Dict, List

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from ..core.adapters import FollowinAPIAdapter
from ..core.models import ContentItem, UserProfile
from ..core.service import FollowinMCPService


logger = logging.getLogger("followin_mcp.mcp.server")


def _item_to_dict(item: ContentItem) -> Dict[str, Any]:
    return {
        "id": item.id,
        "title": item.title,
        "summary": item.summary,
        "content": item.content,
        "url": item.url,
        "source_type": item.source_type,
        "source_name": item.source_name,
        "author": item.author,
        "published_at": item.published_at.isoformat(),
        "language": item.language,
        "entities": {
            "projects": item.entities.projects,
            "tokens": item.entities.tokens,
            "chains": item.entities.chains,
            "topics": item.entities.topics,
            "people": item.entities.people,
        },
        "event_type": item.event_type,
        "importance_score": item.importance_score,
        "credibility_score": item.credibility_score,
        "engagement_score": item.engagement_score,
        "raw_tags": item.raw_tags,
        "entity_sources": item.entity_sources,
        "entity_confidence": item.entity_confidence,
    }


def _items_to_dicts(items: List[ContentItem]) -> List[Dict[str, Any]]:
    return [_item_to_dict(item) for item in items]


def _page_meta(payload: Dict[str, Any], exclude: set[str] | None = None) -> Dict[str, Any]:
    excluded = exclude or set()
    meta: Dict[str, Any] = {}
    defaults: Dict[str, Any] = {
        "cursor": "",
        "next_cursor": "",
        "last_cursor": "",
        "has_more": False,
        "has_next": False,
    }
    for key, default in defaults.items():
        if key in excluded:
            continue
        value = payload.get(key, default)
        if key in {"has_more", "has_next"}:
            meta[key] = bool(value)
        elif value is None:
            meta[key] = default
        else:
            meta[key] = value
    return meta


@lru_cache(maxsize=1)
def _build_service() -> FollowinMCPService:
    load_dotenv()
    api_key = os.getenv("FOLLOWIN_API_KEY")
    if not api_key:
        raise RuntimeError("FOLLOWIN_API_KEY is required to start the MCP server.")

    adapter = FollowinAPIAdapter(
        api_key=api_key,
        lang=os.getenv("FOLLOWIN_LANG", "zh-Hans"),
        timeout=int(os.getenv("FOLLOWIN_TIMEOUT", "15")),
    )
    return FollowinMCPService(adapter)


mcp = FastMCP(
    "followin-mcp",
    host=os.getenv("FOLLOWIN_MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("FOLLOWIN_MCP_PORT", "8001")),
    streamable_http_path=os.getenv("FOLLOWIN_MCP_PATH", "/mcp"),
)


@mcp.tool()
def get_latest_headlines(
    limit: int = 20,
    last_cursor: str | None = None,
    no_tag: bool = False,
    only_important: bool = False,
) -> Dict[str, Any]:
    """Get the latest normalized crypto headlines and breaking news items in chronological order."""
    payload = _build_service().get_latest_headlines(
        limit=limit,
        last_cursor=last_cursor,
        no_tag=no_tag,
        only_important=only_important,
    )
    response = {"items": _items_to_dicts(payload["items"])}
    response.update(_page_meta(payload, exclude={"items"}))
    return response


@mcp.tool()
def get_trending_feeds(
    feed_type: str = "hot_news",
    limit: int = 20,
) -> Dict[str, Any]:
    """Get trending normalized feeds for hot or popular crypto news."""
    payload = _build_service().get_trending_feeds(
        feed_type=feed_type,
        limit=limit,
    )
    return {"items": _items_to_dicts(payload["items"])}


@mcp.tool()
def get_project_feed(
    symbol: str,
    feed_type: str = "tag_information_feed",
    limit: int = 20,
    cursor: str | None = None,
) -> Dict[str, Any]:
    """Get recent normalized factual news for a specific project or token symbol."""
    payload = _build_service().get_project_feed(
        symbol=symbol,
        feed_type=feed_type,
        limit=limit,
        cursor=cursor,
    )
    response = {"items": _items_to_dicts(payload["items"])}
    response.update(_page_meta(payload, exclude={"items"}))
    return response


@mcp.tool()
def get_project_opinions(
    symbol: str,
    limit: int = 20,
    cursor: str | None = None,
) -> Dict[str, Any]:
    """Get opinion-oriented normalized content for a specific project or token symbol."""
    payload = _build_service().get_project_opinions(
        symbol=symbol,
        limit=limit,
        cursor=cursor,
    )
    response = {"items": _items_to_dicts(payload["items"])}
    response.update(_page_meta(payload, exclude={"items"}))
    return response


@mcp.tool()
def get_trending_topics(limit: int = 10, cursor: str | None = None) -> Dict[str, Any]:
    """Get current trending crypto topics and narrative ranking data."""
    payload = _build_service().get_trending_topics(limit=limit, cursor=cursor)
    response = {"items": payload["items"]}
    response.update(_page_meta(payload, exclude={"items"}))
    return response


@mcp.tool()
def search_content(
    query: str,
    limit: int = 20,
) -> Dict[str, Any]:
    """Search normalized content by keyword, project, topic, narrative, or phrase query."""
    payload = _build_service().search_content(
        query=query,
        limit=limit,
    )
    return {"items": _items_to_dicts(payload["items"])}


@mcp.tool()
def get_personal_feed(
    user: Dict[str, Any],
    max_items: int = 5,
    user_message: str = "",
    cursor: str | None = None,
) -> Dict[str, Any]:
    """Build a personalized crypto feed and continue it with a feed session cursor."""
    logger.info("[mcp] get_personal_feed start: max_items=%s cursor=%s", max_items, bool(cursor))
    profile = UserProfile(**user)
    payload = _build_service().build_personal_feed_payload(
        user=profile,
        query=user_message or None,
        max_items=max_items,
        cursor=cursor,
    )
    response = {
        "items": _items_to_dicts(payload["items"]),
        "ranked_clusters": [
            {
                "event_id": cluster.event_id,
                "title": cluster.title,
                "event_type": cluster.event_type,
                "importance_score": cluster.importance_score,
                "projects": cluster.entities.projects,
                "tokens": cluster.entities.tokens,
                "chains": cluster.entities.chains,
                "topics": cluster.entities.topics,
            }
            for cluster in payload["ranked_clusters"]
        ],
        "user": asdict(profile),
    }
    if "next_cursor" in payload:
        response["next_cursor"] = payload["next_cursor"]
    response["has_more"] = bool(payload.get("has_more"))
    logger.info(
        "[mcp] get_personal_feed return: clusters=%s items=%s has_more=%s",
        len(response["ranked_clusters"]),
        len(response["items"]),
        response["has_more"],
    )
    return response


def main() -> None:
    transport = os.getenv("FOLLOWIN_MCP_TRANSPORT", "streamable-http")
    _build_service()
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
