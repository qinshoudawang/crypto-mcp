from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from ..core.adapters import FollowinAPIAdapter
from ..core.models import ContentItem
from ..core.service import FollowinMCPService


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


mcp = FastMCP("followin-mcp")


@mcp.tool()
def get_latest_headlines(limit: int = 20) -> List[Dict[str, Any]]:
    """Get the latest normalized crypto headlines and breaking news items."""
    return _items_to_dicts(_build_service().get_latest_headlines(limit=limit))


@mcp.tool()
def get_trending_feeds(feed_type: str = "hot_news", limit: int = 20) -> List[Dict[str, Any]]:
    """Get trending normalized feeds for the specified trending feed type."""
    return _items_to_dicts(_build_service().get_trending_feeds(feed_type=feed_type, limit=limit))


@mcp.tool()
def get_project_feed(
    symbol: str,
    feed_type: str = "tag_information_feed",
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Get normalized feeds for a specific project or token symbol."""
    return _items_to_dicts(
        _build_service().get_project_feed(symbol=symbol, feed_type=feed_type, limit=limit)
    )


@mcp.tool()
def get_project_opinions(symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Get opinion-oriented normalized content for a specific project or token symbol."""
    return _items_to_dicts(_build_service().get_project_opinions(symbol=symbol, limit=limit))


@mcp.tool()
def get_trending_topics(limit: int = 10) -> List[Dict[str, Any]]:
    """Get current trending crypto topics and narrative ranking data."""
    return _build_service().get_trending_topics(limit=limit)


@mcp.tool()
def search_content(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search normalized content by keyword or project/topic query."""
    return _items_to_dicts(_build_service().search_content(query=query, limit=limit))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
