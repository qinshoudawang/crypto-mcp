from __future__ import annotations

from typing import Any, Dict, List, Set

from .adapters import FollowinSourceAdapter
from .clustering import EventClusterer
from .digest import DigestBuilder
from .models import ContentItem, EventCluster, UserProfile
from .normalizer import ContentNormalizer
from .ranking import UserRanker


class FollowinMCPService:
    def __init__(self, adapter: FollowinSourceAdapter):
        self.adapter = adapter
        self.normalizer = ContentNormalizer()
        self.clusterer = EventClusterer()
        self.ranker = UserRanker()
        self.digest_builder = DigestBuilder()

    # Read APIs

    def get_latest_headlines(self, limit: int = 20) -> List[ContentItem]:
        return [
            self.normalizer.normalize(item)
            for item in self.adapter.get_latest_headlines(limit=limit)
        ]

    def get_trending_feeds(self, feed_type: str = "hot_news", limit: int = 20) -> List[ContentItem]:
        return [
            self.normalizer.normalize(item)
            for item in self.adapter.get_trending_feeds(feed_type=feed_type, limit=limit)
        ]

    def get_project_feed(
        self,
        symbol: str,
        feed_type: str = "tag_information_feed",
        limit: int = 20,
    ) -> List[ContentItem]:
        raw_items = self.adapter.get_project_feed(
            symbol=symbol,
            feed_type=feed_type,
            limit=limit,
        )
        return [self.normalizer.normalize(item) for item in raw_items]

    def get_project_opinions(self, symbol: str, limit: int = 20) -> List[ContentItem]:
        return [
            self.normalizer.normalize(item)
            for item in self.adapter.get_project_opinions(symbol=symbol, limit=limit)
        ]

    def search_content(self, query: str, limit: int = 20) -> List[ContentItem]:
        return [
            self.normalizer.normalize(item)
            for item in self.adapter.search_content(query=query, limit=limit)
        ]

    def get_trending_topics(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.adapter.get_trending_topics(limit=limit)

    # Ranking helpers

    def cluster_same_event(self, items: List[ContentItem]) -> List[EventCluster]:
        return self.clusterer.cluster_same_event(items)

    def recall_candidates_for_user(
        self,
        user: UserProfile,
        latest_limit: int = 10,
        trending_limit: int = 10,
        per_interest_limit: int = 5,
        per_project_limit: int = 5,
    ) -> List[ContentItem]:
        candidates: List[ContentItem] = []
        candidates.extend(self.get_latest_headlines(limit=latest_limit))
        candidates.extend(self.get_trending_feeds(feed_type="hot_news", limit=trending_limit))

        for project in user.followed_projects:
            try:
                candidates.extend(self.get_project_feed(symbol=project, limit=per_project_limit))
            except Exception:
                continue

        for interest in user.interests:
            try:
                candidates.extend(self.search_content(query=interest, limit=per_interest_limit))
            except Exception:
                continue

        deduped: List[ContentItem] = []
        seen_ids: Set[str] = set()
        for item in candidates:
            if item.id in seen_ids:
                continue
            seen_ids.add(item.id)
            deduped.append(item)

        deduped.sort(key=lambda item: item.published_at, reverse=True)
        return deduped

    # User-facing outputs

    def build_personal_digest(
        self,
        user: UserProfile,
        items: List[ContentItem],
        max_items: int = 5,
    ) -> str:
        clusters = self.cluster_same_event(items)
        ranked_clusters = self.ranker.rank_for_user(user, clusters)
        return self.digest_builder.build_personal_digest(
            user=user,
            clusters=ranked_clusters,
            max_items=max_items,
        )
