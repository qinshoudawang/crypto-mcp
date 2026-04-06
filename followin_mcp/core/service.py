from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Sequence, Set, Tuple, TypeVar

from .adapters import FollowinSourceAdapter
from .clustering import EventClusterer
from .models import ContentItem, EventCluster, UserProfile
from .normalizer import ContentNormalizer
from .ranking import UserRanker
from .semantic_recall import SemanticRecallEngine

logger = logging.getLogger("followin_mcp.semantic_service")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[semantic-service] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(
    getattr(logging, os.getenv("FOLLOWIN_SEMANTIC_LOG_LEVEL", "INFO").upper(), logging.INFO)
)
logger.propagate = False

T = TypeVar("T")


@dataclass
class FeedSessionState:
    session_id: str
    user_signature: str
    query: str | None
    source_cursors: Dict[str, str] = field(default_factory=dict)
    delivered_event_ids: Set[str] = field(default_factory=set)
    delivered_item_ids: Set[str] = field(default_factory=set)
    pending_clusters: List[EventCluster] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class FollowinMCPService:
    def __init__(self, adapter: FollowinSourceAdapter):
        self.adapter = adapter
        self.normalizer = ContentNormalizer()
        self.semantic_recall = SemanticRecallEngine.from_env()
        # Clustering reuses the same semantic index so recall and clustering do
        # not need to compute or store separate embeddings.
        self.clusterer = EventClusterer(
            semantic_similarity_fn=self.semantic_recall.similarity_between_items
        )
        self.ranker = UserRanker()
        self._semantic_warmup_thread: threading.Thread | None = None
        self._feed_sessions: Dict[str, FeedSessionState] = {}
        self._feed_session_lock = threading.Lock()
        self._feed_session_ttl_seconds = float(
            os.getenv("FOLLOWIN_FEED_SESSION_TTL_SECONDS", "1800")
        )
        self._feed_snapshot_buffer_size = int(
            os.getenv("FOLLOWIN_FEED_SNAPSHOT_BUFFER_SIZE", "20")
        )
        self._start_semantic_warmup()

    # Read APIs

    def get_latest_headlines(
        self,
        limit: int = 20,
        last_cursor: str | None = None,
        no_tag: bool = False,
        only_important: bool = False,
    ) -> Dict[str, Any]:
        payload = self.adapter.get_latest_headlines_page(
            limit=limit,
            last_cursor=last_cursor,
            no_tag=no_tag,
            only_important=only_important,
        )
        items = [self.normalizer.normalize(item) for item in payload.get("items", [])]
        self.semantic_recall.enqueue_items(items)
        response: Dict[str, Any] = {"items": items}
        for key in ("cursor", "next_cursor", "last_cursor", "has_more", "has_next"):
            if key in payload:
                response[key] = payload[key]
        return response

    def get_trending_feeds(
        self,
        feed_type: str = "hot_news",
        limit: int = 20,
        cursor: str | None = None,
    ) -> Dict[str, Any]:
        # Followin does not expose cursor metadata for trending feeds, so we
        # simulate stable pagination over the current snapshot with an offset cursor.
        fetch_limit = max(limit * 3, 30)
        items = [
            self.normalizer.normalize(item)
            for item in self.adapter.get_trending_feeds(feed_type=feed_type, limit=fetch_limit)
        ]
        self.semantic_recall.enqueue_items(items)
        return self._paginate_sequence(items, limit=limit, cursor=cursor)

    def get_project_feed(
        self,
        symbol: str,
        feed_type: str = "tag_information_feed",
        limit: int = 20,
        cursor: str | None = None,
    ) -> Dict[str, Any]:
        payload = self.adapter.get_project_feed_page(
            symbol=symbol,
            feed_type=feed_type,
            limit=limit,
            cursor=cursor,
        )
        items = [self.normalizer.normalize(item) for item in payload.get("items", [])]
        self.semantic_recall.enqueue_items(items)
        response: Dict[str, Any] = {"items": items}
        for key in ("cursor", "next_cursor", "last_cursor", "has_more", "has_next"):
            if key in payload:
                response[key] = payload[key]
        return response

    def get_project_opinions(
        self,
        symbol: str,
        limit: int = 20,
        cursor: str | None = None,
    ) -> Dict[str, Any]:
        payload = self.adapter.get_project_opinions_page(
            symbol=symbol,
            limit=limit,
            cursor=cursor,
        )
        items = [self.normalizer.normalize(item) for item in payload.get("items", [])]
        self.semantic_recall.enqueue_items(items)
        response: Dict[str, Any] = {"items": items}
        for key in ("cursor", "next_cursor", "last_cursor", "has_more", "has_next"):
            if key in payload:
                response[key] = payload[key]
        return response

    def search_content(
        self,
        query: str,
        limit: int = 20,
        cursor: str | None = None,
    ) -> Dict[str, Any]:
        # Search currently runs over a fresh in-memory snapshot, so pagination is
        # also offset-based until the upstream API exposes a real cursor.
        fetch_limit = max(limit * 3, 30)
        items = [
            self.normalizer.normalize(item)
            for item in self.adapter.search_content(query=query, limit=fetch_limit)
        ]
        self.semantic_recall.enqueue_items(items)
        return self._paginate_sequence(items, limit=limit, cursor=cursor)

    def get_trending_topics(
        self,
        limit: int = 10,
        cursor: str | None = None,
    ) -> Dict[str, Any]:
        return self.adapter.get_trending_topics_page(limit=limit, cursor=cursor)

    # Feed APIs

    def build_personal_feed_payload(
        self,
        user: UserProfile,
        query: str | None = None,
        max_items: int = 5,
        cursor: str | None = None,
    ) -> Dict[str, Any]:
        # This is the public feed façade used by MCP tools. Keep it thin and let
        # the private pipeline own the internal stage orchestration.
        session = self._get_or_create_feed_session(user=user, query=query, cursor=cursor)
        effective_query = session.query if cursor and session.query is not None else query
        logger.info(
            "[feed] request: session_id=%s incoming_cursor=%s effective_query=%r max_items=%s pending_before=%s delivered_events=%s",
            session.session_id,
            cursor or "",
            (effective_query or "").strip()[:80],
            max_items,
            len(session.pending_clusters),
            len(session.delivered_event_ids),
        )
        has_more_sources = self._fill_feed_snapshot_buffer(
            user=user,
            query=effective_query,
            session=session,
            min_buffer_size=max(max_items, self._feed_snapshot_buffer_size),
        )

        page_clusters: List[EventCluster] = []
        with self._feed_session_lock:
            while session.pending_clusters and len(page_clusters) < max_items:
                page_clusters.append(session.pending_clusters.pop(0))

        ranked_items = [item for cluster in page_clusters for item in cluster.items]
        with self._feed_session_lock:
            session.delivered_event_ids.update(cluster.event_id for cluster in page_clusters)
            session.delivered_item_ids.update(item.id for item in ranked_items)
            session.updated_at = time.time()
            has_more = bool(session.pending_clusters) or has_more_sources
            pending_after = len(session.pending_clusters)
        logger.info(
            "[feed] page emitted: session_id=%s clusters=%s items=%s pending_after=%s has_more=%s",
            session.session_id,
            len(page_clusters),
            len(ranked_items),
            pending_after,
            has_more,
        )
        return {
            "items": ranked_items,
            "ranked_clusters": page_clusters,
            "has_more": has_more,
            **({"next_cursor": session.session_id} if has_more else {}),
        }

    def _fill_feed_snapshot_buffer(
        self,
        user: UserProfile,
        query: str | None,
        session: FeedSessionState,
        min_buffer_size: int,
    ) -> bool:
        has_more_sources = False
        remaining_slots = max(1, min_buffer_size)
        logger.info(
            "[feed] buffer fill start: session_id=%s target=%s pending=%s",
            session.session_id,
            min_buffer_size,
            len(session.pending_clusters),
        )

        while True:
            with self._feed_session_lock:
                pending_count = len(session.pending_clusters)
            if pending_count >= min_buffer_size:
                logger.info(
                    "[feed] buffer fill satisfied: session_id=%s pending=%s",
                    session.session_id,
                    pending_count,
                )
                return True

            ranked_clusters, source_has_more = self._build_ranked_feed(
                user=user,
                query=query,
                session_state=session,
                latest_limit=max(10, remaining_slots),
                trending_limit=max(10, remaining_slots),
                per_interest_limit=max(5, min(remaining_slots, 10)),
                per_project_limit=max(5, min(remaining_slots, 10)),
                semantic_pool_limit=max(24, remaining_slots * 2),
                semantic_top_k=max(6, min(remaining_slots, 12)),
            )
            has_more_sources = has_more_sources or source_has_more
            logger.info(
                "[feed] buffer refill: session_id=%s fetched_clusters=%s source_has_more=%s",
                session.session_id,
                len(ranked_clusters),
                source_has_more,
            )
            if not ranked_clusters:
                return has_more_sources

            with self._feed_session_lock:
                session.pending_clusters.extend(ranked_clusters)
                session.updated_at = time.time()
                pending_count = len(session.pending_clusters)

            if pending_count >= min_buffer_size or not source_has_more:
                return pending_count > 0 or has_more_sources

            remaining_slots = max(1, min_buffer_size - pending_count)

    # Internal feed pipeline

    def _build_ranked_feed(
        self,
        user: UserProfile,
        query: str | None = None,
        session_state: FeedSessionState | None = None,
        latest_limit: int = 10,
        trending_limit: int = 10,
        per_interest_limit: int = 5,
        per_project_limit: int = 5,
        semantic_pool_limit: int = 24,
        semantic_top_k: int = 6,
    ) -> Tuple[List[EventCluster], bool]:
        # Feed construction stays stage-based: recall item candidates first,
        # then cluster them into events, then rank at the event level.
        source_cursors = session_state.source_cursors if session_state is not None else {}
        candidates, next_source_cursors = self._recall_candidates_for_user(
            user=user,
            query=query,
            source_cursors=source_cursors,
            latest_limit=latest_limit,
            trending_limit=trending_limit,
            per_interest_limit=per_interest_limit,
            per_project_limit=per_project_limit,
            semantic_pool_limit=semantic_pool_limit,
            semantic_top_k=semantic_top_k,
        )
        fresh_candidates = candidates
        if session_state is not None:
            pending_event_ids = {cluster.event_id for cluster in session_state.pending_clusters}
            fresh_candidates = [
                item for item in candidates if item.id not in session_state.delivered_item_ids
            ]
            with self._feed_session_lock:
                session_state.source_cursors = dict(next_source_cursors)
                session_state.updated_at = time.time()

        ranked_clusters = self._rank_clusters_for_user(user=user, items=fresh_candidates)
        if session_state is not None:
            ranked_clusters = [
                cluster
                for cluster in ranked_clusters
                if cluster.event_id not in session_state.delivered_event_ids
                and cluster.event_id not in pending_event_ids
            ]
        return ranked_clusters, bool(next_source_cursors)

    def _recall_candidates_for_user(
        self,
        user: UserProfile,
        query: str | None = None,
        source_cursors: Dict[str, str] | None = None,
        latest_limit: int = 10,
        trending_limit: int = 10,
        per_interest_limit: int = 5,
        per_project_limit: int = 5,
        semantic_pool_limit: int = 24,
        semantic_top_k: int = 6,
    ) -> Tuple[List[ContentItem], Dict[str, str]]:
        # Mix explicit retrieval paths first so semantic recall expands on the
        # same fresh candidate pool instead of searching in isolation.
        cursor_state = source_cursors or {}
        candidates: List[ContentItem] = []
        latest_page = self.get_latest_headlines(
            limit=latest_limit,
            last_cursor=cursor_state.get("latest") or None,
        )
        candidates.extend(latest_page["items"])
        trending_page = self.get_trending_feeds(
            feed_type="hot_news",
            limit=trending_limit,
            cursor=cursor_state.get("trending") or None,
        )
        candidates.extend(trending_page["items"])

        for project in user.followed_projects:
            try:
                candidates.extend(
                    self.get_project_feed(
                        symbol=project,
                        limit=per_project_limit,
                        cursor=cursor_state.get("project") or None,
                    )["items"]
                )
            except Exception:
                continue

        for interest in user.interests:
            try:
                candidates.extend(
                    self.search_content(
                        query=interest,
                        limit=per_interest_limit,
                        cursor=cursor_state.get("search") or None,
                    )["items"]
                )
            except Exception:
                continue

        candidates.extend(
            self._recall_semantic_candidates(
                user=user,
                query=query,
                base_items=candidates,
                semantic_pool_limit=semantic_pool_limit,
                semantic_top_k=semantic_top_k,
            )
        )

        # When the same item is recalled multiple times, keep the variant with the
        # stronger semantic signal instead of blindly preserving first occurrence.
        best_by_id: Dict[str, ContentItem] = {}
        for item in candidates:
            existing = best_by_id.get(item.id)
            if existing is None or self._should_replace_candidate(existing, item):
                best_by_id[item.id] = item

        deduped = list(best_by_id.values())

        deduped.sort(key=lambda item: item.published_at, reverse=True)
        next_source_cursors = {
            key: value
            for key, value in {
                "latest": latest_page.get("next_cursor"),
                "trending": trending_page.get("next_cursor"),
                "project": cursor_state.get("project"),
                "search": cursor_state.get("search"),
            }.items()
            if value
        }
        return deduped, next_source_cursors

    def _recall_semantic_candidates(
        self,
        user: UserProfile,
        query: str | None,
        base_items: List[ContentItem],
        semantic_pool_limit: int,
        semantic_top_k: int,
    ) -> List[ContentItem]:
        semantic_pool = self._build_semantic_pool(
            base_items=base_items,
            limit=semantic_pool_limit,
        )
        if not semantic_pool:
            return []

        # Semantic recall is a supplement, not a separate feed source: it scores
        # the already-recalled items plus a lightweight expansion pool.
        self.semantic_recall.enqueue_items(semantic_pool)
        hits = self.semantic_recall.recall(
            user=user,
            items=semantic_pool,
            query=query,
            top_k=semantic_top_k,
        )
        logger.info(
            "semantic supplement: base_items=%s pool=%s hits=%s query=%r",
            len(base_items),
            len(semantic_pool),
            len(hits),
            (query or "").strip()[:80],
        )
        return hits

    def _build_semantic_pool(
        self,
        base_items: List[ContentItem],
        limit: int,
    ) -> List[ContentItem]:
        if limit <= 0 or not self.semantic_recall.enabled:
            return []

        # Start from the candidates already recalled in this request, then
        # widen with a lightweight latest/trending pool for semantic expansion.
        semantic_pool: List[ContentItem] = list(base_items)
        try:
            semantic_pool.extend(self.get_latest_headlines(limit=limit)["items"])
        except Exception:
            pass
        try:
            semantic_pool.extend(
                self.get_trending_feeds(feed_type="hot_news", limit=limit)["items"]
            )
        except Exception:
            pass

        pool_by_id: Dict[str, ContentItem] = {}
        for item in semantic_pool:
            pool_by_id.setdefault(item.id, item)
        return list(pool_by_id.values())

    def _should_replace_candidate(self, current: ContentItem, candidate: ContentItem) -> bool:
        current_semantic = getattr(current, "semantic_match_score", 0.0) or 0.0
        candidate_semantic = getattr(candidate, "semantic_match_score", 0.0) or 0.0
        if candidate_semantic != current_semantic:
            return candidate_semantic > current_semantic

        if candidate.importance_score != current.importance_score:
            return candidate.importance_score > current.importance_score

        return candidate.published_at > current.published_at

    def _paginate_sequence(
        self,
        items: Sequence[T],
        limit: int,
        cursor: str | None = None,
    ) -> Dict[str, Any]:
        size = max(1, limit)
        offset = self._decode_offset_cursor(cursor)
        page_items = list(items[offset : offset + size])
        next_offset = offset + len(page_items)
        has_more = next_offset < len(items)
        payload: Dict[str, Any] = {
            "items": page_items,
            "cursor": str(offset),
            "has_more": has_more,
            "has_next": has_more,
        }
        if has_more:
            payload["next_cursor"] = str(next_offset)
        return payload

    def _decode_offset_cursor(self, cursor: str | None) -> int:
        if not cursor:
            return 0
        try:
            return max(0, int(cursor))
        except (TypeError, ValueError):
            return 0

    def _get_or_create_feed_session(
        self,
        user: UserProfile,
        query: str | None,
        cursor: str | None,
    ) -> FeedSessionState:
        self._prune_feed_sessions()
        signature = self._feed_user_signature(user)
        with self._feed_session_lock:
            if cursor:
                existing = self._feed_sessions.get(cursor)
                if existing and existing.user_signature == signature:
                    existing.updated_at = time.time()
                    logger.info(
                        "[feed] session reused: session_id=%s query=%r pending=%s delivered_events=%s",
                        existing.session_id,
                        (existing.query or "").strip()[:80],
                        len(existing.pending_clusters),
                        len(existing.delivered_event_ids),
                    )
                    return existing

            session_id = uuid.uuid4().hex
            session = FeedSessionState(
                session_id=session_id,
                user_signature=signature,
                query=query,
            )
            self._feed_sessions[session_id] = session
            logger.info(
                "[feed] session created: session_id=%s query=%r",
                session_id,
                (query or "").strip()[:80],
            )
            return session

    def _feed_user_signature(self, user: UserProfile) -> str:
        payload = {"user": asdict(user)}
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _prune_feed_sessions(self) -> None:
        if self._feed_session_ttl_seconds <= 0:
            return
        now = time.time()
        with self._feed_session_lock:
            expired = [
                session_id
                for session_id, session in self._feed_sessions.items()
                if now - session.updated_at > self._feed_session_ttl_seconds
            ]
            for session_id in expired:
                self._feed_sessions.pop(session_id, None)

    def _rank_clusters_for_user(
        self,
        user: UserProfile,
        items: List[ContentItem],
    ) -> List[EventCluster]:
        clusters = self._cluster_same_event(items)
        # Personalization happens after event consolidation so ranking decides
        # what event to show, not which near-duplicate article wins.
        return self.ranker.rank_for_user(user, clusters)

    def _cluster_same_event(self, items: List[ContentItem]) -> List[EventCluster]:
        return self.clusterer.cluster_same_event(items)

    def _start_semantic_warmup(self) -> None:
        if not self.semantic_recall.enabled:
            logger.info("semantic warmup skipped: semantic recall disabled")
            return
        if os.getenv("FOLLOWIN_SEMANTIC_WARMUP_ENABLED", "1") == "0":
            logger.info("semantic warmup skipped: FOLLOWIN_SEMANTIC_WARMUP_ENABLED=0")
            return
        if self._semantic_warmup_thread and self._semantic_warmup_thread.is_alive():
            logger.info("semantic warmup skipped: thread already running")
            return

        logger.info("starting semantic warmup thread")
        self._semantic_warmup_thread = threading.Thread(
            target=self._semantic_warmup_job,
            name="followin-semantic-warmup",
            daemon=True,
        )
        self._semantic_warmup_thread.start()

    def _semantic_warmup_job(self) -> None:
        latest_limit = int(os.getenv("FOLLOWIN_SEMANTIC_WARMUP_LATEST_LIMIT", "40"))
        trending_limit = int(os.getenv("FOLLOWIN_SEMANTIC_WARMUP_TRENDING_LIMIT", "40"))
        logger.info(
            "semantic warmup started: latest_limit=%s trending_limit=%s",
            latest_limit,
            trending_limit,
        )

        items: List[ContentItem] = []
        try:
            items.extend(self.get_latest_headlines(limit=latest_limit)["items"])
        except Exception as exc:
            logger.warning("semantic warmup latest fetch failed: %s", exc)
        try:
            items.extend(
                self.get_trending_feeds(feed_type="hot_news", limit=trending_limit)["items"]
            )
        except Exception as exc:
            logger.warning("semantic warmup trending fetch failed: %s", exc)

        if not items:
            logger.info("semantic warmup finished: no items fetched")
            return

        deduped: Dict[str, ContentItem] = {}
        for item in items:
            deduped[item.id] = item
        created = self.semantic_recall.precompute(list(deduped.values()))
        logger.info(
            "semantic warmup finished: fetched=%s deduped=%s indexed=%s",
            len(items),
            len(deduped),
            created,
        )
