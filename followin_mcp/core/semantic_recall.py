from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
from importlib.util import find_spec
from hashlib import md5
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, Iterable, List, Sequence

from openai import OpenAI

from .models import ContentItem, UserProfile

logger = logging.getLogger("followin_mcp.semantic_recall")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[semantic] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(
    getattr(logging, os.getenv("FOLLOWIN_SEMANTIC_LOG_LEVEL", "INFO").upper(), logging.INFO)
)
logger.propagate = False


class SemanticRecallEngine:
    def __init__(
        self,
        api_key: str | None,
        base_url: str | None = None,
        model: str = "text-embedding-3-small",
        db_path: str | None = None,
    ) -> None:
        self._normalize_proxy_env()
        self.enabled = bool(api_key)
        self.model = model
        self.db_path = db_path or os.getenv(
            "FOLLOWIN_SEMANTIC_INDEX_PATH",
            "data/semantic_recall/semantic_index.db",
        )
        self._client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        self._query_cache: Dict[str, List[float]] = {}
        self._enqueue_seen: set[str] = set()
        self._queue: Queue[ContentItem] = Queue()
        self._lock = threading.Lock()
        self._init_db()
        self._worker: threading.Thread | None = None
        if self.enabled:
            logger.info(
                "semantic recall enabled: model=%s db_path=%s",
                self.model,
                self.db_path,
            )
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="semantic-recall-indexer",
                daemon=True,
            )
            self._worker.start()
        else:
            logger.info("semantic recall disabled: missing OPENAI_API_KEY")

    @classmethod
    def from_env(cls) -> "SemanticRecallEngine":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            db_path=os.getenv("FOLLOWIN_SEMANTIC_INDEX_PATH"),
        )

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

    def enqueue_items(self, items: Sequence[ContentItem]) -> None:
        if not self.enabled:
            return
        enqueued = 0
        for item in items:
            item_hash = self._item_hash(item)
            if self._has_indexed_embedding(item.id, item_hash):
                continue
            enqueue_key = f"{item.id}:{item_hash}"
            with self._lock:
                if enqueue_key in self._enqueue_seen:
                    continue
                self._enqueue_seen.add(enqueue_key)
            self._queue.put(item)
            enqueued += 1
        if enqueued:
            logger.info("semantic enqueue: queued=%s", enqueued)

    def recall(
        self,
        user: UserProfile,
        items: Sequence[ContentItem],
        query: str | None = None,
        top_k: int = 6,
    ) -> List[ContentItem]:
        if not self.enabled or not items or top_k <= 0:
            return []

        self.enqueue_items(items)

        query_text = self._build_user_query(user, query=query)
        query_vector = self._embed_query(query_text)
        if not query_vector:
            return []

        scored: List[tuple[float, ContentItem]] = []
        for item in items:
            item_vector = self._load_item_vector(item)
            if not item_vector:
                item.semantic_match_score = 0.0
                continue
            score = self._cosine_similarity(query_vector, item_vector)
            item.semantic_match_score = round(score, 4)
            scored.append((score, item))

        scored.sort(key=lambda row: row[0], reverse=True)
        top_hits = [item for _score, item in scored[:top_k]]
        if top_hits:
            logger.info(
                "semantic recall hit: query=%r pool=%s top_k=%s top_ids=%s top_scores=%s",
                (query or "").strip()[:80],
                len(items),
                len(top_hits),
                [item.id for item in top_hits[:3]],
                [round(score, 4) for score, _item in scored[:3]],
            )
        return top_hits

    def get_item_vector(self, item: ContentItem) -> List[float]:
        if not self.enabled:
            return []
        self.enqueue_items([item])
        return self._load_item_vector(item)

    def similarity_between_items(self, left: ContentItem, right: ContentItem) -> float:
        if not self.enabled:
            return 0.0
        self.enqueue_items([left, right])
        left_vector = self._load_item_vector(left)
        right_vector = self._load_item_vector(right)
        if not left_vector or not right_vector:
            return 0.0
        return self._cosine_similarity(left_vector, right_vector)

    def precompute(self, items: Sequence[ContentItem]) -> int:
        if not self.enabled or not items:
            return 0

        created = 0
        for item in items:
            item_hash = self._item_hash(item)
            if self._has_indexed_embedding(item.id, item_hash):
                continue
            vector = self._embed_item_sync(item)
            if not vector:
                continue
            self._upsert_item_vector(item, item_hash, vector)
            created += 1
        logger.info("semantic precompute finished: input=%s indexed=%s", len(items), created)
        return created

    def _worker_loop(self) -> None:
        logger.info("semantic index worker started")
        while True:
            try:
                item = self._queue.get(timeout=1)
            except Empty:
                continue

            item_hash = self._item_hash(item)
            enqueue_key = f"{item.id}:{item_hash}"
            try:
                if not self._has_indexed_embedding(item.id, item_hash):
                    vector = self._embed_item_sync(item)
                    if vector:
                        self._upsert_item_vector(item, item_hash, vector)
                        logger.info("semantic index worker: indexed item_id=%s", item.id)
            finally:
                with self._lock:
                    self._enqueue_seen.discard(enqueue_key)
                self._queue.task_done()

    def _build_user_query(self, user: UserProfile, query: str | None = None) -> str:
        interests = ", ".join(user.interests) or "crypto market"
        followed = ", ".join(user.followed_projects) or "none"
        muted = ", ".join(user.muted_topics) or "none"
        lines = [
            "User interest profile for crypto content retrieval.\n",
            f"Interests: {interests}\n",
            f"Followed projects: {followed}\n",
            f"Muted topics: {muted}\n",
            f"Risk preference: {user.risk_preference}\n",
            f"Preferred languages: {', '.join(user.preferred_languages) or 'zh'}",
        ]
        if query and query.strip():
            lines.append(f"\nCurrent user request: {query.strip()}")
        return "".join(lines)

    def _item_text(self, item: ContentItem) -> str:
        terms = [
            *item.entities.projects,
            *item.entities.tokens,
            *item.entities.chains,
            *item.entities.topics,
        ]
        return "\n".join(
            [
                f"Title: {item.title}",
                f"Summary: {item.summary}",
                f"Content: {item.content[:800]}",
                f"Entities: {', '.join(terms[:12])}",
                f"Event type: {item.event_type}",
                f"Source: {item.source_name}",
            ]
        )

    def _item_hash(self, item: ContentItem) -> str:
        return md5(self._item_text(item).encode("utf-8")).hexdigest()

    def _embed_query(self, text: str) -> List[float]:
        normalized = text.strip()
        if not normalized or self._client is None:
            return []

        cache_key = md5(f"{self.model}:{normalized}".encode("utf-8")).hexdigest()
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached

        response = self._client.embeddings.create(model=self.model, input=normalized)
        vector = list(response.data[0].embedding) if response.data else []
        self._query_cache[cache_key] = vector
        return vector

    def _embed_item_sync(self, item: ContentItem) -> List[float]:
        if self._client is None:
            return []
        payload = self._item_text(item).strip()
        if not payload:
            return []
        response = self._client.embeddings.create(model=self.model, input=payload)
        return list(response.data[0].embedding) if response.data else []

    def _load_item_vector(self, item: ContentItem) -> List[float]:
        item_hash = self._item_hash(item)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT embedding_json
                FROM item_embeddings
                WHERE item_id = ? AND item_hash = ? AND model = ?
                """,
                (item.id, item_hash, self.model),
            ).fetchone()
        if not row:
            return []
        return list(json.loads(row[0]))

    def _has_indexed_embedding(self, item_id: str, item_hash: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM item_embeddings
                WHERE item_id = ? AND item_hash = ? AND model = ?
                LIMIT 1
                """,
                (item_id, item_hash, self.model),
            ).fetchone()
        return row is not None

    def _upsert_item_vector(self, item: ContentItem, item_hash: str, vector: List[float]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO item_embeddings (
                    item_id,
                    item_hash,
                    model,
                    published_at,
                    title,
                    source_name,
                    embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item_hash,
                    self.model,
                    item.published_at.isoformat(),
                    item.title,
                    item.source_name,
                    json.dumps(vector),
                ),
            )
            conn.commit()

    def _init_db(self) -> None:
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS item_embeddings (
                    item_id TEXT NOT NULL,
                    item_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    published_at TEXT,
                    title TEXT,
                    source_name TEXT,
                    embedding_json TEXT NOT NULL,
                    PRIMARY KEY (item_id, item_hash, model)
                )
                """
            )
            conn.commit()

    @staticmethod
    def _cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
        left_values = list(left)
        right_values = list(right)
        if not left_values or not right_values or len(left_values) != len(right_values):
            return 0.0

        dot = sum(a * b for a, b in zip(left_values, right_values))
        left_norm = math.sqrt(sum(a * a for a in left_values))
        right_norm = math.sqrt(sum(b * b for b in right_values))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)
