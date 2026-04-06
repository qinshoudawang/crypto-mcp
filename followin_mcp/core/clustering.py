from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import datetime
from typing import Callable, Dict, List, Sequence, Set

from .models import ContentItem, Entities, EventCluster
from .taxonomy_rules import EventType


class EventClusterer:
    MAX_CLUSTER_WINDOW_HOURS = 48
    MERGE_THRESHOLD = 0.45

    def __init__(
        self,
        semantic_similarity_fn: Callable[[ContentItem, ContentItem], float] | None = None,
    ) -> None:
        self.semantic_similarity_fn = semantic_similarity_fn

    def cluster_same_event(self, items: List[ContentItem]) -> List[EventCluster]:
        if not items:
            return []

        # Process newer items first so merge decisions compare against recent
        # coverage rather than anchoring on stale reports.
        ordered_items = sorted(items, key=lambda item: item.published_at, reverse=True)
        cluster_buckets: List[List[ContentItem]] = []

        for item in ordered_items:
            best_index = -1
            best_score = 0.0
            # Greedily assign the item to the single best existing bucket.
            for index, bucket in enumerate(cluster_buckets):
                score = self._cluster_match_score(item, bucket)
                if score > best_score:
                    best_score = score
                    best_index = index

            # Only merge when the best candidate bucket clears the match threshold;
            # otherwise start a new event bucket for this item.
            if best_index >= 0 and best_score >= self.MERGE_THRESHOLD:
                cluster_buckets[best_index].append(item)
            else:
                cluster_buckets.append([item])

        clusters = [self._build_cluster(bucket) for bucket in cluster_buckets]
        clusters.sort(key=lambda cluster: cluster.importance_score, reverse=True)
        return clusters

    def _cluster_match_score(self, item: ContentItem, bucket: Sequence[ContentItem]) -> float:
        # Use the newest bucket item as a cheap representative instead of
        # comparing against every item in the candidate cluster.
        representative = bucket[0]
        event_type_score = self._event_type_score(item, bucket)
        if event_type_score == 0.0:
            return 0.0

        time_score = self._time_proximity_score(item.published_at, representative.published_at)
        if time_score == 0.0:
            return 0.0

        project_overlap = self._confidence_overlap_score(item, bucket, "projects")
        token_overlap = self._confidence_overlap_score(item, bucket, "tokens")
        chain_overlap = self._confidence_overlap_score(item, bucket, "chains")
        topic_overlap = self._confidence_overlap_score(item, bucket, "topics")
        title_similarity = self._title_similarity(item.title, representative.title)
        semantic_similarity = self._semantic_similarity(item, representative)

        # Blend hard structural compatibility with softer lexical/semantic
        # similarity; event type and time still act as the main gates.
        score = event_type_score + time_score
        score += 0.30 * project_overlap
        score += 0.18 * token_overlap
        score += 0.12 * chain_overlap
        score += 0.10 * topic_overlap
        score += 0.18 * title_similarity
        score += 0.15 * semantic_similarity

        # Do not merge purely on broad event-type/time agreement when there is no
        # meaningful entity, title, or semantic evidence tying the items together.
        if (
            not any([project_overlap, token_overlap, chain_overlap, topic_overlap])
            and title_similarity < 0.4
            and semantic_similarity < 0.55
        ):
            return 0.0
        return score

    def _event_type_score(self, item: ContentItem, bucket: Sequence[ContentItem]) -> float:
        bucket_event_types = {bucket_item.event_type for bucket_item in bucket}
        if item.event_type in bucket_event_types:
            return 0.35
        # Unknown is treated as weakly compatible so upstream classification
        # noise does not block otherwise obvious merges.
        if item.event_type == EventType.UNKNOWN or EventType.UNKNOWN in bucket_event_types:
            return 0.12
        return 0.0

    def _time_proximity_score(self, left: datetime, right: datetime) -> float:
        delta_hours = abs((left - right).total_seconds()) / 3600
        if delta_hours <= 6:
            return 0.20
        if delta_hours <= 24:
            return 0.12
        if delta_hours <= self.MAX_CLUSTER_WINDOW_HOURS:
            return 0.06
        return 0.0

    def _collect_bucket_entities(self, bucket: Sequence[ContentItem], attr: str) -> Set[str]:
        values: Set[str] = set()
        for item in bucket:
            values.update(getattr(item.entities, attr))
        return values

    def _confidence_overlap_score(
        self,
        item: ContentItem,
        bucket: Sequence[ContentItem],
        attr: str,
    ) -> float:
        # Compare entity overlap using confidence-weighted Jaccard so that
        # strong shared entities count more than weak matches.
        left_weights = self._entity_weight_map(item, attr)
        right_weights = self._bucket_entity_weight_map(bucket, attr)
        if not left_weights or not right_weights:
            return 0.0

        shared_keys = set(left_weights) & set(right_weights)
        if not shared_keys:
            return 0.0

        intersection = sum(min(left_weights[key], right_weights[key]) for key in shared_keys)
        union_keys = set(left_weights) | set(right_weights)
        union = sum(max(left_weights.get(key, 0.0), right_weights.get(key, 0.0)) for key in union_keys)
        if union == 0.0:
            return 0.0
        return intersection / union

    def _entity_weight_map(self, item: ContentItem, attr: str) -> Dict[str, float]:
        values = getattr(item.entities, attr)
        confidence_map = item.entity_confidence.get(attr, {})
        weights: Dict[str, float] = {}
        for value in values:
            # Keep the strongest confidence observed for each normalized entity key.
            weights[value.lower()] = max(
                weights.get(value.lower(), 0.0),
                self._confidence_weight(confidence_map.get(value, "weak")),
            )
        return weights

    def _bucket_entity_weight_map(self, bucket: Sequence[ContentItem], attr: str) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for item in bucket:
            for key, value in self._entity_weight_map(item, attr).items():
                # Bucket-level entity strength reflects the strongest supporting item.
                weights[key] = max(weights.get(key, 0.0), value)
        return weights

    def _confidence_weight(self, level: str) -> float:
        if level == "strong":
            return 1.0
        if level == "medium":
            return 0.7
        return 0.4

    def _title_similarity(self, left: str, right: str) -> float:
        left_tokens = self._title_tokens(left)
        right_tokens = self._title_tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        intersection = left_tokens & right_tokens
        union = left_tokens | right_tokens
        if not union:
            return 0.0
        return len(intersection) / len(union)

    def _title_tokens(self, value: str) -> Set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9\u4e00-\u9fff]+", value.lower())
            if len(token) > 1
        }

    def _semantic_similarity(self, left: ContentItem, right: ContentItem) -> float:
        if self.semantic_similarity_fn is None:
            return 0.0
        try:
            return max(0.0, self.semantic_similarity_fn(left, right))
        except Exception:
            return 0.0

    def _build_cluster(self, items: Sequence[ContentItem]) -> EventCluster:
        grouped_items = sorted(items, key=lambda item: item.published_at)
        # Represent the cluster with the strongest item so downstream title and
        # key-point generation anchor on the best coverage.
        representative = max(
            grouped_items,
            key=lambda item: (item.importance_score, item.published_at.timestamp()),
        )
        event_type = self._dominant_event_type(grouped_items)
        entities = self._merge_entities(grouped_items)
        event_seed = "|".join(sorted(item.id for item in grouped_items))
        return EventCluster(
            event_id=hashlib.md5(event_seed.encode("utf-8")).hexdigest()[:10],
            title=representative.title,
            event_type=event_type,
            entities=entities,
            first_seen_at=grouped_items[0].published_at,
            last_updated_at=grouped_items[-1].published_at,
            importance_score=max(item.importance_score for item in grouped_items),
            items=list(grouped_items),
            key_points=self._build_key_points(grouped_items),
            risks=self._build_risks(grouped_items),
        )

    def _dominant_event_type(self, items: Sequence[ContentItem]) -> EventType:
        counts = Counter(item.event_type for item in items)
        return counts.most_common(1)[0][0]

    def _merge_entities(self, items: Sequence[ContentItem]) -> Entities:
        return Entities(
            projects=self._dedupe([value for item in items for value in item.entities.projects]),
            tokens=self._dedupe([value for item in items for value in item.entities.tokens]),
            chains=self._dedupe([value for item in items for value in item.entities.chains]),
            topics=self._dedupe([value for item in items for value in item.entities.topics]),
            people=self._dedupe([value for item in items for value in item.entities.people]),
        )

    def _dedupe(self, values: Sequence[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    def _build_key_points(self, items: Sequence[ContentItem]) -> List[str]:
        ranked = sorted(
            items,
            key=lambda item: (item.importance_score, item.published_at.timestamp()),
            reverse=True,
        )
        summaries = [item.summary for item in ranked if item.summary]
        return self._dedupe(summaries)[:3]

    def _build_risks(self, items: Sequence[ContentItem]) -> List[str]:
        risks = [
            "Possible security incident, needs closer tracking."
            for item in items
            if item.event_type == EventType.EXPLOIT
        ]
        return self._dedupe(risks)
