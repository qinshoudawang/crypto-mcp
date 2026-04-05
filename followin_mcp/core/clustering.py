from __future__ import annotations

import hashlib
from typing import Dict, List

from .models import ContentItem, EventCluster


class EventClusterer:
    def cluster_same_event(self, items: List[ContentItem]) -> List[EventCluster]:
        buckets: Dict[str, List[ContentItem]] = {}
        for item in items:
            project_key = item.entities.projects[0] if item.entities.projects else "misc"
            key = f"{item.event_type}:{project_key}"
            buckets.setdefault(key, []).append(item)

        clusters: List[EventCluster] = []
        for key, grouped_items in buckets.items():
            grouped_items.sort(key=lambda x: x.published_at)
            first = grouped_items[0]
            clusters.append(
                EventCluster(
                    event_id=hashlib.md5(key.encode("utf-8")).hexdigest()[:10],
                    title=first.title,
                    event_type=first.event_type,
                    entities=first.entities,
                    first_seen_at=grouped_items[0].published_at,
                    last_updated_at=grouped_items[-1].published_at,
                    importance_score=max(item.importance_score for item in grouped_items),
                    items=grouped_items,
                    key_points=self._build_key_points(grouped_items),
                    risks=self._build_risks(grouped_items),
                )
            )

        clusters.sort(key=lambda c: c.importance_score, reverse=True)
        return clusters

    def _build_key_points(self, items: List[ContentItem]) -> List[str]:
        return [item.summary for item in items[:3]]

    def _build_risks(self, items: List[ContentItem]) -> List[str]:
        risks = [
            "Possible security incident, needs closer tracking."
            for item in items
            if item.event_type == "exploit"
        ]
        return list(dict.fromkeys(risks))
