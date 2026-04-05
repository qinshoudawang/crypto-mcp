from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from .models import EventCluster, UserProfile


class UserRanker:
    def rank_for_user(self, user: UserProfile, clusters: List[EventCluster]) -> List[EventCluster]:
        if not clusters:
            return []

        scored = []
        now = datetime.now(timezone.utc)
        for cluster in clusters:
            signals = self._compute_signals(user=user, cluster=cluster, now=now)
            scored.append({
                "cluster": cluster,
                "signals": signals,
                "score": self._combine_signals(signals),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return [item["cluster"] for item in self._rerank(scored)]

    def explain_scores(self, user: UserProfile, clusters: List[EventCluster]) -> List[Dict[str, Any]]:
        if not clusters:
            return []

        now = datetime.now(timezone.utc)
        scored: List[Dict[str, Any]] = []
        for cluster in clusters:
            signals = self._compute_signals(user=user, cluster=cluster, now=now)
            scored.append({
                "cluster": cluster,
                "signals": signals,
                "score": self._combine_signals(signals),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        raw_ranked = scored[:]
        reranked = self._rerank(scored)
        raw_positions = {id(row["cluster"]): idx for idx, row in enumerate(raw_ranked, start=1)}

        explain_rows: List[Dict[str, Any]] = []
        for rank, row in enumerate(reranked, start=1):
            cluster = row["cluster"]
            entity_terms = {
                *cluster.entities.projects,
                *cluster.entities.tokens,
                *cluster.entities.chains,
                *cluster.entities.topics,
            }
            lower_entity_terms = {term.lower() for term in entity_terms}
            explain_rows.append({
                "rank": rank,
                "raw_rank": raw_positions[id(cluster)],
                "title": cluster.title,
                "event_type": cluster.event_type,
                "projects": cluster.entities.projects,
                "tokens": cluster.entities.tokens,
                "chains": cluster.entities.chains,
                "topics": cluster.entities.topics,
                "final_score": row["score"],
                "signals": row["signals"],
                "matched_interests": [
                    interest for interest in user.interests if interest.lower() in lower_entity_terms
                ],
                "matched_follows": [
                    project for project in user.followed_projects if project.lower() in lower_entity_terms
                ],
                "matched_mutes": [
                    topic for topic in user.muted_topics if topic.lower() in lower_entity_terms
                ],
                "entity_sources": cluster.items[0].entity_sources if cluster.items else {},
                "entity_confidence": cluster.items[0].entity_confidence if cluster.items else {},
                "score_breakdown": {
                    "importance": round(0.30 * row["signals"]["importance_score"], 4),
                    "freshness": round(0.20 * row["signals"]["freshness_score"], 4),
                    "follow": round(0.20 * row["signals"]["follow_affinity_score"], 4),
                    "interest": round(0.15 * row["signals"]["interest_match_score"], 4),
                    "source": round(0.10 * row["signals"]["source_quality_score"], 4),
                    "risk": round(0.05 * row["signals"]["risk_boost_score"], 4),
                    "mute_penalty": round(-0.20 * row["signals"]["mute_penalty"], 4),
                },
            })
        return explain_rows

    def _compute_signals(
        self,
        user: UserProfile,
        cluster: EventCluster,
        now: datetime,
    ) -> Dict[str, float]:
        return {
            "importance_score": self._importance_score(cluster),
            "freshness_score": self._freshness_score(cluster, now),
            "follow_affinity_score": self._follow_affinity_score(user, cluster),
            "interest_match_score": self._interest_match_score(user, cluster),
            "source_quality_score": self._source_quality_score(cluster),
            "risk_boost_score": self._risk_boost_score(cluster),
            "mute_penalty": self._mute_penalty(user, cluster),
        }

    def _combine_signals(self, signals: Dict[str, float]) -> float:
        score = (
            0.30 * signals["importance_score"]
            + 0.20 * signals["freshness_score"]
            + 0.20 * signals["follow_affinity_score"]
            + 0.15 * signals["interest_match_score"]
            + 0.10 * signals["source_quality_score"]
            + 0.05 * signals["risk_boost_score"]
            - 0.20 * signals["mute_penalty"]
        )
        return round(score, 4)

    def _importance_score(self, cluster: EventCluster) -> float:
        return self._clamp(cluster.importance_score)

    def _freshness_score(self, cluster: EventCluster, now: datetime) -> float:
        delta_hours = max((now - cluster.last_updated_at).total_seconds() / 3600, 0)
        if delta_hours <= 1:
            return 1.0
        if delta_hours <= 6:
            return 0.85
        if delta_hours <= 24:
            return 0.65
        if delta_hours <= 72:
            return 0.40
        return 0.20

    def _follow_affinity_score(self, user: UserProfile, cluster: EventCluster) -> float:
        if not user.followed_projects:
            return 0.0
        matched_scores: List[float] = []
        followed_set = {p.lower() for p in user.followed_projects}
        for item in cluster.items:
            project_confidence = item.entity_confidence.get("projects", {})
            for project in item.entities.projects:
                if project.lower() not in followed_set:
                    continue
                level = project_confidence.get(project, "weak")
                matched_scores.append(self._confidence_to_score(level, strong=1.0, medium=0.7, weak=0.55))
        return max(matched_scores) if matched_scores else 0.0

    def _interest_match_score(self, user: UserProfile, cluster: EventCluster) -> float:
        if not user.interests:
            return 0.0

        best_matches: Dict[str, float] = {}
        for item in cluster.items:
            confidence_maps = item.entity_confidence
            entity_buckets = {
                "projects": item.entities.projects,
                "tokens": item.entities.tokens,
                "chains": item.entities.chains,
                "topics": item.entities.topics,
            }
            for bucket, values in entity_buckets.items():
                for value in values:
                    confidence = confidence_maps.get(bucket, {}).get(value, "weak")
                    score = self._confidence_to_score(confidence, strong=1.0, medium=0.75, weak=0.55)
                    best_matches[value.lower()] = max(best_matches.get(value.lower(), 0.0), score)

        if not best_matches:
            return 0.0

        matched_score = 0.0
        for interest in user.interests:
            matched_score += best_matches.get(interest.lower(), 0.0)
        return self._clamp(matched_score / max(len(user.interests), 1))

    def _source_quality_score(self, cluster: EventCluster) -> float:
        if not cluster.items:
            return 0.0
        avg = sum(item.credibility_score for item in cluster.items) / len(cluster.items)
        return self._clamp(avg)

    def _risk_boost_score(self, cluster: EventCluster) -> float:
        if cluster.event_type == "exploit":
            return 1.0
        if cluster.risks:
            return 0.7
        return 0.0

    def _mute_penalty(self, user: UserProfile, cluster: EventCluster) -> float:
        if not user.muted_topics:
            return 0.0
        entity_terms = {
            *[x.lower() for x in cluster.entities.projects],
            *[x.lower() for x in cluster.entities.tokens],
            *[x.lower() for x in cluster.entities.chains],
            *[x.lower() for x in cluster.entities.topics],
        }
        for muted in user.muted_topics:
            if muted.lower() in entity_terms:
                return 1.0
        return 0.0

    def _confidence_to_score(self, level: str, strong: float, medium: float, weak: float) -> float:
        if level == "strong":
            return strong
        if level == "medium":
            return medium
        return weak

    def _rerank(self, scored: List[Dict]) -> List[Dict]:
        if len(scored) <= 2:
            return scored
        remaining = scored[:]
        result = []
        used_recent_projects: List[Set[str]] = []
        used_recent_topics: List[Set[str]] = []

        while remaining:
            chosen_idx = None
            for idx, candidate in enumerate(remaining):
                cluster = candidate["cluster"]
                project_set = {x.lower() for x in cluster.entities.projects}
                topic_set = {x.lower() for x in cluster.entities.topics}
                if not result:
                    chosen_idx = idx
                    break

                recent_projects = used_recent_projects[-1] if used_recent_projects else set()
                recent_topics = used_recent_topics[-1] if used_recent_topics else set()
                same_project = bool(project_set and recent_projects and (project_set & recent_projects))
                same_topic = bool(topic_set and recent_topics and (topic_set & recent_topics))
                if not same_project and not same_topic:
                    chosen_idx = idx
                    break

            if chosen_idx is None:
                chosen_idx = 0
            chosen = remaining.pop(chosen_idx)
            cluster = chosen["cluster"]
            result.append(chosen)
            used_recent_projects.append({x.lower() for x in cluster.entities.projects})
            used_recent_topics.append({x.lower() for x in cluster.entities.topics})
        return result

    @staticmethod
    def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
        return max(min_value, min(value, max_value))
