from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from .models import EventCluster, UserProfile
from .taxonomy_rules import EventType


class UserRanker:
    MMR_LAMBDA = 0.78

    def rank_for_user(self, user: UserProfile, clusters: List[EventCluster]) -> List[EventCluster]:
        if not clusters:
            return []

        scored = []
        now = datetime.now(timezone.utc)
        for cluster in clusters:
            # Keep the ranking pipeline explainable: compute named signals first,
            # then combine them with explicit weights.
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
                    "importance": round(0.25 * row["signals"]["importance_score"], 4),
                    "freshness": round(0.20 * row["signals"]["freshness_score"], 4),
                    "follow": round(0.18 * row["signals"]["follow_affinity_score"], 4),
                    "interest": round(0.15 * row["signals"]["interest_match_score"], 4),
                    "semantic": round(0.12 * row["signals"]["semantic_match_score"], 4),
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
        # Separate content quality, user affinity, and negative controls so each
        # family of signals can be tuned independently.
        return {
            "importance_score": self._importance_score(cluster),
            "freshness_score": self._freshness_score(cluster, now),
            "follow_affinity_score": self._follow_affinity_score(user, cluster),
            "interest_match_score": self._interest_match_score(user, cluster),
            "semantic_match_score": self._semantic_match_score(cluster),
            "source_quality_score": self._source_quality_score(cluster),
            "risk_boost_score": self._risk_boost_score(cluster),
            "mute_penalty": self._mute_penalty(user, cluster),
        }

    def _combine_signals(self, signals: Dict[str, float]) -> float:
        # This stays as a lightweight heuristic ranker rather than a learned
        # model, so weights are intentionally explicit and easy to inspect.
        score = (
            0.25 * signals["importance_score"]
            + 0.20 * signals["freshness_score"]
            + 0.18 * signals["follow_affinity_score"]
            + 0.15 * signals["interest_match_score"]
            + 0.12 * signals["semantic_match_score"]
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

    def _semantic_match_score(self, cluster: EventCluster) -> float:
        if not cluster.items:
            return 0.0
        # At cluster level we care whether the event has at least one highly
        # relevant supporting item, so use the best semantic hit instead of an average.
        best = max(getattr(item, "semantic_match_score", 0.0) for item in cluster.items)
        return self._clamp(best)

    def _risk_boost_score(self, cluster: EventCluster) -> float:
        if cluster.event_type == EventType.EXPLOIT:
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
        result: List[Dict] = []

        while remaining:
            if not result:
                # Keep the top-ranked candidate as the anchor item.
                chosen = remaining.pop(0)
                result.append(chosen)
                continue

            chosen_idx = 0
            best_mmr = float("-inf")
            for idx, candidate in enumerate(remaining):
                relevance = candidate["score"]
                # Compare against a short recent window so we penalize local repetition
                # without flattening the whole feed.
                redundancy = max(
                    self._cluster_redundancy(candidate["cluster"], selected["cluster"])
                    for selected in result[-5:]
                )
                # MMR trades off base ranking quality against similarity to already-selected results.
                mmr_score = self.MMR_LAMBDA * relevance - (1 - self.MMR_LAMBDA) * redundancy
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    chosen_idx = idx

            chosen = remaining.pop(chosen_idx)
            result.append(chosen)
        return result

    def _cluster_redundancy(self, left: EventCluster, right: EventCluster) -> float:
        # Redundancy is simpler than clustering-time similarity: here we only
        # need a feed-level "too similar to show back-to-back" signal.
        project_overlap = self._set_overlap(left.entities.projects, right.entities.projects)
        token_overlap = self._set_overlap(left.entities.tokens, right.entities.tokens)
        topic_overlap = self._set_overlap(left.entities.topics, right.entities.topics)
        chain_overlap = self._set_overlap(left.entities.chains, right.entities.chains)
        event_type_overlap = 1.0 if left.event_type == right.event_type else 0.0

        redundancy = (
            0.34 * project_overlap
            + 0.20 * token_overlap
            + 0.20 * topic_overlap
            + 0.10 * chain_overlap
            + 0.16 * event_type_overlap
        )
        return self._clamp(redundancy)

    def _set_overlap(self, left: List[str], right: List[str]) -> float:
        if not left or not right:
            return 0.0
        left_set = {value.lower() for value in left}
        right_set = {value.lower() for value in right}
        intersection = left_set & right_set
        union = left_set | right_set
        if not union:
            return 0.0
        return len(intersection) / len(union)

    @staticmethod
    def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
        return max(min_value, min(value, max_value))
