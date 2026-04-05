from __future__ import annotations

from typing import List

from .models import EventCluster, UserProfile


class DigestBuilder:
    def build_personal_digest(
        self,
        user: UserProfile,
        clusters: List[EventCluster],
        max_items: int = 5,
    ) -> str:
        selected = clusters[:max_items]
        lines = [f"用户 {user.user_id} 今日简报", ""]

        for idx, cluster in enumerate(selected, start=1):
            lines.append(f"{idx}. {cluster.title}")
            lines.append(f"   类型: {cluster.event_type} | 重要性: {cluster.importance_score}")

            if cluster.entities.projects:
                lines.append(f"   项目: {', '.join(cluster.entities.projects)}")
            if cluster.entities.tokens:
                lines.append(f"   代币: {', '.join(cluster.entities.tokens)}")
            if cluster.entities.topics:
                lines.append(f"   主题: {', '.join(cluster.entities.topics)}")
            if cluster.key_points:
                lines.append(f"   要点: {cluster.key_points[0]}")
            if cluster.risks:
                lines.append(f"   风险: {cluster.risks[0]}")
            lines.append("")

        return "\n".join(lines)
