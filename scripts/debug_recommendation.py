from __future__ import annotations

import os

from dotenv import load_dotenv

from followin_mcp.core.adapters import FollowinAPIAdapter, FollowinAPIError
from followin_mcp.core.models import EventCluster, UserProfile
from followin_mcp.core.service import FollowinMCPService


def print_divider(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def format_entities(cluster: EventCluster) -> str:
    parts = []
    if cluster.entities.projects:
        parts.append(f"projects={','.join(cluster.entities.projects)}")
    if cluster.entities.tokens:
        parts.append(f"tokens={','.join(cluster.entities.tokens)}")
    if cluster.entities.chains:
        parts.append(f"chains={','.join(cluster.entities.chains)}")
    if cluster.entities.topics:
        parts.append(f"topics={','.join(cluster.entities.topics)}")
    return " | ".join(parts) if parts else "no-entities"


def format_entity_sources(row: dict) -> str:
    entity_sources = row.get("entity_sources", {})
    entity_confidence = row.get("entity_confidence", {})
    lines = []

    for bucket in ["projects", "tokens", "chains", "topics"]:
        bucket_sources = entity_sources.get(bucket, {})
        if not bucket_sources:
            continue
        formatted = ", ".join(
            f"{entity}({entity_confidence.get(bucket, {}).get(entity, 'unknown')})"
            f"<-{'+'.join(sources)}"
            for entity, sources in bucket_sources.items()
        )
        lines.append(f"{bucket}:{formatted}")

    return " | ".join(lines)


def format_reason_tags(row: dict) -> str:
    reasons = []

    if row["matched_follows"]:
        reasons.append(f"follow:{','.join(row['matched_follows'])}")
    if row["matched_interests"]:
        reasons.append(f"interest:{','.join(row['matched_interests'])}")
    if row["signals"]["risk_boost_score"] > 0:
        reasons.append("risk-boost")
    if row["matched_mutes"]:
        reasons.append(f"muted:{','.join(row['matched_mutes'])}")
    if row["signals"]["freshness_score"] >= 0.85:
        reasons.append("fresh")
    if row["signals"]["importance_score"] >= 0.7:
        reasons.append("high-importance")

    return ", ".join(reasons) if reasons else "general-ranking"


def format_miss_tags(row: dict, user: UserProfile) -> str:
    misses = []

    if user.followed_projects and not row["matched_follows"]:
        misses.append("no-follow-match")
    if user.interests and not row["matched_interests"]:
        misses.append("no-interest-match")
    if row["signals"]["mute_penalty"] == 0:
        misses.append("no-mute-hit")

    return ", ".join(misses)


def print_user_profile(user: UserProfile) -> None:
    print("用户画像")
    print(f"  user_id           : {user.user_id}")
    print(f"  interests         : {', '.join(user.interests) or '-'}")
    print(f"  followed_projects : {', '.join(user.followed_projects) or '-'}")
    print(f"  muted_topics      : {', '.join(user.muted_topics) or '-'}")


def print_recommendation_table(
    service: FollowinMCPService,
    user: UserProfile,
    items,
    max_items: int = 8,
) -> None:
    clusters = service.cluster_same_event(items)
    explain_rows = service.ranker.explain_scores(user, clusters)

    print("Top recommendations")
    for row in explain_rows[:max_items]:
        print(
            f"{row['rank']:>2}. rerank={row['rank']} raw={row['raw_rank']} "
            f"score={row['final_score']:.4f} | "
            f"{row['title'][:48]}"
        )
        print(f"    hit={format_reason_tags(row)}")
        miss_tags = format_miss_tags(row, user)
        if miss_tags:
            print(f"    miss={miss_tags}")
        print(
            "    score_parts="
            f"imp:{row['score_breakdown']['importance']:+.4f} "
            f"fresh:{row['score_breakdown']['freshness']:+.4f} "
            f"follow:{row['score_breakdown']['follow']:+.4f} "
            f"interest:{row['score_breakdown']['interest']:+.4f} "
            f"source:{row['score_breakdown']['source']:+.4f} "
            f"risk:{row['score_breakdown']['risk']:+.4f} "
            f"mute:{row['score_breakdown']['mute_penalty']:+.4f}"
        )
        entity_summary = " | ".join(
            part
            for part in [
                f"projects={','.join(row['projects'])}" if row["projects"] else "",
                f"tokens={','.join(row['tokens'])}" if row["tokens"] else "",
                f"chains={','.join(row['chains'])}" if row["chains"] else "",
                f"topics={','.join(row['topics'])}" if row["topics"] else "",
            ]
            if part
        )
        if entity_summary:
            print(f"    entities={entity_summary}")
        source_summary = format_entity_sources(row)
        if source_summary:
            print(f"    entity_sources={source_summary}")


def print_acceptance_summary(
    service: FollowinMCPService,
    user: UserProfile,
    items,
    top_k: int = 5,
) -> None:
    clusters = service.cluster_same_event(items)
    explain_rows = service.ranker.explain_scores(user, clusters)[:top_k]

    follow_hits = sum(1 for row in explain_rows if row["matched_follows"])
    interest_hits = sum(1 for row in explain_rows if row["matched_interests"])
    mute_hits = sum(1 for row in explain_rows if row["matched_mutes"])

    print(f"Top {top_k} 里命中 followed_projects 的有 {follow_hits} 条")
    print(f"Top {top_k} 里命中 interests 的有 {interest_hits} 条")
    print(f"Top {top_k} 里触发 muted_topics 降权的有 {mute_hits} 条")


def print_baseline_vs_personalized(
    service: FollowinMCPService,
    user: UserProfile,
    items,
    max_items: int = 5,
) -> None:
    clusters = service.cluster_same_event(items)
    baseline = sorted(clusters, key=lambda c: c.importance_score, reverse=True)
    personalized = service.ranker.rank_for_user(user, clusters)

    print("Baseline by importance")
    for idx, cluster in enumerate(baseline[:max_items], start=1):
        print(
            f"{idx:>2}. imp={cluster.importance_score:.2f} | "
            f"{cluster.title[:48]} | {format_entities(cluster)}"
        )

    print("\nPersonalized ranking")
    for idx, cluster in enumerate(personalized[:max_items], start=1):
        print(
            f"{idx:>2}. imp={cluster.importance_score:.2f} | "
            f"{cluster.title[:48]} | {format_entities(cluster)}"
        )


def safe_get_api_key() -> str:
    api_key = os.getenv("FOLLOWIN_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 FOLLOWIN_API_KEY 环境变量。\n"
            "请先设置，例如：\n"
            "export FOLLOWIN_API_KEY='your_api_key'"
        )
    return api_key


def build_demo_user() -> UserProfile:
    return UserProfile(
        user_id="u001",
        interests=["Solana", "DeFi", "AI Agents", "Base"],
        muted_topics=["NFT"],
        followed_projects=["Jupiter", "Base"],
    )


def run_recommendation_debug_demo(service: FollowinMCPService, user: UserProfile) -> None:
    print_divider("USER RECOMMENDATION DEBUG")
    print_user_profile(user)

    items = service.recall_candidates_for_user(
        user=user,
        latest_limit=10,
        trending_limit=10,
        per_interest_limit=5,
        per_project_limit=5,
    )
    print("\n候选集策略: latest + trending + followed_projects + interests")
    print(f"拉取到 {len(items)} 条候选内容")
    print(f"聚合成 {len(service.cluster_same_event(items))} 个事件簇")

    print_divider("BASELINE VS PERSONALIZED")
    print_baseline_vs_personalized(service, user, items, max_items=5)

    print_divider("ACCEPTANCE SUMMARY")
    print_acceptance_summary(service, user, items, top_k=5)

    print_divider("WHY THESE ITEMS RANK HIGH")
    print_recommendation_table(service, user, items, max_items=8)

    print_divider("TOP 5 DIGEST PREVIEW")
    print(service.build_personal_digest(user=user, items=items, max_items=5))


def main() -> None:
    try:
        load_dotenv()
        api_key = safe_get_api_key()

        adapter = FollowinAPIAdapter(
            api_key=api_key,
            lang="zh-Hans",
            timeout=15,
        )
        service = FollowinMCPService(adapter)
        user = build_demo_user()

        run_recommendation_debug_demo(service, user)

    except FollowinAPIError as e:
        print(f"Followin API 返回错误: {e}")
    except Exception as e:
        print(f"运行失败: {e}")


if __name__ == "__main__":
    main()
