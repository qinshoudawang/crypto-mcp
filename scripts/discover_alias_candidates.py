from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Literal, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from followin_mcp.core.adapters import FollowinAPIAdapter
from followin_mcp.core.normalizer import ContentNormalizer


CandidateCategory = Literal["project", "token", "topic", "event_keyword"]


class CandidateMention(BaseModel):
    category: CandidateCategory = Field(description="Candidate type.")
    canonical_name: str = Field(description="Normalized name for the candidate.")
    aliases: List[str] = Field(
        default_factory=list,
        description="Observed aliases or surface forms for this candidate.",
    )
    evidence_title: str = Field(description="One title from the current batch supporting this candidate.")
    evidence_source: str = Field(
        default="unknown",
        description="Source name for the supporting document.",
    )


class CandidateBatch(BaseModel):
    candidates: List[CandidateMention] = Field(default_factory=list)


class AggregatedCandidate(BaseModel):
    category: CandidateCategory
    canonical_name: str
    mention_count: int
    doc_count: int
    aliases: List[str]
    evidence_titles: List[str]
    source_names: List[str]


class CandidateReviewDecision(BaseModel):
    source_id: str = Field(description="Source candidate id in the form category::canonical_name.")
    action: Literal["keep", "merge", "drop", "merge_into_builtin", "merge_into_dynamic"] = Field(
        description="Whether to keep, merge, drop, or merge into an existing builtin/dynamic canonical."
    )
    final_category: CandidateCategory | None = Field(
        default=None,
        description="Final category when action is keep or merge.",
    )
    final_canonical_name: str | None = Field(
        default=None,
        description="Final canonical name when action is keep or merge.",
    )
    final_aliases: List[str] = Field(
        default_factory=list,
        description="Optional cleaned aliases to retain for the final candidate.",
    )
    builtin_category: CandidateCategory | None = Field(
        default=None,
        description="Builtin category when action is merge_into_builtin.",
    )
    builtin_canonical_name: str | None = Field(
        default=None,
        description="Builtin canonical name when action is merge_into_builtin.",
    )
    dynamic_category: CandidateCategory | None = Field(
        default=None,
        description="Dynamic reference category when action is merge_into_dynamic.",
    )
    dynamic_canonical_name: str | None = Field(
        default=None,
        description="Dynamic reference canonical name when action is merge_into_dynamic.",
    )
    reason: str = Field(default="", description="Short reason for the decision.")


class CandidateReviewBatch(BaseModel):
    decisions: List[CandidateReviewDecision] = Field(default_factory=list)


DEFAULT_SQLITE_PATH = "data/candidate_discovery/discovery.db"

GENERIC_NOISE_PATTERNS: Dict[CandidateCategory, tuple[str, ...]] = {
    "project": (
        r"\b(skill|founder|ceo|cto|kol)\b",
        r"(联合创始人|联创|创始人|同事)",
    ),
    "token": (
        r"(多单|空单|交易对|trading pair)",
    ),
    "topic": (
        r"\bskill\b",
        r"(联合创始人|联创|创始人|同事)",
        r"(多单|空单|清算强度|清算图|爆仓图)",
    ),
    "event_keyword": (
        r"\bskill\b",
        r"(赛季|第[一二三四五六七八九十0-9]+期|行动)",
    ),
}

PROJECT_SUFFIX_PATTERNS: tuple[str, ...] = (
    r"\s*平台$",
    r"\s*账户$",
    r"\s*account$",
    r"\s*trading$",
)

EVENT_NORMALIZATION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bhack incident\b", "hack"),
    (r"\bincident\b", ""),
    (r"\s+事件$", ""),
)


def normalize_proxy_env() -> None:
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


def load_raw_documents(adapter: FollowinAPIAdapter, latest_limit: int, trending_limit: int) -> List[Dict]:
    docs: List[Dict] = []
    docs.extend(adapter.get_latest_headlines(limit=latest_limit))
    docs.extend(adapter.get_trending_feeds(feed_type="hot_news", limit=trending_limit))

    deduped: List[Dict] = []
    seen_ids = set()
    for doc in docs:
        key = str(doc.get("id") or doc.get("title") or doc.get("translated_title") or "")
        if not key or key in seen_ids:
            continue
        seen_ids.add(key)
        deduped.append(doc)
    return deduped


def chunked(items: List[Dict], size: int) -> List[List[Dict]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def build_existing_vocab() -> Dict[CandidateCategory, set[str]]:
    project_vocab = {
        name.lower().strip()
        for name, aliases in ContentNormalizer.ENTITY_ALIASES.items()
        for name in [name, *aliases]
        if name.strip()
    }
    token_vocab = {
        name.lower().strip()
        for symbol, aliases in ContentNormalizer.TOKEN_ALIASES.items()
        for name in [symbol, *aliases]
        if name.strip()
    }
    topic_vocab = {
        name.lower().strip()
        for topic, aliases in ContentNormalizer.TOPIC_ALIASES.items()
        for name in [topic, *aliases]
        if name.strip()
    }
    return {
        "project": project_vocab,
        "token": token_vocab,
        "topic": topic_vocab,
        "event_keyword": {
            "exploit",
            "governance",
            "listing",
            "funding",
            "token_unlock",
            "airdrop",
            "partnership",
            "merger",
            "incentive_program",
            "market_structure",
            "institutional_adoption",
            "product_launch",
            "macro",
        },
    }


def build_builtin_reference() -> List[Dict[str, object]]:
    references: List[Dict[str, object]] = []
    for canonical_name, aliases in ContentNormalizer.ENTITY_ALIASES.items():
        references.append(
            {
                "category": "project",
                "canonical_name": canonical_name,
                "aliases": aliases,
            }
        )
    for canonical_name, aliases in ContentNormalizer.TOKEN_ALIASES.items():
        references.append(
            {
                "category": "token",
                "canonical_name": canonical_name,
                "aliases": aliases,
            }
        )
    for canonical_name, aliases in ContentNormalizer.TOPIC_ALIASES.items():
        references.append(
            {
                "category": "topic",
                "canonical_name": canonical_name,
                "aliases": aliases,
            }
        )
    return references


def load_dynamic_reference(sqlite_path: str) -> List[Dict[str, object]]:
    db_path = Path(sqlite_path)
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT category, canonical_name, aliases_json, status
            FROM promoted_candidates
            WHERE status IN ('active', 'cooling')
            ORDER BY category, canonical_name
            """
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    references: List[Dict[str, object]] = []
    for row in rows:
        references.append(
            {
                "category": str(row["category"]),
                "canonical_name": str(row["canonical_name"]),
                "aliases": json.loads(row["aliases_json"]),
                "status": str(row["status"]),
            }
        )
    return references


def normalize_candidate_text(value: str) -> str:
    cleaned = value.strip().strip("\"'`")
    cleaned = cleaned.replace("：", ":")
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def to_canonical_key(value: str) -> str:
    normalized = normalize_candidate_text(value).lower()
    normalized = normalized.replace(" & ", " and ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def format_display_name(category: CandidateCategory, canonical_key: str, fallback: str) -> str:
    if category == "token":
        return fallback or canonical_key.upper()
    if fallback:
        return fallback
    return canonical_key.title()


def normalize_review_display_name(category: CandidateCategory, canonical_name: str) -> str:
    cleaned = normalize_candidate_text(canonical_name)
    if not cleaned:
        return cleaned

    if category == "token":
        return cleaned.upper()

    if re.search(r"[A-Za-z]", cleaned) and cleaned == cleaned.lower():
        words = []
        for word in cleaned.split():
            if word in {"and", "or", "of", "the", "to"}:
                words.append(word)
            elif word.upper() in {"ETF", "FUD", "TVL", "MVRV", "ETH", "BTC", "USDT"}:
                words.append(word.upper())
            else:
                words.append(word.capitalize())
        if words:
            words[0] = words[0][0].upper() + words[0][1:] if words[0] else words[0]
        cleaned = " ".join(words)

    return cleaned


def should_filter_candidate(
    category: CandidateCategory,
    canonical_key: str,
    aliases: List[str],
) -> bool:
    if not canonical_key:
        return True

    combined = " ".join([canonical_key, *[to_canonical_key(alias) for alias in aliases]])
    for pattern in GENERIC_NOISE_PATTERNS.get(category, ()):
        if re.search(pattern, combined, flags=re.IGNORECASE):
            return True

    if category == "event_keyword" and re.search(r"\b(pre-?tge|s[0-9]+)\b", combined, flags=re.IGNORECASE):
        return True

    return False


def normalize_project_key(canonical_key: str) -> str:
    normalized = canonical_key
    for pattern in PROJECT_SUFFIX_PATTERNS:
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)
    return normalized.strip()


def normalize_topic_key(canonical_key: str) -> str:
    return re.sub(r"\s+", " ", canonical_key).strip()


def normalize_event_key(canonical_key: str) -> str:
    normalized = canonical_key
    for pattern, replacement in EVENT_NORMALIZATION_PATTERNS:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def normalize_canonical_key(category: CandidateCategory, canonical_key: str) -> str:
    if category == "project":
        return normalize_project_key(canonical_key)
    if category == "topic":
        return normalize_topic_key(canonical_key)
    if category == "event_keyword":
        return normalize_event_key(canonical_key)
    return canonical_key


def sanitize_candidate(
    candidate: CandidateMention,
) -> Tuple[CandidateCategory, str, str, List[str]] | None:
    category = candidate.category
    original_canonical = normalize_candidate_text(candidate.canonical_name)
    original_aliases = [normalize_candidate_text(alias) for alias in candidate.aliases if normalize_candidate_text(alias)]
    raw_aliases = list(dict.fromkeys([original_canonical, *original_aliases]))

    canonical_key = to_canonical_key(original_canonical)
    canonical_key = normalize_canonical_key(category, canonical_key)

    normalized_aliases: List[str] = []
    for alias in raw_aliases:
        alias_key = to_canonical_key(alias)
        alias_key = normalize_canonical_key(category, alias_key)
        if alias_key == canonical_key:
            continue
        normalized_aliases.append(alias)

    if should_filter_candidate(category, canonical_key, normalized_aliases):
        return None

    display_fallback = original_canonical if to_canonical_key(original_canonical) == canonical_key else ""
    display_name = format_display_name(category, canonical_key, display_fallback)
    deduped_aliases = list(dict.fromkeys(alias for alias in normalized_aliases if alias))
    return category, canonical_key, display_name, deduped_aliases


def format_batch_documents(batch: List[Dict]) -> str:
    lines: List[str] = []
    for idx, doc in enumerate(batch, start=1):
        title = doc.get("translated_title") or doc.get("title") or ""
        content = (
            doc.get("translated_content")
            or doc.get("content")
            or doc.get("translated_full_content")
            or doc.get("full_content")
            or ""
        )
        lines.append(f"[DOC {idx}]")
        lines.append(f"title: {title}")
        lines.append(f"source: {doc.get('source_title') or doc.get('source_name') or 'unknown'}")
        lines.append(f"content: {content[:500]}")
        lines.append("")
    return "\n".join(lines)


def build_extractor(model: str, api_key: str, base_url: str | None):
    client_kwargs = {"model": model, "api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    llm = ChatOpenAI(**client_kwargs)
    return llm.with_structured_output(CandidateBatch)


def build_reviewer(model: str, api_key: str, base_url: str | None):
    client_kwargs = {"model": model, "api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    llm = ChatOpenAI(**client_kwargs)
    return llm.with_structured_output(CandidateReviewBatch)


def aggregate_candidates(
    extracted_batches: List[CandidateBatch],
    existing_vocab: Dict[CandidateCategory, set[str]],
    min_mentions: int,
    min_docs: int,
) -> Tuple[List[AggregatedCandidate], Dict[str, object]]:
    mention_counter: Counter[Tuple[CandidateCategory, str]] = Counter()
    alias_counter: Dict[Tuple[CandidateCategory, str], Counter[str]] = defaultdict(Counter)
    display_name_counter: Dict[Tuple[CandidateCategory, str], Counter[str]] = defaultdict(Counter)
    doc_titles: Dict[Tuple[CandidateCategory, str], set[str]] = defaultdict(set)
    source_names: Dict[Tuple[CandidateCategory, str], set[str]] = defaultdict(set)
    filtered_existing: Counter[Tuple[CandidateCategory, str]] = Counter()
    filtered_cleaning: Counter[Tuple[CandidateCategory, str]] = Counter()
    skipped_empty = 0

    for batch in extracted_batches:
        for candidate in batch.candidates:
            sanitized = sanitize_candidate(candidate)
            if sanitized is None:
                category = candidate.category
                normalized_key = to_canonical_key(candidate.canonical_name)
                filtered_cleaning[(category, normalized_key)] += 1
                continue

            category, normalized_key, display_name, cleaned_aliases = sanitized
            if not normalized_key:
                skipped_empty += 1
                continue

            if normalized_key in existing_vocab.get(category, set()):
                filtered_existing[(category, normalized_key)] += 1
                continue

            key = (category, normalized_key)
            mention_counter[key] += 1
            display_name_counter[key][display_name] += 1
            doc_titles[key].add(candidate.evidence_title.strip())
            if candidate.evidence_source.strip():
                source_names[key].add(candidate.evidence_source.strip())

            for alias in cleaned_aliases:
                if to_canonical_key(alias) == normalized_key:
                    continue
                alias_counter[key][alias] += 1

    results: List[AggregatedCandidate] = []
    filtered_by_threshold: List[Dict[str, object]] = []
    for (category, normalized_key), mention_count in mention_counter.items():
        doc_count = len(doc_titles[(category, normalized_key)])
        if mention_count < min_mentions or doc_count < min_docs:
            filtered_by_threshold.append(
                {
                    "category": category,
                    "canonical_name": normalized_key,
                    "mention_count": mention_count,
                    "doc_count": doc_count,
                }
            )
            continue

        aliases = [
            alias
            for alias, _ in alias_counter[(category, normalized_key)].most_common()
            if alias.lower() != normalized_key
        ]
        evidence_titles = sorted(doc_titles[(category, normalized_key)])[:5]
        canonical_name = display_name_counter[(category, normalized_key)].most_common(1)[0][0]
        results.append(
            AggregatedCandidate(
                category=category,
                canonical_name=canonical_name,
                mention_count=mention_count,
                doc_count=doc_count,
                aliases=aliases[:8],
                evidence_titles=evidence_titles,
                source_names=sorted(source_names[(category, normalized_key)])[:8],
            )
        )

    results.sort(key=lambda item: (item.doc_count, item.mention_count), reverse=True)
    debug_info: Dict[str, object] = {
        "raw_candidate_count": sum(len(batch.candidates) for batch in extracted_batches),
        "unique_candidate_count": len(mention_counter),
        "skipped_empty_count": skipped_empty,
        "filtered_existing_count": sum(filtered_existing.values()),
        "filtered_cleaning_count": sum(filtered_cleaning.values()),
        "filtered_existing_examples": [
            {
                "category": category,
                "canonical_name": canonical_name,
                "count": count,
            }
            for (category, canonical_name), count in filtered_existing.most_common(10)
        ],
        "filtered_cleaning_examples": [
            {
                "category": category,
                "canonical_name": canonical_name,
                "count": count,
            }
            for (category, canonical_name), count in filtered_cleaning.most_common(10)
        ],
        "filtered_threshold_count": len(filtered_by_threshold),
        "filtered_threshold_examples": sorted(
            filtered_by_threshold,
            key=lambda item: (int(item["doc_count"]), int(item["mention_count"])),
            reverse=True,
        )[:10],
    }
    return results, debug_info


def build_batch_debug_summary(
    extracted_batches: List[CandidateBatch],
) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for idx, batch in enumerate(extracted_batches, start=1):
        category_counts: Counter[str] = Counter(candidate.category for candidate in batch.candidates)
        summaries.append(
            {
                "batch_index": idx,
                "candidate_count": len(batch.candidates),
                "category_counts": dict(category_counts),
                "examples": [
                    {
                        "category": candidate.category,
                        "canonical_name": candidate.canonical_name,
                        "aliases": candidate.aliases[:4],
                        "evidence_title": candidate.evidence_title,
                        "evidence_source": candidate.evidence_source,
                    }
                    for candidate in batch.candidates[:8]
                ],
            }
        )
    return summaries


def build_candidate_source_id(candidate: AggregatedCandidate) -> str:
    return f"{candidate.category}::{candidate.canonical_name}"


def apply_review_decisions(
    candidates: List[AggregatedCandidate],
    review_batch: CandidateReviewBatch,
) -> Tuple[List[AggregatedCandidate], Dict[str, object]]:
    decision_map = {
        decision.source_id: decision
        for decision in review_batch.decisions
        if decision.source_id
    }

    merged: Dict[Tuple[CandidateCategory, str], Dict[str, object]] = {}
    default_kept = 0
    kept = 0
    dropped = 0
    merged_count = 0
    builtin_merged = 0
    dynamic_merged = 0

    for candidate in candidates:
        source_id = build_candidate_source_id(candidate)
        decision = decision_map.get(source_id)

        action = decision.action if decision else "keep"
        if action == "drop":
            dropped += 1
            continue
        if action == "merge_into_builtin":
            builtin_merged += 1
            continue
        if action == "merge_into_dynamic":
            dynamic_merged += 1
            continue

        final_category = (decision.final_category if decision else candidate.category) or candidate.category
        final_canonical_name = (
            normalize_candidate_text(decision.final_canonical_name or "")
            if decision
            else candidate.canonical_name
        )
        if not final_canonical_name:
            final_canonical_name = candidate.canonical_name
        final_canonical_name = normalize_review_display_name(final_category, final_canonical_name)
        final_key = (final_category, to_canonical_key(final_canonical_name))

        bucket = merged.setdefault(
            final_key,
            {
                "category": final_category,
                "display_names": Counter(),
                "mention_count": 0,
                "doc_count": 0,
                "aliases": Counter(),
                "evidence_titles": set(),
                "source_names": set(),
            },
        )
        bucket["display_names"][final_canonical_name] += 1
        bucket["mention_count"] += candidate.mention_count
        bucket["doc_count"] = max(int(bucket["doc_count"]), candidate.doc_count)
        bucket["evidence_titles"].update(candidate.evidence_titles)
        bucket["source_names"].update(candidate.source_names)

        candidate_aliases = [*candidate.aliases]
        if decision and decision.final_aliases:
            candidate_aliases.extend(normalize_candidate_text(alias) for alias in decision.final_aliases if normalize_candidate_text(alias))
        for alias in [candidate.canonical_name, *candidate_aliases]:
            cleaned_alias = normalize_candidate_text(alias)
            if not cleaned_alias:
                continue
            if to_canonical_key(cleaned_alias) == final_key[1]:
                continue
            bucket["aliases"][cleaned_alias] += 1

        if decision:
            if action == "merge":
                merged_count += 1
            else:
                kept += 1
        else:
            default_kept += 1

    reviewed: List[AggregatedCandidate] = []
    for (_, _), bucket in merged.items():
        canonical_name = normalize_review_display_name(
            bucket["category"],  # type: ignore[arg-type]
            bucket["display_names"].most_common(1)[0][0],
        )
        reviewed.append(
            AggregatedCandidate(
                category=bucket["category"],  # type: ignore[arg-type]
                canonical_name=canonical_name,
                mention_count=int(bucket["mention_count"]),
                doc_count=max(int(bucket["doc_count"]), len(bucket["evidence_titles"])),
                aliases=[alias for alias, _ in bucket["aliases"].most_common(8)],
                evidence_titles=sorted(bucket["evidence_titles"])[:5],
                source_names=sorted(bucket["source_names"])[:8],
            )
        )

    reviewed.sort(key=lambda item: (item.doc_count, item.mention_count), reverse=True)
    return reviewed, {
        "input_candidate_count": len(candidates),
        "reviewed_candidate_count": len(reviewed),
        "decision_count": len(review_batch.decisions),
        "dropped_count": dropped,
        "keep_decision_count": kept,
        "merge_decision_count": merged_count,
        "merge_into_builtin_count": builtin_merged,
        "merge_into_dynamic_count": dynamic_merged,
        "default_keep_count": default_kept,
    }


def ensure_sqlite_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS discovery_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            model TEXT NOT NULL,
            latest_limit INTEGER NOT NULL,
            trending_limit INTEGER NOT NULL,
            batch_size INTEGER NOT NULL,
            min_mentions INTEGER NOT NULL,
            min_docs INTEGER NOT NULL,
            doc_count INTEGER NOT NULL,
            candidate_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS discovery_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            mention_count INTEGER NOT NULL,
            doc_count INTEGER NOT NULL,
            aliases_json TEXT NOT NULL,
            evidence_titles_json TEXT NOT NULL,
            source_names_json TEXT NOT NULL DEFAULT '[]',
            FOREIGN KEY(run_id) REFERENCES discovery_runs(id)
        );

        CREATE INDEX IF NOT EXISTS idx_discovery_candidates_run_id
        ON discovery_candidates(run_id);

        CREATE INDEX IF NOT EXISTS idx_discovery_candidates_lookup
        ON discovery_candidates(category, canonical_name);
        """
    )
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(discovery_candidates)")
    }
    if "source_names_json" not in columns:
        conn.execute(
            "ALTER TABLE discovery_candidates ADD COLUMN source_names_json TEXT NOT NULL DEFAULT '[]'"
        )
    conn.commit()


def persist_run_to_sqlite(
    sqlite_path: str,
    *,
    model: str,
    latest_limit: int,
    trending_limit: int,
    batch_size: int,
    min_mentions: int,
    min_docs: int,
    doc_count: int,
    results: List[AggregatedCandidate],
) -> int:
    db_path = Path(sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        ensure_sqlite_schema(conn)
        cursor = conn.execute(
            """
            INSERT INTO discovery_runs (
                created_at, model, latest_limit, trending_limit, batch_size,
                min_mentions, min_docs, doc_count, candidate_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                model,
                latest_limit,
                trending_limit,
                batch_size,
                min_mentions,
                min_docs,
                doc_count,
                len(results),
            ),
        )
        run_id = int(cursor.lastrowid)

        for item in results:
            conn.execute(
                """
                INSERT INTO discovery_candidates (
                    run_id, category, canonical_name, mention_count, doc_count,
                    aliases_json, evidence_titles_json, source_names_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    item.category,
                    item.canonical_name,
                    item.mention_count,
                    item.doc_count,
                    json.dumps(item.aliases, ensure_ascii=False),
                    json.dumps(item.evidence_titles, ensure_ascii=False),
                    json.dumps(item.source_names, ensure_ascii=False),
                ),
            )
        conn.commit()
        return run_id
    finally:
        conn.close()


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Discover alias/topic candidates with AI and frequency filters.")
    parser.add_argument("--latest-limit", type=int, default=40, help="Number of latest headlines to inspect.")
    parser.add_argument("--trending-limit", type=int, default=40, help="Number of trending feeds to inspect.")
    parser.add_argument("--batch-size", type=int, default=12, help="Documents per LLM extraction batch.")
    parser.add_argument("--min-mentions", type=int, default=2, help="Minimum raw mentions to keep a candidate.")
    parser.add_argument("--min-docs", type=int, default=2, help="Minimum distinct documents to keep a candidate.")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Enable cold-start mode by lowering discovery thresholds to 1 mention and 1 document.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        help="OpenAI-compatible model for candidate extraction.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--sqlite-path",
        default=DEFAULT_SQLITE_PATH,
        help="SQLite database path for persisting discovery runs.",
    )
    parser.add_argument(
        "--no-sqlite",
        action="store_true",
        help="Disable SQLite persistence and only print or write JSON output.",
    )
    parser.add_argument(
        "--debug-output",
        default="",
        help="Optional JSON path for detailed extraction debug output.",
    )
    args = parser.parse_args()

    if args.bootstrap:
        args.min_mentions = 1
        args.min_docs = 1

    normalize_proxy_env()
    followin_api_key = os.getenv("FOLLOWIN_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")

    if not followin_api_key:
        raise RuntimeError("FOLLOWIN_API_KEY is required.")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    adapter = FollowinAPIAdapter(
        api_key=followin_api_key,
        lang=os.getenv("FOLLOWIN_LANG", "zh-Hans"),
        timeout=int(os.getenv("FOLLOWIN_TIMEOUT", "15")),
    )
    docs = load_raw_documents(adapter, latest_limit=args.latest_limit, trending_limit=args.trending_limit)
    batches = chunked(docs, args.batch_size)
    extractor = build_extractor(model=args.model, api_key=openai_api_key, base_url=openai_base_url)
    reviewer = build_reviewer(model=args.model, api_key=openai_api_key, base_url=openai_base_url)
    existing_vocab = build_existing_vocab()
    builtin_reference = build_builtin_reference()
    dynamic_reference = load_dynamic_reference(args.sqlite_path)

    extracted_batches: List[CandidateBatch] = []
    system_prompt = (
        "You are extracting NEW crypto vocabulary candidates from news documents for a dynamic taxonomy. "
        "Return only candidates that are likely to become reusable normalization entries rather than one-off title phrases.\n\n"
        "<allowed_categories>\n"
        "1. project: crypto-native protocol, exchange, wallet, app, company, product, or platform name.\n"
        "2. token: token symbol or token name.\n"
        "3. topic: reusable crypto narrative or taxonomy concept.\n"
        "4. event_keyword: reusable event class or trigger phrase.\n"
        "</allowed_categories>\n\n"
        "<core_rules>\n"
        "- Prefer stable canonical names that could be reused across many future articles.\n"
        "- If a phrase is just a decorated form of a base entity, return the base entity as canonical_name and put the decorated form in aliases.\n"
        "- Normalize minor wording variants to one canonical concept instead of inventing multiple near-duplicates.\n"
        "- Use the title field exactly as evidence_title for each candidate.\n"
        "- Use the provided source field exactly as evidence_source for each candidate.\n"
        "</core_rules>\n\n"
        "<do_not_return>\n"
        "- People names, founder titles, KOL nicknames, job titles, or person-role phrases.\n"
        "- Government campaigns, law-enforcement actions, geopolitics, macro policy actions, or non-crypto public affairs.\n"
        "- One-off campaign names, season names, promotional activity names, countdown events, or temporary marketing labels.\n"
        "- Headline-specific numeric phrases, liquidation levels, market snapshots, or trading-position descriptions.\n"
        "- Generic words like market, update, platform, account, event, index, liquidation, whale position, staking, ETF, hack, or token unless the phrase is a stable crypto taxonomy concept.\n"
        "- Variants that differ only by spacing, punctuation, capitalization, or language mixing when they clearly refer to the same thing.\n"
        "</do_not_return>\n\n"
        "<canonicalization_examples>\n"
        "- Return 'Hyperliquid' instead of 'Hyperliquid 平台'.\n"
        "- Return 'Schwab Crypto' instead of 'Schwab Crypto账户' or 'Schwab Crypto 账户'.\n"
        "- Return one canonical 'Crypto Fear & Greed Index' concept instead of multiple close English/Chinese variants.\n"
        "- Do not return phrases like 'ETH多单', '主流CEX清算强度', '同事.skill', '天网 2026', or '猎狐专项行动'.\n"
        "- Do not return branded one-off campaign names like temporary launch seasons or promotional rounds unless they are clearly reusable event classes.\n"
        "</canonicalization_examples>\n\n"
        "<topic_guidance>\n"
        "- Topics must be reusable taxonomy concepts such as a stable narrative, mechanism, or market structure concept.\n"
        "- Reject title-fragment concepts that only make sense inside a single article.\n"
        "</topic_guidance>\n\n"
        "<event_guidance>\n"
        "- event_keyword should be reusable event classes like listing, exploit, airdrop, token sale, delisting, unlock, governance vote, treasury purchase.\n"
        "- Reject one-off branded events, campaign names, season names, and incident titles that are too specific.\n"
        "</event_guidance>"
    )

    for batch in batches:
        batch_text = format_batch_documents(batch)
        extracted = extractor.invoke(
            [
                ("system", system_prompt),
                (
                    "human",
                    "Extract candidate entities from these documents.\n"
                    "Only include candidates that appear meaningful for a crypto content taxonomy.\n\n"
                    f"{batch_text}",
                ),
            ]
        )
        extracted_batches.append(extracted)

    results, debug_info = aggregate_candidates(
        extracted_batches=extracted_batches,
        existing_vocab=existing_vocab,
        min_mentions=args.min_mentions,
        min_docs=args.min_docs,
    )
    review_summary: Dict[str, object] = {
        "input_candidate_count": len(results),
        "reviewed_candidate_count": len(results),
        "decision_count": 0,
        "dropped_count": 0,
        "keep_decision_count": 0,
        "merge_decision_count": 0,
        "merge_into_builtin_count": 0,
        "merge_into_dynamic_count": 0,
        "default_keep_count": len(results),
    }
    if results:
        review_input = {
            "meta": {
                "candidate_count": len(results),
                "categories": sorted({item.category for item in results}),
                "builtin_reference_count": len(builtin_reference),
                "dynamic_reference_count": len(dynamic_reference),
            },
            "builtin_reference": builtin_reference,
            "dynamic_reference": dynamic_reference,
            "candidates": [
                {
                    "source_id": build_candidate_source_id(item),
                    "category": item.category,
                    "canonical_name": item.canonical_name,
                    "mention_count": item.mention_count,
                    "doc_count": item.doc_count,
                    "aliases": item.aliases,
                    "evidence_titles": item.evidence_titles,
                    "source_names": item.source_names,
                }
                for item in results
            ],
        }
        review_prompt = (
            "You are reviewing candidate taxonomy entries extracted from crypto news. "
            "Your job is to clean the candidate list by deciding whether to keep, merge, drop, or merge into builtin/dynamic canonical entries.\n\n"
            "<review_rules>\n"
            "- Keep only candidates that look useful for a reusable crypto normalization dictionary.\n"
            "- Be conservative: when unsure, prefer drop over keep.\n"
            "- Drop generic concepts, obvious noise, title fragments, person-role phrases, and one-off event slogans.\n"
            "- Merge near-duplicates that differ only by wording, casing, punctuation, suffixes, or language variants.\n"
            "- If a candidate matches an existing concept in builtin_reference, prefer action=merge_into_builtin instead of keep.\n"
            "- If a candidate matches an existing concept in dynamic_reference, prefer action=merge_into_dynamic instead of keep.\n"
            "- When merging, set final_canonical_name to the single best reusable canonical form.\n"
            "- Prefer stable entity names over decorated forms.\n"
            "- Prefer stable taxonomy labels over article-specific wording.\n"
            "- For event_keyword, do not keep over-generic bare canonicals such as single words like hack, launch, suspension, attack, incident, or event unless the label is clearly a stable reusable event class.\n"
            "- For event_keyword, prefer specific reusable classes like flash loan attack, market delisting, liquidation, token sale, exploit, or governance vote over vague event words.\n"
            "- Drop concepts that are too broad for normalization, such as generic finance terms, macro indicators, generic market activity, or broad geography labels.\n"
            "- Drop labels that mainly describe article context rather than reusable domain vocabulary.\n"
            "- Drop aliases that are only sentence fragments, temporary campaign labels, or trading-state descriptions.\n"
            "- final_aliases should only include aliases worth keeping in the final dictionary.\n"
            "- If a candidate is already good, use action=keep and copy its category/canonical_name.\n"
            "- If a candidate should disappear entirely, use action=drop.\n"
            "- If action=merge_into_builtin, fill builtin_category and builtin_canonical_name and leave final_category/final_canonical_name empty.\n"
            "- If action=merge_into_dynamic, fill dynamic_category and dynamic_canonical_name and leave final_category/final_canonical_name empty.\n"
            "</review_rules>\n\n"
            "<examples_of_likely_drop>\n"
            "- Generic context labels like direct trading, suspension, market shakeout, bot-driven activity, geography-only market labels.\n"
            "- Temporary campaign or season names.\n"
            "- Decorated business phrases when the underlying entity already captures the concept.\n"
            "- Bare event words like hack, launch, event, or incident when they are not normalized into a more specific event class.\n"
            "</examples_of_likely_drop>\n\n"
            "<important>\n"
            "- You must return one decision per input source_id.\n"
            "- Use source_id exactly as provided.\n"
            "- Do not invent extra source_ids.\n"
            "- Prefer merge_into_builtin over keep when builtin_reference already covers the concept under a different string.\n"
            "- Prefer merge_into_dynamic over keep when dynamic_reference already covers the concept under a different string.\n"
            "</important>"
        )
        review_result = reviewer.invoke(
            [
                ("system", review_prompt),
                (
                    "human",
                    "Review this candidate JSON and return cleanup decisions.\n\n"
                    f"{json.dumps(review_input, ensure_ascii=False, indent=2)}",
                ),
            ]
        )
        results, review_summary = apply_review_decisions(results, review_result)
    batch_debug = build_batch_debug_summary(extracted_batches)

    payload = {
        "meta": {
            "latest_limit": args.latest_limit,
            "trending_limit": args.trending_limit,
            "batch_size": args.batch_size,
            "bootstrap": args.bootstrap,
            "min_mentions": args.min_mentions,
            "min_docs": args.min_docs,
            "model": args.model,
            "doc_count": len(docs),
            "candidate_count": len(results),
            "reviewed_by_llm": True,
            "review_summary": review_summary,
        },
        "candidates": [item.model_dump() for item in results],
    }
    debug_payload = {
        **payload,
        "debug": {
            **debug_info,
            "review_summary": review_summary,
            "batch_summaries": batch_debug,
        },
    }

    if not args.no_sqlite:
        run_id = persist_run_to_sqlite(
            args.sqlite_path,
            model=args.model,
            latest_limit=args.latest_limit,
            trending_limit=args.trending_limit,
            batch_size=args.batch_size,
            min_mentions=args.min_mentions,
            min_docs=args.min_docs,
            doc_count=len(docs),
            results=results,
        )
        payload["meta"]["sqlite_path"] = args.sqlite_path
        payload["meta"]["run_id"] = run_id

    output_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.debug_output:
        debug_path = Path(args.debug_output)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(
            json.dumps(debug_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote discovery debug output to {debug_path}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Wrote candidate suggestions to {args.output}")
        return
    print(output_text)


if __name__ == "__main__":
    main()
