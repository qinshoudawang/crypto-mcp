from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from pydantic import BaseModel


CandidateCategory = Literal["project", "token", "topic", "event_keyword"]
DEFAULT_SQLITE_PATH = "data/candidate_discovery/discovery.db"


class PromotedCandidate(BaseModel):
    category: CandidateCategory
    canonical_name: str
    status: str
    run_hits: int
    decayed_run_score: float
    total_mentions: int
    max_doc_count: int
    source_diversity: int
    aliases: List[str]
    evidence_titles: List[str]
    source_names: List[str]
    last_seen_run_id: int
    last_qualified_run_id: int
    stale_cycles: int = 0
    last_evaluated_run_id: int
    cooling_started_at: str | None = None
    archived_at: str | None = None


def ensure_promotion_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS promoted_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            run_hits INTEGER NOT NULL,
            decayed_run_score REAL NOT NULL DEFAULT 0,
            total_mentions INTEGER NOT NULL,
            max_doc_count INTEGER NOT NULL,
            source_diversity INTEGER NOT NULL DEFAULT 0,
            aliases_json TEXT NOT NULL,
            evidence_titles_json TEXT NOT NULL,
            source_names_json TEXT NOT NULL DEFAULT '[]',
            last_seen_run_id INTEGER NOT NULL DEFAULT 0,
            last_qualified_run_id INTEGER NOT NULL DEFAULT 0,
            stale_cycles INTEGER NOT NULL DEFAULT 0,
            last_evaluated_run_id INTEGER NOT NULL DEFAULT 0,
            cooling_started_at TEXT,
            archived_at TEXT,
            promoted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(category, canonical_name)
        );
        """
    )
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(promoted_candidates)")
    }
    if "decayed_run_score" not in columns:
        conn.execute(
            "ALTER TABLE promoted_candidates ADD COLUMN decayed_run_score REAL NOT NULL DEFAULT 0"
        )
    if "status" not in columns:
        conn.execute(
            "ALTER TABLE promoted_candidates ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"
        )
    if "source_diversity" not in columns:
        conn.execute(
            "ALTER TABLE promoted_candidates ADD COLUMN source_diversity INTEGER NOT NULL DEFAULT 0"
        )
    if "source_names_json" not in columns:
        conn.execute(
            "ALTER TABLE promoted_candidates ADD COLUMN source_names_json TEXT NOT NULL DEFAULT '[]'"
        )
    if "last_seen_run_id" not in columns:
        conn.execute(
            "ALTER TABLE promoted_candidates ADD COLUMN last_seen_run_id INTEGER NOT NULL DEFAULT 0"
        )
    if "last_qualified_run_id" not in columns:
        conn.execute(
            "ALTER TABLE promoted_candidates ADD COLUMN last_qualified_run_id INTEGER NOT NULL DEFAULT 0"
        )
    if "stale_cycles" not in columns:
        conn.execute(
            "ALTER TABLE promoted_candidates ADD COLUMN stale_cycles INTEGER NOT NULL DEFAULT 0"
        )
    if "last_evaluated_run_id" not in columns:
        conn.execute(
            "ALTER TABLE promoted_candidates ADD COLUMN last_evaluated_run_id INTEGER NOT NULL DEFAULT 0"
        )
    if "cooling_started_at" not in columns:
        conn.execute(
            "ALTER TABLE promoted_candidates ADD COLUMN cooling_started_at TEXT"
        )
    if "archived_at" not in columns:
        conn.execute(
            "ALTER TABLE promoted_candidates ADD COLUMN archived_at TEXT"
        )
    conn.commit()


def load_recent_candidates(
    conn: sqlite3.Connection,
    lookback_runs: int,
) -> List[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return list(
        conn.execute(
            """
            SELECT dc.category,
                   dc.canonical_name,
                   dc.mention_count,
                   dc.doc_count,
                   dc.aliases_json,
                   dc.evidence_titles_json,
                   dc.source_names_json,
                   dc.run_id,
                   dr.created_at
            FROM discovery_candidates dc
            JOIN discovery_runs dr
            ON dc.run_id = dr.id
            JOIN (
                SELECT id
                FROM discovery_runs
                ORDER BY id DESC
                LIMIT ?
            ) recent_runs
            ON dc.run_id = recent_runs.id
            """,
            (lookback_runs,),
        )
    )


def load_latest_discovery_run_id(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM discovery_runs").fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0] if row is not None else 0)


def aggregate_promotions(
    rows: List[sqlite3.Row],
    *,
    min_run_hits: int,
    min_total_mentions: int,
    min_max_doc_count: int,
    min_source_diversity: int,
) -> List[PromotedCandidate]:
    run_ids: Dict[Tuple[str, str], set[int]] = defaultdict(set)
    mention_counts: Counter[Tuple[str, str]] = Counter()
    max_doc_count: Dict[Tuple[str, str], int] = defaultdict(int)
    alias_counts: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    title_counts: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    source_counts: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    decayed_scores: Dict[Tuple[str, str], float] = defaultdict(float)
    last_seen_run_ids: Dict[Tuple[str, str], int] = defaultdict(int)

    run_times: List[datetime] = []
    parsed_times: Dict[int, datetime] = {}
    for row in rows:
        parsed = datetime.fromisoformat(row["created_at"])
        parsed_times[int(row["run_id"])] = parsed
        run_times.append(parsed)
    newest_time = max(run_times) if run_times else datetime.now(timezone.utc)

    for row in rows:
        key = (row["category"], row["canonical_name"])
        run_ids[key].add(int(row["run_id"]))
        mention_counts[key] += int(row["mention_count"])
        max_doc_count[key] = max(max_doc_count[key], int(row["doc_count"]))
        last_seen_run_ids[key] = max(last_seen_run_ids[key], int(row["run_id"]))
        run_time = parsed_times[int(row["run_id"])]
        age_hours = max((newest_time - run_time).total_seconds() / 3600, 0.0)
        recency_weight = 1.0 / (1.0 + age_hours / 24.0)
        decayed_scores[key] += recency_weight * max(int(row["doc_count"]), 1)

        for alias in json.loads(row["aliases_json"]):
            alias_counts[key][alias] += 1
        for title in json.loads(row["evidence_titles_json"]):
            title_counts[key][title] += 1
        for source_name in json.loads(row["source_names_json"]):
            source_counts[key][source_name] += 1

    promoted: List[PromotedCandidate] = []
    for (category, canonical_name), hits in run_ids.items():
        run_hit_count = len(hits)
        total_mentions = mention_counts[(category, canonical_name)]
        best_doc_count = max_doc_count[(category, canonical_name)]
        source_diversity = len(source_counts[(category, canonical_name)])
        decayed_run_score = round(decayed_scores[(category, canonical_name)], 4)

        if run_hit_count < min_run_hits:
            continue
        if total_mentions < min_total_mentions:
            continue
        if best_doc_count < min_max_doc_count:
            continue
        if source_diversity < min_source_diversity:
            continue

        aliases = [alias for alias, _ in alias_counts[(category, canonical_name)].most_common(8)]
        titles = [title for title, _ in title_counts[(category, canonical_name)].most_common(5)]
        sources = [source for source, _ in source_counts[(category, canonical_name)].most_common(8)]
        promoted.append(
            PromotedCandidate(
                category=category,  # type: ignore[arg-type]
                canonical_name=canonical_name,
                status="active",
                run_hits=run_hit_count,
                decayed_run_score=decayed_run_score,
                total_mentions=total_mentions,
                max_doc_count=best_doc_count,
                source_diversity=source_diversity,
                aliases=aliases,
                evidence_titles=titles,
                source_names=sources,
                last_seen_run_id=last_seen_run_ids[(category, canonical_name)],
                last_qualified_run_id=last_seen_run_ids[(category, canonical_name)],
                stale_cycles=0,
                last_evaluated_run_id=last_seen_run_ids[(category, canonical_name)],
                cooling_started_at=None,
                archived_at=None,
            )
        )

    promoted.sort(
        key=lambda item: (
            item.decayed_run_score,
            item.run_hits,
            item.source_diversity,
            item.total_mentions,
            item.max_doc_count,
        ),
        reverse=True,
    )
    return promoted


def load_existing_promotions(conn: sqlite3.Connection) -> Dict[Tuple[str, str], sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT *
        FROM promoted_candidates
        """
    )
    return {
        (str(row["category"]), str(row["canonical_name"])): row
        for row in rows
    }


def build_latest_seen_run_map(rows: List[sqlite3.Row]) -> Dict[Tuple[str, str], int]:
    latest_seen: Dict[Tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["category"]), str(row["canonical_name"]))
        latest_seen[key] = max(latest_seen.get(key, 0), int(row["run_id"]))
    return latest_seen


def transition_candidate_state(
    existing_row: sqlite3.Row,
    *,
    current_run_id: int,
    now_iso: str,
    cooling_cycles: int,
    last_seen_run_id: int,
) -> PromotedCandidate:
    existing_status = str(existing_row["status"] or "active")
    existing_stale_cycles = int(existing_row["stale_cycles"] or 0)
    last_evaluated_run_id = int(existing_row["last_evaluated_run_id"] or 0)
    next_stale_cycles = existing_stale_cycles
    if current_run_id > last_evaluated_run_id:
        next_stale_cycles += 1

    next_status = existing_status
    cooling_started_at = existing_row["cooling_started_at"]
    archived_at = existing_row["archived_at"]

    if next_status == "active":
        next_status = "cooling"
        cooling_started_at = cooling_started_at or now_iso
        archived_at = None
    elif next_status == "cooling" and next_stale_cycles >= cooling_cycles:
        next_status = "archived"
        archived_at = now_iso
    elif next_status not in {"cooling", "archived"}:
        next_status = "cooling"
        cooling_started_at = cooling_started_at or now_iso
        archived_at = None

    return PromotedCandidate(
        category=existing_row["category"],  # type: ignore[arg-type]
        canonical_name=str(existing_row["canonical_name"]),
        status=next_status,
        run_hits=int(existing_row["run_hits"]),
        decayed_run_score=float(existing_row["decayed_run_score"]),
        total_mentions=int(existing_row["total_mentions"]),
        max_doc_count=int(existing_row["max_doc_count"]),
        source_diversity=int(existing_row["source_diversity"]),
        aliases=json.loads(existing_row["aliases_json"]),
        evidence_titles=json.loads(existing_row["evidence_titles_json"]),
        source_names=json.loads(existing_row["source_names_json"]),
        last_seen_run_id=last_seen_run_id or int(existing_row["last_seen_run_id"] or 0),
        last_qualified_run_id=int(existing_row["last_qualified_run_id"] or 0),
        stale_cycles=next_stale_cycles,
        last_evaluated_run_id=max(current_run_id, last_evaluated_run_id),
        cooling_started_at=str(cooling_started_at) if cooling_started_at else None,
        archived_at=str(archived_at) if archived_at else None,
    )


def apply_promotion_state_machine(
    *,
    promoted: List[PromotedCandidate],
    existing_rows: Dict[Tuple[str, str], sqlite3.Row],
    latest_seen_run_ids: Dict[Tuple[str, str], int],
    current_run_id: int,
    cooling_cycles: int,
) -> List[PromotedCandidate]:
    now_iso = datetime.now(timezone.utc).isoformat()
    final_candidates: Dict[Tuple[str, str], PromotedCandidate] = {}

    for item in promoted:
        key = (item.category, item.canonical_name)
        last_seen_run_id = latest_seen_run_ids.get(key, item.last_seen_run_id)
        final_candidates[key] = item.model_copy(
            update={
                "status": "active",
                "last_seen_run_id": last_seen_run_id,
                "last_qualified_run_id": current_run_id,
                "stale_cycles": 0,
                "last_evaluated_run_id": current_run_id,
                "cooling_started_at": None,
                "archived_at": None,
            }
        )

    for key, row in existing_rows.items():
        if key in final_candidates:
            continue
        final_candidates[key] = transition_candidate_state(
            row,
            current_run_id=current_run_id,
            now_iso=now_iso,
            cooling_cycles=cooling_cycles,
            last_seen_run_id=latest_seen_run_ids.get(key, int(row["last_seen_run_id"] or 0)),
        )

    ordered = list(final_candidates.values())
    ordered.sort(
        key=lambda item: (
            {"active": 0, "cooling": 1, "archived": 2}.get(item.status, 3),
            -item.decayed_run_score,
            -item.run_hits,
            -item.source_diversity,
            -item.total_mentions,
        )
    )
    return ordered


def persist_promotions(conn: sqlite3.Connection, promoted: List[PromotedCandidate]) -> None:
    ensure_promotion_schema(conn)
    for item in promoted:
        conn.execute(
            """
            INSERT INTO promoted_candidates (
                category, canonical_name, status, run_hits, total_mentions, max_doc_count,
                decayed_run_score, source_diversity, aliases_json, evidence_titles_json, source_names_json,
                last_seen_run_id, last_qualified_run_id, stale_cycles, last_evaluated_run_id,
                cooling_started_at, archived_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(category, canonical_name) DO UPDATE SET
                status = excluded.status,
                run_hits = excluded.run_hits,
                decayed_run_score = excluded.decayed_run_score,
                total_mentions = excluded.total_mentions,
                max_doc_count = excluded.max_doc_count,
                source_diversity = excluded.source_diversity,
                aliases_json = excluded.aliases_json,
                evidence_titles_json = excluded.evidence_titles_json,
                source_names_json = excluded.source_names_json,
                last_seen_run_id = excluded.last_seen_run_id,
                last_qualified_run_id = excluded.last_qualified_run_id,
                stale_cycles = excluded.stale_cycles,
                last_evaluated_run_id = excluded.last_evaluated_run_id,
                cooling_started_at = excluded.cooling_started_at,
                archived_at = excluded.archived_at,
                promoted_at = CURRENT_TIMESTAMP
            """,
            (
                item.category,
                item.canonical_name,
                item.status,
                item.run_hits,
                item.total_mentions,
                item.max_doc_count,
                item.decayed_run_score,
                item.source_diversity,
                json.dumps(item.aliases, ensure_ascii=False),
                json.dumps(item.evidence_titles, ensure_ascii=False),
                json.dumps(item.source_names, ensure_ascii=False),
                item.last_seen_run_id,
                item.last_qualified_run_id,
                item.stale_cycles,
                item.last_evaluated_run_id,
                item.cooling_started_at,
                item.archived_at,
            ),
        )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote stable alias candidates from historical discovery runs.")
    parser.add_argument(
        "--sqlite-path",
        default=DEFAULT_SQLITE_PATH,
        help="SQLite database path used by discover_alias_candidates.py.",
    )
    parser.add_argument("--lookback-runs", type=int, default=6, help="How many recent runs to inspect.")
    parser.add_argument("--min-run-hits", type=int, default=3, help="Minimum number of runs a candidate must appear in.")
    parser.add_argument("--min-total-mentions", type=int, default=6, help="Minimum total mentions across recent runs.")
    parser.add_argument("--min-max-doc-count", type=int, default=2, help="Minimum best per-run distinct document count.")
    parser.add_argument("--min-source-diversity", type=int, default=2, help="Minimum number of distinct sources across recent runs.")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Enable cold-start mode by lowering promotion thresholds to 1.",
    )
    parser.add_argument(
        "--cooling-cycles",
        type=int,
        default=2,
        help="How many failed promotion cycles a cooling candidate can survive before being archived.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path. Prints to stdout when omitted.",
    )
    args = parser.parse_args()

    if args.bootstrap:
        args.min_run_hits = 1
        args.min_total_mentions = 1
        args.min_max_doc_count = 1
        args.min_source_diversity = 1

    db_path = Path(args.sqlite_path)
    if not db_path.exists():
        raise RuntimeError(f"SQLite database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        ensure_promotion_schema(conn)
        rows = load_recent_candidates(conn, lookback_runs=args.lookback_runs)
        promoted_now = aggregate_promotions(
            rows,
            min_run_hits=args.min_run_hits,
            min_total_mentions=args.min_total_mentions,
            min_max_doc_count=args.min_max_doc_count,
            min_source_diversity=args.min_source_diversity,
        )
        latest_seen_run_ids = build_latest_seen_run_map(rows)
        current_run_id = load_latest_discovery_run_id(conn)
        existing_rows = load_existing_promotions(conn)
        promoted = apply_promotion_state_machine(
            promoted=promoted_now,
            existing_rows=existing_rows,
            latest_seen_run_ids=latest_seen_run_ids,
            current_run_id=current_run_id,
            cooling_cycles=args.cooling_cycles,
        )
        persist_promotions(conn, promoted)
    finally:
        conn.close()

    payload = {
        "meta": {
            "sqlite_path": str(db_path),
            "lookback_runs": args.lookback_runs,
            "bootstrap": args.bootstrap,
            "min_run_hits": args.min_run_hits,
            "min_total_mentions": args.min_total_mentions,
            "min_max_doc_count": args.min_max_doc_count,
            "min_source_diversity": args.min_source_diversity,
            "cooling_cycles": args.cooling_cycles,
            "active_count": sum(1 for item in promoted if item.status == "active"),
            "cooling_count": sum(1 for item in promoted if item.status == "cooling"),
            "archived_count": sum(1 for item in promoted if item.status == "archived"),
        },
        "promoted_candidates": [item.model_dump() for item in promoted],
    }

    output_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_text, encoding="utf-8")
        print(f"Wrote promoted candidates to {args.output}")
        return
    print(output_text)


if __name__ == "__main__":
    main()
