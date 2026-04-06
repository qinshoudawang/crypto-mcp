from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_SQLITE_PATH = "data/candidate_discovery/discovery.db"
DEFAULT_OUTPUT_PATH = "data/candidate_discovery/promoted_aliases.json"


def load_promoted_candidates(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return list(
        conn.execute(
            """
            SELECT category,
                   canonical_name,
                   aliases_json
            FROM promoted_candidates
            WHERE status = 'active'
            ORDER BY category, canonical_name
            """
        )
    )


def load_export_metadata(conn: sqlite3.Connection) -> Dict[str, Any]:
    conn.row_factory = sqlite3.Row
    status_counts = {
        str(row["status"]): int(row["count"])
        for row in conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM promoted_candidates
            GROUP BY status
            """
        )
    }

    latest_row = conn.execute(
        """
        SELECT
            COALESCE(MAX(last_evaluated_run_id), 0) AS latest_evaluated_run_id,
            COALESCE(MAX(last_qualified_run_id), 0) AS latest_qualified_run_id,
            COALESCE(MAX(last_seen_run_id), 0) AS latest_seen_run_id,
            COALESCE(MAX(promoted_at), '') AS latest_promoted_at
        FROM promoted_candidates
        """
    ).fetchone()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status_counts": {
            "active": status_counts.get("active", 0),
            "cooling": status_counts.get("cooling", 0),
            "archived": status_counts.get("archived", 0),
        },
        "latest_evaluated_run_id": int(latest_row["latest_evaluated_run_id"]),
        "latest_qualified_run_id": int(latest_row["latest_qualified_run_id"]),
        "latest_seen_run_id": int(latest_row["latest_seen_run_id"]),
        "latest_promoted_at": str(latest_row["latest_promoted_at"] or ""),
    }


def build_alias_payload(rows: List[sqlite3.Row]) -> Dict[str, Any]:
    payload = {
        "meta": {},
        "entity_aliases": {},
        "token_aliases": {},
        "topic_aliases": {},
        "event_keywords": {},
    }

    for row in rows:
        category = row["category"]
        canonical_name = row["canonical_name"]
        aliases = json.loads(row["aliases_json"])
        merged_aliases = [canonical_name, *aliases]
        deduped_aliases = list(dict.fromkeys(alias.strip() for alias in merged_aliases if alias.strip()))
        if category == "project":
            payload["entity_aliases"][canonical_name] = deduped_aliases
        elif category == "token":
            payload["token_aliases"][canonical_name.upper()] = deduped_aliases
        elif category == "topic":
            payload["topic_aliases"][canonical_name] = deduped_aliases
        elif category == "event_keyword":
            payload["event_keywords"][canonical_name] = deduped_aliases

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export promoted SQLite candidates into a dynamic alias config.")
    parser.add_argument(
        "--sqlite-path",
        default=DEFAULT_SQLITE_PATH,
        help="SQLite database path used by candidate discovery scripts.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON config path for the normalizer to load.",
    )
    args = parser.parse_args()

    db_path = Path(args.sqlite_path)
    if not db_path.exists():
        raise RuntimeError(f"SQLite database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        rows = load_promoted_candidates(conn)
        export_meta = load_export_metadata(conn)
    finally:
        conn.close()

    payload = build_alias_payload(rows)
    payload["meta"] = {
        **export_meta,
        "sqlite_path": str(db_path),
        "active_export_count": len(rows),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote promoted alias config to {output_path}")


if __name__ == "__main__":
    main()
