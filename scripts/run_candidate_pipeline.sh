#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

bootstrap_args=()
discover_args=()
verbose=false

for arg in "$@"; do
  if [[ "$arg" == "--bootstrap" ]]; then
    bootstrap_args+=("--bootstrap")
  elif [[ "$arg" == "--verbose" ]]; then
    verbose=true
  else
    discover_args+=("$arg")
  fi
done

run_stage() {
  local name="$1"
  shift
  echo "[pipeline] starting ${name}"
  if "$@"; then
    echo "[pipeline] completed ${name}"
  else
    local code=$?
    echo "[pipeline] failed ${name} (exit=${code})"
    return "$code"
  fi
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Pipeline usage:"
  echo "  ./scripts/run_candidate_pipeline.sh [discover_args...]"
  echo
  echo "This runs, in order:"
  echo "  1. discover_alias_candidates.py"
  echo "  2. promote_alias_candidates.py"
  echo "  3. export_promoted_aliases.py"
  echo
  echo "Examples:"
  echo "  ./scripts/run_candidate_pipeline.sh"
  echo "  ./scripts/run_candidate_pipeline.sh --bootstrap"
  echo "  ./scripts/run_candidate_pipeline.sh --bootstrap --verbose"
  exit 0
fi

echo "[pipeline] started alias discovery pipeline"
echo "[pipeline] configured model: ${OPENAI_MODEL:-unset}"
discover_cmd=(python3 scripts/discover_alias_candidates.py)
if ((${#discover_args[@]} > 0)); then
  discover_cmd+=("${discover_args[@]}")
fi
if ((${#bootstrap_args[@]} > 0)); then
  discover_cmd+=("${bootstrap_args[@]}")
fi
discover_output="data/candidate_discovery/discover.latest.json"
discover_cmd+=("--output" "$discover_output")

promote_cmd=(python3 scripts/promote_alias_candidates.py)
if ((${#bootstrap_args[@]} > 0)); then
  promote_cmd+=("${bootstrap_args[@]}")
fi
promote_output="data/candidate_discovery/promote.latest.json"
promote_cmd+=("--output" "$promote_output")

run_stage "discover" "${discover_cmd[@]}" || exit $?
python3 - <<'PY' "$discover_output"
import json, sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
meta = payload.get("meta", {})
print(
    f"[pipeline] discover summary: model={meta.get('model')} "
    f"docs={meta.get('doc_count')} candidates={meta.get('candidate_count')} "
    f"bootstrap={meta.get('bootstrap', False)} "
    f"review_drop={meta.get('review_summary', {}).get('dropped_count', 0)} "
    f"review_merge={meta.get('review_summary', {}).get('merge_decision_count', 0)}"
)
PY
if [[ "$verbose" == true ]]; then
  cat "$discover_output"
fi

run_stage "promote" "${promote_cmd[@]}" || exit $?
python3 - <<'PY' "$promote_output"
import json, sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
meta = payload.get("meta", {})
print(
    f"[pipeline] promote summary: bootstrap={meta.get('bootstrap', False)} "
    f"active={meta.get('active_count')} cooling={meta.get('cooling_count')} "
    f"archived={meta.get('archived_count')}"
)
PY
if [[ "$verbose" == true ]]; then
  cat "$promote_output"
fi

run_stage "export" python3 scripts/export_promoted_aliases.py || exit $?
python3 - <<'PY' "data/candidate_discovery/promoted_aliases.json"
import json, sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
meta = payload.get("meta", {})
counts = meta.get("status_counts", {})
print(
    f"[pipeline] export summary: active_export_count={meta.get('active_export_count')} "
    f"active={counts.get('active', 0)} cooling={counts.get('cooling', 0)} "
    f"archived={counts.get('archived', 0)}"
)
PY
if [[ "$verbose" == true ]]; then
  cat "data/candidate_discovery/promoted_aliases.json"
fi
echo "[pipeline] all stages completed"
