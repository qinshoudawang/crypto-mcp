#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cmd="${1:-dev}"

run_dev() {
  exec python3 -m followin_mcp.dev "${@:2}"
}

case "$cmd" in
  dev)
    run_dev "$@"
    ;;
  *)
    echo "Usage:"
    echo "  ./scripts/start_dev.sh dev [--host HOST --port PORT]"
    echo "  ./scripts/run_candidate_pipeline.sh [discover_args...]"
    exit 1
    ;;
esac
