from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _terminate(process: subprocess.Popen[bytes], name: str) -> None:
    if process.poll() is not None:
        return

    print(f"[dev] stopping {name} (pid={process.pid})")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Followin MCP server and web demo.")
    parser.add_argument("--host", default=os.getenv("FOLLOWIN_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("FOLLOWIN_WEB_PORT", "8000")))
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    python = sys.executable
    env = dict(os.environ)
    env["FOLLOWIN_WEB_HOST"] = args.host
    env["FOLLOWIN_WEB_PORT"] = str(args.port)

    required_vars = ["FOLLOWIN_API_KEY", "OPENAI_API_KEY"]
    missing = [name for name in required_vars if not env.get(name)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    processes = [
        (
            "mcp",
            subprocess.Popen(
                [python, "-m", "followin_mcp.mcp.server"],
                cwd=PROJECT_ROOT,
                env=env,
            ),
        ),
        (
            "web",
            subprocess.Popen(
                [python, "-m", "followin_mcp.demo.webapp"],
                cwd=PROJECT_ROOT,
                env=env,
            ),
        ),
    ]

    print("[dev] started Followin MCP stack")
    print(f"[dev] web url: http://{args.host}:{args.port}")
    print("[dev] press Ctrl+C to stop all processes")

    try:
        while True:
            for name, process in processes:
                code = process.poll()
                if code is not None:
                    raise RuntimeError(f"{name} exited unexpectedly with code {code}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[dev] shutting down")
    finally:
        for name, process in reversed(processes):
            _terminate(process, name)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    main()
