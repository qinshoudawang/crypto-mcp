from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

from dotenv import load_dotenv


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


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


def _cleanup_stale_processes() -> None:
    patterns = [
        "python3 -m followin_mcp.mcp.server",
        "python -m followin_mcp.mcp.server",
        "python3 -m followin_mcp.demo.webapp",
        "python -m followin_mcp.demo.webapp",
    ]

    try:
        output = subprocess.check_output(
            ["ps", "-axo", "pid=,command="],
            text=True,
        )
    except Exception:
        return

    current_pid = os.getpid()
    stale_pids: list[int] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid_text, command = line.split(None, 1)
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        if any(pattern in command for pattern in patterns):
            stale_pids.append(pid)

    if not stale_pids:
        return

    print(f"[dev] cleaning stale processes: {', '.join(str(pid) for pid in stale_pids)}")
    for pid in stale_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue

    time.sleep(0.5)

    for pid in stale_pids:
        try:
            os.kill(pid, 0)
        except OSError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Followin MCP server and web demo as separate processes.")
    parser.add_argument("--host", default=os.getenv("FOLLOWIN_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("FOLLOWIN_WEB_PORT", "8000")))
    parser.add_argument("--mcp-host", default=os.getenv("FOLLOWIN_MCP_HOST", "127.0.0.1"))
    parser.add_argument("--mcp-port", type=int, default=int(os.getenv("FOLLOWIN_MCP_PORT", "8001")))
    args = parser.parse_args()

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

    python = sys.executable
    env = dict(os.environ)
    env["FOLLOWIN_WEB_HOST"] = args.host
    env["FOLLOWIN_WEB_PORT"] = str(args.port)
    env["FOLLOWIN_MCP_HOST"] = args.mcp_host
    env["FOLLOWIN_MCP_PORT"] = str(args.mcp_port)
    env["FOLLOWIN_MCP_TRANSPORT"] = env.get("FOLLOWIN_MCP_TRANSPORT", "streamable-http")
    env["FOLLOWIN_MCP_PATH"] = env.get("FOLLOWIN_MCP_PATH", "/mcp")
    env["FOLLOWIN_MCP_SERVER_URL"] = env.get(
        "FOLLOWIN_MCP_SERVER_URL",
        f"http://{args.mcp_host}:{args.mcp_port}{env['FOLLOWIN_MCP_PATH']}",
    )

    required_vars = ["FOLLOWIN_API_KEY", "OPENAI_API_KEY"]
    missing = [name for name in required_vars if not env.get(name)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    _cleanup_stale_processes()

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
    print(f"[dev] mcp url: {env['FOLLOWIN_MCP_SERVER_URL']}")
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
