#!/usr/bin/env bash
# Stop only the Sparky processes started by start-macos.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT/data/server.pid"
[[ -f "$PID_FILE" ]] || { echo "No Sparky pid file found."; exit 0; }
PID="$(cat "$PID_FILE")"
[[ "$PID" =~ ^[0-9]+$ ]] || { echo "Invalid pid file." >&2; exit 1; }
"$ROOT/runtime/venv/bin/python" - "$PID" "$ROOT" <<'PY'
import os, sys
import psutil

pid, root = int(sys.argv[1]), os.path.realpath(sys.argv[2])
try:
    proc = psutil.Process(pid)
    argv = proc.cmdline()
    # Never kill a process merely because its PID was reused.
    if not any("app/server.py" in a.replace("\\", "/") for a in argv):
        print("PID is not Sparky; refusing to terminate it.")
        sys.exit(1)
    if not any(os.path.realpath(a).startswith(root + os.sep) for a in argv if os.path.isabs(a)):
        print("PID does not belong to this Sparky installation.")
        sys.exit(1)
    children = proc.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.Error:
            pass
    proc.terminate()
    gone, alive = psutil.wait_procs([proc, *children], timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.Error:
            pass
    print("Sparky stopped.")
except psutil.NoSuchProcess:
    print("Sparky is not running.")
PY
rm -f "$PID_FILE"
