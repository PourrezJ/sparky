#!/usr/bin/env bash
# Start Sparky in the background, then open the web UI.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/runtime/venv/bin/python"
[[ -x "$PY" ]] || { echo "Run ./install-macos.sh first." >&2; exit 1; }
PORT="$("$PY" - <<'PY'
import json
from pathlib import Path
f = Path("config.json")
c = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
print(int(c.get("app_port", 7860)))
PY
)"
URL="http://127.0.0.1:$PORT"
if ! curl --silent --fail --max-time 2 "$URL/api/version" >/dev/null; then
  mkdir -p data
  nohup "$PY" app/server.py --port "$PORT" >data/server.log 2>data/server.err.log </dev/null &
  echo $! >data/server.pid
  for _ in $(seq 1 60); do
    if curl --silent --fail --max-time 1 "$URL/api/version" >/dev/null; then break; fi
    if ! kill -0 "$(cat data/server.pid)" 2>/dev/null; then
      echo "Sparky did not start. Read data/server.err.log" >&2
      exit 1
    fi
    sleep 0.5
  done
fi
curl --silent --fail --max-time 2 "$URL/api/version" >/dev/null || { echo "Sparky is not responding (port $PORT)." >&2; exit 1; }
open "$URL"
echo "Sparky: $URL"
