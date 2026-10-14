"""MCP client: a fake stdio server (python script) and a fake Streamable HTTP server, both exposing "add"."""
import json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

FAKE = r'''
import json, sys
TOOLS = [{"name": "add", "description": "Adds two numbers.",
          "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                          "required": ["a"]}},
         {"name": "fail", "description": "Always fails.", "inputSchema": {"type": "object"}}]
def send(m):
    sys.stdout.write(json.dumps(m) + "\n"); sys.stdout.flush()
print("fake server starting", file=sys.stderr, flush=True)
for line in sys.stdin:
    m = json.loads(line)
    if "id" not in m:
        continue                                   # notifications
    if "method" not in m:
        continue                                   # our reply to the ping below
    meth, p, i = m["method"], m.get("params") or {}, m["id"]
    if meth == "initialize":
        send({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "hi"}})
        send({"jsonrpc": "2.0", "id": "srv-1", "method": "roots/list"})   # client must answer with an error
        send({"jsonrpc": "2.0", "id": i, "result": {"protocolVersion": p["protocolVersion"], "capabilities": {"tools": {}},
                                                    "serverInfo": {"name": "fake", "version": "0"}}})
    elif meth == "tools/list":                     # two pages
        page = 1 if p.get("cursor") == "p2" else 0
        r = {"tools": [TOOLS[page]]}
        if not page:
            r["nextCursor"] = "p2"
        send({"jsonrpc": "2.0", "id": i, "result": r})
    elif meth == "tools/call":
        a = p["arguments"]
        if p["name"] == "add":
            send({"jsonrpc": "2.0", "id": i, "result": {"content": [{"type": "text", "text": str(a["a"] + a.get("b", 0))},
                                                                   {"type": "image", "mimeType": "image/png", "data": "AAAA"}]}})
        else:
            send({"jsonrpc": "2.0", "id": i, "result": {"isError": True, "content": [{"type": "text", "text": "boom"}]}})
    else:
        send({"jsonrpc": "2.0", "id": i, "error": {"code": -32601, "message": "nope"}})
'''


@pytest.fixture
def mcp(tmp_path):
    import mcp
    script = tmp_path / "fake_mcp.py"
    script.write_text(FAKE, encoding="utf-8")
    mcp.save({"mcpServers": {"calc": {"command": sys.executable, "args": ["-u", str(script)], "env": {"X": "1"}},
                             "off": {"command": "nothing", "disabled": True}}})
    yield mcp
    mcp.stop_all()
    mcp.save({"mcpServers": {}})


def test_start_list_call_stop(mcp):
    mcp.start_all()
    assert mcp.wait_ready(20)
    st = {s["name"]: s for s in mcp.status()}
    assert st["calc"]["state"] == "ready", st["calc"]["error"]
    assert st["off"]["state"] == "off"
    assert [t["name"] for t in st["calc"]["tools"]] == ["add", "fail"]   # both pages
    names = [t["name"] for t in mcp.tools()]
    assert names == ["calc__add", "calc__fail"]
    out = mcp.call("calc__add", {"a": 2, "b": 3})
    assert out.startswith("5") and "[image image/png" in out
    assert mcp.call("calc__fail", {}) == "Error: boom"
    p = mcp.tools_prompt()
    assert "calc__add" in p and '"a": "number"' in p and '"b?": "number"' in p
    with pytest.raises(mcp.McpError):
        mcp.call("nobody__add", {})
    mcp.stop_all()
    assert mcp.tools() == [] and mcp.tools_prompt() == ""
    assert {s["state"] for s in mcp.status()} == {"off"}


def test_restart_and_disable(mcp):
    mcp.start_all()
    assert mcp.wait_ready(20)
    proc = mcp.SERVERS["calc"].proc
    mcp.restart("calc")
    assert mcp.wait_ready(20) and mcp.SERVERS["calc"].proc is not proc
    assert proc.poll() is not None
    cfg = mcp.load()
    cfg["mcpServers"]["calc"]["disabled"] = True
    mcp.save(cfg)
    mcp.start_all()   # reconciles: the disabled server is stopped
    assert "calc" not in mcp.SERVERS and mcp.status()[0]["state"] == "off"


def test_bad_command_reports_an_error(mcp):
    mcp.save({"mcpServers": {"bad": {"command": sys.executable, "args": ["-c", "import sys; print('oops', file=sys.stderr)"]}}})
    mcp.start_all()
    assert mcp.wait_ready(20)
    st = mcp.status()[0]
    assert st["state"] == "error" and st["error"]


# ---------------------------------------------------------------- Streamable HTTP
class Http(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        m = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.seen.append((m.get("method"), self.headers.get("Mcp-Session-Id")))
        if "id" not in m:
            self.send_response(202)
            self.end_headers()
            return
        if m["method"] == "initialize":
            r = {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "h"}}
        elif m["method"] == "tools/list":
            r = {"tools": [{"name": "add", "inputSchema": {"type": "object"}}]}
        else:
            a = m["params"]["arguments"]
            r = {"content": [{"type": "text", "text": str(a["a"] + a["b"])}]}
        body = json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r})
        self.send_response(200)
        if m["method"] == "tools/call":   # answer as an SSE stream, with a notification first
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'event: message\ndata: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n')
            self.wfile.write(b"data: " + body.encode() + b"\n\n")
            return
        self.send_header("Content-Type", "application/json")
        self.send_header("Mcp-Session-Id", "sess-42")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())


def test_streamable_http(mcp):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Http)
    srv.seen = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        mcp.save({"mcpServers": {"web": {"url": f"http://127.0.0.1:{srv.server_port}/mcp"}}})
        mcp.start_all()
        assert mcp.wait_ready(10)
        assert mcp.status()[0]["state"] == "ready", mcp.status()[0]["error"]
        assert mcp.call("web__add", {"a": 1, "b": 2}) == "3"
        assert srv.seen[0] == ("initialize", None)
        assert all(sid == "sess-42" for _, sid in srv.seen[1:])
    finally:
        mcp.stop_all()
        srv.shutdown()
