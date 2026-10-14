"""Fake cloud providers for the tests: an OpenAI-compatible API under /oai and the Anthropic API under /.

The server records every request (srv.requests) and its behaviour is set through srv.mode:
  "ok"         normal streamed answers
  "strict"     rejects any optional parameter (400) so the client must retry with a bare request
  "refusal"    Anthropic answers with stop_reason "refusal"
Only the key "test-key" is accepted.
"""
import json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEY = "test-key"
OAI_MODELS = {"data": [
    {"id": "mock/vision-chat", "name": "Mock Vision", "architecture": {"input_modalities": ["text", "image"]},
     "supported_parameters": ["reasoning", "response_format"], "context_length": 131072},
    {"id": "mock/text-only", "name": "Mock Text", "architecture": {"input_modalities": ["text"]}, "supported_parameters": []}]}
CLAUDE_MODELS = {"data": [
    {"type": "model", "id": "claude-opus-5-5", "display_name": "Claude Opus 5.5", "created_at": "2026-01-01T00:00:00Z",
     "max_input_tokens": 1000000, "max_tokens": 128000,
     "capabilities": {"image_input": {"supported": True}, "thinking": {"supported": True, "types": {"adaptive": {"supported": True}}},
                      "effort": {"supported": True}}},
    {"type": "model", "id": "claude-old", "display_name": "Claude Old", "created_at": "2024-01-01T00:00:00Z",
     "max_input_tokens": 200000, "max_tokens": 8192,
     "capabilities": {"image_input": {"supported": False}, "thinking": {"supported": False, "types": {"adaptive": {"supported": False}}},
                      "effort": {"supported": False}}}],
    "has_more": False, "first_id": "claude-opus-5-5", "last_id": "claude-old"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self):
        return self.headers.get("Authorization") == "Bearer " + KEY or self.headers.get("x-api-key") == KEY

    def do_GET(self):
        self.server.requests.append(("GET", self.path, None, dict(self.headers)))
        if not self._authed():
            return self._json(401, {"error": {"message": "invalid key"}})
        if self.path.startswith("/oai/models"):
            return self._json(200, OAI_MODELS)
        if self.path.startswith("/v1/models"):
            return self._json(200, CLAUDE_MODELS)
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        self.server.requests.append(("POST", self.path, body, dict(self.headers)))
        if not self._authed():
            return self._json(401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}})
        if self.path.startswith("/oai/chat/completions"):
            return self._oai(body)
        if self.path.startswith("/v1/messages"):
            return self._claude(body)
        self._json(404, {"error": {"message": "not found"}})

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

    def _oai(self, body):
        optional = {"response_format", "reasoning", "temperature", "reasoning_effort"}
        if self.server.mode == "strict" and optional & body.keys():
            return self._json(400, {"error": {"message": "unsupported parameter"}})
        self._sse()
        for d in ({"reasoning": "je réfléchis"}, {"content": "Bon"}, {"content": "jour"}):
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": d}]}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def _claude(self, body):
        if self.server.mode == "strict" and ({"output_config", "thinking"} & body.keys()):
            return self._json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "effort is not supported"}})
        self._sse()
        stop = "refusal" if self.server.mode == "refusal" else "end_turn"

        def ev(kind, data):
            self.wfile.write(f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode())
        ev("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": body.get("model"), "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 1}}})
        idx = 0
        if "thinking" in body:
            ev("content_block_start", {"type": "content_block_start", "index": 0,
                                       "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
            ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}})
            ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}})
            ev("content_block_stop", {"type": "content_block_stop", "index": 0})
            idx = 1
        ev("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}})
        for piece in ("Sa", "lut"):
            ev("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": piece}})
        ev("content_block_stop", {"type": "content_block_stop", "index": idx})
        ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": 3}})
        ev("message_stop", {"type": "message_stop"})


def start():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.requests, srv.mode = [], "ok"
    srv.url = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
