"""Model downloads: parallel byte-range segments, resume of an interrupted file, join (local HTTP server)."""
import http.server, os, socketserver, threading

import pytest

import catalog

DATA = os.urandom(3 * 2**20 + 12345)


class Ranges(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        a, b = self.headers["Range"].split("=")[1].split("-")
        a, b = int(a), min(int(b or len(DATA) - 1), len(DATA) - 1)
        body = DATA[a:b + 1]
        self.send_response(206)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Range", f"bytes {a}-{b}/{len(DATA)}")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def url():
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Ranges)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/model.bin"
    srv.shutdown()


@pytest.mark.parametrize("already", [0, 777777])
def test_segmented_download_and_resume(url, tmp_path, monkeypatch, already):
    monkeypatch.setattr(catalog, "SEGMENT_MIN", 2**20)  # the 3 MB test file is "big"
    part = str(tmp_path / "model.bin.part")
    if already:  # an older single-stream download stopped there: kept as the first segment
        open(part, "wb").write(DATA[:already])
    seen = []
    catalog.fetch(url, part, len(DATA), seen.append)
    assert open(part, "rb").read() == DATA
    assert seen and max(seen) == len(DATA)
    assert not [p for p in os.listdir(tmp_path) if p != "model.bin.part"]  # segments and plan removed


def test_small_file_single_stream(url, tmp_path):
    part = str(tmp_path / "small.part")
    catalog.fetch(url, part, len(DATA), lambda n: None)
    assert open(part, "rb").read() == DATA
