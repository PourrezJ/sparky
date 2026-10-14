"""Regression tests for the issues found by the code review (one test per fix)."""
import json, os, sys, threading, time, urllib.error, urllib.request

import pytest

import config


@pytest.fixture(scope="module")
def servers():
    import attach, server
    attach.server_voice = server.voice_model
    main = server.Server(("127.0.0.1", 0), server.Handler)
    prev = server.Server(("127.0.0.1", 0), server.PreviewHandler)
    for s in (main, prev):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{main.server_address[1]}", f"http://127.0.0.1:{prev.server_address[1]}"
    main.shutdown()
    prev.shutdown()


def req(url, body=None, raw=None, headers=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    r = urllib.request.Request(url, data, {"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(r, timeout=20) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# H1: generated projects cannot reach the API
def test_projects_are_not_served_on_the_api_origin(servers):
    assert req(servers[0] + "/api/proj/x/y/index.html")[0] == 404


def test_preview_origin_cannot_post_to_the_api(servers):
    main, prev = servers
    assert req(prev + "/api/settings", {"settings": {}})[0] == 403  # the preview server has no API
    code, _ = req(main + "/api/settings", {"settings": {}}, headers={"Origin": prev})
    assert code == 403  # a page from the preview origin posting to the API


def test_the_page_itself_can_post(servers):
    main = servers[0]
    assert req(main + "/api/settings", {"settings": {}}, headers={"Origin": main})[0] == 200


# H2: changing the custom provider address drops its key
def test_custom_provider_key_is_not_sent_to_a_new_address(clean_cloud, mock_provider):
    clean_cloud.set_key("custom", "test-key", mock_provider.url + "/oai")
    with pytest.raises(RuntimeError):
        clean_cloud.set_key("custom", None, "http://127.0.0.1:9/v1")
    assert "key" not in clean_cloud.conf("custom")
    assert not any(r[3].get("Authorization") for r in mock_provider.requests if "127.0.0.1:9" in r[1])


# H3: JSON files are written atomically and a damaged file does not stop Sparky
def test_damaged_json_falls_back_to_default(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("{ broken")
    assert config.read_json(str(p), {"ok": 1}) == {"ok": 1}


def test_concurrent_writes_never_produce_broken_json(tmp_path):
    p = str(tmp_path / "s.json")
    ts = [threading.Thread(target=lambda i=i: [config.write_json(p, {"n": i, "pad": "x" * 5000}) for _ in range(30)])
          for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert json.load(open(p))["pad"] == "x" * 5000
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


# H4: a private chat stays private when its id is unknown
@pytest.mark.parametrize("ephemeral", [True, False])
def test_unknown_chat_keeps_its_privacy(servers, ephemeral):
    import server
    _, body = req(servers[0] + "/api/chat", {"id": "20990101-000000-abcd", "ephemeral": ephemeral, "text": "x"})
    ack = next(json.loads(l[6:]) for l in body.decode().split("\n\n") if l.startswith("data: ") and '"ack"' in l)
    assert server.sessions[ack["id"]].ephemeral is ephemeral


# M1: deleting an assistant works when no image model is selected
def test_delete_assistant_without_image_model(servers):
    import harness
    harness.SETTINGS["ckpt"] = None
    assert req(servers[0] + "/api/model/delete", {"id": "gemmae4b"})[0] == 200


# M4: a broken request gets a JSON error instead of a dropped connection
def test_invalid_json_body_gets_an_error(servers):
    code, body = req(servers[0] + "/api/use", raw=b"{not json")
    assert code == 500 and "error" in json.loads(body)


def test_huge_body_is_refused_before_reading(servers):
    r = urllib.request.Request(servers[0] + "/api/attach", b"{}", {"Content-Type": "application/json",
                                                                    "Content-Length": str(10**9)})
    try:
        urllib.request.urlopen(r, timeout=10)
        code = 200
    except urllib.error.HTTPError as e:
        code = e.code
    except (urllib.error.URLError, ConnectionError):
        code = 413  # the server answered and closed before the client finished sending
    assert code == 413


# M6: malformed session ids never become paths
@pytest.mark.parametrize("sid", ["..", "..\\..\\Windows", "../../etc", "20990101-000000-abcd/../.."])
def test_malformed_session_ids_are_rejected(sid):
    import server
    assert server.get_session(sid) is None


# M8: settings are validated
def test_settings_are_clamped():
    import harness, server
    server.update_settings({"max_steps": "", "steps": 9999, "cfg": "abc"})
    assert 1 <= harness.SETTINGS["max_steps"] <= 20 and harness.SETTINGS["steps"] == 150
    server.update_settings({"max_steps": 0})
    assert harness.SETTINGS["max_steps"] == 1


# M5: a test that times out does not leave processes behind
def test_timeout_kills_the_whole_process_tree(tmp_path):
    import checker, psutil
    pidfile = tmp_path / "child.pid"
    code = ("import subprocess, sys, time\n"
            f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            f"open(r'{pidfile}', 'w').write(str(p.pid))\ntime.sleep(60)\n")
    t0 = time.time()
    rc, out, timed_out = checker.run([sys.executable, "-c", code], str(tmp_path), 3)
    assert timed_out and rc is None and time.time() - t0 < 20
    child = int(pidfile.read_text())
    assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE


# M7: one update at a time
def test_second_update_is_refused_while_one_runs(monkeypatch):
    import updater
    monkeypatch.setattr(updater, "status", lambda: {"available": True, "git": True, "changes": []})
    monkeypatch.setitem(updater.state, "updating", True)
    assert updater.update() == (False, "mise à jour déjà en cours")


# perf: the GPU is not polled again once nvidia-smi is known to be missing
def test_gpu_stats_remember_missing_nvidia_smi(monkeypatch):
    import server
    calls = []

    def fake_run(*a, **k):
        calls.append(1)
        raise FileNotFoundError
    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setitem(server._gpu, "missing", False)
    monkeypatch.setitem(server._gpu, "t", 0.0)
    server.gpu_stats()
    server._gpu["t"] = 0.0
    server.gpu_stats()
    assert len(calls) == 1 and server.gpu_stats() == {}


def test_sparky_restart_helper_is_not_detached():
    """Regression: with DETACHED_PROCESS PowerShell had no console and died, Sparky never came back."""
    src = open(os.path.join(os.path.dirname(config.__file__), "updater.py"), encoding="utf-8").read()
    assert "DETACHED_PROCESS" not in src.split("def _restart")[1].split("def update")[0].replace("# no DETACHED_PROCESS", "")
