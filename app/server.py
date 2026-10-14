"""Sparky: chat UI + agent harness (harness.py) + vision LLM (llama-server) + ComfyUI.

Usage: python app\\server.py [--port 7860] [--no-llm]   (or start.bat)
The assistant model, image model and settings are chosen in the UI (Settings).
"""
import argparse, base64, io, json, os, shutil, subprocess, sys, threading, time, urllib.request
import http.server, socketserver

import psutil
from PIL import Image

import catalog
import config
import harness
import localscan
import syscheck
import updater

HERE = os.path.dirname(os.path.abspath(__file__))
NOWIN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
LLM_LOG = os.path.join(config.DATA, "llm.log")
LLMS = catalog.BY_ID    # all models (catalog + Hugging Face additions), filtered by kind when used
sessions = {}           # id -> Session (in memory)
llm_state = {"name": None}
# model switch in progress, tracked by the UI (spinner + steps)
switch = {"kind": None, "id": None, "label": "", "phase": "idle", "steps": [], "t0": 0.0, "eta": 0, "error": ""}


def alive(url):
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


def port_listening(port):
    import socket
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def stop_llm():
    for c in psutil.net_connections("tcp"):
        if c.laddr and c.laddr.port == config.LLM_PORT and c.status == psutil.CONN_LISTEN and c.pid:
            try:
                psutil.Process(c.pid).kill()
            except psutil.Error:
                pass


def start_comfy():
    """Starts ComfyUI (image generation) if it isn't already running."""
    if port_listening(config.COMFY_PORT) or not os.path.exists(os.path.join(config.COMFY_DIR, "main.py")):
        return
    os.makedirs(config.DATA, exist_ok=True)
    subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(config.COMFY_PORT)],
                     cwd=config.COMFY_DIR, creationflags=NOWIN,
                     stdout=open(os.path.join(config.DATA, "comfy.log"), "w"), stderr=subprocess.STDOUT)
    print("ComfyUI en cours de demarrage")


def installed_llms():
    return [k for k, m in LLMS.items() if m["kind"] == "llm" and catalog.installed(m)]


def start_llm(name, restart=False):
    if restart:
        stop_llm()
    elif alive(harness.LLM + "/health"):
        llm_state["name"] = name
        return name
    if not os.path.exists(config.LLAMA_SERVER):
        raise RuntimeError("llama-server introuvable : " + config.LLAMA_SERVER + " (relance install.ps1)")
    if name not in LLMS or LLMS[name]["kind"] != "llm" or not catalog.installed(LLMS[name]):
        ok, rec = installed_llms(), syscheck.current().get("recommended", {}).get("llm")
        if not ok:
            print("Aucun assistant installe : choisis-en un dans l'interface.")
            return None
        name = rec if rec in ok else ok[0]
    try:  # free ComfyUI's VRAM before loading a big model (reloaded on the next image)
        harness.http_json(harness.COMFY + "/free", {"unload_models": True, "free_memory": True}, timeout=10)
    except Exception:
        pass
    args = catalog.llm_args(LLMS[name])
    gpu = syscheck.current()["assess"]["mode"] != "cpu"
    cmd = [config.LLAMA_SERVER, "-m", args[0], "--port", str(config.LLM_PORT), "-ngl", "99" if gpu else "0",
           "-c", "32768", "--jinja", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", *args[1:]]
    os.makedirs(config.DATA, exist_ok=True)
    subprocess.Popen(cmd, creationflags=NOWIN, stdout=open(LLM_LOG, "w"), stderr=subprocess.STDOUT)
    llm_state["name"] = harness.SETTINGS["llm"] = name
    print("Assistant en cours de demarrage :", LLMS[name]["label"])
    return name


def autostart():
    """As soon as a model has just been installed and none is active, activate it automatically."""
    while True:
        time.sleep(4)
        try:
            if switch["phase"] == "loading":
                continue
            ok = installed_llms()
            if not llm_state["name"] and ok:
                rec = syscheck.current().get("recommended", {}).get("llm")
                use_model(rec if rec in ok else ok[0])
            elif not harness.SETTINGS.get("ckpt") or not any(
                    m["kind"] == "image" and m["ckpt"] == harness.SETTINGS["ckpt"] and catalog.installed(m) for m in catalog.CATALOG):
                imgs = [m for m in catalog.CATALOG if m["kind"] == "image" and catalog.installed(m)]
                if imgs and alive(harness.COMFY + "/system_stats"):  # loaded through the switch: the UI shows it
                    rec = syscheck.current().get("recommended", {}).get("image")
                    use_model(next((x for x in imgs if x["id"] == rec), imgs[0])["id"])
        except Exception as e:
            print("autostart :", e)


def _step(text, state="run"):
    if switch["steps"] and switch["steps"][-1]["state"] == "run":
        switch["steps"][-1]["state"] = "ok"
    if text:
        switch["steps"].append({"text": text, "state": state, "t": time.time() - switch["t0"]})


def _switch(mid):
    m = catalog.BY_ID[mid]
    try:
        if m["kind"] == "llm":
            old = LLMS.get(llm_state["name"], {}).get("label")
            if old and alive(harness.LLM + "/health"):
                _step(f"Décharge {old}")
                stop_llm()
                for _ in range(40):
                    if not alive(harness.LLM + "/health"):
                        break
                    time.sleep(0.25)
            _step("Libère la mémoire")
            switch["eta"] = round(m["size_gb"] * 1.3 + 6)
            start_llm(mid, restart=False)
            _step(f"Charge {m['label']}")
            t0 = time.time()
            while not alive(harness.LLM + "/health"):
                if time.time() - t0 > 15 and not port_listening(config.LLM_PORT):
                    tail = open(LLM_LOG, encoding="utf-8", errors="replace").read()[-600:]
                    raise RuntimeError("le modèle n'a pas pu se charger (mémoire insuffisante ?)\n" + tail[-300:])
                if time.time() - t0 > 600:
                    raise RuntimeError("chargement trop long")
                time.sleep(0.5)
        else:
            harness.SETTINGS["ckpt"] = m["ckpt"]
            _step("Décharge l'ancien modèle d'image")
            try:  # ComfyUI replies with an empty body: not an error
                harness.http_json(harness.COMFY + "/free", {"unload_models": True, "free_memory": True}, timeout=15)
            except Exception:
                pass
            switch["eta"] = 25
            _step(f"Charge {m['label']}")
            if m.get("engine") in harness.INSTRUCT:
                switch["eta"] = 60 if m.get("engine") == "qwen21" else 20
                warm = harness.build_instruct(m, "warmup", 1, size=(256, 256), steps=1)
            else:
                warm = harness.build_graph("warmup", "", 1, size=(256, 256))
                warm["10"]["inputs"]["steps"] = 1
            warm["20"] = {"class_type": "PreviewImage", "inputs": {"images": warm["20"]["inputs"]["images"]}}
            with harness.busy("image"):
                harness._comfy_run(warm)  # tiny first image: forces the checkpoint to load
        harness.SETTINGS["llm" if m["kind"] == "llm" else "ckpt"] = mid if m["kind"] == "llm" else m["ckpt"]
        harness.save_settings()
        _step(None)
        switch["phase"] = "ready"
    except Exception as e:
        if switch["steps"]:
            switch["steps"][-1]["state"] = "err"
        switch.update(phase="error", error=str(e))


def voice_model():
    """Selected dictation model (or the recommended / first installed one), and whether it is ready."""
    voices = [m for m in catalog.CATALOG if m["kind"] == "voice"]
    ok = [m for m in voices if catalog.installed(m)]
    rec = syscheck.current().get("recommended", {}).get("voice")
    pick = next((m for m in ok if m["id"] == harness.SETTINGS.get("voice")), None) or \
        next((m for m in ok if m["id"] == rec), None) or (ok[0] if ok else catalog.BY_ID.get(rec) or voices[0])
    return pick, catalog.installed(pick)


def use_model(mid):
    m = catalog.BY_ID.get(mid)
    if not m or not catalog.installed(m) or switch["phase"] == "loading":
        return False
    if m["kind"] == "embed":  # Sparky Bot's memory model: always used once installed
        return True
    if m["kind"] == "voice":  # nothing to load: Whisper loads on first dictation
        harness.SETTINGS["voice"] = mid
        harness.save_settings()
        return True
    switch.update(kind=m["kind"], id=mid, label=m["label"], phase="loading", steps=[], t0=time.time(), eta=0, error="")
    threading.Thread(target=_switch, args=(mid,), daemon=True).start()
    return True


def get_session(sid):
    if sid not in sessions:
        if not os.path.exists(os.path.join(harness.DATA, sid, "session.json")):
            return None
        sessions[sid] = harness.Session.load(sid)
    return sessions[sid]


def checkpoints():
    try:
        info = harness.http_json(harness.COMFY + "/object_info/CheckpointLoaderSimple", timeout=5)
        return info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    except Exception:
        return []


_gpu = {"t": 0.0, "v": {}}


def num(v):
    try:
        return float(v)
    except ValueError:
        return None


def metrics():
    if time.time() - _gpu["t"] > 1.5:  # nvidia-smi is slow: at most one read every 1.5 s
        _gpu["t"] = time.time()
        try:
            line = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                                   "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                                  creationflags=NOWIN).stdout.splitlines()[0]
            u, mu, mt, tp, pw = [num(x.strip()) for x in line.split(",")]
            _gpu["v"] = {"util": u, "used": mu, "total": mt, "temp": tp, "power": pw}
        except Exception:
            _gpu["v"] = {}
    vm = psutil.virtual_memory()
    llm_up = alive(harness.LLM + "/health")
    try:
        q = harness.http_json(harness.COMFY + "/queue", timeout=2)
        comfy = {"up": True, "running": len(q["queue_running"]), "pending": len(q["queue_pending"])}
    except Exception:
        comfy = {"up": False, "running": 0, "pending": 0}
    a = syscheck.current()
    return {"cpu": psutil.cpu_percent(), "ram": {"used": vm.used / 2**30, "total": vm.total / 2**30},
            "gpu": _gpu["v"], "gpuName": a["assess"].get("gpu"), "cores": a["hw"].get("cores"),
            "unified": a["assess"]["mode"] == "unified",
            "llm": {"name": LLMS.get(llm_state["name"], {}).get("label"), "up": llm_up, "active": harness.ACTIVITY["llm"] > 0},
            "image": {"name": harness.image_model().get("label") if harness.SETTINGS.get("ckpt") else None, "up": comfy["up"],
                      "active": harness.ACTIVITY["image"] > 0 or comfy["running"] > 0, "queue": comfy["pending"]},
            "switch": {"phase": switch["phase"], "kind": switch["kind"], "label": switch["label"]}}


def run_project(s, pid, p):
    """Run button: same engine as the automatic tests (Python, Node.js with npm, website in a headless browser)."""
    import checker
    r = checker.check(s, pid, p, config.APP_PORT)
    return {"code": r["code"], "output": r["output"] or "(aucune sortie)", "secs": r["secs"], "ok": r["ok"], "kind": r["kind"]}


def decode_image(data_url):
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def query(self):
        from urllib.parse import parse_qs, urlparse
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

    def do_GET(self):
        path = self.path.split("?")[0]
        q = self.query()
        if path in ("/", "/index.html"):
            return self.send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
        if path == "/logo.png":
            return self.send(200, open(os.path.join(HERE, "logo.png"), "rb").read(), "image/png")
        if path == "/api/health":
            return self.send(200, {"comfy": alive(harness.COMFY + "/system_stats"), "llm": alive(harness.LLM + "/health"),
                                   "llmName": LLMS.get(llm_state["name"], {}).get("label")})
        if path == "/api/metrics":
            return self.send(200, metrics())
        if path == "/api/sessions":
            return self.send(200, harness.Session.list_all())
        if path == "/api/state":
            s = sessions.get(q.get("id")) or get_session(q.get("id", ""))
            return self.send(200, {"id": s.id, "events": s.events} if s else {"id": None, "events": []})
        if path.startswith("/api/img/"):  # /api/img/<session>/<n>
            try:
                sid, n = path[len("/api/img/"):].rsplit("/", 1)
                s = sessions.get(sid) or get_session(sid)
                return self.send(200, open(s.images[int(n)]["path"], "rb").read(), "image/png")
            except (AttributeError, KeyError, ValueError, OSError):
                return self.send(404, {"error": "image inconnue"})
        if path.startswith("/api/proj/"):  # /api/proj/<session>/<project>/<path> : project files, for the live preview
            import mimetypes
            from urllib.parse import unquote
            try:
                sid, pid, fpath = unquote(path[len("/api/proj/"):]).split("/", 2)
                s = sessions.get(sid) or get_session(sid)
                files = s.projects[pid]["files"]
                img = s.asset_file(pid, fpath) if fpath else None
                if img and os.path.exists(img):  # generated image used by the project
                    return self.send(200, open(img, "rb").read(), mimetypes.guess_type(fpath)[0] or "image/png")
                body = files.get(fpath) if fpath else None
                if body is None:
                    return self.send(404, b"not found", "text/plain")
                ctype = mimetypes.guess_type(fpath)[0] or "text/plain"
                return self.send(200, body.encode(), ctype + ("; charset=utf-8" if ctype.startswith(("text/", "application/javascript", "application/json", "image/svg")) else ""))
            except (AttributeError, KeyError, ValueError):
                return self.send(404, b"not found", "text/plain")
        if path == "/api/projzip":  # whole project as a ZIP
            import zipfile
            s = sessions.get(q.get("id")) or get_session(q.get("id", ""))
            p = (s.projects if s else {}).get(q.get("project"))
            if not p:
                return self.send(404, {"error": "projet inconnu"})
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for fp, content in p["files"].items():
                    z.writestr(f"{q['project']}/{fp}", content)
                for fp in p.get("assets", {}):
                    src = s.asset_file(q["project"], fp)
                    if src and os.path.exists(src):
                        z.write(src, f"{q['project']}/{fp}")
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{q["project"]}.zip"')
            self.send_header("Content-Length", str(buf.tell()))
            self.end_headers()
            self.wfile.write(buf.getvalue())
            return
        if path == "/api/settings":
            known ={m["ckpt"] for m in catalog.CATALOG if m["kind"] == "image"}
            others = [c for c in checkpoints() if c not in known and not catalog.blocked(c)]
            return self.send(200, {"settings": harness.SETTINGS, "catalog": catalog.listing(harness.SETTINGS),
                                   "otherCheckpoints": others, "active": {"llm": llm_state["name"], "voice": voice_model()[0]["id"]},
                                   "syscheck": syscheck.current(), "onboarded": bool(harness.SETTINGS.get("onboarded"))})
        if path == "/api/syscheck":
            return self.send(200, syscheck.current())
        if path == "/api/bot":
            import companion
            e5 = catalog.BY_ID["e5-small"]
            if not catalog.installed(e5) and not catalog.jobs.get("e5-small"):
                catalog.install("e5-small")  # memory search model (0.5 GB), downloaded on first use
            return self.send(200, companion.snapshot() | {"embedJob": catalog.jobs.get("e5-small")})
        if path == "/api/voice":
            m, ready = voice_model()
            return self.send(200, {"id": m["id"], "label": m["label"], "size_gb": m["size_gb"], "ready": ready,
                                   "job": catalog.jobs.get(m["id"])})
        if path == "/api/localscan":  # models already on this PC (LM Studio, Ollama...)
            return self.send(200, localscan.status())
        if path == "/api/version":
            return self.send(200, updater.status())
        if path == "/api/switch":
            return self.send(200, switch | {"elapsed": round(time.time() - switch["t0"], 1) if switch["t0"] else 0})
        if path == "/api/hf/search":
            try:
                return self.send(200, catalog.hf_search(q.get("q", ""), q.get("kind", "llm")))
            except Exception as e:
                return self.send(502, {"error": str(e)})
        if path == "/api/hf/files":
            try:
                return self.send(200, catalog.hf_files(q.get("repo", ""), q.get("kind", "llm")))
            except Exception as e:
                return self.send(502, {"error": str(e)})
        self.send(404, {"error": "introuvable"})

    def do_POST(self):
        if self.path.startswith("/api/transcribe"):  # body = raw WAV recorded by the UI
            data = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            m, ready = voice_model()
            if not ready:
                return self.send(409, {"error": "dictée vocale non installée", "id": m["id"]})
            try:
                import stt
                t0 = time.time()
                text = stt.transcribe(data, m["id"], self.query().get("lang") or None)
                harness.record_speed("img:" + m["id"], (time.time() - t0) * 10 / max(1.0, len(data) / 32000))
                return self.send(200, {"text": text})
            except Exception as e:
                return self.send(500, {"error": f"transcription impossible : {e}"})
        b = self.body()
        if self.path == "/api/new":
            s = harness.Session(ephemeral=bool(b.get("ephemeral")))
            # a previous ephemeral conversation disappears as soon as another one is opened
            for k in [k for k, v in sessions.items() if v.ephemeral]:
                sessions.pop(k).delete()
            sessions[s.id] = s
            return self.send(200, {"id": s.id, "ephemeral": s.ephemeral})
        if self.path == "/api/delete":
            s = sessions.pop(b.get("id"), None) or get_session(b.get("id", ""))
            sessions.pop(b.get("id"), None)
            if s:
                s.delete()
            return self.send(200, {"ok": True})
        if self.path == "/api/install":
            for mid in b.get("ids") or [b.get("id")]:
                catalog.install(mid)
            return self.send(200, {"ok": True})
        if self.path == "/api/syscheck":
            return self.send(200, syscheck.run())
        if self.path == "/api/projrun":  # runs a Python project, only when the user clicks Run
            s = sessions.get(b.get("id")) or get_session(b.get("id") or "")
            p = (s.projects if s else {}).get(b.get("project"))
            if not p:
                return self.send(404, {"error": "projet inconnu"})
            return self.send(200, run_project(s, b["project"], p))
        if self.path.startswith("/api/bot/"):
            return self.bot(self.path[len("/api/bot/"):], b)
        if self.path == "/api/onboarded":
            harness.SETTINGS["onboarded"] = True
            harness.save_settings()
            return self.send(200, {"ok": True})
        if self.path == "/api/use":
            return self.send(200, {"ok": use_model(b.get("id"))})
        if self.path == "/api/localscan":
            localscan.start()
            return self.send(200, {"ok": True})
        if self.path == "/api/localscan/use":
            try:
                mid = localscan.use(b.get("key"))
            except Exception as e:
                return self.send(400, {"error": str(e)})
            return self.send(200, {"id": mid, "label": catalog.BY_ID[mid]["label"], "kind": catalog.BY_ID[mid]["kind"],
                                   "ok": use_model(mid)})
        if self.path == "/api/version/check":
            updater.check(force=True)
            return self.send(200, updater.status())
        if self.path == "/api/update":
            ok, msg = updater.update()
            return self.send(200 if ok else 409, {"ok": ok, "message": msg})
        if self.path == "/api/hf/install":
            try:
                return self.send(200, {"id": catalog.hf_install(b["repo"], b["path"], b.get("kind", "llm"))})
            except Exception as e:
                return self.send(502, {"error": str(e)})
        if self.path == "/api/model/delete":
            mid = b.get("id")
            if mid in (llm_state["name"],) or catalog.BY_ID.get(mid, {}).get("ckpt") == harness.SETTINGS["ckpt"]:
                return self.send(409, {"error": "modèle en cours d'utilisation"})
            catalog.delete(mid)
            return self.send(200, {"ok": True})
        if self.path == "/api/settings":
            new = b.get("settings", {})
            for k in ("steps", "cfg", "max_steps"):
                if k in new:
                    harness.SETTINGS[k] = new[k]
            if new.get("think") in harness.THINK:
                harness.SETTINGS["think"] = new["think"]
            if "autotest" in new:
                harness.SETTINGS["autotest"] = bool(new["autotest"])
            if new.get("ui_lang") in ("en", "fr"):
                harness.SETTINGS["ui_lang"] = new["ui_lang"]
            harness.save_settings()
            return self.send(200, {"ok": True})
        if self.path != "/api/chat":
            return self.send(404, {"error": "introuvable"})

        s = sessions.get(b.get("id")) or get_session(b.get("id") or "")
        if s is None:
            s = harness.Session()
            sessions[s.id] = s
        # event stream (Server-Sent Events)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def emit(e):
            try:
                self.wfile.write(("data: " + json.dumps(e, ensure_ascii=False) + "\n\n").encode())
                self.wfile.flush()
            except OSError:
                pass  # the tab was closed: the turn continues and stays in the history

        emit({"type": "ack", "id": s.id, "busy": s.lock.locked()})  # received by the server
        if not llm_state["name"]:
            emit({"type": "error", "text": "Aucun assistant n'est installé : ouvre Paramètres > Bibliothèque de modèles."})
            emit({"type": "done"})
            return
        uploads = [decode_image(u).convert("RGB") for u in b.get("images", [])]
        mask = decode_image(b["mask"]).convert("RGB") if b.get("mask") else None
        s.turn(b.get("text", ""), uploads, mask, b.get("maskFor"), emit)


    def stream(self):
        """Opens a Server-Sent Events response and returns an emit(event) function."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def emit(e):
            try:
                self.wfile.write(("data: " + json.dumps(e, ensure_ascii=False) + "\n\n").encode())
                self.wfile.flush()
            except OSError:
                pass
        return emit

    def bot(self, action, b):
        """Sparky Bot: companion with long-term memory (see companion.py)."""
        import companion
        S = companion.S
        if action in ("chat", "greet"):
            if not llm_state["name"]:
                return self.send(409, {"error": "Aucun assistant n'est installé : ouvre Paramètres > Bibliothèque de modèles."})
            emit = self.stream()
            try:
                if action == "chat":
                    companion.chat((b.get("text") or "").strip(), emit)
                else:
                    t = companion.greeting(emit)
                    emit({"type": "reply", "text": t, "bond": companion.bond()} if t else {"type": "noop"})
            except Exception as e:
                emit({"type": "error", "text": f"assistant indisponible : {e}"})
            emit({"type": "done"})
            return
        if action == "memory":  # add / edit / delete / pin one memory
            if b.get("delete"):
                S.delete_memory(int(b["id"]))
            elif b.get("id"):
                S.update_memory(int(b["id"]), b.get("text"), b.get("category"), b.get("importance"), b.get("pinned"))
            else:
                S.add_memory(b.get("text", ""), b.get("category", "other"), int(b.get("importance") or 4), source="user")
            return self.send(200, {"ok": True})
        if action == "style":
            S.set("style", {k: v for k, v in (b.get("style") or {}).items() if v})
            return self.send(200, {"ok": True})
        if action == "profile":
            p = companion.profile()
            p.update({k: b[k] for k in ("name", "persona", "custom", "memory_on") if k in b})
            S.set("profile", p)
            return self.send(200, {"ok": True})
        if action == "reflect":
            return self.send(200, {"ok": companion.reflect(force=True)})
        if action == "forget":
            S.forget_all()
            return self.send(200, {"ok": True})
        return self.send(404, {"error": "introuvable"})


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=config.APP_PORT)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-comfy", action="store_true")
    a = ap.parse_args()
    config.APP_PORT = a.port  # the automatic project tests open the preview on this port
    harness.load_settings()
    if harness.SETTINGS.get("llm") and harness.SETTINGS["llm"] not in LLMS:
        harness.SETTINGS["llm"] = None  # old setting pointing to a model removed from the catalog
    shutil.rmtree(os.path.join(harness.DATA, "_tmp"), ignore_errors=True)  # leftovers from ephemeral conversations
    syscheck.current()
    import companion  # Sparky Bot: opens its memory and starts the background reflection loop
    if not a.no_comfy:
        start_comfy()
    if not a.no_llm:
        try:
            if alive(harness.LLM + "/health"):  # llama-server still running from a previous session
                start_llm(harness.SETTINGS["llm"])
            else:  # load through the switch so the UI shows the loading steps
                ok, rec = installed_llms(), syscheck.current().get("recommended", {}).get("llm")
                pick = next((i for i in (harness.SETTINGS["llm"], rec) if i in ok), ok[0] if ok else None)
                if pick:
                    use_model(pick)
                else:
                    print("Aucun assistant installe : choisis-en un dans l'interface.")
        except Exception as e:
            print(e)
        threading.Thread(target=autostart, daemon=True).start()
    updater.start()
    print(f"Sparky {updater.current()} : http://127.0.0.1:{a.port}")
    Server(("127.0.0.1", a.port), Handler).serve_forever()
