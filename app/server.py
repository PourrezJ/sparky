"""Defend Local IA : interface de chat + harnais d'agent (harness.py) + LLM vision (llama-server) + ComfyUI.

Usage : python app\\server.py [--port 7860] [--no-llm]   (ou start.bat)
Le modele d'assistant, le modele d'image et les reglages se choisissent dans l'interface (Parametres).
"""
import argparse, base64, io, json, os, shutil, subprocess, sys, threading, time, urllib.request
import http.server, socketserver

import psutil
from PIL import Image

import catalog
import config
import harness
import syscheck

HERE = os.path.dirname(os.path.abspath(__file__))
NOWIN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
LLM_LOG = os.path.join(config.DATA, "llm.log")
LLMS = catalog.BY_ID    # tous les modeles (catalogue + ajouts Hugging Face), filtres par kind a l'usage
sessions = {}           # id -> Session (en memoire)
llm_state = {"name": None}
# changement de modele en cours, suivi par l'interface (spinner + etapes)
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
    """Lance ComfyUI (generation d'images) s'il ne tourne pas deja."""
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
    try:  # libere la VRAM de ComfyUI avant de charger un gros modele (rechargee a la prochaine image)
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
    """Des qu'un modele vient d'etre installe et qu'aucun n'est actif, on l'active tout seul."""
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
                if imgs:
                    rec = syscheck.current().get("recommended", {}).get("image")
                    m = next((x for x in imgs if x["id"] == rec), imgs[0])
                    harness.SETTINGS["ckpt"] = m["ckpt"]
                    harness.save_settings()
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
            try:  # ComfyUI repond par un corps vide : ce n'est pas une erreur
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
                harness._comfy_run(warm)  # premiere image minuscule : force le chargement du checkpoint
        harness.SETTINGS["llm" if m["kind"] == "llm" else "ckpt"] = mid if m["kind"] == "llm" else m["ckpt"]
        harness.save_settings()
        _step(None)
        switch["phase"] = "ready"
    except Exception as e:
        if switch["steps"]:
            switch["steps"][-1]["state"] = "err"
        switch.update(phase="error", error=str(e))


def use_model(mid):
    m = catalog.BY_ID.get(mid)
    if not m or not catalog.installed(m) or switch["phase"] == "loading":
        return False
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
    if time.time() - _gpu["t"] > 1.5:  # nvidia-smi est lent : au plus une lecture toutes les 1,5 s
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
                      "active": harness.ACTIVITY["image"] > 0 or comfy["running"] > 0, "queue": comfy["pending"]}}


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
        if path == "/logo.svg":
            return self.send(200, open(os.path.join(HERE, "logo.svg"), "rb").read(), "image/svg+xml")
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
        if path == "/api/settings":
            known = {m["ckpt"] for m in catalog.CATALOG if m["kind"] == "image"}
            others = [c for c in checkpoints() if c not in known and not catalog.blocked(c)]
            return self.send(200, {"settings": harness.SETTINGS, "catalog": catalog.listing(harness.SETTINGS),
                                   "otherCheckpoints": others, "active": {"llm": llm_state["name"]},
                                   "syscheck": syscheck.current(), "onboarded": bool(harness.SETTINGS.get("onboarded"))})
        if path == "/api/syscheck":
            return self.send(200, syscheck.current())
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
        b = self.body()
        if self.path == "/api/new":
            s = harness.Session(ephemeral=bool(b.get("ephemeral")))
            # une conversation ephemere precedente disparait des qu'on en ouvre une autre
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
        if self.path == "/api/onboarded":
            harness.SETTINGS["onboarded"] = True
            harness.save_settings()
            return self.send(200, {"ok": True})
        if self.path == "/api/use":
            return self.send(200, {"ok": use_model(b.get("id"))})
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
            harness.save_settings()
            return self.send(200, {"ok": True})
        if self.path != "/api/chat":
            return self.send(404, {"error": "introuvable"})

        s = sessions.get(b.get("id")) or get_session(b.get("id") or "")
        if s is None:
            s = harness.Session()
            sessions[s.id] = s
        # flux d'evenements (Server-Sent Events)
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
                pass  # l'onglet a ete ferme : le tour continue et reste dans l'historique

        emit({"type": "ack", "id": s.id, "busy": s.lock.locked()})  # recu par le serveur
        if not llm_state["name"]:
            emit({"type": "error", "text": "Aucun assistant n'est installé : ouvre Paramètres > Bibliothèque de modèles."})
            emit({"type": "done"})
            return
        uploads = [decode_image(u).convert("RGB") for u in b.get("images", [])]
        mask = decode_image(b["mask"]).convert("RGB") if b.get("mask") else None
        s.turn(b.get("text", ""), uploads, mask, b.get("maskFor"), emit)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=config.APP_PORT)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-comfy", action="store_true")
    a = ap.parse_args()
    harness.load_settings()
    if harness.SETTINGS.get("llm") and harness.SETTINGS["llm"] not in LLMS:
        harness.SETTINGS["llm"] = None  # ancien reglage vers un modele retire du catalogue
    shutil.rmtree(os.path.join(harness.DATA, "_tmp"), ignore_errors=True)  # restes de conversations ephemeres
    syscheck.current()
    if not a.no_comfy:
        start_comfy()
    if not a.no_llm:
        try:
            start_llm(harness.SETTINGS["llm"])
        except Exception as e:
            print(e)
    if not a.no_llm:
        threading.Thread(target=autostart, daemon=True).start()
    print(f"Defend Local IA : http://127.0.0.1:{a.port}")
    Server(("127.0.0.1", a.port), Handler).serve_forever()
