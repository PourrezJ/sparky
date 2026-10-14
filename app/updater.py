"""Sparky version and updates.

The version lives in the VERSION file at the project root. Once an hour (and at startup) Sparky reads
the VERSION file of the GitHub repository; when it is newer, the UI offers the update:
  - git install, no local change: "git pull --ff-only" then Sparky restarts by itself;
  - local changes, or an install from the ZIP: the UI explains why and gives the command to run.
"""
import os, re, shutil, subprocess, sys, threading, time, urllib.request

import config

REPO = "anisayari/sparky"
REMOTE_VERSION = os.environ.get("SPARKY_VERSION_URL") or f"https://raw.githubusercontent.com/{REPO}/main/VERSION"
INSTALL_CMD = f"irm https://raw.githubusercontent.com/{REPO}/main/install.ps1 | iex"
NOWIN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
CHECK_EVERY = 3600

state = {"latest": None, "checked": 0.0, "error": "", "updating": False, "log": ""}


def current():
    try:
        return open(os.path.join(config.ROOT, "VERSION"), encoding="utf-8").read().strip()
    except OSError:
        return "0.0.0"


def _num(v):
    return tuple(int(x) for x in re.findall(r"\d+", v or "")[:3]) or (0,)


def _git(*args, timeout=60):
    r = subprocess.run(["git", "-C", config.ROOT, *args], capture_output=True, text=True, timeout=timeout,
                       creationflags=NOWIN, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout + r.stderr  # not stripped: "git status --porcelain" lines start with a space


def is_git():
    return bool(shutil.which("git")) and os.path.isdir(os.path.join(config.ROOT, ".git"))


def local_changes():
    """Tracked files modified by hand (untracked files do not block a fast-forward pull)."""
    if not is_git():
        return []
    code, out = _git("status", "--porcelain", "--untracked-files=no")
    return [l[3:].strip() for l in out.splitlines() if len(l) > 3] if code == 0 else []


def check(force=False):
    if not force and time.time() - state["checked"] < CHECK_EVERY:
        return
    state["checked"] = time.time()
    try:
        req = urllib.request.Request(REMOTE_VERSION + f"?t={int(time.time())}", headers={"Cache-Control": "no-cache"})
        with urllib.request.urlopen(req, timeout=10) as r:
            state.update(latest=r.read().decode().strip(), error="")
    except Exception as e:  # offline: Sparky works without Internet, so this is not an error for the user
        state["error"] = str(e)


def loop():
    while True:
        try:
            check()
        except Exception:
            pass
        time.sleep(300)


def status():
    cur, latest = current(), state["latest"]
    avail = bool(latest) and _num(latest) > _num(cur)
    git, changes = is_git(), local_changes() if avail else []
    return {"version": cur, "latest": latest, "available": avail, "git": git, "changes": changes,
            "auto": avail and git and not changes, "updating": state["updating"], "log": state["log"],
            "command": INSTALL_CMD, "root": config.ROOT}


def _restart():
    """Starts a new Sparky once this process has released its port, then exits."""
    ps = os.path.join(config.ROOT, "start.ps1")
    py = sys.executable.replace("'", "''")
    port = config.APP_PORT
    script = (f"Start-Sleep -Seconds 2; Start-Process '{py}' -ArgumentList 'app\\server.py','--port','{port}' "
              f"-WorkingDirectory '{config.ROOT}' -WindowStyle Hidden "
              f"-RedirectStandardOutput '{os.path.join(config.DATA, 'server.log')}' "
              f"-RedirectStandardError '{os.path.join(config.DATA, 'server.err.log')}'")
    if os.name != "nt" or not os.path.exists(ps):
        return
    subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                     creationflags=NOWIN | getattr(subprocess, "DETACHED_PROCESS", 0), close_fds=True)
    time.sleep(0.5)
    os._exit(0)


def update():
    """git pull --ff-only, then restart. Returns (ok, message)."""
    s = status()
    if not s["available"]:
        return False, "déjà à jour"
    if not s["git"]:
        return False, "installation sans git : relance la commande d'installation"
    if s["changes"]:
        return False, "modifications locales : " + ", ".join(s["changes"][:8])
    state.update(updating=True, log="")
    code, out = _git("pull", "--ff-only", timeout=180)
    out = out.strip()
    state["log"] = out[-1500:]
    if code != 0:
        state["updating"] = False
        return False, "git pull a échoué : " + out[-400:]
    # new Python packages, if any (quick when nothing changed)
    req = os.path.join(config.ROOT, "requirements.txt")
    if os.path.exists(req):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req], creationflags=NOWIN, timeout=900)
    threading.Thread(target=_restart, daemon=True).start()
    return True, out[-400:]


def start():
    threading.Thread(target=loop, daemon=True).start()
