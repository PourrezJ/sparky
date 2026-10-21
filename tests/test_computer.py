"""Computer mode: security levels, protected folders, file tools, approval flow in the agent."""
import os, threading, time

import pytest

import computer
import config
import harness


def test_levels():
    na = computer.needs_approval
    assert na("off", "list") == "deny" and na("off", "run") == "deny"
    assert na("read", "read") is None and na("read", "write") == "deny" and na("read", "run") == "deny"
    assert na("ask", "search") is None and na("ask", "write") == "ask" and na("ask", "trash") == "ask"
    assert na("ask_all", "list") == "ask" and na("full", "run") is None


def test_protected_folders():
    paths = (os.environ.get("SystemRoot", r"C:\Windows"),
             os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "x"), "C:\\") if os.name == "nt" else (
             "/", "/usr/bin/example", config.ROOT)
    for p in paths:
        with pytest.raises(PermissionError):
            computer.check_writable(os.path.abspath(p))


def test_places_and_resolve():
    pl = computer.places()
    assert os.path.isabs(pl["desktop"]) and computer.resolve("desktop\\a.txt") == os.path.join(pl["desktop"], "a.txt")
    if os.name != "nt":
        assert computer.resolve("documents/sub/a.txt") == os.path.join(pl["documents"], "sub", "a.txt")


def test_file_tools(tmp_path):
    d = str(tmp_path)
    out = computer.RUN["write"]({"path": os.path.join(d, "notes", "a.txt"), "content": "bonjour Sparky"})
    assert "créé" in out
    assert "a.txt" in computer.RUN["list"]({"path": os.path.join(d, "notes")})
    assert "bonjour Sparky" in computer.RUN["read"]({"path": os.path.join(d, "notes", "a.txt")})
    assert "1 fichier" in computer.RUN["search"]({"root": d, "name": "*.txt", "text": "sparky"})
    computer.RUN["copy"]({"src": os.path.join(d, "notes", "a.txt"), "dst": os.path.join(d, "b.txt")})
    computer.RUN["move"]({"src": os.path.join(d, "b.txt"), "dst": os.path.join(d, "c.txt")})
    assert os.path.exists(os.path.join(d, "c.txt")) and not os.path.exists(os.path.join(d, "b.txt"))
    with pytest.raises(FileExistsError):
        computer.RUN["copy"]({"src": os.path.join(d, "c.txt"), "dst": os.path.join(d, "notes", "a.txt")})


@pytest.mark.skipif(os.name != "nt", reason="Recycle Bin and PowerShell: Windows only")
def test_trash_and_run(tmp_path):
    p = tmp_path / "old.txt"
    p.write_text("x")
    assert "corbeille" in computer.RUN["trash"]({"path": str(p)}) and not p.exists()
    out = computer.RUN["run"]({"command": "Write-Output (6*7)", "cwd": str(tmp_path)})
    assert "code de sortie 0" in out and "42" in out


class _T:
    computer, free, tools_used = 0, 0, False

    def spare(self):
        self.free += 1


def _session(level):
    s = harness.Session(ephemeral=True)
    s.config["computer"] = level
    return s


def test_agent_denied_by_level(tmp_path):
    s, evs = _session("read"), []
    s._on_computer(_T(), {"tool": "write", "path": str(tmp_path / "x.txt"), "content": "x"}, evs.append)
    assert not (tmp_path / "x.txt").exists() and "[REFUSÉ]" in s.history[-1]["text"]
    assert [e["type"] for e in evs] == ["ctool", "cresult"]


def test_agent_approval_flow(tmp_path):
    s, evs = _session("ask"), []
    target = tmp_path / "ok.txt"

    def user():  # the user clicks "Allow" on the card
        for _ in range(100):
            ap = [e for e in evs if e["type"] == "approve"]
            if ap:
                assert s.approve(ap[0]["id"], "always")
                return
            time.sleep(0.05)
    threading.Thread(target=user).start()
    s._on_computer(_T(), {"tool": "write", "path": str(target), "content": "fait"}, evs.append)
    assert target.read_text() == "fait"
    assert [e["type"] for e in evs] == ["ctool", "approve", "approved", "cresult"] and evs[2]["decision"] == "always"
    evs.clear()  # "always": the next write runs without asking
    s._on_computer(_T(), {"tool": "write", "path": str(target), "content": "encore"}, evs.append)
    assert target.read_text() == "encore" and "approve" not in [e["type"] for e in evs]


def test_agent_user_denies(tmp_path):
    s, evs = _session("ask_all"), []

    def user():
        for _ in range(100):
            ap = [e for e in evs if e["type"] == "approve"]
            if ap:
                s.approve(ap[0]["id"], "deny")
                return
            time.sleep(0.05)
    threading.Thread(target=user).start()
    s._on_computer(_T(), {"tool": "list", "path": str(tmp_path)}, evs.append)
    assert "REFUSÉ PAR L'UTILISATEUR" in s.history[-1]["text"]


def test_batch_disks_and_info(tmp_path):
    for n in ("a.txt", "b.txt", "c.png"):
        (tmp_path / n).write_text(n)
    out = computer.RUN["move"]({"paths": [str(tmp_path / "a.txt"), str(tmp_path / "b.txt")], "dst": str(tmp_path / "Notes")})
    assert out.startswith("2/2") and sorted(os.listdir(tmp_path / "Notes")) == ["a.txt", "b.txt"]
    out = computer.RUN["copy"]({"paths": [str(tmp_path / "c.png"), str(tmp_path / "missing.png")], "dst": str(tmp_path / "Img")})
    assert out.startswith("1/2") and "✗" in out
    assert "libres" in computer.RUN["info"]({"path": str(tmp_path)}) and "fichiers" in computer.RUN["info"]({"path": str(tmp_path)})
    assert "libres sur" in computer.RUN["disks"]({})
    assert computer.summary("move", {"paths": ["x", "y"], "dst": "z"}).startswith("2 éléments → z")


def test_macos_trash_preserves_collisions_and_originals(tmp_path, monkeypatch):
    monkeypatch.setattr(computer.sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    home = tmp_path / "home"
    home.mkdir()
    first = tmp_path / "one" / "notes.txt"
    second = tmp_path / "two" / "notes.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first")
    second.write_text("second")
    assert "corbeille" in computer.t_trash(str(first))
    assert "corbeille" in computer.t_trash(str(second))
    trash = home / ".Trash"
    assert not first.exists() and not second.exists()
    assert sorted(p.read_text() for p in trash.iterdir()) == ["first", "second"]
    assert len({p.name for p in trash.iterdir()}) == 2


def test_macos_open_uses_open_and_commands_use_zsh(tmp_path, monkeypatch):
    monkeypatch.setattr(computer.sys, "platform", "darwin")
    calls = []

    def popen(args, **kw):
        calls.append(("open", args, kw))

    def run(args, **kw):
        calls.append(("run", args, kw))
        from types import SimpleNamespace
        return SimpleNamespace(stdout=b"42", stderr=b"", returncode=0)

    monkeypatch.setattr(computer.subprocess, "Popen", popen)
    monkeypatch.setattr(computer.subprocess, "run", run)
    target = tmp_path / "notes.txt"
    target.write_text("hi")
    assert computer.t_open(str(target)).startswith("ouvert")
    computer.t_open("TextEdit")
    out = computer.t_run("echo 42", str(tmp_path))
    assert "code de sortie 0" in out and "42" in out
    assert calls[0][1] == ["open", str(target)]
    assert calls[1][1] == ["open", "-a", "TextEdit"]
    assert calls[2][1] == ["/bin/zsh", "-c", "echo 42"]
    assert computer.needs_approval("ask", "run") == "ask"


def test_protected_tree_cannot_be_modified_through_symlink(tmp_path):
    link = tmp_path / "shortcut"
    link.symlink_to(config.ROOT, target_is_directory=True)
    with pytest.raises(PermissionError):
        computer.check_writable(str(link / "app" / "server.py"))
