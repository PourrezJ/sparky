"""Portable project checker: bundled Node/npm and macOS browser discovery."""
import json
import os
from types import SimpleNamespace

import pytest

import checker

# Simulating sys.platform does not change os.name, pathlib or the Windows
# subprocess implementation. Run the actual POSIX behavior on Linux/macOS.
pytestmark = pytest.mark.skipif(os.name == "nt", reason="macOS/POSIX paths require a POSIX test host")


def test_macos_bundled_node_and_npm_are_both_used(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    node_dir = runtime / "node"
    node_dir.mkdir(parents=True)
    (node_dir / "node").touch()
    (node_dir / "npm").touch()
    monkeypatch.setattr(checker.config, "RUNTIME", str(runtime))
    monkeypatch.setattr(checker.sys, "platform", "darwin")
    assert checker.node_dir() == str(node_dir)
    commands = []
    monkeypatch.setattr(checker, "run", lambda cmd, cwd, timeout: (commands.append(cmd) or (0, "", False)))
    session = SimpleNamespace(dir=str(tmp_path), projects={}, id="test")
    project = {
        "entry": "server.js",
        "files": {
            "server.js": "console.log('ok')",
            "package.json": json.dumps({
                "dependencies": {"express": "1.0"},
                "scripts": {"build": "node --version", "test": "node --version"},
            }),
        },
    }
    result = checker.check(session, "p1", project, 8765)
    assert result["ok"]
    assert [os.path.basename(cmd[0]) for cmd in commands] == ["npm", "npm", "npm", "node"]
    assert commands[0][1] == "install"
    assert commands[1][1:] == ["run", "build"]
    assert commands[2][1:] == ["run", "test"]


def test_macos_browser_application_bundle_is_detected(monkeypatch):
    monkeypatch.setattr(checker.sys, "platform", "darwin")
    want = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    monkeypatch.setattr(checker.os.path, "isfile", lambda path: path == want)
    assert checker.browser_path() == want


def test_no_npm_produces_a_clear_failure(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    (runtime / "node").mkdir(parents=True)
    (runtime / "node" / "node").touch()
    monkeypatch.setattr(checker.config, "RUNTIME", str(runtime))
    monkeypatch.setattr(checker.shutil, "which", lambda name: None)
    session = SimpleNamespace(dir=str(tmp_path), projects={}, id="test")
    project = {
        "entry": "main.js",
        "files": {"main.js": "console.log('hello')", "package.json": '{"dependencies":{"express":"1.0"}}'},
    }
    result = checker.check(session, "p2", project, 8765)
    assert not result["ok"] and "npm is not installed" in result["report"]
