"""Intel macOS and Hackintosh default to conservative CPU inference."""
import json
from types import SimpleNamespace

import catalog
import config
import server
import syscheck


def intel_machine(ram):
    hw = {"os": "Darwin 26", "arch": "x86_64", "cpu": "Intel Core i5", "cores": 4,
          "ram_gb": ram, "disk_free_gb": 100,
          "gpus": [{"name": "Intel UHD Graphics 620", "vendor": "other", "vram_gb": 0, "driver": ""}]}
    return {"hw": hw, "assess": syscheck.assess(hw)}


def test_intel_mac_model_budget():
    for ram in (8, 16):
        info = intel_machine(ram)
        assert info["assess"]["mode"] == "cpu"
        assert info["assess"]["model_mem_gb"] < ram
        assert not info["assess"]["images"]
        rec, notes = syscheck.recommend(info)
        assert rec["llm"] in ("qwen3508", "qwen352", "qwen354")
        assert rec["voice"] == "whisper-base"
        assert any("Mac Intel" in msg for msg in notes)


def test_explicit_gpu_layers_override(monkeypatch, tmp_path):
    executable = tmp_path / "llama-server"
    executable.touch()
    monkeypatch.setattr(config, "LLAMA_SERVER", str(executable))
    monkeypatch.setattr(config, "_cfg", {"gpu_layers": 0})
    monkeypatch.setattr(catalog, "llm_args", lambda model: ["test.gguf"])
    monkeypatch.setattr(server.syscheck, "current", lambda: {"assess": {"mode": "unified", "model_mem_gb": 10}})
    spawned = []
    monkeypatch.setattr(server, "_spawn", lambda argv, log: spawned.append(argv))
    server.launch_llm({"label": "test", "size_gb": 1.3})
    argv = spawned[0]
    assert argv[argv.index("-ngl") + 1] == "0"
    assert argv[argv.index("-fa") + 1] == "off"


def test_intel_hardware_scan(monkeypatch, tmp_path):
    monkeypatch.setattr(syscheck.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(syscheck.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(config, "LLM_DIR", str(tmp_path))
    def fake_run(cmd, timeout=10):
        return ("Intel Core i5" if cmd[0] == "sysctl" else
                json.dumps({"SPDisplaysDataType": [{"sppci_model": "Intel UHD Graphics 620"}]})
                if cmd[0] == "system_profiler" else "")
    monkeypatch.setattr(syscheck, "_run", fake_run)
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=8 * 2**30))
    info = syscheck.detect()
    assert info["cpu"] == "Intel Core i5"
    assert info["gpus"][0]["vendor"] == "other"
    assert syscheck.assess(info)["mode"] == "cpu"
