"""Apple Silicon hardware budgeting and conservative Metal model estimates."""
import json
from types import SimpleNamespace

import catalog
import config
import syscheck


def mac(ram, chip="Apple M3 Pro"):
    hw = {"os": "Darwin 26", "arch": "arm64", "cpu": chip, "cores": 12, "ram_gb": ram,
          "gpus": [{"name": chip, "vendor": "apple", "driver": "Metal", "vram_gb": ram, "unified": True}],
          "disk_free_gb": 250}
    return {"hw": hw, "assess": syscheck.assess(hw)}


def test_mac_scan_system_profiler_and_unified_gpu(monkeypatch):
    monkeypatch.setattr(syscheck.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(syscheck.platform, "machine", lambda: "arm64")
    calls = []

    def fake_run(cmd, timeout=10):
        calls.append(cmd)
        if cmd[0] == "system_profiler":
            return json.dumps({"SPHardwareDataType": [{"chip_type": "Apple M4 Pro"}]})
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(syscheck, "_run", fake_run)
    monkeypatch.setattr(syscheck, "shutil", SimpleNamespace(disk_usage=lambda _: SimpleNamespace(free=90 * 2**30)))
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=32 * 2**30))
    info = syscheck.detect()
    assert info["cpu"] == "Apple M4 Pro"
    assert info["gpus"] == [{"name": "Apple M4 Pro", "vendor": "apple", "driver": "Metal",
                             "vram_gb": 32.0, "unified": True}]
    assert syscheck.assess(info)["bandwidth"] == 273
    assert all(c[0] != "nvidia-smi" for c in calls)


def test_mac_scan_without_system_profiler_still_detects_metal(monkeypatch):
    monkeypatch.setattr(syscheck.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(syscheck.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(syscheck.platform, "processor", lambda: "")
    monkeypatch.setattr(syscheck, "_run", lambda *args, **kwargs: "")
    monkeypatch.setattr(syscheck, "shutil", SimpleNamespace(disk_usage=lambda _: SimpleNamespace(free=2**30)))
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=16 * 2**30))
    info = syscheck.detect()
    assert info["gpus"][0]["vendor"] == "apple"
    assert info["gpus"][0]["name"] == "Apple Silicon"


def test_apple_budget_preserves_memory_for_the_os():
    for ram, expected in ((8, 4), (16, 11.2), (32, 22.4), (64, 48)):
        a = mac(ram)["assess"]
        assert a["mode"] == "unified"
        assert a["model_mem_gb"] == expected
        assert a["model_mem_gb"] < ram
    assert not mac(16)["assess"]["images"]
    assert mac(32)["assess"]["images"]


def test_mac_recommends_metal_llm_and_sdxl_when_memory_allows():
    low = mac(16)
    low["recommended"], low["notes"] = syscheck.recommend(low)
    assert low["recommended"]["llm"] in catalog.BY_ID
    assert low["recommended"].get("image") is None
    assert not any("GPU NVIDIA" in note for note in low["notes"])
    assert any("Mémoire disponible insuffisante" in note for note in low["notes"])

    high = mac(64, "Apple M4 Max")
    high["recommended"], high["notes"] = syscheck.recommend(high)
    assert high["recommended"]["voice"] == "whisper-turbo"
    assert high["recommended"]["image"] == "juggernaut"
    assert any("PyTorch MPS" in note for note in high["notes"])


def test_mac_speed_is_an_unmeasured_estimate(monkeypatch):
    model = catalog.BY_ID["qwen359"]
    info = mac(32, "Apple M4 Pro")
    estimated = catalog.indicators(model, sysinfo=info)
    assert estimated["fit"] >= 1
    assert estimated["measured"] is False
    assert estimated["tps"] < model["ref_tps"] * info["assess"]["bandwidth"] / catalog.REF_BW


def test_mtp_is_not_added_to_unverified_metal_launch(monkeypatch, tmp_path):
    draft = tmp_path / "draft.gguf"
    draft.touch()
    monkeypatch.setattr(catalog, "MTP", str(draft))
    monkeypatch.setattr(catalog, "has_vision", lambda m: False)
    monkeypatch.setattr(catalog.platform, "system", lambda: "Darwin")
    assert "--spec-type" not in catalog.llm_args(catalog.BY_ID["qwen38"])
    monkeypatch.setattr(catalog.platform, "system", lambda: "Windows")
    assert "--spec-type" in catalog.llm_args(catalog.BY_ID["qwen38"])


def test_config_expands_unix_home_and_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKY_TEST_MODELS", "llms")
    monkeypatch.setattr(config, "_cfg", {"llm_dir": str(tmp_path) + "/$SPARKY_TEST_MODELS"})
    assert config._p("llm_dir", "ignored") == str(tmp_path / "llms")
