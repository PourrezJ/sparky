"""Hardware -> recommended models (regression: what each kind of PC gets by default)."""
import pytest

import syscheck


def machine(gpu=None, vram=0, ram=32, disk=500):
    gpus = [{"name": gpu, "vram_gb": vram, "driver": "600", "vendor": "nvidia"}] if gpu else []
    hw = {"os": "Windows 11", "arch": "AMD64", "cpu": "x", "cores": 8, "ram_gb": ram, "gpus": gpus, "disk_free_gb": disk}
    info = {"hw": hw, "assess": syscheck.assess(hw)}
    info["recommended"], info["notes"] = syscheck.recommend(info)
    return info


def test_rtx_5090_gets_the_smartest_model_and_fast_images():
    i = machine("NVIDIA GeForce RTX 5090", 31.8, 64)
    assert i["assess"]["tier_label"] == "Excellente"
    assert i["recommended"] == {"llm": "qwen38", "image": "klein4b", "voice": "whisper-turbo"} and not i["notes"]


@pytest.mark.parametrize("gpu,vram", [("NVIDIA GeForce RTX 4060", 8), ("NVIDIA GeForce RTX 3060", 12)])
def test_mid_range_gpus_get_a_model_that_fits(gpu, vram):
    i = machine(gpu, vram)
    import catalog
    llm = catalog.BY_ID[i["recommended"]["llm"]]
    assert llm["size_gb"] + 2.5 <= i["assess"]["model_mem_gb"]
    assert i["recommended"]["image"] and i["recommended"]["voice"] == "whisper-turbo"


def test_no_gpu_runs_a_small_model_without_images():
    i = machine(None, ram=16)
    assert i["assess"]["mode"] == "cpu" and i["assess"]["tier"] == 1
    assert "image" not in i["recommended"] and i["recommended"]["voice"] == "whisper-base"
    assert any("processeur" in n for n in i["notes"])


def test_low_disk_space_is_reported():
    assert any("Espace disque faible" in n for n in machine("NVIDIA GeForce RTX 5090", 31.8, 64, disk=12)["notes"])


def test_unified_memory_machines_keep_memory_for_windows():
    i = machine("NVIDIA GB10 Spark", 120, 128)
    assert i["assess"]["mode"] == "unified" and i["assess"]["model_mem_gb"] == 118
