"""Portable speech backend selection without requiring access to a Mac."""
import sys
from types import SimpleNamespace

import numpy as np

import stt
import tts


def fake_torch(cuda=False, mps=False):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                           backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
                           float16="float16", float32="float32")


def test_whisper_prefers_cuda_then_mps_then_cpu():
    assert stt._device(fake_torch(cuda=True, mps=True)) == ("cuda:0", "float16")
    assert stt._device(fake_torch(mps=True)) == ("mps", "float16")
    assert stt._device(fake_torch()) == ("cpu", "float32")


def test_whisper_falls_back_if_mps_pipeline_cannot_load(monkeypatch):
    attempts = []

    def pipeline(task, *, model, dtype, device):
        attempts.append(device)
        if device == "mps":
            raise RuntimeError("MPS operator unsupported")
        return lambda *a, **kw: {"text": "Hello."}

    monkeypatch.setitem(sys.modules, "torch", fake_torch(mps=True))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=pipeline))
    monkeypatch.setattr(stt, "_st", {"id": None, "pipe": None, "used": 0.0})
    stt._load("whisper-base")
    assert attempts == ["mps", "cpu"]
    assert stt._st["device"] == "cpu"


def test_whisper_falls_back_if_mps_inference_fails(monkeypatch):
    def broken(*a, **kw):
        raise RuntimeError("not implemented on MPS")

    monkeypatch.setitem(sys.modules, "torch", fake_torch(mps=True))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        pipeline=lambda *a, **kw: lambda *args, **kwargs: {"text": "Recorded speech."}))
    monkeypatch.setattr(stt, "_st", {"id": "whisper-base", "pipe": broken, "used": 0.0, "device": "mps"})
    assert stt.transcribe_array(np.full(16000, 0.2, dtype=np.float32), "whisper-base") == "Recorded speech."
    assert stt._st["device"] == "cpu"


def test_homebrew_espeak_ng_location_fallback(monkeypatch):
    # Ensure the bundled Python loader is unavailable in this test.
    monkeypatch.setitem(sys.modules, "espeakng_loader", None)
    monkeypatch.setattr(tts.ctypes.util, "find_library", lambda name: None)
    monkeypatch.setattr(tts.ctypes, "CDLL", lambda path: object())
    monkeypatch.setattr(tts.os.path, "isdir", lambda p: p.endswith("/share/espeak-ng-data"))
    with monkeypatch.context() as patch:
        patch.setattr(tts.sys, "platform", "darwin")
        path, data = tts._espeak_location()
    assert path == "/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib"
    assert data == "/opt/homebrew/opt/espeak-ng/share/espeak-ng-data"
