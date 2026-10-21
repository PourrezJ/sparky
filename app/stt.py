"""Offline voice dictation: Whisper (transformers + PyTorch), loaded on demand and freed after 10 min of inactivity."""
import io, re, threading, time, wave

import numpy as np

import catalog

_st = {"id": None, "pipe": None, "used": 0.0}
_lock = threading.Lock()
IDLE = 600


def _device(torch):
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _load(mid):
    import torch
    from transformers import pipeline
    m = catalog.BY_ID[mid]
    device, dtype = _device(torch)
    try:
        pipe = pipeline("automatic-speech-recognition", model=m["dir"], dtype=dtype, device=device)
    except (RuntimeError, ValueError, NotImplementedError):
        if device != "mps":
            raise
        # A missing Metal kernel must not break dictation on the Mac.
        pipe = pipeline("automatic-speech-recognition", model=m["dir"], dtype=torch.float32, device="cpu")
        device = "cpu"
    _st["pipe"] = pipe
    _st["id"] = mid
    _st["device"] = device


def _unload_when_idle():
    while True:
        time.sleep(60)
        with _lock:
            if _st["pipe"] is not None and time.time() - _st["used"] > IDLE:
                device = _st.get("device", "cpu")
                _st.update(pipe=None, id=None, device=None)
                try:
                    import torch
                    if device == "cuda:0":
                        torch.cuda.empty_cache()
                    elif device == "mps" and hasattr(torch, "mps"):
                        torch.mps.empty_cache()
                except Exception:
                    pass


threading.Thread(target=_unload_when_idle, daemon=True).start()


def read_wav(data):
    """16-bit PCM WAV sent by the UI -> float32 mono 16 kHz."""
    with wave.open(io.BytesIO(data)) as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        x = np.linspace(0, len(pcm) - 1, int(len(pcm) * 16000 / sr))
        pcm = np.interp(x, np.arange(len(pcm)), pcm).astype(np.float32)
    return pcm


# phrases Whisper invents on silence or noise (seen in its subtitle training data)
ARTEFACT = re.compile(r"^\s*(thanks for watching!?|merci d'avoir regardé.*|sous-titres? .*|sous-titrage .*|.*amara\.org.*|"
                      r"abonnez-vous.*|♪+|\.+)\s*$", re.I)


def clean(text):
    """Removes subtitle artifacts and sentences repeated in a loop (Whisper hallucinations)."""
    out, reps = [], 0
    for p in re.split(r"(?<=[.!?])\s+", text.strip()):
        if ARTEFACT.match(p):
            continue
        if out and p.strip().lower() == out[-1].strip().lower():
            reps += 1  # the same sentence repeated back to back
            continue
        out.append(p)
    if reps >= 2 and len(out) == 1 and len(out[0].split()) <= 3:
        return ""  # "Thank you. Thank you. Thank you..." on noise: nothing was said
    return " ".join(out).strip()


def transcribe(data, mid, language=None):
    return transcribe_array(read_wav(data), mid, language)


def transcribe_array(audio, mid, language=None):
    """float32 mono 16 kHz -> text (also used for attached audio and video files)."""
    if len(audio) < 16000 * 0.3 or float(np.abs(audio).max()) < 0.01:
        return ""  # too short or silent: Whisper would make up text
    with _lock:
        if _st["id"] != mid:
            _load(mid)
        _st["used"] = time.time()
        kw = {"task": "transcribe"} | ({"language": language} if language else {})
        # beyond 30 s, Whisper chains the windows itself (long, timestamped transcription)
        params = {"generate_kwargs": kw, "return_timestamps": len(audio) > 30 * 16000}
        try:
            out = _st["pipe"]({"raw": audio, "sampling_rate": 16000}, **params)
        except (RuntimeError, NotImplementedError):
            if _st.get("device") != "mps":
                raise
            # Some Whisper ops are not implemented in PyTorch MPS releases.
            import torch
            from transformers import pipeline
            _st["pipe"] = pipeline("automatic-speech-recognition", model=catalog.BY_ID[mid]["dir"],
                                   dtype=torch.float32, device="cpu")
            _st["device"] = "cpu"
            out = _st["pipe"]({"raw": audio, "sampling_rate": 16000}, **params)
        _st["used"] = time.time()
    return clean(out["text"])
