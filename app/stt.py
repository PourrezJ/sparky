"""Dictee vocale hors ligne : Whisper (transformers + PyTorch), charge a la demande et libere apres 10 min sans usage."""
import io, threading, time, wave

import numpy as np

import catalog

_st = {"id": None, "pipe": None, "used": 0.0}
_lock = threading.Lock()
IDLE = 600


def _load(mid):
    import torch
    from transformers import pipeline
    m = catalog.BY_ID[mid]
    gpu = torch.cuda.is_available()
    _st["pipe"] = pipeline("automatic-speech-recognition", model=m["dir"],
                           dtype=torch.float16 if gpu else torch.float32, device="cuda:0" if gpu else "cpu")
    _st["id"] = mid


def _unload_when_idle():
    while True:
        time.sleep(60)
        with _lock:
            if _st["pipe"] is not None and time.time() - _st["used"] > IDLE:
                _st.update(pipe=None, id=None)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass


threading.Thread(target=_unload_when_idle, daemon=True).start()


def read_wav(data):
    """WAV PCM 16 bits envoye par l'interface -> float32 mono 16 kHz."""
    with wave.open(io.BytesIO(data)) as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        x = np.linspace(0, len(pcm) - 1, int(len(pcm) * 16000 / sr))
        pcm = np.interp(x, np.arange(len(pcm)), pcm).astype(np.float32)
    return pcm


import re

# phrases que Whisper invente sur du silence ou du bruit (vues dans ses donnees de sous-titres)
ARTEFACT = re.compile(r"^\s*(thanks for watching!?|merci d'avoir regardé.*|sous-titres? .*|sous-titrage .*|.*amara\.org.*|"
                      r"abonnez-vous.*|♪+|\.+)\s*$", re.I)


def clean(text):
    """Retire les artefacts de sous-titres et les phrases repetees en boucle (hallucinations de Whisper)."""
    out, reps = [], 0
    for p in re.split(r"(?<=[.!?])\s+", text.strip()):
        if ARTEFACT.match(p):
            continue
        if out and p.strip().lower() == out[-1].strip().lower():
            reps += 1  # la meme phrase repetee d'affilee
            continue
        out.append(p)
    if reps >= 2 and len(out) == 1 and len(out[0].split()) <= 3:
        return ""  # « Thank you. Thank you. Thank you… » sur du bruit : rien n'a ete dit
    return " ".join(out).strip()


def transcribe(data, mid, language=None):
    audio = read_wav(data)
    if len(audio) < 16000 * 0.3 or float(np.abs(audio).max()) < 0.01:
        return ""  # trop court ou silence : Whisper inventerait du texte
    with _lock:
        if _st["id"] != mid:
            _load(mid)
        _st["used"] = time.time()
        kw = {"task": "transcribe"} | ({"language": language} if language else {})
        # au-dela de 30 s, Whisper enchaine lui-meme les fenetres (transcription longue, horodatee)
        out = _st["pipe"]({"raw": audio, "sampling_rate": 16000}, generate_kwargs=kw,
                          return_timestamps=len(audio) > 30 * 16000)
        _st["used"] = time.time()
    return clean(out["text"])
