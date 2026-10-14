"""Offline read-aloud: Piper voices (ONNX, CPU) run with onnxruntime, phonemized by espeak-ng (espeakng-loader).

The piper-tts package has no Windows ARM64 wheel, so its small pipeline is reproduced here:
text -> espeak-ng IPA phonemes -> phoneme ids (voice .onnx.json) -> VITS model -> 16-bit WAV.
"""
import ctypes, io, json, os, re, threading, unicodedata, urllib.request, wave

import numpy as np

import config

DIR = os.path.join(config.VOICE_DIR, "piper")
HF = "https://huggingface.co/rhasspy/piper-voices/resolve/main/{path}"
VOICES = [
    {"id": "en_US-lessac-medium", "lang": "en", "label": "English (Lessac)", "size_mb": 60},
    {"id": "fr_FR-siwis-medium", "lang": "fr", "label": "Français (Siwis)", "size_mb": 60},
    {"id": "de_DE-thorsten-medium", "lang": "de", "label": "Deutsch (Thorsten)", "size_mb": 60},
    {"id": "es_ES-davefx-medium", "lang": "es", "label": "Español (Davefx)", "size_mb": 60},
    {"id": "it_IT-paola-medium", "lang": "it", "label": "Italiano (Paola)", "size_mb": 61},
]
BY_ID = {v["id"]: v for v in VOICES}
LANG_NAMES = {"english": "en", "anglais": "en", "french": "fr", "français": "fr", "francais": "fr",
              "german": "de", "deutsch": "de", "allemand": "de", "spanish": "es", "español": "es", "espagnol": "es",
              "italian": "it", "italiano": "it", "italien": "it"}
CODE_WORD = {"en": "code block.", "fr": "bloc de code.", "de": "Codeblock.", "es": "bloque de código.",
             "it": "blocco di codice."}
SILENCE = 0.25  # seconds between sentences
MAX_PHONEMES = 400  # longer sentences are cut at spaces (memory and attention of the model)

jobs = {}
_voices = {}  # id -> (onnxruntime session, config), loaded once
_lock = threading.Lock()  # espeak-ng has a global state
_jobs_lock = threading.Lock()
_espeak = None


def _files(vid):
    base = os.path.join(DIR, vid)
    return base + ".onnx", base + ".onnx.json"


def _url(vid, ext):
    lang, rest = vid.split("-", 1)
    name, quality = rest.rsplit("-", 1)
    return HF.format(path=f"{lang.split('_')[0]}/{lang}/{name}/{quality}/{vid}{ext}")


def ready(vid):
    return all(os.path.exists(f) for f in _files(vid))


def installed():
    try:
        import onnxruntime, espeakng_loader  # noqa: F401
        return True
    except ImportError:
        return False


def status():
    return {"installed": installed(),
            "voices": [{k: v[k] for k in ("id", "lang", "label", "size_mb")}
                       | {"ready": ready(v["id"]), "job": jobs.get(v["id"])} for v in VOICES]}


def _install(vid):
    job = jobs[vid]
    try:
        os.makedirs(DIR, exist_ok=True)
        for local, ext in zip(_files(vid), (".onnx", ".onnx.json")):
            if os.path.exists(local):
                continue
            part = local + ".part"
            for _ in range(8):  # slow or flaky connection: resume where it stopped
                done = os.path.getsize(part) if os.path.exists(part) else 0
                req = urllib.request.Request(_url(vid, ext), headers={"Range": f"bytes={done}-"} if done else {})
                try:
                    with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if r.status == 206 else "wb") as f:
                        done = done if r.status == 206 else 0
                        total = done + int(r.headers.get("Content-Length") or 0)
                        while chunk := r.read(1 << 20):
                            f.write(chunk)
                            done += len(chunk)
                            if ext == ".onnx" and total:
                                job["progress"] = round(done / total, 3)
                except OSError:
                    continue
                if not total or done >= total:
                    break
            else:
                raise RuntimeError("téléchargement interrompu (connexion instable), réessaie")
            os.replace(part, local)
        job.update(state="ok", progress=1.0)
    except Exception as e:
        job.update(state="error", error=str(e))


def install(vid):
    """Downloads a voice in the background; progress in status()["voices"][i]["job"]."""
    with _jobs_lock:  # two clicks must not start two downloads writing the same file
        if vid not in BY_ID or ready(vid) or jobs.get(vid, {}).get("state") == "running":
            return
        jobs[vid] = {"state": "running", "progress": 0.0, "error": ""}
    threading.Thread(target=_install, args=(vid,), daemon=True).start()


# --- text preparation ---

def strip_markdown(text, lang="en"):
    """Markdown -> plain text to read: code blocks announced, links -> their text, tables -> cells."""
    code = " " + CODE_WORD.get(lang, CODE_WORD["en"]) + "\n"
    t = re.sub(r"<think>.*?(</think>|$)", " ", text, flags=re.S)
    t = re.sub(r"^\s*(```|~~~).*?(^\s*\1|\Z)", code, t, flags=re.S | re.M)
    t = re.sub(r"\$\$.+?\$\$", " ", t, flags=re.S)
    t = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", t)  # images -> alt text
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)  # links -> text
    t = re.sub(r"<(https?://[^>]+)>|https?://\S+", " ", t)  # raw URLs are not read
    t = re.sub(r"</?[a-zA-Z][^>]*>", " ", t)  # HTML tags
    t = re.sub(r"`([^`]*)`", r"\1", t)
    lines = []
    for line in t.splitlines():
        s = line.strip()
        if re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", s) or re.fullmatch(r"[-*_]{3,}", s):
            continue  # table separator, horizontal rule
        if s.startswith("|"):
            s = ", ".join(c.strip() for c in s.strip("|").split("|") if c.strip())
        s = re.sub(r"^(#{1,6}\s+|>\s*|[-*+]\s+|\d+[.)]\s+)+", "", s)  # headings, quotes, list markers
        s = re.sub(r"(\*\*|__|~~)(.+?)\1", r"\2", s)
        s = re.sub(r"(?<![\w*])[*_](?!\s)(.+?)(?<!\s)[*_](?![\w*])", r"\1", s)  # *italic*, not snake_case
        s = s.replace("*", "").strip()
        if s:
            lines.append(s if s[-1] in ".!?:;,…" else s + ".")  # a line break is a pause
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def norm_lang(lang):
    if not lang:
        return None
    lang = lang.strip().lower()
    return LANG_NAMES.get(lang) or re.split(r"[-_]", lang)[0][:2]


_STOP = {"fr": "le la les des est une et que pour pas vous dans ce qui sur avec",
         "en": "the is and to of you that it for are with this be not have",
         "de": "der die das und ist nicht ich sie ein eine mit zu den auf",
         "es": "el la los las es y que de en un una por para con no",
         "it": "il la di che è e un una per non sono con del della gli"}


def guess_lang(text):
    """Very small guess from common words, used when the reply language is unknown."""
    words = re.findall(r"[a-zàâäçéèêëîïôöùûüßñíóúì]+", text.lower()[:2000])
    score = {k: sum(w in set(v.split()) for w in words) for k, v in _STOP.items()}
    best = max(score, key=score.get)
    return best if score[best] else "en"


def pick_voice(lang=None, voice_id=None):
    """Voice id for a language ("fr", "French", "fr-FR"...); prefers an installed voice; English otherwise."""
    if voice_id in BY_ID:
        return voice_id
    lang = norm_lang(lang) or "en"
    cands = [v["id"] for v in VOICES if v["lang"] == lang] or [v["id"] for v in VOICES if v["lang"] == "en"]
    return next((c for c in cands if ready(c)), cands[0])


# --- espeak-ng phonemization (same calls as piper-phonemize) ---

def _espeak_lib():
    global _espeak
    if _espeak is None:
        import espeakng_loader
        lib = ctypes.cdll.LoadLibrary(espeakng_loader.get_library_path())
        lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
        lib.espeak_TextToPhonemes.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int]
        lib.terminator = hasattr(lib, "espeak_TextToPhonemesWithTerminator")  # espeak-ng >= 1.52 only
        if lib.terminator:
            lib.espeak_TextToPhonemesWithTerminator.restype = ctypes.c_char_p
            lib.espeak_TextToPhonemesWithTerminator.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                                                ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        if lib.espeak_Initialize(2, 0, espeakng_loader.get_data_path().encode(), 0) < 0:  # 2 = synchronous
            raise RuntimeError("espeak-ng initialisation failed")
        _espeak = lib
    return _espeak


def phonemize(text, voice):
    """Text -> list of sentences of IPA phonemes, with punctuation kept (the voices were trained with it)."""
    lib = _espeak_lib()
    if lib.espeak_SetVoiceByName(voice.encode()) != 0:
        raise RuntimeError("unknown espeak voice: " + voice)
    if not lib.terminator:  # older espeak-ng: sentences and clauses are split here, punctuation added back
        sentences = []
        for sent in re.findall(r"[^.!?…]+[.!?…]*", text):
            end = (re.search(r"[.!?]", sent[::-1]) or [None])[0] if sent.strip() else None
            parts = []
            for clause in filter(None, (x.strip() for x in re.split(r"[,;:]", sent))):
                buf = ctypes.create_string_buffer(clause.encode("utf-8"))
                ptr = ctypes.c_void_p(ctypes.addressof(buf))
                ph = []
                while ptr.value:
                    ph.append(re.sub(r"\([a-z-]+\)", "", (lib.espeak_TextToPhonemes(ctypes.byref(ptr), 1, 2) or b"").decode("utf-8", "ignore")).strip())
                parts.append(" ".join(filter(None, ph)))
            if any(parts):
                sentences.append(", ".join(p for p in parts if p) + (end or "."))
    else:
        sentences = _phonemize_terminator(lib, text)
    out = []
    for s in filter(None, (s.strip() for s in sentences)):
        while len(s) > MAX_PHONEMES:
            cut = s.rfind(" ", 0, MAX_PHONEMES)
            cut = cut if cut > 0 else MAX_PHONEMES
            out.append(s[:cut])
            s = s[cut:].strip()
        out.append(s)
    return out


def _phonemize_terminator(lib, text):
    buf = ctypes.create_string_buffer(text.encode("utf-8"))
    ptr, term = ctypes.c_void_p(ctypes.addressof(buf)), ctypes.c_int(0)
    sentences, cur = [], []
    while ptr.value:
        ph = lib.espeak_TextToPhonemesWithTerminator(ctypes.byref(ptr), 1, 2, ctypes.byref(term))  # UTF-8, IPA
        cur.append(re.sub(r"\([a-z-]+\)", "", (ph or b"").decode("utf-8", "ignore")).strip())
        t = term.value
        intonation, kind = t & 0x7000, t & 0xF0000
        if kind == 0x80000:  # end of sentence
            cur[-1] += {0x2000: "?", 0x3000: "!"}.get(intonation, ".")
            sentences.append(" ".join(cur))
            cur = []
        elif kind == 0x40000:  # clause: comma, colon, semicolon
            cur[-1] += ";" if t & 0xFFF == 30 and intonation == 0x1000 else ":" if t & 0xFFF == 30 else ","
    if any(cur):
        sentences.append(" ".join(cur))
    return sentences


def phoneme_ids(phonemes, id_map):
    ids = id_map["^"] + id_map["_"]
    for p in unicodedata.normalize("NFD", phonemes):
        if p in id_map:  # unknown symbols are dropped, as Piper does
            ids = ids + id_map[p] + id_map["_"]
    return ids + id_map["$"]


# --- synthesis ---

def _load(vid):
    if vid not in _voices:
        import onnxruntime as ort
        model, cfg = _files(vid)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(model, sess_options=opts, providers=["CPUExecutionProvider"])
        with open(cfg, encoding="utf-8") as f:
            _voices[vid] = (sess, json.load(f))
    return _voices[vid]


def _wav(pcm, rate):
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return out.getvalue()


def speak(text, lang=None, voice_id=None, speed=1.0):
    """Markdown text -> WAV bytes (16-bit mono). The voice must be installed (install())."""
    lang = norm_lang(lang) or (BY_ID[voice_id]["lang"] if voice_id in BY_ID else guess_lang(text))
    vid = pick_voice(lang, voice_id)
    if not ready(vid):
        raise RuntimeError(f"voice {vid} not installed")
    plain = strip_markdown(text, BY_ID[vid]["lang"])
    with _lock:
        sess, cfg = _load(vid)
        rate = cfg["audio"]["sample_rate"]
        inf = cfg.get("inference", {})
        scales = np.array([inf.get("noise_scale", 0.667), inf.get("length_scale", 1.0) / max(speed, 0.1),
                           inf.get("noise_w", 0.8)], dtype=np.float32)
        gap = np.zeros(int(rate * SILENCE), dtype=np.float32)
        names = {i.name for i in sess.get_inputs()}
        chunks = []
        for sent in phonemize(plain, cfg["espeak"]["voice"]) if plain else []:
            ids = np.array([phoneme_ids(sent, cfg["phoneme_id_map"])], dtype=np.int64)
            feed = {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64), "scales": scales}
            if "sid" in names:
                feed["sid"] = np.array([0], dtype=np.int64)
            chunks += [sess.run(None, feed)[0].squeeze(), gap]
    audio = np.concatenate(chunks) if chunks else np.zeros(int(rate * 0.1), dtype=np.float32)
    return _wav((np.clip(audio, -1, 1) * 32767).astype(np.int16), rate)
