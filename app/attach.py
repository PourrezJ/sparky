"""Attached files -> text given to the assistant as context (everything stays on this PC).

  text and code   read as they are
  PDF             pypdf
  Word, PowerPoint, Excel, OpenDocument   their XML, without extra package
  audio and video transcribed offline with Whisper (decoded by PyAV)
Images do not come here: the UI sends them as pictures the assistant can see.
"""
import base64, html, io, os, re, threading, time, uuid, zipfile

MAX_CHARS = 40000          # per file (~10k tokens): the context of the assistant is 32k tokens
MAX_BYTES = 200 * 2**20
TEXT_EXT = {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg",
            ".conf", ".log", ".py", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".html", ".htm", ".css", ".scss", ".java", ".kt", ".c",
            ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".sql", ".sh", ".bat", ".ps1", ".r", ".lua",
            ".tex", ".srt", ".vtt", ".rtf", ".env", ".gitignore", ".dockerfile"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".flac", ".wma", ".aiff", ".amr"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".wmv", ".mpeg", ".mpg", ".3gp"}
OFFICE = {".docx": "word/document.xml", ".pptx": "ppt/slides/", ".xlsx": "xl/sharedStrings.xml",
          ".odt": "content.xml", ".odp": "content.xml", ".ods": "content.xml"}

files = {}   # id -> {"name", "kind", "state": "reading|ok|error", "text", "error", "chars", "truncated", "t"}
_lock = threading.Lock()


def kind_of(name, mime=""):
    ext = os.path.splitext(name.lower())[1]
    if ext == ".pdf" or mime == "application/pdf":
        return "pdf"
    if ext in OFFICE:
        return "doc"
    if ext in AUDIO_EXT or mime.startswith("audio/"):
        return "audio"
    if ext in VIDEO_EXT or mime.startswith("video/"):
        return "video"
    if ext in TEXT_EXT or mime.startswith("text/") or mime in ("application/json", "application/xml"):
        return "text"
    return "other"


def _decode_text(data):
    for enc in ("utf-8-sig", "utf-16") if data[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _looks_text(data):
    head = data[:8192]
    return b"\x00" not in head and (not head or sum(32 <= b < 127 or b in (9, 10, 13) or b >= 128 for b in head) / len(head) > 0.95)


def _xml_text(xml):
    xml = re.sub(r"</(w:p|a:p|text:p|text:h|row)>", "\n", xml)
    xml = re.sub(r"<(w:tab|text:tab)/>", "\t", xml)
    return html.unescape(re.sub(r"<[^>]+>", "", xml))


def _office(data, ext):
    z = zipfile.ZipFile(io.BytesIO(data))
    if ext == ".pptx":
        slides = sorted((n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
                        key=lambda n: int(re.findall(r"\d+", n)[-1]))
        return "\n\n".join(f"--- Diapositive {i + 1} ---\n" + _xml_text(z.read(n).decode("utf-8", "replace")).strip()
                           for i, n in enumerate(slides))
    if ext == ".xlsx":  # shared strings + inline values of each sheet, row by row
        shared = re.findall(r"<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode("utf-8", "replace"), re.S) \
            if "xl/sharedStrings.xml" in z.namelist() else []
        shared = [html.unescape(re.sub(r"<[^>]+>", "", s)) for s in shared]
        out = []
        for n in sorted(x for x in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", x)):
            out.append("--- Feuille " + re.findall(r"\d+", n)[-1] + " ---")
            for row in re.findall(r"<row[^>]*>(.*?)</row>", z.read(n).decode("utf-8", "replace"), re.S):
                cells = []
                for attrs, body in re.findall(r"<c([^>]*)>(.*?)</c>", row, re.S):
                    v = re.search(r"<v>(.*?)</v>", body) or re.search(r"<t[^>]*>(.*?)</t>", body)
                    val = html.unescape(v.group(1)) if v else ""
                    if 't="s"' in attrs and val.isdigit() and int(val) < len(shared):
                        val = shared[int(val)]
                    cells.append(val)
                out.append("\t".join(cells))
        return "\n".join(out)
    return _xml_text(z.read(OFFICE[ext]).decode("utf-8", "replace"))


def _pdf(data):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("lecture des PDF indisponible (relance l'installation)")
    r = PdfReader(io.BytesIO(data))
    pages = []
    for i, p in enumerate(r.pages):
        pages.append(f"--- Page {i + 1} ---\n" + (p.extract_text() or "").strip())
        if sum(len(x) for x in pages) > MAX_CHARS * 1.2:
            break
    text = "\n\n".join(pages)
    if not re.sub(r"--- Page \d+ ---|\s", "", text):
        raise RuntimeError("PDF scanné sans texte (images seulement)")
    return text


def _audio(data):
    """Any audio or video file -> 16 kHz mono -> Whisper."""
    import av
    import numpy as np
    import stt
    m, ready = server_voice()
    if not ready:
        raise RuntimeError("installe la dictée vocale (Whisper) pour transcrire l'audio")
    chunks = []
    with av.open(io.BytesIO(data)) as c:
        if not c.streams.audio:
            raise RuntimeError("aucune piste audio")
        res = av.AudioResampler(format="flt", layout="mono", rate=16000)
        for frame in c.decode(audio=0):
            for f in res.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in res.resample(None):
            chunks.append(f.to_ndarray().reshape(-1))
    audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    text = stt.transcribe_array(audio, m["id"])
    if not text:
        raise RuntimeError("aucune parole détectée")
    return f"[Transcription de l'audio, {len(audio) / 16000:.0f} s]\n" + text


def _extract(fid, name, data, mime):
    f = files[fid]
    try:
        k, ext = f["kind"], os.path.splitext(name.lower())[1]
        if k == "pdf":
            text = _pdf(data)
        elif k == "doc":
            text = _office(data, ext)
        elif k in ("audio", "video"):
            text = _audio(data)
        elif k == "text" or _looks_text(data):
            f["kind"], text = "text", _decode_text(data)
        else:
            raise RuntimeError("format non lu : seuls le texte, les PDF, Office, l'audio et la vidéo sont pris en compte")
        text = text.replace("\r\n", "\n").strip()
        if not text:
            raise RuntimeError("fichier vide")
        f.update(truncated=len(text) > MAX_CHARS, chars=len(text), text=text[:MAX_CHARS], state="ok")
    except Exception as e:
        f.update(state="error", error=str(e) or e.__class__.__name__)


def add(name, data_b64, mime=""):
    """Starts reading a file; the UI polls status(id)."""
    data = base64.b64decode(data_b64.split(",", 1)[-1])
    fid = uuid.uuid4().hex[:12]
    with _lock:
        for k in [k for k, v in files.items() if time.time() - v["t"] > 6 * 3600]:  # forget old unsent files
            files.pop(k, None)
        files[fid] = {"id": fid, "name": name, "kind": kind_of(name, mime), "state": "reading", "text": "",
                      "error": "", "chars": 0, "truncated": False, "size": len(data), "t": time.time()}
    if len(data) > MAX_BYTES:
        files[fid].update(state="error", error="fichier trop gros (200 Mo max.)")
    else:
        threading.Thread(target=_extract, args=(fid, name, data, mime), daemon=True).start()
    return status(fid)


def status(fid):
    f = files.get(fid)
    return {k: v for k, v in f.items() if k != "text"} if f else None


def take(ids):
    """Files ready to send: [{name, kind, text, truncated}]."""
    return [{"name": f["name"], "kind": f["kind"], "text": f["text"], "truncated": f["truncated"], "size": f["size"]}
            for f in (files.get(i) for i in ids or []) if f and f["state"] == "ok"]


server_voice = None   # set by server.py: returns (voice model, installed)
