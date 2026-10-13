"""Sparky Bot long-term memory: SQLite store + hybrid retrieval (semantic embeddings, keywords,
importance, recency). Every memory is visible and editable by the user.
"""
import json, math, os, re, sqlite3, threading, time

import config

DB = os.path.join(config.DATA, "companion.db")
CATEGORIES = ("identity", "preference", "project", "people", "goal", "event", "style", "other")
EMBED_ID = "e5-small"


class Embedder:
    """multilingual-e5-small through transformers (CPU, ~20 ms per sentence). None until downloaded."""

    def __init__(self):
        self.model = self.tok = None
        self.lock = threading.Lock()

    def ready(self):
        import catalog
        m = catalog.BY_ID.get(EMBED_ID)
        return bool(m) and catalog.installed(m)

    def _load(self):
        import catalog
        from transformers import AutoModel, AutoTokenizer
        d = catalog.BY_ID[EMBED_ID]["dir"]
        self.tok = AutoTokenizer.from_pretrained(d)
        self.model = AutoModel.from_pretrained(d).eval()

    def encode(self, texts, query=False):
        if not self.ready():
            return None
        import torch
        with self.lock:
            if self.model is None:
                self._load()
            pre = "query: " if query else "passage: "
            b = self.tok([pre + t for t in texts], padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                h = self.model(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1).float()
            v = (h * mask).sum(1) / mask.sum(1)
            v = torch.nn.functional.normalize(v, dim=-1)
        return [x.tolist() for x in v]


EMB = Embedder()
WORD = re.compile(r"[a-zà-ÿ0-9]{3,}", re.I)
STOP = set("the and for you your are was with this that have has not but from they she him her his our out what when "
           "les des une est pas pour avec dans que qui sur son ses mes tes nous vous ils elle mais par cette tout plus".split())


def words(t):
    return {w for w in WORD.findall(t.lower()) if w not in STOP}


class Store:
    def __init__(self, path=DB):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, text TEXT, category TEXT, importance INTEGER,
                created REAL, updated REAL, last_used REAL, uses INTEGER DEFAULT 0, pinned INTEGER DEFAULT 0,
                source TEXT DEFAULT 'chat', vec TEXT);
            CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, role TEXT, text TEXT, ts REAL);
            CREATE TABLE IF NOT EXISTS journal(id INTEGER PRIMARY KEY, ts REAL, text TEXT);
            CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);""")
            self.db.commit()

    # -- key/value (persona, style, bond, summary, check-in...)
    def get(self, key, default=None):
        r = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    def set(self, key, value):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO kv VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False)))
            self.db.commit()

    # -- messages
    def add_message(self, role, text):
        with self.lock:
            c = self.db.execute("INSERT INTO messages(role,text,ts) VALUES(?,?,?)", (role, text, time.time()))
            self.db.commit()
            return c.lastrowid

    def messages(self, limit=200, after=0):
        rows = self.db.execute("SELECT * FROM messages WHERE id>? ORDER BY id DESC LIMIT ?", (after, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    # -- memories
    def memories(self):
        return [dict(r) | {"vec": None} for r in self.db.execute(
            "SELECT * FROM memories ORDER BY pinned DESC, importance DESC, updated DESC").fetchall()]

    def add_memory(self, text, category="other", importance=3, source="chat", pinned=0):
        text = text.strip()
        if not text:
            return None
        vec = EMB.encode([text])
        now = time.time()
        with self.lock:
            c = self.db.execute("INSERT INTO memories(text,category,importance,created,updated,last_used,source,pinned,vec) "
                                "VALUES(?,?,?,?,?,?,?,?,?)",
                                (text, category if category in CATEGORIES else "other", int(max(1, min(5, importance))),
                                 now, now, now, source, int(pinned), json.dumps(vec[0]) if vec else None))
            self.db.commit()
            return c.lastrowid

    def update_memory(self, mid, text=None, category=None, importance=None, pinned=None):
        r = self.db.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone()
        if not r:
            return False
        text = (text or r["text"]).strip()
        vec = EMB.encode([text]) if text != r["text"] or not r["vec"] else None
        with self.lock:
            self.db.execute("UPDATE memories SET text=?, category=?, importance=?, pinned=?, updated=?, vec=COALESCE(?,vec) WHERE id=?",
                            (text, category if category in CATEGORIES else r["category"],
                             int(max(1, min(5, importance))) if importance else r["importance"],
                             int(pinned) if pinned is not None else r["pinned"], time.time(),
                             json.dumps(vec[0]) if vec else None, mid))
            self.db.commit()
        return True

    def delete_memory(self, mid):
        with self.lock:
            self.db.execute("DELETE FROM memories WHERE id=?", (mid,))
            self.db.commit()

    def reindex(self):
        """Computes the missing vectors (memories saved before the embedding model was downloaded)."""
        rows = self.db.execute("SELECT id,text FROM memories WHERE vec IS NULL").fetchall()
        if not rows or not EMB.ready():
            return 0
        vecs = EMB.encode([r["text"] for r in rows])
        with self.lock:
            for r, v in zip(rows, vecs):
                self.db.execute("UPDATE memories SET vec=? WHERE id=?", (json.dumps(v), r["id"]))
            self.db.commit()
        return len(rows)

    def nearest(self, text, exclude=None):
        """(id, cosine, text) of the closest memory by meaning, or None (no embeddings / no memories)."""
        qv = EMB.encode([text])
        if not qv:
            return None
        best = None
        for r in self.db.execute("SELECT id,text,vec FROM memories WHERE vec IS NOT NULL").fetchall():
            if r["id"] == exclude:
                continue
            cos = sum(a * b for a, b in zip(qv[0], json.loads(r["vec"])))
            if not best or cos > best[1]:
                best = (r["id"], cos, r["text"])
        return best

    def similarity(self, a, b):
        v = EMB.encode([a, b])
        if v:
            return sum(x * y for x, y in zip(v[0], v[1]))
        wa, wb = words(a), words(b)  # without embeddings: shared words (Jaccard), rescaled to a cosine-like range
        return 0.75 + 0.25 * len(wa & wb) / max(1, len(wa | wb))

    def search(self, query, k=12):
        """Hybrid ranking: meaning (cosine) or shared words, + importance, + recency. Pinned and identity always first."""
        rows = [dict(r) for r in self.db.execute("SELECT * FROM memories").fetchall()]
        if not rows:
            return []
        qv = EMB.encode([query], query=True) if query.strip() else None
        qw = words(query)
        now = time.time()
        for r in rows:
            if qv and r["vec"]:
                v = json.loads(r["vec"])
                sim = max(0.0, (sum(a * b for a, b in zip(qv[0], v)) - 0.80) / 0.15)  # e5 cosines live in ~0.80..0.95
            else:
                mw = words(r["text"])
                sim = len(qw & mw) / math.sqrt(len(mw) + 1) if qw else 0.0
            age = (now - max(r["updated"], r["last_used"] or 0)) / 86400
            r["score"] = 0.6 * min(1.0, sim) + 0.25 * r["importance"] / 5 + 0.15 * math.exp(-age / 30) \
                + (1.0 if r["pinned"] else 0) + (0.3 if r["category"] == "identity" else 0)
        rows.sort(key=lambda r: -r["score"])
        top = rows[:k]
        with self.lock:
            for r in top:
                self.db.execute("UPDATE memories SET last_used=?, uses=uses+1 WHERE id=?", (now, r["id"]))
            self.db.commit()
        return [{k2: r[k2] for k2 in ("id", "text", "category", "importance", "pinned")} for r in top]

    # -- journal
    def add_journal(self, text):
        with self.lock:
            self.db.execute("INSERT INTO journal(ts,text) VALUES(?,?)", (time.time(), text))
            self.db.commit()

    def journal(self, limit=30):
        return [dict(r) for r in self.db.execute("SELECT * FROM journal ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]

    def forget_all(self):
        with self.lock:
            for t in ("memories", "messages", "journal", "kv"):
                self.db.execute(f"DELETE FROM {t}")
            self.db.commit()
