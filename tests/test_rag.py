"""Knowledge base: chunking, ingestion (text + PDF pages), keyword search without the embedding model, delete."""
import time

import pytest

import memory
import rag
from test_attach import pdf


@pytest.fixture(autouse=True)
def no_embedder(monkeypatch):
    """Tests never need e5-small: encode() behaves as when the model is not installed."""
    monkeypatch.setattr(memory.EMB, "encode", lambda texts, query=False: None)
    monkeypatch.setattr(memory.EMB, "ready", lambda: False)


def wait(did):
    t0 = time.time()
    while time.time() - t0 < 20:
        d = next(d for c in rag.collections() for d in c["docs"] if d["id"] == did)
        if d["state"] != "indexing":
            return d
        time.sleep(0.02)
    raise TimeoutError


def test_chunk_sizes_and_overlap():
    text = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(300))
    parts = rag.chunk(text)
    assert len(parts) > 5
    assert all(len(p) <= rag.CHUNK for p in parts)
    assert all(p.endswith(".") for p in parts[:-1])          # cut at sentence boundaries
    assert parts[0][-60:].split(".")[-2] in parts[1]          # overlap: end of one chunk starts the next
    assert rag.chunk("short") == ["short"] and rag.chunk("  ") == []


def test_chunk_prefers_paragraphs():
    text = "\n\n".join(("Paragraph %d. " % i) + "word " * 60 for i in range(10))
    assert all(p.startswith("Paragraph") or p.startswith("word") for p in rag.chunk(text))


def test_text_ingestion_search_and_delete():
    cid = rag.create_collection("Recettes")
    body = ("Notes de cuisine.\n\n" + "Le gâteau au chocolat demande 200 g de chocolat noir et 4 oeufs. " * 3 + "\n\n"
            + "Filler paragraph about nothing in particular. " * 60 + "\n\n"
            + "The sourdough bread needs a levain fed twelve hours before baking.")
    did = rag.add_document(cid, "cuisine.txt", body.encode("utf-8"))
    d = wait(did)
    assert d["state"] == "ready" and d["chunks"] >= 2 and d["kind"] == "text"

    hits = rag.search("sourdough levain", [cid])
    assert hits and "levain" in hits[0]["text"] and hits[0]["doc"] == "cuisine.txt"
    assert hits[0]["collection"] == "Recettes" and hits[0]["page"] is None
    assert rag.search("quantum chromodynamics", [cid]) == []
    texts = [h["text"] for h in rag.search("chocolat", [cid], k=10)]
    assert len(texts) == len(set(texts))

    rag.rename_collection(cid, "Cuisine")
    assert any(c["name"] == "Cuisine" for c in rag.collections())
    rag.delete_document(did)
    assert rag.search("sourdough levain", [cid]) == []
    rag.delete_collection(cid)
    assert all(c["id"] != cid for c in rag.collections())


def test_pdf_keeps_pages_and_citations():
    cid = rag.create_collection("Docs")
    did = rag.add_document(cid, "manual.pdf", pdf(["Introduction to the device", "Battery replacement takes ten minutes"]))
    assert wait(did)["state"] == "ready"
    hits = rag.search("battery replacement", [cid])
    assert hits[0]["page"] == 2
    block = rag.context_block(hits)
    assert "[1] manual.pdf (p. 2): Battery replacement" in block and "[1]" in block
    assert rag.context_block([]) == ""
    rag.delete_collection(cid)


def test_collection_filter_and_errors():
    a, b = rag.create_collection("A"), rag.create_collection("B")
    wait(rag.add_document(a, "a.md", b"Zebras live in Africa."))
    bad = wait(rag.add_document(b, "x.bin", bytes(range(256)) * 10))
    assert bad["state"] == "error" and bad["error"]
    assert rag.search("zebras", [b]) == [] and rag.search("zebras", [a])
    assert rag.search("zebras")  # all collections
    with pytest.raises(ValueError):
        rag.add_document(999999, "x.txt", b"hi")
    rag.delete_collection(a)
    rag.delete_collection(b)
