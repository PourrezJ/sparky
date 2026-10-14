"""Attached files -> text (regression: every format the + button accepts)."""
import base64, io, time, zipfile

import pytest

import attach


def read(name, data, mime=""):
    st = attach.add(name, base64.b64encode(data).decode(), mime)
    t0 = time.time()
    while attach.files[st["id"]]["state"] == "reading" and time.time() - t0 < 20:
        time.sleep(0.02)
    return attach.files[st["id"]]


def zipped(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return buf.getvalue()


def pdf(pages):
    """Smallest valid PDF with one text line per page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for t in pages:
        stream = f"BT /F1 14 Tf 50 750 Td ({t}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {len(objs)} 0 R "
                    "/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offs = b"%PDF-1.4\n", []
    for i, x in enumerate(objs):
        offs.append(len(out))
        out += f"{i + 1} 0 obj\n{x}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode() + b"".join(f"{o:010d} 00000 n \n".encode() for o in offs)
    return out + f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()


@pytest.mark.parametrize("name,kind", [("a.PDF", "pdf"), ("b.docx", "doc"), ("c.mp3", "audio"), ("d.mov", "video"),
                                       ("e.py", "text"), ("f.unknown", "other"), ("noext", "other")])
def test_kind_of(name, kind):
    assert attach.kind_of(name) == kind


def test_text_encodings():
    assert read("u.txt", "été ✓".encode())["text"] == "été ✓"
    assert read("w.txt", "café".encode("cp1252"))["text"] == "café"
    assert read("x.txt", "hé".encode("utf-16"))["text"] == "hé"  # regression: the BOM stayed in the text
    assert read("y.txt", "﻿hé".encode("utf-8"))["text"] == "hé"


def test_unknown_extension_with_text_content_is_read():
    f = read("notes.weird", b"plain text content")
    assert f["state"] == "ok" and f["kind"] == "text"


def test_binary_is_refused():
    f = read("blob.bin", bytes(range(256)) * 10)
    assert f["state"] == "error" and "format non lu" in f["error"]


def test_empty_file():
    assert read("e.txt", b"")["state"] == "error"


def test_pdf_pages():
    f = read("r.pdf", pdf(["Budget 48000 euros", "Livraison en mars"]))
    assert f["state"] == "ok", f["error"]
    assert "Page 1" in f["text"] and "48000" in f["text"] and "mars" in f["text"]


def test_docx():
    f = read("c.docx", zipped({"word/document.xml": "<w:document><w:body><w:p><w:r><w:t>Loyer 950</w:t></w:r></w:p>"
                                                      "<w:p><w:r><w:t>euros</w:t></w:r></w:p></w:body></w:document>"}))
    assert f["text"] == "Loyer 950\neuros"


def test_xlsx_shared_strings_and_numbers():
    f = read("v.xlsx", zipped({
        "xl/sharedStrings.xml": "<sst><si><t>Mois</t></si><si><t>Janvier</t></si></sst>",
        "xl/worksheets/sheet1.xml": '<worksheet><sheetData><row><c t="s"><v>0</v></c></row>'
                                    '<row><c t="s"><v>1</v></c><c><v>1200</v></c></row></sheetData></worksheet>'}))
    assert "Mois" in f["text"] and "Janvier\t1200" in f["text"]


def test_pptx_slides_in_order():
    f = read("s.pptx", zipped({"ppt/slides/slide10.xml": "<p:sld><a:p><a:t>dix</a:t></a:p></p:sld>",
                               "ppt/slides/slide2.xml": "<p:sld><a:p><a:t>deux</a:t></a:p></p:sld>"}))
    assert f["text"].index("deux") < f["text"].index("dix")


def test_long_file_is_truncated():
    f = read("big.txt", b"a" * (attach.MAX_CHARS + 500))
    assert f["truncated"] and len(f["text"]) == attach.MAX_CHARS


def test_audio_without_voice_model_explains_what_to_do():
    attach.server_voice = lambda: ({"id": "whisper-turbo"}, False)
    f = read("v.wav", b"RIFF....WAVE")
    assert f["state"] == "error"


def test_take_only_returns_ready_files():
    ok = attach.add("a.txt", base64.b64encode(b"hello").decode())["id"]
    time.sleep(0.3)
    assert [f["name"] for f in attach.take([ok, "missing"])] == ["a.txt"]
    assert "text" not in attach.status(ok)  # the UI never gets the extracted text back
