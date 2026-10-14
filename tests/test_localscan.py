"""Models already on the PC: what is found, what is ignored, and that Sparky never deletes the user's files."""
import json, os

import pytest

import catalog
import localscan


def gguf(path, size=1024):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"GGUF" + b"\0" * (size - 4))
    return path


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(localscan, "MIN_GB", 0)
    lm = tmp_path / "lmstudio" / "models" / "pub" / "repo"
    gguf(str(lm / "Mistral-7B-Instruct-Q4_K_M.gguf"))
    gguf(str(lm / "mmproj-Mistral-F16.gguf"))
    gguf(str(tmp_path / "dl" / "Qwen3.5-9B-Q4_K_M.gguf"))          # same name as a catalog model
    gguf(str(tmp_path / "dl" / "model-00002-of-00003.gguf"))        # split part: only the first is usable
    gguf(str(tmp_path / "dl" / "nomic-embed-text.gguf"))            # embeddings are not assistants
    (tmp_path / "dl" / "fake.gguf").write_bytes(b"NOPE" * 300)      # not a GGUF file
    # Ollama store
    ol = tmp_path / "ollama"
    blob = gguf(str(ol / "blobs" / "sha256-abc"))
    man = ol / "manifests" / "registry.ollama.ai" / "library" / "llama3.2"
    man.mkdir(parents=True)
    (man / "3b").write_text(json.dumps({"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:abc"}]}))
    return tmp_path, blob


def scan(tmp_path):
    return {os.path.basename(i["path"]) if i["source"] != "Ollama" else i["label"]: i
            for i in localscan.find([(str(tmp_path), 6)], ollama=str(tmp_path / "ollama"))}


def test_finds_gguf_models_and_their_vision_file(tree):
    found = scan(tree[0])
    m = found["Mistral-7B-Instruct-Q4_K_M.gguf"]
    assert m["kind"] == "llm" and m["mmproj"].endswith("mmproj-Mistral-F16.gguf") and not m["match"]


def test_recognises_catalog_models(tree):
    assert scan(tree[0])["Qwen3.5-9B-Q4_K_M.gguf"]["match"] == "qwen359"


def test_ignores_split_parts_embeddings_and_non_gguf(tree):
    names = set(scan(tree[0]))
    assert not names & {"model-00002-of-00003.gguf", "nomic-embed-text.gguf", "fake.gguf", "mmproj-Mistral-F16.gguf"}


def test_finds_ollama_models(tree):
    m = scan(tree[0])["llama3.2:3b"]
    assert m["path"] == tree[1] and m["source"] == "Ollama"


def test_use_in_place_and_delete_keeps_the_users_file(tree):
    found = scan(tree[0])
    localscan.state["items"] = list(found.values())
    item = found["Mistral-7B-Instruct-Q4_K_M.gguf"]
    mid = localscan.use(item["key"])
    m = catalog.BY_ID[mid]
    assert m["local"] and catalog.installed(m) and catalog.has_vision(m)
    catalog.delete(mid)
    assert os.path.exists(item["path"]), "Sparky must never delete a model it did not download"
    assert mid not in catalog.BY_ID


def test_catalog_model_is_linked_where_sparky_expects_it(tree):
    found = scan(tree[0])
    localscan.state["items"] = list(found.values())
    mid = localscan.use(found["Qwen3.5-9B-Q4_K_M.gguf"]["key"])
    assert mid == "qwen359" and catalog.installed(catalog.BY_ID["qwen359"])
    assert localscan.status()["items"]  # the item is then reported as added
    os.remove(catalog.BY_ID["qwen359"]["files"][0][2])


def test_unknown_key_is_an_error():
    localscan.state["items"] = []
    with pytest.raises(RuntimeError):
        localscan.use("nope")
