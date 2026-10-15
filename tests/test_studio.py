"""Studio: video graphs (LTX-2.5 and Wan 2.2), library, job queue (no ComfyUI needed)."""
import json, os

import catalog
import studio


def test_ltx_graph_text_and_image_to_video():
    m = catalog.BY_ID["ltx25"]
    g, info = studio.build_ltx(m, "a fox in the snow", 1, "16:9", 5)
    assert info["frames"] % 8 == 1 and info["width"] % 32 == 0 and info["height"] % 32 == 0 and info["width"] > info["height"]
    assert g["10"]["inputs"]["video_latent"] == ["8", 0] and g["20"]["class_type"] == "SaveVideo"
    g, info = studio.build_ltx(m, "", 1, "9:16", 3, image="start.png")
    assert g["10"]["inputs"]["video_latent"] == ["33", 0] and g["30"]["inputs"]["image"] == "start.png"
    assert info["height"] > info["width"]


def test_wan_graph():
    m = catalog.BY_ID["wan22-5b"]
    g, info = studio.build_wan(m, "waves", 1, "1:1", 5, image="a.png")
    assert info["frames"] % 4 == 1 and info["width"] == info["height"]
    assert g["7"]["inputs"]["start_image"] == ["31", 0] and g["2"]["inputs"]["type"] == "wan"


def test_video_models_in_catalog():
    vids = [m for m in catalog.CATALOG if m["kind"] == "video"]
    assert {m["engine"] for m in vids} == {"ltx", "wan"}
    for m in vids:
        assert all(f[1].endswith(".safetensors") for f in m["files"])
        ind = catalog.indicators(m)
        assert {"speed", "fit", "secs"} <= ind.keys()


def test_library_lists_studio_and_chat_media(data_dir):
    from PIL import Image
    os.makedirs(studio.DIR, exist_ok=True)
    Image.new("RGB", (8, 8)).save(os.path.join(studio.DIR, "abc123.png"))
    studio._add({"id": "abc123", "type": "image", "file": "abc123.png", "prompt": "red", "created": 1.0})
    sid = "20260101-000000-abcd"
    d = os.path.join(data_dir, sid)
    os.makedirs(d, exist_ok=True)
    Image.new("RGB", (8, 8)).save(os.path.join(d, "1.png"))
    json.dump({"images": {"1": {"path": os.path.join(d, "1.png"), "kind": "gen", "desc": "blue"}}, "events": [], "title": "t"},
              open(os.path.join(d, "session.json"), "w"))
    lib = studio.library()
    assert {"abc123", f"chat-{sid}-1"} <= {i["id"] for i in lib}
    assert studio.file_path("abc123.png") and not studio.file_path("../x.png")
    assert studio.source_image(f"chat-{sid}-1").size == (8, 8)
    assert studio.delete("abc123") and not studio.file_path("abc123.png")


def test_submit_validates():
    import pytest
    with pytest.raises(ValueError):
        studio.submit("video", "", source=None)
    with pytest.raises(ValueError):
        studio.submit("gif", "x")
