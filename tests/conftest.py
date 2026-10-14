"""Test setup: Sparky runs on throw-away folders (no model, no GPU, no network needed).

config.py reads SPARKY_CONFIG when it is first imported, so it is set here before any app module loads.
"""
import json, os, sys, tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
TMP = tempfile.mkdtemp(prefix="sparky-tests-")
_cfg = {k: os.path.join(TMP, v) for k, v in {
    "runtime_dir": "runtime", "llm_dir": "models/llm", "mmproj_dir": "models/mmproj", "voice_dir": "models/voice",
    "embed_dir": "models/embed", "data_dir": "data", "llama_server": "runtime/llama/llama-server.exe"}.items()}
_cfg |= {"app_port": 17860, "llm_port": 18080, "comfy_port": 18188}  # nothing listens there
with open(os.path.join(TMP, "config.json"), "w", encoding="utf-8") as f:
    json.dump(_cfg, f)
os.environ["SPARKY_CONFIG"] = os.path.join(TMP, "config.json")
sys.path.insert(0, APP)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def data_dir():
    return _cfg["data_dir"]


@pytest.fixture
def mock_provider():
    """Fake OpenAI-compatible and Anthropic servers on a free port (see mocks.py)."""
    import mocks
    srv = mocks.start()
    yield srv
    srv.shutdown()


@pytest.fixture
def clean_cloud(data_dir):
    import cloud
    if os.path.exists(cloud.FILE):
        os.remove(cloud.FILE)
    yield cloud
    if os.path.exists(cloud.FILE):
        os.remove(cloud.FILE)
