"""Cloud providers: keys, model lists, streaming, retries, errors (against the fake servers of mocks.py)."""
import json

import pytest

import mocks


def custom(cloud, srv, key=mocks.KEY):
    cloud.set_key("custom", key, srv.url + "/oai")
    return cloud.entry("custom", "mock/vision-chat", "Mock Vision", vision=True, think=True)


def test_no_provider_is_configured_by_default(clean_cloud):
    assert not any(p["hasKey"] for p in clean_cloud.status())


def test_status_never_returns_the_key(clean_cloud, mock_provider):
    custom(clean_cloud, mock_provider, key="test-key")
    dump = json.dumps(clean_cloud.status())
    assert "test-key" not in dump
    assert clean_cloud.mask("sk-or-v1-abcdefghijklmnop").endswith("mnop") and "efgh" not in clean_cloud.mask("sk-or-v1-abcdefghijklmnop")


def test_wrong_key_is_refused_with_a_clear_message(clean_cloud, mock_provider):
    with pytest.raises(RuntimeError, match="clé API refusée"):
        custom(clean_cloud, mock_provider, key="bad")


def test_openai_compatible_models_and_capabilities(clean_cloud, mock_provider):
    custom(clean_cloud, mock_provider)
    ms = {m["id"]: m for m in clean_cloud.models("custom")}
    assert ms["mock/vision-chat"]["vision"] and ms["mock/vision-chat"]["think"]
    assert not ms["mock/text-only"]["vision"] and not ms["mock/text-only"]["think"]


def test_openai_compatible_stream_with_reasoning(clean_cloud, mock_provider):
    m = custom(clean_cloud, mock_provider)
    pieces = list(clean_cloud.stream(m, [{"role": "user", "content": "salut"}], 100, think="medium", json_mode=True))
    assert "".join(t for t, _ in pieces) == "Bonjour"
    assert "".join(r for _, r in pieces) == "je réfléchis"
    sent = [r for r in mock_provider.requests if r[0] == "POST"][-1][2]
    assert sent["model"] == "mock/vision-chat" and sent["stream"] is True


def test_optional_parameters_are_dropped_when_the_provider_refuses_them(clean_cloud, mock_provider):
    m = custom(clean_cloud, mock_provider)
    mock_provider.mode = "strict"
    assert clean_cloud.complete(m, [{"role": "user", "content": "x"}]) == "Bonjour"
    posts = [r[2] for r in mock_provider.requests if r[0] == "POST"]
    assert "temperature" in posts[-2] and "temperature" not in posts[-1]


def test_entry_is_a_cloud_llm(clean_cloud):
    e = clean_cloud.entry("openrouter", "anthropic/claude-x", "Claude X", vision=True)
    assert e["id"] == "cloud:openrouter:anthropic/claude-x" and e["kind"] == "llm" and e["cloud"] and e["vision"]
    assert e["files"] == [] and e["size_gb"] == 0


# ---------------------------------------------------------------- Anthropic (official SDK)
@pytest.fixture
def claude(clean_cloud, mock_provider, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", mock_provider.url)
    clean_cloud.set_key("anthropic", mocks.KEY)
    return clean_cloud


def test_claude_models_read_capabilities(claude):
    ms = {m["id"]: m for m in claude.models("anthropic")}
    assert ms["claude-opus-5-5"] == {"id": "claude-opus-5-5", "label": "Claude Opus 5.5", "context": 1000000,
                                     "vision": True, "think": True, "effort": True}
    assert not ms["claude-old"]["think"] and not ms["claude-old"]["vision"]


def test_claude_stream_text_and_thinking(claude, mock_provider):
    m = claude.entry("anthropic", "claude-opus-5-5", think=True, effort=True, vision=True)
    pieces = list(claude.stream(m, [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}], 500, think="high"))
    assert "".join(t for t, _ in pieces) == "Salut" and "".join(r for _, r in pieces) == "hmm"
    sent = [r for r in mock_provider.requests if r[1].startswith("/v1/messages")][-1]
    body, headers = sent[2], {k.lower(): v for k, v in sent[3].items()}
    assert body["system"] == "sys" and body["thinking"]["type"] == "adaptive"
    assert body["output_config"] == {"effort": "high"}
    assert body["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in headers.get("anthropic-beta", "")
    assert "temperature" not in body  # removed on current Claude models


def test_claude_retries_without_options_the_model_does_not_take(claude, mock_provider):
    m = claude.entry("anthropic", "claude-old", effort=True)
    mock_provider.mode = "strict"
    assert claude.complete(m, [{"role": "user", "content": "hi"}]) == "Salut"


def test_claude_refusal_is_reported(claude, mock_provider):
    m = claude.entry("anthropic", "claude-opus-5-5")
    mock_provider.mode = "refusal"
    with pytest.raises(RuntimeError, match="refusé"):
        claude.complete(m, [{"role": "user", "content": "hi"}])


def test_messages_are_converted_to_claude_blocks(clean_cloud):
    system, msgs = clean_cloud._to_claude([
        {"role": "system", "content": "A"},
        {"role": "assistant", "content": "{}"},
        {"role": "user", "content": [{"type": "text", "text": "voici"},
                                     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}]}])
    assert system == "A"
    assert msgs[0]["role"] == "user"  # Claude conversations must start with the user
    img = msgs[-1]["content"][1]
    assert img == {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}}
