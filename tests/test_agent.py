"""Agent loop (Session.turn) with a scripted model: every path of a turn, without any real model."""
import json

import pytest

import harness


@pytest.fixture
def agent(monkeypatch):
    """Returns run(script, text) -> events. Each script item is what the model answers to one request."""
    monkeypatch.setitem(harness.SETTINGS, "max_steps", 4)
    monkeypatch.setitem(harness.SETTINGS, "autotest", False)
    monkeypatch.setitem(harness.SETTINGS, "llm", None)
    asked = []

    def run(script, text="Bonjour, écris-moi quelque chose s'il te plaît", files=()):
        script = list(script)

        def fake(body, think="off"):
            asked.append(body["messages"])
            out = script.pop(0) if script else {"action": "reply", "args": {"text": "fin du script"}}
            yield (out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)), ""
        monkeypatch.setattr(harness, "chat_deltas", fake)
        monkeypatch.setattr(harness, "chat_once", lambda body: body["messages"][0]["content"].split("\n\n", 1)[-1])
        s = harness.Session(ephemeral=True)
        events = []
        s.turn(text, [], None, None, events.append, files=files)
        return s, events
    run.asked = asked
    return run


def kinds(events):
    return [e["type"] for e in events if e["type"] not in ("read", "typing", "status")]


def test_plain_reply(agent):
    s, ev = agent([{"action": "reply", "args": {"text": "Voici ma réponse."}}])
    assert kinds(ev) == ["user", "reply", "done"]
    assert next(e for e in ev if e["type"] == "reply")["text"] == "Voici ma réponse."
    assert s.history[-1]["role"] == "assistant"


def test_artifact_then_reply(agent):
    code = "<!doctype html><html><body><h1>Bonjour</h1><script>console.log('une vraie page complète');</script></body></html>"
    s, ev = agent([{"action": "artifact", "args": {"id": "page", "title": "Page", "language": "html", "content": code}},
                   {"action": "reply", "args": {"text": "La page est prête."}}])
    assert kinds(ev) == ["user", "artifact", "reply", "done"]
    assert s.artifacts and next(iter(s.artifacts.values()))["versions"][-1] == code


def test_packed_actions_run_in_order_without_asking_again(agent):
    """Regression: artifact + reply packed in ONE object (duplicate "action" keys)."""
    code = "print('un programme python complet qui affiche quelque chose')\nfor i in range(3):\n    print(i)\n"
    raw = ('{"action": "artifact", "args": {"id": "prog", "title": "Prog", "language": "python", "content": %s}, '
           '"action": "reply", "args": {"text": "Programme écrit."}}') % json.dumps(code)
    s, ev = agent([raw])
    assert kinds(ev) == ["user", "artifact", "reply", "done"]
    assert len(agent.asked) == 1


def test_project_files(agent):
    html = "<!doctype html><html><head><link rel='stylesheet' href='style.css'></head><body><h1>Site</h1></body></html>"
    css = "body { font-family: sans-serif; margin: 0; padding: 2rem; background: #fafafa; color: #222; }"
    s, ev = agent([{"action": "artifact", "args": {"project": "mon-site", "path": "index.html", "content": html}},
                   {"action": "artifact", "args": {"project": "mon-site", "path": "style.css", "content": css}},
                   {"action": "reply", "args": {"text": "Le site est prêt."}}])
    assert kinds(ev) == ["user", "project", "project", "reply", "done"]
    assert set(s.projects["mon-site"]["files"]) == {"index.html", "style.css"}


def test_missing_project_file_is_asked_before_replying(agent):
    html = "<!doctype html><html><head><script src='app.js'></script></head><body><h1>Jeu complet</h1></body></html>"
    js = "document.body.append('le jeu démarre'); console.log('script principal du jeu, complet');"
    s, ev = agent([{"action": "artifact", "args": {"project": "jeu", "path": "index.html", "content": html}},
                   {"action": "reply", "args": {"text": "C'est prêt."}},
                   {"action": "artifact", "args": {"project": "jeu", "path": "app.js", "content": js}},
                   {"action": "reply", "args": {"text": "C'est prêt, avec le script."}}])
    assert any("app.js" in m["content"][0]["text"] for m in agent.asked[2] if m["role"] == "user" and isinstance(m["content"], list))
    assert ev[-2]["text"] == "C'est prêt, avec le script."


def test_empty_reply_is_retried(agent):
    s, ev = agent([{"action": "reply", "args": {}}, {"action": "reply", "args": {"text": "Réponse complète."}}])
    assert [e["text"] for e in ev if e["type"] == "reply"] == ["Réponse complète."]


def test_invalid_json_is_retried_once_then_used_as_text(agent):
    s, ev = agent(["pas du json du tout", "toujours pas du json"])
    assert [e["text"] for e in ev if e["type"] == "reply"] == ["toujours pas du json"]


def test_repeated_reply_is_refused(agent):
    s, _ = agent([{"action": "reply", "args": {"text": "Même réponse."}}])
    events = []
    agent_script = [{"action": "reply", "args": {"text": "Même réponse."}}, {"action": "reply", "args": {"text": "Nouvelle réponse."}}]

    def fake(body, think="off"):
        yield json.dumps(agent_script.pop(0)), ""
    harness.chat_deltas = fake
    s.turn("Et encore une question en français ?", [], None, None, events.append)
    assert [e["text"] for e in events if e["type"] == "reply"] == ["Nouvelle réponse."]


def test_attached_files_reach_the_model(agent):
    f = {"name": "contrat.docx", "kind": "doc", "text": "Le loyer est de 950 euros.", "truncated": False, "size": 10}
    s, ev = agent([{"action": "reply", "args": {"text": "Le loyer est de 950 euros."}}], files=[f])
    first_user = next(m for m in agent.asked[0] if m["role"] == "user")
    assert "Le loyer est de 950 euros." in first_user["content"][0]["text"]
    assert ev[0]["files"] == [{"name": "contrat.docx", "kind": "doc", "size": 10}]


def test_out_of_steps_ends_with_an_honest_reply(agent, monkeypatch):
    monkeypatch.setitem(harness.SETTINGS, "max_steps", 2)
    s, ev = agent([{"action": "generate", "args": {"prompt": f"chat {i}"}} for i in range(10)])
    assert kinds(ev)[-2:] == ["reply", "done"]
    assert "error" in kinds(ev)  # no image engine in the tests: the tool error is reported


def test_model_failure_is_reported(agent, monkeypatch):
    def boom(body, think="off"):
        raise RuntimeError("connexion impossible")
    s = harness.Session(ephemeral=True)
    monkeypatch.setattr(harness, "chat_deltas", boom)
    events = []
    s.turn("salut", [], None, None, events.append)
    assert kinds(events) == ["user", "error", "done"] and "connexion impossible" in events[1]["text"]
