"""Scheduled prompts: next_run maths (local time) and the runner loop driven through tick()."""
import datetime, os

import pytest


def ts(*a):
    return datetime.datetime(*a).timestamp()  # local time


@pytest.fixture
def sched(data_dir):
    import scheduler
    if os.path.exists(scheduler.FILE):
        os.remove(scheduler.FILE)
    yield scheduler
    if os.path.exists(scheduler.FILE):
        os.remove(scheduler.FILE)


# 2026-10-14 is a Wednesday (weekday 2)
def test_daily(sched):
    w = {"type": "daily", "time": "08:00"}
    assert sched.next_run(w, ts(2026, 10, 14, 7, 0)) == ts(2026, 10, 14, 8, 0)
    assert sched.next_run(w, ts(2026, 10, 14, 8, 0)) == ts(2026, 10, 15, 8, 0)  # exactly now -> tomorrow
    assert sched.next_run(w, ts(2026, 10, 14, 23, 59)) == ts(2026, 10, 15, 8, 0)
    assert sched.next_run(w, ts(2026, 12, 31, 9, 0)) == ts(2027, 1, 1, 8, 0)  # year rollover


def test_weekly(sched):
    assert sched.next_run({"type": "weekly", "day": 4, "time": "09:30"}, ts(2026, 10, 14, 12, 0)) == ts(2026, 10, 16, 9, 30)
    assert sched.next_run({"type": "weekly", "day": 0, "time": "09:30"}, ts(2026, 10, 14, 12, 0)) == ts(2026, 10, 19, 9, 30)
    w = {"type": "weekly", "day": 2, "time": "09:30"}  # same weekday
    assert sched.next_run(w, ts(2026, 10, 14, 9, 0)) == ts(2026, 10, 14, 9, 30)
    assert sched.next_run(w, ts(2026, 10, 14, 10, 0)) == ts(2026, 10, 21, 9, 30)
    assert sched.next_run({"type": "weekly", "day": 3, "time": "00:00"}, ts(2026, 10, 31, 1, 0)) == ts(2026, 11, 5, 0, 0)


def test_interval_and_once(sched):
    now = ts(2026, 10, 14, 12, 0)
    assert sched.next_run({"type": "interval", "minutes": 90}, now) == now + 5400
    assert sched.next_run({"type": "once", "at": now + 60}, now) == now + 60
    assert sched.next_run({"type": "once", "at": now - 60}, now) is None


@pytest.mark.parametrize("when", [None, {"type": "hourly"}, {"type": "daily", "time": "25:00"}, {"type": "daily", "time": "8h"},
                                  {"type": "weekly", "day": 7, "time": "08:00"}, {"type": "interval", "minutes": 0}])
def test_invalid_when(sched, when):
    with pytest.raises(ValueError):
        sched.save_task({"prompt": "x", "when": when})


def test_crud(sched):
    now = ts(2026, 10, 14, 7, 0)
    t = sched.save_task({"title": "News", "prompt": "Summarize the news", "when": {"type": "daily", "time": "8:00"}}, now)
    assert t["when"]["time"] == "08:00" and t["next_run"] == ts(2026, 10, 14, 8, 0) and t["enabled"]
    t2 = sched.save_task(t | {"enabled": False}, now)
    assert t2["id"] == t["id"] and t2["next_run"] is None and len(sched.list_()) == 1
    assert sched.save_task({"prompt": "  hello  ", "when": {"type": "interval", "minutes": 5}}, now)["title"] == "hello"
    with pytest.raises(ValueError):
        sched.save_task({"prompt": " ", "when": {"type": "interval", "minutes": 5}})
    assert sched.delete(t["id"]) and not sched.delete(t["id"]) and len(sched.list_()) == 1


def test_due_task_runs(sched):
    calls = []
    runner = lambda prompt, title: calls.append((prompt, title)) or "sess-1"  # noqa: E731
    now = ts(2026, 10, 14, 7, 0)
    t = sched.save_task({"title": "News", "prompt": "Summarize", "when": {"type": "daily", "time": "08:00"}}, now)
    late = sched.save_task({"prompt": "later", "when": {"type": "daily", "time": "20:00"}}, now)
    assert sched.tick(ts(2026, 10, 14, 7, 59), runner) == [] and calls == []
    assert sched.tick(ts(2026, 10, 14, 8, 0, 10), runner) == [t["id"]] and calls == [("Summarize", "News")]
    t = sched.get(t["id"])
    assert t["last_session"] == "sess-1" and t["last_error"] is None and t["last_run"]
    assert t["next_run"] > ts(2026, 10, 14, 8, 0) and sched.unread() == [t["id"]]
    assert sched.get(late["id"])["last_run"] is None
    sched.mark_read(t["id"])
    assert sched.unread() == []


def test_disabled_and_no_runner(sched, monkeypatch):
    now = ts(2026, 10, 14, 7, 0)
    sched.save_task({"prompt": "x", "when": {"type": "interval", "minutes": 1}, "enabled": False}, now)
    assert sched.tick(now + 3600, lambda p, t: "s") == []
    sched.save_task({"prompt": "y", "when": {"type": "interval", "minutes": 1}}, now)
    monkeypatch.setattr(sched, "_runner", None)
    assert sched.tick(now + 3600) == []  # no runner registered: nothing runs


def test_once_runs_then_disables(sched):
    now = ts(2026, 10, 14, 7, 0)
    t = sched.save_task({"prompt": "x", "when": {"type": "once", "at": now + 60}}, now)
    assert sched.tick(now + 120, lambda p, title: "s9") == [t["id"]]
    t = sched.get(t["id"])
    assert t["enabled"] is False and t["next_run"] is None and t["last_session"] == "s9"
    assert sched.tick(now + 99999, lambda p, title: "s10") == []


def test_runner_error_is_recorded(sched):
    def boom(prompt, title):
        raise RuntimeError("no assistant loaded")
    now = ts(2026, 10, 14, 7, 0)
    t = sched.save_task({"prompt": "x", "when": {"type": "interval", "minutes": 30}}, now)
    sched.tick(now + 1800, boom)
    t = sched.get(t["id"])
    assert t["last_error"] == "no assistant loaded" and t["next_run"] is not None and not t["unread"]


def test_run_now(sched):
    now = ts(2026, 10, 14, 7, 0)
    t = sched.save_task({"prompt": "x", "when": {"type": "weekly", "day": 0, "time": "08:00"}}, now)
    assert not sched.run_now("missing", wait=True, runner=lambda p, title: "s")
    assert sched.run_now(t["id"], wait=True, runner=lambda p, title: "s2")
    assert sched.get(t["id"])["last_session"] == "s2"
