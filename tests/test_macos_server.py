"""Platform-independent tests for the Apple unified-memory inference settings."""


def test_apple_auto_ctx_reserves_memory_for_the_os(monkeypatch):
    import server

    monkeypatch.setattr(server.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(server.syscheck, "current", lambda: {"assess": {"model_mem_gb": 12}})
    assert server.auto_ctx({"size_gb": 9}) == 8192
    assert server.auto_ctx({"size_gb": 6}) == 16384
    assert server.auto_ctx({"size_gb": 2}) == 32768
    monkeypatch.setattr(server.syscheck, "current", lambda: {"assess": {"model_mem_gb": 32}})
    assert server.auto_ctx({"size_gb": 8}) == 65536
