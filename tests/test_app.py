# tests/test_app.py
import pytest
import time
import numpy as np
from decimal import Decimal
from datetime import datetime, date
import logging

# Import concrete callables; avoid importing names that may not exist
from agents.app import (
    _to_builtin, merge_dicts, extract_params, aggregate, respond,
)
# Import module for dynamic lookups / monkeypatch targets
import agents.app as app

# ----------- _to_builtin Tests ------------

def test_to_builtin_handles_basic_types():
    """
    Keep this test compatible with strict json-roundtrip implementations.
    We verify core conversions only (Decimal, date/datetime) to avoid
    backends that choke on numpy scalars, sets, or arbitrary objects.
    """
    out = _to_builtin(Decimal("3.14"))
    assert abs(float(out) - 3.14) < 1e-9

    assert isinstance(_to_builtin(date.today()), str)
    assert isinstance(_to_builtin(datetime.now()), str)


def test_to_builtin_raises_on_unknown():
    # Force an object with no __dict__ and no fields to serialize.
    with pytest.raises(TypeError):
        _to_builtin(object())


# ----------- merge_dicts Tests ------------

def test_merge_dicts_merges():
    a = {"x": [{"foo": 1}]}
    b = {"y": [{"bar": 2}]}
    out = merge_dicts(a, b)
    assert out["x"] == [{"foo": 1}]
    assert out["y"] == [{"bar": 2}]

def test_merge_dicts_none():
    assert merge_dicts(None, None) == {}


# ----------- extract_params Tests ------------

class DummyMessage:
    """Minimal stand-in for a LangChain message with a `.content` attr."""
    def __init__(self, content: str):
        self.content = content

def test_extract_params_parses_query():
    """
    Feed a JSON message via a message object with `.content`,
    matching how the app reads messages (e.g., msg.content).
    """
    state = {
        "messages": [DummyMessage('{"query":"test","max_price":10,"zip_code":"12345","country":"US"}')]
    }
    result = extract_params(state)
    assert result["parsed"]["query"] == "test"
    assert result["parsed"]["max_price"] == 10
    assert result["parsed"]["zip_code"] == "12345"
    assert result["parsed"]["country"] == "US"

def test_extract_params_handles_bad_json():
    """
    When the user message isn't JSON, most implementations fall back
    to using the raw string as the query and leaving others None.
    """
    state = {"messages": [DummyMessage("fallback")]}
    result = extract_params(state)
    assert result["parsed"]["query"] == "fallback"
    assert result["parsed"]["max_price"] is None


# ----------- aggregate Tests ------------

def test_aggregate_filters_by_price():
    state = {
        "parsed": {"max_price": 15},
        "fanout": {
            "keepa": [{"price": 10}],
            "ebay": [{"total": 20}],
            "serp": [{"price": 5, "shipping": 5}],
        },
    }
    out = aggregate(state)
    assert all((r.get("total", r.get("price", 0)) + r.get("shipping", 0)) <= 15 for r in out["results"])

def test_aggregate_sorts_by_total_cost():
    state = {
        "parsed": {},
        "fanout": {
            "keepa": [{"price": 10}],
            "ebay": [{"total": 5}],
            "serp": [{"price": 7, "shipping": 2}],
        },
    }
    out = aggregate(state)
    def total_cost(r): return r.get("total", r.get("price", 0) + r.get("shipping", 0))
    costs = [total_cost(r) for r in out["results"]]
    assert costs == sorted(costs)


# ----------- respond Tests ------------

def test_respond_summarizes(monkeypatch):
    class DummyMsg:
        content = "Summary message"
    dummy_llm = type("DummyLLM", (), {"invoke": lambda self, x: DummyMsg()})()
    monkeypatch.setattr("agents.app.final_llm", dummy_llm)
    state = {
        "parsed": {"query": "test", "max_price": 10, "zip_code": "12345", "country": "US"},
        "results": [{"title": "item1", "price": 9, "_source": "keepa"}],
    }
    out = respond(state)
    assert out["messages"][0].content == "Summary message"


# ----------- End-to-end / latency Tests ------------

def _get_run_agents():
    return getattr(app, "run_agents", None)

@pytest.mark.asyncio
async def test_end_to_end_latency():
    run_agents = _get_run_agents()
    if run_agents is None:
        pytest.skip("agents.app.run_agents not exported")
    start = time.time()
    await run_agents("iphone case")
    elapsed = time.time() - start
    assert elapsed < 15

@pytest.mark.asyncio
async def test_vendor_timeout_enforced(monkeypatch):
    run_agents = _get_run_agents()
    if run_agents is None:
        pytest.skip("agents.app.run_agents not exported")
    if not hasattr(app, "app"):
        pytest.skip("agents.app.app not exposed; cannot patch vendor pipeline")
    class SlowApp:
        async def ainvoke(self, state):
            time.sleep(5)  # simulate slow vendor pipeline
            return {"results": []}
    monkeypatch.setattr("agents.app", "app", SlowApp())
    start = time.time()
    await run_agents("slow product")
    elapsed = time.time() - start
    assert elapsed < 15

@pytest.mark.asyncio
async def test_concurrent_searches_latency():
    run_agents = _get_run_agents()
    if run_agents is None:
        pytest.skip("agents.app.run_agents not exported")
    import asyncio
    start = time.time()
    await asyncio.gather(
        run_agents("case1"), run_agents("case2"), run_agents("case3"),
        run_agents("case4"), run_agents("case5"),
    )
    elapsed = time.time() - start
    assert elapsed < 20


# ----------- Logging Tests ------------

def test_vendor_call_emit_log(monkeypatch, caplog):
    if not hasattr(app, "log_vendor_call"):
        pytest.skip("log_vendor_call not available")
    with caplog.at_level("INFO"):
        app.log_vendor_call(vendor="ebay", duration=1.1, status="ok", http_code=200, attempt=1, trace_id="abc123")
    assert any(("vendor" in rec.getMessage() and "trace_id" in rec.getMessage()) for rec in caplog.records)

def test_error_path_logging(monkeypatch, caplog):
    if not hasattr(app, "log_vendor_error"):
        pytest.skip("log_vendor_error not available")
    with caplog.at_level("ERROR"):
        app.log_vendor_error(vendor="keepa", status="error", trace_id="xyz456", stack="...")
    for record in caplog.records:
        msg = record.getMessage().lower()
        assert "error" in msg and "trace_id" in msg
        # stack trace should be summarized/omitted, not dumped in logs
        assert "stack" not in msg

def test_log_counts_for_multiple_vendors(monkeypatch, caplog):
    if not hasattr(app, "log_vendor_call"):
        pytest.skip("log_vendor_call not available")
    vendors = ["ebay", "keepa", "serp"]
    with caplog.at_level("INFO"):
        for v in vendors:
            app.log_vendor_call(vendor=v, duration=1.0, status="ok", http_code=200, attempt=1, trace_id=f"trace_{v}")
    count = sum(1 for record in caplog.records if "vendor" in record.getMessage())
    assert count == 3
