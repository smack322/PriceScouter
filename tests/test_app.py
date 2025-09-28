import pytest
import time
import numpy as np
from decimal import Decimal
from datetime import datetime, date
from app import (
    _to_builtin, merge_dicts, extract_params, aggregate, respond,
    State, Parsed
)

# ----------- _to_builtin Tests ------------

def test_to_builtin_handles_basic_types():
    assert _to_builtin(Decimal("3.14")) == 3.14
    assert _to_builtin(np.int64(42)) == 42
    assert _to_builtin(np.float64(3.2)) == 3.2
    assert _to_builtin(np.bool_(True)) is True
    assert _to_builtin(np.array([1,2,3])).tolist() == [1,2,3]
    today = date.today()
    now = datetime.now()
    assert isinstance(_to_builtin(today), str)
    assert isinstance(_to_builtin(now), str)
    assert _to_builtin(set([1,2])) == [1,2]
    class Dummy: pass
    obj = Dummy()
    obj.x = 1
    obj.y = 2
    assert _to_builtin(obj) == {'x': 1, 'y': 2}

def test_to_builtin_raises_on_unknown():
    class NotSerializable: pass
    with pytest.raises(TypeError):
        _to_builtin(NotSerializable())

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

def test_extract_params_parses_query(monkeypatch):
    # Mock the LLM .invoke to return a dummy response
    class DummyMsg:
        content = '{"query": "test", "max_price": 10, "zip_code": "12345", "country": "US"}'
    dummy_llm = type("DummyLLM", (), {"invoke": lambda self, x: DummyMsg()})()
    monkeypatch.setattr("app.extract_llm", dummy_llm)
    state = {"messages": [{"content": "test"}]}
    result = extract_params(state)
    assert result["parsed"]["query"] == "test"
    assert result["parsed"]["max_price"] == 10

def test_extract_params_handles_bad_json(monkeypatch):
    class DummyMsg:
        content = "not json"
    dummy_llm = type("DummyLLM", (), {"invoke": lambda self, x: DummyMsg()})()
    monkeypatch.setattr("app.extract_llm", dummy_llm)
    state = {"messages": [{"content": "fallback"}]}
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
            "serp": [{"price": 5, "shipping": 5}]
        }
    }
    out = aggregate(state)
    # Only items with total cost <= 15 should remain
    assert all((r.get("price", r.get("total", 0)) + r.get("shipping", 0)) <= 15 for r in out["results"])

def test_aggregate_sorts_by_total_cost():
    state = {
        "parsed": {},
        "fanout": {
            "keepa": [{"price": 10}],
            "ebay": [{"total": 5}],
            "serp": [{"price": 7, "shipping": 2}]
        }
    }
    out = aggregate(state)
    results = out["results"]
    # Should be sorted by total cost
    costs = []
    for r in results:
        if "total" in r:
            costs.append(r["total"])
        else:
            costs.append(r.get("price", 0) + r.get("shipping", 0))
    assert costs == sorted(costs)

# ----------- respond Tests ------------

def test_respond_summarizes(monkeypatch):
    # Mock the LLM .invoke to return a dummy message
    class DummyMsg:
        content = "Summary message"
    dummy_llm = type("DummyLLM", (), {"invoke": lambda self, x: DummyMsg()})()
    monkeypatch.setattr("app.final_llm", dummy_llm)
    state = {
        "parsed": {"query": "test", "max_price": 10, "zip_code": "12345", "country": "US"},
        "results": [{"title": "item1", "price": 9, "_source": "keepa"}]
    }
    out = respond(state)
    assert out["messages"][0].content == "Summary message"

@pytest.mark.asyncio
async def test_end_to_end_latency():
    start = time.time()
    await run_agents("iphone case")
    elapsed = time.time() - start
    assert elapsed < 15

@pytest.mark.asyncio
async def test_vendor_timeout_enforced(monkeypatch):
    class SlowApp:
        async def ainvoke(self, state):
            time.sleep(5)  # Simulate slow vendor
            return {"results": []}
    monkeypatch.setattr("agents.app", "app", SlowApp())
    start = time.time()
    await run_agents("slow product")
    elapsed = time.time() - start
    assert elapsed < 15  # Still completes in overall limit

@pytest.mark.asyncio
async def test_concurrent_searches_latency():
    import asyncio
    start = time.time()
    await asyncio.gather(
        run_agents("case1"), run_agents("case2"), run_agents("case3"),
        run_agents("case4"), run_agents("case5")
    )
    elapsed = time.time() - start
    assert elapsed < 20  # P99 requirement

def test_vendor_call_emit_log(monkeypatch, caplog):
    with caplog.at_level("INFO"):
        app.log_vendor_call(vendor="ebay", duration=1.1, status="ok", http_code=200, attempt=1, trace_id="abc123")
    found = False
    for record in caplog.records:
        if "vendor" in record.getMessage() and "trace_id" in record.getMessage():
            found = True
    assert found

def test_error_path_logging(monkeypatch, caplog):
    with caplog.at_level("ERROR"):
        app.log_vendor_error(vendor="keepa", status="error", trace_id="xyz456", stack="...")
    for record in caplog.records:
        assert "error" in record.getMessage()
        assert "trace_id" in record.getMessage()
        assert "stack" not in record.getMessage()  # stack trace should be summarized or omitted

def test_log_counts_for_multiple_vendors(monkeypatch, caplog):
    vendors = ["ebay", "keepa", "serp"]
    with caplog.at_level("INFO"):
        for v in vendors:
            app.log_vendor_call(vendor=v, duration=1.0, status="ok", http_code=200, attempt=1, trace_id=f"trace_{v}")
    count = sum(1 for record in caplog.records if "vendor" in record.getMessage())
    assert count == 3
