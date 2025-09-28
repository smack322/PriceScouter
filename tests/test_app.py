import pytest
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
