# tests/test_agent_serp.py
import pytest

# Import the AGENT wrapper you want to test.
# If your agent is in a different module (e.g., agents.agent_google_shopping),
# change this import accordingly.
from agents.agent_serp import google_shopping
import logging


def _call_google_shopping(q, num=None, location=None):
    """Call google_shopping whether it's a LangChain Tool or a plain function."""
    if hasattr(google_shopping, "invoke"):
        payload = {"q": q}
        if num is not None:
            payload["num"] = num
        if location is not None:
            payload["location"] = location
        return google_shopping.invoke(payload)
    # Plain function fallback
    kwargs = {}
    if num is not None:
        kwargs["num"] = num
    if location is not None:
        kwargs["location"] = location
    return google_shopping(q, **kwargs)


def fake_google_shopping_search(q, num=20, location=None):
    # Return a fixed result for testing
    loc = location or "Default"
    data = [
        {"title": f"Product for {q}", "price": 19.99, "location": loc},
        {"title": f"Another {q}", "price": 25.00, "location": loc},
    ]
    return data[: num if num is not None else 20]


@pytest.fixture
def patch_google_shopping_search(monkeypatch):
    # Patch the IMPORT SITE that the agent uses internally
    # If your agent module name differs, update this dotted path.
    monkeypatch.setattr(
        "agents.agent_serp.google_shopping_search",
        fake_google_shopping_search,
        raising=True,
    )


def test_google_shopping_basic(patch_google_shopping_search):
    results = _call_google_shopping("iphone case")
    assert isinstance(results, list)
    assert len(results) >= 1
    assert results[0]["title"].startswith("Product for iphone case")


def test_google_shopping_num_limit(patch_google_shopping_search):
    results = _call_google_shopping("ipad", num=1)
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["title"] == "Product for ipad"


def test_google_shopping_location(patch_google_shopping_search):
    loc = "Philadelphia, Pennsylvania, United States"
    results = _call_google_shopping("macbook", location=loc)
    assert isinstance(results, list)
    assert len(results) >= 1
    assert all(r["location"] == loc for r in results)


def test_google_shopping_empty_result(monkeypatch):
    def empty_search(q, num=20, location=None):
        return []
    # Patch the same import site for this test
    monkeypatch.setattr(
        "agents.agent_serp.google_shopping_search",
        empty_search,
        raising=True,
    )
    results = _call_google_shopping("nonexistent")
    assert results == []
