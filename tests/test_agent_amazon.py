# tests/test_req_025_amazon_agent.py
import asyncio
import json
import types
import pytest

import agents.app as app

#
# ---------- T-025-1: Unit ----------
#

def test_graph_has_amazon_not_keepa():
    """
    T-025-1 (Unit)
    Validate that graph.nodes includes "amazon" and not "keepa".
    """
    nodes = set(app.graph.nodes.keys())
    assert "amazon" in nodes, "amazon node missing from graph"
    assert "keepa" not in nodes, "keepa node should be removed from graph"


#
# ---------- T-025-2: Integration ----------
#

@pytest.mark.asyncio
async def test_integration_amazon_results_have_source_amazon(monkeypatch):
    import json, types
    import agents.app as app

    class _FakeLLM:
        def invoke(self, _msgs):
            class _Resp:
                content = json.dumps({
                    "query": "iphone 15 case",
                    "max_price": None,
                    "zip_code": "19406",
                    "country": "US",
                })
            return _Resp()
    monkeypatch.setattr(app, "extract_llm", _FakeLLM())

    class _FakeTool:
        def __init__(self, data): self._data, self.name = data, "fake_tool"
        def invoke(self, _kwargs): return self._data

    monkeypatch.setattr(app, "amazon_products", _FakeTool([
        {"title": "Case A", "price": 9.99, "product_link": "https://www.amazon.com/dp/ASINCASEA"},
        {"title": "Case B", "price": 12.49, "product_link": "https://www.amazon.com/dp/ASINCASEB"},
    ]))
    monkeypatch.setattr(app, "google_shopping", _FakeTool([]))
    monkeypatch.setattr(app, "ebay_search", _FakeTool([]))

    state = {"messages": [types.SimpleNamespace(content="find iphone 15 case under $20")]}
    state.update(app.extract_params(state))

    fanout = {}
    for node in (app.call_amazon, app.call_serp, app.call_ebay):
        out = await node({"parsed": state["parsed"]})
        fanout.update(out.get("fanout", {}))

    agg = app.aggregate({"parsed": state["parsed"], "fanout": fanout})
    results = agg["results"]

    assert results, "No aggregated results"
    assert all(r.get("_source") == "amazon" for r in results)
    assert all(isinstance(r.get("product_link"), str) and r["product_link"].startswith("http") for r in results)
    assert not any(r.get("_source") == "keepa" for r in results)



#
# ---------- T-025-3: Regression (UI-path schema smoke test) ----------
#

def test_ui_amazon_search_schema_smoke(monkeypatch):
    import frontend.chatbot_ui as ui

    monkeypatch.setattr(ui, "AMAZON_TOOLS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(ui, "AMAZON_TOOLS_HAS_AMAZON", True, raising=False)

    class _FakeAmazonTools:
        @staticmethod
        def amazon_search(**kwargs):
            return [
                {"title": "Case Z", "price": 10.0, "price_str": "$10.00", "seller": "ShopCo",
                 "rating": 4.5, "link": "https://www.amazon.com/dp/ASINZ", "asin": "ASINZ"},
                {"title": "Case Y", "price": 14.99, "shipping_str": "Free delivery", "seller_domain": "amazon.com",
                 "asin": "ASINY"},
            ]
    monkeypatch.setattr(ui, "amazon_tools", _FakeAmazonTools(), raising=False)

    df = ui.amazon_search(q="iphone 15 case", key="", loc="United States", n=5)

    assert not df.empty
    assert "price_display" in df.columns
    assert "product_link" in df.columns

    if "_source" not in df.columns:
        df["_source"] = "amazon"
    assert df["_source"].astype(str).eq("amazon").all()

    df_clean = df.drop(columns=[c for c in ui.BANNED_GOOGLE_UI_COLS if c in df.columns], errors="ignore")

    for banned in ui.BANNED_GOOGLE_UI_COLS:
        assert banned not in df_clean.columns
    for col in ["title", "price_display", "product_link"]:
        assert col in df_clean.columns
    assert df_clean["product_link"].notna().all()
    assert df_clean["product_link"].astype(str).str.startswith("http").all()
