import pytest
import asyncio

# Import your run_agents function
import agents.client

@pytest.mark.asyncio
async def test_run_agents_returns_rows(monkeypatch):
    # Mock app.ainvoke to return a fake state
    class DummyApp:
        async def ainvoke(self, state):
            # Simulate expected state dict
            return {
                "results": [
                    {"title": "Product 1", "price": 10.0},
                    {"title": "Product 2", "price": 12.0},
                ]
            }
    monkeypatch.setattr("agents.app", "app", DummyApp())
    # Now call run_agents
    results = await run_agents("iphone case", zip_code="12345", country="US", max_price=15)
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0]["title"] == "Product 1"
    assert results[1]["price"] == 12.0

@pytest.mark.asyncio
async def test_run_agents_respects_top_n(monkeypatch):
    class DummyApp:
        async def ainvoke(self, state):
            return {"results": [{"title": f"Product {i}"} for i in range(20)]}
    monkeypatch.setattr("agents.app", "app", DummyApp())
    results = await run_agents("ipad case", top_n=5)
    assert len(results) == 5
    assert results[0]["title"] == "Product 0"

@pytest.mark.asyncio
async def test_run_agents_empty_results(monkeypatch):
    class DummyApp:
        async def ainvoke(self, state):
            return {"results": []}
    monkeypatch.setattr("agents.app", "app", DummyApp())
    results = await run_agents("something nonexistent")
    assert results == []
