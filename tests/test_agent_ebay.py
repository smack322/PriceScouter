import agents.agent_ebay

def test_ebay_search_returns_list(monkeypatch):
    def fake_search(params):
        return [{"title": "Test Item", "price": 9.99, "link": "http://example.com"}]
    monkeypatch.setattr(agents.agent_ebay, "ebay_search", fake_search)
    items = agents.agent_ebay.ebay_search({"query": "test", "category_id": "12345", "max_results": 1})
    assert isinstance(items, list)
    assert items[0]["title"] == "Test Item"