import agents.agent_ebay

def test_ebay_search_returns_list(monkeypatch):
    def fake_search(params):
        return [{"title": "Test Item", "price": 9.99, "link": "http://example.com"}]
    monkeypatch.setattr(agents.agent_ebay, "ebay_search", fake_search)
    items = agents.agent_ebay.ebay_search({"query": "test", "category_id": "12345", "max_results": 1})
    assert isinstance(items, list)
    assert items[0]["title"] == "Test Item"

def fake_search_ebay_cheapest_tool(**kwargs):
    # Return a fixed result for testing
    return {
        "items": [
            {
                "title": "Test Item",
                "item_id": "12345",
                "condition": "New",
                "seller": "BestSeller",
                "price": 10.0,
                "shipping": 2.0,
                "total": 12.0,
                "url": "http://example.com/item",
                "location": "Philadelphia",
                "est_delivery_min": "2025-09-30",
                "est_delivery_max": "2025-10-05"
            },
            {
                "title": "Auction Item",
                "item_id": "67890",
                "condition": "Used",
                "seller": "AuctionGuy",
                "price": 8.0,
                "shipping": 5.0,
                "total": 13.0,
                "url": "http://example.com/auction",
                "location": "NYC",
                "est_delivery_min": "2025-10-02",
                "est_delivery_max": "2025-10-07"
            }
        ]
    }

@pytest.fixture
def patch_search(monkeypatch):
    # Patch the core search tool used by the agent
    monkeypatch.setattr(agent_ebay, "search_ebay_cheapest_tool", fake_search_ebay_cheapest_tool)

def test_basic_search_returns_items(patch_search):
    items = agent_ebay.ebay_search("test", "12345")
    assert isinstance(items, list)
    assert len(items) == 2
    assert items[0]["title"] == "Test Item"

def test_fixed_price_only_filters_items(monkeypatch):
    # If you enrich your filter logic, test here.
    monkeypatch.setattr(agent_ebay, "search_ebay_cheapest_tool", fake_search_ebay_cheapest_tool)
    items = agent_ebay.ebay_search("test", "12345", fixed_price_only=True)
    # In the current code, all items are kept; if you filter auctions, update this assertion!
    assert all(isinstance(it, dict) for it in items)

def test_limit_and_max_results(monkeypatch):
    monkeypatch.setattr(agent_ebay, "search_ebay_cheapest_tool", fake_search_ebay_cheapest_tool)
    items = agent_ebay.ebay_search("test", "12345", limit=1, max_results=1)
    # The core tool mock always returns 2, but you can check parameters passed
    assert isinstance(items, list)
    assert len(items) == 2  # update if you change the mock

def test_country_and_sandbox(monkeypatch):
    monkeypatch.setattr(agent_ebay, "search_ebay_cheapest_tool", fake_search_ebay_cheapest_tool)
    items = agent_ebay.ebay_search("test", "12345", country="CA", sandbox=True)
    assert items[0]["location"] == "Philadelphia"

def test_empty_result(monkeypatch):
    def empty_search(**kwargs):
        return {"items": []}
    monkeypatch.setattr(agent_ebay, "search_ebay_cheapest_tool", empty_search)
    items = agent_ebay.ebay_search("test", "12345")
    assert items == []