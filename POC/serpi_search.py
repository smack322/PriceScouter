#API_KEY="cdd0ca35d1632a42e72b59dcab3206a40eb32a65c53c305e9839be62751d739a"

# pip install google-search-results
import os
from serpapi import GoogleSearch

API_KEY =  ""

params = {
    "engine": "google_shopping",
    "q": "iphone 15 case",
    "hl": "en",
    "gl": "us",
    "num": "20",
    "api_key": API_KEY,         # <-- include your key
    # Optional: narrow location if you want regional pricing
    # "location": "Philadelphia, Pennsylvania, United States"
}

search = GoogleSearch(params)
results = search.get_dict()

# Quick debug: print top-level keys and any error message
print("keys:", list(results.keys()))
if "error" in results:
    print("SerpApi error:", results["error"])

items = []
for r in results.get("shopping_results", []):
    items.append({
        "title": r.get("title"),
        "price": r.get("price"),
        "source": r.get("source"),
        "rating": r.get("rating"),
        "link": r.get("link"),
        "product_link": r.get("product_link")
    })

print(f"found {len(items)} items")
for it in items[:10]:
    print(it)

