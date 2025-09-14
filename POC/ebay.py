import requests

TOKEN = "YOUR_OAUTH_TOKEN"
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"
}

params = {
    "q": "nike air zoom",
    "limit": "10",
    "filter": "buyingOptions:{FIXED_PRICE};price:[50..200]"
}

r = requests.get(
    "https://api.ebay.com/buy/browse/v1/item_summary/search",
    headers=headers, params=params
)
items = r.json().get("itemSummaries", [])
