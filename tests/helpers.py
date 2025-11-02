import numpy as np
AMAZON_US = "ATVPDKIKX0DER"

class DummyEmbedder:
    def __init__(self, dim=384):
        self.model = "dummy"
        self.dim = dim
    def embed(self, texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=self.dim).astype("float32")
            v /= np.linalg.norm(v) + 1e-9
            out.append(v.tolist())
        return out

def bb_history(cents):
    # Keepa histories use -1 for “no data”; we include one valid latest price
    return [-1, cents]

def mk_product(
    asin="TESTASIN1",
    title="iPhone",
    brand="Apple",
    buybox_seller_id=AMAZON_US,
    buybox_cents=9999,
    new_cents=11999,
    sales_rank=5000,
    offers=None,
    review_rating=4.5,
    review_count=250,
):
    if offers is None:
        offers = [{
            "sellerId": buybox_seller_id,
            "isAmazon": (buybox_seller_id == AMAZON_US),
            "isBuyBoxWinner": True,
            "isFBA": False,
            "isPrime": True,
        }]

    return {
        "asin": asin,
        "title": title,
        "brand": brand,
        "stats": {
            "buyBoxSellerId": buybox_seller_id,
            "buyBoxAvg90": 10999,
        },
        "offers": offers,
        # Your keepa_tools expects histories under "data" (or "csv")
        "data": {
            "BUY_BOX_SHIPPING": bb_history(buybox_cents),
            "NEW": bb_history(new_cents),
            "SALES": [sales_rank],
            "COUNT_NEW": [3],
            "COUNT_USED": [0],
        },
        "reviewRating": review_rating,
        "reviewCount": review_count,
    }
    
def upsert_titles(vstore, items):
    """items: list of (product_id, title) to embed+upsert."""
    vecs = vstore.embedder.embed([t for _, t in items])
    vstore.upsert_embeddings(list(zip([i for i, _ in items], vecs)))