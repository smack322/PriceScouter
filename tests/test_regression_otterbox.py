# tests/test_regression_otterbox.py
from sqlalchemy import text

def test_otterbox_bestbuy_regression(conn):
    # Using the known BestBuy Otterbox product_id key from the seed
    ck = "7920095867532335495"
    row = conn.execute(text("""
      SELECT title, min_price, avg_price, max_price, seller_count, total_listings
      FROM canonical_product_view WHERE canonical_key=:k
    """), {"k": ck}).mappings().first()
    assert row is not None
    assert "Otterbox" in row["title"] or "OtterBox" in row["title"]
    assert abs(row["min_price"] - 39.09) < 1e-6
    assert abs(row["avg_price"] - 39.09) < 1e-6
    assert abs(row["max_price"] - 39.09) < 1e-6
    assert row["seller_count"] == 1
    assert row["total_listings"] == 1
