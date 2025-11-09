# tests/test_canonical_view_unit.py
from sqlalchemy import text

def test_canonical_groups_by_product_id(conn):
    rows = conn.execute(text("""
    SELECT canonical_key, COUNT(*) AS n
    FROM canonical_product_view
    GROUP BY canonical_key
    """)).mappings().all()
    keys = {r["canonical_key"] for r in rows}
    # 3 distinct product_ids in sample seed:
    #   7920095867532335495 (Otterbox BestBuy)
    #   3917673451152378843 (Apple)
    #   792030822556606110  (Speck)
    #   + Walmart row has a *different* product_id, so total 4 keys
    assert len(keys) == 4

def test_canonical_aggregates_min_avg_max(conn):
    # Pull Otterbox BestBuy key
    otter_key = "7920095867532335495"
    row = conn.execute(text("""
      SELECT min_price, avg_price, max_price, seller_count, total_listings
      FROM canonical_product_view WHERE canonical_key=:k
    """), {"k": otter_key}).mappings().first()
    assert row is not None
    # Only one listing for that key in our seed
    assert row["min_price"] == 39.09
    assert row["max_price"] == 39.09
    assert abs(row["avg_price"] - 39.09) < 1e-6
    assert row["seller_count"] == 1
    assert row["total_listings"] == 1
