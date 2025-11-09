# tests/test_queries_integration.py
import pandas as pd
from sqlalchemy import text

def fetch_canonicals(conn, q=None, limit=100):
    where = ""
    params = {"limit": limit}
    if q:
        where = "WHERE title LIKE :q"
        params["q"] = f"%{q}%"
    sql = f"""
      SELECT canonical_key, title, min_price, avg_price, max_price, seller_count, total_listings
      FROM canonical_product_view
      {where}
      ORDER BY avg_price IS NULL, avg_price ASC
      LIMIT :limit
    """
    return pd.read_sql_query(sql, conn.connection, params=params)

def fetch_variants(conn, canonical_key):
    sql = """
    WITH base AS (
      SELECT
        id AS row_id,
        COALESCE(json_extract(raw,'$.product_id'), lower(trim(title))) AS canonical_key,
        title, seller, source, link,
        COALESCE(link, json_extract(raw,'$.product_link')) AS product_url,
        CAST(total AS REAL) AS total_price,
        CAST(price AS REAL) AS unit_price
      FROM product_results
    )
    SELECT seller, source, title AS listing_title,
           COALESCE(total_price, unit_price) AS price, product_url
    FROM base WHERE canonical_key = :ck
    ORDER BY price IS NULL, price ASC
    """
    return pd.read_sql_query(sql, conn.connection, params={"ck": canonical_key})

def test_fetch_canonicals_returns_one_row_per_key(conn):
    df = fetch_canonicals(conn, q="iPhone 15")
    # Ensure titles exist and dedup works (no duplicate canonical_key)
    assert not df.empty
    assert df["canonical_key"].is_unique

def test_fetch_variants_lists_all_vendor_rows(conn):
    # pick any key from the view and ensure >= 1 variant row
    ck = conn.execute(text("SELECT canonical_key FROM canonical_product_view LIMIT 1")).scalar_one()
    vdf = fetch_variants(conn, ck)
    assert not vdf.empty
    # columns present
    for col in ["seller", "listing_title", "price", "product_url"]:
        assert col in vdf.columns
