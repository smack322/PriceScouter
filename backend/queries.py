# backend/queries.py
from typing import Optional
import pandas as pd
from sqlalchemy import text
from .local_db.db import engine

def fetch_canonicals(limit: int = 200, q: Optional[str] = None) -> pd.DataFrame:
    """
    Return canonical products with aggregate pricing, no view required.
    Columns returned:
      canonical_id, canonical_key, title, min_price, avg_price, max_price,
      seller_count, total_listings, representative_url
    """
    where = ""
    params = {"limit": limit}
    if q:
        where = "WHERE LOWER(cp.title) LIKE :q OR LOWER(COALESCE(cp.brand, '')) LIKE :q"
        params["q"] = f"%{q.lower()}%"

    sql = f"""
    SELECT
        cp.id AS canonical_id,
        COALESCE(cp.variant_key, cp.model, CAST(cp.id AS TEXT)) AS canonical_key,
        cp.title AS title,
        MIN(v.price) AS min_price,
        AVG(v.price) AS avg_price,
        MAX(v.price) AS max_price,
        COUNT(DISTINCT v.seller) AS seller_count,
        COUNT(v.id) AS total_listings,
        MIN(v.url) AS representative_url
    FROM canonical_product cp
    LEFT JOIN variant v ON v.canonical_id = cp.id
    {where}
    GROUP BY cp.id, canonical_key, cp.title
    ORDER BY (avg_price IS NULL), avg_price ASC
    LIMIT :limit
    """
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params)

def fetch_variants(
    canonical_id: Optional[int] = None,
    canonical_key: Optional[str] = None,
    limit: int = 200
) -> pd.DataFrame:
    """
    Return variants for a canonical. Prefer canonical_id; falls back to canonical_key.
    Returns columns: seller, source, price, currency, product_url, listing_title, shipping, condition, created_at
    """
    if not canonical_id and not canonical_key:
        raise ValueError("Provide canonical_id or canonical_key")

    where = "v.canonical_id = :cid" if canonical_id else "cp.variant_key = :ck OR CAST(cp.id AS TEXT) = :ck"
    params = {"cid": canonical_id} if canonical_id else {"ck": str(canonical_key)}

    sql = f"""
    SELECT
        v.seller,
        v.source,
        v.price,
        v.currency,
        v.url AS product_url,         -- alias to match UI
        v.title AS listing_title,     -- alias to match UI
        v.shipping,
        v.condition,
        v.created_at
    FROM variant v
    JOIN canonical_product cp ON cp.id = v.canonical_id
    WHERE {where}
    ORDER BY v.price ASC NULLS LAST, v.created_at DESC
    LIMIT :limit
    """
    params["limit"] = limit
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params)