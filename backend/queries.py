# backend/queries.py
from typing import List, Dict, Any, Optional
import pandas as pd
from sqlalchemy import text
from .db import engine  # your SQLAlchemy engine factory

from .chart_adapter import df_to_chart_points

def fetch_chart_data_for_search(search_id: int) -> list[dict]:
    """
    Returns a list[ChartPointDict] for the chart UI.
    """
    df: pd.DataFrame = fetch_canonicals_for_chart(search_id)  # your existing query
    return df_to_chart_points(df)

def fetch_canonicals(limit: int = 200, q: Optional[str] = None) -> pd.DataFrame:
    where = ""
    params = {"limit": limit}
    if q:
        where = "WHERE title LIKE :q"
        params["q"] = f"%{q}%"
    sql = f"""
    SELECT canonical_id, canonical_key, title,
           min_price, avg_price, max_price,
           seller_count, total_listings, representative_url
    FROM canonical_product_view
    {where}
    ORDER BY avg_price IS NULL, avg_price ASC
    LIMIT :limit
    """
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params)

def fetch_variants(canonical_key: str) -> List[Dict[str, Any]]:
    sql = """
    WITH base AS (
      SELECT
        id AS row_id,
        COALESCE(
          json_extract(raw, '$.product_id'),
          lower(trim(replace(replace(title, '®',''), '™','')))
        )                               AS canonical_key,
        title,
        seller,
        source,
        link,
        COALESCE(link, json_extract(raw, '$.product_link')) AS product_url,
        CAST(total AS REAL) AS total_price,
        CAST(price AS REAL) AS unit_price,
        shipping,
        currency,
        created_at,
        json_extract(raw, '$.condition_guess') AS condition,
        json_extract(raw, '$.brand_guess')     AS brand
      FROM product_results
    )
    SELECT
      seller,
      source,
      title AS listing_title,
      COALESCE(total_price, unit_price) AS price,
      shipping,
      currency,
      condition,
      brand,
      product_url
    FROM base
    WHERE canonical_key = :ck
    ORDER BY price IS NULL, price ASC, created_at ASC
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"ck": canonical_key}).mappings().all()
    return [dict(r) for r in rows]
