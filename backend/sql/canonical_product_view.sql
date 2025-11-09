-- backend/sql/canonical_product_view.sql
-- Canonical key: product_id from JSON; fallback to normalized title
CREATE VIEW IF NOT EXISTS canonical_product_view AS
WITH base AS (
  SELECT
    id                              AS row_id,
    COALESCE(
      json_extract(raw, '$.product_id'),
      lower(trim(replace(replace(title, '®',''), '™','')))
    )                               AS canonical_key,
    title,
    seller,
    source,
    link,                                    -- raw link column (may be NULL)
    COALESCE(link, json_extract(raw, '$.product_link')) AS product_url,
    CAST(total AS REAL)               AS total_price,
    CAST(price AS REAL)               AS unit_price,
    shipping,
    currency,
    created_at
  FROM product_results
)
SELECT
  -- a stable id for UI; in SQLite we can hash the key
  abs(cast(substr(hex(sha1(canonical_key)), 1, 15) AS INTEGER)) AS canonical_id,
  canonical_key,
  -- choose a representative title (cheapest row’s title)
  (SELECT b2.title FROM base b2
   WHERE b2.canonical_key = b.canonical_key
   ORDER BY COALESCE(b2.total_price, b2.unit_price) ASC NULLS LAST, b2.created_at ASC
   LIMIT 1)                          AS title,
  -- aggregated stats
  MIN(COALESCE(total_price, unit_price))    AS min_price,
  AVG(COALESCE(total_price, unit_price))    AS avg_price,
  MAX(COALESCE(total_price, unit_price))    AS max_price,
  COUNT(DISTINCT seller)             AS seller_count,
  COUNT(*)                           AS total_listings,
  -- optional: a representative URL (from cheapest row)
  (SELECT b2.product_url FROM base b2
   WHERE b2.canonical_key = b.canonical_key
   ORDER BY COALESCE(b2.total_price, b2.unit_price) ASC NULLS LAST, b2.created_at ASC
   LIMIT 1)                          AS representative_url
FROM base b
GROUP BY canonical_key;
