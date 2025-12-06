-- backend/sql/canonical_product_view.sql

DROP VIEW IF EXISTS canonical_product_view;

CREATE VIEW canonical_product_view AS
WITH base AS (
  SELECT
    pr.id AS row_id,
    pr.search_id AS search_id,
    sh.query AS orig_query,
    sh.zip_code,
    sh.country,

    COALESCE(
      json_extract(pr.raw, '$.product_id'),
      lower(trim(replace(replace(pr.title, '®',''), '™','')))
    ) AS canonical_key,

    pr.title,
    pr.seller,
    pr.source,
    pr.link, -- raw link column (may be NULL)
    COALESCE(pr.link, json_extract(pr.raw, '$.product_link')) AS product_url,
    CAST(pr.total AS REAL) AS total_price,
    CAST(pr.price AS REAL) AS unit_price,
    pr.shipping,
    pr.currency,
    pr.created_at
  FROM product_results pr
  LEFT JOIN search_history sh ON pr.search_id = sh.id
)
SELECT
  -- simple, stable integer id per canonical_key
  MIN(row_id) AS canonical_id,
  canonical_key,

  -- representative title (cheapest non-null price; if tie, earliest created)
  (
    SELECT b2.title
    FROM base b2
    WHERE b2.canonical_key = b.canonical_key
    ORDER BY
      (COALESCE(b2.total_price, b2.unit_price) IS NULL), -- non-null first
      COALESCE(b2.total_price, b2.unit_price),
      b2.created_at
    LIMIT 1
  ) AS title,

  -- aggregated stats
  MIN(COALESCE(total_price, unit_price)) AS min_price,
  AVG(COALESCE(total_price, unit_price)) AS avg_price,
  MAX(COALESCE(total_price, unit_price)) AS max_price,
  COUNT(DISTINCT seller) AS seller_count,
  COUNT(*) AS total_listings,

  -- representative URL (from cheapest row)
  (
    SELECT b2.product_url
    FROM base b2
    WHERE b2.canonical_key = b.canonical_key
    ORDER BY
      (COALESCE(b2.total_price, b2.unit_price) IS NULL),
      COALESCE(b2.total_price, b2.unit_price),
      b2.created_at
    LIMIT 1
  ) AS representative_url
FROM base b
GROUP BY canonical_key;
