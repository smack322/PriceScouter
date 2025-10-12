# db_check.py
from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///./app.db", future=True, connect_args={"check_same_thread": False})

def print_rows(title, rows):
    print(f"\n{title}")
    for r in rows:
        print(dict(r._mapping))

with engine.begin() as conn:
    # ------- What tables exist -------
    tables = list(conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'").scalars())
    print("Tables:", tables)

    # ------- Schemas -------
    sh_cols = conn.exec_driver_sql("PRAGMA table_info(search_history)").all()
    print("\nsearch_history columns:")
    for c in sh_cols:
        print(c)

    pr_cols = conn.exec_driver_sql("PRAGMA table_info(product_results)").all()
    print("\nproduct_results columns:")
    for c in pr_cols:
        print(c)

    # ------- Counts -------
    sh_count = conn.execute(text("SELECT COUNT(*) FROM search_history")).scalar_one()
    pr_count = conn.execute(text("SELECT COUNT(*) FROM product_results")).scalar_one()
    print(f"\nsearch_history row count: {sh_count}")
    print(f"product_results row count: {pr_count}")

    # ------- Latest 5 search_history rows -------
    rows = conn.execute(text("""
        SELECT id, agent, status, query, results_count, duration_ms, created_at
        FROM search_history
        ORDER BY id DESC
        LIMIT 5
    """)).all()
    print_rows("Latest 5 search_history rows:", rows)

    # ------- Latest 10 product rows (any search) -------
    rows = conn.execute(text("""
        SELECT id, search_id, source, seller, title, price, shipping, total, currency, created_at
        FROM product_results
        ORDER BY id DESC
        LIMIT 10
    """)).all()
    print_rows("Latest 10 product_results rows:", rows)

    # ------- Find the most recent aggregate search_id (merged results) -------
    latest_agg = conn.execute(text("""
        SELECT id
        FROM search_history
        WHERE agent = 'aggregate'
        ORDER BY id DESC
        LIMIT 1
    """)).scalar_one_or_none()
    print(f"\nMost recent aggregate search_id: {latest_agg}")

    if latest_agg is not None:
        # Cheapest 10 items for the latest aggregate results
        rows = conn.execute(text("""
            SELECT id, source, seller, title, total, price, shipping, link
            FROM product_results
            WHERE search_id = :sid
            ORDER BY (total IS NULL), total ASC, price ASC
            LIMIT 10
        """), {"sid": latest_agg}).all()
        print_rows("Cheapest 10 items (latest aggregate):", rows)

        # Per-source stats for that search
        rows = conn.execute(text("""
            SELECT source,
                   COUNT(*)                 AS n,
                   ROUND(AVG(total), 2)     AS avg_total,
                   MIN(total)               AS min_total,
                   MAX(total)               AS max_total
            FROM product_results
            WHERE search_id = :sid
            GROUP BY source
            ORDER BY n DESC
        """), {"sid": latest_agg}).all()
        print_rows("Per-source stats (latest aggregate):", rows)

        # Top sellers by count (latest aggregate)
        rows = conn.execute(text("""
            SELECT COALESCE(seller, '(unknown)') AS seller,
                   source,
                   COUNT(*) AS n
            FROM product_results
            WHERE search_id = :sid
            GROUP BY seller, source
            ORDER BY n DESC, seller ASC
            LIMIT 10
        """), {"sid": latest_agg}).all()
        print_rows("Top sellers (latest aggregate):", rows)

    # ------- Join products back to query text (recent 20 rows) -------
    rows = conn.execute(text("""
        SELECT pr.id, pr.source, pr.title, pr.total, pr.link, sh.query, pr.created_at
        FROM product_results pr
        JOIN search_history sh ON sh.id = pr.search_id
        ORDER BY pr.id DESC
        LIMIT 20
    """)).all()
    print_rows("Recent 20 products with their original query:", rows)
