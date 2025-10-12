from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///./app.db", future=True, connect_args={"check_same_thread": False})

with engine.begin() as conn:
    # What tables exist?
    tables = list(conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'").scalars())
    print("Tables:", tables)

    # Schema for search_history
    cols = conn.exec_driver_sql("PRAGMA table_info(search_history)").all()
    print("\nsearch_history columns:")
    for c in cols:
        print(c)

    # Count
    n = conn.execute(text("SELECT COUNT(*) FROM search_history")).scalar_one()
    print(f"\nRow count: {n}")

    # Latest 5 rows
    rows = conn.execute(text("""
        SELECT id, agent, status, query, results_count, duration_ms, created_at
        FROM search_history
        ORDER BY id DESC
        LIMIT 5
    """)).all()
    print("\nLatest 5:")
    for r in rows:
        print(dict(r._mapping))
