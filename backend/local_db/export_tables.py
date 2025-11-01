# export_all_tables.py
import os
import sys
import csv
from datetime import datetime, UTC
from typing import Iterable

# --- Reuse the exact same engine/path as your app ---
try:
    from db import engine  # pulls the engine configured in db.py (absolute path, PRAGMAs, etc.)
except Exception as e:
    print("ERROR: Could not import engine from db.py:", e)
    sys.exit(1)

from sqlalchemy import text, inspect

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def list_tables_and_views() -> tuple[list[str], list[str]]:
    insp = inspect(engine)
    tables = insp.get_table_names()
    views  = insp.get_view_names()
    return tables, views

def dump_query_to_csv(sql: str, out_path: str) -> int:
    with engine.begin() as conn:
        rs = conn.execute(text(sql))
        cols = list(rs.keys())

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)

            # Iterate as dict-like rows so name-based access works
            count = 0
            for row in rs.mappings():           # <— THIS is the key change
                w.writerow([row.get(c) for c in cols])
                count += 1

    return count
    return len(rows)

def dump_table(table_name: str, out_dir: str) -> int:
    out_path = os.path.join(out_dir, f"{table_name}.csv")
    count = dump_query_to_csv(f'SELECT * FROM "{table_name}"', out_path)
    return count

def main(include_views: bool = False):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(base_dir, "data", "exports", ts)
    ensure_dir(out_dir)

    # Diagnostics so we know we’re on the right DB
    with engine.begin() as conn:
        db_list = conn.exec_driver_sql("PRAGMA database_list").all()
        sqlite_ver = conn.exec_driver_sql("SELECT sqlite_version()").scalar_one()
    print("SQLite version:", sqlite_ver)
    print("PRAGMA database_list:", db_list)
    print("Export folder:", out_dir)

    tables, views = list_tables_and_views()
    if not tables and not (include_views and views):
        print("No tables (or views) found. Are we pointing at the correct DB?")
        return

    if tables:
        print("Tables found:", tables)
    if include_views and views:
        print("Views found:", views)

    total_files = 0
    total_rows  = 0

    # Dump tables
    for t in tables:
        n = dump_table(t, out_dir)
        print(f'Wrote table {t} -> {t}.csv ({n} rows)')
        total_files += 1
        total_rows  += n

    # Optionally dump views too
    if include_views:
        for v in views:
            out_path = os.path.join(out_dir, f"{v}__view.csv")
            n = dump_query_to_csv(f'SELECT * FROM "{v}"', out_path)
            print(f'Wrote view  {v} -> {v}__view.csv ({n} rows)')
            total_files += 1
            total_rows  += n

    print(f"Done. Files: {total_files}, Total rows: {total_rows}")

if __name__ == "__main__":
    # Pass INCLUDE_VIEWS=1 to also export views
    include_views = os.environ.get("INCLUDE_VIEWS") == "1"
    main(include_views=include_views)
