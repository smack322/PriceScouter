import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = "app.db"  # the one that has product_results + search_history
SQL_PATH = Path("backend/sql/canonical_product_view.sql")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 1) Apply the SQL file (it handles DROP + CREATE)
with SQL_PATH.open("r", encoding="utf-8") as f:
    cur.executescript(f.read())
conn.commit()

# 2) Debug: list tables + views
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cur.fetchall())

cur.execute("SELECT name FROM sqlite_master WHERE type='view';")
print("Views:", cur.fetchall())

# 3) Try querying the view
df = pd.read_sql_query(
    """
    SELECT *
    FROM canonical_product_view
    LIMIT 10;
    """,
    conn,
)

print(df.head())
conn.close()
