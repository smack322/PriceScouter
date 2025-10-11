import os, sqlite3, time, datetime as dt
import pytest
from pathlib import Path

# --- ADAPT ME: wire to your repo functions if you have them already ----------
# Expected interface:
# - create_table_if_missing(conn)
# - insert_search(conn, query: str, ts: str|int|None = None) -> int (row id)
#
# If you don't have them yet, these helpers mimic your eventual repo.
def create_table_if_missing(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()

def insert_search(conn: sqlite3.Connection, query: str, ts: str | None = None) -> int:
    q = (query or "").strip()
    if not q:
        return 0
    if ts is None:
        ts = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    cur = conn.execute("INSERT INTO search_history (query, timestamp) VALUES (?, ?)", (q, ts))
    conn.commit()
    return int(cur.lastrowid)
# -----------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path: Path):
    db_path = tmp_path / "test_search_history.sqlite3"
    conn = sqlite3.connect(db_path.as_posix())
    try:
        yield conn
    finally:
        conn.close()

def _table_info(conn):
    return conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name='search_history'").fetchone()

def test_schema_exists_with_required_columns(tmp_db):
    create_table_if_missing(tmp_db)
    name, sql = _table_info(tmp_db)
    assert name == "search_history"
    # columns present (loose match to avoid driver diffs)
    for col in ("id", "query", "timestamp"):
        assert col in sql.lower()

def test_insert_and_readback(tmp_db):
    create_table_if_missing(tmp_db)
    before = time.time()
    rid = insert_search(tmp_db, "iphone 15 case")
    assert rid > 0
    row = tmp_db.execute("SELECT id, query, timestamp FROM search_history WHERE id=?", (rid,)).fetchone()
    assert row is not None
    _id, q, ts = row
    assert q == "iphone 15 case"
    # timestamp present & parseable
    parsed = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed.timestamp() >= before - 2  # within a small window

def test_auto_timestamp_when_missing(tmp_db):
    create_table_if_missing(tmp_db)
    rid = insert_search(tmp_db, "airpods pro", ts=None)
    (_id, _q, ts) = tmp_db.execute("SELECT id, query, timestamp FROM search_history WHERE id=?", (rid,)).fetchone()
    assert ts and isinstance(ts, str)

def test_multiple_inserts_order(tmp_db):
    create_table_if_missing(tmp_db)
    qlist = ["ipad case", "iphone case", "macbook stand"]
    ids = [insert_search(tmp_db, q) for q in qlist]
    rows = tmp_db.execute("SELECT query FROM search_history ORDER BY id ASC").fetchall()
    got = [r[0] for r in rows]
    assert got == qlist

def test_empty_query_not_persisted(tmp_db):
    create_table_if_missing(tmp_db)
    rid = insert_search(tmp_db, "   ")
    assert rid == 0
    count = tmp_db.execute("SELECT COUNT(*) FROM search_history").fetchone()[0]
    assert count == 0
