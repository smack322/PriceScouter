# tests/test_vector_pg_integration.py
import os
import pytest

pytestmark = pytest.mark.integration

def _ensure_table(conn, table, dim):
    with conn.cursor() as cur:
        cur.execute("create extension if not exists vector;")
        cur.execute(f"""
            create table if not exists {table} (
              product_id text primary key,
              embedding vector({dim}),
              model text not null,
              dim int not null,
              updated_at timestamptz not null default now()
            );
        """)
        cur.execute(f"""
            create index if not exists idx_{table}_ivf
            on {table} using ivfflat (embedding vector_cosine_ops) with (lists=100);
        """)
    conn.commit()

def _count(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"select count(*) from {table};")
        return cur.fetchone()[0]

@pytest.mark.usefixtures("pg_env")
def test_pgvector_round_trip(pg_env, monkeypatch):
    import psycopg
    table = os.getenv("PGVECTOR_TABLE", "product_embeddings")
    dim = int(os.getenv("PGVECTOR_DIM", "384"))

    # Force supabase backend in this test
    monkeypatch.setenv("VECTOR_BACKEND", "supabase")

    # Connect & ensure table
    conn = psycopg.connect(pg_env)
    _ensure_table(conn, table, dim)

    # Instantiate store with dummy embedder
    from conftest import DummyEmbedder
    from vector_utils import VectorStore
    vs = VectorStore(embedder=DummyEmbedder(dim=dim))
    # upsert a couple rows
    ids = ["p1", "p2", "p3"]
    vecs = vs.embedder.embed(["iphone 13", "iphone 13 128gb", "garden hose"])
    vs.upsert_embeddings(list(zip(ids, vecs)))

    assert _count(conn, table) >= 3

    # search
    hits = vs.search("iphone 13", k=2)
    got = [h[0] for h in hits]
    assert set(got) & {"p1", "p2"}
