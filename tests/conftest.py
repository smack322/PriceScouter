import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import os
from pathlib import Path

import importlib.util
import sys
from pathlib import Path
import contextlib, textwrap, pandas as pd
import pytest
import json
import types
import shutil
import tempfile
import numpy as np
import re

from tests.db_fixtures import *
from sqlalchemy import create_engine, text


@pytest.fixture(autouse=True)
def _ci_env():
    # prevent network/LLM during tests
    os.environ.setdefault("DISABLE_LLM", "1")

@pytest.fixture
def parsed_default():
    return {"query": "test", "max_price": None, "zip_code": "19406", "country": "US"}

# ----- Fake Streamlit -----
class _FakeLinkColumn:
    def __init__(self, label=None, display_text=None, help=None):
        self.label = label
        self.display_text = display_text
        self.help = help

class _FakeColumnConfig:
    LinkColumn = _FakeLinkColumn

class FakeSt:
    def __init__(self):
        self.calls = {"dataframe": [], "link_button": [], "markdown": [], "caption": []}
        self.column_config = _FakeColumnConfig()

    def dataframe(self, *args, **kwargs):
        self.calls["dataframe"].append((args, kwargs))

    def link_button(self, label, url):
        self.calls["link_button"].append((label, url))

    def markdown(self, md):
        self.calls["markdown"].append(md)

    def caption(self, cap):
        self.calls["caption"].append(cap)

    @contextlib.contextmanager
    def expander(self, _title):
        yield

def _load_module_as(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find module at: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register under requested name (alias)
    spec.loader.exec_module(mod)
    return mod

@pytest.fixture()
def fake_st(monkeypatch):
    # Adjust this path if your repo layout differs
    repo_root = Path(__file__).resolve().parents[1]   # repo root (.. from tests/)
    app_path = repo_root / "frontend" / "chatbot_ui.py"

    # Load chatbot_ui.py but expose it to tests as module name "streamlit_ui"
    ui = _load_module_as("streamlit_ui", app_path)

    fake = FakeSt()
    # Replace the 'st' object used inside the UI with our fake
    monkeypatch.setattr(ui, "st", fake, raising=True)
    yield ui, fake

def _load_env_once():
    # Load .env if present (local dev). In CI we rely on injected env vars.
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)

_load_env_once()

os.environ.setdefault("DISABLE_LLM", "1")



class _FakeLLM:
    def invoke(self, _msgs):
        class _Resp:
            content = json.dumps({"query": "fallback", "vendor": None, "limit": None})
        return _Resp()

def test_extract_params_handles_bad_json(monkeypatch):
    import agents.app as app
    monkeypatch.setattr(app, "extract_llm", _FakeLLM())  # <- no network
    state = {"messages": [DummyMessage("fallback")]}
    result = app.extract_params(state)
    assert result["query"] == "fallback"
    assert result["vendor"] is None

    AMAZON_US = "ATVPDKIKX0DER"

def make_fake_api(products):
    """
    products: list[dict] shaped like Keepa products.
    """
    class FakeAPI:
        def __init__(self, prods):
            self._prods = prods

        def product_finder(self, criteria, domain="US"):
            # We don't need to parse criteria for tests; return the ASINs we have.
            return [p.get("asin") for p in self._prods]

        def query(self, asins, domain="US", **kwargs):
            # Return only the requested ASINs in the same order.
            keep = {p.get("asin"): p for p in self._prods}
            return [keep[a] for a in asins if a in keep]

    return FakeAPI(products)

# ---------- Deterministic Dummy Embedder ----------
class DummyEmbedder:
    def __init__(self, dim=384):
        self.model = "dummy"
        self.dim = dim
    def embed(self, texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=self.dim).astype("float32")
            v /= np.linalg.norm(v) + 1e-9
            out.append(v.tolist())
        return out

@pytest.fixture
def dummy_embedder_384():
    return DummyEmbedder(dim=384)

@pytest.fixture
def dummy_embedder_1536():
    return DummyEmbedder(dim=1536)

# ---------- FAISS-backed VectorStore with temp index ----------
@pytest.fixture
def faiss_store_dummy(tmp_path, monkeypatch, dummy_embedder_384):
    # Force FAISS backend, no network
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")
    monkeypatch.setenv("EMBED_PROVIDER", "sbert")
    monkeypatch.setenv("PGVECTOR_DIM", "384")
    monkeypatch.setenv("FAISS_INDEX_PATH", str(tmp_path / "index.bin"))
    monkeypatch.setenv("FAISS_IDS_PATH", str(tmp_path / "ids.json"))

    from vector_utils import VectorStore
    vs = VectorStore(embedder=dummy_embedder_384)
    return vs

def upsert_titles(vstore, items):
    """items: list of (product_id, title) to embed+upsert."""
    vecs = vstore.embedder.embed([t for _, t in items])
    vstore.upsert_embeddings(list(zip([i for i, _ in items], vecs)))

def seed_n(vstore, n=1000, prefix="item"):
    rows = [(f"pid-{i}", f"{prefix} {i}") for i in range(n)]
    upsert_titles(vstore, rows)

def timed_ms(fn):
    import time
    t0 = time.perf_counter()
    _ = fn()
    return (time.perf_counter() - t0) * 1000.0

# ---------- pgvector integration helpers ----------
def pg_available():
    try:
        import psycopg  # noqa: F401
        return True
    except Exception:
        return False

@pytest.fixture
def pg_env(monkeypatch):
    """Skip if Postgres env not configured."""
    if not pg_available():
        pytest.skip("psycopg not installed")
    dsn = os.getenv("SUPABASE_DB_URL") or os.getenv("PG_DSN")
    if not dsn:
        pytest.skip("No Postgres DSN in SUPABASE_DB_URL / PG_DSN")
    # Force supabase backend
    monkeypatch.setenv("VECTOR_BACKEND", "supabase")
    monkeypatch.setenv("PGVECTOR_TABLE", os.getenv("PGVECTOR_TABLE", "product_embeddings"))
    return dsn

@pytest.fixture
def mini_product_results(tmp_path):
    """Small product_results.csv with guaranteed duplicates via product_id."""
    p = tmp_path / "product_results.csv"
    # 4 rows: 2 duplicates for pid=A, 2 duplicates for pid=B
    rows = [
        dict(id=1, search_id=9, source="unknown",
             title="Otterbox Defender iPhone 15", link="",
             seller="Best Buy", price=39.09, shipping="", total=39.09, currency="USD",
             rating=4.7, reviews_count=632, extra="{}",
             raw=json.dumps({"title":"Otterbox Defender iPhone 15","price":39.09,"total_cost":39.09,
                             "seller":"Best Buy","product_id":"A","brand_guess":"OtterBox"})),
        dict(id=2, search_id=9, source="unknown",
             title="Otterbox Defender for iPhone 15", link="",
             seller="Target", price=38.99, shipping="", total=38.99, currency="USD",
             rating=4.5, reviews_count=100, extra="{}",
             raw=json.dumps({"title":"Otterbox Defender for iPhone 15","price":38.99,"total_cost":38.99,
                             "seller":"Target","product_id":"A","brand_guess":"OtterBox"})),
        dict(id=3, search_id=9, source="unknown",
             title="Apple MagSafe Case iPhone 15", link="",
             seller="Apple", price=49.00, shipping="", total=49.00, currency="USD",
             rating=4.4, reviews_count=295, extra="{}",
             raw=json.dumps({"title":"Apple MagSafe Case iPhone 15","price":49.0,"total_cost":49.0,
                             "seller":"Apple","product_id":"B","brand_guess":"Apple"})),
        dict(id=4, search_id=9, source="unknown",
             title="iPhone 15 MagSafe Case by Apple", link="",
             seller="Apple", price=49.00, shipping="", total=49.00, currency="USD",
             rating=4.4, reviews_count=296, extra="{}",
             raw=json.dumps({"title":"iPhone 15 MagSafe Case by Apple","price":49.0,"total_cost":49.0,
                             "seller":"Apple","product_id":"B","brand_guess":"Apple"})),
    ]
    pd.DataFrame(rows).to_csv(p, index=False)
    return p

@pytest.fixture
def mini_search_history(tmp_path):
    """Search history isn’t required by the builder, but we supply a stub path for consistency."""
    p = tmp_path / "search_history.csv"
    p.write_text("id,query\n1,iphone 15 case\n")
    return p

SAMPLE_ROWS = [
    # Best Buy Otterbox (39.09)
    dict(id=1, search_id=9, source="unknown",
         title="Otterbox Defender Pro Case for Apple iPhone 15 / iPhone 14 / iPhone 13",
         link=None, seller="Best Buy", price=39.09, shipping=None, total=39.09,
         currency=None, rating=4.7, reviews_count=632, extra="{}",
         raw=json.dumps({"title":"Otterbox Defender Pro Case for Apple iPhone 15 / iPhone 14 / iPhone 13",
                         "price":39.09,"price_str":"$39.09","seller":"Best Buy",
                         "product_link":"https://www.google.com/search?ibp=oshop&q=iphone 15 case&prds=catalogid:7920095867532335495",
                         "rating":4.7,"reviews_count":632,"product_id":"7920095867532335495",
                         "total_cost":39.09,"brand_guess":"Otterbox"})),
    # Apple MagSafe (49)
    dict(id=2, search_id=9, source="unknown",
         title="Apple MagSafe Case for iPhone 15",
         link=None, seller="Apple", price=49.0, shipping=None, total=49.0,
         currency=None, rating=4.4, reviews_count=295, extra="{}",
         raw=json.dumps({"title":"Apple MagSafe Case for iPhone 15",
                         "price":49.0,"price_str":"$49.00","seller":"Apple",
                         "product_link":"https://www.google.com/search?ibp=oshop&q=iphone 15 case&prds=catalogid:3917673451152378843",
                         "rating":4.4,"reviews_count":295,"product_id":"3917673451152378843",
                         "total_cost":49.0,"brand_guess":"Apple"})),
    # Best Buy Speck (29.99)
    dict(id=3, search_id=9, source="unknown",
         title="Speck Presidio Perfect Clear MagSafe Case for Apple iPhone 15 / iPhone 14 / iPhone 13",
         link=None, seller="Best Buy", price=29.99, shipping=None, total=29.99,
         currency=None, rating=4.6, reviews_count=153, extra="{}",
         raw=json.dumps({"title":"Speck Presidio Perfect Clear MagSafe Case for Apple iPhone 15 / iPhone 14 / iPhone 13",
                         "price":29.99,"product_link":"https://www.google.com/search?ibp=oshop&q=iphone 15 case&prds=catalogid:792030822556606110",
                         "rating":4.6,"reviews_count":153,"product_id":"792030822556606110",
                         "total_cost":29.99,"brand_guess":"Speck"})),
    # Walmart Otterbox (39) – same product_id as new one to force canonical collapse
    dict(id=4, search_id=9, source="unknown",
         title="OtterBox Defender Series Pro MagSafe Case for Apple iPhone 15",
         link=None, seller="Walmart", price=39.0, shipping=None, total=39.0,
         currency=None, rating=None, reviews_count=None, extra="{}",
         raw=json.dumps({"title":"OtterBox Defender Series Pro MagSafe Case for Apple iPhone 15",
                         "price":39.0,"product_link":"https://www.google.com/search?ibp=oshop&q=iphone 15 case&prds=catalogid:10259123632034479621",
                         "product_id":"10259123632034479621",
                         "total_cost":39.0,"brand_guess":"OtterBox"})),
]

CREATE_TABLES = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS product_results (
  id INTEGER PRIMARY KEY,
  search_id INTEGER,
  source TEXT,
  title TEXT,
  link TEXT,
  seller TEXT,
  price REAL,
  shipping TEXT,
  total REAL,
  currency TEXT,
  rating REAL,
  reviews_count INTEGER,
  extra TEXT,
  raw TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS search_history (
  id INTEGER PRIMARY KEY,
  query TEXT,
  zip_code TEXT,
  country TEXT,
  agent TEXT,
  status TEXT,
  duration_ms INTEGER,
  results_count INTEGER,
  results_sample TEXT,
  full_payload TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

DROP_VIEW = "DROP VIEW IF EXISTS canonical_product_view;"
CREATE_VIEW = """
CREATE VIEW canonical_product_view AS
WITH base AS (
  SELECT
    id AS row_id,
    COALESCE(
      json_extract(raw, '$.product_id'),
      lower(trim(replace(replace(title, '®',''), '™','')))
    ) AS canonical_key,
    title, seller, source, link,
    COALESCE(link, json_extract(raw, '$.product_link')) AS product_url,
    CAST(total AS REAL) AS total_price,
    CAST(price AS REAL) AS unit_price,
    currency,
    created_at
  FROM product_results
)
SELECT
  row_number() OVER (ORDER BY (SELECT NULL)) AS canonical_id,
  canonical_key,
  (SELECT b2.title FROM base b2
   WHERE b2.canonical_key = b.canonical_key
   ORDER BY COALESCE(b2.total_price, b2.unit_price) ASC, b2.created_at ASC
   LIMIT 1) AS title,
  MIN(COALESCE(total_price, unit_price)) AS min_price,
  AVG(COALESCE(total_price, unit_price)) AS avg_price,
  MAX(COALESCE(total_price, unit_price)) AS max_price,
  COUNT(DISTINCT seller) AS seller_count,
  COUNT(*) AS total_listings,
  (SELECT b2.product_url FROM base b2
   WHERE b2.canonical_key = b.canonical_key
   ORDER BY COALESCE(b2.total_price, b2.unit_price) ASC, b2.created_at ASC
   LIMIT 1) AS representative_url
FROM base b
GROUP BY canonical_key;
"""

def _run_many(conn, sql_block: str):
    parts = [p.strip() for p in re.split(r';\s*(?:\n|$)', sql_block) if p.strip()]
    for stmt in parts:
        conn.exec_driver_sql(stmt)

@pytest.fixture(scope="session")
def engine():
    eng = create_engine("sqlite+pysqlite:///:memory:", future=True)
    with eng.begin() as conn:
        # one statement at a time
        _run_many(conn, CREATE_TABLES)
        # seed
        for r in SAMPLE_ROWS:
            cols = ",".join(r.keys())
            vals = ":" + ",:".join(r.keys())
            conn.execute(text(f"INSERT INTO product_results ({cols}) VALUES ({vals})"), r)
        conn.execute(text(
            "INSERT INTO search_history (id, query, agent, status, results_count) "
            "VALUES (1,'iphone 15 case','serp','success',40)"
        ))
        # drop then create view (two separate calls)
        conn.exec_driver_sql(DROP_VIEW)
        conn.exec_driver_sql(CREATE_VIEW)
    return eng

@pytest.fixture()
def conn(engine):
    with engine.begin() as c:
        yield c
