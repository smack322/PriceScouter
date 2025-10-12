# db.py
from __future__ import annotations

import json
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, DateTime, Float, UniqueConstraint, ForeignKey, event
)
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON

import numpy as np

# -----------------------------
# Engine / PRAGMAs (SQLite)
# -----------------------------
engine = create_engine(
    "sqlite:///./app.db",
    future=True,
    echo=False,
    connect_args={"check_same_thread": False},  # Streamlit + threads
)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")   # better concurrency
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()

metadata = MetaData()

# -----------------------------
# Tables
# -----------------------------
search_history = Table(
    "search_history",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("query", String(512), nullable=False),
    Column("zip_code", String(32)),
    Column("country", String(8)),
    Column("agent", String(64), nullable=False),      # "keepa" | "serp" | "ebay" | "aggregate"
    Column("status", String(32), nullable=False),     # "success" | "error"
    Column("duration_ms", Integer),
    Column("results_count", Integer),
    Column("results_sample", SQLITE_JSON),            # small sample for quick debugging
    Column("full_payload", SQLITE_JSON),              # entire agent output if desired
    Column("created_at", DateTime, default=datetime.utcnow, nullable=False),
)

product_results = Table(
    "product_results",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("search_id", Integer, ForeignKey("search_history.id", ondelete="CASCADE"), nullable=False),
    Column("source", String(32), nullable=False),     # "keepa" | "serp" | "ebay" | etc.
    Column("title", String(1024)),
    Column("link", String(2048)),
    Column("seller", String(256)),
    Column("price", Float),
    Column("shipping", Float),
    Column("total", Float),
    Column("currency", String(8)),
    Column("rating", Float),
    Column("reviews_count", Integer),
    Column("extra", SQLITE_JSON),
    Column("raw", SQLITE_JSON),
    Column("created_at", DateTime, default=datetime.utcnow, nullable=False),
    UniqueConstraint("search_id", "link", "source", name="uq_product_per_search"),
)

def init_db() -> None:
    metadata.create_all(engine)

# -----------------------------
# JSON sanitization helpers
# -----------------------------
def _to_builtin(o: Any):
    # numpy / pandas scalars & arrays
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    # datetimes / decimals
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    # pydantic models
    if hasattr(o, "model_dump"):
        try:
            return o.model_dump()
        except Exception:
            pass
    # generic objects with __dict__
    if hasattr(o, "__dict__"):
        try:
            return {k: _to_builtin(v) for k, v in vars(o).items()}
        except Exception:
            pass
    # last resort: let json raise so we learn about new types
    raise TypeError(f"{type(o).__name__} is not JSON serializable")

def _jsonable(obj: Any):
    if obj is None:
        return None
    return json.loads(json.dumps(obj, default=_to_builtin, ensure_ascii=False))

# -----------------------------
# Search event logger
# -----------------------------
def log_search_event(
    *,
    query: str,
    agent: str,
    status: str = "success",
    zip_code: Optional[str] = None,
    country: Optional[str] = None,
    duration_ms: Optional[int] = None,
    results: Optional[List[Dict[str, Any]]] = None,
    full_payload: Optional[Dict[str, Any] | List[Dict[str, Any]]] = None,
    sample_n: int = 3,
) -> int:
    """Insert one row into search_history and return its ID."""
    results = results or []
    row = {
        "query": query,
        "zip_code": zip_code,
        "country": country,
        "agent": agent,
        "status": status,
        "duration_ms": int(duration_ms) if duration_ms is not None else None,
        "results_count": int(len(results)),
        "results_sample": _jsonable(results[:sample_n]),
        "full_payload": _jsonable(full_payload if full_payload is not None else results),
        "created_at": datetime.utcnow(),
    }
    with engine.begin() as conn:
        res = conn.execute(search_history.insert().values(**row))
        return int(res.inserted_primary_key[0])

# -----------------------------
# Normalization helpers
# -----------------------------
def _coerce_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            s = x.strip().replace(",", "")
            if s and not s[0].isdigit() and s[0] not in "+-":
                s = s[1:]
            return float(s)
        return float(x)
    except Exception:
        return None

def _first(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default

def _norm_source_tag(row: Dict[str, Any]) -> str:
    if "_source" in row:
        return str(row["_source"])
    for k in ("keepa", "serp", "ebay"):
        if k in (row.get("source") or "").lower():
            return k
    return "unknown"

def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    source = _norm_source_tag(row)
    title = _first(row.get("title"), row.get("name"), row.get("product_title"))
    link  = _first(row.get("link"), row.get("url"), row.get("product_url"))

    price = _first(
        _coerce_float(row.get("total")),            # if agent mislabels "total" as price
        _coerce_float(row.get("price")),
        _coerce_float(row.get("current_price")),
        _coerce_float(row.get("buybox_price")),
        _coerce_float(row.get("regular_price")),
        default=None,
    )
    shipping = _coerce_float(row.get("shipping"))

    total_from_field = _coerce_float(row.get("total"))
    total = _first(total_from_field, None)
    if total is None and (price is not None or shipping is not None):
        total = (price or 0.0) + (shipping or 0.0)

    currency = _first(row.get("currency"), row.get("iso_currency_code"))
    rating = _coerce_float(_first(row.get("rating"), row.get("stars")))

    reviews_count = None
    for key in ("reviews_count", "review_count", "reviews"):
        v = row.get(key)
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()):
            reviews_count = int(float(v))
            break

    seller = _first(row.get("seller"), row.get("source"), row.get("store"))

    extra: Dict[str, Any] = {}
    for k in ("sales_rank", "asin", "sku", "brand", "condition", "shipping_from",
              "shipping_to", "country", "zip_code"):
        if row.get(k) is not None:
            extra[k] = row.get(k)

    return {
        "source": source,
        "title": title,
        "link": link,
        "seller": seller,
        "price": price,
        "shipping": shipping,
        "total": total,
        "currency": currency,
        "rating": rating,
        "reviews_count": reviews_count,
        "extra": _jsonable(extra),
        "raw": _jsonable(row),
    }

# -----------------------------
# Bulk save normalized products
# -----------------------------
def save_product_results(search_id: int, rows: List[Dict[str, Any]]) -> int:
    """
    Bulk-insert normalized product rows for a given search_id.
    Returns the number of rows inserted (skips duplicates per unique constraint).
    """
    if not rows:
        return 0

    normalized = [_normalize_row(r) for r in rows]
    for r in normalized:
        r["search_id"] = search_id
        r["created_at"] = datetime.utcnow()

    inserted = 0
    with engine.begin() as conn:
        for r in normalized:
            try:
                conn.execute(product_results.insert().values(**r))
                inserted += 1
            except Exception:
                # Likely unique constraint: (search_id, link, source)
                pass
    return inserted
