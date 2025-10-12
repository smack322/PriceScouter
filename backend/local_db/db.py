# db.py
from __future__ import annotations

import json
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, DateTime
)
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON  # uses TEXT under the hood
from sqlalchemy import event

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
# Table definition
# -----------------------------
search_history = Table(
    "search_history",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("query", String(512), nullable=False),
    Column("zip_code", String(32)),
    Column("country", String(8)),
    Column("agent", String(64), nullable=False),      # e.g., "google_shopping", "keepa", "ebay", "aggregate"
    Column("status", String(32), nullable=False),     # "success" | "error"
    Column("duration_ms", Integer),
    Column("results_count", Integer),
    Column("results_sample", SQLITE_JSON),            # small sample for quick debugging
    Column("full_payload", SQLITE_JSON),              # entire agent output if desired
    Column("created_at", DateTime, default=datetime.utcnow, nullable=False),
)

def init_db() -> None:
    metadata.create_all(engine)

# -----------------------------
# JSON sanitization helpers
# -----------------------------
def _to_builtin(o: Any):
    """Convert common non-JSON-native Python/NumPy/Pandas types to builtins."""
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

    # As a last resort, let json raise a TypeError so we notice new types
    raise TypeError(f"{type(o).__name__} is not JSON serializable")

def _jsonable(obj: Any):
    """Round-trip through json to ensure pure-Python types (lists/dicts/str/int/float/bool/None)."""
    if obj is None:
        return None
    return json.loads(json.dumps(obj, default=_to_builtin, ensure_ascii=False))

# -----------------------------
# Logger
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
    """Insert one row into search_history."""
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
