# app.py — Option 2: single fanout key with merge reducer + correct respond()
from __future__ import annotations

import asyncio
import json
import os
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv

# Load env BEFORE importing tools
load_dotenv(override=False)
_ = os.environ.get("OPENAI_API_KEY")

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

# from agent_keepa import keepa_search
# from agent_serp import google_shopping
# from agent_ebay import ebay_search

from agents.agent_amazon import amazon_products
from agents.agent_keepa import keepa_search
from agents.agent_serp import google_shopping
from agents.agent_ebay import ebay_search

# --- add this helper ---
from decimal import Decimal
from datetime import date, datetime
import numpy as np

import time
import uuid
from backend.local_db.db import init_db, log_search_event, save_product_results

import os
import json

def _llm_enabled() -> bool:
    # Read at call-time (not import-time) so CI env is honored
    return os.getenv("DISABLE_LLM", "").lower() not in {"1", "true", "yes"}

# Lazy holder; only build when enabled
_extract_llm = None

def _get_extract_llm():
    global _extract_llm
    if _extract_llm is None:
        # Only construct if enabled
        if _llm_enabled():
            from langchain_openai import ChatOpenAI
            _extract_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        else:
            class _FakeLLM:
                def invoke(self, _msgs):
                    class _Resp:
                        content = json.dumps({"query": "fallback", "vendor": None, "limit": None})
                    return _Resp()
            _extract_llm = _FakeLLM()
    return _extract_llm

def extract_params(state):
    """
    Your existing docstring…
    """
    # FAST PATH when LLM is disabled: avoid any network calls
    if not _llm_enabled():
        # emulate your “bad JSON -> fallback” path deterministically
        msgs = state.get("messages") or []
        text = getattr(msgs[-1], "content", None) if msgs else None
        text = text or ""
        return {"query": text, "vendor": None, "limit": None}

    # LLM path
    extract_llm = _get_extract_llm()
    msg = extract_llm.invoke([
        {"role": "system", "content": "Extract fields as JSON only."},
        {"role": "user", "content": f"{state['messages'][-1].content}\n\nReturn only JSON."}
    ])
    return json.loads(msg.content)

init_db()

def _to_builtin(o):
    """Coerce common non-JSON-native types to JSON-serializable builtins."""
    # numpy / pandas scalars & arrays
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    # sets / generators
    if isinstance(o, set):
        return list(o)
    # datetimes / decimals
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    # pydantic models
    if hasattr(o, "model_dump"):
        return o.model_dump()
    # objects with __dict__
    if hasattr(o, "__dict__"):
        return {k: _to_builtin(v) for k, v in vars(o).items()}
    # give up → let json raise
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

# ---------------- Reducer for concurrent writes to 'fanout' ---------------- #
def merge_dicts(
    a: Dict[str, List[dict]] | None,
    b: Dict[str, List[dict]] | None
) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    if a:
        out.update(a)
    if b:
        out.update(b)
    return out


# --------------------------------- State ---------------------------------- #
class Parsed(TypedDict):
    query: str
    max_price: Optional[float]
    zip_code: Optional[str]
    country: Optional[str]

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    parsed: Parsed
    # Single key for parallel writes; the reducer merges them
    fanout: Annotated[Dict[str, List[dict]], merge_dicts]
    results: List[dict]


# ------------------------ 1) Extract parameters ---------------------------- #
extract_llm = init_chat_model("openai:gpt-4.1-mini")

def extract_params(state: State):
    sys = (
        "Extract shopping params as JSON with keys: "
        "query (string, required), max_price (float | null), zip_code (string | null), country (string | null). "
        "Be conservative; if unsure, use null."
    )
    last = state["messages"][-1].content if state["messages"] else ""
    msg = extract_llm.invoke([
        ("system", sys),
        ("user", f"User request:\n{last}\n\nReturn only JSON.")
    ])

    try:
        parsed = json.loads(msg.content)
    except Exception:
        parsed = {"query": last, "max_price": None, "zip_code": None, "country": None}

    parsed.setdefault("query", last)
    parsed.setdefault("max_price", None)
    parsed.setdefault("zip_code", "19406")
    parsed.setdefault("country", "US")

    # Coerce max_price to float if string
    try:
        if parsed["max_price"] is not None:
            parsed["max_price"] = float(parsed["max_price"])
    except Exception:
        parsed["max_price"] = None

    return {"parsed": parsed}


# --------------------- 2) Fan-out: run tools in parallel ------------------- #
async def _safe_ainvoke(tool, kwargs):
    """
    Call LangChain tools/agents whether they are async, sync, or StructuredTool.
    """
    try:
        # Runnable-style async
        if hasattr(tool, "ainvoke") and callable(getattr(tool, "ainvoke")):
            return await tool.ainvoke(kwargs)

        # Runnable-style sync (StructuredTool usually implements this)
        if hasattr(tool, "invoke") and callable(getattr(tool, "invoke")):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: tool.invoke(kwargs))

        # Async callable
        if inspect.iscoroutinefunction(tool):
            return await tool(kwargs)

        # Plain callable
        if callable(tool):
            res = tool(kwargs)
            if inspect.iscoroutine(res):
                return await res
            return res

        return [{"_error": f"Unsupported tool type: {type(tool).__name__}"}]
    except Exception as e:
        name = getattr(tool, "name", getattr(tool, "__name__", "tool"))
        return [{"_error": f"{name}: {e}"}]
# --- add this new node (and delete the old call_keepa function) ---
async def call_amazon(state: State):
    p = state["parsed"]
    t0 = time.time()
    rows = await _safe_ainvoke(
        amazon_products,
        {
            "q": p["query"],
            "page": 1,
            "max_pages": 2,                 # tweak as you like
            "amazon_domain": "amazon.com",
            "gl": (p.get("country") or "US").lower(),  # 'us', 'uk', 'de', …
        },
    )
    dt = int((time.time() - t0) * 1000)
    status = "error" if any("_error" in r for r in rows) else "success"
    search_id = log_search_event(
        agent="amazon",                    # <- label for analytics
        query=p["query"],
        zip_code=p.get("zip_code"),
        country=p.get("country"),
        status=status,
        duration_ms=dt,
        results=rows
    )
    save_product_results(search_id, rows)
    return {"fanout": {"amazon": rows}}

async def call_keepa(state: State):
    p = state["parsed"]
    t0 = time.time()
    rows = await _safe_ainvoke(
        keepa_search,
        {"keyword": p["query"], "domain": p.get("country", "US"), "max_results": 10},
    )
    # status if any row carries an error
    dt = int((time.time() - t0) * 1000)
    status = "error" if any("_error" in r for r in rows) else "success"
    search_id = log_search_event(
        agent="keepa",
        query=p["query"],
        zip_code=p.get("zip_code"),
        country=p.get("country"),
        status=status,
        duration_ms=dt,
        results=rows
    )
    save_product_results(search_id, rows)
    return {"fanout": {"keepa": rows}}

# async def call_serp(state: State):
#     p = state["parsed"]
#     rows = await _safe_ainvoke(
#         google_shopping,
#         {"keyword": p["query"], "max_results": 10, "country": p.get("country", "US")},
#     )
#     return {"fanout": {"serp": rows}}

async def call_serp(state: State):
    p = state["parsed"]
    t0 = time.time()
    rows = await _safe_ainvoke(
        google_shopping,
        {
            "q": p["query"],
            "num": 10,
            "location": p.get("zip_code") or p.get("country", "US"),  # or more relevant location string
        },
        
    )
    dt = int((time.time() - t0) * 1000)
    status = "error" if any("_error" in r for r in rows) else "success"
    search_id = log_search_event(
        agent="serp",
        query=p["query"],
        zip_code=p.get("zip_code"),
        country=p.get("country"),
        status=status,
        duration_ms=dt,
        results=rows
    )
    save_product_results(search_id, rows)
    return {"fanout": {"serp": rows}}

async def call_ebay(state: State):
    p = state["parsed"]
    t0 = time.time()
    rows = await _safe_ainvoke(
        ebay_search,
        {
            "keyword": p["query"],
            "zip_code": p.get("zip_code") or "19406",
            "country": p.get("country", "US"),
            "limit": 50,
            "max_results": 10,
            "fixed_price_only": False,
        },
    )
    dt = int((time.time() - t0) * 1000)
    status = "error" if any("_error" in r for r in rows) else "success"
    search_id = log_search_event(
        agent="ebay",
        query=p["query"],
        zip_code=p.get("zip_code"),
        country=p.get("country"),
        status=status,
        duration_ms=dt,
        results=rows
    )
    save_product_results(search_id, rows)
    return {"fanout": {"ebay": rows}}


# -------------------------- 3) Aggregate results --------------------------- #
def aggregate(state: State):
    p = state["parsed"]
    max_price = p.get("max_price")

    fan = state.get("fanout") or {}
    # keepa_rows = fan.get("keepa", []) or []
    amazon_rows = fan.get("amazon", []) or []
    serp_rows  = fan.get("serp",  []) or []
    ebay_rows  = fan.get("ebay",  []) or []

    all_rows: List[dict] = []
    for src, rows in (("amazon", amazon_rows), ("serp", serp_rows), ("ebay", ebay_rows)):
        for r in rows:
            r["_source"] = src
        all_rows.extend(rows)

    def total_cost(r: dict) -> float:
        # eBay: already has 'total'
        if isinstance(r.get("total"), (int, float)):
            return float(r["total"])
        # Keepa/Serp: price (+ shipping if provided)
        price = None
        for k in ("buybox_price", "price", "current_price"):
            if r.get(k) is not None:
                try:
                    price = float(r[k])
                    break
                except Exception:
                    pass
        shipping = 0.0
        if r.get("shipping") is not None:
            try:
                shipping = float(r["shipping"])
            except Exception:
                pass
        return (price or 0.0) + shipping

    if isinstance(max_price, (int, float)):
        all_rows = [r for r in all_rows if total_cost(r) <= float(max_price)]

    all_rows.sort(key=total_cost)
    # log the merged/filtered set and store them too
    search_id = log_search_event(
        agent="aggregate",
        query=p["query"],
        zip_code=p.get("zip_code"),
        country=p.get("country"),
        status="success",
        results=all_rows[:50],      # sample
        full_payload=all_rows,      # complete merged list
    )
    save_product_results(search_id, all_rows)
    return {"results": all_rows}


# -------------------------- 4) Final response ------------------------------ #
final_llm = init_chat_model("openai:gpt-4.1")

def respond(state: State):
    p = state["parsed"]
    rows = state.get("results", [])[:15]

    sys = "You are a concise shopping assistant that merges multi-source listings."

    # ✅ serialize with a default that converts numpy/pandas/etc. to builtins
    rows_json = json.dumps(rows, ensure_ascii=False, default=_to_builtin)

    user = (
        f"Query: {p['query']}\n"
        f"Max price: {p.get('max_price')}\n"
        f"ZIP: {p.get('zip_code')} Country: {p.get('country')}\n\n"
        "Here are aggregated results in JSON. Summarize a short ranked list with: "
        "estimated total (or price), shipping if known, source (keepa/serp/ebay), title, and a direct link.\n\n"
        f"{rows_json}"
    )

    msg = final_llm.invoke([("system", sys), ("user", user)])
    return {"messages": [msg]}


SESSION_ID = uuid.uuid4().hex[:12]
# --------------------------- Build the graph ------------------------------- #
graph = StateGraph(State)

graph.add_node("extract_params", extract_params)
graph.add_node("amazon", call_amazon)
graph.add_node("serp", call_serp)
graph.add_node("ebay", call_ebay)
graph.add_node("aggregate", aggregate)
graph.add_node("respond", respond)

graph.add_edge(START, "extract_params")
graph.add_edge("extract_params", "amazon")
graph.add_edge("extract_params", "serp")
graph.add_edge("extract_params", "ebay")
graph.add_edge("amazon", "aggregate")
graph.add_edge("serp", "aggregate")
graph.add_edge("ebay", "aggregate")
graph.add_edge("aggregate", "respond")
graph.add_edge("respond", END)


app = graph.compile()


# --------------------------------- Main ------------------------------------ #
if __name__ == "__main__":
    async def main():
        user_msg = {
            "role": "user",
            "content": (
                "Find iPhone 15 cases under $20. Show buybox price, regular price, "
                "shipping cost, sales rank and direct link to the product. Ship to 19406, US."
            ),
        }
        result = await app.ainvoke({"messages": [user_msg]})
        print("\n=== FINAL ANSWER ===")
        print(result["messages"][-1].content)

    asyncio.run(main())
