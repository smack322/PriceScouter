# tests/perf/test_latency.py
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

# Import your main agent entrypoint
import agents.app as app  # ensure app.run_agents exists


# -------- Helpers --------

REP_QUERIES = [
    "iphone 15 case",
    "usb c charger",
    "airpods pro",
    "macbook stand",
    "ipad mini cover",
    "gaming mouse",
    "mechanical keyboard",
    "4k monitor",
    "wireless earbuds",
    "portable ssd",
    "webcam 1080p",
    "hdmi 2.1 cable",
    "bluetooth speaker",
    "android tablet case",
    "noise cancelling headphones",
]

def _now_ms():
    return int(time.time() * 1000)

def _percentiles_ms(samples_ms):
    arr = np.array(samples_ms, dtype=float)
    return {
        "n": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }

def _save_report(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# -------- Tests --------

@pytest.mark.asyncio
def test_smoke_single_query_under_15s():
    """
    TC-LAT-001: Smoke guard that a single representative query is < 15s.
    """
    if not hasattr(app, "run_agents"):
        pytest.skip("agents.app.run_agents not exported")
    q = "iphone 15 case"
    t0 = time.perf_counter()
    # If your run_agents is async, await; if sync, call directly.
    coro = app.run_agents(q)
    if asyncio.iscoroutine(coro):
        asyncio.get_event_loop().run_until_complete(coro)
    t = (time.perf_counter() - t0)
    assert t < 15.0


@pytest.mark.performance
@pytest.mark.asyncio
async def test_p95_latency_under_15s(tmp_path):
    """
    TC-LAT-002: Compute percentiles over a small batch; assert P95 ≤ 15s.
    Produces a report artifact for docs.
    """
    if not hasattr(app, "run_agents"):
        pytest.skip("agents.app.run_agents not exported")

    # Warm-up
    await app.run_agents("usb c charger")

    lat_ms = []
    # Light concurrency to avoid overloading CI runners
    for q in REP_QUERIES:
        t0 = _now_ms()
        await app.run_agents(q)
        lat_ms.append(_now_ms() - t0)

    stats = _percentiles_ms(lat_ms)
    report = {
        "suite": "p95_e2e_latency",
        "stats": stats,
        "queries": REP_QUERIES,
        "timestamp_ms": _now_ms(),
    }
    _save_report(report, tmp_path / "reports" / "latency_p95.json")

    assert stats["p95"] <= 15000.0, f"P95 too high: {stats['p95']} ms"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_small_load_concurrent_10():
    """
    TC-LAT-003: Run 10 concurrent searches; ensure wall time stays bounded.
    """
    if not hasattr(app, "run_agents"):
        pytest.skip("agents.app.run_agents not exported")

    queries = REP_QUERIES[:10]
    t0 = time.perf_counter()
    await asyncio.gather(*(app.run_agents(q) for q in queries))
    elapsed = time.perf_counter() - t0
    # Allow a small overhead margin over 15s for aggregate wall time
    assert elapsed < 20.0


@pytest.mark.performance
@pytest.mark.asyncio
async def test_vendor_timeout_enforced_under_budget(monkeypatch):
    """
    TC-LAT-004: Simulate a slow vendor stage; ensure SLA still holds (<15s).
    """
    if not hasattr(app, "run_agents"):
        pytest.skip("agents.app.run_agents not exported")
    if not hasattr(app, "app"):
        pytest.skip("agents.app.app not exposed for patching")

    class SlowApp:
        async def ainvoke(self, state):
            # Simulate a blocking vendor path
            time.sleep(6)  # simulate slow I/O path; your code should guard this
            return {"results": []}

    monkeypatch.setattr("agents.app.app", SlowApp())

    t0 = time.perf_counter()
    await app.run_agents("slow vendor path")
    elapsed = time.perf_counter() - t0
    assert elapsed < 15.0


@pytest.mark.performance
@pytest.mark.asyncio
async def test_retry_backoff_within_sla(monkeypatch):
    """
    TC-LAT-005: Force two quick transient errors then success; ensure overall time < 15s.
    Patch the specific vendor call your pipeline uses (example: ebay token fetch).
    """
    # Example for ebay_tools._request_token; adjust to your pipeline’s hot path.
    try:
        import agents.ebay_tools as ebay
    except Exception:
        pytest.skip("ebay tools not present")

    calls = {"n": 0}

    def flaky_token(url):
        calls["n"] += 1
        if calls["n"] < 3:
            # Quick failure to exercise retry without long sleeps
            raise ebay.requests.ConnectionError("transient")
        return "ok_token"

    monkeypatch.setattr(ebay, "_request_token", flaky_token)

    t0 = time.perf_counter()
    # Choose a query that triggers eBay path; if not deterministic, just run_agents.
    await app.run_agents("iphone 15 case")
    elapsed = time.perf_counter() - t0
    assert elapsed < 15.0
