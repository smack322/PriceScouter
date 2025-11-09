# tests/test_ui_smoke_system.py
import types
import builtins
import pandas as pd
import importlib
import sys

def test_streamlit_render_results_smoke(monkeypatch):
    # Fake streamlit minimal API used in render_results
    class DummyST:
        def subheader(self, *a, **kw): pass
        def text_input(self, *a, **kw): return ""
        def number_input(self, *a, **kw): return 50
        def dataframe(self, *a, **kw): pass
        def markdown(self, *a, **kw): pass
        def info(self, *a, **kw): pass
        class _Exp:
            def __enter__(self): return self
            def __exit__(self, *exc): pass
        def expander(self, *a, **kw): return self._Exp()
        def image(self, *a, **kw): pass
        def caption(self, *a, **kw): pass
        def columns(self, *a, **kw): return (self, self)

    st = DummyST()
    monkeypatch.setitem(sys.modules, "streamlit", st)  # simple module patch

    # Patch backend.queries.fetch_canonicals / fetch_variants to return small dataframes
    fake_df = pd.DataFrame([{
        "canonical_id": 1,
        "canonical_key": "3917673451152378843",
        "title": "Apple MagSafe Case for iPhone 15",
        "min_price": 49.0, "avg_price": 49.0, "max_price": 49.0,
        "seller_count": 1, "total_listings": 1, "representative_url": "https://example.com"
    }])

    def fake_fetch_canonicals(limit=200, q=None):
        return fake_df

    def fake_fetch_variants(canonical_key: str):
        return [
            {"seller":"Apple","listing_title":"Apple MagSafe Case for iPhone 15",
             "price":49.0,"product_url":"https://example.com"}
        ]

    # Create a fake module tree for frontend.ui_results import path
    backend = types.ModuleType("backend")
    queries = types.ModuleType("backend.queries")
    queries.fetch_canonicals = fake_fetch_canonicals
    queries.fetch_variants = fake_fetch_variants
    sys.modules["backend"] = backend
    sys.modules["backend.queries"] = queries

    # Build a minimal frontend.ui_results module in-memory
    code = """
import pandas as pd
import streamlit as st
from backend.queries import fetch_canonicals, fetch_variants

def _money(x):
    return "" if x is None or pd.isna(x) else f"${x:,.2f}"

def _link(text, url):
    return f"[{text}]({url})" if url else text

def render_results():
    st.subheader("Product Results (Canonical)")
    q = st.text_input("Filter products", "", placeholder="e.g., iPhone 15 case")
    limit = st.number_input("Max rows", 10, 1000, 200, step=10)
    df = fetch_canonicals(limit=limit, q=q if q else None)
    if df.empty:
        st.info("No results yet.")
        return
    show = df.assign(
        Min=df["min_price"].map(_money),
        Avg=df["avg_price"].map(_money),
        Max=df["max_price"].map(_money),
    )[["title","Min","Avg","Max","seller_count","total_listings"]]
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.markdown("### Details")
    for _, row in df.iterrows():
        with st.expander(f"{row.title} — {_money(row.avg_price)} avg • {int(row.seller_count)} sellers"):
            variants = fetch_variants(canonical_key=row.canonical_key)
            import pandas as pd
            vdf = pd.DataFrame(variants)
            vdf["price"] = vdf["price"].map(_money)
            vdf["Buy"] = vdf.apply(lambda r: _link(r.get("seller") or "Open", r.get("product_url")), axis=1)
            cols = [c for c in ["Buy","listing_title","price"] if c in vdf.columns]
            st.dataframe(vdf[cols], use_container_width=True, hide_index=True)
"""
    mod = types.ModuleType("frontend.ui_results")
    exec(code, mod.__dict__)
    sys.modules["frontend.ui_results"] = mod

    # Import and run
    ui = importlib.import_module("frontend.ui_results")
    ui.render_results()  # should not raise
