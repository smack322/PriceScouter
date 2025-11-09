import os, runpy, csv, pandas as pd
from pathlib import Path

# Resolve repo root once
REPO_ROOT = Path(__file__).resolve().parents[1]  # <repo>/
SCRIPT_PATH = REPO_ROOT / "backend" / "make_eval_from_csv.py"

def run_builder(env: dict, script_path: Path):
    """Run the builder with env vars. Use absolute script path; preserve original env."""
    prev = os.environ.copy()
    os.environ.update(env)
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        os.environ.clear()
        os.environ.update(prev)

def test_builder_creates_outputs(tmp_path, mini_product_results, mini_search_history):
    # Keep outputs in a simple, consistent folder
    out_prod  = tmp_path / "data" / "eval" / "products_eval.csv"
    out_pairs = tmp_path / "data" / "eval" / "labeled_pairs.csv"
    out_prod.parent.mkdir(parents=True, exist_ok=True)

    env = {
        "SRC_RESULTS": str(mini_product_results),
        "SRC_SEARCH":  str(mini_search_history),
        "OUT_PROD":    str(out_prod),
        "OUT_PAIRS":   str(out_pairs),
    }

    run_builder(env, SCRIPT_PATH)

    assert out_prod.exists(), "products_eval.csv should be created"
    assert out_pairs.exists(), "labeled_pairs.csv should be created"

    dfp = pd.read_csv(out_prod)
    assert {"listing_id","vendor","title","brand","upc","price"}.issubset(dfp.columns)
    # ensure we mapped product_id->upc and have duplicates in the mini fixture
    assert dfp["upc"].nunique() == 2
    assert len(dfp) == 4

    # at least one positive duplicate pair expected
    with out_pairs.open() as f:
        rdr = csv.DictReader(f)
        pos = sum(1 for r in rdr if r["is_duplicate"] == "1")
    assert pos >= 2, "should contain positive duplicate pairs"

def test_builder_handles_missing_raw_field(tmp_path, mini_product_results, mini_search_history):
    # Create a version without 'raw' for one row; builder should not crash.
    df = pd.read_csv(mini_product_results)
    df.loc[0, "raw"] = "{}"
    alt = tmp_path / "product_results_noraw.csv"
    df.to_csv(alt, index=False)

    out_prod  = tmp_path / "data" / "eval" / "products_eval.csv"
    out_pairs = tmp_path / "data" / "eval" / "labeled_pairs.csv"
    out_prod.parent.mkdir(parents=True, exist_ok=True)

    env = {
        "SRC_RESULTS": str(alt),
        "SRC_SEARCH":  str(mini_search_history),
        "OUT_PROD":    str(out_prod),
        "OUT_PAIRS":   str(out_pairs),
    }

    run_builder(env, SCRIPT_PATH)

    assert out_prod.exists() and out_pairs.exists()
