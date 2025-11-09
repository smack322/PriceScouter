import os, sys, subprocess, pandas as pd

def test_no_positive_pairs_yields_zero_pairwise(tmp_path):
    # products with distinct product_id → builder will make no positives if you customize it that way
    prod = tmp_path / "data/eval/products_eval.csv"
    pairs = tmp_path / "data/eval/labeled_pairs.csv"
    (tmp_path / "data/eval").mkdir(parents=True, exist_ok=True)

    # Make products_eval with no repeats
    pd.DataFrame([
        {"listing_id":"L1","vendor":"v","title":"Alpha case","brand":"BrandA","upc":"P1","price":10.0},
        {"listing_id":"L2","vendor":"v","title":"Bravo case","brand":"BrandB","upc":"P2","price":12.0},
    ]).to_csv(prod, index=False)

    # Labeled pairs: none positive
    pairs.write_text("pair_id,id_a,id_b,is_duplicate\n")

    reports = tmp_path / "reports"; reports.mkdir()
    env_eval = {"PYTHONPATH": os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))}
    cmd = [sys.executable, "backend/run_clustering_eval.py",
           "--products", str(prod), "--pairs", str(pairs),
           "--theta", "0.85", "--outdir", str(reports)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=env_eval["PYTHONPATH"])
    assert res.returncode == 0

    import csv
    with (reports/"clustering_metrics.csv").open() as f:
        rows = {r["metric"]: float(r["value"]) for r in csv.DictReader(f)}
    # pairwise metrics are 0; purity is >= 0 (often 1.0 because all singletons)
    assert rows["Pairwise F1"] == 0.0
