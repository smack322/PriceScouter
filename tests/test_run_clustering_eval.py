import os, subprocess, sys, csv, re, pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_cmd(cmd, env=None, cwd=None):
    env0 = os.environ.copy()
    if env:
        env0.update(env)
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or REPO_ROOT)
    assert res.returncode == 0, f"Command failed: {cmd}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return res

def test_eval_end_to_end(tmp_path, mini_product_results, mini_search_history):
    # 1) Build eval inputs
    out_prod  = tmp_path / "backend/local_db/data/eval/products_eval.csv"
    out_pairs = tmp_path / "backend/local_db/data/eval/labeled_pairs.csv"
    (tmp_path / "data/eval").mkdir(parents=True, exist_ok=True)

    env = {
        "SRC_RESULTS": str(mini_product_results),
        "SRC_SEARCH":  str(mini_search_history),
        "OUT_PROD":    str(out_prod),
        "OUT_PAIRS":   str(out_pairs),
    }
    run_cmd([sys.executable, "backend/make_eval_from_csv.py"], env=env)

    assert out_prod.exists() and out_pairs.exists()

    # 2) Run clustering eval (with UPC enabled -> should be near perfect)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(exist_ok=True)
    cmd = [
        sys.executable, "backend/run_clustering_eval.py",
        "--products", str(out_prod),
        "--pairs",    str(out_pairs),
        "--theta",    "0.85",
        "--outdir",   str(reports_dir),
    ]
    # Ensure PYTHONPATH sees the repo root so `comparison` is importable
    env_eval = {"PYTHONPATH": REPO_ROOT}
    run_cmd(cmd, env=env_eval)

    csv_path = reports_dir / "clustering_metrics.csv"
    md_path  = reports_dir / "clustering_metrics.md"
    assert csv_path.exists() and md_path.exists()

    # Basic regression: check required metric rows exist
    with csv_path.open() as f:
        rdr = {r["metric"]: r["value"] for r in csv.DictReader(f)}
    for k in ["Pairwise Precision","Pairwise Recall","Pairwise F1","Cluster Purity","Runtime (s)"]:
        assert k in rdr

    # Markdown sanity: header + Git SHA line
    txt = md_path.read_text()
    assert "# Clustering Evaluation Metrics" in txt
    assert re.search(r"Git SHA:\s+\S+", txt)

def test_eval_ignore_upc_changes_behavior(tmp_path, mini_product_results, mini_search_history):
    # Build eval
    out_prod  = tmp_path / "data/eval/products_eval.csv"
    out_pairs = tmp_path / "data/eval/labeled_pairs.csv"
    (tmp_path / "data/eval").mkdir(parents=True, exist_ok=True)
    env = {
        "SRC_RESULTS": str(mini_product_results),
        "SRC_SEARCH":  str(mini_search_history),
        "OUT_PROD":    str(out_prod),
        "OUT_PAIRS":   str(out_pairs),
    }
    run_cmd([sys.executable, "backend/make_eval_from_csv.py"], env=env)

    # Run with UPC (baseline)
    reports1 = tmp_path / "reports1"; reports1.mkdir()
    env_eval = {"PYTHONPATH": REPO_ROOT}
    cmd1 = [sys.executable, "backend/run_clustering_eval.py",
            "--products", str(out_prod), "--pairs", str(out_pairs),
            "--theta","0.85","--outdir",str(reports1)]
    run_cmd(cmd1, env=env_eval)

    # Run ignoring UPC (hard key off) – expect F1 to be <= baseline
    reports2 = tmp_path / "reports2"; reports2.mkdir()
    cmd2 = [sys.executable, "backend/run_clustering_eval.py",
            "--products", str(out_prod), "--pairs", str(out_pairs),
            "--theta","0.85","--ignore-upc","--outdir",str(reports2)]
    run_cmd(cmd2, env=env_eval)

    import csv as _csv
    def f1(path):
        with (path/"clustering_metrics.csv").open() as f:
            rows = {r["metric"]: float(r["value"]) for r in _csv.DictReader(f)}
        return rows["Pairwise F1"]

    f1_hard = f1(reports1)
    f1_soft = f1(reports2)
    assert f1_soft <= f1_hard + 1e-9  # ignoring UPC shouldn't improve beyond baseline
