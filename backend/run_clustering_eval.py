#!/usr/bin/env python3
import argparse, csv, os, time, subprocess, sys
from datetime import datetime
import pandas as pd
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from comparison.clustering import ProductRecord, cluster_products
from comparison.metrics import cluster_purity, pairwise_prf  # add pairwise_prf per prior snippet

def load_products_csv(path: str):
    df = pd.read_csv(path)
    recs = []
    for _, r in df.iterrows():
        recs.append(ProductRecord(
            listing_id=str(r["listing_id"]),
            vendor=str(r.get("vendor","unknown")),
            title=str(r["title"]),
            brand=(None if pd.isna(r.get("brand")) else str(r.get("brand"))),
            upc=(None if pd.isna(r.get("upc")) else str(r.get("upc"))),
            attrs={},
            price=(None if pd.isna(r.get("price")) else float(r.get("price")))
        ))
    return recs

def load_labels_from_pairs(path: str):
    df = pd.read_csv(path)
    # union-find on positive pairs to convert to gold clusters
    parent = {}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra!=rb: parent[rb]=ra

    for _, r in df.iterrows():
        if int(r["is_duplicate"])==1:
            union(str(r["id_a"]), str(r["id_b"]))

    # flatten to {listing_id: gold_label}
    labels = {lid: find(lid) for lid in list(parent.keys())}
    return labels

def memberships_to_clusters(memberships):
    by = defaultdict(list)
    for m in memberships:
        by[m["canonical_product_id"]].append(m["listing_id"])
    return list(by.values())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", default="local_db/data/eval/products_eval.csv")
    ap.add_argument("--pairs",    default="local_db/data/eval/labeled_pairs.csv")
    ap.add_argument("--theta",    type=float, default=0.85)
    ap.add_argument("--outdir",   default="reports")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    recs = load_products_csv(args.products)
    labels = load_labels_from_pairs(args.pairs)

    t0 = time.perf_counter()
    outputs, memberships = cluster_products(recs, theta=args.theta)
    elapsed = time.perf_counter() - t0

    pred_clusters = memberships_to_clusters(memberships)
    # Keep only items that exist in labels for fair comparison
    pred_clusters = [[x for x in c if x in labels] for c in pred_clusters]
    pred_clusters = [c for c in pred_clusters if c]

    prf = pairwise_prf(pred_clusters, labels)
    purity = cluster_purity(pred_clusters, labels)

    rows = [
        ("Pairwise Precision", prf["precision"]),
        ("Pairwise Recall",    prf["recall"]),
        ("Pairwise F1",        prf["f1"]),
        ("Cluster Purity",     purity),
        ("Runtime (s)",        elapsed),
    ]

    sha = subprocess.run(["git","rev-parse","--short","HEAD"], capture_output=True, text=True).stdout.strip()
    dt  = datetime.utcnow().isoformat(timespec="seconds")+"Z"

    csv_path = os.path.join(args.outdir, "clustering_metrics.csv")
    md_path  = os.path.join(args.outdir, "clustering_metrics.md")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric","value"]); w.writerows(rows)

    with open(md_path, "w") as f:
        f.write(f"# Clustering Evaluation Metrics\n\nRun: {dt}  \nGit SHA: {sha}\n\n")
        f.write("| Metric | Value |\n|---|---:|\n")
        for k,v in rows: f.write(f"| {k} | {v:.4f} |\n")
        f.write("\n**Dataset:** `products={}`; `pairs={}`  \n".format(args.products, args.pairs))
        f.write("**Params:** theta={}  \n".format(args.theta))
        f.write("**Notes:** Gold clusters from positive pairs; labels derived from Google Shopping `product_id` within the same `search_id`.\n")

    print(f"Wrote {csv_path} and {md_path}. Runtime={elapsed:.3f}s")

if __name__ == "__main__":
    main()
