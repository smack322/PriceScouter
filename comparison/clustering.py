# comparison/clustering.py
from __future__ import annotations
import argparse, math, uuid, json
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---- Data contracts ----
@dataclass
class ProductRecord:
    listing_id: str
    vendor: str
    title: str
    brand: Optional[str]
    upc: Optional[str]
    attrs: Dict[str, str]
    price: Optional[float]

# ---- Normalization helpers ----
STOPWORDS = {"the","a","an","for","with","and","or","to","of","by","in"}

def normalize_text(s: str) -> str:
    return " ".join(
        "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in s or "")
        .split()
    )

def normalize_brand(b: Optional[str]) -> Optional[str]:
    if not b: return None
    b = normalize_text(b)
    ALIASES = {
        "p g":"procter gamble",
        "p&g":"procter gamble",
        "hp inc":"hp",
    }
    return ALIASES.get(b, b)

def tokens(s: str) -> List[str]:
    return [t for t in s.split() if t not in STOPWORDS and len(t) > 1]

def blocking_key(rec: ProductRecord) -> Tuple[str, Tuple[str, ...]]:
    t = tokens(normalize_text(rec.title))
    key_tokens = tuple(sorted(t[:5]))
    b = normalize_brand(rec.brand) or ""
    return (b, key_tokens)

# ---- Embedding (toy BoW to keep tests fast) ----
def embed_texts(texts: List[str]) -> np.ndarray:
    vocab: Dict[str,int] = {}
    rows: List[Dict[int,float]] = []
    for s in texts:
        vec: Dict[int,float] = {}
        for tok in tokens(normalize_text(s)):
            idx = vocab.setdefault(tok, len(vocab))
            vec[idx] = vec.get(idx, 0.0) + 1.0
        rows.append(vec)
    dim = len(vocab)
    X = np.zeros((len(rows), dim), dtype=np.float32)
    for i, vec in enumerate(rows):
        for j, v in vec.items():
            X[i, j] = v
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / norms

# ---- Graph clustering (Union-Find) ----
class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a,b):
        a,b = self.find(a), self.find(b)
        if a==b: return
        if self.r[a]<self.r[b]: a,b=b,a
        self.p[b]=a
        if self.r[a]==self.r[b]: self.r[a]+=1

def cluster_block(records: List[ProductRecord], theta: float) -> List[List[int]]:
    if not records: return []
    titles = [r.title for r in records]
    X = embed_texts(titles)
    sim = cosine_similarity(X)
    n = len(records)
    dsu = DSU(n)
    norm_brand = [normalize_brand(r.brand) for r in records]
    for i in range(n):
        for j in range(i+1, n):
            if norm_brand[i] and norm_brand[j] and norm_brand[i] != norm_brand[j]:
                continue
            if sim[i, j] >= theta:
                dsu.union(i, j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = dsu.find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())

def stable_uuid(brand: str, upc: Optional[str], key_tokens: Tuple[str, ...]) -> str:
    ns = uuid.NAMESPACE_URL
    name = json.dumps({"brand": brand, "upc": upc or "", "kt": key_tokens}, sort_keys=True)
    return str(uuid.uuid5(ns, name))

# ---- End-to-end pipeline ----
def cluster_products(records: Iterable[ProductRecord], theta: float=0.85):
    by_upc: Dict[str, List[ProductRecord]] = {}
    soft_pool: List[ProductRecord] = []
    for r in records:
        if r.upc:
            by_upc.setdefault(r.upc, []).append(r)
        else:
            soft_pool.append(r)

    clusters: List[List[ProductRecord]] = []
    # UPC groups → split by brand purity
    for upc, rs in by_upc.items():
        by_brand: Dict[str, List[ProductRecord]] = {}
        for r in rs:
            b = normalize_brand(r.brand) or ""
            by_brand.setdefault(b, []).append(r)
        clusters.extend(by_brand.values())

    # Token blocking on the rest
    blocks: Dict[Tuple[str, Tuple[str, ...]], List[ProductRecord]] = {}
    for r in soft_pool:
        blocks.setdefault(blocking_key(r), []).append(r)

    # Intra-block clustering via embeddings
    for key, rs in blocks.items():
        idx_groups = cluster_block(rs, theta)
        for g in idx_groups:
            clusters.append([rs[i] for i in g])

    # Canonicalization & aggregates
    outputs = []
    memberships = []
    for group in clusters:
        if not group: continue
        brand = normalize_brand(group[0].brand) or ""
        kt = blocking_key(group[0])[1]
        upc = group[0].upc
        cid = stable_uuid(brand, upc, kt)
        prices = [r.price for r in group if r.price is not None and not math.isnan(r.price)]
        price_min = min(prices) if prices else None
        price_avg = sum(prices)/len(prices) if prices else None
        price_max = max(prices) if prices else None
        # medoid title
        titles = [r.title for r in group]
        X = embed_texts(titles)
        sims = cosine_similarity(X)
        medoid_idx = int(np.argmax(sims.mean(axis=1)))
        canonical_title = titles[medoid_idx]
        outputs.append({
            "canonical_product_id": cid,
            "brand": brand,
            "title": canonical_title,
            "price_min": price_min,
            "price_avg": price_avg,
            "price_max": price_max,
            "listing_count": len(group),
            "attrs_json": json.dumps({})  # extend later
        })
        for i, r in enumerate(group):
            memberships.append({
                "canonical_product_id": cid,
                "listing_id": r.listing_id,
                "vendor": r.vendor,
                "similarity_to_centroid": float(sims[medoid_idx, i]),
            })
    return outputs, memberships

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=0.85)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    # TODO: replace with ORM fetch
    records = load_records_batch(limit=args.batch_size)

    outputs, memberships = cluster_products(records, theta=args.theta)

    if args.write:
        persist(outputs, memberships)
    else:
        print(f"Preview canonical products: {len(outputs)}")
        print(json.dumps(outputs[:5], indent=2))

if __name__ == "__main__":
    main()
