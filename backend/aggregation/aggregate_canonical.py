# backend/aggregation/aggregate_canonical.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple
import math
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from backend.db.models import CanonicalProduct, Variant
from backend.db.models import Base  # if you need create_all in tests
from backend.db.models import Listing  # your existing listing model

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Safe cosine distance (1 - cosine similarity)
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    if da == 0 or db == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (da * db))

def _pick_medoid(listings: list[Listing]) -> Tuple[Optional[str], bool]:
    """
    Choose a medoid listing for display.
    Priority:
      1) If embeddings exist for >=2 items: true medoid by min average cosine distance.
      2) Else: choose the listing whose price is closest to the cluster median (robust proxy).
    Returns (listing_id, is_true_medoid)
    """
    # Try embedding medoid
    embeds = []
    id_order = []
    for l in listings:
        emb = getattr(l, "embedding", None)
        if emb is not None:
            # Expect bytes or list; normalize to np.ndarray
            if isinstance(emb, (bytes, bytearray)):
                try:
                    v = np.frombuffer(emb, dtype=np.float32)
                except Exception:
                    v = None
            elif isinstance(emb, (list, tuple)):
                v = np.asarray(emb, dtype=np.float32)
            else:
                v = None
            if v is not None and v.size > 0:
                embeds.append(v)
                id_order.append(l.id)

    if len(embeds) >= 2:
        E = np.stack(embeds)
        # pairwise average distance for each point
        dmat = np.zeros((E.shape[0], E.shape[0]), dtype=np.float32)
        for i in range(E.shape[0]):
            for j in range(i + 1, E.shape[0]):
                d = _cosine_distance(E[i], E[j])
                dmat[i, j] = d
                dmat[j, i] = d
        avg_d = dmat.mean(axis=1)
        argmin = int(np.argmin(avg_d))
        return id_order[argmin], True

    # Fallback: price-median proxy
    priced = [(l.id, l.price) for l in listings if l.price is not None and not math.isnan(l.price)]
    if not priced:
        return (listings[0].id if listings else None), False
    prices = np.array([p for _, p in priced], dtype=np.float32)
    median = np.median(prices)
    # closest to median
    best = min(priced, key=lambda pair: abs(pair[1] - median))
    return best[0], False


def aggregate_cluster(session: Session, cluster_id: str) -> CanonicalProduct:
    """
    Compute aggregates for a single cluster and upsert CanonicalProduct + Variants.
    """
    listings: list[Listing] = session.scalars(
        select(Listing).where(Listing.cluster_id == cluster_id)
    ).all()

    if not listings:
        raise ValueError(f"No listings found for cluster_id={cluster_id}")

    prices = [l.price for l in listings if l.price is not None]
    price_min = float(min(prices)) if prices else None
    price_max = float(max(prices)) if prices else None
    price_avg = float(sum(prices) / len(prices)) if prices else None

    sellers = {getattr(l, "seller", None) for l in listings if getattr(l, "seller", None)}
    seller_count = len(sellers)

    # Medoid / representative
    rep_id, is_true_medoid = _pick_medoid(listings)

    # Choose a reasonable display title/brand: most common non-empty
    def _mode_nonempty(values: Iterable[Optional[str]]) -> Optional[str]:
        vals = [v.strip() for v in values if v and isinstance(v, str) and v.strip()]
        if not vals:
            return None
        from collections import Counter
        return Counter(vals).most_common(1)[0][0]

    title = _mode_nonempty(l.title for l in listings) or (listings[0].title if listings else None)
    brand = _mode_nonempty(l.brand for l in listings) or (listings[0].brand if listings else None)

    # Upsert canonical
    canonical: CanonicalProduct | None = session.scalar(
        select(CanonicalProduct).where(CanonicalProduct.cluster_id == cluster_id)
    )
    if canonical is None:
        canonical = CanonicalProduct(cluster_id=cluster_id)
        session.add(canonical)

    canonical.title = title
    canonical.brand = brand
    canonical.price_min = price_min
    canonical.price_max = price_max
    canonical.price_avg = price_avg
    canonical.seller_count = seller_count
    canonical.representative_listing_id = rep_id
    canonical.representative_is_medoid = bool(is_true_medoid)

    # Rebuild variants for this canonical
    # Strategy: one variant per listing row (simple & deterministic).
    # If you want to dedup variants by (color,size), switch to a group-by.
    # First, delete existing variants for this canonical
    for v in list(canonical.variants):
        session.delete(v)
    session.flush()

    variant_rows = []
    for l in listings:
        variant_rows.append(
            Variant(
                canonical_id=canonical.id,
                listing_id=l.id,
                color=getattr(l, "variant_color", None),
                size=getattr(l, "variant_size", None),
                material=getattr(l, "variant_material", None),
                price=l.price,
            )
        )
    canonical.variant_count = len(variant_rows)
    session.add_all(variant_rows)

    # Optional: record some stats JSON for debugging
    canonical.stats_json = {
        "cluster_size": len(listings),
        "has_embedding_medoid": bool(is_true_medoid),
    }

    session.flush()
    return canonical


def run_aggregation_for_all_clusters(session: Session) -> int:
    """
    Aggregates all distinct clusters found in `listing`.
    Returns number of canonical products processed.
    """
    cluster_ids = session.scalars(select(func.distinct(Listing.cluster_id))).all()
    count = 0
    for cid in cluster_ids:
        if cid:
            aggregate_cluster(session, cid)
            count += 1
    session.commit()
    return count
