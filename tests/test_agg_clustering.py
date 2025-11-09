import math
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.models import Base, Listing, CanonicalProduct, Variant
from backend.aggregation.aggregate_canonical import aggregate_cluster

@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    with Session() as s:
        yield s

def _f32(v):
    arr = np.asarray(v, dtype=np.float32)
    return arr.tobytes()

def test_aggregate_single_cluster(session):
    # Seed listings
    cluster_id = "C1"
    L = [
        Listing(
            id="L1", cluster_id=cluster_id, title="ACME Widget", brand="ACME",
            price=10.0, seller="s1", variant_color="red", variant_size="S",
            embedding=_f32([0.1, 0.2, 0.7]),
        ),
        Listing(
            id="L2", cluster_id=cluster_id, title="ACME Widget", brand="ACME",
            price=12.0, seller="s2", variant_color="red", variant_size="M",
            embedding=_f32([0.09, 0.21, 0.70]),
        ),
        Listing(
            id="L3", cluster_id=cluster_id, title="ACME Widget", brand="ACME",
            price=20.0, seller="s2", variant_color="blue", variant_size="M",
            embedding=_f32([0.9, 0.1, 0.0]),
        ),
    ]
    session.add_all(L)
    session.commit()

    canonical = aggregate_cluster(session, cluster_id)
    session.commit()

    # Acceptance Criteria: min, max, avg price and seller count
    assert canonical.price_min == 10.0
    assert canonical.price_max == 20.0
    assert math.isclose(canonical.price_avg, (10.0 + 12.0 + 20.0) / 3, rel_tol=1e-6)
    assert canonical.seller_count == 2

    # Variants stored
    variants = session.query(Variant).filter(Variant.canonical_id == canonical.id).all()
    assert len(variants) == 3
    # at least verifies attributes copied over
    colors = {v.color for v in variants}
    sizes = {v.size for v in variants}
    assert colors == {"red", "blue"}
    assert sizes == {"S", "M"}

    # Medoid identified & flagged as display record
    assert canonical.representative_listing_id in {"L1", "L2"}  # these two are closer in embedding space
    assert canonical.representative_is_medoid is True

    # Basic title/brand rollups set
    assert canonical.title == "ACME Widget"
    assert canonical.brand == "ACME"

def test_aggregate_handles_missing_prices_and_no_embeddings(session):
    cluster_id = "C2"
    session.add_all([
        Listing(id="A", cluster_id=cluster_id, title="Thing", brand="BrandX", price=None, seller="x"),
        Listing(id="B", cluster_id=cluster_id, title="Thing", brand="BrandX", price=9.0, seller="y"),
        Listing(id="C", cluster_id=cluster_id, title="Thing", brand="BrandX", price=11.0, seller="y"),
    ])
    session.commit()

    canonical = aggregate_cluster(session, cluster_id)
    session.commit()

    assert canonical.price_min == 9.0
    assert canonical.price_max == 11.0
    assert math.isclose(canonical.price_avg, (9.0 + 11.0) / 2, rel_tol=1e-6)
    assert canonical.seller_count == 2
    assert canonical.variant_count == 3
    # No embeddings -> fallback to price-median proxy => rep is B or C (closest to 10 median is either 9 or 11)
    assert canonical.representative_listing_id in {"B", "C"}
    assert canonical.representative_is_medoid is False