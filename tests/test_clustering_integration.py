# tests/test_clustering_integration.py
import math
import uuid
import pytest
from datetime import datetime

pytestmark = pytest.mark.integration

from comparison.clustering import ProductRecord, cluster_products

# -------- helpers --------
def _rec(id, title, price, brand="Apple", vendor="s1", upc=None, attrs=None):
    return ProductRecord(
        listing_id=id,
        vendor=vendor,
        title=title,
        brand=brand,
        upc=upc,
        attrs=attrs or {},
        price=price,
    )

def _define_test_models(db_session):
    """
    Define ISOLATED test tables so we don't collide with any existing schema.
    """
    from sqlalchemy.orm import declarative_base
    from sqlalchemy import Column, String, Numeric, Integer, DateTime, ForeignKey

    Base = declarative_base()

    class TCanonicalProduct(Base):
        __tablename__ = "t_canonical_product"  # test-only table name
        canonical_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        cluster_id = Column(String)
        title = Column(String, nullable=False)
        # keep a small set of columns sufficient for the assertions
        medoid_product_id = Column(String)
        price_min = Column(Numeric)
        price_max = Column(Numeric)
        price_avg = Column(Numeric)
        seller_count = Column(Integer)
        updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    class TCanonicalVariant(Base):
        __tablename__ = "t_canonical_variant"  # test-only table name
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        canonical_id = Column(String, ForeignKey("t_canonical_product.canonical_id"), nullable=False)
        product_id = Column(String, nullable=False)
        color = Column(String)
        size = Column(String)

    engine = db_session.get_bind()
    Base.metadata.create_all(engine)
    return TCanonicalProduct, TCanonicalVariant

def _only_model_columns(model_cls, data: dict):
    cols = set(model_cls.__table__.columns.keys())
    return {k: v for k, v in data.items() if k in cols}

# -------- the test --------
def test_write_canonical_and_variants(db_session):
    CanonicalProduct, CanonicalVariant = _define_test_models(db_session)

    # Keep identical tokens so they cluster together under the toy embedder/blocking
    records = [
        _rec("A", "Apple iPhone 13 128GB Black", 599.0, brand="Apple", vendor="s1"),
        _rec("B", "Apple iPhone 13 128GB Black", 579.0, brand="Apple", vendor="s2"),
        _rec("C", "Apple iPhone 13 128GB Black", 589.0, brand="Apple", vendor="s1"),
    ]

    outputs, memberships = cluster_products(records, theta=0.60)
    assert len(outputs) == 1
    canon_out = outputs[0]
    cid = canon_out["canonical_product_id"]

    # derive medoid & seller_count from memberships
    mems = [m for m in memberships if m["canonical_product_id"] == cid]
    assert len(mems) == 3
    medoid_row = max(mems, key=lambda m: m["similarity_to_centroid"])
    medoid_pid = medoid_row["listing_id"]
    seller_count = len({m["vendor"] for m in mems})

    # upsert canonical product into the test table
    cp_data = {
        "canonical_id": cid,
        "cluster_id": cid,
        "title": canon_out["title"],
        "price_min": canon_out["price_min"],
        "price_avg": canon_out["price_avg"],
        "price_max": canon_out["price_max"],
        "seller_count": seller_count,
        "medoid_product_id": medoid_pid,
    }
    cp = CanonicalProduct(**_only_model_columns(CanonicalProduct, cp_data))
    db_session.merge(cp)
    db_session.flush()

    # fetch canonical by canonical_id
    row = db_session.query(CanonicalProduct).filter_by(canonical_id=getattr(cp, "canonical_id")).one()

    # insert variants
    for m in mems:
        v_data = {
            "canonical_id": getattr(row, "canonical_id"),
            "product_id": m["listing_id"],
            "color": None,
            "size": None,
        }
        db_session.add(CanonicalVariant(**_only_model_columns(CanonicalVariant, v_data)))

    db_session.commit()

    # ---- Assertions on stored stats ----
    assert math.isclose(float(row.price_min), 579.0, rel_tol=1e-6)
    assert math.isclose(float(row.price_max), 599.0, rel_tol=1e-6)
    assert math.isclose(float(row.price_avg), (599.0 + 579.0 + 589.0) / 3.0, rel_tol=1e-6)
    assert int(row.seller_count) == 2
    assert row.medoid_product_id in {"A", "B", "C"}

    # variants count
    vcount = db_session.query(CanonicalVariant).filter_by(canonical_id=getattr(row, "canonical_id")).count()
    assert vcount == 3
