# backend/db/models.py
from __future__ import annotations
import uuid
from sqlalchemy import (
    Column, String, Float, Integer, ForeignKey, UniqueConstraint, Index, Boolean
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.types import JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

def _uuid():
    return str(uuid.uuid4())

class CanonicalProduct(Base):
    __tablename__ = "canonical_product"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    cluster_id: Mapped[str] = mapped_column(String, index=True)  # ties back to clustering output

    # Display / representative fields
    representative_listing_id: Mapped[str | None] = mapped_column(String, ForeignKey("listing.id"), nullable=True)
    representative_is_medoid: Mapped[bool] = mapped_column(Boolean, default=False)

    # Aggregates
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    brand: Mapped[str | None] = mapped_column(String, nullable=True)
    price_min: Mapped[float | None] = mapped_column(Float)
    price_max: Mapped[float | None] = mapped_column(Float)
    price_avg: Mapped[float | None] = mapped_column(Float)
    seller_count: Mapped[int | None] = mapped_column(Integer)
    variant_count: Mapped[int | None] = mapped_column(Integer)

    # Optional: store some rollup JSON for debugging/analytics
    stats_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Relationships
    variants: Mapped[list[Variant]] = relationship(
        "Variant", back_populates="canonical", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("cluster_id", name="uq_canonical_cluster"),
        Index("ix_canonical_cluster", "cluster_id"),
    )


class Variant(Base):
    __tablename__ = "variant"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    canonical_id: Mapped[str] = mapped_column(String, ForeignKey("canonical_product.id"), index=True)

    # Link the concrete listing this variant row came from (for drill-down in UI)
    listing_id: Mapped[str] = mapped_column(String, ForeignKey("listing.id"), index=True)

    # Variant attributes (extend as needed)
    color: Mapped[str | None] = mapped_column(String, nullable=True)
    size: Mapped[str | None] = mapped_column(String, nullable=True)
    material: Mapped[str | None] = mapped_column(String, nullable=True)

    # Snapshot pricing for this variant/listing
    price: Mapped[float | None] = mapped_column(Float)

    canonical: Mapped[CanonicalProduct] = relationship("CanonicalProduct", back_populates="variants")

# backend/db/models.py (excerpt)
from sqlalchemy import Column, String, Float, LargeBinary

class Listing(Base):
    __tablename__ = "listing"

    id = Column(String, primary_key=True)
    cluster_id = Column(String, index=True)
    title = Column(String)
    brand = Column(String)
    price = Column(Float)
    seller = Column(String)

    # optional variant attrs
    variant_color = Column(String)
    variant_size = Column(String)
    variant_material = Column(String)

    # optional embedding (float32 array)
    embedding = Column(LargeBinary, nullable=True)

