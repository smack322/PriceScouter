# tests/db_fixtures.py
import os, uuid, pytest
from sqlalchemy import create_engine, Column, String, Numeric, Integer, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

TEST_DB_URL = os.getenv("TEST_DB_URL", "sqlite+pysqlite:///:memory:")

@pytest.fixture
def db_session():
    engine = create_engine(TEST_DB_URL, future=True)
    from models.canonical import Base
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    sess = Session()
    try:
        yield sess
    finally:
        sess.close()
        Base.metadata.drop_all(engine)

TEST_DB_URL = os.getenv("TEST_DB_URL", "sqlite+pysqlite:///:memory:")
Base = declarative_base()

class CanonicalProduct(Base):
    __tablename__ = "canonical_product"
    canonical_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    cluster_id = Column(String)
    title = Column(String, nullable=False)
    medoid_product_id = Column(String)
    price_min = Column(Numeric)
    price_max = Column(Numeric)
    price_avg = Column(Numeric)
    seller_count = Column(Integer)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class CanonicalVariant(Base):
    __tablename__ = "canonical_variant"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    canonical_id = Column(String, ForeignKey("canonical_product.canonical_id"), nullable=False)
    product_id = Column(String, nullable=False)
    color = Column(String)
    size = Column(String)

@pytest.fixture
def db_session():
    engine = create_engine(TEST_DB_URL, future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    sess = Session()
    try:
        yield sess
    finally:
        sess.close()
        Base.metadata.drop_all(engine)
