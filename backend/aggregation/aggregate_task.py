import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.aggregation.aggregate_canonical import run_aggregation_for_all_clusters
from backend.db.models import Base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./price_scouter.db")

def main():
    engine = create_engine(DATABASE_URL, future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, future=True)
    with Session() as session:
        n = run_aggregation_for_all_clusters(session)
        print(f"Aggregated {n} canonical products.")

if __name__ == "__main__":
    main()