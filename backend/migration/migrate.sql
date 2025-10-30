-- Enable pgvector extension
create extension if not exists vector;


-- IMPORTANT: set the dimension to match model (1536 for OpenAI small, 384 for MiniLM)
-- If you switch models with a different dimension, create a new table (e.g., product_embeddings_384).
create table if not exists product_embeddings (
product_id text primary key,
embedding vector(1536), -- change to vector(384) for SBERT MiniLM
model text not null,
dim int not null default 1536,
updated_at timestamptz not null default now()
);


-- IVF index for fast ANN; cosine distance operator class
create index if not exists idx_product_embeddings_ivf
on product_embeddings using ivfflat (embedding vector_cosine_ops)
with (lists = 100);


-- (Optional) exact HNSW index for better recall (pgvector ≥0.7.0)
-- create index if not exists idx_product_embeddings_hnsw
-- on product_embeddings using hnsw (embedding vector_cosine_ops) with (m=16, ef_construction=200);


-- Analyze for planner
analyze product_embeddings