# backend/embeddings/vector_utils.py
import os
import json
from typing import List, Dict, Any, Optional

import numpy as np
import pyarrow.parquet as pq
import pandas as pd


class Embedder:
    def __init__(self, provider: str = "openai", model: str = "", dim: Optional[int] = None):
        self.provider = (provider or "openai").strip().lower()
        self.dim = dim

        # --- choose model from env by provider ---
        if self.provider == "openai":
            model = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        else:
            model = model or os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        # --- guardrail: prevent cross-provider model names ---
        openai_like = any(x in model for x in ["text-embedding-3", "text-embedding-ada", "ada-002"])
        sbert_like  = any("/" in model or x in model.lower()
                          for x in ["sentence-transformers", "all-minilm", "e5", "gte", "mpnet", "distil"])
        if self.provider == "sbert" and openai_like:
            # auto-correct to a sane SBERT default
            print(f"[Embedder] '{model}' looks like an OpenAI model. Using SBERT default instead.")
            model = "sentence-transformers/all-MiniLM-L6-v2"
        if self.provider == "openai" and sbert_like:
            raise ValueError(f"[Embedder] '{model}' looks like a HuggingFace model. "
                             f"Set OPENAI_EMBED_MODEL to an OpenAI embedding model name.")

        self.model = model

        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set. Either set it or use EMBED_PROVIDER=sbert.")
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=api_key)
            self._backend = "openai"
        else:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._sbert = SentenceTransformer(self.model)
            self._backend = "sbert"

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Returns L2-normalized float32 embeddings for cosine/IP search.
        """
        if self._backend == "openai":
            # Requires OPENAI_API_KEY in env
            out = self._client.embeddings.create(model=self.model, input=texts)
            vecs = np.array([d.embedding for d in out.data], dtype="float32")
            # Normalize
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (vecs / norms).astype("float32")
        else:
            # SBERT can normalize internally
            from sentence_transformers import SentenceTransformer  # type: ignore
            return self._sbert.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")


class VectorStore:
    """
    FAISS-backed local vector store loader + search.
    Expects out_dir to contain:
      - index.faiss
      - idmap.npy  (np.int64 product ids aligned with FAISS rows)
      - meta.parquet (columns: id, text, metadata_json)
    """
    def __init__(self, embedder: Embedder, out_dir: str = "./data/faiss"):
        import faiss  # lazy import
        self.embedder = embedder
        self.out_dir = out_dir
        self.faiss = faiss

        idx_path = os.path.join(out_dir, "index.faiss")
        idmap_path = os.path.join(out_dir, "idmap.npy")
        meta_path = os.path.join(out_dir, "meta.parquet")

        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"Missing FAISS index: {idx_path}")
        if not os.path.exists(idmap_path):
            raise FileNotFoundError(f"Missing idmap: {idmap_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")

        self.index = faiss.read_index(idx_path)
        self.idmap = np.load(idmap_path)

        meta_df = pq.read_table(meta_path).to_pandas()
        if "id" not in meta_df.columns or "metadata_json" not in meta_df.columns or "text" not in meta_df.columns:
            raise ValueError("meta.parquet must contain columns: id, text, metadata_json")
        self.meta_df: pd.DataFrame = meta_df.set_index("id")

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        qv = self.embedder.embed([query])
        scores, idxs = self.index.search(qv, k)
        scores = scores[0]
        idxs = idxs[0]

        results: List[Dict[str, Any]] = []
        for pi, score in zip(idxs, scores):
            if pi < 0:
                continue
            prod_id = int(self.idmap[pi])
            row = self.meta_df.loc[prod_id]
            meta = json.loads(row["metadata_json"])
            results.append({
                "score": float(score),
                "id": prod_id,
                "text": row["text"],
                **meta,  # includes title/source/seller/total/currency/link/query...
            })
        return results
