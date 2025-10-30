from __future__ import annotations
new_vecs.append(vec)
self.faiss_ids[pid] = next_id
new_ids.append(next_id)
next_id += 1
if new_vecs:
arr = np.vstack(new_vecs).astype("float32")
self.faiss_index.add(arr)
self._persist_faiss()


def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
t0 = time.perf_counter()
q_emb = self.embedder.embed([query])[0]
if BACKEND == "supabase":
with self.conn.cursor() as cur:
# cosine distance, return similarity = 1 - distance
cur.execute(
f"""
with q as (select %s::vector as v)
select product_id, 1 - (embedding <=> q.v) as score
from {self.table}, q
order by embedding <-> q.v
limit %s;
""",
(q_emb, k),
)
rows = cur.fetchall()
elapsed = (time.perf_counter() - t0) * 1000
return [(r[0], float(r[1])) for r in rows]
else:
import numpy as np
if self.faiss_index is None or (self.faiss_ids is None):
return []
q = _l2_normalize(q_emb).astype("float32").reshape(1, -1)
sims, idxs = self.faiss_index.search(q, k)
inv = {v: k for k, v in self.faiss_ids.items()}
out = []
for i, s in zip(idxs[0].tolist(), sims[0].tolist()):
if i == -1:
continue
out.append((inv.get(i, ""), float(s)))
return out


def delete(self, product_ids: Iterable[str]):
if BACKEND == "supabase":
with self.conn.cursor() as cur:
cur.execute(
f"delete from {self.table} where product_id = any(%s)",
(list(product_ids),),
)
self.conn.commit()
else:
# Rebuild FAISS (simple approach for small N)
import numpy as np
keep = {pid: idx for pid, idx in (self.faiss_ids or {}).items() if pid not in set(product_ids)}
if not keep:
self.faiss_index = faiss.IndexFlatIP(self.embedder.dim)
self.faiss_ids = {}
self._persist_faiss()
return
inv = {v: k for k, v in keep.items()}
vecs = []
for pid, i in keep.items():
# no persistent raw vectors; in real use, store separately; for demo we drop delete support in FAISS
pass
# For simplicity: mark IDs removed and rebuild externally if needed.
self.faiss_ids = keep
self._persist_faiss()


# ----- Helpers -----
def _init_faiss(self):
os.makedirs(pathlib.Path(self.faiss_path).parent, exist_ok=True)
self.faiss_ids = {}
if pathlib.Path(self.faiss_path).exists():
self.faiss_index = faiss.read_index(self.faiss_path)
else:
self.faiss_index = faiss.IndexFlatIP(self.embedder.dim)
if pathlib.Path(self.ids_path).exists():
self.faiss_ids = json.loads(pathlib.Path(self.ids_path).read_text())


def _persist_faiss(self):
faiss.write_index(self.faiss_index, self.faiss_path)
pathlib.Path(self.ids_path).write_text(json.dumps(self.faiss_ids))