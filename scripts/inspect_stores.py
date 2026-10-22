import os
import faiss
import json

targets = [
    ("COCO CLIP", "storage/coco/clip"),
    ("COCO SigLIP", "storage/coco/siglip"),
    ("NewsSeek CLIP", "storage/newsseek/clip"),
    ("NewsSeek SigLIP", "storage/newsseek/siglip"),
]

for name, p in targets:
    idx_p = os.path.join(p, "index.faiss")
    meta_p = os.path.join(p, "metadata.json")
    if os.path.exists(idx_p) and os.path.exists(meta_p):
        idx = faiss.read_index(idx_p)
        meta = json.load(open(meta_p, encoding="utf-8"))
        count = len(meta.get("items", []))
        size_mb = os.path.getsize(idx_p) / (1024 * 1024)
        print(f"{name}: READY - {idx.ntotal} vectors (dim {idx.d}, {size_mb:.2f} MB), metadata count: {count}")
    else:
        print(f"{name}: NOT FOUND or INCOMPLETE in {p}")
