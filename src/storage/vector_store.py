from __future__ import annotations

import os
import json
from typing import Any, List, Dict, Optional, Tuple, Union
from pathlib import Path

import faiss
import numpy as np


SUPPORTED_INDEX_TYPES: tuple[str, ...] = ("FlatIP", "FlatL2")


class VectorStore:
    """FAISS-based Vector Store for Cross-Modal Semantic Retrieval.
    
    Supports:
    - High-performance Cosine Similarity search via IndexFlatIP (L2 normalized vectors).
    - Structured metadata persistence (JSON) synchronized with vector indices.
    - Zero external dependencies beyond faiss-cpu / faiss-gpu.
    """

    def __init__(self, dim: int = 512, index_type: str = "FlatIP", model_name: str = "clip") -> None:
        if index_type not in SUPPORTED_INDEX_TYPES:
            raise ValueError(f"index_type must be one of {SUPPORTED_INDEX_TYPES}")
        
        self.dim: int = dim
        self.index_type: str = index_type
        self.model_name: str = model_name
        self.index: faiss.Index = self._create_index()
        self.metadata: list[dict[str, Any]] = []

    def _create_index(self) -> faiss.Index:
        if self.index_type == "FlatIP":
            return faiss.IndexFlatIP(self.dim)
        return faiss.IndexFlatL2(self.dim)

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata_list: list[dict[str, Any]],
    ) -> None:
        """Adds normalized vector embeddings and their corresponding metadata."""
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be a 2D numpy array, got shape {vectors.shape}")
        if vectors.shape[0] != len(metadata_list):
            raise ValueError(f"Number of vectors ({vectors.shape[0]}) does not match metadata count ({len(metadata_list)})")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dim ({vectors.shape[1]}) does not match store dim ({self.dim})")

        # Ensure float32 contiguous array
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        
        # Ensure L2 normalization for Inner Product (Cosine similarity)
        if self.index_type == "FlatIP":
            faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.metadata.extend(metadata_list)

    def search(
        self, query_vector: np.ndarray, k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Searches nearest vectors for a single query vector."""
        if self.index.ntotal == 0:
            raise RuntimeError("VectorStore is empty. Please add vectors before searching.")

        query = np.ascontiguousarray(query_vector, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if self.index_type == "FlatIP":
            faiss.normalize_L2(query)

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query, k)
        return distances[0], indices[0]

    def search_with_metadata(
        self, query_vector: np.ndarray, k: int = 5
    ) -> list[dict[str, Any]]:
        """Searches Top-K nearest items and returns enriched metadata dicts with scores."""
        distances, indices = self.search(query_vector, k=k)

        results: list[dict[str, Any]] = []
        for dist, idx in zip(distances, indices):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(dist)
            item["rank"] = len(results) + 1
            results.append(item)
        return results

    def save_db(self, folder_path: Union[str, Path]) -> None:
        """Saves FAISS binary index and metadata JSON to disk."""
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        index_path = folder / "index.faiss"
        meta_path = folder / "metadata.json"

        faiss.write_index(self.index, str(index_path))
        print(f"[VectorStore] Saved FAISS index ({self.index.ntotal} vectors) -> {index_path}")

        payload = {
            "dim": self.dim,
            "index_type": self.index_type,
            "model_name": self.model_name,
            "count": len(self.metadata),
            "items": self.metadata,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[VectorStore] Saved metadata ({len(self.metadata)} items) -> {meta_path}")

    @classmethod
    def load_db(cls, folder_path: Union[str, Path]) -> VectorStore:
        """Loads FAISS index and metadata JSON from a saved directory."""
        folder = Path(folder_path)
        index_path = folder / "index.faiss"
        meta_path = folder / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata JSON not found at: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        dim: int = data.get("dim", 512)
        index_type: str = data.get("index_type", "FlatIP")
        model_name: str = data.get("model_name", "clip")
        items: list[dict[str, Any]] = data.get("items", [])

        index = faiss.read_index(str(index_path))

        store = cls.__new__(cls)
        store.dim = dim
        store.index_type = index_type
        store.model_name = model_name
        store.index = index
        store.metadata = items

        print(f"[VectorStore] Successfully loaded {index.ntotal} vectors ({dim}D, {model_name}) from {folder}")
        return store

    @property
    def count(self) -> int:
        return self.index.ntotal

    def __len__(self) -> int:
        return self.index.ntotal


class DualModelVectorStore:
    """Unified container managing parallel CLIP and SigLIP stores for a given dataset."""

    def __init__(self, clip_store: Optional[VectorStore] = None, siglip_store: Optional[VectorStore] = None):
        self.clip = clip_store
        self.siglip = siglip_store

    @classmethod
    def load(cls, base_dir: Union[str, Path]) -> DualModelVectorStore:
        """Loads both CLIP and SigLIP sub-stores from base_dir/clip and base_dir/siglip."""
        base = Path(base_dir)
        clip_dir = base / "clip"
        siglip_dir = base / "siglip"

        clip_store = VectorStore.load_db(clip_dir) if clip_dir.exists() else None
        siglip_store = VectorStore.load_db(siglip_dir) if siglip_dir.exists() else None

        return cls(clip_store=clip_store, siglip_store=siglip_store)

    def save(self, base_dir: Union[str, Path]) -> None:
        """Saves both stores under base_dir/clip and base_dir/siglip."""
        base = Path(base_dir)
        if self.clip is not None:
            self.clip.save_db(base / "clip")
        if self.siglip is not None:
            self.siglip.save_db(base / "siglip")
