"""Comparative Search Engine: Direct Side-by-Side Dual-Model Retrieval (CLIP vs SigLIP).

Provides:
1. Unified querying across dual FAISS vector stores (512D CLIP & 768D SigLIP).
2. Query-level comparative benchmark metrics (latency, score separation margin, rank agreement).
3. Support for both COCO Karpathy images and NewsSeek video keyframes.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..storage.vector_store import VectorStore, DualModelVectorStore
from ..models.clip_wrapper import CLIPWrapper
from ..models.siglip_wrapper import SigLIPWrapper


class ComparativeSearchEngine:
    """Orchestrates comparative retrieval and query-level benchmarking between CLIP and SigLIP."""

    def __init__(
        self,
        clip_store: Optional[VectorStore] = None,
        siglip_store: Optional[VectorStore] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_store = clip_store
        self.siglip_store = siglip_store
        
        # Lazy loaded model wrappers
        self._clip_model: Optional[CLIPWrapper] = None
        self._siglip_model: Optional[SigLIPWrapper] = None

    @property
    def clip_model(self) -> CLIPWrapper:
        if self._clip_model is None:
            self._clip_model = CLIPWrapper(device=self.device)
        return self._clip_model

    @property
    def siglip_model(self) -> SigLIPWrapper:
        if self._siglip_model is None:
            self._siglip_model = SigLIPWrapper(device=self.device)
        return self._siglip_model

    def set_stores(self, clip_store: Optional[VectorStore], siglip_store: Optional[VectorStore]) -> None:
        """Dynamically switches active vector stores (e.g., COCO vs NewsSeek)."""
        self.clip_store = clip_store
        self.siglip_store = siglip_store

    def search_text(
        self,
        query_text: str,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Performs parallel text-to-visual retrieval for both models with live benchmark metrics.
        
        Args:
            query_text: Natural language query (multilingual supported).
            k: Top-K items to retrieve per model.
            
        Returns:
            Dictionary containing:
            - 'clip_results': List of enriched metadata dicts with 'rank', 'score'
            - 'siglip_results': List of enriched metadata dicts with 'rank', 'score'
            - 'benchmark': Comparative query-level metrics dict
        """
        query_text = query_text.strip()
        if not query_text:
            raise ValueError("Query text cannot be empty.")

        # 1. CLIP Retrieval
        clip_res = []
        clip_metrics = {}
        if self.clip_store is not None and self.clip_store.count > 0:
            t0 = time.perf_counter()
            with torch.no_grad():
                clip_feat = self.clip_model.get_text_embeddings([query_text]).cpu().numpy()
            t_enc = (time.perf_counter() - t0) * 1000.0  # ms

            t0 = time.perf_counter()
            clip_res = self.clip_store.search_with_metadata(clip_feat[0], k=k)
            t_search = (time.perf_counter() - t0) * 1000.0  # ms

            scores = [r["score"] for r in clip_res]
            top1 = scores[0] if scores else 0.0
            top_rest = np.mean(scores[1:5]) if len(scores) > 1 else top1
            margin = float(top1 - top_rest)

            clip_metrics = {
                "encode_latency_ms": round(t_enc, 2),
                "search_latency_ms": round(t_search, 2),
                "total_latency_ms": round(t_enc + t_search, 2),
                "top1_score": round(float(top1), 4),
                "mean_top5_score": round(float(np.mean(scores[:5])), 4) if scores else 0.0,
                "separation_margin": round(margin, 4),
            }

        # 2. SigLIP Retrieval
        siglip_res = []
        siglip_metrics = {}
        if self.siglip_store is not None and self.siglip_store.count > 0:
            t0 = time.perf_counter()
            with torch.no_grad():
                siglip_feat = self.siglip_model.get_text_embeddings([query_text]).cpu().numpy()
            t_enc = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            siglip_res = self.siglip_store.search_with_metadata(siglip_feat[0], k=k)
            t_search = (time.perf_counter() - t0) * 1000.0

            scores = [r["score"] for r in siglip_res]
            top1 = scores[0] if scores else 0.0
            top_rest = np.mean(scores[1:5]) if len(scores) > 1 else top1
            margin = float(top1 - top_rest)

            siglip_metrics = {
                "encode_latency_ms": round(t_enc, 2),
                "search_latency_ms": round(t_search, 2),
                "total_latency_ms": round(t_enc + t_search, 2),
                "top1_score": round(float(top1), 4),
                "mean_top5_score": round(float(np.mean(scores[:5])), 4) if scores else 0.0,
                "separation_margin": round(margin, 4),
            }

        # 3. Comparative Inter-Model Agreement
        agreement_metrics = self._compute_agreement(clip_res, siglip_res, k=k)

        return {
            "query": query_text,
            "query_type": "text",
            "k": k,
            "clip_results": clip_res,
            "siglip_results": siglip_res,
            "benchmark": {
                "clip": clip_metrics,
                "siglip": siglip_metrics,
                "agreement": agreement_metrics,
            },
        }

    def search_image(
        self,
        query_image: Image.Image,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Performs parallel visual-to-visual retrieval for both models."""
        if not isinstance(query_image, Image.Image):
            raise TypeError("query_image must be a PIL Image instance.")

        # 1. CLIP Retrieval
        clip_res = []
        clip_metrics = {}
        if self.clip_store is not None and self.clip_store.count > 0:
            t0 = time.perf_counter()
            with torch.no_grad():
                clip_feat = self.clip_model.get_image_embeddings([query_image]).cpu().numpy()
            t_enc = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            clip_res = self.clip_store.search_with_metadata(clip_feat[0], k=k)
            t_search = (time.perf_counter() - t0) * 1000.0

            scores = [r["score"] for r in clip_res]
            top1 = scores[0] if scores else 0.0
            top_rest = np.mean(scores[1:5]) if len(scores) > 1 else top1
            clip_metrics = {
                "encode_latency_ms": round(t_enc, 2),
                "search_latency_ms": round(t_search, 2),
                "total_latency_ms": round(t_enc + t_search, 2),
                "top1_score": round(float(top1), 4),
                "separation_margin": round(float(top1 - top_rest), 4),
            }

        # 2. SigLIP Retrieval
        siglip_res = []
        siglip_metrics = {}
        if self.siglip_store is not None and self.siglip_store.count > 0:
            t0 = time.perf_counter()
            with torch.no_grad():
                siglip_feat = self.siglip_model.get_image_embeddings([query_image]).cpu().numpy()
            t_enc = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            siglip_res = self.siglip_store.search_with_metadata(siglip_feat[0], k=k)
            t_search = (time.perf_counter() - t0) * 1000.0

            scores = [r["score"] for r in siglip_res]
            top1 = scores[0] if scores else 0.0
            top_rest = np.mean(scores[1:5]) if len(scores) > 1 else top1
            siglip_metrics = {
                "encode_latency_ms": round(t_enc, 2),
                "search_latency_ms": round(t_search, 2),
                "total_latency_ms": round(t_enc + t_search, 2),
                "top1_score": round(float(top1), 4),
                "separation_margin": round(float(top1 - top_rest), 4),
            }

        agreement_metrics = self._compute_agreement(clip_res, siglip_res, k=k)

        return {
            "query": "[User Uploaded Image]",
            "query_type": "image",
            "k": k,
            "clip_results": clip_res,
            "siglip_results": siglip_res,
            "benchmark": {
                "clip": clip_metrics,
                "siglip": siglip_metrics,
                "agreement": agreement_metrics,
            },
        }

    def _compute_agreement(
        self, clip_res: List[Dict[str, Any]], siglip_res: List[Dict[str, Any]], k: int = 10
    ) -> Dict[str, Any]:
        """Calculates Top-K Jaccard similarity and rank overlap between CLIP and SigLIP."""
        clip_ids = [r.get("id") or r.get("filename") or r.get("relative_path") for r in clip_res]
        siglip_ids = [r.get("id") or r.get("filename") or r.get("relative_path") for r in siglip_res]

        set_clip = set(filter(None, clip_ids))
        set_siglip = set(filter(None, siglip_ids))

        if not set_clip or not set_siglip:
            return {"jaccard_overlap_pct": 0.0, "intersection_count": 0, "rank_1_match": False}

        intersection = set_clip.intersection(set_siglip)
        union = set_clip.union(set_siglip)
        jaccard = (len(intersection) / len(union)) * 100.0 if union else 0.0

        rank_1_match = bool(clip_ids and siglip_ids and clip_ids[0] == siglip_ids[0])

        return {
            "jaccard_overlap_pct": round(float(jaccard), 2),
            "intersection_count": len(intersection),
            "rank_1_match": rank_1_match,
        }
