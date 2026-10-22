"""Unified Comparative Search Engine (CLIP vs SigLIP).

Streamlined to ONE canonical retrieval algorithm designed specifically for
controlled, rigorous benchmarking across diverse query taxonomies:

1. Canonical Pipeline:
   - Parallel text encoding (CLIP 512D unit-norm vs SigLIP 768D unit-norm).
   - Exact Inner Product similarity search via FAISS IndexFlatIP.
   - Near-duplicate temporal suppression (optional, for video keyframes).

2. Automated Query Taxonomy Profiler:
   - Language classification: English vs Vietnamese / Multilingual.
   - Length / Granularity: Short Entity (<= 3 words), Phrase (4-8 words), Complex Compositional (> 8 words).
   - Relational / Spatial cues: Prepositions, action verbs, color attributes.

3. Query-Level Benchmark Telemetry:
   - Inference latency (encode + search ms).
   - Score distribution & separation margin (Top-1 vs Mean Top-2..5).
   - Inter-model rank agreement (Jaccard Overlap @ K, Rank-1 agreement).
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..storage.vector_store import VectorStore
from ..models.clip_wrapper import CLIPWrapper
from ..models.siglip_wrapper import SigLIPWrapper


def profile_query(query_text: str) -> Dict[str, Any]:
    """Analyzes query linguistic features to support benchmark categorization."""
    text = query_text.strip()
    words = text.split()
    word_count = len(words)

    # Detect language: Vietnamese diacritics check
    vn_diacritics = re.compile(
        r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
        re.IGNORECASE,
    )
    is_vietnamese = bool(vn_diacritics.search(text))
    language = "vi" if is_vietnamese else ("en" if text.isascii() else "multilingual")

    # Granularity classification
    if word_count <= 3:
        granularity = "short_entity"
    elif word_count <= 8:
        granularity = "phrase"
    else:
        granularity = "complex_compositional"

    # Semantic attributes presence
    relational_cues = ["on", "in", "under", "next to", "near", "with", "behind", "front", "trên", "dưới", "trong", "cạnh"]
    has_relational = any(cue in text.lower().split() for cue in relational_cues)

    return {
        "word_count": word_count,
        "char_count": len(text),
        "language": language,
        "granularity": granularity,
        "has_relational_cue": has_relational,
    }


class ComparativeSearchEngine:
    """The canonical dual-model comparative search engine for CLIP vs SigLIP."""

    def __init__(
        self,
        clip_store: Optional[VectorStore] = None,
        siglip_store: Optional[VectorStore] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_store = clip_store
        self.siglip_store = siglip_store

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
        self.clip_store = clip_store
        self.siglip_store = siglip_store

    def search_text(
        self,
        query_text: str,
        k: int = 10,
        filter_duplicates: bool = False,
        min_time_gap_sec: float = 1.5,
    ) -> Dict[str, Any]:
        """Canonical semantic retrieval comparing CLIP and SigLIP on the exact same query.

        Args:
            query_text: Input query string.
            k: Top-K results to retrieve.
            filter_duplicates: Whether to apply temporal duplicate suppression (for video keyframes).
            min_time_gap_sec: Minimum temporal distance for frames from the same video.
        """
        query_text = query_text.strip()
        if not query_text:
            raise ValueError("Query text cannot be empty.")

        query_profile = profile_query(query_text)
        candidate_k = max(k * 3, 30) if filter_duplicates else k

        # 1. CLIP Retrieval (512D)
        clip_res = []
        clip_metrics = {}
        if self.clip_store is not None and self.clip_store.count > 0:
            t0 = time.perf_counter()
            with torch.no_grad():
                clip_feat = self.clip_model.get_text_embeddings([query_text]).cpu().numpy()
            t_enc = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            clip_res = self.clip_store.search_with_metadata(clip_feat[0], k=candidate_k)
            t_search = (time.perf_counter() - t0) * 1000.0

            if filter_duplicates:
                clip_res = self._filter_near_duplicates(clip_res, min_time_gap_sec)
            clip_res = clip_res[:k]

            scores = [r["score"] for r in clip_res]
            top1 = scores[0] if scores else 0.0
            top_rest = np.mean(scores[1:5]) if len(scores) > 1 else top1

            clip_metrics = {
                "encode_latency_ms": round(t_enc, 2),
                "search_latency_ms": round(t_search, 2),
                "total_latency_ms": round(t_enc + t_search, 2),
                "top1_score": round(float(top1), 4),
                "mean_top5_score": round(float(np.mean(scores[:5])), 4) if scores else 0.0,
                "separation_margin": round(float(top1 - top_rest), 4),
            }

        # 2. SigLIP Retrieval (768D)
        siglip_res = []
        siglip_metrics = {}
        if self.siglip_store is not None and self.siglip_store.count > 0:
            t0 = time.perf_counter()
            with torch.no_grad():
                siglip_feat = self.siglip_model.get_text_embeddings([query_text]).cpu().numpy()
            t_enc = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            siglip_res = self.siglip_store.search_with_metadata(siglip_feat[0], k=candidate_k)
            t_search = (time.perf_counter() - t0) * 1000.0

            if filter_duplicates:
                siglip_res = self._filter_near_duplicates(siglip_res, min_time_gap_sec)
            siglip_res = siglip_res[:k]

            scores = [r["score"] for r in siglip_res]
            top1 = scores[0] if scores else 0.0
            top_rest = np.mean(scores[1:5]) if len(scores) > 1 else top1

            siglip_metrics = {
                "encode_latency_ms": round(t_enc, 2),
                "search_latency_ms": round(t_search, 2),
                "total_latency_ms": round(t_enc + t_search, 2),
                "top1_score": round(float(top1), 4),
                "mean_top5_score": round(float(np.mean(scores[:5])), 4) if scores else 0.0,
                "separation_margin": round(float(top1 - top_rest), 4),
            }

        # 3. Inter-Model Agreement
        agreement = self._compute_agreement(clip_res, siglip_res, k=k)

        # 4. Grouped View
        video_groups = self._aggregate_video_groups(siglip_res if siglip_res else clip_res)

        return {
            "query": query_text,
            "query_type": "text",
            "query_profile": query_profile,
            "k": k,
            "clip_results": clip_res,
            "siglip_results": siglip_res,
            "video_groups": video_groups,
            "benchmark": {
                "clip": clip_metrics,
                "siglip": siglip_metrics,
                "agreement": agreement,
                "query_profile": query_profile,
            },
        }

    def search_image(
        self,
        query_image: Image.Image,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Canonical visual-to-visual search for reverse image retrieval."""
        if not isinstance(query_image, Image.Image):
            raise TypeError("query_image must be a PIL Image instance.")

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
            clip_metrics = {
                "encode_latency_ms": round(t_enc, 2),
                "search_latency_ms": round(t_search, 2),
                "total_latency_ms": round(t_enc + t_search, 2),
                "top1_score": round(float(scores[0]), 4) if scores else 0.0,
            }

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
            siglip_metrics = {
                "encode_latency_ms": round(t_enc, 2),
                "search_latency_ms": round(t_search, 2),
                "total_latency_ms": round(t_enc + t_search, 2),
                "top1_score": round(float(scores[0]), 4) if scores else 0.0,
            }

        agreement = self._compute_agreement(clip_res, siglip_res, k=k)
        video_groups = self._aggregate_video_groups(siglip_res if siglip_res else clip_res)

        return {
            "query": "[Uploaded Image]",
            "query_type": "image",
            "k": k,
            "clip_results": clip_res,
            "siglip_results": siglip_res,
            "video_groups": video_groups,
            "benchmark": {
                "clip": clip_metrics,
                "siglip": siglip_metrics,
                "agreement": agreement,
            },
        }

    def _filter_near_duplicates(self, items: List[Dict[str, Any]], min_time_gap_sec: float) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        last_time_by_video: Dict[str, float] = {}

        for item in items:
            vid = item.get("video_id")
            if not vid:
                kept.append(item)
                continue

            ts = float(item.get("timestamp_sec", 0.0))
            if vid in last_time_by_video:
                if abs(ts - last_time_by_video[vid]) < min_time_gap_sec:
                    continue

            kept.append(item)
            last_time_by_video[vid] = ts

        for idx, item in enumerate(kept, start=1):
            item["rank"] = idx
        return kept

    def _aggregate_video_groups(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in items:
            vid = item.get("video_id") or item.get("id", "single_item")
            grouped.setdefault(vid, []).append(item)

        video_list = []
        for vid, kfs in grouped.items():
            kfs.sort(key=lambda x: float(x.get("timestamp_sec", 0.0)))
            max_score = max(float(x.get("score", 0.0)) for x in kfs)
            first = kfs[0]
            video_list.append({
                "video_id": vid,
                "video_title": first.get("video_title", vid),
                "language": first.get("language", "en"),
                "max_score": round(max_score, 4),
                "frame_count": len(kfs),
                "keyframes": kfs,
            })

        video_list.sort(key=lambda v: v["max_score"], reverse=True)
        return video_list

    def _compute_agreement(
        self, clip_res: List[Dict[str, Any]], siglip_res: List[Dict[str, Any]], k: int = 10
    ) -> Dict[str, Any]:
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
