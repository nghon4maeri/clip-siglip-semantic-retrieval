"""Video-level Semantic Search Engine.

Supports:
1. Max-Pooling & Top-K pooling aggregation across video keyframes.
2. Direct CLIP, SigLIP, or Reciprocal Rank Fusion (RRF) ensemble retrieval.
3. Near-duplicate result filtering via BLAS matrix prefiltering.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F

from .fusion import rrf
from .similarity_filter import filter_similar_results


class VideoSearchEngine:
    def __init__(
        self,
        clip_wrapper=None,
        siglip_wrapper=None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            clip_wrapper: Instance of CLIPWrapper (optional).
            siglip_wrapper: Instance of SigLIPWrapper (optional).
            device: 'cuda' or 'cpu'.
        """
        self.clip = clip_wrapper
        self.siglip = siglip_wrapper
        self.device = device

        # Video catalog metadata
        # video_id -> { 'video_id': ..., 'title': ..., 'language': ..., 'keyframes': [...] }
        self.catalog: Dict[str, Dict[str, Any]] = {}
        
        # Precomputed video embeddings:
        # model_name ('clip' | 'siglip') -> {
        #    'frame_ids': list of str,
        #    'frame_to_video': list of video_id,
        #    'frame_metadata': list of dict,
        #    'matrix': np.ndarray of shape (N_frames, D) (normalized)
        # }
        self.indexes: Dict[str, Dict[str, Any]] = {}

    def build_index(
        self,
        manifest_entries: List[Dict[str, Any]],
        batch_size: int = 64,
        models_to_index: Optional[List[str]] = None,
    ):
        """Builds keyframe embeddings index for all videos in manifest.
        
        Args:
            manifest_entries: List of video manifest dicts, each with:
                'video_id', 'title', 'language', 'keyframes' (list of dicts with 'image_path' / 'full_path')
            batch_size: Extraction batch size.
            models_to_index: List containing 'clip', 'siglip', or both.
        """
        if models_to_index is None:
            models_to_index = []
            if self.clip is not None:
                models_to_index.append("clip")
            if self.siglip is not None:
                models_to_index.append("siglip")

        # Flatten all keyframes across videos
        flat_frames = []
        for v in manifest_entries:
            vid = v["video_id"]
            self.catalog[vid] = v
            for kf in v.get("keyframes", []):
                flat_frames.append({
                    "video_id": vid,
                    "frame_number": kf.get("frame_number", 0),
                    "timestamp_sec": kf.get("timestamp_sec", 0.0),
                    "image_path": kf.get("image_path") or kf.get("relative_path"),
                    "full_path": kf.get("full_path") or kf.get("image_path"),
                    "frame_id": f"{vid}_{kf.get('frame_number', 0):06d}",
                })

        print(f"[VideoSearchEngine] Indexing {len(flat_frames)} keyframes from {len(manifest_entries)} videos.")

        for m_name in models_to_index:
            model = self.clip if m_name == "clip" else self.siglip
            if model is None:
                continue

            print(f"--> Extracting {m_name.upper()} keyframe embeddings...")
            all_vecs = []
            
            # Batch extraction
            from PIL import Image
            for i in range(0, len(flat_frames), batch_size):
                batch_items = flat_frames[i : i + batch_size]
                batch_images = []
                for item in batch_items:
                    img_p = item["full_path"]
                    try:
                        img = Image.open(img_p).convert("RGB")
                    except Exception as e:
                        # Fallback dummy image if missing
                        img = Image.new("RGB", (224, 224), (0, 0, 0))
                    batch_images.append(img)

                with torch.no_grad():
                    vecs = model.encode_images(batch_images)
                    vecs = vecs / vecs.norm(dim=-1, keepdim=True)
                    all_vecs.append(vecs.cpu().numpy().astype(np.float32))

            matrix = np.concatenate(all_vecs, axis=0) if all_vecs else np.empty((0, 512))
            
            self.indexes[m_name] = {
                "frame_ids": [item["frame_id"] for item in flat_frames],
                "frame_to_video": [item["video_id"] for item in flat_frames],
                "frame_metadata": flat_frames,
                "matrix": matrix,
            }
            print(f"--> {m_name.upper()} index ready: {matrix.shape}")

    def search_frames(
        self,
        query: str,
        model_name: str = "siglip",
        top_k: int = 50,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Frame-level retrieval for a text query."""
        if model_name not in self.indexes:
            raise ValueError(f"Model index '{model_name}' not built.")

        idx_data = self.indexes[model_name]
        model = self.clip if model_name == "clip" else self.siglip

        # Encode text query
        with torch.no_grad():
            q_vec = model.encode_text(query)
            q_vec = q_vec / q_vec.norm(dim=-1, keepdim=True)
            q_np = q_vec.cpu().numpy().astype(np.float32).reshape(-1)

        sims = idx_data["matrix"] @ q_np
        top_indices = np.argsort(-sims)[:top_k]

        results = []
        for idx in top_indices:
            frame_meta = dict(idx_data["frame_metadata"][idx])
            vid = frame_meta["video_id"]
            video_meta = self.catalog.get(vid, {})
            results.append({
                "id": frame_meta["frame_id"],
                "frame_id": frame_meta["frame_id"],
                "video_id": vid,
                "score": float(sims[idx]),
                "frame_number": frame_meta["frame_number"],
                "timestamp_sec": frame_meta["timestamp_sec"],
                "image_path": frame_meta["image_path"],
                "title": video_meta.get("title", ""),
                "language": video_meta.get("language", ""),
                "source_name": video_meta.get("source_name", ""),
            })

        if similarity_threshold is not None:
            # Near-duplicate suppression
            vec_dict = {
                res["frame_id"]: idx_data["matrix"][top_indices[i]]
                for i, res in enumerate(results)
            }
            results = filter_similar_results(results, vec_dict, threshold=similarity_threshold)

        return results

    def search_videos(
        self,
        query: str,
        model_name: str = "siglip",
        aggregation: str = "max",  # 'max' or 'top3_avg'
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Video-level retrieval using Max-Pooling / Top-K aggregation across keyframes.
        
        Formula for max pooling:
            Score(V_i, q) = max_{k in Keyframes(V_i)} ( cosine_sim(k, q) )
        """
        frame_hits = self.search_frames(query, model_name=model_name, top_k=len(self.indexes[model_name]["frame_ids"]))
        
        # Group by video_id
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for hit in frame_hits:
            vid = hit["video_id"]
            grouped.setdefault(vid, []).append(hit)

        video_results = []
        for vid, hits in grouped.items():
            scores = [h["score"] for h in hits]
            if aggregation == "max":
                video_score = max(scores)
                best_hit = hits[0]  # Already sorted by score descending
            elif aggregation == "top3_avg":
                top3 = sorted(scores, reverse=True)[:3]
                video_score = sum(top3) / len(top3)
                best_hit = hits[0]
            else:
                video_score = max(scores)
                best_hit = hits[0]

            v_meta = self.catalog.get(vid, {})
            video_results.append({
                "id": vid,
                "video_id": vid,
                "video_score": float(video_score),
                "best_frame_number": best_hit["frame_number"],
                "best_timestamp_sec": best_hit["timestamp_sec"],
                "best_image_path": best_hit["image_path"],
                "title": v_meta.get("title", ""),
                "language": v_meta.get("language", ""),
                "source_name": v_meta.get("source_name", ""),
                "duration_seconds": v_meta.get("duration_seconds", 0.0),
                "num_keyframes": len(hits),
            })

        video_results.sort(key=lambda x: x["video_score"], reverse=True)
        return video_results[:top_k]

    def search_videos_rrf(
        self,
        query: str,
        k: int = 60,
        weights: Optional[List[float]] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Ensemble Video Retrieval fusing CLIP and SigLIP rankings via RRF."""
        if "clip" not in self.indexes or "siglip" not in self.indexes:
            raise RuntimeError("Both 'clip' and 'siglip' indexes must be built for RRF search.")

        clip_hits = self.search_videos(query, model_name="clip", top_k=top_k * 2)
        siglip_hits = self.search_videos(query, model_name="siglip", top_k=top_k * 2)

        fused = rrf([clip_hits, siglip_hits], key="video_id", k=k, weights=weights)
        return fused[:top_k]
