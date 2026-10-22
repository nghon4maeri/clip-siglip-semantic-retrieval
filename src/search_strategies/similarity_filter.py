"""Greedy Near-Duplicate Suppression and Visual Diversity Filter.

Adapted from the reference multimodal retrieval architecture:
- Suppresses repetitive adjacent keyframes belonging to the same visual scene.
- Operates on pre-computed embeddings or checks video_id + temporal proximity.
- Guarantees diversity in Top-K result presentations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np


def filter_near_duplicate_frames(
    results: List[Dict[str, Any]],
    embeddings_map: Optional[Dict[str, np.ndarray]] = None,
    *,
    threshold: float = 0.88,
    min_time_gap_sec: float = 1.5,
    key: str = "id",
) -> List[Dict[str, Any]]:
    """Filters near-duplicate keyframes to maximize visual entropy and diversity.

    A candidate is marked as duplicate if:
    1. Both frames belong to the SAME video_id, AND
    2. EITHER their cosine similarity exceeds `threshold` (if embeddings provided),
       OR their timestamp difference is less than `min_time_gap_sec`.

    Args:
        results: Ranked list of search result dictionaries.
        embeddings_map: Optional dict mapping item identity to normalized float32 vector.
        threshold: Cosine similarity cutoff (default 0.88).
        min_time_gap_sec: Minimum temporal distance (in seconds) between retained frames of same video.
        key: Identity key.

    Returns:
        Filtered list of diverse result items.
    """
    if not results:
        return []

    kept: List[Dict[str, Any]] = []
    kept_by_video: Dict[str, List[Dict[str, Any]]] = {}

    for item in results:
        vid = item.get("video_id")
        # For non-video datasets (like COCO), fallback to image-level vector similarity
        if not vid:
            if embeddings_map and key in item and str(item[key]) in embeddings_map:
                vec = embeddings_map[str(item[key])]
                is_dup = False
                for prev in kept:
                    prev_id = str(prev.get(key, ""))
                    if prev_id in embeddings_map:
                        prev_vec = embeddings_map[prev_id]
                        sim = float(np.dot(vec, prev_vec))
                        if sim >= threshold:
                            is_dup = True
                            break
                if not is_dup:
                    kept.append(item)
            else:
                kept.append(item)
            continue

        # Video-level near duplicate checking
        prev_video_frames = kept_by_video.get(vid, [])
        is_duplicate = False

        item_ts = float(item.get("timestamp_sec", 0.0))
        item_id = str(item.get(key, ""))
        item_vec = embeddings_map.get(item_id) if embeddings_map else None

        for prev in prev_video_frames:
            prev_ts = float(prev.get("timestamp_sec", 0.0))
            time_diff = abs(item_ts - prev_ts)

            # Check 1: Temporal proximity
            if time_diff < min_time_gap_sec:
                is_duplicate = True
                break

            # Check 2: Cosine similarity
            if item_vec is not None:
                prev_id = str(prev.get(key, ""))
                prev_vec = embeddings_map.get(prev_id)
                if prev_vec is not None:
                    sim = float(np.dot(item_vec, prev_vec))
                    if sim >= threshold:
                        is_duplicate = True
                        break

        if not is_duplicate:
            kept.append(item)
            kept_by_video.setdefault(vid, []).append(item)

    # Re-assign ranks
    for rank_idx, item in enumerate(kept, start=1):
        item["rank"] = rank_idx

    return kept
