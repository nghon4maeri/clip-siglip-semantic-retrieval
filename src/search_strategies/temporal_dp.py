"""Dynamic Programming (DP) Temporal Sequence Search Strategy.

Adapted from the reference repository's DuyTemporalSearch & _duy_temporal_core:
- Executes multi-step video event sequence search.
- Solves optimal frame chain alignment using monotonic timestamp DP.
- Respects minimum time offsets or bounded interval windows [min_sec, max_sec].
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Sequence
import numpy as np


def match_temporal_sequence(
    event_rankings: Sequence[List[Dict[str, Any]]],
    temporal_offsets_sec: Optional[Sequence[float]] = None,
    interval_bounds_sec: Optional[Tuple[float, float]] = None,
    event_weights: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Finds optimal video-level sequential event chains via Dynamic Programming.

    Args:
        event_rankings: List of ranking lists, one per sequential event query.
        temporal_offsets_sec: Required minimum seconds after previous event for each step.
                              (e.g. [0.0, 5.0, 10.0]).
        interval_bounds_sec: Optional (min_interval_sec, max_interval_sec) span between start and end.
        event_weights: Relative importance weights for each event step.

    Returns:
        List of fused chain results sorted by confidence, with 'steps', 'confidence', and 'evidence'.
    """
    num_events = len(event_rankings)
    if num_events == 0:
        return []
    if num_events == 1:
        # Degenerate case: single query
        return event_rankings[0]

    if temporal_offsets_sec is None:
        temporal_offsets_sec = [0.0] * num_events
    if len(temporal_offsets_sec) != num_events:
        temporal_offsets_sec = [float(temporal_offsets_sec[i]) if i < len(temporal_offsets_sec) else 0.0 for i in range(num_events)]

    if event_weights is None:
        event_weights = [1.0] * num_events
    weights = [float(w) for w in event_weights]
    weight_total = sum(weights) or 1.0

    # Group frame hits by video_id and event level
    levels_by_video: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    for level, ranking in enumerate(event_rankings):
        for hit in ranking:
            vid = hit.get("video_id")
            if not vid:
                continue
            levels_by_video.setdefault(vid, {}).setdefault(level, []).append(hit)

    results: List[Dict[str, Any]] = []

    for vid, levels in levels_by_video.items():
        # Only evaluate videos that contain candidates for all requested event levels
        if len(levels) != num_events:
            continue

        # Sort frames at each level by timestamp
        for lvl in range(num_events):
            levels[lvl].sort(key=lambda f: float(f.get("timestamp_sec", 0.0)))

        if interval_bounds_sec is None:
            chains = _dp_unbounded(levels, temporal_offsets_sec, weights, num_events)
        else:
            min_sec, max_sec = interval_bounds_sec
            chains = _dp_interval(levels, temporal_offsets_sec, min_sec, max_sec, weights, num_events)

        for chain in chains:
            last_frame = dict(chain["path"][-1])
            confidence = round(chain["score"] / weight_total, 4)
            last_frame["confidence"] = confidence
            last_frame["score"] = confidence
            last_frame["steps"] = chain["path"]
            last_frame["evidence"] = chain["path"]
            last_frame["is_temporal_chain"] = True
            results.append(last_frame)

    results.sort(key=lambda x: x["confidence"], reverse=True)
    for idx, item in enumerate(results, start=1):
        item["rank"] = idx

    return results


def _dp_unbounded(
    levels: Dict[int, List[Dict[str, Any]]],
    offsets: Sequence[float],
    weights: Sequence[float],
    num_events: int,
) -> List[Dict[str, Any]]:
    """Monotonic DP chain for unbounded intervals."""
    # Level 0 initialization
    dp: List[Optional[Dict[str, Any]]] = [
        {"score": float(f.get("score", 0.0)) * weights[0], "path": [f]}
        for f in levels[0]
    ]
    prev_frames = levels[0]

    for level in range(1, num_events):
        curr_dp: List[Optional[Dict[str, Any]]] = []
        best: Optional[Dict[str, Any]] = None
        ptr = 0
        min_offset = offsets[level]

        for frame in levels[level]:
            curr_ts = float(frame.get("timestamp_sec", 0.0))
            threshold = curr_ts - min_offset

            while ptr < len(prev_frames) and float(prev_frames[ptr].get("timestamp_sec", 0.0)) <= threshold:
                candidate = dp[ptr]
                if candidate is not None and (best is None or candidate["score"] > best["score"]):
                    best = candidate
                ptr += 1

            if best is not None:
                step_score = float(frame.get("score", 0.0)) * weights[level]
                curr_dp.append({
                    "score": best["score"] + step_score,
                    "path": best["path"] + [frame],
                })
            else:
                curr_dp.append(None)

        prev_frames = levels[level]
        dp = curr_dp

    return [entry for entry in dp if entry is not None]


def _dp_interval(
    levels: Dict[int, List[Dict[str, Any]]],
    offsets: Sequence[float],
    min_span_sec: float,
    max_span_sec: float,
    weights: Sequence[float],
    num_events: int,
) -> List[Dict[str, Any]]:
    """Bounded span DP: total duration between start and end must fall in [min_span, max_span]."""
    last_level = num_events - 1
    chains: List[Dict[str, Any]] = []

    for start_frame in levels[0]:
        start_ts = float(start_frame.get("timestamp_sec", 0.0))
        frontier = [{"score": float(start_frame.get("score", 0.0)) * weights[0], "path": [start_frame]}]
        frontier_frames = [start_frame]

        for level in range(1, last_level):
            next_frontier = []
            next_frames = []
            best = None
            ptr = 0
            min_offset = offsets[level]

            for frame in levels[level]:
                curr_ts = float(frame.get("timestamp_sec", 0.0))
                threshold = curr_ts - min_offset

                while ptr < len(frontier_frames) and float(frontier_frames[ptr].get("timestamp_sec", 0.0)) <= threshold:
                    cand = frontier[ptr]
                    if best is None or cand["score"] > best["score"]:
                        best = cand
                    ptr += 1

                if best is not None:
                    next_frames.append(frame)
                    next_frontier.append({
                        "score": best["score"] + float(frame.get("score", 0.0)) * weights[level],
                        "path": best["path"] + [frame],
                    })

            frontier = next_frontier
            frontier_frames = next_frames
            if not frontier:
                break

        if not frontier:
            continue

        # Connect to last level
        ptr = 0
        best = None
        min_offset = offsets[last_level]

        for end_frame in levels[last_level]:
            end_ts = float(end_frame.get("timestamp_sec", 0.0))
            span = end_ts - start_ts
            if span > max_span_sec:
                break

            threshold = end_ts - min_offset
            while ptr < len(frontier_frames) and float(frontier_frames[ptr].get("timestamp_sec", 0.0)) <= threshold:
                cand = frontier[ptr]
                if best is None or cand["score"] > best["score"]:
                    best = cand
                ptr += 1

            if span >= min_span_sec and best is not None:
                chains.append({
                    "score": best["score"] + float(end_frame.get("score", 0.0)) * weights[last_level],
                    "path": best["path"] + [end_frame],
                })

    return chains
