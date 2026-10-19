"""Scene-aware keyframe selection policies.
Supports 'tiered' (duration buckets) and 'linear' (proportional sampling) strategies.
"""
from __future__ import annotations

import math
import json
from collections.abc import Iterable
from pathlib import Path


KEYFRAME_STRATEGIES = ("tiered", "linear")


def selection_metadata(
    *, strategy: str, keyframes_per_second: float = 0.2,
    min_keyframes_per_scene: int = 1, max_keyframes_per_scene: int = 20,
) -> dict:
    return {
        "strategy": strategy,
        "keyframes_per_second": keyframes_per_second,
        "min_keyframes_per_scene": min_keyframes_per_scene,
        "max_keyframes_per_scene": max_keyframes_per_scene,
    }


def keyframe_count(
    duration_s: float,
    *,
    strategy: str = "tiered",
    keyframes_per_second: float = 0.2,
    min_keyframes_per_scene: int = 1,
    max_keyframes_per_scene: int = 20,
) -> int:
    """Return the requested sample count for one scene or segment."""
    if strategy not in KEYFRAME_STRATEGIES:
        raise ValueError(f"Unsupported keyframe strategy: {strategy}. Must be one of {KEYFRAME_STRATEGIES}")
    if keyframes_per_second <= 0:
        raise ValueError("keyframes_per_second must be positive")
    if min_keyframes_per_scene < 1:
        raise ValueError("min_keyframes_per_scene must be positive")
    if max_keyframes_per_scene < 0:
        raise ValueError("max_keyframes_per_scene must be non-negative")
    if max_keyframes_per_scene and max_keyframes_per_scene < min_keyframes_per_scene:
        raise ValueError("max_keyframes_per_scene must be >= min_keyframes_per_scene")

    if strategy == "tiered":
        # Tiered duration bucket policy:
        # <= 3s -> 1 keyframe (center)
        # <= 10s -> 3 keyframes
        # > 10s -> 5 keyframes
        if duration_s <= 3.0:
            return 1
        if duration_s <= 10.0:
            return 3
        return 5

    # Linear Policy: proportional to duration
    requested = math.ceil(max(0.0, duration_s) * keyframes_per_second)
    count = max(min_keyframes_per_scene, requested)
    return min(max_keyframes_per_scene, count) if max_keyframes_per_scene else count


def select_keyframes(
    scenes: Iterable[tuple[int, int]],
    fps: float,
    *,
    strategy: str = "tiered",
    keyframes_per_second: float = 0.2,
    min_keyframes_per_scene: int = 1,
    max_keyframes_per_scene: int = 20,
) -> list[dict]:
    """Select evenly spaced frame centers inside every detected scene/segment.
    
    Args:
        scenes: Iterable of (start_frame, end_frame) tuples.
        fps: Video frame rate.
        strategy: 'tiered' or 'linear'.
        keyframes_per_second: Frame rate density in linear mode.
        min_keyframes_per_scene: Lower bound per scene.
        max_keyframes_per_scene: Upper bound per scene.

    Returns:
        List of dicts: [{'frame_number': int, 'timestamp_sec': float, 'scene_index': int}, ...]
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    selected: list[dict] = []
    
    for scene_index, (start_raw, end_raw) in enumerate(scenes):
        start_frame, end_frame = int(start_raw), int(end_raw)
        duration_frames = end_frame - start_frame + 1
        if duration_frames <= 0:
            continue
            
        duration_s = duration_frames / fps
        count = min(duration_frames, keyframe_count(
            duration_s,
            strategy=strategy,
            keyframes_per_second=keyframes_per_second,
            min_keyframes_per_scene=min_keyframes_per_scene,
            max_keyframes_per_scene=max_keyframes_per_scene,
        ))
        
        numbers = sorted({
            min(end_frame, max(start_frame, start_frame + round((index + 0.5) * duration_frames / count)))
            for index in range(count)
        })
        
        for number in numbers:
            selected.append({
                "frame_number": int(number),
                "timestamp_sec": round(number / fps, 3),
                "scene_index": int(scene_index)
            })
            
    return selected
