"""Video Processing Pipeline for Scene-Aware Keyframe Extraction.

Combines neural (TransNetV2) or content-aware (OpenCV histogram / windowed) scene detection
with tiered / linear keyframe selection policies, producing high-quality representative keyframes
and structured keyframes.json manifests.
"""
from __future__ import annotations

import os
import json
import math
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import cv2
import numpy as np

from .keyframe_selection import select_keyframes, selection_metadata, KEYFRAME_STRATEGIES


class VideoProcessor:
    def __init__(
        self,
        strategy: str = "tiered",
        keyframes_per_second: float = 0.2,
        min_keyframes_per_scene: int = 1,
        max_keyframes_per_scene: int = 10,
        scene_mode: str = "auto",  # 'auto', 'transnetv2', 'content_aware', 'fixed_window'
        scene_window_seconds: float = 10.0,
        scene_threshold: float = 0.35,
        max_dimension: int = 720,
        jpeg_quality: int = 90,
    ):
        """
        Args:
            strategy: 'tiered' (duration buckets) or 'linear' (proportional).
            keyframes_per_second: Sample rate for 'linear' mode.
            min_keyframes_per_scene: Min keyframes per scene.
            max_keyframes_per_scene: Max keyframes per scene.
            scene_mode: 'auto' (content-aware with fallback windowing),
                        'transnetv2' (deep neural shot boundary detection),
                        'content_aware' (histogram-based shot detection),
                        'fixed_window' (uniform temporal blocks).
            scene_window_seconds: Duration of temporal blocks for fixed window / fallback.
            scene_threshold: Threshold for scene detection (TransNet probability or histogram diff).
            max_dimension: Max width/height to resize keyframes (0 to keep original).
            jpeg_quality: Quality of saved JPEG images (1-100).
        """
        if strategy not in KEYFRAME_STRATEGIES:
            raise ValueError(f"strategy must be one of {KEYFRAME_STRATEGIES}")

        self.strategy = strategy
        self.keyframes_per_second = keyframes_per_second
        self.min_keyframes_per_scene = min_keyframes_per_scene
        self.max_keyframes_per_scene = max_keyframes_per_scene
        self.scene_mode = scene_mode
        self.scene_window_seconds = scene_window_seconds
        self.scene_threshold = scene_threshold
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality

    def probe_video(self, video_path: str | Path) -> Dict[str, Any]:
        """Probes video metadata using OpenCV."""
        path_str = str(video_path)
        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {path_str}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0 or math.isnan(fps):
            fps = 25.0

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = num_frames / fps if fps > 0 else 0.0

        cap.release()
        return {
            "fps": float(fps),
            "num_frames": int(num_frames),
            "width": int(width),
            "height": int(height),
            "duration_seconds": float(duration),
        }

    def _detect_scenes_transnetv2(
        self,
        video_path: str | Path,
        fps: float,
        num_frames: int,
    ) -> List[Tuple[int, int]]:
        """Deep neural shot transition detection using TransNetV2."""
        try:
            from transnetv2_pytorch import TransNetV2
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = TransNetV2().to(device).eval()

            # TransNetV2 predict_video handles batched 100-frame 48x27 evaluation
            _, single_frame_predictions, _ = model.predict_video(str(video_path))
            threshold = self.scene_threshold if self.scene_threshold < 1.0 else 0.5
            scenes = model.predictions_to_scenes(single_frame_predictions, threshold=threshold)
            return [(int(s[0]), int(s[1])) for s in scenes]
        except ImportError:
            print("[Warning] transnetv2-pytorch not installed. Falling back to content-aware histogram detection.")
            return self._detect_scenes_content_aware(video_path, fps, num_frames)
        except Exception as e:
            print(f"[Warning] TransNetV2 inference failed ({e}). Falling back to content-aware detection.")
            return self._detect_scenes_content_aware(video_path, fps, num_frames)

    def _detect_scenes_content_aware(
        self,
        video_path: str | Path,
        fps: float,
        num_frames: int,
    ) -> List[Tuple[int, int]]:
        """Content-aware shot detection using HSV color histogram diffs."""
        if num_frames <= 0:
            return [(0, 0)]

        scenes: List[Tuple[int, int]] = []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [(0, num_frames - 1)]

        sample_step = max(1, int(fps * 0.2))  # Sample every 0.2s
        prev_hist = None
        current_scene_start = 0
        max_scene_frames = int(30.0 * fps)  # Force split if scene exceeds 30s in 'auto' mode

        frame_idx = 0
        while frame_idx < num_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            small_frame = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_NEAREST)
            hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                duration_in_scene = frame_idx - current_scene_start
                is_cut = diff > self.scene_threshold
                is_timeout = (self.scene_mode == "auto" and duration_in_scene >= max_scene_frames)

                if is_cut or is_timeout:
                    scenes.append((current_scene_start, frame_idx - 1))
                    current_scene_start = frame_idx

            prev_hist = hist
            frame_idx += sample_step

        cap.release()

        if current_scene_start < num_frames:
            scenes.append((current_scene_start, num_frames - 1))

        return scenes if scenes else [(0, num_frames - 1)]

    def detect_scenes(
        self,
        video_path: str | Path,
        fps: float,
        num_frames: int,
    ) -> List[Tuple[int, int]]:
        """Segments video into scene intervals [(start_frame, end_frame), ...]."""
        if num_frames <= 0:
            return [(0, 0)]

        if self.scene_mode == "transnetv2":
            return self._detect_scenes_transnetv2(video_path, fps, num_frames)

        if self.scene_mode == "fixed_window":
            window_frames = max(1, int(self.scene_window_seconds * fps))
            scenes = []
            for start in range(0, num_frames, window_frames):
                end = min(num_frames - 1, start + window_frames - 1)
                scenes.append((start, end))
            return scenes

        # 'auto' or 'content_aware'
        return self._detect_scenes_content_aware(video_path, fps, num_frames)

    def process_video(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        video_id: Optional[str] = None,
        save_images: bool = True,
    ) -> Dict[str, Any]:
        """Runs end-to-end video processing: probe -> scene detection -> keyframe selection -> extraction.

        Args:
            video_path: Path to .mp4 file.
            output_dir: Base directory to store video keyframes & metadata.
            video_id: Optional ID (defaults to file stem).
            save_images: Whether to save decoded keyframe images to disk.

        Returns:
            Dictionary with video metadata, scenes, keyframes, and output paths.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        vid = video_id or video_path.stem
        video_out_dir = Path(output_dir) / vid
        keyframes_dir = video_out_dir / "keyframes"
        keyframes_dir.mkdir(parents=True, exist_ok=True)

        meta = self.probe_video(video_path)
        fps = meta["fps"]
        num_frames = meta["num_frames"]

        # Step 1: Detect scenes
        scenes = self.detect_scenes(video_path, fps, num_frames)

        # Step 2: Select keyframes using selection policy
        raw_keyframes = select_keyframes(
            scenes,
            fps,
            strategy=self.strategy,
            keyframes_per_second=self.keyframes_per_second,
            min_keyframes_per_scene=self.min_keyframes_per_scene,
            max_keyframes_per_scene=self.max_keyframes_per_scene,
        )

        # Step 3: Extract & save keyframe images
        extracted_keyframes: List[Dict[str, Any]] = []
        if save_images and raw_keyframes:
            cap = cv2.VideoCapture(str(video_path))
            for item in raw_keyframes:
                frame_no = item["frame_number"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # Resize if needed
                h, w = frame.shape[:2]
                if self.max_dimension > 0 and max(h, w) > self.max_dimension:
                    scale = self.max_dimension / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                img_filename = f"frame_{frame_no:06d}.jpg"
                img_path = keyframes_dir / img_filename
                cv2.imwrite(
                    str(img_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )

                extracted_keyframes.append({
                    "frame_number": frame_no,
                    "timestamp_sec": item["timestamp_sec"],
                    "scene_index": item["scene_index"],
                    "filename": img_filename,
                    "relative_path": f"{vid}/keyframes/{img_filename}",
                })
            cap.release()
        else:
            for item in raw_keyframes:
                frame_no = item["frame_number"]
                img_filename = f"frame_{frame_no:06d}.jpg"
                extracted_keyframes.append({
                    "frame_number": frame_no,
                    "timestamp_sec": item["timestamp_sec"],
                    "scene_index": item["scene_index"],
                    "filename": img_filename,
                    "relative_path": f"{vid}/keyframes/{img_filename}",
                })

        # Save keyframes.json
        keyframes_payload = {
            "video_id": vid,
            "source_filename": video_path.name,
            "fps": fps,
            "width": meta["width"],
            "height": meta["height"],
            "num_frames": num_frames,
            "duration_seconds": meta["duration_seconds"],
            "num_scenes": len(scenes),
            "selection": selection_metadata(
                strategy=self.strategy,
                keyframes_per_second=self.keyframes_per_second,
                min_keyframes_per_scene=self.min_keyframes_per_scene,
                max_keyframes_per_scene=self.max_keyframes_per_scene,
            ),
            "num_keyframes": len(extracted_keyframes),
            "keyframes": extracted_keyframes,
        }

        keyframes_json_path = video_out_dir / "keyframes.json"
        with open(keyframes_json_path, "w", encoding="utf-8") as f:
            json.dump(keyframes_payload, f, indent=2, ensure_ascii=False)

        return keyframes_payload
