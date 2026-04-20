from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np


@dataclass
class LoadedVideo:
    frames: List[np.ndarray]
    fps: float

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def frame_at_progress(self, progress: float) -> np.ndarray:
        if not self.frames:
            raise RuntimeError("No frames available in loaded video")

        progress = max(0.0, min(1.0, progress))
        position = progress * (self.frame_count - 1)
        i0 = int(position)
        i1 = min(i0 + 1, self.frame_count - 1)
        alpha = position - i0

        if i0 == i1:
            return self.frames[i0]

        frame0 = self.frames[i0]
        frame1 = self.frames[i1]
        blended = cv2.addWeighted(frame0, 1.0 - alpha, frame1, alpha, 0.0)
        return blended


class DualVideoRenderer:
    def __init__(self, video1_path: str, video2_path: str) -> None:
        self.video1 = self._load_video(video1_path)
        self.video2 = self._load_video(video2_path)

    @staticmethod
    def _resolve_path(path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate

        project_root = Path(__file__).resolve().parent.parent
        return project_root / candidate

    def _load_video(self, path: str) -> LoadedVideo:
        resolved_path = self._resolve_path(path)
        cap = cv2.VideoCapture(str(resolved_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {resolved_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 30.0

        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)

        cap.release()
        if not frames:
            raise RuntimeError(f"Video has no frames: {resolved_path}")

        return LoadedVideo(frames=frames, fps=fps)

    def render(self, video_index: int, progress: float) -> np.ndarray:
        if video_index == 1:
            return self.video1.frame_at_progress(progress)
        if video_index == 2:
            return self.video2.frame_at_progress(progress)
        raise ValueError("video_index must be 1 or 2")
