from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp

from .config import (
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)

Point = Tuple[float, float, float]


@dataclass
class PoseFrame:
    frame_width: int
    frame_height: int
    landmarks: Dict[int, Point]

    @property
    def has_person(self) -> bool:
        return bool(self.landmarks)


class PoseDetector:
    def __init__(
        self,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        visibility_threshold: float,
    ) -> None:
        self._visibility_threshold = visibility_threshold
        self._use_legacy_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")

        if self._use_legacy_solutions:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._landmarker = None
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = Path(__file__).resolve().parent.parent / "pose_landmarker_lite.task"
            if not model_path.exists():
                raise RuntimeError(f"Pose model not found: {model_path}")

            safe_model_path = self._prepare_ascii_safe_model_path(model_path)

            options = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(safe_model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            self._pose = None

    @staticmethod
    def _prepare_ascii_safe_model_path(model_path: Path) -> Path:
        try:
            str(model_path).encode("ascii")
            return model_path
        except UnicodeEncodeError:
            pass

        safe_dir = Path(tempfile.gettempdir()) / "video_interaction_mp"
        safe_dir.mkdir(parents=True, exist_ok=True)
        safe_path = safe_dir / model_path.name

        # Refresh copy when source changed.
        if (not safe_path.exists()) or (model_path.stat().st_mtime > safe_path.stat().st_mtime):
            shutil.copy2(model_path, safe_path)

        return safe_path

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
        if self._landmarker is not None:
            self._landmarker.close()

    def detect(self, frame_bgr) -> PoseFrame:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._use_legacy_solutions:
            result = self._pose.process(frame_rgb)
            if not result.pose_landmarks:
                return PoseFrame(frame_width=w, frame_height=h, landmarks={})
            source_landmarks = result.pose_landmarks.landmark
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._landmarker.detect(mp_image)
            if not result.pose_landmarks:
                return PoseFrame(frame_width=w, frame_height=h, landmarks={})
            source_landmarks = result.pose_landmarks[0]

        selected = [
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_ELBOW,
            RIGHT_ELBOW,
            LEFT_WRIST,
            RIGHT_WRIST,
            LEFT_HIP,
            RIGHT_HIP,
        ]
        landmarks: Dict[int, Point] = {}
        for index in selected:
            lm = source_landmarks[index]
            visibility = float(getattr(lm, "visibility", 1.0))
            local_threshold = self._visibility_threshold
            if index in (LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST):
                local_threshold = max(0.15, self._visibility_threshold - 0.2)

            if visibility >= local_threshold:
                landmarks[index] = (lm.x, lm.y, visibility)

        return PoseFrame(frame_width=w, frame_height=h, landmarks=landmarks)

    @staticmethod
    def get_xy(pose_frame: PoseFrame, index: int) -> Optional[Tuple[float, float]]:
        point = pose_frame.landmarks.get(index)
        if point is None:
            return None
        return point[0], point[1]
