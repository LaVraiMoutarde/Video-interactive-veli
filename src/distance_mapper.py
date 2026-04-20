from __future__ import annotations

import math
from typing import Optional

from .config import LEFT_SHOULDER, RIGHT_SHOULDER
from .pose_detector import PoseDetector, PoseFrame


class DistanceEstimator:
    def __init__(self, camera_hfov_deg: float, shoulder_width_m: float, minimum_shoulder_px: float) -> None:
        self.camera_hfov_deg = camera_hfov_deg
        self.shoulder_width_m = shoulder_width_m
        self.minimum_shoulder_px = minimum_shoulder_px

    def estimate_distance_m(self, pose_frame: PoseFrame) -> Optional[float]:
        left = PoseDetector.get_xy(pose_frame, LEFT_SHOULDER)
        right = PoseDetector.get_xy(pose_frame, RIGHT_SHOULDER)
        if left is None or right is None:
            return None

        shoulder_px = abs(left[0] - right[0]) * pose_frame.frame_width
        if shoulder_px < self.minimum_shoulder_px:
            return None

        fov_rad = math.radians(self.camera_hfov_deg)
        focal_px = pose_frame.frame_width / (2.0 * math.tan(fov_rad / 2.0))
        distance_m = (self.shoulder_width_m * focal_px) / shoulder_px
        return max(distance_m, 0.1)


class TemporalMapper:
    def __init__(
        self,
        switch_distance_m: float,
        video2_far_distance_m: float,
        video1_min_travel_m: float,
        video2_min_travel_m: float,
    ) -> None:
        self.switch_distance_m = switch_distance_m
        self.video2_far_distance_m = video2_far_distance_m
        self.video1_min_travel_m = max(0.05, video1_min_travel_m)
        self.video2_min_travel_m = max(0.05, video2_min_travel_m)

        self.video1_trigger_distance_m = None
        self.video2_switch_distance_m = None

    def reset(self) -> None:
        self.video1_trigger_distance_m = None
        self.video2_switch_distance_m = None

    @staticmethod
    def _clamp01(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def set_video1_trigger_distance(self, distance_m: float) -> None:
        self.video1_trigger_distance_m = distance_m

    def set_video2_switch_distance(self, distance_m: float) -> None:
        self.video2_switch_distance_m = distance_m

    def progress_video1(self, distance_m: float) -> float:
        if self.video1_trigger_distance_m is None:
            return 0.0

        d0 = self.video1_trigger_distance_m
        raw_range = max(0.0, d0 - self.switch_distance_m)
        effective_range = max(self.video1_min_travel_m, raw_range)
        progress = (d0 - distance_m) / effective_range
        return self._clamp01(progress)

    def progress_video2(self, distance_m: float) -> float:
        if self.video2_switch_distance_m is None:
            return 0.0

        d0 = self.video2_switch_distance_m
        raw_range = max(0.0, self.video2_far_distance_m - d0)
        effective_range = max(self.video2_min_travel_m, raw_range)
        progress = (distance_m - d0) / effective_range
        return self._clamp01(progress)
