from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .config import LEFT_SHOULDER, RIGHT_SHOULDER
from .pose_detector import PoseDetector, PoseFrame


@dataclass
class UserSignature:
    center: Tuple[float, float]
    shoulder_span: float


class SingleUserTracker:
    def __init__(
        self,
        center_tolerance: float = 0.16,
        shoulder_ratio_min: float = 0.72,
        shoulder_ratio_max: float = 1.42,
        acquisition_center_tolerance: float = 0.30,
        update_alpha: float = 0.18,
        hold_last_pose_frames: int = 8,
        unlock_after_mismatch_frames: int = 18,
    ) -> None:
        self.center_tolerance = center_tolerance
        self.shoulder_ratio_min = shoulder_ratio_min
        self.shoulder_ratio_max = shoulder_ratio_max
        self.acquisition_center_tolerance = acquisition_center_tolerance
        self.update_alpha = update_alpha
        self.hold_last_pose_frames = hold_last_pose_frames
        self.unlock_after_mismatch_frames = unlock_after_mismatch_frames

        self._locked_signature: Optional[UserSignature] = None
        self._last_accepted_pose: Optional[PoseFrame] = None
        self._mismatch_frames = 0

    @property
    def is_locked(self) -> bool:
        return self._locked_signature is not None

    def unlock(self) -> None:
        self._locked_signature = None
        self._last_accepted_pose = None
        self._mismatch_frames = 0

    def lock_from_pose(self, pose_frame: PoseFrame) -> bool:
        signature = self._extract_signature(pose_frame)
        if signature is None:
            return False

        if abs(signature.center[0] - 0.5) > self.acquisition_center_tolerance:
            return False

        self._locked_signature = signature
        self._last_accepted_pose = PoseFrame(
            frame_width=pose_frame.frame_width,
            frame_height=pose_frame.frame_height,
            landmarks=dict(pose_frame.landmarks),
        )
        self._mismatch_frames = 0
        return True

    def filter_pose(self, pose_frame: PoseFrame) -> PoseFrame:
        if not self.is_locked:
            return pose_frame

        if not pose_frame.has_person:
            self._mismatch_frames += 1
            if self._mismatch_frames <= self.hold_last_pose_frames and self._last_accepted_pose is not None:
                return self._last_accepted_pose
            if self._mismatch_frames >= self.unlock_after_mismatch_frames:
                self.unlock()
            return pose_frame

        signature = self._extract_signature(pose_frame)
        if signature is None:
            self._mismatch_frames += 1
            if self._mismatch_frames <= self.hold_last_pose_frames and self._last_accepted_pose is not None:
                return self._last_accepted_pose
            if self._mismatch_frames >= self.unlock_after_mismatch_frames:
                self.unlock()
            return PoseFrame(
                frame_width=pose_frame.frame_width,
                frame_height=pose_frame.frame_height,
                landmarks={},
            )

        if self._matches_locked_user(signature):
            self._mismatch_frames = 0
            self._last_accepted_pose = PoseFrame(
                frame_width=pose_frame.frame_width,
                frame_height=pose_frame.frame_height,
                landmarks=dict(pose_frame.landmarks),
            )
            return pose_frame

        self._mismatch_frames += 1
        if self._mismatch_frames <= self.hold_last_pose_frames and self._last_accepted_pose is not None:
            return self._last_accepted_pose
        if self._mismatch_frames >= self.unlock_after_mismatch_frames:
            self.unlock()

        return PoseFrame(
            frame_width=pose_frame.frame_width,
            frame_height=pose_frame.frame_height,
            landmarks={},
        )

    def _blend_signature(self, ref: UserSignature, current: UserSignature) -> UserSignature:
        a = min(1.0, max(0.0, self.update_alpha))
        blended_center = (
            (1.0 - a) * ref.center[0] + a * current.center[0],
            (1.0 - a) * ref.center[1] + a * current.center[1],
        )
        blended_span = (1.0 - a) * ref.shoulder_span + a * current.shoulder_span
        return UserSignature(center=blended_center, shoulder_span=blended_span)

    def _extract_signature(self, pose_frame: PoseFrame) -> Optional[UserSignature]:
        left = PoseDetector.get_xy(pose_frame, LEFT_SHOULDER)
        right = PoseDetector.get_xy(pose_frame, RIGHT_SHOULDER)
        if left is None or right is None:
            return None

        center = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)
        shoulder_span = abs(left[0] - right[0])
        if shoulder_span <= 1e-6:
            return None

        return UserSignature(center=center, shoulder_span=shoulder_span)

    def _matches_locked_user(self, current: UserSignature) -> bool:
        if self._locked_signature is None:
            return False

        ref = self._locked_signature

        dx = current.center[0] - ref.center[0]
        dy = current.center[1] - ref.center[1]
        center_distance = (dx * dx + dy * dy) ** 0.5

        shoulder_ratio = current.shoulder_span / ref.shoulder_span

        if center_distance > self.center_tolerance:
            return False
        if shoulder_ratio < self.shoulder_ratio_min or shoulder_ratio > self.shoulder_ratio_max:
            return False

        # Update lock slowly to avoid drifting onto another person.
        self._locked_signature = self._blend_signature(ref, current)
        return True
