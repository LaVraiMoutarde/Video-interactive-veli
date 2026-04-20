from __future__ import annotations

from dataclasses import dataclass

from .config import LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_WRIST
from .pose_detector import PoseDetector, PoseFrame


@dataclass
class GestureSignals:
    both_hands_triggered: bool = False
    right_hand_triggered: bool = False
    both_hands_up: bool = False
    right_hand_up: bool = False
    both_hands_hold_ratio: float = 0.0
    right_hand_hold_ratio: float = 0.0


class GestureRecognizer:
    def __init__(
        self,
        hand_above_shoulder_margin: float,
        start_hold_seconds: float,
        switch_hold_seconds: float,
        cooldown_seconds: float,
        dropout_grace_seconds: float,
    ) -> None:
        self.margin = hand_above_shoulder_margin
        self.start_hold_seconds = start_hold_seconds
        self.switch_hold_seconds = switch_hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.dropout_grace_seconds = dropout_grace_seconds

        self._last_update_s = None
        self._both_hold_s = 0.0
        self._right_hold_s = 0.0
        self._both_lost_s = 0.0
        self._right_lost_s = 0.0
        self._cooldown_until = 0.0

    def _is_arm_up(self, pose_frame: PoseFrame, wrist_idx: int, elbow_idx: int, shoulder_idx: int) -> bool:
        shoulder = PoseDetector.get_xy(pose_frame, shoulder_idx)
        if shoulder is None:
            return False

        wrist = PoseDetector.get_xy(pose_frame, wrist_idx)
        elbow = PoseDetector.get_xy(pose_frame, elbow_idx)

        if wrist is not None:
            # Standard case: wrist visible.
            return wrist[1] < (shoulder[1] - self.margin)

        if elbow is not None:
            # Fallback when wrist goes out of frame at full extension.
            relaxed_margin = max(0.005, self.margin * 0.25)
            return elbow[1] < (shoulder[1] - relaxed_margin)

        return False

    def _is_left_arm_up(self, pose_frame: PoseFrame) -> bool:
        return self._is_arm_up(pose_frame, LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER)

    def _is_right_arm_up(self, pose_frame: PoseFrame) -> bool:
        return self._is_arm_up(pose_frame, RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER)

    def update(self, pose_frame: PoseFrame, now_s: float) -> GestureSignals:
        if self._last_update_s is None:
            dt = 0.0
        else:
            dt = max(0.0, min(0.12, now_s - self._last_update_s))
        self._last_update_s = now_s

        left_up = self._is_left_arm_up(pose_frame)
        right_up = self._is_right_arm_up(pose_frame)
        both_up = left_up and right_up
        right_up_only = right_up and not left_up

        if both_up:
            self._both_hold_s = min(self.start_hold_seconds, self._both_hold_s + dt)
            self._both_lost_s = 0.0
        else:
            self._both_lost_s += dt
            if self._both_lost_s > self.dropout_grace_seconds:
                self._both_hold_s = max(0.0, self._both_hold_s - (dt * 0.80))

        if right_up_only:
            self._right_hold_s = min(self.switch_hold_seconds, self._right_hold_s + dt)
            self._right_lost_s = 0.0
        else:
            self._right_lost_s += dt
            if self._right_lost_s > self.dropout_grace_seconds:
                self._right_hold_s = max(0.0, self._right_hold_s - (dt * 0.95))

        both_hold_ratio = 0.0 if self.start_hold_seconds <= 1e-6 else min(1.0, max(0.0, self._both_hold_s / self.start_hold_seconds))

        right_hold_ratio = 0.0 if self.switch_hold_seconds <= 1e-6 else min(1.0, max(0.0, self._right_hold_s / self.switch_hold_seconds))

        signals = GestureSignals(
            both_hands_up=both_up,
            right_hand_up=right_up_only,
            both_hands_hold_ratio=both_hold_ratio,
            right_hand_hold_ratio=right_hold_ratio,
        )
        if now_s < self._cooldown_until:
            return signals

        if self._both_hold_s >= self.start_hold_seconds:
            signals.both_hands_triggered = True
            self._cooldown_until = now_s + self.cooldown_seconds
            self._both_hold_s = 0.0
            self._right_hold_s = 0.0
            self._both_lost_s = 0.0
            self._right_lost_s = 0.0
            return signals

        if self._right_hold_s >= self.switch_hold_seconds:
            signals.right_hand_triggered = True
            self._cooldown_until = now_s + self.cooldown_seconds
            self._right_hold_s = 0.0
            self._both_hold_s = 0.0
            self._both_lost_s = 0.0
            self._right_lost_s = 0.0

        return signals
