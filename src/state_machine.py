from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .distance_mapper import TemporalMapper
from .gesture_recognizer import GestureSignals


class InteractionState(str, Enum):
    WAITING_START = "WAITING_START"
    VIDEO1_CONTROL = "VIDEO1_CONTROL"
    VIDEO2_DETECTION = "VIDEO2_DETECTION"
    VIDEO2_CONTROL = "VIDEO2_CONTROL"
    TRACKING_LOST = "TRACKING_LOST"


@dataclass
class StateOutput:
    state: InteractionState
    active_video: Optional[int]
    progress: float
    countdown_s: Optional[float] = None
    info_message: Optional[str] = None
    launch_window_active: bool = False
    video2_trigger_window_active: bool = False


class InteractionStateMachine:
    def __init__(
        self,
        mapper: TemporalMapper,
        launch_min_distance_m: float,
        launch_max_distance_m: float,
        launch_window_hysteresis_m: float,
        switch_distance_m: float,
        switch_hysteresis_m: float,
        video2_trigger_min_distance_m: float,
        video2_trigger_max_distance_m: float,
        video2_trigger_hysteresis_m: float,
        tracking_lost_timeout_s: float,
        tracking_reset_delay_s: float = 2.0,
    ) -> None:
        self.mapper = mapper
        self.launch_min_distance_m = launch_min_distance_m
        self.launch_max_distance_m = launch_max_distance_m
        self.launch_window_hysteresis_m = launch_window_hysteresis_m
        self.switch_distance_m = switch_distance_m
        self.switch_hysteresis_m = switch_hysteresis_m
        self.video2_trigger_min_distance_m = video2_trigger_min_distance_m
        self.video2_trigger_max_distance_m = video2_trigger_max_distance_m
        self.video2_trigger_hysteresis_m = video2_trigger_hysteresis_m
        self.tracking_lost_timeout_s = tracking_lost_timeout_s
        self.tracking_reset_delay_s = tracking_reset_delay_s

        self.state = InteractionState.WAITING_START
        self._last_seen_person_s = 0.0
        self._tracking_lost_since_s = None
        self._video1_progress = 0.0
        self._video2_progress = 1.0
        self._launch_window_active = False
        self._video2_trigger_window_active = False
        self._last_valid_distance_m = None

    def reset_session(self) -> None:
        self.mapper.reset()
        self.state = InteractionState.WAITING_START
        self._tracking_lost_since_s = None
        self._last_seen_person_s = 0.0
        self._video1_progress = 0.0
        self._video2_progress = 1.0
        self._launch_window_active = False
        self._video2_trigger_window_active = False
        self._last_valid_distance_m = None

    def _switch_enter_distance(self) -> float:
        return self.switch_distance_m - self.switch_hysteresis_m

    def _switch_exit_distance(self) -> float:
        return self.switch_distance_m + self.switch_hysteresis_m

    def _update_window_with_hysteresis(
        self,
        distance_m: float,
        was_active: bool,
        min_distance_m: float,
        max_distance_m: float,
        hysteresis_m: float,
    ) -> bool:
        if hysteresis_m <= 0.0:
            return min_distance_m <= distance_m <= max_distance_m

        entry_min = min_distance_m + hysteresis_m
        entry_max = max_distance_m - hysteresis_m
        if entry_min > entry_max:
            entry_min = min_distance_m
            entry_max = max_distance_m

        if was_active:
            return (min_distance_m - hysteresis_m) <= distance_m <= (max_distance_m + hysteresis_m)

        return entry_min <= distance_m <= entry_max

    def _is_in_launch_window(self, distance_m: float) -> bool:
        self._launch_window_active = self._update_window_with_hysteresis(
            distance_m,
            self._launch_window_active,
            self.launch_min_distance_m,
            self.launch_max_distance_m,
            self.launch_window_hysteresis_m,
        )
        return self._launch_window_active

    def _is_in_video2_trigger_window(self, distance_m: float) -> bool:
        self._video2_trigger_window_active = self._update_window_with_hysteresis(
            distance_m,
            self._video2_trigger_window_active,
            self.video2_trigger_min_distance_m,
            self.video2_trigger_max_distance_m,
            self.video2_trigger_hysteresis_m,
        )
        return self._video2_trigger_window_active

    def update(
        self,
        now_s: float,
        has_person: bool,
        distance_m: Optional[float],
        gestures: GestureSignals,
    ) -> StateOutput:
        info_message = None
        launch_window_active = False
        video2_trigger_window_active = False

        if distance_m is not None:
            self._last_valid_distance_m = distance_m

        if has_person:
            self._last_seen_person_s = now_s
            self._tracking_lost_since_s = None
        elif (now_s - self._last_seen_person_s) > self.tracking_lost_timeout_s:
            self.state = InteractionState.TRACKING_LOST
            if self._tracking_lost_since_s is None:
                self._tracking_lost_since_s = now_s

        if self.state == InteractionState.TRACKING_LOST:
            if has_person:
                self.state = InteractionState.WAITING_START
                self._tracking_lost_since_s = None
            else:
                elapsed = now_s - self._tracking_lost_since_s if self._tracking_lost_since_s is not None else 0.0
                if elapsed >= self.tracking_reset_delay_s:
                    self.reset_session()
                    return StateOutput(self.state, None, 0.0, countdown_s=0.0, info_message="session reinitialisee")

                countdown = max(0.0, self.tracking_reset_delay_s - elapsed)
                return StateOutput(self.state, None, 0.0, countdown_s=countdown, info_message="tracking perdu")

        if self.state == InteractionState.WAITING_START:
            self._video1_progress = 0.0
            self._video2_progress = 1.0
            if distance_m is not None:
                launch_window_active = self._is_in_launch_window(distance_m)
                if launch_window_active:
                    info_message = "Levez les 2 mains pour commencer"
                elif distance_m > self.launch_max_distance_m:
                    info_message = "Avancez pour commencer"
                elif distance_m < self.launch_min_distance_m:
                    info_message = "Reculez pour commencer"
                else:
                    info_message = "Ajustez votre distance"
            else:
                launch_window_active = self._launch_window_active
                if has_person:
                    info_message = "Stabilisez votre position"
                else:
                    info_message = "Placez-vous face a la camera"

            if gestures.both_hands_triggered:
                distance_for_start = distance_m if distance_m is not None else self._last_valid_distance_m
                if distance_for_start is None:
                    info_message = "Distance indisponible"
                elif self._is_in_launch_window(distance_for_start):
                    self.mapper.set_video1_trigger_distance(distance_for_start)
                    self.state = InteractionState.VIDEO1_CONTROL
                    info_message = "Video 1 lancee"
                else:
                    info_message = f"Placez-vous entre {self.launch_min_distance_m:.1f} m et {self.launch_max_distance_m:.1f} m"

        elif self.state == InteractionState.VIDEO1_CONTROL:
            if distance_m is not None:
                self._video1_progress = self.mapper.progress_video1(distance_m)
                launch_window_active = self._is_in_launch_window(distance_m)
                if distance_m <= self._switch_enter_distance():
                    self.state = InteractionState.VIDEO2_DETECTION
                    info_message = "Levez la main droite"

        elif self.state == InteractionState.VIDEO2_DETECTION:
            if distance_m is not None:
                self._video1_progress = self.mapper.progress_video1(distance_m)
                launch_window_active = self._is_in_launch_window(distance_m)
                video2_trigger_window_active = self._is_in_video2_trigger_window(distance_m)
                if distance_m >= self._switch_exit_distance():
                    self.state = InteractionState.VIDEO1_CONTROL
                    info_message = "Reculez pour revenir a la video 1"
                elif not video2_trigger_window_active:
                    info_message = "Ajustez votre distance"
                elif gestures.right_hand_triggered:
                    if video2_trigger_window_active:
                        self.mapper.set_video2_switch_distance(distance_m)
                        self.state = InteractionState.VIDEO2_CONTROL
                        info_message = "Video 2 lancee"
                    else:
                        info_message = (
                            f"Main droite valide entre {self.video2_trigger_min_distance_m:.1f} m et {self.video2_trigger_max_distance_m:.1f} m"
                        )

        elif self.state == InteractionState.VIDEO2_CONTROL:
            if distance_m is not None:
                self._video2_progress = self.mapper.progress_video2(distance_m)
                video2_trigger_window_active = self._is_in_video2_trigger_window(distance_m)

        if self.state == InteractionState.WAITING_START:
            return StateOutput(
                self.state,
                None,
                0.0,
                info_message=info_message,
                launch_window_active=launch_window_active,
                video2_trigger_window_active=video2_trigger_window_active,
            )
        if self.state in (InteractionState.VIDEO1_CONTROL, InteractionState.VIDEO2_DETECTION):
            return StateOutput(
                self.state,
                1,
                self._video1_progress,
                info_message=info_message,
                launch_window_active=launch_window_active,
                video2_trigger_window_active=video2_trigger_window_active,
            )
        if self.state == InteractionState.VIDEO2_CONTROL:
            return StateOutput(
                self.state,
                2,
                self._video2_progress,
                info_message=info_message,
                launch_window_active=launch_window_active,
                video2_trigger_window_active=video2_trigger_window_active,
            )
        return StateOutput(
            self.state,
            None,
            0.0,
            info_message=info_message,
            launch_window_active=launch_window_active,
            video2_trigger_window_active=video2_trigger_window_active,
        )
