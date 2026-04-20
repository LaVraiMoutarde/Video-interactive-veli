from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2

from .config import AppConfig


@dataclass
class CalibrationParam:
    name: str
    label: str
    small_step: float
    large_step: float
    minimum: float
    maximum: float
    decimals: int = 2


class CalibrationModeController:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.enabled = False
        self.selected_index = 0
        self.dirty = False
        self.status_message = ""
        self.guided_active = False
        self.guided_step = 0

        self.params: List[CalibrationParam] = [
            CalibrationParam("launch_min_distance_m", "Debut video 1 - min (m)", 0.05, 0.2, 1.0, 6.0),
            CalibrationParam("launch_max_distance_m", "Debut video 1 - max (m)", 0.05, 0.2, 1.2, 6.0),
            CalibrationParam("launch_window_hysteresis_m", "Stabilite debut (m)", 0.01, 0.05, 0.0, 0.4),
            CalibrationParam("video2_trigger_min_distance_m", "Trigger video 2 - min (m)", 0.05, 0.2, 0.8, 5.5),
            CalibrationParam("video2_trigger_max_distance_m", "Trigger video 2 - max (m)", 0.05, 0.2, 1.0, 6.0),
            CalibrationParam("video2_trigger_hysteresis_m", "Stabilite trigger v2 (m)", 0.01, 0.05, 0.0, 0.4),
            CalibrationParam("switch_distance_m", "Distance de bascule (m)", 0.02, 0.1, 1.0, 4.5),
            CalibrationParam("video1_min_travel_m", "Video 1 - distance min 0->100 (m)", 0.02, 0.1, 0.05, 3.0),
            CalibrationParam("switch_hysteresis_m", "Stabilite bascule (m)", 0.01, 0.05, 0.01, 0.5),
            CalibrationParam("video2_far_distance_m", "Video 2 - distance loin (m)", 0.05, 0.2, 2.0, 8.0),
            CalibrationParam("video2_min_travel_m", "Video 2 - distance min 0->100 (m)", 0.02, 0.1, 0.05, 4.0),
            CalibrationParam("camera_hfov_deg", "Angle camera (deg)", 0.5, 2.0, 40.0, 120.0, decimals=1),
            CalibrationParam("shoulder_width_m", "Largeur epaules (m)", 0.01, 0.05, 0.25, 0.7),
            CalibrationParam("hand_above_shoulder_margin", "Marge main/epaule", 0.005, 0.02, 0.01, 0.2, decimals=3),
        ]

        self.guided_steps: List[Tuple[str, str]] = [
            (
                "launch_min_distance_m",
                "Etape 1/4: placez-vous a la position de depart la plus PROCHE de la camera, puis ESPACE",
            ),
            (
                "launch_max_distance_m",
                "Etape 2/4: placez-vous a la position de depart la plus ELOIGNEE de la camera, puis ESPACE",
            ),
            (
                "switch_distance_m",
                "Etape 3/4: placez-vous a la distance de BASCULE entre video 1 et video 2, puis ESPACE",
            ),
            (
                "video2_trigger_min_distance_m",
                "Etape 4/4: placez-vous a la distance la plus PROCHE ou le geste video 2 doit marcher, puis ESPACE",
            ),
        ]

    def toggle(self) -> str:
        self.enabled = not self.enabled
        if self.enabled:
            self.status_message = "Mode calibration active"
        else:
            self.guided_active = False
            self.status_message = "Mode calibration desactive"
        return self.status_message

    def toggle_guided(self) -> str:
        self.guided_active = not self.guided_active
        if self.guided_active:
            self.guided_step = 0
            self.status_message = self.guided_steps[0][1]
            return "Assistant guide active"

        self.status_message = "Assistant guide desactive"
        return "Assistant guide desactive"

    def selected_param(self) -> CalibrationParam:
        return self.params[self.selected_index]

    def _set_param_value(self, param: CalibrationParam, value: float) -> None:
        clamped = min(param.maximum, max(param.minimum, value))
        setattr(self.config, param.name, round(clamped, param.decimals))

    def _apply_delta(self, delta: float) -> None:
        param = self.selected_param()
        current = float(getattr(self.config, param.name))
        self._set_param_value(param, current + delta)
        self._enforce_consistency()
        self.dirty = True

    def _enforce_consistency(self) -> None:
        min_gap = 0.05

        if self.config.launch_min_distance_m >= self.config.launch_max_distance_m - min_gap:
            self.config.launch_max_distance_m = round(self.config.launch_min_distance_m + min_gap, 2)

        if self.config.video2_trigger_min_distance_m >= self.config.video2_trigger_max_distance_m - min_gap:
            self.config.video2_trigger_max_distance_m = round(self.config.video2_trigger_min_distance_m + min_gap, 2)

        if self.config.switch_distance_m < self.config.video2_trigger_min_distance_m + min_gap:
            self.config.switch_distance_m = round(self.config.video2_trigger_min_distance_m + min_gap, 2)

        if self.config.switch_distance_m > self.config.launch_max_distance_m - min_gap:
            self.config.switch_distance_m = round(self.config.launch_max_distance_m - min_gap, 2)

        if self.config.video2_far_distance_m < self.config.launch_max_distance_m + 0.2:
            self.config.video2_far_distance_m = round(self.config.launch_max_distance_m + 0.2, 2)

        self.config.launch_min_distance_m = max(1.0, self.config.launch_min_distance_m)
        self.config.launch_max_distance_m = min(6.0, self.config.launch_max_distance_m)
        self.config.video2_trigger_min_distance_m = max(0.8, self.config.video2_trigger_min_distance_m)
        self.config.video2_trigger_max_distance_m = min(6.0, self.config.video2_trigger_max_distance_m)

    def _capture_guided_step(self, distance_m: Optional[float]) -> str:
        if distance_m is None:
            return "Distance indisponible: impossible de capturer"

        field_name, _ = self.guided_steps[self.guided_step]
        setattr(self.config, field_name, round(distance_m, 2))

        if field_name == "launch_max_distance_m":
            self.config.video2_trigger_max_distance_m = round(distance_m, 2)

        self.guided_step += 1
        self.dirty = True

        if self.guided_step >= len(self.guided_steps):
            self.guided_active = False
            self._enforce_consistency()
            return "Calibration guidee terminee"

        self._enforce_consistency()
        self.status_message = self.guided_steps[self.guided_step][1]
        return self.status_message

    def handle_key(self, key: int, raw_distance: Optional[float], filtered_distance: Optional[float]) -> Optional[str]:
        if key == ord("g"):
            return self.toggle_guided()

        if self.guided_active:
            if key == 32:
                capture_distance = filtered_distance if filtered_distance is not None else raw_distance
                return self._capture_guided_step(capture_distance)
            if key == ord("x"):
                self.guided_active = False
                return "Calibration guidee annulee"
            return None

        if key == ord("["):
            self.selected_index = (self.selected_index - 1) % len(self.params)
            return f"Parametre: {self.selected_param().label}"

        if key == ord("]"):
            self.selected_index = (self.selected_index + 1) % len(self.params)
            return f"Parametre: {self.selected_param().label}"

        if key in (ord("+"), ord("=")):
            self._apply_delta(self.selected_param().small_step)
            return f"Mis a jour: {self.selected_param().label}"

        if key in (ord("-"), ord("_")):
            self._apply_delta(-self.selected_param().small_step)
            return f"Mis a jour: {self.selected_param().label}"

        if key == ord("}"):
            self._apply_delta(self.selected_param().large_step)
            return f"Mis a jour: {self.selected_param().label}"

        if key == ord("{"):
            self._apply_delta(-self.selected_param().large_step)
            return f"Mis a jour: {self.selected_param().label}"

        if key == ord("s"):
            return "save"

        if key == ord("r"):
            return "reload"

        return None

    def _draw_distance_bar(self, frame, raw_distance: Optional[float], x: int, y: int, w: int, h: int) -> None:
        d_min = 0.0
        d_max = max(5.0, self.config.video2_far_distance_m + 0.5)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (34, 34, 34), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (155, 155, 155), 1)

        def to_x(distance_m: float) -> int:
            ratio = (distance_m - d_min) / max(1e-6, (d_max - d_min))
            ratio = min(1.0, max(0.0, ratio))
            return x + int(ratio * w)

        launch_x0 = to_x(self.config.launch_min_distance_m)
        launch_x1 = to_x(self.config.launch_max_distance_m)
        cv2.rectangle(frame, (launch_x0, y + 4), (launch_x1, y + h - 4), (70, 125, 80), -1)

        trig_x0 = to_x(self.config.video2_trigger_min_distance_m)
        trig_x1 = to_x(self.config.video2_trigger_max_distance_m)
        cv2.rectangle(frame, (trig_x0, y + 10), (trig_x1, y + h - 10), (100, 92, 62), -1)

        sw = to_x(self.config.switch_distance_m)
        cv2.line(frame, (sw, y), (sw, y + h), (120, 170, 230), 2)

        if raw_distance is not None:
            cx = to_x(raw_distance)
            cv2.line(frame, (cx, y - 6), (cx, y + h + 6), (95, 210, 120), 2)

        cv2.putText(frame, "Zone debut v1", (launch_x0, y - 8), cv2.FONT_HERSHEY_DUPLEX, 0.45, (150, 220, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, "Zone trigger v2", (trig_x0, y + h + 18), cv2.FONT_HERSHEY_DUPLEX, 0.45, (215, 200, 150), 1, cv2.LINE_AA)

    def _draw_guided_progress_hud(self, frame, x: int, y: int, w: int) -> None:
        total_steps = len(self.guided_steps)
        current_step = min(self.guided_step + 1, total_steps)

        step_palette = [
            ((55, 55, 55), (190, 230, 200), (70, 70, 70)),
            ((55, 55, 55), (190, 230, 200), (70, 70, 70)),
            ((55, 55, 55), (205, 225, 245), (70, 70, 70)),
            ((55, 55, 55), (220, 225, 238), (70, 70, 70)),
        ]
        palette_idx = min(max(self.guided_step, 0), len(step_palette) - 1)
        title_color, fill_color, text_color = step_palette[palette_idx]

        title = f"CALIBRATION GUIDEE {current_step}/{total_steps}"
        cv2.putText(frame, title, (x, y - 12), cv2.FONT_HERSHEY_DUPLEX, 0.95, title_color, 2, cv2.LINE_AA)

        bar_h = 24
        cv2.rectangle(frame, (x, y), (x + w, y + bar_h), (246, 246, 246), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + bar_h), (195, 195, 195), 2)

        fill_ratio = self.guided_step / max(1, total_steps)
        fill_w = int(w * min(1.0, max(0.0, fill_ratio)))
        if fill_w > 0:
            cv2.rectangle(frame, (x + 2, y + 2), (x + fill_w - 2, y + bar_h - 2), fill_color, -1)

        for i in range(total_steps + 1):
            tick_x = x + int((i / total_steps) * w)
            cv2.line(frame, (tick_x, y + bar_h + 2), (tick_x, y + bar_h + 9), (160, 160, 160), 1)

        if self.guided_step < total_steps:
            _, step_text = self.guided_steps[self.guided_step]
            cv2.putText(frame, step_text, (x, y + bar_h + 34), cv2.FONT_HERSHEY_DUPLEX, 0.65, text_color, 2, cv2.LINE_AA)
            cv2.putText(frame, "ESPACE: capturer | X: annuler", (x, y + bar_h + 62), cv2.FONT_HERSHEY_DUPLEX, 0.62, text_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Calibration guidee terminee", (x, y + bar_h + 34), cv2.FONT_HERSHEY_DUPLEX, 0.65, (70, 130, 85), 2, cv2.LINE_AA)

    def _draw_compact_guided_block(self, frame, x: int, y: int, w: int) -> int:
        total_steps = len(self.guided_steps)
        current_step = min(self.guided_step + 1, total_steps)

        title = f"CALIBRATION GUIDEE {current_step}/{total_steps}"
        cv2.putText(frame, title, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.72, (235, 235, 235), 2, cv2.LINE_AA)
        y += 24

        bar_h = 16
        cv2.rectangle(frame, (x, y), (x + w, y + bar_h), (34, 34, 34), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + bar_h), (155, 155, 155), 1)
        fill_ratio = self.guided_step / max(1, total_steps)
        fill_w = int(w * min(1.0, max(0.0, fill_ratio)))
        if fill_w > 2:
            cv2.rectangle(frame, (x + 2, y + 2), (x + fill_w - 2, y + bar_h - 2), (120, 170, 230), -1)
        y += 30

        if self.guided_step < total_steps:
            _, step_text = self.guided_steps[self.guided_step]
            cv2.putText(frame, step_text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.52, (210, 210, 210), 1, cv2.LINE_AA)
            y += 24
            cv2.putText(frame, "ESPACE: capturer | X: annuler", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.50, (175, 175, 175), 1, cv2.LINE_AA)
            y += 20
        else:
            cv2.putText(frame, "Calibration guidee terminee", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.54, (95, 210, 125), 2, cv2.LINE_AA)
            y += 24

        return y

    def draw_overlay(self, frame, raw_distance: Optional[float], filtered_distance: Optional[float]) -> None:
        h, w = frame.shape[:2]
        panel_w = min(620, int(w * 0.52))
        panel_h = min(520, int(h * 0.75))
        x0, y0 = 20, 20

        panel = frame.copy()
        cv2.rectangle(panel, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 24), -1)
        cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (210, 210, 210), 2)

        y = y0 + 30
        cv2.putText(frame, "MODE CALIBRATION", (x0 + 14, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
        y += 28

        instructions = "Workflow simple: G -> ESPACE x4 -> S -> K"
        cv2.putText(frame, instructions, (x0 + 14, y), cv2.FONT_HERSHEY_DUPLEX, 0.50, (210, 210, 210), 1, cv2.LINE_AA)
        y += 26
        cv2.putText(frame, "Touches: [ ] param | +/- fin | { } large | R recharger", (x0 + 14, y), cv2.FONT_HERSHEY_DUPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        y += 22

        if self.guided_active:
            compact_w = panel_w - 28
            y = self._draw_compact_guided_block(frame, x0 + 14, y, compact_w)
            y += 6

        raw_text = f"Distance brute: {raw_distance:.2f} m" if raw_distance is not None else "Distance brute: n/a"
        fil_text = f"Distance filtree: {filtered_distance:.2f} m" if filtered_distance is not None else "Distance filtree: n/a"
        cv2.putText(frame, raw_text, (x0 + 14, y), cv2.FONT_HERSHEY_DUPLEX, 0.52, (140, 220, 160), 2, cv2.LINE_AA)
        y += 24
        cv2.putText(frame, fil_text, (x0 + 14, y), cv2.FONT_HERSHEY_DUPLEX, 0.52, (140, 220, 160), 2, cv2.LINE_AA)
        y += 30

        if not self.guided_active:
            for idx, param in enumerate(self.params):
                value = getattr(self.config, param.name)
                color = (235, 235, 235) if idx == self.selected_index else (175, 175, 175)
                cv2.putText(
                    frame,
                    f"{param.label}: {value:.{param.decimals}f}",
                    (x0 + 18, y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.52,
                    color,
                    2 if idx == self.selected_index else 1,
                    cv2.LINE_AA,
                )
                y += 24

        bar_y = y0 + panel_h - 80
        self._draw_distance_bar(frame, raw_distance, x0 + 16, bar_y, panel_w - 32, 28)

        if self.status_message and not self.guided_active:
            cv2.putText(
                frame,
                self.status_message,
                (x0 + 14, y0 + panel_h - 12),
                cv2.FONT_HERSHEY_DUPLEX,
                0.52,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )

