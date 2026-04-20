from __future__ import annotations

import math
import threading
import time
import traceback

import cv2
import numpy as np

from src.calibration_mode import CalibrationModeController
from src.calibration_store import load_persistent_config, save_persistent_config
from src.distance_mapper import DistanceEstimator, TemporalMapper
from src.filters import EMAFilter, SlewRateLimiter
from src.gesture_recognizer import GestureRecognizer
from src.pose_detector import PoseDetector
from src.single_user_tracker import SingleUserTracker
from src.state_machine import InteractionStateMachine, InteractionState
from src.video_player import DualVideoRenderer


POSE_EDGES = [
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
]


def draw_debug_overlay(frame, state_output, distance_m, filtered_distance_m, user_lock_active):
    lines = [
        f"state: {state_output.state}",
        f"video: {state_output.active_video}",
        f"progress: {state_output.progress:.3f}",
        f"distance_m: {distance_m:.2f}" if distance_m is not None else "distance_m: n/a",
        f"filtered_m: {filtered_distance_m:.2f}" if filtered_distance_m is not None else "filtered_m: n/a",
        f"launch_window_active: {'ON' if state_output.launch_window_active else 'OFF'}",
        f"video2_window_active: {'ON' if state_output.video2_trigger_window_active else 'OFF'}",
        f"single_user_lock: {'ON' if user_lock_active else 'OFF'}",
        f"reset in: {state_output.countdown_s:.1f}s" if state_output.countdown_s is not None else "reset in: -",
        f"info: {state_output.info_message}" if state_output.info_message else "info: -",
        "q: quit | f: fullscreen both | c: save calibration | k: calibration mode | b: detection",
    ]

    y = 28
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_DUPLEX, 0.62, (30, 240, 30), 2, cv2.LINE_AA)
        y += 28


def setup_window(window_name: str, x: int, y: int, fullscreen: bool) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, int(x), int(y))
    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
    )


def set_windows_fullscreen(window_names: tuple[str, ...], fullscreen: bool) -> None:
    mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
    for name in window_names:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, mode)


def draw_camera_pose_inset(output_frame, camera_frame, pose_frame):
    out_h, out_w = output_frame.shape[:2]
    inset_w = max(220, int(out_w * 0.24))
    inset_h = max(140, int(out_h * 0.24))
    x0, y0 = 14, 14

    preview = cv2.resize(camera_frame, (inset_w, inset_h), interpolation=cv2.INTER_AREA)

    for start, end in POSE_EDGES:
        p0 = pose_frame.landmarks.get(start)
        p1 = pose_frame.landmarks.get(end)
        if p0 is None or p1 is None:
            continue

        x_start = int(p0[0] * inset_w)
        y_start = int(p0[1] * inset_h)
        x_end = int(p1[0] * inset_w)
        y_end = int(p1[1] * inset_h)
        cv2.line(preview, (x_start, y_start), (x_end, y_end), (0, 220, 220), 2, cv2.LINE_AA)

    for _, (x_norm, y_norm, _) in pose_frame.landmarks.items():
        x = int(x_norm * inset_w)
        y = int(y_norm * inset_h)
        cv2.circle(preview, (x, y), 4, (20, 255, 40), -1, cv2.LINE_AA)
        cv2.circle(preview, (x, y), 7, (20, 120, 20), 1, cv2.LINE_AA)

    output_frame[y0 : y0 + inset_h, x0 : x0 + inset_w] = preview
    cv2.rectangle(output_frame, (x0 - 2, y0 - 2), (x0 + inset_w + 2, y0 + inset_h + 2), (245, 245, 245), 2)
    cv2.putText(output_frame, "camera tracking", (x0 + 8, y0 + inset_h + 20), cv2.FONT_HERSHEY_DUPLEX, 0.55, (240, 240, 240), 2, cv2.LINE_AA)


def _draw_step_line(frame, x, y, text, done):
    if done:
        color = (95, 220, 135)
        marker = "[OK]"
    else:
        color = (215, 215, 215)
        marker = "[  ]"
    cv2.putText(frame, f"{marker} {text}", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_bottom_gesture_bar(frame, progress, primary_text, secondary_text, color):
    h, w = frame.shape[:2]
    bar_margin_x = max(28, int(w * 0.04))
    bar_w = w - (bar_margin_x * 2)
    bar_h = 32
    bar_y = h - 56

    overlay = frame.copy()
    cv2.rectangle(overlay, (bar_margin_x - 8, bar_y - 12), (bar_margin_x + bar_w + 8, bar_y + bar_h + 32), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    cv2.rectangle(frame, (bar_margin_x, bar_y), (bar_margin_x + bar_w, bar_y + bar_h), (38, 38, 38), -1)
    cv2.rectangle(frame, (bar_margin_x, bar_y), (bar_margin_x + bar_w, bar_y + bar_h), (165, 165, 165), 1)

    fill_ratio = min(1.0, max(0.0, progress))
    fill_w = int(bar_w * fill_ratio)
    if fill_w > 2:
        cv2.rectangle(frame, (bar_margin_x + 2, bar_y + 2), (bar_margin_x + fill_w - 2, bar_y + bar_h - 2), color, -1)

    if primary_text:
        cv2.putText(frame, primary_text, (bar_margin_x, bar_y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.72, (242, 242, 242), 2, cv2.LINE_AA)
    if secondary_text:
        cv2.putText(frame, secondary_text, (bar_margin_x, bar_y + 55), cv2.FONT_HERSHEY_DUPLEX, 0.52, (205, 205, 205), 1, cv2.LINE_AA)


def draw_prestart_overlay(frame, cfg, state_output, has_person, distance_m, gestures):
    h, w = frame.shape[:2]
    panel_w = min(int(w * 0.72), 980)
    panel_h = min(int(h * 0.46), 430)
    x0 = (w - panel_w) // 2
    y0 = max(20, int(h * 0.08))

    panel = frame.copy()
    cv2.rectangle(panel, (x0, y0), (x0 + panel_w, y0 + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(panel, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (220, 220, 220), 2)

    title = "PRE-PHASE GUIDEE"
    cv2.putText(frame, title, (x0 + 22, y0 + 42), cv2.FONT_HERSHEY_DUPLEX, 1.0, (230, 235, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, "Suivez ces etapes pour lancer l'experience", (x0 + 22, y0 + 72), cv2.FONT_HERSHEY_DUPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)

    in_launch_window = state_output.launch_window_active

    both_hands_ok = gestures.both_hands_up and in_launch_window
    hold_ratio = gestures.both_hands_hold_ratio if in_launch_window else 0.0
    hold_ratio = min(1.0, max(0.0, hold_ratio))

    y = y0 + 115
    _draw_step_line(frame, x0 + 28, y, "Entrez dans le champ camera", has_person)
    y += 38
    _draw_step_line(
        frame,
        x0 + 28,
        y,
        f"Placez-vous entre {cfg.launch_min_distance_m:.1f}m et {cfg.launch_max_distance_m:.1f}m",
        in_launch_window,
    )
    y += 38
    _draw_step_line(frame, x0 + 28, y, "Levez les 2 mains", both_hands_ok)
    y += 38

    hint = state_output.info_message if state_output.info_message else "Levez les 2 mains pour demarrer"
    cv2.putText(frame, hint, (x0 + 22, y0 + panel_h - 16), cv2.FONT_HERSHEY_DUPLEX, 0.62, (235, 220, 165), 2, cv2.LINE_AA)

    draw_bottom_gesture_bar(
        frame=frame,
        progress=hold_ratio,
        primary_text="Maintenir les 2 mains levees pour lancer",
        secondary_text=f"{hold_ratio * cfg.start_gesture_hold_seconds:.2f}s / {cfg.start_gesture_hold_seconds:.2f}s",
        color=(90, 220, 255),
    )


def draw_video2_switch_overlay(frame, cfg, state_output, distance_m, gestures):
    h, w = frame.shape[:2]
    panel_w = min(int(w * 0.56), 820)
    panel_h = min(int(h * 0.28), 250)
    x0 = (w - panel_w) // 2
    y0 = max(20, int(h * 0.10))

    panel = frame.copy()
    cv2.rectangle(panel, (x0, y0), (x0 + panel_w, y0 + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(panel, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (220, 220, 220), 2)

    in_v2_window = state_output.video2_trigger_window_active

    right_ok = gestures.right_hand_up and in_v2_window
    hold_ratio = gestures.right_hand_hold_ratio if in_v2_window else 0.0
    hold_ratio = min(1.0, max(0.0, hold_ratio))

    marker = "[OK]" if right_ok else "[  ]"
    marker_color = (100, 220, 145) if right_ok else (205, 205, 205)
    cv2.putText(frame, f"{marker} Main droite levee", (x0 + 20, y0 + 110), cv2.FONT_HERSHEY_DUPLEX, 0.62, marker_color, 2, cv2.LINE_AA)

    hint = state_output.info_message if state_output.info_message else "Main droite pour basculer"
    cv2.putText(frame, hint, (x0 + 20, y0 + panel_h - 16), cv2.FONT_HERSHEY_DUPLEX, 0.60, (235, 220, 165), 2, cv2.LINE_AA)

    draw_bottom_gesture_bar(
        frame=frame,
        progress=hold_ratio,
        primary_text="",
        secondary_text="",
        color=(110, 230, 255),
    )


def draw_startup_cooldown_overlay(frame, seconds_left, elapsed):
    h, w = frame.shape[:2]
    panel_w = min(int(w * 0.62), 860)
    panel_h = min(int(h * 0.30), 250)
    x0 = (w - panel_w) // 2
    y0 = (h - panel_h) // 2

    frame[:, :, :] = (14, 14, 16)
    gradient = np.tile(np.linspace(0, 36, w, dtype=np.uint8), (h, 1))
    frame[:, :, 0] = np.clip(frame[:, :, 0] + gradient // 4, 0, 255)
    frame[:, :, 1] = np.clip(frame[:, :, 1] + gradient // 5, 0, 255)
    frame[:, :, 2] = np.clip(frame[:, :, 2] + gradient // 3, 0, 255)

    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (24, 24, 28), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (210, 210, 210), 2)

    cv2.putText(frame, "initialisation de l'idee", (x0 + 28, y0 + 68), cv2.FONT_HERSHEY_DUPLEX, 1.05, (235, 235, 235), 2, cv2.LINE_AA)
    bar_x = x0 + 28
    bar_y = y0 + 132
    bar_w = panel_w - 56
    bar_h = 24
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (42, 42, 42), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (175, 175, 175), 1)

    progress = 1.0 - min(1.0, max(0.0, seconds_left / 3.0))
    fill_w = int(bar_w * progress)
    if fill_w > 2:
        pulse = int(18 * (0.5 + 0.5 * math.sin(elapsed * 3.0)))
        cv2.rectangle(frame, (bar_x + 2, bar_y + 2), (bar_x + fill_w - 2, bar_y + bar_h - 2), (120 + pulse // 3, 155 + pulse // 2, 230), -1)

def load_renderer_with_min_cooldown(cfg, window_name, min_seconds=3.0):
    result = {"renderer": None, "error": None}

    def _worker():
        try:
            result["renderer"] = DualVideoRenderer(cfg.video1_path, cfg.video2_path)
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        seconds_left = max(0.0, min_seconds - elapsed)

        screen = np.zeros((720, 1280, 3), dtype=np.uint8)
        draw_startup_cooldown_overlay(screen, seconds_left, elapsed)
        cv2.imshow(window_name, screen)

        key = cv2.waitKey(16) & 0xFF
        if key == ord("q"):
            raise RuntimeError("Startup interrupted by user")

        if result["error"] is not None:
            raise RuntimeError(f"Video loading failed: {result['error']}")

        if elapsed >= min_seconds and result["renderer"] is not None:
            return result["renderer"]


def run_camera_diagnostic() -> None:
    print("--- DIAGNOSTIC CAMERAS ---")
    available_ids = []

    for i in range(11):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"[OK] Camera detectee a l'ID: {i}")
                available_ids.append(i)
        cap.release()

    if not available_ids:
        print("[ERREUR] Aucune camera detectee.")
    else:
        print(f"Resume: IDs detectes = {available_ids}")


def open_camera_with_fallback(camera_id):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(camera_id, backend)
        if not cap.isOpened():
            cap.release()
            continue

        ok, _ = cap.read()
        if ok:
            return cap

        cap.release()

    return None


def draw_startup_menu(frame, selected_index, menu_items):
    h, w = frame.shape[:2]
    frame[:, :, :] = (12, 12, 14)

    gradient = np.tile(np.linspace(0, 45, w, dtype=np.uint8), (h, 1))
    frame[:, :, 0] = np.clip(frame[:, :, 0] + gradient // 5, 0, 255)
    frame[:, :, 1] = np.clip(frame[:, :, 1] + gradient // 5, 0, 255)
    frame[:, :, 2] = np.clip(frame[:, :, 2] + gradient // 3, 0, 255)

    panel_w = min(int(w * 0.58), 860)
    panel_h = min(int(h * 0.64), 540)
    x0 = (w - panel_w) // 2
    y0 = (h - panel_h) // 2

    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (24, 24, 28), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (208, 208, 208), 2)

    cv2.putText(frame, "Evolution interactive", (x0 + 28, y0 + 56), cv2.FONT_HERSHEY_DUPLEX, 1.05, (236, 236, 236), 2, cv2.LINE_AA)
    cv2.putText(frame, "Menu de demarrage", (x0 + 30, y0 + 88), cv2.FONT_HERSHEY_DUPLEX, 0.62, (190, 190, 190), 1, cv2.LINE_AA)

    y = y0 + 145
    for idx, label in enumerate(menu_items):
        is_selected = idx == selected_index
        if is_selected:
            cv2.rectangle(frame, (x0 + 22, y - 28), (x0 + panel_w - 22, y + 16), (44, 44, 52), -1)
            cv2.rectangle(frame, (x0 + 22, y - 28), (x0 + panel_w - 22, y + 16), (150, 150, 160), 1)

        color = (235, 235, 235) if is_selected else (176, 176, 176)
        marker = ">" if is_selected else " "
        cv2.putText(frame, f"{marker} {label}", (x0 + 36, y), cv2.FONT_HERSHEY_DUPLEX, 0.82, color, 2, cv2.LINE_AA)
        y += 54

    cv2.putText(
        frame,
        "Z/S ou Fleches: naviguer | Entree/Espace: valider | Q: quitter",
        (x0 + 28, y0 + panel_h - 24),
        cv2.FONT_HERSHEY_DUPLEX,
        0.52,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )


def show_startup_menu(window_name):
    menu_items = [
        "Lancer l'experience",
        "Mode calibration",
        "Test camera",
        "Quitter",
    ]
    actions = ["experience", "calibration", "test_camera", "quit"]
    selected_index = 0

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        draw_startup_menu(frame, selected_index, menu_items)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(16) & 0xFF
        if key in (ord("q"), 27):
            return "quit"
        if key in (ord("w"), ord("z"), 82):
            selected_index = (selected_index - 1) % len(menu_items)
            continue
        if key in (ord("s"), 84):
            selected_index = (selected_index + 1) % len(menu_items)
            continue
        if key in (13, 10, 32):
            return actions[selected_index]


def run_experience(start_in_calibration=False) -> None:
    cfg, calibration_status = load_persistent_config()
    print(calibration_status)

    calibration_mode = CalibrationModeController(cfg)
    if start_in_calibration:
        calibration_mode.enabled = True
        calibration_mode.guided_active = True
        calibration_mode.guided_step = 0
        calibration_mode.status_message = calibration_mode.guided_steps[0][1]

    camera = open_camera_with_fallback(cfg.camera_id)
    if camera is None or not camera.isOpened():
        raise RuntimeError("Unable to open camera. Check camera ID and USB connection.")

    pose_detector = PoseDetector(
        min_detection_confidence=cfg.pose_min_detection_confidence,
        min_tracking_confidence=cfg.pose_min_tracking_confidence,
        visibility_threshold=cfg.landmark_visibility_threshold,
    )
    gesture_recognizer = GestureRecognizer(
        hand_above_shoulder_margin=cfg.hand_above_shoulder_margin,
        start_hold_seconds=cfg.start_gesture_hold_seconds,
        switch_hold_seconds=cfg.switch_gesture_hold_seconds,
        cooldown_seconds=cfg.gesture_cooldown_seconds,
        dropout_grace_seconds=cfg.gesture_dropout_grace_seconds,
    )

    distance_estimator = DistanceEstimator(
        camera_hfov_deg=cfg.camera_hfov_deg,
        shoulder_width_m=cfg.shoulder_width_m,
        minimum_shoulder_px=cfg.minimum_shoulder_px,
    )

    mapper = TemporalMapper(
        switch_distance_m=cfg.switch_distance_m,
        video2_far_distance_m=cfg.video2_far_distance_m,
        video1_min_travel_m=cfg.video1_min_travel_m,
        video2_min_travel_m=cfg.video2_min_travel_m,
    )

    state_machine = InteractionStateMachine(
        mapper=mapper,
        launch_min_distance_m=cfg.launch_min_distance_m,
        launch_max_distance_m=cfg.launch_max_distance_m,
        launch_window_hysteresis_m=cfg.launch_window_hysteresis_m,
        switch_distance_m=cfg.switch_distance_m,
        switch_hysteresis_m=cfg.switch_hysteresis_m,
        video2_trigger_min_distance_m=cfg.video2_trigger_min_distance_m,
        video2_trigger_max_distance_m=cfg.video2_trigger_max_distance_m,
        video2_trigger_hysteresis_m=cfg.video2_trigger_hysteresis_m,
        tracking_lost_timeout_s=cfg.tracking_lost_timeout_s,
        tracking_reset_delay_s=cfg.tracking_reset_delay_s,
    )

    ema_filter = EMAFilter(alpha=cfg.ema_alpha)
    limiter = SlewRateLimiter(max_delta_per_second=cfg.max_distance_jump_m_per_s)
    fullscreen = True
    projector_window = "Projector Experience"
    debug_window = "Debug Monitor"
    setup_window(projector_window, cfg.projector_monitor_x, cfg.projector_monitor_y, fullscreen)
    setup_window(debug_window, cfg.debug_monitor_x, cfg.debug_monitor_y, fullscreen)
    runtime_windows = (projector_window, debug_window)

    renderer = None
    if not calibration_mode.enabled:
        renderer = load_renderer_with_min_cooldown(cfg, projector_window, min_seconds=3.0)
    single_user_tracker = SingleUserTracker()

    last_time = time.perf_counter()
    previous_state = InteractionState.WAITING_START

    def reset_runtime_state() -> None:
        nonlocal distance_estimator, mapper, state_machine, ema_filter, limiter, previous_state

        distance_estimator = DistanceEstimator(
            camera_hfov_deg=cfg.camera_hfov_deg,
            shoulder_width_m=cfg.shoulder_width_m,
            minimum_shoulder_px=cfg.minimum_shoulder_px,
        )

        mapper = TemporalMapper(
            switch_distance_m=cfg.switch_distance_m,
            video2_far_distance_m=cfg.video2_far_distance_m,
            video1_min_travel_m=cfg.video1_min_travel_m,
            video2_min_travel_m=cfg.video2_min_travel_m,
        )

        state_machine = InteractionStateMachine(
            mapper=mapper,
            launch_min_distance_m=cfg.launch_min_distance_m,
            launch_max_distance_m=cfg.launch_max_distance_m,
            launch_window_hysteresis_m=cfg.launch_window_hysteresis_m,
            switch_distance_m=cfg.switch_distance_m,
            switch_hysteresis_m=cfg.switch_hysteresis_m,
            video2_trigger_min_distance_m=cfg.video2_trigger_min_distance_m,
            video2_trigger_max_distance_m=cfg.video2_trigger_max_distance_m,
            video2_trigger_hysteresis_m=cfg.video2_trigger_hysteresis_m,
            tracking_lost_timeout_s=cfg.tracking_lost_timeout_s,
            tracking_reset_delay_s=cfg.tracking_reset_delay_s,
        )

        ema_filter = EMAFilter(alpha=cfg.ema_alpha)
        limiter = SlewRateLimiter(max_delta_per_second=cfg.max_distance_jump_m_per_s)
        previous_state = InteractionState.WAITING_START
        single_user_tracker.unlock()

    try:
        while True:
            ok, camera_frame = camera.read()
            if not ok:
                break

            now = time.perf_counter()
            dt = max(1e-4, now - last_time)
            last_time = now

            raw_pose_frame = pose_detector.detect(camera_frame)

            if (not single_user_tracker.is_locked) and raw_pose_frame.has_person:
                single_user_tracker.lock_from_pose(raw_pose_frame)

            pose_frame = single_user_tracker.filter_pose(raw_pose_frame)
            gestures = gesture_recognizer.update(pose_frame, now)

            raw_distance = distance_estimator.estimate_distance_m(pose_frame)
            filtered_distance = None
            if raw_distance is not None:
                smoothed = ema_filter.update(raw_distance)
                filtered_distance = limiter.update(smoothed, dt)

            if calibration_mode.enabled:
                projector_frame = camera_frame.copy()
                debug_frame = camera_frame.copy()
                calibration_mode.draw_overlay(debug_frame, raw_distance, filtered_distance)

                cv2.imshow(projector_window, projector_frame)
                cv2.imshow(debug_window, debug_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("f"):
                    fullscreen = not fullscreen
                    set_windows_fullscreen(runtime_windows, fullscreen)
                if key == ord("k"):
                    calibration_mode.toggle()
                    reset_runtime_state()
                    calibration_mode.dirty = False
                    if not calibration_mode.enabled and renderer is None:
                        renderer = load_renderer_with_min_cooldown(cfg, projector_window, min_seconds=3.0)
                    continue

                action = calibration_mode.handle_key(key, raw_distance, filtered_distance)
                if action == "save":
                    try:
                        saved_path = save_persistent_config(cfg)
                        calibration_mode.status_message = f"saved: {saved_path}"
                    except ValueError as exc:
                        calibration_mode.status_message = str(exc)
                elif action == "reload":
                    reloaded_cfg, status = load_persistent_config()
                    for field_name, value in reloaded_cfg.to_dict().items():
                        setattr(cfg, field_name, value)
                    reset_runtime_state()
                    calibration_mode.dirty = False
                    calibration_mode.status_message = status
                elif action is not None:
                    calibration_mode.status_message = action
                continue

            state_output = state_machine.update(
                now_s=now,
                has_person=pose_frame.has_person,
                distance_m=filtered_distance,
                gestures=gestures,
            )

            if state_output.state == InteractionState.TRACKING_LOST:
                single_user_tracker.unlock()

            if previous_state == InteractionState.TRACKING_LOST and state_output.state == InteractionState.WAITING_START:
                ema_filter.reset()
                limiter.reset()

            previous_state = state_output.state

            if renderer is None:
                renderer = load_renderer_with_min_cooldown(cfg, projector_window, min_seconds=3.0)

            if state_output.active_video in (1, 2):
                base_frame = renderer.render(state_output.active_video, state_output.progress)
            else:
                base_frame = camera_frame.copy()

            projector_frame = base_frame.copy()

            if state_output.state == InteractionState.WAITING_START:
                draw_prestart_overlay(
                    frame=projector_frame,
                    cfg=cfg,
                    state_output=state_output,
                    has_person=pose_frame.has_person,
                    distance_m=filtered_distance,
                    gestures=gestures,
                )
            elif state_output.state == InteractionState.VIDEO2_DETECTION:
                draw_video2_switch_overlay(
                    frame=projector_frame,
                    cfg=cfg,
                    state_output=state_output,
                    distance_m=filtered_distance,
                    gestures=gestures,
                )

            debug_frame = projector_frame.copy()
            draw_camera_pose_inset(debug_frame, camera_frame, pose_frame)
            draw_debug_overlay(debug_frame, state_output, raw_distance, filtered_distance, single_user_tracker.is_locked)

            cv2.imshow(projector_window, projector_frame)
            cv2.imshow(debug_window, debug_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):
                fullscreen = not fullscreen
                set_windows_fullscreen(runtime_windows, fullscreen)
            if key == ord("c"):
                try:
                    saved_path = save_persistent_config(cfg)
                    print(f"calibration saved: {saved_path}")
                except ValueError as exc:
                    print(str(exc))
            if key == ord("k"):
                calibration_mode.toggle()
                reset_runtime_state()
            if key in (ord("b"), ord("B")):
                reset_runtime_state()
                continue

    finally:
        pose_detector.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    window_name = "Video Interaction"

    while True:
        action = show_startup_menu(window_name)
        if action == "quit":
            cv2.destroyAllWindows()
            break
        if action == "test_camera":
            run_camera_diagnostic()
            continue

        try:
            cv2.destroyWindow(window_name)
            for _ in range(3):
                cv2.waitKey(1)
            run_experience(start_in_calibration=(action == "calibration"))
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[ERREUR] Execution interrompue: {exc}")
            traceback.print_exc()
            print("Retour au menu de demarrage...")
            continue

