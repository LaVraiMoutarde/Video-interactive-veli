from dataclasses import asdict, dataclass


@dataclass
class AppConfig:
    camera_id: int = 0
    camera_hfov_deg: float = 70.0
    calibration_file_path: str = "calibration.json"

    # Window origins in virtual desktop coordinates.
    # Typical setup: laptop screen at (0, 0), projector at (1920, 0).
    debug_monitor_x: int = 0
    debug_monitor_y: int = 0
    projector_monitor_x: int = 1920
    projector_monitor_y: int = 0

    video1_path: str = "video 1.mp4"
    video2_path: str = "video 2.mp4"

    pose_min_detection_confidence: float = 0.5
    pose_min_tracking_confidence: float = 0.5
    landmark_visibility_threshold: float = 0.5

    hand_above_shoulder_margin: float = 0.04
    start_gesture_hold_seconds: float = 3.0
    switch_gesture_hold_seconds: float = 1.5
    gesture_cooldown_seconds: float = 1.0
    gesture_dropout_grace_seconds: float = 0.45

    ema_alpha: float = 0.25
    max_distance_jump_m_per_s: float = 2.0

    # Launch video 1 only when user is in this distance window.
    launch_min_distance_m: float = 3.0
    launch_max_distance_m: float = 3.7
    launch_window_hysteresis_m: float = 0.08

    switch_distance_m: float = 2.0
    video1_min_travel_m: float = 0.8
    switch_hysteresis_m: float = 0.12
    video2_far_distance_m: float = 4.5
    video2_min_travel_m: float = 0.8

    # Right-hand gesture for switching to video 2 is only accepted in this window.
    video2_trigger_min_distance_m: float = 1.2
    video2_trigger_max_distance_m: float = 3.7
    video2_trigger_hysteresis_m: float = 0.08

    tracking_lost_timeout_s: float = 0.75
    tracking_reset_delay_s: float = 2.0

    shoulder_width_m: float = 0.38
    minimum_shoulder_px: float = 12.0

    def to_dict(self) -> dict:
        return asdict(self)


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
