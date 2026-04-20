from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Tuple, get_type_hints

from .config import AppConfig


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _fallback_calibration_path() -> Path:
    home = Path.home()
    return home / ".video_interaction" / "calibration.json"


def _resolve_calibration_path(path_value: str) -> Path:
    raw = Path(path_value)
    if raw.is_absolute():
        return raw
    # Keep relative config files anchored to the project, not the terminal cwd.
    return _project_root() / raw


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _coerce_value(value: Any, expected_type: Any) -> Any:
    if expected_type is int:
        return int(value)
    if expected_type is float:
        return float(value)
    if expected_type is bool:
        return bool(value)
    if expected_type is str:
        return str(value)
    return value


def _filtered_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    field_defs = {f.name: f for f in fields(AppConfig)}
    type_hints = get_type_hints(AppConfig)

    overrides: Dict[str, Any] = {}
    for key, value in raw.items():
        if key not in field_defs:
            continue

        expected_type = type_hints.get(key)
        if expected_type is None:
            overrides[key] = value
            continue

        try:
            overrides[key] = _coerce_value(value, expected_type)
        except (ValueError, TypeError):
            # Skip invalid values and keep default config value.
            continue

    return overrides


def validate_config(config: AppConfig) -> list[str]:
    errors: list[str] = []

    if config.launch_min_distance_m >= config.launch_max_distance_m:
        errors.append("launch_min_distance_m must be lower than launch_max_distance_m")

    if config.video2_trigger_min_distance_m >= config.video2_trigger_max_distance_m:
        errors.append("video2_trigger_min_distance_m must be lower than video2_trigger_max_distance_m")

    if config.video2_trigger_min_distance_m < 1.0:
        errors.append("video2_trigger_min_distance_m must be >= 1.0")

    if not (config.video2_trigger_min_distance_m <= config.switch_distance_m <= config.launch_max_distance_m):
        errors.append("switch_distance_m must stay between video2_trigger_min_distance_m and launch_max_distance_m")

    if config.video2_far_distance_m <= config.launch_max_distance_m:
        errors.append("video2_far_distance_m must be greater than launch_max_distance_m")

    if config.switch_hysteresis_m <= 0:
        errors.append("switch_hysteresis_m must be > 0")

    if config.launch_window_hysteresis_m < 0:
        errors.append("launch_window_hysteresis_m must be >= 0")

    if config.video2_trigger_hysteresis_m < 0:
        errors.append("video2_trigger_hysteresis_m must be >= 0")

    if config.video1_min_travel_m <= 0:
        errors.append("video1_min_travel_m must be > 0")

    if config.video2_min_travel_m <= 0:
        errors.append("video2_min_travel_m must be > 0")

    launch_half_width = (config.launch_max_distance_m - config.launch_min_distance_m) / 2.0
    if config.launch_window_hysteresis_m > launch_half_width:
        errors.append("launch_window_hysteresis_m is too large for launch window width")

    v2_half_width = (config.video2_trigger_max_distance_m - config.video2_trigger_min_distance_m) / 2.0
    if config.video2_trigger_hysteresis_m > v2_half_width:
        errors.append("video2_trigger_hysteresis_m is too large for video2 trigger window width")

    return errors


def save_persistent_config(config: AppConfig) -> Path:
    errors = validate_config(config)
    if errors:
        raise ValueError("Calibration validation failed: " + "; ".join(errors))

    path = _resolve_calibration_path(config.calibration_file_path)
    payload = config.to_dict()

    try:
        _write_json(path, payload)
    except PermissionError:
        fallback_path = _fallback_calibration_path()
        _write_json(fallback_path, payload)
        return fallback_path

    return path


def load_persistent_config() -> Tuple[AppConfig, str]:
    default_config = AppConfig()
    path = _resolve_calibration_path(default_config.calibration_file_path)
    fallback_path = _fallback_calibration_path()

    candidate_paths = [path]
    if fallback_path != path:
        candidate_paths.append(fallback_path)

    readable_path = None
    for candidate in candidate_paths:
        if candidate.exists():
            readable_path = candidate
            break

    if readable_path is None:
        try:
            saved_path = save_persistent_config(default_config)
            return default_config, f"calibration created: {saved_path}"
        except OSError:
            return default_config, "calibration unavailable (permission), using defaults"

    try:
        with readable_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return default_config, "calibration invalid, using defaults"

    if not isinstance(raw, dict):
        return default_config, "calibration invalid format, using defaults"

    overrides = _filtered_overrides(raw)
    merged = default_config.to_dict()
    merged.update(overrides)

    config = AppConfig(**merged)
    return config, f"calibration loaded: {readable_path}"
