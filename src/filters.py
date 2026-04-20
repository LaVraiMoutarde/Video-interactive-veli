from __future__ import annotations


class EMAFilter:
    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._value = None

    def reset(self) -> None:
        self._value = None

    def update(self, sample: float) -> float:
        if self._value is None:
            self._value = sample
            return sample

        self._value = (self.alpha * sample) + ((1.0 - self.alpha) * self._value)
        return self._value


class SlewRateLimiter:
    def __init__(self, max_delta_per_second: float) -> None:
        if max_delta_per_second <= 0.0:
            raise ValueError("max_delta_per_second must be > 0")
        self.max_delta_per_second = max_delta_per_second
        self._last_value = None

    def reset(self) -> None:
        self._last_value = None

    def update(self, target: float, dt_s: float) -> float:
        if self._last_value is None or dt_s <= 0.0:
            self._last_value = target
            return target

        max_delta = self.max_delta_per_second * dt_s
        delta = target - self._last_value
        if delta > max_delta:
            delta = max_delta
        elif delta < -max_delta:
            delta = -max_delta

        self._last_value = self._last_value + delta
        return self._last_value
