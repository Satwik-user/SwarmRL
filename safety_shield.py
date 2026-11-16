"""Continuous swept-path safety filter for drone velocity commands."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Optional, Sequence, Tuple

from flight_dynamics import Vector3


def _add(a: Vector3, b: Vector3) -> Vector3:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return tuple(x - y for x, y in zip(a, b))  # type: ignore[return-value]


def _scale(value: Vector3, factor: float) -> Vector3:
    return tuple(component * factor for component in value)  # type: ignore[return-value]


def _dot(a: Vector3, b: Vector3) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(value: Vector3) -> float:
    return sqrt(_dot(value, value))


def _normalize(value: Vector3) -> Vector3:
    magnitude = _norm(value)
    return _scale(value, 1.0 / magnitude) if magnitude else (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class AABB:
    minimum: Vector3
    maximum: Vector3

    def __post_init__(self) -> None:
        if any(low > high for low, high in zip(self.minimum, self.maximum)):
            raise ValueError("AABB minimum must not exceed maximum")

    def expanded(self, amount: float) -> "AABB":
        return AABB(
            _sub(self.minimum, (amount, amount, amount)),
            _add(self.maximum, (amount, amount, amount)),
        )


@dataclass(frozen=True)
class Neighbor:
    position: Vector3
    velocity: Vector3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SafetyConfig:
    drone_radius: float = 0.5
    separation_margin: float = 0.5
    prediction_horizon: float = 1.0
    max_speed: float = 12.0
    repulsion_gain: float = 2.0

    def __post_init__(self) -> None:
        if min(
            self.drone_radius,
            self.separation_margin,
            self.prediction_horizon,
            self.max_speed,
        ) <= 0:
            raise ValueError("safety distances, horizon and speed must be positive")
        if self.repulsion_gain < 0:
            raise ValueError("repulsion gain must be non-negative")


@dataclass(frozen=True)
class SafetyResult:
    velocity: Vector3
    safe: bool
    modified: bool
    reason: str
    minimum_clearance: float


def segment_intersects_aabb(start: Vector3, end: Vector3, box: AABB) -> bool:
    """Return whether a line segment intersects an axis-aligned box.

    The slab test is continuous; it does not sample points and therefore
    catches high-speed tunnelling through thin obstacles.
    """
    direction = _sub(end, start)
    lower, upper = 0.0, 1.0
    for origin, delta, minimum, maximum in zip(
        start, direction, box.minimum, box.maximum
    ):
        if delta == 0:
            if origin < minimum or origin > maximum:
                return False
            continue
        first = (minimum - origin) / delta
        second = (maximum - origin) / delta
        if first > second:
            first, second = second, first
        lower = max(lower, first)
        upper = min(upper, second)
        if lower > upper:
            return False
    return True


def _distance_to_segment(point: Vector3, start: Vector3, end: Vector3) -> float:
    direction = _sub(end, start)
    denominator = _dot(direction, direction)
    if denominator == 0:
        return _norm(_sub(point, start))
    fraction = max(0.0, min(1.0, _dot(_sub(point, start), direction) / denominator))
    return _norm(_sub(point, _add(start, _scale(direction, fraction))))


class SafetyShield:
    """Projects desired velocities onto a deterministic collision-free set."""

    def __init__(self, config: Optional[SafetyConfig] = None) -> None:
        self.config = config or SafetyConfig()

    @property
    def clearance(self) -> float:
        return self.config.drone_radius + self.config.separation_margin

    def is_path_safe(
        self,
        position: Vector3,
        velocity: Vector3,
        *,
        obstacles: Sequence[AABB] = (),
        neighbors: Sequence[Neighbor] = (),
    ) -> bool:
        endpoint = _add(
            position, _scale(velocity, self.config.prediction_horizon)
        )
        if any(
            segment_intersects_aabb(position, endpoint, obstacle.expanded(self.clearance))
            for obstacle in obstacles
        ):
            return False
        for neighbor in neighbors:
            relative_end = _add(
                _sub(position, neighbor.position),
                _scale(
                    _sub(velocity, neighbor.velocity),
                    self.config.prediction_horizon,
                ),
            )
            if _distance_to_segment(
                (0.0, 0.0, 0.0),
                _sub(position, neighbor.position),
                relative_end,
            ) <= self.clearance:
                return False
        return True

    def filter_velocity(
        self,
        position: Vector3,
        desired_velocity: Vector3,
        *,
        obstacles: Sequence[AABB] = (),
        neighbors: Sequence[Neighbor] = (),
    ) -> SafetyResult:
        desired = _scale(
            _normalize(desired_velocity),
            min(_norm(desired_velocity), self.config.max_speed),
        )
        if self.is_path_safe(position, desired, obstacles=obstacles, neighbors=neighbors):
            return SafetyResult(
                desired, True, False, "desired velocity is safe", self._clearance(
                    position, desired, obstacles, neighbors
                )
            )

        candidates = [(0.0, 0.0, 0.0)]
        for axis in range(3):
            for sign in (-1.0, 1.0):
                vector = list(desired)
                vector[axis] = 0.0
                vector[axis] += sign * self.config.max_speed * 0.5
                candidates.append(tuple(vector))  # type: ignore[arg-type]
        for obstacle in obstacles:
            away = _sub(position, _scale(_add(obstacle.minimum, obstacle.maximum), 0.5))
            candidates.append(_scale(_normalize(away), self.config.max_speed))
        for neighbor in neighbors:
            away = _sub(position, neighbor.position)
            candidates.append(_add(neighbor.velocity, _scale(_normalize(away), self.config.max_speed)))

        safe_candidates = [
            candidate
            for candidate in candidates
            if self.is_path_safe(position, candidate, obstacles=obstacles, neighbors=neighbors)
        ]
        if not safe_candidates:
            return SafetyResult(
                (0.0, 0.0, 0.0),
                False,
                True,
                "no safe velocity found within the speed limit",
                self._clearance(position, (0.0, 0.0, 0.0), obstacles, neighbors),
            )
        projected = min(
            safe_candidates,
            key=lambda candidate: _norm(_sub(candidate, desired)),
        )
        return SafetyResult(
            projected,
            True,
            True,
            "unsafe command projected by deterministic shield",
            self._clearance(position, projected, obstacles, neighbors),
        )

    def _clearance(
        self,
        position: Vector3,
        velocity: Vector3,
        obstacles: Iterable[AABB],
        neighbors: Iterable[Neighbor],
    ) -> float:
        endpoint = _add(position, _scale(velocity, self.config.prediction_horizon))
        values = []
        for obstacle in obstacles:
            center = _scale(_add(obstacle.minimum, obstacle.maximum), 0.5)
            values.append(_distance_to_segment(center, position, endpoint) - self.clearance)
        for neighbor in neighbors:
            values.append(
                _distance_to_segment(
                    neighbor.position, position, endpoint
                ) - self.clearance
            )
        return min(values, default=float("inf"))
