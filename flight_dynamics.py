"""Deterministic 3-D flight dynamics for the swarm simulator."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional, Tuple

Vector3 = Tuple[float, float, float]


def _add(a: Vector3, b: Vector3) -> Vector3:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return tuple(x - y for x, y in zip(a, b))  # type: ignore[return-value]


def _scale(value: Vector3, factor: float) -> Vector3:
    return tuple(component * factor for component in value)  # type: ignore[return-value]


def _norm(value: Vector3) -> float:
    return sqrt(sum(component * component for component in value))


def _clamp_norm(value: Vector3, maximum: float) -> Vector3:
    magnitude = _norm(value)
    if magnitude <= maximum or magnitude == 0:
        return value
    return _scale(value, maximum / magnitude)


@dataclass(frozen=True)
class FlightDynamicsConfig:
    """Physical and battery limits used by :class:`FlightDynamics`."""

    max_acceleration: float = 4.0
    max_speed: float = 12.0
    drag_coefficient: float = 0.12
    battery_capacity: float = 100.0
    idle_drain_per_second: float = 0.08
    movement_drain_per_second: float = 0.018
    wind_drain_per_second: float = 0.006

    def __post_init__(self) -> None:
        if self.max_acceleration <= 0 or self.max_speed <= 0:
            raise ValueError("acceleration and speed limits must be positive")
        if self.drag_coefficient < 0 or self.battery_capacity <= 0:
            raise ValueError("drag must be non-negative and battery capacity positive")
        if min(
            self.idle_drain_per_second,
            self.movement_drain_per_second,
            self.wind_drain_per_second,
        ) < 0:
            raise ValueError("battery drain rates must be non-negative")


@dataclass(frozen=True)
class FlightState:
    position: Vector3 = (0.0, 0.0, 0.0)
    velocity: Vector3 = (0.0, 0.0, 0.0)
    battery: float = 100.0


class FlightDynamics:
    """Integrates bounded acceleration, drag, wind and battery depletion."""

    def __init__(self, config: Optional[FlightDynamicsConfig] = None) -> None:
        self.config = config or FlightDynamicsConfig()

    def step(
        self,
        state: FlightState,
        desired_acceleration: Vector3,
        *,
        wind_velocity: Vector3 = (0.0, 0.0, 0.0),
        dt: float = 0.1,
    ) -> FlightState:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if state.battery < 0:
            raise ValueError("battery cannot be negative")

        acceleration = _clamp_norm(desired_acceleration, self.config.max_acceleration)
        relative_velocity = _sub(state.velocity, wind_velocity)
        drag = _scale(relative_velocity, -self.config.drag_coefficient)
        velocity = _clamp_norm(
            _add(state.velocity, _scale(_add(acceleration, drag), dt)),
            self.config.max_speed,
        )
        position = _add(state.position, _scale(velocity, dt))

        drain = dt * (
            self.config.idle_drain_per_second
            + self.config.movement_drain_per_second * _norm(velocity) ** 2
            + self.config.wind_drain_per_second * _norm(wind_velocity) ** 2
        )
        battery = max(0.0, state.battery - drain)
        return FlightState(position=position, velocity=velocity, battery=battery)

    def velocity_command(
        self,
        current_velocity: Vector3,
        desired_velocity: Vector3,
    ) -> Vector3:
        """Convert a velocity target into a bounded acceleration command."""
        return _clamp_norm(
            _sub(desired_velocity, current_velocity),
            self.config.max_acceleration,
        )
