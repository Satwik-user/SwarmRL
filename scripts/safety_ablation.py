"""Reproducible safety-shield ablation experiment.

Run with ``python -m scripts.safety_ablation`` to compare raw policy velocities with
the deterministic shield over identical obstacle and neighbor scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from src.safety.safety_shield import (
    AABB,
    Neighbor,
    SafetyShield,
    Vector3,
    segment_intersects_aabb,
)


@dataclass(frozen=True)
class Scenario:
    position: Vector3
    desired_velocity: Vector3
    obstacle: AABB
    neighbor: Neighbor


def scenarios(count: int = 100) -> Iterable[Scenario]:
    for index in range(count):
        offset = (index % 10) * 0.15
        yield Scenario(
            position=(0.0, offset, 0.0),
            desired_velocity=(8.0, 0.0, 0.0),
            obstacle=AABB((3.0, -0.75, -0.75), (4.0, 0.75, 0.75)),
            neighbor=Neighbor((2.5, 2.0 + offset, 0.0)),
        )


def run_ablation(items: Iterable[Scenario] = scenarios()) -> Dict[str, float]:
    shield = SafetyShield()
    raw_collisions = 0
    shield_collisions = 0
    raw_near_misses = 0
    shield_near_misses = 0
    total = 0

    for item in items:
        total += 1
        raw_end = (
            item.position[0] + item.desired_velocity[0],
            item.position[1] + item.desired_velocity[1],
            item.position[2] + item.desired_velocity[2],
        )
        raw_collision = segment_intersects_aabb(
            item.position,
            raw_end,
            item.obstacle.expanded(shield.clearance),
        )
        raw_collisions += int(raw_collision)
        raw_near_misses += int(
            segment_intersects_aabb(item.position, raw_end, item.obstacle)
            and not raw_collision
        )

        result = shield.filter_velocity(
            item.position,
            item.desired_velocity,
            obstacles=(item.obstacle,),
            neighbors=(item.neighbor,),
        )
        shield_end = (
            item.position[0] + result.velocity[0],
            item.position[1] + result.velocity[1],
            item.position[2] + result.velocity[2],
        )
        shield_collision = segment_intersects_aabb(
            item.position,
            shield_end,
            item.obstacle.expanded(shield.clearance),
        )
        shield_collisions += int(shield_collision or not result.safe)
        shield_near_misses += int(
            segment_intersects_aabb(item.position, shield_end, item.obstacle)
            and not shield_collision
        )

    total = max(1, total)
    return {
        "scenarios": float(total),
        "shield_off_collisions": float(raw_collisions),
        "shield_on_collisions": float(shield_collisions),
        "shield_off_near_misses": float(raw_near_misses),
        "shield_on_near_misses": float(shield_near_misses),
    }


if __name__ == "__main__":
    for metric, value in run_ablation().items():
        print(f"{metric}: {int(value) if value.is_integer() else value}")
