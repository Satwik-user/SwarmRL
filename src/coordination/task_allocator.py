"""
task_allocator.py  --  Owner: Somnath Saha
Dynamic role / task allocation by AUCTION.

IDEA: Tasks appear (scan here, deliver kit there, relay comms over there).
Every drone computes a BID = "how good am I for this task?" Highest bid wins.
Because every drone sees the same bids and uses the same tie-break rule, all
drones reach the SAME assignment with no central boss.

Standard library only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

Vec3 = Tuple[float, float, float]


class Role(str, Enum):
    THERMAL_SCANNER = "thermal_scanner"   # searches with heat camera
    KIT_TRANSPORT = "kit_transport"       # carries emergency kit to a survivor
    COMMS_RELAY = "comms_relay"           # hovers to extend radio range


@dataclass
class DroneState:
    # >>> NEEDS FROM SATWIK: his environment must supply position, battery and alive
    # >>> for every drone each step so you can build these objects.
    drone_id: int
    position: Vec3
    battery: float                         # fraction 0..1
    capabilities: Set[Role]                # roles this airframe can physically do
    speed: float = 8.0                     # Satwik's environment max speed (m/s)
    base: Vec3 = (0.0, 0.0, 0.0)           # return point
    # >>> NEEDS FROM ARKENDU: battery fraction used per metre flown. Ask him for the real
    # >>> number from flight_dynamics.py and replace 0.0005 below (must match exactly).
    energy_per_meter: float = 0.0005   # PLACEHOLDER: replace with the value Arkendu gives you
    alive: bool = True
    current_task: Optional[int] = None


@dataclass
class Task:
    task_id: int
    role: Role                             # which role is needed
    position: Vec3                         # where it must be done
    priority: float = 1.0                  # bigger = more urgent (locked survivor ~ 10)


def dist(a: Vec3, b: Vec3) -> float:
    return math.dist(a, b)  # Euclidean distance


# >>> YOU (SOMNATH) MUST: write the glue that turns locked survivors into tasks
# >>> (see tasks_from_survivors below) and call run_auction every few steps.
def bid(drone: DroneState, task: Task, reserve: float = 0.15,
        distance_weight: float = 1.0) -> Optional[float]:
    """
    Return the drone's bid, or None if it must NOT take the task.
    bid = priority_value - travel_cost   (higher is better)
    Feasibility: drone must be alive, capable, and keep `reserve` battery
    after going to the task AND then flying home.
    """
    if not drone.alive or task.role not in drone.capabilities:
        return None
    d_to = dist(drone.position, task.position)
    d_home = dist(task.position, drone.base)
    need = (d_to + d_home) * drone.energy_per_meter + reserve
    if drone.battery < need:
        return None  # would strand itself -> refuse
    # Cost is travel TIME; slightly favours drones with more battery left.
    cost = distance_weight * (d_to / max(drone.speed, 1e-6))
    return task.priority * 10.0 - cost + 0.5 * drone.battery


def run_auction(drones: List[DroneState], tasks: List[Task]) -> Dict[int, int]:
    """
    Greedy global auction. Repeat: take the single best (drone, task) bid among
    free drones and open tasks, assign it, remove both. Returns {task_id: drone_id}.
    Deterministic tie-break: higher bid, then lower drone_id, then lower task_id,
    so every drone running this on the same inputs gets the same answer.
    NOTE: greedy is fast and consistent but not guaranteed globally optimal.
    """
    free = {d.drone_id: d for d in drones if d.alive and d.current_task is None}
    open_tasks = {t.task_id: t for t in tasks}
    assignment: Dict[int, int] = {}
    while free and open_tasks:
        best: Optional[Tuple[float, int, int]] = None
        for d in free.values():
            for t in open_tasks.values():
                b = bid(d, t)
                if b is None:
                    continue
                cand = (b, -d.drone_id, -t.task_id)  # max() picks high bid, low ids
                if best is None or cand > best:
                    best = cand
        if best is None:
            break  # nobody can take any remaining task
        _, neg_d, neg_t = best
        did, tid = -neg_d, -neg_t
        assignment[tid] = did
        free[did].current_task = tid  # mark busy
        del free[did]
        del open_tasks[tid]
    return assignment


_ROLE_CODE = {Role.THERMAL_SCANNER: 1, Role.KIT_TRANSPORT: 2, Role.COMMS_RELAY: 3}


def task_id_for(role: Role, cell: Tuple[int, int, int]) -> int:
    """Deterministic task id from (role, grid cell). Every drone that knows about the
    same survivor computes the SAME id, so claims/duplicates can be matched without
    a central counter. Cell coordinates must be within [-500, 499]."""
    assert all(-500 <= c <= 499 for c in cell)
    cx, cy, cz = (c + 500 for c in cell)
    return _ROLE_CODE[role] * 10**12 + (cx * 1000 + cy) * 1000 + cz


def tasks_from_survivors(survivors, cell_of, priority: float = 10.0) -> List[Task]:
    """Locked survivors -> KIT_TRANSPORT tasks (high priority).
    survivors = ConsensusEngine.locked_survivors(); cell_of = ConsensusEngine.cell_of."""
    return [Task(task_id_for(Role.KIT_TRANSPORT, cell_of(pos)), Role.KIT_TRANSPORT,
                 tuple(pos), priority) for pos, _conf in survivors]


def tasks_for_verification(candidates, cell_of, priority: float = 5.0) -> List[Task]:
    """Suspected-but-unconfirmed survivors -> THERMAL_SCANNER 'go look again' tasks
    (medium priority). candidates = ConsensusEngine.unlocked_candidates()."""
    return [Task(task_id_for(Role.THERMAL_SCANNER, cell_of(pos)), Role.THERMAL_SCANNER,
                 tuple(pos), priority) for pos, _conf in candidates]


def release_tasks_of(drones: List[DroneState], dead_ids: Set[int]) -> List[int]:
    """When drones die, free the tasks they held so they can be re-auctioned.
    Returns the task ids that became unassigned (re-create them as Tasks)."""
    freed = []
    for d in drones:
        if d.drone_id in dead_ids:
            d.alive = False
            if d.current_task is not None:
                freed.append(d.current_task)
                d.current_task = None
    return freed