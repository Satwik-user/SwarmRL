"""
self_healing_manager.py  --  Owner: Somnath Saha
Heartbeat failure detection + Voronoi sector re-partitioning.

IDEA: Every step each drone says "I'm alive" (heartbeat). If a peer is silent
for 3 steps it is declared failed. Survivors then re-divide the search area:
each map cell is owned by the NEAREST alive drone (a "Voronoi partition"), so a
dead drone's cells are automatically absorbed by its neighbours.

Needs: numpy.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set

import numpy as np


class HeartbeatMonitor:
    """Runs on each drone; tracks when it last heard from each peer."""

    # >>> NEEDS FROM CHANDAN: his comm_mesh.py must deliver heartbeat packets (each drone
    # >>> sends "I'm alive, step N" every step). YOU call record_heartbeat() per packet received.
    def __init__(self, peer_ids: Sequence[int], timeout_intervals: int = 3, start_step: int = 0):
        assert timeout_intervals >= 1
        self.timeout = timeout_intervals
        # Assume everyone was just heard at start so nobody is falsely "dead" at step 0.
        self.last_seen: Dict[int, int] = {p: start_step for p in peer_ids}
        self.failed: Set[int] = set()

    def record_heartbeat(self, peer_id: int, step: int) -> None:
        """Call when a heartbeat packet from `peer_id` is received."""
        # Only FRESHER information counts. Stale rumours (older step) must not
        # resurrect a peer that was already declared failed.
        if step > self.last_seen.get(peer_id, -1):
            self.last_seen[peer_id] = step
            self.failed.discard(peer_id)  # fresh news -> peer is alive again (rejoin)

    def check(self, step: int) -> Set[int]:
        """Call once per step. Returns peers NEWLY declared failed this step.
        Failed = silent for `timeout` consecutive intervals."""
        newly = set()
        for peer, seen in self.last_seen.items():
            if step - seen >= self.timeout:
                if peer not in self.failed:
                    self.failed.add(peer)
                    newly.add(peer)
            else:
                self.failed.discard(peer)  # heard recently enough -> alive
        return newly

    def alive_peers(self) -> List[int]:
        return sorted(p for p in self.last_seen if p not in self.failed)


def voronoi_assign(cell_centers: np.ndarray, drone_ids: Sequence[int],
                   drone_positions: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Assign every cell to its nearest drone.
      cell_centers   : (M,3) array of cell centre coordinates
      drone_ids      : K ids of ALIVE drones
      drone_positions: (K,3) their positions
    Returns {drone_id: array of cell indices it owns}.
    Equal-distance ties are split between the tied drones by a coordinate hash (deterministic).
    """
    cell_centers = np.asarray(cell_centers, dtype=float)
    drone_positions = np.asarray(drone_positions, dtype=float)
    if len(drone_ids) == 0:
        return {}
    order = np.argsort(np.asarray(drone_ids))        # sort by id for stable tie-break
    ids_sorted = np.asarray(drone_ids)[order]
    pos_sorted = drone_positions[order]
    # (M,K) matrix of distances from every cell to every drone.
    d = np.linalg.norm(cell_centers[:, None, :] - pos_sorted[None, :, :], axis=2)
    # Ties (equal distance, e.g. all drones sitting at the base at take-off) must NOT all go to
    # the lowest id, or one drone would own everything. Split tied cells between the tied drones
    # using a hash of the cell coordinates: deterministic, so every drone computes the same answer.
    tied = d <= d.min(axis=1, keepdims=True) + 1e-9   # (M,K) True where a drone is (co-)nearest
    ci = np.rint(cell_centers).astype(np.int64)
    h = np.abs((ci[:, 0] * 73856093) ^ (ci[:, 1] * 19349663) ^ (ci[:, 2] * 83492791))
    pick = h % tied.sum(axis=1) + 1                   # which of the tied drones (1st, 2nd, ...)
    owner = np.argmax(tied & (np.cumsum(tied, axis=1) == pick[:, None]), axis=1)
    return {int(ids_sorted[k]): np.flatnonzero(owner == k) for k in range(len(ids_sorted))}


class SelfHealingManager:
    """Glue: monitor heartbeats, and re-partition when someone drops out."""

    # >>> NEEDS FROM BHARGAV: cell_centers = (M,3) array of the centres of the map cells
    # >>> to be searched (from his occupancy grid). (test_somnath.py shows a simple example grid.)
    # >>> NEEDS FROM SATWIK: after repartition(), his exploration policy/reward must USE the
    # >>> sectors (e.g. give each drone its own sector as input, reward coverage inside it).
    def __init__(self, drone_ids: Sequence[int], cell_centers: np.ndarray,
                 timeout_intervals: int = 3):
        self.monitor = HeartbeatMonitor(drone_ids, timeout_intervals)
        self.cell_centers = np.asarray(cell_centers, dtype=float)
        self.sectors: Dict[int, np.ndarray] = {}

    def repartition(self, positions: Dict[int, Sequence[float]],
                    cells: np.ndarray = None) -> Dict[int, np.ndarray]:
        """Recompute sectors using only drones believed alive.
        `cells` (optional): partition THESE cell centres (e.g. only the current
        frontier cells) instead of the full list given at construction."""
        alive = [i for i in self.monitor.alive_peers() if i in positions]
        pts = np.array([positions[i] for i in alive], dtype=float).reshape(-1, 3)
        centers = self.cell_centers if cells is None else np.asarray(cells, dtype=float).reshape(-1, 3)
        self.sectors = voronoi_assign(centers, alive, pts)
        return self.sectors

    def step(self, step: int, heard: Dict[int, int], positions: Dict[int, Sequence[float]]):
        """
        One control-loop tick.
          heard    : {peer_id: step_of_heartbeat} received this tick
          positions: last known positions of peers
        Returns (newly_failed_set, sectors_if_changed_else_None).
        """
        for pid, s in heard.items():
            self.monitor.record_heartbeat(pid, s)
        newly = self.monitor.check(step)
        if newly:
            return newly, self.repartition(positions)
        return newly, None