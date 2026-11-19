"""Global mission state and per-drone knowledge, kept strictly separate.

Hidden victim coordinates are never copied into knowledge -- faint contact
positions are reconstructed from sensor returns (drone pos + relative bearing).
"""
from typing import Dict, Set
import numpy as np


class DroneKnowledge:
    __slots__ = ("faint", "confirmed", "visited")

    def __init__(self, grid_shape):
        self.faint: Dict[str, tuple] = {}      # victim_id -> (sensor_pos, step)
        self.confirmed: Set[str] = set()
        self.visited = np.zeros(grid_shape, dtype=bool)


class MissionState:
    def __init__(self, world, cell_size: float = 25.0):
        self.world = world
        self.cell_size = float(cell_size)
        a = world.arena
        self.nx = max(1, int(np.ceil(a.size / self.cell_size)))
        self.ny = self.nx
        self.discover_targets = world.discover_target_ids
        self.confirm_targets = world.confirm_target_ids
        self.reset()

    @property
    def grid_shape(self):
        return (self.nx, self.ny)

    def reset(self):
        self.team_visited = np.zeros(self.grid_shape, dtype=bool)
        self.team_discovered: Set[str] = set()
        self.team_confirmed: Set[str] = set()
        self.discovered_by: Dict[str, str] = {}
        self.confirmed_by: Dict[str, str] = {}
        self.cell_entries = 0
        self.redundant_entries = 0
        self.collision_events = 0
        self.collision_steps = 0
        self.proximity_steps = 0
        self.comm_links = 0
        self.comm_share_events = 0

    def cell_of(self, pos):
        a = self.world.arena
        ix = int((pos[0] - a.x_bounds[0]) / self.cell_size)
        iy = int((pos[1] - a.y_bounds[0]) / self.cell_size)
        return (int(np.clip(ix, 0, self.nx - 1)), int(np.clip(iy, 0, self.ny - 1)))

    def enter_cell(self, knowledge, pos):
        """Called when a drone crosses into a new grid cell -> (team_new, self_new)."""
        c = self.cell_of(pos)
        team_new = not bool(self.team_visited[c])
        self_new = not bool(knowledge.visited[c])
        self.team_visited[c] = True
        knowledge.visited[c] = True
        self.cell_entries += 1
        if not team_new:
            self.redundant_entries += 1
        return team_new, self_new

    def mark_current_cell(self, knowledge, pos):
        c = self.cell_of(pos)
        self.team_visited[c] = True
        knowledge.visited[c] = True

    def update_contacts(self, knowledge, drone, contacts, step: int):
        """Fold thermal sensor returns into knowledge + mission; return events."""
        events = {"personal_new_faint": 0, "team_discoveries": 0,
                  "team_confirmations": 0, "coop_confirmations": 0}
        pos = np.asarray(drone.pos)
        for c in contacts["faint"]:
            if c.get("type") == "fire":
                continue
            vid = c["id"]
            if vid not in knowledge.faint and vid not in knowledge.confirmed:
                knowledge.faint[vid] = (pos + np.asarray(c["relative_position"]), step)
                events["personal_new_faint"] += 1
            if vid in self.discover_targets and vid not in self.team_discovered:
                self.team_discovered.add(vid)
                self.discovered_by[vid] = drone.id
                events["team_discoveries"] += 1
        for c in contacts["confirmed"]:
            vid = c["id"]
            if vid not in knowledge.confirmed:
                knowledge.confirmed.add(vid)
                knowledge.faint.pop(vid, None)
            if vid in self.discover_targets and vid not in self.team_discovered:
                self.team_discovered.add(vid)
                self.discovered_by[vid] = drone.id
                events["team_discoveries"] += 1
            if vid in self.confirm_targets and vid not in self.team_confirmed:
                self.team_confirmed.add(vid)
                self.confirmed_by[vid] = drone.id
                events["team_confirmations"] += 1
                if self.discovered_by.get(vid, drone.id) != drone.id:
                    events["coop_confirmations"] += 1
        return events

    @property
    def coverage(self) -> float:
        return float(self.team_visited.mean())

    @property
    def discovered_fraction(self) -> float:
        n = len(self.discover_targets)
        return len(self.team_discovered) / n if n else 0.0

    @property
    def confirmed_fraction(self) -> float:
        n = len(self.confirm_targets)
        return len(self.team_confirmed) / n if n else 0.0

    @property
    def all_confirmed(self) -> bool:
        n = len(self.confirm_targets)
        return n > 0 and len(self.team_confirmed) >= n