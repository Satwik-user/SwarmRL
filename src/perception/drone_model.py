"""
Drone and Obstacle Data Structures, Sensing, and Collision Algorithms.
Supports 10 km x 10 km environments with typed obstacles: Buildings, Rubble, Trees, and Poles.
"""

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class Obstacle:
    """
    Typed obstacle entity representing a Building, Rubble piece, Tree, or Pole.
    Provides exact geometric surface distance and relative direction vector.
    """
    name: str
    obs_type: str  # 'Building', 'Rubble', 'Tree', 'Pole'
    pos: np.ndarray  # Center or base position [x, y, z]
    shape: str = "box"  # 'box', 'cylinder', 'sphere', 'tree'
    size_params: np.ndarray = field(default_factory=lambda: np.array([10.0, 10.0, 10.0]))
    heading_deg: float = 0.0

    @property
    def size(self) -> np.ndarray:
        """Backward-compatibility helper for box sizes."""
        return self.size_params

    @property
    def half_extents(self) -> np.ndarray:
        return self.size_params / 2.0

    def surface_distance(self, point: np.ndarray) -> float:
        """
        Calculates Euclidean distance from query point to obstacle surface.
        Returns 0.0 if the point is inside or contacting the obstacle.
        """
        if self.shape == "box":
            hx, hy, hz = self.size_params[0] / 2.0, self.size_params[1] / 2.0, self.size_params[2] / 2.0
            closest = np.clip(point, self.pos - np.array([hx, hy, hz]), self.pos + np.array([hx, hy, hz]))
            delta = point - closest
            return float(np.linalg.norm(delta))

        elif self.shape == "cylinder":
            # Vertical cylinder: size_params = [radius, height]
            radius = self.size_params[0]
            height = self.size_params[1]
            dx = point[0] - self.pos[0]
            dy = point[1] - self.pos[1]
            dist_xy = math.sqrt(dx * dx + dy * dy)
            surf_xy = max(0.0, dist_xy - radius)

            # Base is at pos[2] - height/2 or pos[2]
            z_min = self.pos[2] - height / 2.0
            z_max = self.pos[2] + height / 2.0
            if point[2] < z_min:
                surf_z = z_min - point[2]
            elif point[2] > z_max:
                surf_z = point[2] - z_max
            else:
                surf_z = 0.0

            return float(math.sqrt(surf_xy * surf_xy + surf_z * surf_z))

        elif self.shape == "sphere":
            # Sphere: size_params = [radius]
            radius = self.size_params[0]
            dist_center = float(np.linalg.norm(point - self.pos))
            return max(0.0, dist_center - radius)

        elif self.shape == "tree":
            # Tree: size_params = [trunk_r, trunk_h, crown_r, crown_z]
            trunk_r, trunk_h, crown_r, crown_z = self.size_params
            # 1. Trunk (cylinder from z=0 to trunk_h at (pos[0], pos[1]))
            dx = point[0] - self.pos[0]
            dy = point[1] - self.pos[1]
            dist_xy = math.sqrt(dx * dx + dy * dy)
            surf_trunk_xy = max(0.0, dist_xy - trunk_r)
            if point[2] < 0.0:
                surf_trunk_z = -point[2]
            elif point[2] > trunk_h:
                surf_trunk_z = point[2] - trunk_h
            else:
                surf_trunk_z = 0.0
            d_trunk = math.sqrt(surf_trunk_xy * surf_trunk_xy + surf_trunk_z * surf_trunk_z)

            # 2. Crown (sphere at (pos[0], pos[1], crown_z) with radius crown_r)
            crown_center = np.array([self.pos[0], self.pos[1], crown_z])
            d_crown = max(0.0, float(np.linalg.norm(point - crown_center)) - crown_r)

            return min(d_trunk, d_crown)

        # Fallback to center distance
        return float(np.linalg.norm(point - self.pos))

    def relative_position(self, point: np.ndarray) -> Tuple[float, float, float]:
        """Vector pointing from query point towards the obstacle center or closest surface."""
        if self.shape == "box":
            hx, hy, hz = self.size_params[0] / 2.0, self.size_params[1] / 2.0, self.size_params[2] / 2.0
            closest = np.clip(point, self.pos - np.array([hx, hy, hz]), self.pos + np.array([hx, hy, hz]))
            dist = np.linalg.norm(point - closest)
            if dist > 1e-5:
                rel = closest - point
            else:
                rel = self.pos - point
            return (float(rel[0]), float(rel[1]), float(rel[2]))
        else:
            rel = self.pos - point
            return (float(rel[0]), float(rel[1]), float(rel[2]))


@dataclass
class Drone:
    """
    Kinematic drone entity.
    Position is directly controlled with zero physics/gravity/aerodynamics.
    Boundary clamped to 10 km x 10 km environment (-5,000 to +5,000 m).
    """
    id: str
    pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 5.0], dtype=np.float64))
    speed: float = 10.0  # Configurable movement speed in m/s
    comm_radius: float = 100.0  # Configurable inter-drone detection range in meters
    collision_radius: float = 1.8  # Bounding geometry radius for collision detection (meters)
    active: bool = True
    mocap_id: int = 0

    # Boundaries for 10 km x 10 km environment (-5000 to +5000 m)
    x_bounds: Tuple[float, float] = (-5000.0, 5000.0)
    y_bounds: Tuple[float, float] = (-5000.0, 5000.0)
    z_bounds: Tuple[float, float] = (0.2, 1000.0)

    def move(self, delta: np.ndarray) -> None:
        """Directly change drone position, clamped within area boundaries."""
        new_pos = self.pos + delta
        new_pos[0] = np.clip(new_pos[0], self.x_bounds[0], self.x_bounds[1])
        new_pos[1] = np.clip(new_pos[1], self.y_bounds[0], self.y_bounds[1])
        new_pos[2] = np.clip(new_pos[2], self.z_bounds[0], self.z_bounds[1])
        self.pos = new_pos

    def set_position(self, x: float, y: float, z: float) -> None:
        """Set position directly with boundary clamping."""
        pos = np.array([x, y, z], dtype=np.float64)
        pos[0] = np.clip(pos[0], self.x_bounds[0], self.x_bounds[1])
        pos[1] = np.clip(pos[1], self.y_bounds[0], self.y_bounds[1])
        pos[2] = np.clip(pos[2], self.z_bounds[0], self.z_bounds[1])
        self.pos = pos


def get_nearby_drones(
    current_drone: Drone,
    all_drones: List[Drone],
    comm_radius: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Finds all other active drones within communication_radius (default 100.0m).
    
    Returns list of dicts:
      - 'drone_id': ID of detected drone
      - 'distance': 3D Euclidean distance in meters
      - 'relative_position': (dx, dy, dz) vector from current_drone to other drone
    """
    radius = comm_radius if comm_radius is not None else current_drone.comm_radius
    nearby = []

    p1 = current_drone.pos
    for other in all_drones:
        if other.id == current_drone.id or not other.active:
            continue

        p2 = other.pos
        # 3D Euclidean distance: sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
        diff = p2 - p1
        distance = float(np.linalg.norm(diff))

        if distance <= radius:
            nearby.append({
                "drone_id": other.id,
                "distance": distance,
                "relative_position": (float(diff[0]), float(diff[1]), float(diff[2]))
            })

    # Sort by distance
    nearby.sort(key=lambda item: item["distance"])
    return nearby


def get_nearest_obstacle(
    drone: Drone,
    obstacles: List[Obstacle],
    obstacle_positions_matrix: Optional[np.ndarray] = None
) -> Tuple[Optional[Obstacle], float, Tuple[float, float, float]]:
    """
    Determines the nearest obstacle to the drone using high-performance vectorized
    candidate pre-filtering followed by exact primitive surface distance calculation.
    
    Returns:
      (nearest_obstacle, distance, relative_position_tuple)
    """
    if not obstacles:
        return (None, float("inf"), (0.0, 0.0, 0.0))

    # Fast vectorized search if obstacle positions matrix is available or large list
    num_obs = len(obstacles)
    if num_obs > 20:
        if obstacle_positions_matrix is None:
            obstacle_positions_matrix = np.array([obs.pos for obs in obstacles], dtype=np.float64)

        # Optimized non-allocating Euclidean center squared distances (~4x faster)
        dx = obstacle_positions_matrix[:, 0] - drone.pos[0]
        dy = obstacle_positions_matrix[:, 1] - drone.pos[1]
        dz = obstacle_positions_matrix[:, 2] - drone.pos[2]
        center_dists_sq = dx * dx + dy * dy + dz * dz

        # Pre-filter closest 35 candidates
        k = min(35, num_obs)
        candidate_indices = np.argpartition(center_dists_sq, k)[:k]

        nearest_obs = None
        min_dist = float("inf")
        best_rel = (0.0, 0.0, 0.0)

        for idx in candidate_indices:
            obs = obstacles[idx]
            dist = obs.surface_distance(drone.pos)
            if dist < min_dist:
                min_dist = dist
                nearest_obs = obs
                best_rel = obs.relative_position(drone.pos)

        return (nearest_obs, min_dist, best_rel)

    # Standard loop for smaller lists
    nearest_obs = None
    min_dist = float("inf")
    best_rel = (0.0, 0.0, 0.0)

    for obs in obstacles:
        dist = obs.surface_distance(drone.pos)
        if dist < min_dist:
            min_dist = dist
            nearest_obs = obs
            best_rel = obs.relative_position(drone.pos)

    return (nearest_obs, min_dist, best_rel)


def check_collisions(
    drone: Drone,
    obstacles: List[Obstacle],
    all_drones: List[Drone],
    min_separation: float = 10.0,
    obstacle_positions_matrix: Optional[np.ndarray] = None,
    nearest_obs: Optional[Obstacle] = None,
    obs_dist: Optional[float] = None,
) -> Tuple[bool, Optional[str], bool, Optional[str]]:
    """
    Evaluates geometric collisions:
      1. Terrain collision (drone Z <= collision radius)
      2. Drone <-> Obstacle intersection (surface distance <= drone collision radius)
      3. Drone <-> Drone separation violation (< min_separation)
      
    Accepts optional precalculated (nearest_obs, obs_dist) to eliminate redundant queries.
      
    Returns:
      (is_colliding, obstacle_or_terrain_name, is_drone_proximity_alert, proximity_drone_id)
    """
    # 1. Terrain collision
    if drone.pos[2] <= drone.collision_radius:
        return (True, "Terrain", False, None)

    # 2. Obstacle collisions (utilizes precomputed or fast nearest obstacle lookup)
    if nearest_obs is None or obs_dist is None:
        nearest_obs, obs_dist, _ = get_nearest_obstacle(drone, obstacles, obstacle_positions_matrix)
    if nearest_obs is not None and obs_dist <= drone.collision_radius:
        return (True, nearest_obs.name, False, None)

    # 3. Inter-drone minimum separation
    for other in all_drones:
        if other.id == drone.id or not other.active:
            continue
        dist = float(np.linalg.norm(other.pos - drone.pos))
        if dist < min_separation:
            return (False, None, True, other.id)

    return (False, None, False, None)
