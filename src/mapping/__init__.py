"""
Module: src.mapping
Owner: Bhargav Kumar
Responsibilities:
- 3D Volumetric Mapping & Occupancy Voxel Grid (OctoMap memory).
- Frontier Delta generation for lightweight P2P mesh broadcasting.
- Distributed map stitching and 3D Disaster Digital Twin reconstruction (Bhargav-Kumar2007/SwarmRL integration).
- Decentralized P2P local knowledge sharing & registry merging (Confirmed and ExploredMap sets).
- Per-drone knowledge state and global mission tracking.
"""
from .occupancy_grid import OccupancyGrid3D
from .communication import share_knowledge
from .mission import DroneKnowledge, MissionState

__all__ = [
    "OccupancyGrid3D",
    "share_knowledge",
    "DroneKnowledge",
    "MissionState",
]

