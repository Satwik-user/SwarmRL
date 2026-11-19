"""
Module: src.perception
Owner: Bhargav Kumar
Responsibilities:
- 3D LiDAR Raycasting sensor simulation for drone obstacle perception.
- Two-stage thermal perception engine (50m faint radar, 10m confirmation).
- Line-of-Sight (LOS) occlusion math (_los_clear, ray-box, ray-cylinder, ray-sphere).
- Thermal entity definitions (Human, Fire, ColdZone).
- Obstacle geometry, surface distance computation, and drone representations.
"""
from .lidar_sensor import LiDAR3D
from .entities import Human, Fire, ColdZone
from .drone_model import Obstacle, Drone
from .thermal import (
    get_thermal_contacts,
    _los_clear,
    FAINT_SIGNAL_RADIUS,
    CONFIRM_RADIUS,
    FIRE_ALERT_RADIUS,
)

__all__ = [
    "LiDAR3D",
    "Human",
    "Fire",
    "ColdZone",
    "Obstacle",
    "Drone",
    "get_thermal_contacts",
    "_los_clear",
    "FAINT_SIGNAL_RADIUS",
    "CONFIRM_RADIUS",
    "FIRE_ALERT_RADIUS",
]

