"""
Module: src.perception
Owner: Bhargav Kumar
Responsibilities:
- 3D LiDAR Raycasting sensor simulation for drone obstacle perception.
- Sensor observation generation (feeds into env.sensor_hook).
- Multi-modal sensor simulation (thermal signatures, visual cameras, acoustic cues).
"""
from .lidar_sensor import LiDAR3D

__all__ = ["LiDAR3D"]
