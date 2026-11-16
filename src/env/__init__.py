"""
Module: src.env
Owner: Satwik Choudhury
Responsibilities:
- 3D Continuous PettingZoo Multi-Agent Simulation Environment.
- Volumetric Coverage Grid & Obstacle Terrain generation.
- Core simulation loop and observation/action space definitions.
- Plug-in hooks for Safety Shield, LiDAR Sensors, and Ground Station Telemetry.
"""
from .swarm_env_3d import SwarmEnv

__all__ = ["SwarmEnv"]
