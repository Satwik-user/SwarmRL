"""
Module: src.ground_station.digital_twin
Owner: Bhargav Kumar (Procedural Disaster City & MuJoCo Visualizer)
Responsibilities:
- Procedural disaster city generation with 3D buildings, rubble, fires, and victims.
- MuJoCo digital twin MJCF export and high-fidelity rendering.
- Interactive 3D single-drone inspector and telemetry visualizer.
- High-performance GPU utility setup.
"""

from .environment_generator import ProceduralEnvironmentGenerator, GeneratedObstacle
# Backward-compatibility alias
ProceduralDisasterEnvironment = ProceduralEnvironmentGenerator

from .swarm_scene_builder import (
    build_swarm_scene_xml,
    generate_drone_body_xml,
    generate_compact_disaster_city,
    DRONE_PALETTES,
)

try:
    from .view_swarm import SwarmMuJoCoViewer
except (ImportError, ModuleNotFoundError):
    SwarmMuJoCoViewer = None

try:
    from .view_drone import SingleDroneViewer
except (ImportError, ModuleNotFoundError):
    SingleDroneViewer = None

try:
    from .gpu_utils import setup_high_performance_gpu
except (ImportError, ModuleNotFoundError):
    def setup_high_performance_gpu():
        return False

__all__ = [
    "ProceduralEnvironmentGenerator",
    "ProceduralDisasterEnvironment",
    "GeneratedObstacle",
    "SingleDroneViewer",
    "SwarmMuJoCoViewer",
    "build_swarm_scene_xml",
    "generate_drone_body_xml",
    "generate_compact_disaster_city",
    "DRONE_PALETTES",
    "setup_high_performance_gpu",
]

