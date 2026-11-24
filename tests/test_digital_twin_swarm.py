"""
Unit Tests for Digital Twin Multi-Drone MuJoCo Swarm Integration.
Tests:
- Dynamic multi-drone MJCF XML generation & mocap body injection
- MuJoCo scene compilation with arbitrary swarm sizes
- End-to-end headless flight simulation and telemetry recording
"""

import os
import unittest
import numpy as np

from src.ground_station.digital_twin.swarm_scene_builder import (
    build_swarm_scene_xml,
    generate_drone_body_xml,
    generate_compact_disaster_city,
    DRONE_PALETTES,
)
from src.ground_station.digital_twin.view_swarm import SwarmMuJoCoViewer

try:
    import mujoco
    MUJOCO_INSTALLED = True
except ImportError:
    MUJOCO_INSTALLED = False


class TestSwarmSceneBuilder(unittest.TestCase):
    def test_compact_disaster_city_generation(self):
        xml = generate_compact_disaster_city(num_drones=6)
        self.assertIn('<mujoco model="swarmrl_disaster_city_compact">', xml)
        self.assertEqual(xml.count('<body name="drone_'), 6)
        self.assertEqual(xml.count('mocap="true"'), 6)
        self.assertIn("Tower_Alpha", xml)
        self.assertIn("Victim_01", xml)

    def test_drone_palettes_assignment(self):
        body_0 = generate_drone_body_xml(0, 0.0, 0.0)
        body_1 = generate_drone_body_xml(1, 5.0, 5.0)
        self.assertIn('name="drone_0"', body_0)
        self.assertIn('name="drone_1"', body_1)
        self.assertIn(DRONE_PALETTES[0]["dome_rgba"], body_0)
        self.assertIn(DRONE_PALETTES[1]["dome_rgba"], body_1)

    @unittest.skipUnless(MUJOCO_INSTALLED, "MuJoCo not installed in environment")
    def test_mujoco_xml_compilation(self):
        xml = build_swarm_scene_xml(num_drones=4, compact=True)
        model = mujoco.MjModel.from_xml_string(xml)
        self.assertIsNotNone(model)
        self.assertEqual(model.nmocap, 4)
        self.assertGreater(model.nbody, 4)


class TestSwarmViewerExecution(unittest.TestCase):
    def test_headless_swarm_simulation_run(self):
        viewer = SwarmMuJoCoViewer(
            num_drones=3,
            max_steps=15,
            compact_scene=True,
            headless=True,
            save_trajectory_path="data/test_trajectory.json",
            save_plot_path="data/test_plot.png",
        )
        summary = viewer.run()
        self.assertGreaterEqual(summary["total_steps"], 10)
        self.assertIn("final_coverage", summary)
        self.assertIn("total_rescued", summary)
        self.assertTrue(os.path.exists("data/test_trajectory.json"))
        self.assertTrue(os.path.exists("data/test_plot.png"))

        # Clean up test files
        if os.path.exists("data/test_trajectory.json"):
            os.remove("data/test_trajectory.json")
        if os.path.exists("data/test_plot.png"):
            os.remove("data/test_plot.png")


if __name__ == "__main__":
    unittest.main()
