"""
Unit tests for Bhargav Kumar's Perception, Mapping, and Heuristic modules.
Tests:
- Thermal LOS Occlusion & Two-Stage Search (thermal.py)
- Thermal Entities (entities.py)
- Typed Obstacle Surface Distance Math (drone_model.py)
- P2P Knowledge Sharing & Registry Merging (communication.py, mission.py)
- Rule Policy Expert Heuristic (pretrain_bc.py)
"""

import unittest
import numpy as np

from src.perception.entities import Human, Fire, ColdZone
from src.perception.drone_model import Obstacle, Drone
from src.perception.thermal import (
    get_thermal_contacts,
    _los_clear,
    _segment_hits_obstacle,
    _segment_hits_cylinder,
    FAINT_SIGNAL_RADIUS,
    CONFIRM_RADIUS,
    FIRE_ALERT_RADIUS,
)
from src.mapping.communication import share_knowledge
from src.mapping.mission import DroneKnowledge, MissionState
from scripts.pretrain_bc import rule_policy
from swarm_env import SwarmEnv


class ThermalPerceptionTests(unittest.TestCase):
    def test_los_clear_and_occlusion_box(self):
        # Box obstacle centered at (5, 0, 0) with size 2x2x2
        box = Obstacle(
            name="test_box",
            obs_type="Building",
            pos=np.array([5.0, 0.0, 0.0]),
            shape="box",
            size_params=np.array([2.0, 2.0, 2.0]),
        )
        # Line from (0, 0, 0) to (10, 0, 0) hits the box -> LOS should NOT be clear
        self.assertFalse(_los_clear(np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0]), [box]))

        # Line from (0, 5, 0) to (10, 5, 0) is offset by 5m -> LOS SHOULD be clear
        self.assertTrue(_los_clear(np.array([0.0, 5.0, 0.0]), np.array([10.0, 5.0, 0.0]), [box]))

    def test_cylinder_intersection_math(self):
        # Cylinder at origin with radius 1.0 and height 4.0 (z from -2 to +2)
        center = np.array([0.0, 0.0, 0.0])
        # Direct hit across cylinder
        self.assertTrue(_segment_hits_cylinder(np.array([-5.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]), center, 1.0, 4.0))
        # Ray passing above the top of cylinder
        self.assertFalse(_segment_hits_cylinder(np.array([-5.0, 0.0, 5.0]), np.array([5.0, 0.0, 5.0]), center, 1.0, 4.0))

    def test_obstacle_surface_distance(self):
        box = Obstacle(
            name="b1",
            obs_type="Building",
            pos=np.array([0.0, 0.0, 0.0]),
            shape="box",
            size_params=np.array([2.0, 2.0, 2.0]),
        )
        # Point at (4, 0, 0): box extends to x = 1.0 -> distance should be 3.0
        self.assertAlmostEqual(box.surface_distance(np.array([4.0, 0.0, 0.0])), 3.0)

        sphere = Obstacle(
            name="s1",
            obs_type="Tree",
            pos=np.array([0.0, 0.0, 0.0]),
            shape="sphere",
            size_params=np.array([2.0]),  # radius 2.0
        )
        # Point at (5, 0, 0) -> distance should be 5 - 2 = 3.0
        self.assertAlmostEqual(sphere.surface_distance(np.array([5.0, 0.0, 0.0])), 3.0)

    def test_two_stage_thermal_contacts_and_heat_masking(self):
        drone = Drone(id="d1", pos=np.array([0.0, 0.0, 10.0]))

        # Human 1 at 8m (within CONFIRM_RADIUS = 10m)
        h_confirmed = Human(id="victim_1", pos=np.array([0.0, 0.0, 2.0]), surface_temp_c=32.5)
        # Human 2 at 30m (between CONFIRM_RADIUS and FAINT_SIGNAL_RADIUS = 50m)
        h_faint = Human(id="victim_2", pos=np.array([30.0, 0.0, 10.0]), surface_temp_c=32.5)

        # Ambient temperature function
        ambient_normal = lambda p: 20.0
        contacts = get_thermal_contacts(drone, [h_confirmed, h_faint], [], [], ambient_normal)

        confirmed_ids = [c["id"] for c in contacts["confirmed"]]
        faint_ids = [c["id"] for c in contacts["faint"]]

        self.assertIn("victim_1", confirmed_ids)
        self.assertIn("victim_2", faint_ids)

        # Ambient heat masking test: when ambient exceeds surface temp, faint detection is masked
        ambient_hot = lambda p: 40.0
        contacts_hot = get_thermal_contacts(drone, [h_faint], [], [], ambient_hot)
        self.assertEqual(len(contacts_hot["faint"]), 0)


class P2PMappingTests(unittest.TestCase):
    def test_decentralized_knowledge_sharing(self):
        # Two drones
        d1 = Drone(id="drone_0", pos=np.array([0.0, 0.0, 0.0]), comm_radius=100.0)
        d2 = Drone(id="drone_1", pos=np.array([20.0, 0.0, 0.0]), comm_radius=100.0)
        drones = [d1, d2]

        grid_shape = (10, 10)
        k1 = DroneKnowledge(grid_shape)
        k2 = DroneKnowledge(grid_shape)

        # Drone 0 knows victim A and visited cell (1, 1)
        k1.confirmed.add("victim_A")
        k1.visited[1, 1] = True

        # Drone 1 knows victim B and visited cell (2, 2)
        k2.confirmed.add("victim_B")
        k2.visited[2, 2] = True

        class MockArena:
            size = 250.0
            x_bounds = (0.0, 250.0)
            y_bounds = (0.0, 250.0)

        class MockWorld:
            arena = MockArena()
            discover_target_ids = {"victim_A", "victim_B"}
            confirm_target_ids = {"victim_A", "victim_B"}

        mission = MissionState(MockWorld(), cell_size=25.0)

        # Nearby list indicating d2 is within comm_radius of d1
        nearby = [
            [{"drone_id": "drone_1", "dist": 20.0}],
            [{"drone_id": "drone_0", "dist": 20.0}],
        ]

        # Trigger P2P knowledge exchange
        share_events = share_knowledge(drones, [k1, k2], nearby, mission, share_maps=True)

        self.assertGreater(share_events, 0)
        # Confirmed sets should be unioned: Confirmed_A ∪ Confirmed_B
        self.assertEqual(k1.confirmed, {"victim_A", "victim_B"})
        self.assertEqual(k2.confirmed, {"victim_A", "victim_B"})

        # Explored maps should be unioned: ExploredMap_A ∪ ExploredMap_B
        self.assertTrue(k1.visited[1, 1] and k1.visited[2, 2])
        self.assertTrue(k2.visited[1, 1] and k2.visited[2, 2])


class RuleExpertHeuristicTests(unittest.TestCase):
    def test_rule_policy_bounds_and_validity(self):
        env = SwarmEnv(num_agents=4)
        obs, _ = env.reset(seed=123)
        rng = np.random.default_rng(123)

        actions = rule_policy(obs, env, rng)

        self.assertEqual(len(actions), 4)
        for agent_id, act in actions.items():
            self.assertEqual(act.shape, (3,))
            self.assertTrue(np.all(act >= -1.0))
            self.assertTrue(np.all(act <= 1.0))


if __name__ == "__main__":
    unittest.main()
