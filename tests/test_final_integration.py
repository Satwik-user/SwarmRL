"""
SwarmRL Final Integration Test Suite.
Validates the end-to-end integration across all 5 member sub-systems:
- Member 1 (Satwik Chowdhury): PettingZoo SwarmEnv, MAPPOAgent, BC warmstart
- Member 2 (Arkendu Chatterjee): SafetyShield deterministic swept-path projection
- Member 3 (Somnath Saha): ConsensusEngine Bayesian voting, TaskAllocator auction
- Member 4 (Bhargav Kumar): Thermal LOS occlusion, typed Obstacle math
- Member 5 (Chandan Mondal): MeshNetworkSimulator, ground station logging
"""

import unittest
import numpy as np
import torch

from swarm_env import SwarmEnv
from models.mappo_agent import MAPPOAgent
from src.safety.safety_shield import SafetyShield, SafetyConfig, AABB, Neighbor
from src.coordination.consensus_engine import ConsensusEngine, Observation
from src.coordination.task_allocator import TaskAllocator, DroneState, Task, Role
from src.communication.comm_mesh import MeshNetworkSimulator
from src.perception.drone_model import Obstacle, Drone
from src.perception.thermal import _los_clear
from src.mapping.communication import share_knowledge
from src.mapping.mission import DroneKnowledge, MissionState


class TestModuleImports(unittest.TestCase):
    """Verify that all core modules and sub-packages import without error."""
    def test_package_exports(self):
        import src.ground_station
        import src.ground_station.metrics_logger
        import src.coordination
        import src.dynamics.flight_dynamics
        import src.safety.safety_shield
        import src.communication.comm_mesh
        import src.perception
        import src.mapping
        self.assertIsNotNone(src.ground_station.MetricsLogger)
        self.assertIsNotNone(src.coordination.ConsensusEngine)
        self.assertIsNotNone(src.coordination.TaskAllocator)


class TestSafetyShieldIntegration(unittest.TestCase):
    """Verify Arkendu's continuous swept-path safety projection."""
    def test_shield_collision_prevention(self):
        shield = SafetyShield(SafetyConfig(drone_radius=1.5, separation_margin=1.0, max_speed=8.0))
        # Drone at origin moving directly into a box obstacle at (5, 0, 0)
        box = AABB(minimum=(3.0, -2.0, -2.0), maximum=(7.0, 2.0, 2.0))
        pos = (0.0, 0.0, 0.0)
        v_unsafe = (8.0, 0.0, 0.0)  # Direct impact velocity

        result = shield.filter_velocity(pos, v_unsafe, obstacles=[box])
        # Safety result must deflect or brake to prevent penetration
        self.assertTrue(result.modified or not result.safe)
        self.assertNotEqual(result.velocity, v_unsafe)


class TestConsensusAndAuctionIntegration(unittest.TestCase):
    """Verify Somnath's Bayesian consensus engine and task allocation auction."""
    def test_consensus_locking_and_task_auction(self):
        # Two drones observing the same location (100.0, 100.0, 10.0)
        target_loc = (100.0, 100.0, 10.0)
        engine_a = ConsensusEngine(lock_threshold=0.85)
        engine_b = ConsensusEngine(lock_threshold=0.85)

        # Drone 0 reports 0.70 confidence at step 10
        obs0 = Observation(drone_id=0, sensor="thermal", position=target_loc, confidence=0.70, timestamp=10)
        engine_a.add_observation(obs0)
        self.assertEqual(len(engine_a.locked_survivors()), 0)  # Not locked yet (only 1 drone)

        # Drone 1 reports 0.75 confidence at step 12
        obs1 = Observation(drone_id=1, sensor="thermal", position=target_loc, confidence=0.75, timestamp=12)
        engine_b.add_observation(obs1)

        # Peer-to-peer merge over mesh
        engine_a.merge(engine_b.export_state())
        locked = engine_a.locked_survivors()
        self.assertGreaterEqual(len(locked), 1)  # Locked by consensus!

        # Auction task generation from locked survivor
        tasks = TaskAllocator.create_tasks_from_survivors(locked, engine_a.cell_of, priority=10.0)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].role, Role.KIT_TRANSPORT)

        # Run auction with 2 drones
        d0 = DroneState(drone_id=0, position=(50.0, 50.0, 10.0), battery=0.8, capabilities={Role.KIT_TRANSPORT, Role.THERMAL_SCANNER})
        d1 = DroneState(drone_id=1, position=(90.0, 90.0, 10.0), battery=0.8, capabilities={Role.KIT_TRANSPORT})

        allocator = TaskAllocator()
        assignments = allocator.allocate([d0, d1], tasks)
        self.assertIn(tasks[0].task_id, assignments)
        # d1 is closer to (100, 100, 10) than d0 -> d1 should win the auction
        self.assertEqual(assignments[tasks[0].task_id], 1)


class TestThermalLOSIntegration(unittest.TestCase):
    """Verify Bhargav's thermal line-of-sight occlusion in SwarmEnv."""
    def test_thermal_los_radar_blocking(self):
        env = SwarmEnv(num_agents=1, num_survivors=1, enable_thermal_los=True)
        env.reset(seed=42)

        # Place drone on one side of building and survivor on the other side
        # Obstacle 0: [30, 30, 0, 30, 30, 35] -> X: 30 to 60, Y: 30 to 60
        drone_pos = np.array([10.0, 45.0, 10.0], dtype=np.float32)
        survivor_behind_wall = np.array([80.0, 45.0, 10.0], dtype=np.float32)

        env.agent_positions["drone_0"] = drone_pos
        env.survivor_positions[0] = survivor_behind_wall

        # Check LOS occlusion directly
        los = _los_clear(drone_pos, survivor_behind_wall, env.perception_obstacles)
        self.assertFalse(los)

        # In observation, survivor must NOT be detected because wall blocks LOS
        obs = env._get_observation("drone_0")
        detected_flag = obs[-1]
        self.assertEqual(detected_flag, 0.0)

        # Now move drone to unoccluded line-of-sight
        drone_clear_pos = np.array([80.0, 20.0, 10.0], dtype=np.float32)
        env.agent_positions["drone_0"] = drone_clear_pos
        los_clear_check = _los_clear(drone_clear_pos, survivor_behind_wall, env.perception_obstacles)
        self.assertTrue(los_clear_check)

        obs_clear = env._get_observation("drone_0")
        self.assertEqual(obs_clear[-1], 1.0)


class TestMeshNetworkIntegration(unittest.TestCase):
    """Verify Chandan's lossy mesh simulator with P2P knowledge sharing."""
    def test_mesh_out_of_range_suppression(self):
        mesh = MeshNetworkSimulator(comm_radius=50.0, base_drop_rate=0.0)

        d1 = Drone(id="drone_0", pos=np.array([0.0, 0.0, 0.0]))
        d2_near = Drone(id="drone_1", pos=np.array([30.0, 0.0, 0.0]))
        d3_far = Drone(id="drone_2", pos=np.array([100.0, 0.0, 0.0]))

        can_comm_near, loss_near = mesh.can_communicate(d1.pos, d2_near.pos)
        self.assertTrue(can_comm_near)
        self.assertLess(loss_near, 1.0)

        can_comm_far, loss_far = mesh.can_communicate(d1.pos, d3_far.pos)
        self.assertFalse(can_comm_far)
        self.assertEqual(loss_far, 1.0)


class TestSelfHealingAndVoxelSLAM(unittest.TestCase):
    """Verify Somnath's SelfHealingManager and Bhargav's LiDAR3D / OccupancyGrid3D."""
    def test_self_healing_attrition_repartition(self):
        from src.coordination.self_healing import SelfHealingManager
        cells = np.array([
            [10.0, 10.0, 5.0],
            [10.0, 90.0, 5.0],
            [90.0, 10.0, 5.0],
            [90.0, 90.0, 5.0],
        ])
        manager = SelfHealingManager(drone_ids=[0, 1], cell_centers=cells, timeout_intervals=2)
        positions = {0: (10.0, 10.0, 5.0), 1: (90.0, 90.0, 5.0)}

        # Both drones active at step 1
        manager.step(1, {0: 1, 1: 1}, positions)
        self.assertEqual(len(manager.monitor.failed), 0)

        # Drone 1 goes silent at step 2 and step 3
        newly_failed, repartitioned = manager.step(3, {0: 3}, positions)
        self.assertIn(1, newly_failed)
        self.assertIsNotNone(repartitioned)
        # Remaining alive drone (0) absorbs all cells
        self.assertEqual(len(repartitioned[0]), 4)

    def test_lidar_and_voxel_occupancy_pipeline(self):
        from src.perception.lidar_sensor import LiDAR3D
        from src.mapping.occupancy_grid import OccupancyGrid3D
        env = SwarmEnv(num_agents=2, num_survivors=1)
        env.reset(seed=42)

        # Test LiDAR scan returns finite values in [0, 1]
        lidar = LiDAR3D(num_azimuth=8, num_elevation=2, max_range=40.0)
        readings = lidar.scan("drone_0", env)
        self.assertEqual(len(readings), 16)
        self.assertTrue(np.all(readings >= 0.0) and np.all(readings <= 1.0))

        # Test Occupancy Grid integration & delta extraction
        grid = OccupancyGrid3D(bounds=(200, 200, 50), resolution=10.0)
        point_cloud = [np.array([15.0, 15.0, 5.0]), np.array([55.0, 55.0, 15.0])]
        grid.integrate_scan((0, 0, 0), point_cloud)

        deltas = grid.extract_deltas()
        self.assertEqual(len(deltas), 2)
        # After extraction, new deltas are cleared
        self.assertEqual(len(grid.extract_deltas()), 0)

        # Peer drone merges deltas
        peer_grid = OccupancyGrid3D(bounds=(200, 200, 50), resolution=10.0)
        peer_grid.merge_deltas(deltas)
        self.assertEqual(peer_grid.grid[1, 1, 0], 2)
        self.assertEqual(peer_grid.grid[5, 5, 1], 2)


class TestFullEndToEndSimulation(unittest.TestCase):
    """Execute end-to-end simulation steps with all safety and perception systems active."""
    def test_full_environment_stepping_with_shield(self):
        env = SwarmEnv(
            num_agents=6,
            num_survivors=4,
            max_steps=20,
            enable_thermal_los=True,
            enable_safety_shield=True,
        )
        obs, _ = env.reset(seed=99)
        agent = MAPPOAgent(k_neighbors=env.k_neighbors, state_dim=env.state_dim, action_dim=3, device="cpu")

        total_steps = 0
        for _ in range(15):
            actions, _ = agent.select_actions(obs, deterministic=True)
            obs, rewards, terms, truncs, infos = env.step(actions)
            total_steps += 1
            # Collisions must stay 0 with deterministic safety shield active
            self.assertEqual(env.episode_collisions, 0)
            if any(terms.values()) or any(truncs.values()):
                break

        self.assertGreater(total_steps, 0)
        # Check telemetry output
        telemetry = env.get_telemetry()
        self.assertIn("coverage_pct", telemetry)
        self.assertIn("drones", telemetry)
        self.assertEqual(telemetry["collisions"], 0)


if __name__ == "__main__":
    unittest.main()
