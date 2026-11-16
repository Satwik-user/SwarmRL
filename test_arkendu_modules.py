import unittest

from flight_dynamics import FlightDynamics, FlightDynamicsConfig, FlightState
from safety_shield import AABB, Neighbor, SafetyShield, segment_intersects_aabb


class FlightDynamicsTests(unittest.TestCase):
    def test_acceleration_and_speed_are_bounded(self):
        dynamics = FlightDynamics(FlightDynamicsConfig(max_speed=2.0, max_acceleration=1.0))
        state = dynamics.step(FlightState(), (100.0, 0.0, 0.0), dt=1.0)
        self.assertAlmostEqual(state.velocity[0], 1.0)
        for _ in range(10):
            state = dynamics.step(state, (100.0, 0.0, 0.0), dt=1.0)
        self.assertLessEqual(state.velocity[0], 2.0)

    def test_wind_and_battery_are_modelled(self):
        dynamics = FlightDynamics()
        state = dynamics.step(
            FlightState(battery=10.0),
            (0.0, 0.0, 0.0),
            wind_velocity=(3.0, 0.0, 0.0),
            dt=1.0,
        )
        self.assertLess(state.battery, 10.0)
        self.assertNotEqual(state.velocity, (0.0, 0.0, 0.0))


class SafetyShieldTests(unittest.TestCase):
    def test_swept_path_detects_tunnelling(self):
        obstacle = AABB((4.0, -1.0, -1.0), (5.0, 1.0, 1.0))
        self.assertTrue(segment_intersects_aabb((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), obstacle))

    def test_obstacle_command_is_projected(self):
        shield = SafetyShield()
        result = shield.filter_velocity(
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            obstacles=(AABB((4.0, -1.0, -1.0), (5.0, 1.0, 1.0)),),
        )
        self.assertTrue(result.safe)
        self.assertTrue(result.modified)
        self.assertTrue(
            shield.is_path_safe(
                (0.0, 0.0, 0.0),
                result.velocity,
                obstacles=(AABB((4.0, -1.0, -1.0), (5.0, 1.0, 1.0)),),
            )
        )

    def test_neighbor_separation_is_enforced(self):
        shield = SafetyShield()
        result = shield.filter_velocity(
            (0.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
            neighbors=(Neighbor((3.0, 0.0, 0.0)),),
        )
        self.assertTrue(result.safe)
        self.assertTrue(result.modified)


if __name__ == "__main__":
    unittest.main()
