import numpy as np

class FlightDynamics:
    """
    Simulates physical drone flight dynamics, environmental wind, and battery usage.
    Owner: Arkendu Chatterjee
    """
    def __init__(
        self,
        mass=1.2,              # kg
        drag_coeff=0.1,        # Aerodynamic drag factor
        max_thrust=15.0,       # Newtons
        battery_capacity=100.0,# %
        battery_burn_rate=0.05 # % per step
    ):
        self.mass = mass
        self.drag_coeff = drag_coeff
        self.max_thrust = max_thrust
        self.battery_capacity = battery_capacity
        self.battery_burn_rate = battery_burn_rate

    def apply_forces(self, current_velocity, thrust_vector, wind_vector=None):
        """
        Integrates net force = thrust + wind - drag to return updated velocity.
        """
        if wind_vector is None:
            wind_vector = np.zeros(3, dtype=np.float32)

        rel_wind = current_velocity - wind_vector
        drag_force = -self.drag_coeff * rel_wind * np.linalg.norm(rel_wind)
        net_force = (thrust_vector * self.max_thrust) + drag_force

        accel = net_force / self.mass
        return accel

    def step_battery(self, current_battery, thrust_magnitude):
        """Simulates non-linear battery degradation under motor load."""
        burn = self.battery_burn_rate * (1.0 + 0.5 * thrust_magnitude)
        return max(0.0, current_battery - burn)
