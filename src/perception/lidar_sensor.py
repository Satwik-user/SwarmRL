import numpy as np

class LiDAR3D:
    """
    Simulated 3D LiDAR Sensor.
    Owner: Bhargav Kumar
    
    Casts multi-directional rays in 3D to measure distance to obstacles and ground.
    Plugs directly into `env.sensor_hook`.
    """
    def __init__(self, num_azimuth=8, num_elevation=2, max_range=40.0):
        self.num_azimuth = num_azimuth
        self.num_elevation = num_elevation
        self.max_range = float(max_range)
        self.total_rays = num_azimuth * num_elevation

    def scan(self, agent_id, env):
        """
        Calculates distance readings for all rays from agent's position.
        Returns normalized array [total_rays] in [0, 1].
        """
        pos = env.agent_positions[agent_id]
        readings = []
        obstacles = getattr(env, "perception_obstacles", [])

        azimuths = np.linspace(-np.pi, np.pi, self.num_azimuth, endpoint=False)
        elevations = np.linspace(-np.pi / 6.0, np.pi / 6.0, self.num_elevation)

        for el in elevations:
            for az in azimuths:
                ray_dir = np.array([
                    np.cos(el) * np.cos(az),
                    np.cos(el) * np.sin(az),
                    np.sin(el),
                ], dtype=np.float32)

                min_dist = self.max_range
                # Ground plane intersection
                if ray_dir[2] < -1e-4:
                    ground_dist = -pos[2] / ray_dir[2]
                    if 0.0 < ground_dist < min_dist:
                        min_dist = ground_dist

                # Obstacle surface proximity
                for obs in obstacles:
                    d = obs.surface_distance(pos)
                    if d < min_dist:
                        min_dist = min(min_dist, max(0.0, d))

                readings.append(float(min(min_dist, self.max_range) / self.max_range))

        return np.array(readings, dtype=np.float32)

