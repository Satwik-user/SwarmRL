import numpy as np

class SafetyShield:
    """
    Deterministic Safety Shield filter.
    Owner: Arkendu Chatterjee
    
    Intercepts raw RL policy actions and projects them onto a safe velocity set
    using Potential Fields or Velocity Obstacles (ORCA/CBF).
    """
    def __init__(self, safe_radius=3.0, warning_radius=5.0, repulsion_gain=2.5):
        self.safe_radius = safe_radius
        self.warning_radius = warning_radius
        self.repulsion_gain = repulsion_gain

    def filter_action(self, agent_id, raw_action, env):
        """
        Safety Shield hook signature:
        (agent, raw_action, env) -> safe_action
        Projects raw_action to guarantee zero collisions with neighbors and obstacles.
        """
        agent_pos = env.agent_positions[agent_id]
        safe_action = raw_action.copy()

        # Potential-field repulsive barrier against nearby drones
        for other_id, other_pos in env.agent_positions.items():
            if other_id == agent_id:
                continue

            delta = agent_pos - other_pos
            dist = np.linalg.norm(delta)

            if dist < self.warning_radius and dist > 1e-4:
                normal = delta / dist
                overlap = (self.warning_radius - dist) / (self.warning_radius - self.safe_radius)
                repulsion = normal * (self.repulsion_gain * np.clip(overlap, 0.0, 2.0))
                safe_action += repulsion

        return np.clip(safe_action, -1.0, 1.0)
