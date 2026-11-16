import numpy as np

class MeshNetworkSimulator:
    """
    Simulated P2P Mesh Communication Network.
    Owner: Chandan Mondal
    
    Models realistic ad-hoc wireless links between drones:
    - Euclidean distance cutoff (`comm_radius`)
    - Distance-dependent packet loss probability
    - Latency jitter
    """
    def __init__(self, comm_radius=50.0, base_drop_rate=0.05):
        self.comm_radius = comm_radius
        self.base_drop_rate = base_drop_rate

    def can_communicate(self, pos1, pos2):
        dist = np.linalg.norm(pos1 - pos2)
        if dist > self.comm_radius:
            return False, 0.0

        # Loss probability increases quadratically near the edge of the range
        loss_prob = self.base_drop_rate + (1.0 - self.base_drop_rate) * (dist / self.comm_radius)**2
        success = np.random.uniform(0.0, 1.0) > loss_prob
        return success, dist
