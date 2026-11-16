import numpy as np


class MeshNetworkSimulator:
    """Simulates wireless communication between drones."""

    def __init__(
        self,
        comm_radius=50.0,
        base_drop_rate=0.05,
        base_latency_ms=20.0,
        latency_jitter_ms=10.0
    ):
        if comm_radius <= 0:
            raise ValueError("Communication radius must be positive")
        if not 0 <= base_drop_rate <= 1:
            raise ValueError("Base drop rate must be between 0 and 1")
        if latency_jitter_ms < 0:
            raise ValueError("Latency jitter cannot be negative")

        self.comm_radius = float(comm_radius)
        self.base_drop_rate = float(base_drop_rate)
        self.base_latency_ms = float(base_latency_ms)
        self.latency_jitter_ms = float(latency_jitter_ms)

    def can_communicate(self, pos1, pos2):
        pos1 = np.asarray(pos1, dtype=float)
        pos2 = np.asarray(pos2, dtype=float)

        if pos1.shape != pos2.shape:
            raise ValueError("Drone positions must have matching dimensions")
        if not np.all(np.isfinite(pos1)) or not np.all(np.isfinite(pos2)):
            raise ValueError("Positions must contain finite values")

        dist = float(np.linalg.norm(pos1 - pos2))

        if dist > self.comm_radius:
            return False, 1.0

        ratio = dist / self.comm_radius
        loss_prob = self.base_drop_rate + (
            1.0 - self.base_drop_rate
        ) * ratio ** 2

        return True, float(min(loss_prob, 0.99))

    def is_packet_delivered(self, packet_loss_prob):
        if not 0 <= packet_loss_prob <= 1:
            raise ValueError("Invalid packet loss probability")
        return bool(np.random.random() >= packet_loss_prob)

    def get_latency(self):
        jitter = np.random.uniform(
            -self.latency_jitter_ms,
            self.latency_jitter_ms
        )
        return max(0.0, self.base_latency_ms + jitter)
