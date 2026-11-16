class SelfHealingManager:
    """
    Self-Healing Swarm Manager.
    Owner: Somnath Saha
    
    Tracks neighbor heartbeats and triggers sector rebalancing if a drone crashes or runs out of battery.
    """
    def __init__(self, heartbeat_timeout_steps=10):
        self.heartbeat_timeout = heartbeat_timeout_steps
        self.last_heartbeats = {}

    def record_heartbeat(self, agent_id, step):
        self.last_heartbeats[agent_id] = step

    def check_dropouts(self, current_step):
        dead_drones = [
            agent for agent, last_seen in self.last_heartbeats.items()
            if (current_step - last_seen) > self.heartbeat_timeout
        ]
        return dead_drones
