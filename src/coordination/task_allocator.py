class TaskAllocator:
    """
    Dynamic Role & Task Allocation.
    Owner: Somnath Saha
    
    Assigns drone roles (Explorer, Responder, Comms Relay) based on battery and payload.
    """
    def __init__(self):
        self.roles = {}  # {agent_id: "explorer" | "responder" | "relay"}

    def evaluate_role(self, agent_id, battery_level, payload_type):
        if battery_level < 25.0:
            return "rtl"  # Return to launch
        return "explorer"
