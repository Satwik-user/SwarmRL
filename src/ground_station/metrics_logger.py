
import csv
import json
import os
from copy import deepcopy


class MetricsLogger:
    """
    Ground Station Metrics Logger.
    Owner: Chandan Mondal

    Features:
    - Drone telemetry logging
    - Episode performance tracking
    - JSON and CSV exports
    - Dashboard visualization data
    """

    def __init__(self, log_dir="data"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.records = []
        self.telemetry_history = []

    def log_episode(
        self, episode_idx, coverage_pct, rescued,
        collisions, near_misses, steps
    ):
        entry = {
            "episode": int(episode_idx),
            "coverage_pct": float(coverage_pct),
            "survivors_rescued": int(rescued),
            "collisions": int(collisions),
            "near_misses": int(near_misses),
            "steps": int(steps)
        }

        self.records.append(entry)

    def log_telemetry(self, telemetry_data):
        """Store telemetry from env.get_telemetry()."""

        if not isinstance(telemetry_data, dict):
            raise TypeError("Telemetry must be a dictionary")

        required = [
            "step", "drones", "survivors", "obstacles",
            "coverage_pct", "collisions", "near_misses"
        ]

        for field in required:
            if field not in telemetry_data:
                raise ValueError(f"Missing telemetry field: {field}")

        self.telemetry_history.append(deepcopy(telemetry_data))

    def export_summary(self, filename="benchmark_summary.json"):
        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, "w") as file:
            json.dump(self.records, file, indent=2)

        print(f"[METRICS] Summary exported: {filepath}")

    def export_telemetry_json(self, filename="telemetry_history.json"):
        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, "w") as file:
            json.dump(self.telemetry_history, file, indent=2)

        print(f"[METRICS] JSON exported: {filepath}")

    def export_telemetry_csv(self, filename="telemetry_history.csv"):
        filepath = os.path.join(self.log_dir, filename)

        fields = [
            "step", "drone_count", "survivor_count",
            "rescued_count", "obstacle_count",
            "coverage_pct", "collisions", "near_misses",
            "drones", "survivors", "obstacles"
        ]

        with open(filepath, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()

            for data in self.telemetry_history:
                survivors = data["survivors"]

                writer.writerow({
                    "step": data["step"],
                    "drone_count": len(data["drones"]),
                    "survivor_count": len(survivors),
                    "rescued_count": sum(
                        1 for s in survivors if s["rescued"]
                    ),
                    "obstacle_count": len(data["obstacles"]),
                    "coverage_pct": data["coverage_pct"],
                    "collisions": data["collisions"],
                    "near_misses": data["near_misses"],
                    "drones": json.dumps(data["drones"]),
                    "survivors": json.dumps(survivors),
                    "obstacles": json.dumps(data["obstacles"])
                })

        print(f"[METRICS] CSV exported: {filepath}")

    def get_latest_telemetry(self):
        if not self.telemetry_history:
            return None

        return deepcopy(self.telemetry_history[-1])

    def get_telemetry_history(self):
        return deepcopy(self.telemetry_history)

