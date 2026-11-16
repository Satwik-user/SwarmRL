import json
import os

class MetricsLogger:
    """
    Automated Metrics Logger and Report Exporter.
    Owner: Chandan Mondal
    
    Logs episode metrics, exports benchmark JSON/CSVs, and prepares data for plotting.
    """
    def __init__(self, log_dir="data"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.records = []

    def log_episode(self, episode_idx, coverage_pct, rescued, collisions, near_misses, steps):
        entry = {
            "episode": episode_idx,
            "coverage_pct": float(coverage_pct),
            "survivors_rescued": int(rescued),
            "collisions": int(collisions),
            "near_misses": int(near_misses),
            "steps": int(steps),
        }
        self.records.append(entry)

    def export_summary(self, filename="benchmark_summary.json"):
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.records, f, indent=2)
        print(f"[METRICS] Successfully exported benchmark telemetry to {filepath}")
