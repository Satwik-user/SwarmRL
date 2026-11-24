#!/usr/bin/env python3
"""
MuJoCo Swarm Playback & Live Digital Twin Flight Bridge.
Single-line CLI runner for SwarmRL multi-drone flight execution.

Usage:
  python scripts/replay_mujoco_swarm.py --drones 6 --steps 200
  python scripts/replay_mujoco_swarm.py --drones 4 --compact --steps 150
"""

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.ground_station.digital_twin.view_swarm import main

if __name__ == "__main__":
    main()
