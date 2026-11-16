# SwarmRL / SWARMDR: System Directory Tree & Architecture Specification

This document defines the modular directory structure, task segregation matrix, and inter-module interface contracts for the SwarmRL project.

---

## 1. Visual Directory Tree

```
swarmRL/
├── configs/                            # Centralized Simulation & Hyperparameter Configurations
│   └── swarm_config.json               # Shared physical limits, world bounds, and RL parameters
│
├── src/                                # Core Modular Source Code
│   ├── env/                            # [OWNER: Satwik Choudhury]
│   │   ├── __init__.py                 # Exports SwarmEnv
│   │   └── swarm_env_3d.py             # 3D Continuous PettingZoo Simulation Environment
│   │
│   ├── rl/                             # [OWNER: Satwik Choudhury]
│   │   ├── __init__.py                 # Exports Actor, Critic, MAPPOAgent
│   │   └── mappo_agent.py              # Permutation-Invariant Self-Attention Actor & CTDE Critic
│   │
│   ├── dynamics/                       # [OWNER: Arkendu Chatterjee]
│   │   ├── __init__.py                 # Exports FlightDynamics
│   │   └── flight_dynamics.py          # Bounded acceleration, aerodynamics, wind gusts, battery
│   │
│   ├── safety/                         # [OWNER: Arkendu Chatterjee]
│   │   ├── __init__.py                 # Exports SafetyShield
│   │   └── safety_shield.py            # Deterministic velocity filter (ORCA / CBF / Potential Field)
│   │
│   ├── coordination/                   # [OWNER: Somnath Saha]
│   │   ├── __init__.py                 # Exports ConsensusEngine, TaskAllocator, SelfHealing
│   │   ├── consensus_engine.py         # Distributed Bayesian/Byzantine survivor voting protocol
│   │   ├── task_allocator.py           # Dynamic role auctioning (thermal vs kit vs comms relay)
│   │   └── self_healing.py             # Heartbeat monitor & Voronoi sector redistribution
│   │
│   ├── perception/                     # [OWNER: Bhargav Kumar]
│   │   ├── __init__.py                 # Exports LiDAR3D
│   │   └── lidar_sensor.py             # Multi-beam 3D LiDAR raycasting & proximity sensor
│   │
│   ├── mapping/                        # [OWNER: Bhargav Kumar]
│   │   ├── __init__.py                 # Exports OccupancyGrid3D
│   │   └── occupancy_grid.py           # Onboard Sparse 3D OctoMap & P2P frontier delta exchange
│   │
│   ├── communication/                  # [OWNER: Chandan Mondal]
│   │   ├── __init__.py                 # Exports MeshNetworkSimulator
│   │   └── comm_mesh.py                # Range-constrained lossy wireless mesh & packet jitter
│   │
│   └── ground_station/                 # [OWNER: Chandan Mondal]
│       ├── __init__.py                 # Exports MetricsLogger
│       └── metrics_logger.py           # Telemetry ingestion, benchmark exporter & plot generator
│
├── scripts/                            # Executable CLI Workflows
│   ├── train.py                        # Master MAPPO training pipeline
│   └── benchmark.py                    # Multi-episode academic benchmark runner
│
├── checkpoints/                        # Trained Neural Network Weights (.pt, .onnx)
│   ├── best_policy.pt                  # Active highest-coverage checkpoint
│   └── policy_iter_*.pt                # Intermediate epoch checkpoints
│
├── data/                               # Generated Outputs & Benchmark Metrics
│   └── benchmark_results.json          # Academic benchmark logs (coverage, collisions, rescues)
│
├── tests/                              # Module Verification & Test Suite
├── SwarmRL_Development_Lifecycle.md    # Master University Project Charter & Roadmap
├── requirements.txt                    # Project dependencies (torch, gymnasium, pettingzoo)
└── README.md                           # Quickstart guide
```

---

## 2. Module Ownership & Interface Contracts

| Directory | Owner | Core Class / Hook | Input Contract | Output Contract |
| :--- | :--- | :--- | :--- | :--- |
| **`src/env/`** | **Satwik Choudhury** | `SwarmEnv` | Action dict: `{agent_id: np.ndarray(3,)}` in $[-1, 1]$ | `(obs_dict, rewards_dict, term_dict, trunc_dict, info_dict)` |
| **`src/rl/`** | **Satwik Choudhury** | `MAPPOAgent` | Local observation vector (38 dims) | Action vector (3 dims) + log-prob |
| **`src/dynamics/`** | **Arkendu Chatterjee**| `FlightDynamics` | `(velocity, thrust_cmd, wind_vector)` | Acceleration vector $\vec{a}$ & updated battery level |
| **`src/safety/`** | **Arkendu Chatterjee**| `SafetyShield` | `(agent_id, raw_action, env_state)` | Projected guaranteed safe action vector $\vec{a}_{\text{safe}}$ |
| **`src/coordination/`**| **Somnath Saha** | `ConsensusEngine` | `(drone_id, target_coords, confidence)` | Quorum agreement flag (`True` / `False`) |
| **`src/perception/`** | **Bhargav Kumar** | `LiDAR3D` | `(agent_id, env_state)` | Normalized distance array `np.ndarray(16,)` |
| **`src/mapping/`** | **Bhargav Kumar** | `OccupancyGrid3D` | Sensor point clouds & neighbor deltas | Sparse 3D occupancy voxel grid & frontier deltas |
| **`src/communication/`**| **Chandan Mondal** | `MeshNetworkSimulator` | `(pos_agent1, pos_agent2)` | `(connected_bool, packet_loss_prob)` |
| **`src/ground_station/`**| **Chandan Mondal**| `MetricsLogger` | `env.get_telemetry()` dictionary | JSON/CSV exports & 3D visualization stream |

---

## 3. Inter-Module Integration Hooks

To maintain total independence between team members and prevent code conflicts, `SwarmEnv` provides clean integration hooks:

```python
# 1. Satwik's Environment Initialization
from src.env import SwarmEnv
env = SwarmEnv()

# 2. Arkendu's Safety Shield Hook
from src.safety import SafetyShield
shield = SafetyShield(safe_radius=3.0)
env.safety_shield_hook = shield.filter_action  # Intercepts actions before physical update

# 3. Bhargav's LiDAR Perception Hook
from src.perception import LiDAR3D
lidar = LiDAR3D()
env.sensor_hook = lidar.scan                  # Appends LiDAR rays to agent observation vector

# 4. Chandan's Ground Station Telemetry Hook
telemetry_data = env.get_telemetry()           # Returns real-time 3D coordinates & swarm health
```

---

## 5. Standardized Execution Commands

```bash
# Train the 3D MAPPO Swarm Policy
python3 scripts/train.py --agents 20 --iterations 30

# Run Academic Benchmarks (Exports to data/benchmark_results.json)
python3 scripts/benchmark.py --agents 20 --episodes 5 --checkpoint checkpoints/best_policy.pt
```
