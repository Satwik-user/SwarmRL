# SwarmRL
Innovative Project 5th sem. Drone Swarm technology for disaster management.


---

## 🚀 Teammate Machine Setup Guide: Running All Ends of SwarmRL

This step-by-step guide is designed for teammates (e.g. running Windows with NVIDIA GPUs, Ubuntu, or WSL2) to set up and run every component ("all ends") of the **SwarmRL** project with single-line, copy-pasteable commands.

---

### Step 0: Clone Repository & Virtual Environment Setup

#### On Windows (PowerShell as Administrator or Standard User)
```powershell
git clone https://github.com/SatwikChowdhury/swarmRL.git; cd swarmRL; python -m venv venv; .\venv\Scripts\Activate.ps1; python -m pip install --upgrade pip; pip install -r requirements.txt
```

#### On Linux / macOS / WSL2 (Bash)
```bash
git clone https://github.com/SatwikChowdhury/swarmRL.git && cd swarmRL && python3 -m venv venv && source venv/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt
```

> **Note for Windows Execution Policy**: If PowerShell blocks script activation, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

### End 1: Run Full Integration & Verification Test Suite

Verify that all modules across all 5 sub-systems (MARL environment, Control Barrier SafetyShield, Bayesian consensus, typed LiDAR perception, and MuJoCo digital twin) pass with 100% success:

```bash
python -m unittest discover tests/
```
*(Expected: Ran 23 tests, 0 failures, 0 errors, OK).*

---

### End 2: 3D MuJoCo Digital Twin Multi-Drone Swarm Flight (Flagship Visualizer)

Renders the trained multi-drone swarm flying inside the high-fidelity 3D MuJoCo disaster city. On machines with an active display / GPU, this launches a hardware-accelerated 60 FPS interactive window. On headless systems, it simulates telemetry and exports 3D plots.

#### 1. Full 10 km Congested Disaster City (6 Drones with MAPPO Policy & SafetyShield)
```bash
python scripts/replay_mujoco_swarm.py --drones 6 --steps 200 --shield
```

#### 2. Fast Lightweight City (Instant Startup on Laptops)
```bash
python scripts/replay_mujoco_swarm.py --drones 4 --compact --steps 150
```

#### 3. Single Drone Model & Helipad Inspector
Inspects the detailed quadcopter chassis, rotating rotors, gimbal cameras, and landing gear:
```bash
python src/ground_station/digital_twin/view_drone.py
```

#### 🎮 Interactive MuJoCo 3D Viewer Keybindings:
| Key | Action | Description |
| :--- | :--- | :--- |
| **`SPACE`** | Auto-Orbit Toggle | Smooth 360° turntable camera rotation around the swarm |
| **`F`** | Camera Mode | Cycles: `Flock` (tracks swarm centroid) $\rightarrow$ `Follow` (chase drone) $\rightarrow$ `Orbit` $\rightarrow$ `Free` |
| **`0` – `9`** | Chase Drone ID | Directly locks third-person follow camera onto drone $0, 1, 2, \dots$ |
| **`P`** | Pause / Resume | Freezes and unfreezes flight physics in real time |
| **`R`** | Reset Camera | Restores default wide-angle mission perspective |
| **`ESC`** | Close Window | Cleanly terminates the 3D viewer session |

---

### End 3: Ground Station Telemetry Dashboard (Web HUD)

Launches the live search-and-rescue mission operations command center.

#### Option A: Standalone Web Server (Port 8502)
```bash
python src/ground_station/dashboard.py --standalone --port 8502
```
*Open your web browser at: [http://localhost:8502](http://localhost:8502)*

#### Option B: Streamlit Dashboard
```bash
streamlit run src/ground_station/dashboard.py
```

---

### End 4: Certified Safety Benchmark & Performance Suite

Evaluates the trained MAPPO policy with Arkendu's continuous Control Barrier Function (CBF) `SafetyShield` enabled vs. unshielded baseline across collision rates, near-misses, and survivor discovery:

```bash
python benchmark.py --shield --episodes 5
```
*(Results are automatically recorded and saved to [data/benchmark_results.json](file:///home/satwik/dev/swarmRL/data/benchmark_results.json)).*

---

### End 5: 3D Trajectory Visualizer & Animated GIF Exporter

Generates an interactive Matplotlib 3D volumetric trajectory view and exports an animated mission GIF:

```bash
python scripts/visualize_swarm_3d.py --agents 6 --steps 100 --shield --save data/swarm_mission_3d.gif
```
*(View the generated GIF animation in [data/swarm_mission_3d.gif](file:///home/satwik/dev/swarmRL/data/swarm_mission_3d.gif)).*

---

### End 6: Reinforcement Learning Policy Training (MAPPO)

Trains the decentralized multi-agent actor-critic neural network in the 3D PettingZoo continuous flight domain.

#### Quick Verification Training (5 Iterations with BC Warmstart & SafetyShield)
```bash
python scripts/train.py --warmstart --iterations 5 --shield
```

#### Production Full Training (30 Iterations)
```bash
python train_mappo.py --warmstart --iterations 30 --shield
```
*(Saves trained neural network weights to [checkpoints/best_policy.pt](file:///home/satwik/dev/swarmRL/checkpoints/best_policy.pt)).*

---

### End 7: Behavioral Cloning (BC) Expert Pre-training

Generates demonstration state-action pairs using Bhargav's heuristic rule-policy expert and pretrains the MAPPO actor network:

```bash
python scripts/pretrain_bc.py --episodes 100 --output checkpoints/bc_warmstart.pt
```

---

### 📋 One-Line Quick Reference Cheat Sheet

| End | Component | Single-Line Pasteable Command |
| :--- | :--- | :--- |
| **Unit Tests** | Full Test Suite (23 Tests) | `python -m unittest discover tests/` |
| **MuJoCo Swarm** | 3D Digital Twin City (6 Drones) | `python scripts/replay_mujoco_swarm.py --drones 6 --steps 200 --shield` |
| **MuJoCo Compact** | Fast 3D City Simulation | `python scripts/replay_mujoco_swarm.py --drones 4 --compact --steps 150` |
| **MuJoCo Drone** | Single Drone Inspector | `python src/ground_station/digital_twin/view_drone.py` |
| **Dashboard** | Web Telemetry HUD (Port 8502) | `python src/ground_station/dashboard.py --standalone --port 8502` |
| **Benchmark** | Safety Evaluation Suite | `python benchmark.py --shield --episodes 5` |
| **3D Trajectory** | Flight Paths & GIF Export | `python scripts/visualize_swarm_3d.py --agents 6 --steps 100 --shield --save data/swarm_mission_3d.gif` |
| **RL Training** | MAPPO Policy Training | `python scripts/train.py --warmstart --iterations 5 --shield` |
| **BC Pretraining** | Expert Policy Demonstrations | `python scripts/pretrain_bc.py --episodes 100 --output checkpoints/bc_warmstart.pt` |
