import numpy as np
import gymnasium as gym
from pettingzoo.utils.env import ParallelEnv

from src.perception.drone_model import Obstacle
from src.perception.thermal import _los_clear
from src.safety.safety_shield import SafetyShield, SafetyConfig, AABB, Neighbor

class SwarmEnv(ParallelEnv):
    """
    SwarmRL 3D Multi-Agent Reinforcement Learning Environment.
    
    Part of Satwik's Core Architecture:
    - True 3D continuous flight domain (X, Y, Z coordinates and velocities).
    - Continuous 3D action space [ax, ay, az] in [-1.0, 1.0].
    - Decoupled modular design with clear integration hooks for team members:
        * Arkendu: `safety_shield_hook` (ORCA / Control Barrier Functions)
        * Bhargav: `sensor_hook` (3D LiDAR raycasting & OctoMap)
        * Chandan: `get_telemetry()` (Ground Station digital twin streaming)
        * Somnath: `task_allocation_hook` (Dynamic role & sector management)
    - Headless, ultra-fast simulation engine (pure NumPy tensor math, zero UI bloat).
    """
    metadata = {
        "name": "swarm_env_3d_v1",
        "render_modes": [],
    }

    def __init__(
        self,
        num_agents=20,
        comm_radius=50.0,
        num_survivors=8,
        world_size=(200.0, 200.0, 50.0),  # 200m x 200m x 50m altitude
        max_steps=300,
        k_neighbors=4,
        enable_thermal_los=True,
        enable_safety_shield=False,
    ):
        super().__init__()
        self.num_agents_count = num_agents
        self.comm_radius = float(comm_radius)
        self.num_survivors = num_survivors
        self.world_size = np.array(world_size, dtype=np.float32)
        self.max_steps = max_steps
        self.k_neighbors = k_neighbors
        self.enable_thermal_los = enable_thermal_los
        self.enable_safety_shield = enable_safety_shield

        self.possible_agents = [f"drone_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]

        # Flight dynamics limits
        self.drone_radius = 1.5           # Collision sphere radius (meters)
        self.warning_radius = 4.0         # Graded near-miss boundary (meters)
        self.max_speed = 8.0              # Max velocity (m/s)
        self.accel_scale = 2.0            # Max acceleration impulse (m/s^2)
        self.survivor_detect_radius = 30.0
        self.survivor_rescue_radius = 5.0

        # 3D Obstacles: [x_min, y_min, z_min, dx, dy, dz]
        self.obstacles = np.array([
            [30.0, 30.0, 0.0, 30.0, 30.0, 35.0],
            [120.0, 40.0, 0.0, 40.0, 30.0, 40.0],
            [40.0, 120.0, 0.0, 35.0, 45.0, 30.0],
            [130.0, 120.0, 0.0, 35.0, 40.0, 45.0],
            [80.0, 80.0, 0.0, 30.0, 30.0, 25.0],
        ], dtype=np.float32)

        # Perception obstacles (Bhargav's model for thermal LOS occlusion)
        self.perception_obstacles = [
            Obstacle(
                name=f"building_{i}",
                obs_type="Building",
                pos=np.array([obs[0] + obs[3] / 2.0, obs[1] + obs[4] / 2.0, obs[2] + obs[5] / 2.0]),
                shape="box",
                size_params=np.array([obs[3], obs[4], obs[5]]),
            )
            for i, obs in enumerate(self.obstacles)
        ]

        # Safety AABBs (Arkendu's swept-path collision shield)
        self.safety_aabbs = [
            AABB(
                minimum=(float(obs[0]), float(obs[1]), float(obs[2])),
                maximum=(float(obs[0] + obs[3]), float(obs[1] + obs[4]), float(obs[2] + obs[5])),
            )
            for obs in self.obstacles
        ]
        self.safety_shield = SafetyShield(
            SafetyConfig(
                drone_radius=self.drone_radius,
                separation_margin=2.0,
                max_speed=self.max_speed,
            )
        )

        # Volumetric 3D Search Coverage Grid: 20x20x5 voxels
        self.grid_dims = np.array([20, 20, 5], dtype=np.int32)
        self.voxel_size = self.world_size / self.grid_dims

        # Continuous 3D Action Space: [ax, ay, az] thrust vectors
        self.action_spaces = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Decentralized Observation Space Breakdown (per agent):
        # 1. Self State (6): [x/X, y/Y, z/Z, vx/V, vy/V, vz/V]
        # 2. K-Nearest Neighbors (K * 7): [dx/r, dy/r, dz/r, dvx, dvy, dvz, active_mask]
        # 3. Survivor Search Radar (4): [rel_x, rel_y, rel_z, detected_flag]
        # Total = 6 + (K * 7) + 4 = 10 + (4 * 7) = 38 floats
        self.obs_dim = 6 + (self.k_neighbors * 7) + 4
        self.observation_spaces = {
            agent: gym.spaces.Box(low=-3.0, high=3.0, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Centralized State Dimension (for CTDE Critic):
        # All drones (N * 6) + All survivors (M * 4: x, y, z, is_found)
        self.state_dim = (self.num_agents_count * 6) + (self.num_survivors * 4)

        # Team Integration Hooks
        self.safety_shield_hook = None  # Arkendu: function(agent, raw_action, state) -> safe_action
        self.sensor_hook = None         # Bhargav: function(agent, state) -> extra_sensor_features

        # Internal simulation state
        self.agent_positions = {}
        self.agent_velocities = {}
        self.survivor_positions = np.zeros((self.num_survivors, 3), dtype=np.float32)
        self.survivor_found = np.zeros(self.num_survivors, dtype=bool)
        self.coverage_grid = np.zeros(tuple(self.grid_dims), dtype=bool)

        self.current_step = 0
        self.episode_collisions = 0
        self.episode_near_misses = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.episode_collisions = 0
        self.episode_near_misses = 0
        self.coverage_grid.fill(False)

        # Initialize 3D positions outside obstacles with initial separation
        self.agent_positions = {}
        self.agent_velocities = {}
        for agent in self.agents:
            self.agent_positions[agent] = self._sample_free_position(min_z=5.0, others=list(self.agent_positions.values()))
            self.agent_velocities[agent] = np.zeros(3, dtype=np.float32)

        # Initialize 3D survivor locations (ground or collapsed debris levels)
        for i in range(self.num_survivors):
            self.survivor_positions[i] = self._sample_free_position(min_z=0.5, max_z=10.0)
        self.survivor_found.fill(False)

        self._update_coverage()

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _sample_free_position(self, min_z=2.0, max_z=None, min_dist_to_others=6.0, others=None):
        max_alt = max_z if max_z is not None else self.world_size[2] - 2.0
        for _ in range(500):
            pt = np.array([
                np.random.uniform(5.0, self.world_size[0] - 5.0),
                np.random.uniform(5.0, self.world_size[1] - 5.0),
                np.random.uniform(min_z, max_alt),
            ], dtype=np.float32)

            inside = False
            for obs in self.obstacles:
                if (obs[0] - self.drone_radius <= pt[0] <= obs[0] + obs[3] + self.drone_radius and
                    obs[1] - self.drone_radius <= pt[1] <= obs[1] + obs[4] + self.drone_radius and
                    obs[2] <= pt[2] <= obs[2] + obs[5] + self.drone_radius):
                    inside = True
                    break
            if inside:
                continue

            if others:
                if any(np.linalg.norm(pt - o) < min_dist_to_others for o in others):
                    continue

            return pt
        return np.array([10.0, 10.0, min_z], dtype=np.float32)

    def _update_coverage(self):
        new_cells = 0
        for agent in self.agents:
            pos = self.agent_positions[agent]
            idx = np.clip(np.floor(pos / self.voxel_size).astype(np.int32), 0, self.grid_dims - 1)
            if not self.coverage_grid[tuple(idx)]:
                self.coverage_grid[tuple(idx)] = True
                new_cells += 1
        return new_cells

    def _get_observation(self, agent):
        pos = self.agent_positions[agent]
        vel = self.agent_velocities[agent]

        # 1. Self State (6)
        self_state = np.concatenate([
            pos / self.world_size,
            vel / self.max_speed,
        ]).astype(np.float32)

        # 2. K-Nearest Neighbors within comm_radius (K * 7)
        neighbor_data = []
        for other in self.agents:
            if other == agent:
                continue
            other_pos = self.agent_positions[other]
            other_vel = self.agent_velocities[other]
            dist = float(np.linalg.norm(other_pos - pos))
            neighbor_data.append((dist, other_pos, other_vel))

        neighbor_data.sort(key=lambda x: x[0])

        neighbor_feats = []
        for k in range(self.k_neighbors):
            if k < len(neighbor_data):
                d, n_pos, n_vel = neighbor_data[k]
                if d <= self.comm_radius:
                    d_rel = (n_pos - pos) / self.comm_radius
                    v_rel = (n_vel - vel) / (2.0 * self.max_speed)
                    neighbor_feats.extend([d_rel[0], d_rel[1], d_rel[2], v_rel[0], v_rel[1], v_rel[2], 1.0])
                    continue
            neighbor_feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # 3. Survivor Radar (4)
        survivor_feat = [0.0, 0.0, 0.0, 0.0]
        unfound = np.where(~self.survivor_found)[0]
        if len(unfound) > 0:
            dists = np.linalg.norm(self.survivor_positions[unfound] - pos, axis=1)
            sorted_indices = np.argsort(dists)
            for s_cand in sorted_indices:
                cand_idx = unfound[s_cand]
                cand_dist = dists[s_cand]
                if cand_dist > self.survivor_detect_radius:
                    break
                target_pos = self.survivor_positions[cand_idx]
                has_los = True
                if self.enable_thermal_los and self.perception_obstacles:
                    has_los = _los_clear(pos, target_pos, self.perception_obstacles)
                if has_los:
                    rel_t = (target_pos - pos) / self.world_size
                    survivor_feat = [rel_t[0], rel_t[1], rel_t[2], 1.0]
                    break

        obs = np.concatenate([self_state, neighbor_feats, survivor_feat]).astype(np.float32)

        # Optional Hook for Bhargav's LiDAR/sensor extension
        if self.sensor_hook is not None:
            extra = self.sensor_hook(agent, self)
            obs = np.concatenate([obs, extra]).astype(np.float32)

        return obs

    def get_global_state(self):
        """Global state vector for CTDE Centralized Critic."""
        agent_feats = []
        for agent in self.possible_agents:
            if agent in self.agent_positions:
                p = self.agent_positions[agent]
                v = self.agent_velocities[agent]
                agent_feats.extend([*(p / self.world_size), *(v / self.max_speed)])
            else:
                agent_feats.extend([0.0] * 6)

        survivor_feats = []
        for i in range(self.num_survivors):
            sp = self.survivor_positions[i]
            found = 1.0 if self.survivor_found[i] else 0.0
            survivor_feats.extend([*(sp / self.world_size), found])

        return np.concatenate([agent_feats, survivor_feats]).astype(np.float32)

    def step(self, actions):
        self.current_step += 1
        old_positions = {a: self.agent_positions[a].copy() for a in self.agents}
        rewards = {agent: -0.01 for agent in self.agents}

        # Apply Safety Shield Hook if provided by Arkendu or built-in deterministic shield
        effective_actions = {}
        for agent in self.agents:
            raw_act = actions.get(agent, np.zeros(3, dtype=np.float32))
            if self.safety_shield_hook is not None:
                effective_actions[agent] = self.safety_shield_hook(agent, raw_act, self)
            elif self.enable_safety_shield and self.safety_shield is not None:
                p_cur = tuple(float(x) for x in self.agent_positions[agent])
                v_des = tuple(float(x) for x in self.agent_velocities[agent] + np.clip(raw_act, -1.0, 1.0) * self.accel_scale)
                neighbors = [
                    Neighbor(
                        position=tuple(float(x) for x in self.agent_positions[o]),
                        velocity=tuple(float(x) for x in self.agent_velocities[o]),
                    )
                    for o in self.agents if o != agent
                ]
                shield_res = self.safety_shield.filter_velocity(
                    p_cur, v_des, obstacles=self.safety_aabbs, neighbors=neighbors
                )
                self.agent_velocities[agent] = np.array(shield_res.velocity, dtype=np.float32)
                effective_actions[agent] = None
            else:
                effective_actions[agent] = raw_act

        # 1. 3D Kinematics Integration
        for agent in self.agents:
            if effective_actions[agent] is not None:
                act = np.clip(effective_actions[agent], -1.0, 1.0)
                self.agent_velocities[agent] += act * self.accel_scale

                # Velocity capping
                spd = np.linalg.norm(self.agent_velocities[agent])
                if spd > self.max_speed:
                    self.agent_velocities[agent] = (self.agent_velocities[agent] / spd) * self.max_speed

            # 3D Position update
            self.agent_positions[agent] += self.agent_velocities[agent]

            # 3D World Boundary Restraints
            for dim in range(3):
                if self.agent_positions[agent][dim] < self.drone_radius:
                    self.agent_positions[agent][dim] = self.drone_radius
                    self.agent_velocities[agent][dim] *= -0.5
                elif self.agent_positions[agent][dim] > self.world_size[dim] - self.drone_radius:
                    self.agent_positions[agent][dim] = self.world_size[dim] - self.drone_radius
                    self.agent_velocities[agent][dim] *= -0.5

        # 2. 3D Obstacle Collisions
        for agent in self.agents:
            pos = self.agent_positions[agent]
            for obs in self.obstacles:
                if (obs[0] - self.drone_radius <= pos[0] <= obs[0] + obs[3] + self.drone_radius and
                    obs[1] - self.drone_radius <= pos[1] <= obs[1] + obs[4] + self.drone_radius and
                    obs[2] <= pos[2] <= obs[2] + obs[5] + self.drone_radius):
                    rewards[agent] -= 15.0
                    self.episode_collisions += 1
                    self.agent_velocities[agent] *= -0.5
                    self.agent_positions[agent] = old_positions[agent]
                    break

        # 3. 3D Continuous Swept-Path Inter-Agent Collision Check
        agent_list = list(self.agents)
        for i in range(len(agent_list)):
            a1 = agent_list[i]
            p1_old, p1_new = old_positions[a1], self.agent_positions[a1]
            for j in range(i + 1, len(agent_list)):
                a2 = agent_list[j]
                p2_old, p2_new = old_positions[a2], self.agent_positions[a2]

                dp0 = p1_old - p2_old
                dv = (p1_new - p1_old) - (p2_new - p2_old)
                dv_sq = np.dot(dv, dv)
                s_star = np.clip(-np.dot(dp0, dv) / dv_sq, 0.0, 1.0) if dv_sq > 1e-6 else 0.0
                min_dist = float(np.linalg.norm(dp0 + s_star * dv))

                if min_dist < (2.0 * self.drone_radius):
                    rewards[a1] -= 20.0
                    rewards[a2] -= 20.0
                    self.episode_collisions += 1
                elif min_dist < self.warning_radius:
                    penalty = 2.0 * ((self.warning_radius - min_dist) / (self.warning_radius - 2.0 * self.drone_radius))
                    rewards[a1] -= penalty
                    rewards[a2] -= penalty
                    self.episode_near_misses += 1

        # 4. Volumetric Coverage Exploration Reward
        new_voxels = self._update_coverage()
        cov_reward = new_voxels * 0.2
        for agent in self.agents:
            rewards[agent] += cov_reward

        # 5. 3D Survivor Search & Rescue
        for s_idx in range(self.num_survivors):
            if not self.survivor_found[s_idx]:
                s_pos = self.survivor_positions[s_idx]
                for agent in self.agents:
                    dist_to_survivor = float(np.linalg.norm(self.agent_positions[agent] - s_pos))
                    if dist_to_survivor <= self.survivor_rescue_radius:
                        has_los = True
                        if self.enable_thermal_los and self.perception_obstacles:
                            has_los = _los_clear(self.agent_positions[agent], s_pos, self.perception_obstacles)
                        if has_los:
                            self.survivor_found[s_idx] = True
                            rewards[agent] += 15.0
                            for other in self.agents:
                                rewards[other] += 5.0
                            break

        done = self.current_step >= self.max_steps or bool(np.all(self.survivor_found))
        terminations = {agent: bool(np.all(self.survivor_found)) for agent in self.agents}
        truncations = {agent: self.current_step >= self.max_steps for agent in self.agents}
        infos = {
            agent: {
                "coverage_pct": float(np.mean(self.coverage_grid) * 100.0),
                "survivors_rescued": int(np.sum(self.survivor_found)),
                "collisions": self.episode_collisions,
                "near_misses": self.episode_near_misses,
            }
            for agent in self.agents
        }

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def get_telemetry(self):
        """
        Export structured 3D telemetry dictionary for Chandan's Ground Station UI.
        """
        return {
            "step": self.current_step,
            "drones": {
                agent: {
                    "position": self.agent_positions[agent].tolist(),
                    "velocity": self.agent_velocities[agent].tolist(),
                }
                for agent in self.agents
            },
            "survivors": [
                {
                    "position": self.survivor_positions[i].tolist(),
                    "rescued": bool(self.survivor_found[i]),
                }
                for i in range(self.num_survivors)
            ],
            "obstacles": self.obstacles.tolist(),
            "coverage_pct": float(np.mean(self.coverage_grid) * 100.0),
            "collisions": self.episode_collisions,
            "near_misses": self.episode_near_misses,
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def get_thermal_contacts(self, agent_id: str):
        """
        Extracts two-stage thermal contacts (faint vs confirmed) for an agent,
        incorporating line-of-sight obstruction from obstacles (Bhargav's model).
        """
        pos = self.agent_positions[agent_id]
        faint_contacts = []
        confirmed_contacts = []
        for s_idx in range(self.num_survivors):
            if self.survivor_found[s_idx]:
                continue
            s_pos = self.survivor_positions[s_idx]
            dist = float(np.linalg.norm(s_pos - pos))
            if dist <= self.survivor_detect_radius:
                los_clear = True
                if self.enable_thermal_los and self.perception_obstacles:
                    los_clear = _los_clear(pos, s_pos, self.perception_obstacles)
                if los_clear:
                    contact_info = {
                        "id": f"survivor_{s_idx}",
                        "index": s_idx,
                        "distance": dist,
                        "position": s_pos.copy(),
                    }
                    if dist <= self.survivor_rescue_radius:
                        confirmed_contacts.append(contact_info)
                    else:
                        faint_contacts.append(contact_info)
        return {"faint": faint_contacts, "confirmed": confirmed_contacts}

    def get_consensus_observations(self, step: int):
        """
        Generates Observation reports from all drones detecting survivors,
        ready for decentralized Bayesian fusion in Somnath's ConsensusEngine.
        """
        from src.coordination.consensus_engine import Observation
        observations = []
        for idx, agent_id in enumerate(self.agents):
            contacts = self.get_thermal_contacts(agent_id)
            for c in contacts["confirmed"]:
                observations.append(
                    Observation(
                        drone_id=idx,
                        sensor="thermal",
                        position=(float(c["position"][0]), float(c["position"][1]), float(c["position"][2])),
                        confidence=0.95,
                        timestamp=step,
                    )
                )
            for f in contacts["faint"]:
                conf = max(0.55, 0.85 - (f["distance"] / self.survivor_detect_radius) * 0.3)
                observations.append(
                    Observation(
                        drone_id=idx,
                        sensor="thermal",
                        position=(float(f["position"][0]), float(f["position"][1]), float(f["position"][2])),
                        confidence=float(conf),
                        timestamp=step,
                    )
                )
        return observations

