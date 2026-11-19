"""
Behavioral Cloning (BC) & DAgger Warm-Start Pipeline for 3D MAPPO SwarmRL.
Owner: Bhargav Kumar

Ports the rule-based expert heuristic (evaluate.py) to warm-start the
Decentralized MAPPO Actor (pretrain_bc.py) before RL fine-tuning.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swarm_env import SwarmEnv
from models.mappo_agent import MAPPOAgent


def rule_policy(
    obs_dict: Dict[str, np.ndarray],
    env: SwarmEnv,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Sensor-driven frontier explorer and survivor acquisition expert heuristic.
    Ported from evaluate.py: Uses local observations, survivor detection radar,
    and boundary/neighbor avoidance geometry to produce expert continuous 3D actions.
    """
    actions: Dict[str, np.ndarray] = {}
    world_bounds = env.world_size

    for agent_id in env.agents:
        obs = obs_dict[agent_id]
        pos = env.agent_positions[agent_id]
        vel = env.agent_velocities[agent_id]

        # Deconstruct observation: self_state (6), neighbors (k*7), survivor (4)
        survivor_feat = obs[-4:]  # [rel_x, rel_y, rel_z, detected]
        is_detected = survivor_feat[3] > 0.5

        if is_detected:
            # Stage 1: Survivor in range -> Direct approach vector to confirm / rescue
            rel_vec = survivor_feat[:3] * world_bounds
            dist = np.linalg.norm(rel_vec)
            if dist > 1e-4:
                desired_dir = rel_vec / dist
            else:
                desired_dir = np.zeros(3, dtype=np.float32)
            actions[agent_id] = np.clip(desired_dir, -1.0, 1.0).astype(np.float32)
        else:
            # Stage 2: Exploration mode -> Frontier probe heuristic across 8 azimuth angles
            speed = np.linalg.norm(vel)
            if speed > 0.1:
                base_angle = np.arctan2(vel[1], vel[0])
            else:
                base_angle = rng.uniform(-np.pi, np.pi)

            best_score = -1e9
            best_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            # Probe 8 candidate azimuth directions
            for k in range(8):
                angle = base_angle + k * (np.pi / 4.0)
                probe_dir = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
                probe_pt = pos + probe_dir * 30.0

                score = 0.0

                # 1. Boundary clearance penalty
                margin = 15.0
                if (probe_pt[0] < margin or probe_pt[0] > world_bounds[0] - margin or
                    probe_pt[1] < margin or probe_pt[1] > world_bounds[1] - margin):
                    score -= 5.0

                # 2. Altitude centering (stay between 15m and 35m)
                if pos[2] < 15.0:
                    probe_dir[2] = 0.5
                elif pos[2] > 35.0:
                    probe_dir[2] = -0.5

                # 3. Neighbor dispersion penalty (avoid crowding)
                for other_id in env.agents:
                    if other_id == agent_id:
                        continue
                    n_pos = env.agent_positions[other_id]
                    n_dist = np.linalg.norm(probe_pt - n_pos)
                    if n_dist < env.comm_radius * 0.5:
                        score -= 2.0 / max(n_dist, 1.0)

                # 4. Persistence bias for continuing current heading
                if k == 0:
                    score += 0.2

                if score > best_score:
                    best_score = score
                    best_dir = probe_dir

            actions[agent_id] = np.clip(best_dir, -1.0, 1.0).astype(np.float32)

    return actions


def clone_action(
    obs_dict: Dict[str, np.ndarray],
    agent: MAPPOAgent,
) -> Dict[str, np.ndarray]:
    """Queries current MAPPO actor policy for executing agent actions."""
    return agent.select_actions(obs_dict, deterministic=False)


def run_pretrain_bc(
    agents: int = 10,
    episodes_per_round: int = 5,
    rounds: int = 3,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    out_path: str = "checkpoints/bc_warmstart.pt",
    device: str = "cpu",
) -> str:
    """Runs Behavioral Cloning + DAgger warm-start training."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    print("==================================================================")
    print("   SwarmRL: Behavioral Cloning (BC/DAgger) Actor Warm-Start      ")
    print("==================================================================")
    print(f"Agents: {agents} Drones")
    print(f"DAgger Rounds: {rounds} (Round 0 = Teacher flies, Round > 0 = Clone flies)")
    print(f"Episodes per Round: {episodes_per_round}")
    print(f"Fitting Epochs: {epochs} (Batch Size: {batch_size}, LR: {lr})")
    print(f"Output Checkpoint: {out_path}")
    print("------------------------------------------------------------------\n")

    env = SwarmEnv(num_agents=agents)
    agent = MAPPOAgent(
        k_neighbors=env.k_neighbors,
        state_dim=env.state_dim,
        action_dim=3,
        lr_actor=lr,
        device=device,
    )

    dataset_obs: List[np.ndarray] = []
    dataset_act: List[np.ndarray] = []

    for round_idx in range(rounds):
        flyer = "Expert Teacher" if round_idx == 0 else "Student Clone"
        print(f"--- DAgger Round {round_idx + 1}/{rounds} [{flyer} Flying] ---")

        for ep in range(episodes_per_round):
            ep_seed = seed + 100 * round_idx + ep
            obs_dict, _ = env.reset(seed=ep_seed)
            step_count = 0

            while step_count < env.max_steps:
                # 1. Expert provides ground-truth labels for current states
                expert_actions = rule_policy(obs_dict, env, rng)

                # 2. Decide who executes actions in the environment
                if round_idx == 0:
                    exec_actions = expert_actions
                else:
                    exec_actions = clone_action(obs_dict, agent)

                # 3. Record (observation, expert_action) pairs
                for agent_id in env.agents:
                    dataset_obs.append(obs_dict[agent_id].copy())
                    dataset_act.append(expert_actions[agent_id].copy())

                # 4. Step environment
                next_obs, _, terminations, truncations, _ = env.step(exec_actions)
                obs_dict = next_obs
                step_count += 1

                if all(terminations.values()) or all(truncations.values()):
                    break

        total_samples = len(dataset_obs)
        print(f"Dataset Size: {total_samples} samples collected.")

        # Supervised fitting of actor policy
        X = torch.as_tensor(np.array(dataset_obs), dtype=torch.float32, device=device)
        Y = torch.as_tensor(np.array(dataset_act), dtype=torch.float32, device=device)

        dataset_size = X.size(0)
        optimizer = torch.optim.Adam(agent.actor.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            perm = torch.randperm(dataset_size, device=device)
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, dataset_size, batch_size):
                batch_idx = perm[i : i + batch_size]
                x_b, y_b = X[batch_idx], Y[batch_idx]

                action_mean, _ = agent.actor(x_b)
                loss = F.mse_loss(action_mean, y_b)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            if epoch % 2 == 0 or epoch == epochs:
                print(f"  Epoch {epoch:2d}/{epochs} | BC Imitation MSE: {avg_loss:.5f}")

    # Save warm-started checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    agent.save(out_path)
    print(f"\nWarm-start completed successfully! Checkpoint saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwarmRL Behavioral Cloning & DAgger Warm-Start")
    parser.add_argument("--agents", type=int, default=10, help="Number of drones")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per DAgger round")
    parser.add_argument("--rounds", type=int, default=2, help="DAgger rounds")
    parser.add_argument("--epochs", type=int, default=4, help="Actor fitting epochs per round")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Actor learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="checkpoints/bc_warmstart.pt", help="Output checkpoint path")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")

    args = parser.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    run_pretrain_bc(
        agents=args.agents,
        episodes_per_round=args.episodes,
        rounds=args.rounds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        out_path=args.out,
        device=device,
    )
