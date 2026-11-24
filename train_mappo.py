import os
import time
import argparse
import numpy as np
import torch
from swarm_env import SwarmEnv
from models.mappo_agent import MAPPOAgent

def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    print("==================================================================")
    print("      SwarmRL: 3D MAPPO Multi-Agent Policy Training Engine        ")
    print("==================================================================")
    print(f"Device: {device.upper()}")
    print(f"Agents: {args.agents} 3D Drones")
    print(f"World Bounds: 200m x 200m x 50m (Volume: 2,000,000 m³)")
    print(f"Rollout Length: {args.rollout_steps} steps ({args.agents * args.rollout_steps} transitions/iter)")
    print(f"Total Iterations: {args.iterations}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print("------------------------------------------------------------------\n")

    env = SwarmEnv(
        num_agents=args.agents,
        comm_radius=args.comm_radius,
        num_survivors=args.survivors,
        max_steps=args.max_steps,
        k_neighbors=args.k_neighbors,
    )

    agent = MAPPOAgent(
        k_neighbors=args.k_neighbors,
        state_dim=env.state_dim,
        action_dim=3,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        device=device,
    )

    if args.warmstart and os.path.exists(args.warmstart):
        agent.load(args.warmstart, load_critic=False)
        print(f"[STATUS] Initialized actor weights from BC warm-start checkpoint: {args.warmstart}\n")

    obs, infos = env.reset(seed=args.seed)
    best_coverage = 0.0

    start_time = time.time()

    for iteration in range(1, args.iterations + 1):
        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_values = []
        batch_dones = []
        batch_global_states = []

        ep_rewards = []
        ep_coverages = []
        ep_rescues = []
        ep_collisions = []

        # 1. Rollout Collection
        for step in range(args.rollout_steps):
            g_state = env.get_global_state()
            v_val = agent.get_value(g_state)

            actions, log_probs = agent.select_actions(obs)
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            done = any(terminations.values()) or any(truncations.values())

            for a_name in env.agents:
                batch_obs.append(obs[a_name])
                batch_actions.append(actions[a_name])
                batch_log_probs.append(log_probs[a_name])
                batch_rewards.append(rewards[a_name])
                batch_values.append(v_val)
                batch_dones.append(float(done))
                batch_global_states.append(g_state)

            obs = next_obs

            if done:
                first_info = list(infos.values())[0]
                ep_rewards.append(sum(rewards.values()))
                ep_coverages.append(first_info["coverage_pct"])
                ep_rescues.append(first_info["survivors_rescued"])
                ep_collisions.append(first_info["collisions"])
                obs, infos = env.reset()

        if not ep_coverages:
            first_info = list(infos.values())[0]
            ep_coverages.append(first_info["coverage_pct"])
            ep_rescues.append(first_info["survivors_rescued"])
            ep_collisions.append(first_info["collisions"])

        # 2. GAE Advantage Calculation
        final_g_state = env.get_global_state()
        final_value = agent.get_value(final_g_state)

        T = len(batch_rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            next_val = final_value if t == T - 1 else batch_values[t + 1]
            non_terminal = 1.0 - batch_dones[t]
            delta = batch_rewards[t] + (args.gamma * next_val * non_terminal) - batch_values[t]
            last_gae = delta + (args.gamma * args.gae_lambda * non_terminal * last_gae)
            advantages[t] = last_gae
            returns[t] = advantages[t] + batch_values[t]

        rollouts = {
            "obs": np.array(batch_obs, dtype=np.float32),
            "actions": np.array(batch_actions, dtype=np.float32),
            "log_probs": np.array(batch_log_probs, dtype=np.float32),
            "advantages": advantages,
            "returns": returns,
            "global_states": np.array(batch_global_states, dtype=np.float32),
        }

        # 3. Policy Update
        actor_loss, critic_loss = agent.update(rollouts, ppo_epochs=args.epochs, batch_size=args.batch_size)

        # 4. Logging
        avg_cov = np.mean(ep_coverages)
        avg_resc = np.mean(ep_rescues)
        avg_coll = np.mean(ep_collisions)
        fps = int((args.agents * args.rollout_steps) / (time.time() - start_time))

        print(
            f"Iter [{iteration:03d}/{args.iterations:03d}] | "
            f"3D Coverage: {avg_cov:5.1f}% | "
            f"Rescued: {avg_resc:3.1f}/{args.survivors} | "
            f"Collisions: {avg_coll:4.1f} | "
            f"Actor Loss: {actor_loss:6.3f} | "
            f"Critic Loss: {critic_loss:6.2f} | "
            f"Speed: {fps:4d} FPS"
        )

        if iteration == 1 or avg_cov > best_coverage:
            best_coverage = avg_cov
            agent.save(os.path.join(args.checkpoint_dir, "best_policy.pt"))

        if iteration % args.save_interval == 0:
            agent.save(os.path.join(args.checkpoint_dir, f"policy_iter_{iteration}.pt"))

    env.close()
    print("\nTraining completed successfully! Model saved to checkpoints/best_policy.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwarmRL 3D MAPPO Training Engine")
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument("--comm-radius", type=float, default=50.0)
    parser.add_argument("--survivors", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--k-neighbors", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--warmstart", type=str, default=None, help="Path to BC warm-start actor checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    train(args)
