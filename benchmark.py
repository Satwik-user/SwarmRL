import os
import argparse
import json
import numpy as np
import torch
from swarm_env import SwarmEnv
from models.mappo_agent import MAPPOAgent

def run_benchmark(args):
    print("==================================================================")
    print("         SwarmRL: Automated MARL Benchmark Harness                ")
    print("==================================================================")
    print(f"Model Checkpoint: {args.checkpoint}")
    print(f"Swarm Size: {args.agents} Drones in 3D Space")
    print(f"Communication Radius: {args.comm_radius}m")
    print(f"Safety Shield: {'ENABLED (Zero-Collision Projection)' if args.shield else 'DISABLED (Raw Baseline)'}")
    print(f"Benchmark Episodes: {args.episodes}")
    print("------------------------------------------------------------------\n")

    env = SwarmEnv(
        num_agents=args.agents,
        comm_radius=args.comm_radius,
        num_survivors=args.survivors,
        max_steps=args.max_steps,
        k_neighbors=args.k_neighbors,
        enable_safety_shield=args.shield,
    )

    agent = MAPPOAgent(
        k_neighbors=args.k_neighbors,
        state_dim=env.state_dim,
        action_dim=3,
        device="cpu",
    )

    if os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)
        print(f"[STATUS] Successfully loaded checkpoint from {args.checkpoint}\n")
    else:
        print(f"[WARNING] Checkpoint {args.checkpoint} not found. Running baseline with random initial weights.\n")

    episode_metrics = []

    for ep in range(1, args.episodes + 1):
        obs, infos = env.reset(seed=args.seed + ep if args.seed else None)
        done = False
        step = 0

        while not done:
            actions, _ = agent.select_actions(obs, deterministic=True)
            obs, rewards, terminations, truncations, infos = env.step(actions)
            step += 1
            done = any(terminations.values()) or any(truncations.values())

        first_info = list(infos.values())[0]
        ep_result = {
            "episode": ep,
            "steps": step,
            "coverage_pct": first_info["coverage_pct"],
            "survivors_rescued": first_info["survivors_rescued"],
            "collisions": first_info["collisions"],
            "near_misses": first_info["near_misses"],
        }
        episode_metrics.append(ep_result)

        print(
            f"Episode {ep:02d}/{args.episodes:02d} | "
            f"3D Coverage: {ep_result['coverage_pct']:5.1f}% | "
            f"Survivors: {ep_result['survivors_rescued']}/{args.survivors} | "
            f"Collisions: {ep_result['collisions']:3d} | "
            f"Near-Misses: {ep_result['near_misses']:3d} | "
            f"Steps: {step}"
        )

    coverages = [m["coverage_pct"] for m in episode_metrics]
    rescues = [m["survivors_rescued"] for m in episode_metrics]
    collisions = [m["collisions"] for m in episode_metrics]
    near_misses = [m["near_misses"] for m in episode_metrics]

    summary = {
        "mean_coverage_pct": float(np.mean(coverages)),
        "std_coverage_pct": float(np.std(coverages)),
        "mean_survivors_rescued": float(np.mean(rescues)),
        "mean_collisions": float(np.mean(collisions)),
        "mean_near_misses": float(np.mean(near_misses)),
        "total_episodes": args.episodes,
    }

    print("\n------------------------------------------------------------------")
    print("                     ACADEMIC BENCHMARK SUMMARY                   ")
    print("------------------------------------------------------------------")
    print(f"Mean 3D Coverage     : {summary['mean_coverage_pct']:6.2f}% +/- {summary['std_coverage_pct']:4.2f}%")
    print(f"Mean Survivors Found : {summary['mean_survivors_rescued']:6.2f} / {args.survivors}")
    print(f"Mean Collisions      : {summary['mean_collisions']:6.2f}")
    print(f"Mean Near-Misses     : {summary['mean_near_misses']:6.2f}")
    print("==================================================================")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"summary": summary, "episodes": episode_metrics}, f, indent=2)
        print(f"Exported metrics to {args.output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwarmRL Automated Benchmark Harness")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_policy.pt")
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument("--comm-radius", type=float, default=50.0)
    parser.add_argument("--survivors", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--k-neighbors", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--shield", action="store_true", help="Enable Arkendu's deterministic safety shield during benchmark")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output-json", type=str, default="benchmark_results.json")

    args = parser.parse_args()
    run_benchmark(args)
