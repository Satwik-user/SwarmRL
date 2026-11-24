"""
3D Flight Visualizer & Mission Playback Engine for SwarmRL.
Visualizes drone swarm trajectories, volumetric exploration, obstacles,
and survivor rescues in an interactive 3D matplotlib scene or saves as an animation.

Usage:
  python scripts/visualize_swarm_3d.py --episodes 1 --steps 100 --shield --save data/swarm_mission_3d.gif
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from swarm_env import SwarmEnv
from models.mappo_agent import MAPPOAgent


def create_box_polys(min_pt, size):
    """Generates 3D polygonal faces for an axis-aligned box obstacle."""
    x, y, z = min_pt
    dx, dy, dz = size
    corners = np.array([
        [x, y, z],
        [x + dx, y, z],
        [x + dx, y + dy, z],
        [x, y + dy, z],
        [x, y, z + dz],
        [x + dx, y, z + dz],
        [x + dx, y + dy, z + dz],
        [x, y + dy, z + dz],
    ])
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # Bottom
        [corners[4], corners[5], corners[6], corners[7]],  # Top
        [corners[0], corners[1], corners[5], corners[4]],  # Front
        [corners[2], corners[3], corners[7], corners[6]],  # Back
        [corners[0], corners[3], corners[7], corners[4]],  # Left
        [corners[1], corners[2], corners[6], corners[5]],  # Right
    ]
    return faces


def run_visualization(args):
    # Set backend if headless saving requested
    if args.save:
        matplotlib.use("Agg")

    print("==================================================================")
    print("         SwarmRL: 3D Multi-UAV Mission Visualizer                 ")
    print("==================================================================")
    print(f"Swarm Size: {args.agents} Drones")
    print(f"Survivors: {args.survivors} Target Victims")
    print(f"Safety Shield: {'ENABLED' if args.shield else 'DISABLED'}")
    print(f"Max Steps: {args.steps}")
    print("------------------------------------------------------------------\n")

    env = SwarmEnv(
        num_agents=args.agents,
        num_survivors=args.survivors,
        comm_radius=args.comm_radius,
        max_steps=args.steps,
        enable_thermal_los=True,
        enable_safety_shield=args.shield,
    )

    agent = MAPPOAgent(
        k_neighbors=env.k_neighbors,
        state_dim=env.state_dim,
        action_dim=3,
        device="cpu",
    )

    if os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)
        print(f"[STATUS] Loaded trained policy: {args.checkpoint}")
    else:
        print("[WARNING] Checkpoint not found. Running with baseline initial policy.")

    # Collect trajectory frames
    obs, _ = env.reset(seed=args.seed)
    drone_trails = {a: [] for a in env.agents}
    survivor_states = []
    telemetry_steps = []

    print("[STATUS] Simulating mission trajectory...")
    for step in range(args.steps):
        actions, _ = agent.select_actions(obs, deterministic=True)
        obs, rewards, terms, truncs, infos = env.step(actions)

        # Record drone positions
        for a in env.agents:
            drone_trails[a].append(env.agent_positions[a].copy())

        survivor_states.append(env.survivor_found.copy())
        telemetry_steps.append({
            "step": step + 1,
            "coverage": infos[env.agents[0]]["coverage_pct"],
            "rescued": infos[env.agents[0]]["survivors_rescued"],
            "collisions": env.episode_collisions,
            "near_misses": env.episode_near_misses,
        })

        if any(terms.values()) or any(truncs.values()):
            break

    total_frames = len(telemetry_steps)
    print(f"[STATUS] Simulation complete ({total_frames} frames). Rendering 3D animation...")

    # Setup 3D Figure
    fig = plt.figure(figsize=(12, 9), facecolor="#111317")
    ax = fig.add_subplot(111, projection="3d", facecolor="#111317")

    # Set world bounds
    ax.set_xlim(0, env.world_size[0])
    ax.set_ylim(0, env.world_size[1])
    ax.set_zlim(0, env.world_size[2])

    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Y (m)", color="white")
    ax.set_zlabel("Altitude Z (m)", color="white")
    ax.tick_params(colors="gray")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Render static 3D obstacle boxes
    for obs_data in env.obstacles:
        min_p = obs_data[:3]
        dims = obs_data[3:]
        faces = create_box_polys(min_p, dims)
        poly = Poly3DCollection(faces, facecolors="#3A404D", linewidths=0.5, edgecolors="#5A6275", alpha=0.6)
        ax.add_collection3d(poly)

    # Initial plot elements
    colors = plt.cm.tab10(np.linspace(0, 1, len(env.agents)))
    agent_plots = {}
    trail_plots = {}

    for i, a in enumerate(env.agents):
        init_p = drone_trails[a][0]
        agent_plots[a] = ax.plot([init_p[0]], [init_p[1]], [init_p[2]], marker="o", markersize=6, color=colors[i])[0]
        trail_plots[a] = ax.plot([], [], [], color=colors[i], alpha=0.4, linewidth=1.2)[0]

    # Survivor scatter markers
    surv_scatter = ax.scatter(
        env.survivor_positions[:, 0],
        env.survivor_positions[:, 1],
        env.survivor_positions[:, 2],
        c=["#FF3B30" if not s else "#34C759" for s in survivor_states[0]],
        marker="^",
        s=60,
        label="Survivors",
    )

    title_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, color="white", fontsize=11, family="monospace")

    def update_frame(frame_idx):
        t_data = telemetry_steps[frame_idx]
        title_text.set_text(
            f"Step: {t_data['step']:03d}/{total_frames} | "
            f"3D Coverage: {t_data['coverage']:4.1f}% | "
            f"Rescued: {t_data['rescued']}/{env.num_survivors} | "
            f"Collisions: {t_data['collisions']} | "
            f"Shield: {'ON' if args.shield else 'OFF'}"
        )

        # Update drone positions and trails
        for a in env.agents:
            history = np.array(drone_trails[a][: frame_idx + 1])
            cur_p = history[-1]
            agent_plots[a].set_data([cur_p[0]], [cur_p[1]])
            agent_plots[a].set_3d_properties([cur_p[2]])

            trail_plots[a].set_data(history[:, 0], history[:, 1])
            trail_plots[a].set_3d_properties(history[:, 2])

        # Update survivor colors (Red = Lost, Green = Rescued)
        current_found = survivor_states[frame_idx]
        surv_scatter._facecolor3d = np.array([
            matplotlib.colors.to_rgba("#34C759") if current_found[i] else matplotlib.colors.to_rgba("#FF3B30")
            for i in range(env.num_survivors)
        ])

        # Gentle camera rotation for turntable 3D effect
        ax.view_init(elev=25, azim=45 + (frame_idx * 0.4))
        return list(agent_plots.values()) + list(trail_plots.values()) + [title_text]

    ani = animation.FuncAnimation(fig, update_frame, frames=total_frames, interval=50, blit=False)

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        print(f"[STATUS] Saving animation to {args.save} (this may take a few seconds)...")
        ani.save(args.save, writer="pillow", fps=15)
        print(f"[SUCCESS] Mission animation saved to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwarmRL 3D Flight Visualizer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_policy.pt")
    parser.add_argument("--agents", type=int, default=6)
    parser.add_argument("--survivors", type=int, default=4)
    parser.add_argument("--comm-radius", type=float, default=50.0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shield", action="store_true", help="Enable deterministic safety shield")
    parser.add_argument("--save", type=str, default=None, help="Output GIF path (e.g. data/swarm_mission_3d.gif)")

    args = parser.parse_args()
    run_visualization(args)
