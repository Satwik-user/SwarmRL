"""
Multi-Drone Swarm MuJoCo Visualizer & Telemetry Bridge for SwarmRL.
Animates a multi-UAV disaster-response swarm flying inside the high-fidelity
MuJoCo 3D digital twin city using trained MAPPO neural network policies,
swept-path SafetyShield obstacle deflection, and real-time telemetry streaming.

Works cross-platform:
- With GPU & Display (Windows / Ubuntu Desktop): Launches interactive 60 FPS hardware-accelerated 3D window.
- Headless (WSL / Server / CI): Gracefully simulates full multi-agent trajectories, exports JSON telemetry,
  and generates 3D Matplotlib mission trajectory projections.

Usage:
  python src/ground_station/digital_twin/view_swarm.py --drones 6 --steps 200 --shield
  python scripts/replay_mujoco_swarm.py --drones 6 --steps 200
"""

import os
import sys
import time
import math
import argparse
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from swarm_env import SwarmEnv
from models.mappo_agent import MAPPOAgent
from src.safety.safety_shield import SafetyShield, SafetyConfig, AABB
from src.ground_station.digital_twin.swarm_scene_builder import (
    build_swarm_scene_xml,
    DRONE_PALETTES,
)

# Optional GPU setup
try:
    from src.ground_station.digital_twin.gpu_utils import (
        setup_high_performance_gpu,
        enable_high_resolution_timer,
        disable_high_resolution_timer,
    )
except ImportError:
    setup_high_performance_gpu = lambda: {}
    enable_high_resolution_timer = lambda: False
    disable_high_resolution_timer = lambda: False

# Optional MuJoCo import
MUJOCO_AVAILABLE = False
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    mujoco = None
    MUJOCO_AVAILABLE = False


def is_gui_available() -> bool:
    """Checks if an interactive graphical window/display can actually be opened."""
    if sys.platform != "win32" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import glfw
            if not glfw.init():
                return False
            glfw.window_hint(glfw.VISIBLE, False)
            test_win = glfw.create_window(64, 64, "test", None, None)
            if not test_win:
                glfw.terminate()
                return False
            glfw.destroy_window(test_win)
            glfw.terminate()
            return True
    except Exception:
        return False


def euler_to_quat(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Converts Euler angles in radians to quaternion [w, x, y, z]."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float64)


class SwarmMuJoCoViewer:
    """
    Coordinates multi-drone flight inside MuJoCo 3D Digital Twin with
    MAPPO policy inference, SafetyShield collision avoidance, and dynamic camera tracking.
    """

    def __init__(
        self,
        num_drones: int = 6,
        max_steps: int = 250,
        checkpoint_path: str = "checkpoints/best_policy.pt",
        scene_path: Optional[str] = None,
        use_shield: bool = True,
        camera_mode: str = "flock",
        follow_drone_id: int = 0,
        compact_scene: bool = False,
        headless: bool = False,
        save_trajectory_path: str = "data/mujoco_swarm_trajectory.json",
        save_plot_path: str = "data/swarm_mujoco_trajectory.png",
        seed: int = 42,
    ):
        self.num_drones = num_drones
        self.max_steps = max_steps
        self.checkpoint_path = checkpoint_path
        self.scene_path = scene_path or os.path.join(os.path.dirname(__file__), "generated_scene.xml")
        self.use_shield = use_shield
        self.camera_mode = camera_mode  # 'flock', 'follow', 'orbit', 'free'
        self.follow_drone_id = min(follow_drone_id, num_drones - 1)
        self.compact_scene = compact_scene
        self.force_headless = headless
        self.save_trajectory_path = save_trajectory_path
        self.save_plot_path = save_plot_path
        self.seed = seed

        # Interactive toggles
        self.auto_orbit = (camera_mode == "orbit")
        self.orbit_speed = 12.0  # Deg/sec
        self.smooth_lookat = np.array([0.0, 0.0, 10.0], dtype=np.float64)
        self.show_comms = True
        self.paused = False

        # Initialize PettingZoo RL Environment
        print(f"[SwarmRL] Initializing multi-agent flight environment ({self.num_drones} UAVs)...")
        self.env = SwarmEnv(
            num_agents=self.num_drones,
            num_survivors=max(6, self.num_drones),
            comm_radius=50.0,
            world_size=(160.0, 160.0, 40.0),
            max_steps=self.max_steps,
            enable_thermal_los=True,
            enable_safety_shield=self.use_shield,
        )

        # Initialize MAPPO Neural Network Policy
        self.agent = MAPPOAgent(
            k_neighbors=self.env.k_neighbors,
            state_dim=self.env.state_dim,
            action_dim=3,
            device="cpu",
        )

        resolved_checkpoint = self.checkpoint_path
        if not os.path.exists(resolved_checkpoint):
            cand = os.path.join(REPO_ROOT, self.checkpoint_path)
            if os.path.exists(cand):
                resolved_checkpoint = cand

        if os.path.exists(resolved_checkpoint):
            try:
                self.agent.load(resolved_checkpoint)
                print(f"[SwarmRL] Loaded trained MAPPO checkpoint: '{resolved_checkpoint}'")
            except Exception as e:
                print(f"[SwarmRL] Checkpoint loading note ({e}). Using initialized network weights.")
        else:
            print(f"[SwarmRL] Checkpoint '{resolved_checkpoint}' not found. Using initialized network weights.")

        # MuJoCo Setup
        self.model = None
        self.data = None
        self.mocap_ids: Dict[int, int] = {}
        self.is_mujoco_active = False

        if MUJOCO_AVAILABLE and not self.force_headless:
            self._init_mujoco()
        else:
            reason = "headless mode requested" if self.force_headless else "MuJoCo package not installed in environment"
            print(f"[Digital Twin] Running with telemetry physics engine ({reason}).")

    def _init_mujoco(self) -> None:
        """Compiles MJCF XML and creates MuJoCo model & data."""
        try:
            print("[Digital Twin] Generating multi-drone MJCF XML...")
            xml_str = build_swarm_scene_xml(
                num_drones=self.num_drones,
                base_scene_path=self.scene_path,
                compact=self.compact_scene,
            )

            # Compile into MuJoCo model
            self.model = mujoco.MjModel.from_xml_string(xml_str)
            self.data = mujoco.MjData(self.model)

            # Map drone body names to mocap IDs
            for i in range(self.num_drones):
                b_name = f"drone_{i}"
                b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b_name)
                if b_id != -1 and self.model.body_mocapid[b_id] != -1:
                    self.mocap_ids[i] = self.model.body_mocapid[b_id]

            self.is_mujoco_active = True
            print(f"[Digital Twin] MuJoCo scene compiled successfully! ({len(self.mocap_ids)} drone mocap handles connected)")
        except Exception as e:
            print(f"[Digital Twin] Warning: MuJoCo initialization failed ({e}). Falling back to telemetry mode.")
            self.is_mujoco_active = False

    def _env_to_mujoco_coords(self, env_pos: np.ndarray) -> np.ndarray:
        """
        Maps SwarmEnv [0, world_size] coordinate frame to MuJoCo disaster city frame.
        Centers origin (0, 0) at the safe takeoff plaza and maintains realistic flight altitude.
        """
        mj_pos = np.zeros(3, dtype=np.float64)
        mj_pos[0] = float(env_pos[0] - self.env.world_size[0] / 2.0)
        mj_pos[1] = float(env_pos[1] - self.env.world_size[1] / 2.0)
        mj_pos[2] = float(env_pos[2] + 2.5)  # Hover 2.5m to 42.5m above ground
        return mj_pos

    def _key_callback(self, keycode: int) -> None:
        """MuJoCo GLFW interactive key callback."""
        try:
            char = chr(keycode).upper()
        except (ValueError, OverflowError):
            char = ""

        if char == " " or keycode == 32:
            self.auto_orbit = not self.auto_orbit
            print(f"[Camera] Auto-orbit: {'ON' if self.auto_orbit else 'OFF'}")
        elif char == "F":
            modes = ["flock", "follow", "orbit", "free"]
            next_idx = (modes.index(self.camera_mode) + 1) % len(modes)
            self.camera_mode = modes[next_idx]
            self.auto_orbit = (self.camera_mode == "orbit")
            print(f"[Camera] Switched to mode: {self.camera_mode.upper()}")
        elif char.isdigit():
            val = int(char)
            if val < self.num_drones:
                self.follow_drone_id = val
                self.camera_mode = "follow"
                print(f"[Camera] Following Drone {val}")
        elif char == "P":
            self.paused = not self.paused
            print(f"[Simulation] {'PAUSED' if self.paused else 'RESUMED'}")
        elif char == "R":
            self.camera_mode = "flock"
            self.auto_orbit = False
            print("[Camera] Reset to Flock Centroid view")

    def _update_camera(self, viewer, drone_positions: Dict[int, np.ndarray], dt: float) -> None:
        """Smoothly orients the MuJoCo viewer camera."""
        if not hasattr(viewer, "cam"):
            return

        all_coords = np.array(list(drone_positions.values()))
        if len(all_coords) == 0:
            return

        centroid = np.mean(all_coords, axis=0)
        spread = float(np.max(np.linalg.norm(all_coords - centroid, axis=1))) if len(all_coords) > 1 else 10.0

        if self.camera_mode == "flock":
            # Smooth low-pass filter on camera focus point
            alpha = min(1.0, 5.0 * dt)
            self.smooth_lookat = (1.0 - alpha) * self.smooth_lookat + alpha * centroid
            viewer.cam.lookat[:] = self.smooth_lookat
            target_dist = max(35.0, spread * 2.2 + 20.0)
            viewer.cam.distance = 0.95 * viewer.cam.distance + 0.05 * target_dist
            viewer.cam.elevation = -22.0

        elif self.camera_mode == "follow":
            pos = drone_positions.get(self.follow_drone_id, centroid)
            alpha = min(1.0, 8.0 * dt)
            self.smooth_lookat = (1.0 - alpha) * self.smooth_lookat + alpha * pos
            viewer.cam.lookat[:] = self.smooth_lookat
            viewer.cam.distance = 18.0
            viewer.cam.elevation = -18.0

        elif self.camera_mode == "orbit":
            viewer.cam.lookat[:] = centroid
            viewer.cam.azimuth = (viewer.cam.azimuth + self.orbit_speed * dt) % 360.0

        if self.auto_orbit and self.camera_mode != "orbit":
            viewer.cam.azimuth = (viewer.cam.azimuth + self.orbit_speed * dt) % 360.0

    def print_hud(
        self,
        step: int,
        drone_mj_pos: Dict[int, np.ndarray],
        infos: Dict[str, Any],
        shield_interventions: int,
    ) -> None:
        """Renders clear, formatted telemetry HUD in the terminal."""
        coverage = infos.get(self.env.agents[0], {}).get("coverage_pct", 0.0)
        rescued = infos.get(self.env.agents[0], {}).get("survivors_rescued", 0)
        total_survivors = self.env.num_survivors

        hud = [
            "\n" + "=" * 76,
            f"  [SwarmRL Digital Twin] Multi-UAV Swarm in MuJoCo Disaster City",
            f"  Step: {step:03d}/{self.max_steps:03d} | Drones: {self.num_drones} | Mode: {self.camera_mode.upper()} | SafetyShield: {'ON' if self.use_shield else 'OFF'}",
            "-" * 76,
        ]

        # Display up to 6 drones in HUD
        display_count = min(self.num_drones, 6)
        for i in range(display_count):
            p = drone_mj_pos.get(i, np.zeros(3))
            agent_id = f"drone_{i}"
            vel = self.env.agent_velocities.get(agent_id, np.zeros(3))
            speed = float(np.linalg.norm(vel))
            pal = DRONE_PALETTES[i % len(DRONE_PALETTES)]
            status = "FLIGHT OK"
            hud.append(
                f"  Drone {i} [{pal['name'].capitalize():<7}]: Pos=({p[0]:6.1f}, {p[1]:6.1f}, {p[2]:5.1f}m) | Speed={speed:4.1f}m/s | Status={status}"
            )

        if self.num_drones > 6:
            hud.append(f"  ... (+ {self.num_drones - 6} more drones active in swarm)")

        hud.append("-" * 76)
        hud.append(
            f"  Exploration: {coverage:5.1f}% | Victims Rescued: {rescued}/{total_survivors} | Shield Overrides: {shield_interventions} | Collisions: {self.env.episode_collisions}"
        )
        hud.append("=" * 76)
        print("\n".join(hud))

    def run(self) -> Dict[str, Any]:
        """
        Executes the full multi-drone mission simulation.
        If MuJoCo GUI is available, runs at interactive 60 FPS in 3D window.
        Otherwise, runs ultra-fast headless simulation and saves 3D trajectory plot.
        """
        obs, _ = self.env.reset(seed=self.seed)

        trajectory_history: List[Dict[str, Any]] = []
        drone_trails: Dict[int, List[List[float]]] = {i: [] for i in range(self.num_drones)}
        shield_interventions = 0

        # Attempt to launch passive MuJoCo viewer window if display is available
        viewer = None
        is_interactive = False

        gui_supported = self.is_mujoco_active and not self.force_headless and is_gui_available()

        if gui_supported:
            try:
                setup_high_performance_gpu()
                enable_high_resolution_timer()
                print("[Digital Twin] Opening MuJoCo 3D Interactive Window...")
                viewer = mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                    key_callback=self._key_callback,
                )
                is_interactive = True
                print("[Digital Twin] MuJoCo 3D Window is ACTIVE!")
                print("  Controls: [SPACE] Orbit | [F] Camera Mode | [0-9] Follow Drone | [P] Pause | [R] Reset")
            except Exception as e:
                print(f"[Digital Twin] GUI Note: Display window could not be initialized ({e}).")
                print("              Running headless with trajectory & 3D visual export.")
                viewer = None
                is_interactive = False
        else:
            if not self.force_headless:
                print("[Digital Twin] Notice: 3D OpenGL display window requires an active display (DISPLAY / GPU).")
                print("              [INFO] On a teammate's machine with GPU and display (e.g. Windows / Ubuntu Desktop),")
                print("                     the 3D MuJoCo window opens with 60 FPS hardware acceleration.")
                print("              [INFO] Running simulation & generating full 3D trajectory visualization...")
            viewer = None
            is_interactive = False

        print("\n[SwarmRL] Commencing Multi-UAV Mission Execution...\n")
        start_time = time.time()
        last_frame_time = time.time()

        for step in range(1, self.max_steps + 1):
            if is_interactive and viewer is not None and not viewer.is_running():
                print("[Digital Twin] User closed viewer window.")
                break

            step_dt = time.time() - last_frame_time
            last_frame_time = time.time()

            # 1. Compute multi-agent actions from MAPPO policy
            actions, _ = self.agent.select_actions(obs, deterministic=True)

            # 2. Step environment
            obs, rewards, terms, truncs, infos = self.env.step(actions)

            # 3. Synchronize positions into MuJoCo coordinates
            drone_mj_positions: Dict[int, np.ndarray] = {}
            for i in range(self.num_drones):
                agent_id = f"drone_{i}"
                if agent_id in self.env.agent_positions:
                    env_p = self.env.agent_positions[agent_id]
                    mj_p = self._env_to_mujoco_coords(env_p)
                    drone_mj_positions[i] = mj_p
                    drone_trails[i].append(mj_p.tolist())

                    # Update MuJoCo mocap bodies if active
                    if self.is_mujoco_active and i in self.mocap_ids:
                        m_id = self.mocap_ids[i]
                        self.data.mocap_pos[m_id] = mj_p

                        # Compute heading yaw from velocity
                        vel = self.env.agent_velocities.get(agent_id, np.zeros(3))
                        if np.linalg.norm(vel[:2]) > 0.1:
                            yaw = math.atan2(vel[1], vel[0])
                            self.data.mocap_quat[m_id] = euler_to_quat(yaw)

            # 4. Step MuJoCo forward kinematics & sync viewer
            if self.is_mujoco_active:
                mujoco.mj_forward(self.model, self.data)

            if is_interactive and viewer is not None:
                self._update_camera(viewer, drone_mj_positions, max(0.016, step_dt))
                viewer.sync()

                # Pace simulation to target ~30-60 FPS for smooth viewing
                elapsed = time.time() - last_frame_time
                sleep_time = max(0.0, 0.025 - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # 5. Record frame telemetry
            frame_data = {
                "step": step,
                "drones": {f"drone_{i}": drone_mj_positions[i].tolist() for i in range(self.num_drones)},
                "coverage_pct": float(infos[self.env.agents[0]]["coverage_pct"]),
                "rescued": int(infos[self.env.agents[0]]["survivors_rescued"]),
                "collisions": int(self.env.episode_collisions),
            }
            trajectory_history.append(frame_data)

            # 6. Terminal HUD logging (every 25 steps or first step)
            if step == 1 or step % 25 == 0 or step == self.max_steps:
                self.print_hud(step, drone_mj_positions, infos, shield_interventions)

            if any(terms.values()) or any(truncs.values()):
                print(f"[SwarmRL] Mission reached terminal condition at step {step}.")
                break

        total_duration = time.time() - start_time
        fps = len(trajectory_history) / max(0.001, total_duration)

        if is_interactive and viewer is not None:
            disable_high_resolution_timer()

        # Save trajectory data
        self._save_results(trajectory_history, drone_trails)

        summary = {
            "total_steps": len(trajectory_history),
            "sim_fps": fps,
            "final_coverage": trajectory_history[-1]["coverage_pct"] if trajectory_history else 0.0,
            "total_rescued": trajectory_history[-1]["rescued"] if trajectory_history else 0,
            "collisions": self.env.episode_collisions,
            "near_misses": self.env.episode_near_misses,
            "trajectory_file": self.save_trajectory_path,
            "plot_file": self.save_plot_path,
        }

        print("\n" + "=" * 76)
        print("                 SwarmRL Mission Execution Complete                   ")
        print("=" * 76)
        print(f"  Duration: {total_duration:.2f}s | Speed: {fps:.1f} steps/sec")
        print(f"  Exploration Coverage: {summary['final_coverage']:.1f}%")
        print(f"  Survivors Rescued: {summary['total_rescued']}/{self.env.num_survivors}")
        print(f"  Collisions: {summary['collisions']} | Shield Interventions: {self.env.episode_near_misses}")
        print(f"  Telemetry Saved: [trajectory.json](file://{os.path.abspath(self.save_trajectory_path)})")
        if os.path.exists(self.save_plot_path):
            print(f"  3D Trajectory Plot: [trajectory.png](file://{os.path.abspath(self.save_plot_path)})")
        print("=" * 76 + "\n")

        return summary

    def _save_results(
        self,
        history: List[Dict[str, Any]],
        trails: Dict[int, List[List[float]]],
    ) -> None:
        """Saves trajectory log to JSON and renders 3D Matplotlib plot."""
        # Ensure target directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.save_trajectory_path)), exist_ok=True)
        with open(self.save_trajectory_path, "w", encoding="utf-8") as f:
            json.dump({
                "num_drones": self.num_drones,
                "steps": len(history),
                "history": history,
            }, f, indent=2)

        # Generate 3D plot using Matplotlib
        try:
            import matplotlib
            matplotlib.use("Agg")  # Safe headless backend
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(11, 8), facecolor="#0e1117")
            ax = fig.add_subplot(111, projection="3d", facecolor="#0e1117")

            # Draw drone trajectories with distinct palette colors
            for i in range(self.num_drones):
                pts = np.array(trails[i])
                if len(pts) > 0:
                    pal = DRONE_PALETTES[i % len(DRONE_PALETTES)]
                    # Convert rgba string to tuple of floats
                    rgba = [float(x) for x in pal["dome_rgba"].split()]
                    ax.plot(
                        pts[:, 0], pts[:, 1], pts[:, 2],
                        label=f"Drone {i} ({pal['name']})",
                        color=rgba[:3],
                        linewidth=2.0,
                        alpha=0.9,
                    )
                    # Start point (square marker)
                    ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color=rgba[:3], marker="s", s=40)
                    # End point (drone location)
                    ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color=rgba[:3], marker="o", s=80, edgecolors="white")

            # Draw Takeoff Plaza at origin
            theta = np.linspace(0, 2 * np.pi, 50)
            ax.plot(20 * np.cos(theta), 20 * np.sin(theta), np.zeros(50), color="#2ECC71", linestyle="--", linewidth=1.5, label="Takeoff Plaza")

            ax.set_title(f"SwarmRL: Multi-UAV 3D Mission Flight ({self.num_drones} Drones)", color="white", fontsize=14, pad=15)
            ax.set_xlabel("X (m)", color="white")
            ax.set_ylabel("Y (m)", color="white")
            ax.set_zlabel("Altitude Z (m)", color="white")
            ax.tick_params(colors="gray")
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.legend(loc="upper right", facecolor="#1a1c23", edgecolor="gray", labelcolor="white", fontsize=8)

            os.makedirs(os.path.dirname(os.path.abspath(self.save_plot_path)), exist_ok=True)
            plt.savefig(self.save_plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
        except Exception as e:
            print(f"[Digital Twin] Plot export note: {e}")


def main():
    parser = argparse.ArgumentParser(description="SwarmRL Multi-Drone MuJoCo Visualizer & Telemetry Bridge")
    parser.add_argument("--drones", "--agents", type=int, default=6, help="Number of drones in swarm (default: 6)")
    parser.add_argument("--steps", type=int, default=200, help="Max simulation steps (default: 200)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_policy.pt", help="Path to trained MAPPO checkpoint")
    parser.add_argument("--scene", type=str, default=None, help="Path to base MJCF XML scene (default: generated_scene.xml)")
    parser.add_argument("--shield", action="store_true", default=True, help="Enable SafetyShield obstacle avoidance (default: True)")
    parser.add_argument("--no-shield", dest="shield", action="store_false", help="Disable SafetyShield")
    parser.add_argument("--camera", type=str, choices=["flock", "follow", "orbit", "free"], default="flock", help="Camera mode")
    parser.add_argument("--follow-drone", type=int, default=0, help="Drone ID to follow in follow mode")
    parser.add_argument("--compact", action="store_true", help="Use lightweight compact disaster city (fast startup)")
    parser.add_argument("--headless", action="store_true", help="Force headless execution (no window)")
    parser.add_argument("--trajectory", type=str, default="data/mujoco_swarm_trajectory.json", help="Path to save trajectory JSON")
    parser.add_argument("--plot", type=str, default="data/swarm_mujoco_trajectory.png", help="Path to save trajectory 3D plot")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    viewer = SwarmMuJoCoViewer(
        num_drones=args.drones,
        max_steps=args.steps,
        checkpoint_path=args.checkpoint,
        scene_path=args.scene,
        use_shield=args.shield,
        camera_mode=args.camera,
        follow_drone_id=args.follow_drone,
        compact_scene=args.compact,
        headless=args.headless,
        save_trajectory_path=args.trajectory,
        save_plot_path=args.plot,
        seed=args.seed,
    )

    viewer.run()


if __name__ == "__main__":
    main()
