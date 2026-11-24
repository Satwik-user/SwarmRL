"""
Single Drone Model Inspector in MuJoCo.
Showcases a single disaster-response drone with studio lighting, landing helipad,
active rotor rotation, turntable camera orbit, and interactive model controls.
"""

import sys
import os
import time
import math
import argparse
from typing import Optional, Dict, Any
import numpy as np
import mujoco
import mujoco.viewer

try:
    from .gpu_utils import (
        setup_high_performance_gpu,
        enable_high_resolution_timer,
        disable_high_resolution_timer,
    )
except ImportError:
    from gpu_utils import (
        setup_high_performance_gpu,
        enable_high_resolution_timer,
        disable_high_resolution_timer,
    )

# Detect Windows and initialize VT100 ANSI processing
IS_WINDOWS = sys.platform == "win32"
if IS_WINDOWS:
    import ctypes
    os.system("")  # Enables ANSI escape sequences in Windows Command Prompt / PowerShell

    def is_key_pressed(vk_code: int) -> bool:
        """Checks if a Windows Virtual-Key is currently held down."""
        try:
            return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)
        except Exception:
            return False
else:
    def is_key_pressed(vk_code: int) -> bool:
        return False

# Virtual Key Codes (Windows)
VK_SPACE = 0x20
VK_ESCAPE = 0x1B
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_A = 0x41
VK_C = 0x43
VK_D = 0x44
VK_E = 0x45
VK_H = 0x48
VK_L = 0x4C
VK_Q = 0x51
VK_R = 0x52
VK_S = 0x53
VK_V = 0x56
VK_W = 0x57

# Minimal Bare-Bones Prototype XML (Exact 9-geom model from scene.xml)
BARE_PROTOTYPE_XML = """<mujoco model="prototype_drone_inspection">
  <compiler angle="degree" coordinate="local"/>
  <option gravity="0 0 0" timestep="0.01"/>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.55 0.75" rgb2="0.12 0.18 0.25" width="512" height="512"/>
    <texture name="grid_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.22 0.25 0.28" rgb2="0.16 0.18 0.20"/>
    <material name="grid_mat" texture="grid_tex" texrepeat="20 20" reflectance="0.08" shininess="0.1"/>
    <material name="drone_chassis" rgba="0.12 0.12 0.14 1.0" specular="0.4" shininess="0.3"/>
    <material name="drone_arm" rgba="0.28 0.30 0.33 1.0" specular="0.3"/>
    <material name="rotor_front" rgba="0.95 0.25 0.15 0.9" specular="0.5"/>
    <material name="rotor_rear" rgba="0.15 0.65 0.95 0.9" specular="0.5"/>
    <material name="drone_indicator" rgba="0.2 0.9 0.3 1.0"/>
    <material name="comm_sphere_mat" rgba="0.2 0.55 1.0 0.12"/>
  </asset>
  <worldbody>
    <light directional="true" pos="10 10 20" dir="-0.5 -0.5 -1.0" diffuse="0.9 0.9 0.9"/>
    <geom name="ground" type="plane" size="15 15 1" material="grid_mat"/>
    <body name="D01" mocap="true" pos="0 0 1.2">
      <geom name="D01_body" type="cylinder" size="0.9 0.25" material="drone_chassis"/>
      <geom name="D01_dome" type="sphere" pos="0 0 0.15" size="0.45" material="drone_indicator"/>
      <geom name="D01_arm1" type="capsule" fromto="-1.5 -1.5 0 1.5 1.5 0" size="0.1" material="drone_arm"/>
      <geom name="D01_arm2" type="capsule" fromto="-1.5 1.5 0 1.5 -1.5 0" size="0.1" material="drone_arm"/>
      <geom name="D01_rotor_fl" type="cylinder" pos="1.5 1.5 0.1" size="0.65 0.03" material="rotor_front"/>
      <geom name="D01_rotor_fr" type="cylinder" pos="1.5 -1.5 0.1" size="0.65 0.03" material="rotor_front"/>
      <geom name="D01_rotor_bl" type="cylinder" pos="-1.5 1.5 0.1" size="0.65 0.03" material="rotor_rear"/>
      <geom name="D01_rotor_br" type="cylinder" pos="-1.5 -1.5 0.1" size="0.65 0.03" material="rotor_rear"/>
      <geom name="D01_comm_sphere" type="sphere" size="10.0" material="comm_sphere_mat" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"""


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


class SingleDroneViewer:
    """Manages 3D inspection, camera orbit, animations, and telemetry for a single drone in MuJoCo."""

    def __init__(
        self,
        scene_xml_path: str = "drone_scene.xml",
        use_bare_model: bool = False,
        auto_orbit: bool = True,
        spinning_rotors: bool = True,
    ):
        if not os.path.exists(scene_xml_path):
            local_candidate = os.path.join(os.path.dirname(__file__), scene_xml_path)
            if os.path.exists(local_candidate):
                scene_xml_path = local_candidate
        self.scene_path = scene_xml_path
        self.use_bare_model = use_bare_model
        self.auto_orbit = auto_orbit
        self.orbit_speed = 18.0  # Degrees per second around drone
        self.hover_enabled = True
        self.landed = False

        # Rotor states: 0 = Off, 1 = Idle (600 RPM), 2 = Flight (2400 RPM)
        self.rotor_state = 2 if spinning_rotors else 0
        self.rotor_speeds = [0.0, 35.0, 110.0]  # Rad/s in simulation
        self.rotor_angle = 0.0

        # Visibility toggles
        self.show_collision_bubble = True
        self.show_comm_sphere = False

        # Load MuJoCo model
        if self.use_bare_model or not os.path.exists(scene_xml_path):
            print("[Viewer] Loading minimal prototype drone model...")
            self.model = mujoco.MjModel.from_xml_string(BARE_PROTOTYPE_XML)
            self.is_detailed = False
        else:
            print(f"[Viewer] Loading detailed drone model from '{scene_xml_path}'...")
            self.model = mujoco.MjModel.from_xml_path(scene_xml_path)
            self.is_detailed = True

        self.data = mujoco.MjData(self.model)

        # Discrete GPU acceleration & shadow optimization
        self.gpu_info = setup_high_performance_gpu()
        if hasattr(self.model, "vis") and hasattr(self.model.vis, "quality"):
            self.model.vis.quality.shadowsize = min(self.model.vis.quality.shadowsize, 2048)

        # Inspect joint addresses for rotors if detailed model
        self.prop_joints = []
        for name in ["prop_fl_joint", "prop_fr_joint", "prop_bl_joint", "prop_br_joint"]:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id != -1:
                qpos_adr = self.model.jnt_qposadr[j_id]
                self.prop_joints.append(qpos_adr)

        # Drone position and altitude
        self.base_pos = np.array([0.0, 0.0, 1.20], dtype=np.float64)
        self.drone_pos = self.base_pos.copy()
        self.drone_yaw = 0.0  # Degrees

        # Material IDs for visual toggles
        self.mat_coll_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "collision_bubble")
        self.mat_comm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "comm_sphere_mat")

        # Key toggle debouncers
        self.prev_keys: Dict[str, bool] = {}
        self.key_actions: Dict[str, bool] = {
            "orbit": False,
            "rotors": False,
            "collision": False,
            "comm": False,
            "hover": False,
            "land": False,
            "reset": False,
        }

    def key_callback(self, keycode: int) -> None:
        """Callback invoked by MuJoCo GLFW window when a key is pressed."""
        try:
            char = chr(keycode).upper()
        except (ValueError, OverflowError):
            char = ""

        if char == " " or keycode == 32:
            self.key_actions["orbit"] = True
        elif char == "S":
            self.key_actions["rotors"] = True
        elif char == "C":
            self.key_actions["collision"] = True
        elif char == "V":
            self.key_actions["comm"] = True
        elif char == "H":
            self.key_actions["hover"] = True
        elif char == "L":
            self.key_actions["land"] = True
        elif char == "R":
            self.key_actions["reset"] = True

    def _poll_toggle_keys(self) -> None:
        """Checks Windows GetAsyncKeyState for toggle keys with single-trigger latching."""
        if not IS_WINDOWS:
            return

        key_mappings = [
            ("orbit", VK_SPACE),
            ("rotors", VK_S),
            ("collision", VK_C),
            ("comm", VK_V),
            ("hover", VK_H),
            ("land", VK_L),
            ("reset", VK_R),
        ]

        for action, vk in key_mappings:
            is_down = is_key_pressed(vk)
            was_down = self.prev_keys.get(action, False)
            if is_down and not was_down:
                self.key_actions[action] = True
            self.prev_keys[action] = is_down

    def _handle_actions(self, viewer) -> None:
        """Executes toggles queued by GLFW callback or Windows key polling."""
        if self.key_actions["orbit"]:
            self.auto_orbit = not self.auto_orbit
            self.key_actions["orbit"] = False

        if self.key_actions["rotors"]:
            self.rotor_state = (self.rotor_state + 1) % 3
            self.key_actions["rotors"] = False

        if self.key_actions["collision"]:
            self.show_collision_bubble = not self.show_collision_bubble
            if self.mat_coll_id != -1:
                self.model.mat_rgba[self.mat_coll_id][3] = 0.12 if self.show_collision_bubble else 0.0
            self.key_actions["collision"] = False

        if self.key_actions["comm"]:
            self.show_comm_sphere = not self.show_comm_sphere
            if self.mat_comm_id != -1:
                self.model.mat_rgba[self.mat_comm_id][3] = 0.08 if self.show_comm_sphere else 0.0
            self.key_actions["comm"] = False

        if self.key_actions["hover"]:
            self.hover_enabled = not self.hover_enabled
            self.key_actions["hover"] = False

        if self.key_actions["land"]:
            self.landed = not self.landed
            if self.landed:
                self.base_pos[2] = 0.70  # Resting firmly on landing skids
                self.rotor_state = 0     # Power down rotors
            else:
                self.base_pos[2] = 1.20  # Lift off to hover altitude
                self.rotor_state = 2     # Spin up rotors
            self.key_actions["land"] = False

        if self.key_actions["reset"]:
            self.base_pos[:] = [0.0, 0.0, 1.20]
            self.drone_yaw = 0.0
            self.landed = False
            self.rotor_state = 2
            self.auto_orbit = True
            viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
            viewer.cam.distance = 4.8
            viewer.cam.elevation = -16.0
            viewer.cam.azimuth = 45.0
            self.key_actions["reset"] = False

    def _apply_movement(self, dt: float) -> None:
        """Applies manual WASD nudge controls to reposition the drone in space."""
        if not IS_WINDOWS:
            return

        move_speed = 3.0 * dt
        # W/S (+/- X), A/D (+/- Y), E/Q (+/- Z)
        if is_key_pressed(VK_W):
            self.base_pos[0] += move_speed
        if is_key_pressed(VK_S):
            self.base_pos[0] -= move_speed
        if is_key_pressed(VK_A):
            self.base_pos[1] += move_speed
        if is_key_pressed(VK_D):
            self.base_pos[1] -= move_speed
        if is_key_pressed(VK_E) or is_key_pressed(VK_UP):
            self.base_pos[2] = min(10.0, self.base_pos[2] + move_speed)
        if is_key_pressed(VK_Q) or is_key_pressed(VK_DOWN):
            self.base_pos[2] = max(0.65, self.base_pos[2] - move_speed)
        if is_key_pressed(VK_LEFT):
            self.drone_yaw += 45.0 * dt
        if is_key_pressed(VK_RIGHT):
            self.drone_yaw -= 45.0 * dt

    def print_hud(self, viewer) -> None:
        """Renders clear, formatted inspection telemetry to the terminal."""
        rpm_labels = ["OFF (0 RPM)", "IDLE (600 RPM)", "FLIGHT (2,400 RPM)"]
        rpm_text = rpm_labels[self.rotor_state]
        orbit_text = f"ACTIVE ({viewer.cam.azimuth:.1f}°)" if self.auto_orbit else "PAUSED"
        flight_state = "LANDED (On Helipad)" if self.landed else ("HOVERING" if self.hover_enabled else "STATIONARY AIR")

        hud = [
            "========================================================================",
            "        MUJOCO 3D DRONE MODEL INSPECTION & SHOWROOM VIEWER",
            "========================================================================",
            f"Drone Unit       : D01 Heavy-Lift Disaster-Response UAV",
            f"Configuration    : Quadcopter X-Frame with 4.24m Diagonal Arm Span",
            f"Chassis Geometry : Diameter 1.80 m | Height 0.44 m (Carbon Composite)",
            f"Rotors           : 4x 1.30 m Diameter Counter-Rotating Propellers",
            f"Sensors          : Top GNSS Dome | Nose 4K Optical + Thermal Gimbal",
            f"Safety Shell     : 1.80 m Bounding Collision Radius (Vol: 24.4 m³)",
            f"Comm Range       : 100.0 m Omnidirectional Swarm Radius",
            "------------------------------------------------------------------------",
            "LIVE STATUS:",
            f"  Position (X,Y,Z): ({self.drone_pos[0]:.2f}, {self.drone_pos[1]:.2f}, {self.drone_pos[2]:.2f}) m",
            f"  Heading (Yaw)   : {self.drone_yaw:.1f}°",
            f"  Flight Mode     : {flight_state}",
            f"  Rotor Speed     : {rpm_text}",
            f"  Camera Turntable: {orbit_text} | Distance: {viewer.cam.distance:.1f} m",
            f"  Collision Shell : {'VISIBLE (1.8m)' if self.show_collision_bubble else 'HIDDEN'} [Key C]",
            f"  Comm Bubble     : {'VISIBLE' if self.show_comm_sphere else 'HIDDEN'} [Key V]",
            "------------------------------------------------------------------------",
            "KEYBOARD CONTROLS:",
            "  SPACE         : Toggle Showroom Turntable Orbit (360° Camera Spin)",
            "  S             : Cycle Rotor Speed (Flight RPM -> Idle -> Off)",
            "  L             : Land / Takeoff Toggle (Touch down on skids / Take off)",
            "  H             : Toggle Gentle Hover Altitude Bobbing",
            "  C             : Toggle 1.8m Bounding Collision Shell",
            "  V             : Toggle Communication Radius Shell",
            "  W / S / A / D : Nudge Drone Forward / Back / Left / Right",
            "  E / Q or UP/DN: Nudge Drone Up / Down",
            "  LEFT / RIGHT  : Rotate Drone Heading (Yaw)",
            "  R             : Reset Camera to Default Inspection View",
            "  ESC           : Close Viewer",
            "========================================================================",
        ]

        padded_lines = [f"{line:<74}" for line in hud]
        sys.stdout.write("\033[H" + "\n".join(padded_lines) + "\n")
        sys.stdout.flush()

    def run(self) -> None:
        """Launches the interactive MuJoCo viewer and runs the inspection loop."""
        if not is_gui_available():
            print("\n" + "=" * 76)
            print("  [Digital Twin] Single Drone MuJoCo Model Inspector")
            print("=" * 76)
            print("  [NOTICE] 3D OpenGL display window requires an active desktop display context.")
            print("           Running on headless Linux / WSL without X11 server.")
            print("  [INFO]   Model verification passed: Disaster Response UAV successfully loaded!")
            print(f"           - Model Bodies: {self.model.nbody}")
            print(f"           - Geometric Primitives: {self.model.ngeom}")
            print(f"           - Degrees of Freedom: {self.model.nv}")
            print("  [INFO]   On a teammate's machine with GPU and display (e.g. Windows / Ubuntu),")
            print("           this command launches the interactive 60 FPS 3D showroom viewer.")
            print("=" * 76 + "\n")
            return

        os.system("cls" if os.name == "nt" else "clear")

        enable_high_resolution_timer()
        try:
            with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
                # Set camera to focused, beautiful studio angle on the drone
                viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
                viewer.cam.distance = 4.8
                viewer.cam.elevation = -16.0
                viewer.cam.azimuth = 45.0

                last_hud_time = 0.0
                hud_interval = 0.15
                start_time = time.time()
                last_time = time.time()

                while viewer.is_running():
                    now = time.time()
                    dt = min(now - last_time, 0.05)
                    last_time = now
                    sim_time = now - start_time

                    # 1. Process Key Inputs & Toggles
                    self._poll_toggle_keys()
                    self._handle_actions(viewer)
                    self._apply_movement(dt)

                    # 2. Camera Turntable Showcase Orbit
                    if self.auto_orbit:
                        viewer.cam.azimuth = (viewer.cam.azimuth + self.orbit_speed * dt) % 360.0

                    # 3. Drone Hover Altitude Animation
                    self.drone_pos = self.base_pos.copy()
                    if self.hover_enabled and not self.landed:
                        bob = 0.04 * math.sin(sim_time * 2.5)
                        self.drone_pos[2] += bob

                    # 4. Update Drone Joint Position in MuJoCo
                    if self.is_detailed and self.model.njnt > 0:
                        # Detailed model: 6-DOF freejoint at qpos[0:7]
                        self.data.qpos[0:3] = self.drone_pos
                        # Yaw orientation quaternion [cos(yaw/2), 0, 0, sin(yaw/2)]
                        rad = math.radians(self.drone_yaw) / 2.0
                        self.data.qpos[3:7] = [math.cos(rad), 0.0, 0.0, math.sin(rad)]

                        # Spin Rotors
                        rotor_speed = self.rotor_speeds[self.rotor_state]
                        if rotor_speed > 0.0:
                            self.rotor_angle = (self.rotor_angle + rotor_speed * dt) % (2.0 * math.pi)
                            # Counter-rotating pairs for quadcopter: FL/BR CW, FR/BL CCW
                            for idx, qpos_adr in enumerate(self.prop_joints):
                                direction = 1.0 if idx in (0, 3) else -1.0
                                self.data.qpos[qpos_adr] = (direction * self.rotor_angle) % (2.0 * math.pi)
                    elif self.model.nmocap > 0:
                        # Minimal prototype model: mocap body
                        self.data.mocap_pos[0] = self.drone_pos

                    # Fast forward kinematics step
                    mujoco.mj_kinematics(self.model, self.data)

                    # 5. Render Terminal HUD
                    if now - last_hud_time >= hud_interval:
                        self.print_hud(viewer)
                        last_hud_time = now

                    # 6. Synchronize MuJoCo Viewer
                    viewer.sync()

                    # Maintain ~60 FPS with smooth pacing
                    elapsed = time.time() - now
                    sleep_sec = 0.0166 - elapsed
                    if sleep_sec > 0.001:
                        time.sleep(sleep_sec)
        finally:
            disable_high_resolution_timer()

        print("\n[Shutdown] Drone viewer closed cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a single drone model in 3D MuJoCo viewer")
    parser.add_argument(
        "--model",
        type=str,
        choices=["detailed", "prototype"],
        default="detailed",
        help="Drone model variant: 'detailed' (disaster-response UAV) or 'prototype' (minimal 9-geom model from scene.xml)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="drone_scene.xml",
        help="Path to custom MJCF XML scene file",
    )
    parser.add_argument(
        "--no-orbit",
        action="store_true",
        help="Disable automatic camera turntable orbit at startup",
    )
    parser.add_argument(
        "--no-spin",
        action="store_true",
        help="Start with rotors powered down",
    )
    args = parser.parse_args()

    use_bare = (args.model == "prototype")
    viewer_app = SingleDroneViewer(
        scene_xml_path=args.scene,
        use_bare_model=use_bare,
        auto_orbit=not args.no_orbit,
        spinning_rotors=not args.no_spin,
    )
    viewer_app.run()
