"""
Multi-Drone Scene Generator & MJCF XML Injector for SwarmRL.
Dynamically injects N disaster-response drones into a MuJoCo scene XML
with distinctive color beacons, high-fidelity geometry, active rotor discs,
and mocap control handles for seamless real-time flight synchronization.
"""

import os
import re
import math
from typing import List, Tuple, Optional

# Distinct high-contrast palette for multi-drone swarm visual identification
DRONE_PALETTES = [
    {"name": "cyan",    "dome_rgba": "0.10 0.85 1.00 1.0", "comm_rgba": "0.10 0.85 1.00 0.08", "accent": "0.10 0.85 1.00 1.0"},
    {"name": "amber",   "dome_rgba": "1.00 0.55 0.10 1.0", "comm_rgba": "1.00 0.55 0.10 0.08", "accent": "1.00 0.55 0.10 1.0"},
    {"name": "lime",    "dome_rgba": "0.20 0.95 0.35 1.0", "comm_rgba": "0.20 0.95 0.35 0.08", "accent": "0.20 0.95 0.35 1.0"},
    {"name": "magenta", "dome_rgba": "0.95 0.20 0.85 1.0", "comm_rgba": "0.95 0.20 0.85 0.08", "accent": "0.95 0.20 0.85 1.0"},
    {"name": "gold",    "dome_rgba": "1.00 0.88 0.15 1.0", "comm_rgba": "1.00 0.88 0.15 0.08", "accent": "1.00 0.88 0.15 1.0"},
    {"name": "crimson", "dome_rgba": "1.00 0.25 0.25 1.0", "comm_rgba": "1.00 0.25 0.25 0.08", "accent": "1.00 0.25 0.25 1.0"},
    {"name": "violet",  "dome_rgba": "0.65 0.30 1.00 1.0", "comm_rgba": "0.65 0.30 1.00 0.08", "accent": "0.65 0.30 1.00 0.08"},
    {"name": "teal",    "dome_rgba": "0.15 0.95 0.85 1.0", "comm_rgba": "0.15 0.95 0.85 0.08", "accent": "0.15 0.95 0.85 1.0"},
    {"name": "orange",  "dome_rgba": "1.00 0.40 0.00 1.0", "comm_rgba": "1.00 0.40 0.00 0.08", "accent": "1.00 0.40 0.00 1.0"},
    {"name": "white",   "dome_rgba": "0.95 0.95 0.95 1.0", "comm_rgba": "0.95 0.95 0.95 0.08", "accent": "0.95 0.95 0.95 1.0"},
]


def generate_drone_body_xml(
    drone_idx: int,
    init_x: float,
    init_y: float,
    init_z: float = 3.0,
    drone_name: Optional[str] = None,
) -> str:
    """
    Generates a high-fidelity disaster response quadcopter body MJCF element.
    Uses mocap='true' for zero-drift real-time trajectory streaming from MAPPO / SwarmEnv.
    """
    name = drone_name or f"drone_{drone_idx}"
    p_idx = drone_idx % len(DRONE_PALETTES)
    pal = DRONE_PALETTES[p_idx]

    xml = f"""
    <!-- Multi-Agent Drone: {name} (Color Theme: {pal['name'].capitalize()}) -->
    <body name="{name}" mocap="true" pos="{init_x:.2f} {init_y:.2f} {init_z:.2f}">
      <!-- Central Chassis Core (Carbon composite, 0.9m radius, 0.22m height) -->
      <geom name="{name}_chassis" type="cylinder" size="0.90 0.22" material="drone_chassis" contype="0" conaffinity="1"/>
      <geom name="{name}_top_bezel" type="cylinder" pos="0 0 0.23" size="0.65 0.02" rgba="{pal['accent']}" contype="0" conaffinity="0"/>

      <!-- Distinct Color GNSS / Status Beacon Dome -->
      <geom name="{name}_dome" type="sphere" pos="0 0 0.25" size="0.45" rgba="{pal['dome_rgba']}" contype="0" conaffinity="0"/>

      <!-- Forward Search & Rescue Sensor Lens (+X forward) -->
      <geom name="{name}_sensor_mount" type="cylinder" pos="0.92 0 -0.08" size="0.10 0.08" material="drone_arm" contype="0" conaffinity="0"/>
      <geom name="{name}_lens" type="sphere" pos="1.02 0 -0.10" size="0.14" rgba="0.1 0.7 1.0 1.0" contype="0" conaffinity="0"/>

      <!-- Cross Boom Arms (Span 4.24m Diagonal) -->
      <geom name="{name}_arm1" type="capsule" fromto="-1.5 -1.5 0 1.5 1.5 0" size="0.10" material="drone_arm" contype="0" conaffinity="1"/>
      <geom name="{name}_arm2" type="capsule" fromto="-1.5 1.5 0 1.5 -1.5 0" size="0.10" material="drone_arm" contype="0" conaffinity="1"/>

      <!-- Rotor Motors and Spinning Rotor Discs -->
      <geom name="{name}_rotor_fl" type="cylinder" pos="1.5 1.5 0.10" size="0.65 0.02" material="rotor_front" contype="0" conaffinity="0"/>
      <geom name="{name}_rotor_fr" type="cylinder" pos="1.5 -1.5 0.10" size="0.65 0.02" material="rotor_front" contype="0" conaffinity="0"/>
      <geom name="{name}_rotor_bl" type="cylinder" pos="-1.5 1.5 0.10" size="0.65 0.02" material="rotor_rear" contype="0" conaffinity="0"/>
      <geom name="{name}_rotor_br" type="cylinder" pos="-1.5 -1.5 0.10" size="0.65 0.02" material="rotor_rear" contype="0" conaffinity="0"/>

      <!-- Landing Skids -->
      <geom name="{name}_skid_l" type="capsule" fromto="-1.1 0.75 -0.55 1.1 0.75 -0.55" size="0.04" rgba="0.25 0.27 0.30 1.0" contype="0" conaffinity="0"/>
      <geom name="{name}_skid_r" type="capsule" fromto="-1.1 -0.75 -0.55 1.1 -0.75 -0.55" size="0.04" rgba="0.25 0.27 0.30 1.0" contype="0" conaffinity="0"/>
      <geom name="{name}_strut_fl" type="capsule" fromto="0.6 0.6 -0.22 0.75 0.75 -0.55" size="0.03" rgba="0.25 0.27 0.30 1.0" contype="0" conaffinity="0"/>
      <geom name="{name}_strut_fr" type="capsule" fromto="0.6 -0.6 -0.22 0.75 -0.75 -0.55" size="0.03" rgba="0.25 0.27 0.30 1.0" contype="0" conaffinity="0"/>
      <geom name="{name}_strut_bl" type="capsule" fromto="-0.6 0.6 -0.22 -0.75 0.75 -0.55" size="0.03" rgba="0.25 0.27 0.30 1.0" contype="0" conaffinity="0"/>
      <geom name="{name}_strut_br" type="capsule" fromto="-0.6 -0.6 -0.22 -0.75 -0.75 -0.55" size="0.03" rgba="0.25 0.27 0.30 1.0" contype="0" conaffinity="0"/>

      <!-- Dynamic Mesh Network Translucent Communication Bubble (Visual RF field) -->
      <geom name="{name}_comm_sphere" type="sphere" size="12.0" rgba="{pal['comm_rgba']}" contype="0" conaffinity="0"/>
    </body>"""
    return xml


def generate_compact_disaster_city(num_drones: int = 6) -> str:
    """
    Generates a lightweight, self-contained MuJoCo disaster city MJCF XML
    with an active search-and-rescue mission area, collapsed structures, fires, and victims.
    Loads instantaneously (<100ms) on any system without requiring 900KB disk models.
    """
    drone_xmls = []
    # Place drones in circular staging formation around origin
    formation_radius = 8.0 + (num_drones * 1.5)
    for i in range(num_drones):
        angle = 2.0 * math.pi * i / num_drones
        x = formation_radius * math.cos(angle)
        y = formation_radius * math.sin(angle)
        drone_xmls.append(generate_drone_body_xml(i, x, y, init_z=2.5))

    drones_block = "\n".join(drone_xmls)

    xml = f"""<mujoco model="swarmrl_disaster_city_compact">
  <compiler angle="degree" coordinate="local"/>
  <option gravity="0 0 0" timestep="0.0166667"/>
  <size memory="32M"/>

  <default>
    <geom contype="1" conaffinity="0"/>
  </default>

  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.45 0.45 0.48" diffuse="0.80 0.80 0.80" specular="0.2 0.2 0.2"/>
    <map znear="0.1" zfar="5000.0"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.55 0.75" rgb2="0.12 0.18 0.25" width="512" height="512"/>
    <texture name="floor_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.22 0.24 0.26" rgb2="0.16 0.18 0.20"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="50 50" reflectance="0.06" shininess="0.1"/>

    <!-- Materials -->
    <material name="drone_chassis" rgba="0.12 0.13 0.15 1.0" specular="0.5" shininess="0.4"/>
    <material name="drone_arm" rgba="0.28 0.30 0.33 1.0" specular="0.4"/>
    <material name="rotor_front" rgba="0.95 0.25 0.15 0.85" specular="0.5"/>
    <material name="rotor_rear" rgba="0.15 0.65 0.95 0.85" specular="0.5"/>

    <material name="building_glass" rgba="0.22 0.38 0.52 1.0" specular="0.7" shininess="0.6"/>
    <material name="building_concrete" rgba="0.48 0.49 0.52 1.0" specular="0.2"/>
    <material name="building_brick" rgba="0.55 0.35 0.28 1.0" specular="0.15"/>
    <material name="building_damaged" rgba="0.32 0.30 0.28 1.0" specular="0.1"/>
    <material name="rubble_mat" rgba="0.45 0.42 0.38 1.0" specular="0.1"/>
    <material name="fire_mat" rgba="1.00 0.35 0.05 0.85" emission="0.9"/>
  </asset>

  <worldbody>
    <!-- Sunlight and fill lighting -->
    <light directional="true" pos="150 150 300" dir="-0.4 -0.4 -0.9" diffuse="0.9 0.9 0.9" specular="0.4 0.4 0.4"/>
    <light directional="true" pos="-100 -100 200" dir="0.3 0.3 -0.7" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>

    <!-- Ground Plane (600m x 600m) -->
    <geom name="ground" type="plane" size="300 300 1" material="floor_mat"/>

    <!-- Staging Takeoff Plaza Center Marker -->
    <geom name="takeoff_plaza" type="cylinder" pos="0 0 0.05" size="25.0 0.02" rgba="0.2 0.8 0.4 0.20" contype="0" conaffinity="0"/>
    <geom name="origin_beacon" type="sphere" pos="0 0 0.3" size="0.6" rgba="1 1 1 1" contype="0" conaffinity="0"/>

    <!-- ==================== SWARM DRONES ({num_drones}) ==================== -->
{drones_block}

    <!-- ==================== DISASTER BUILDINGS & CITY BLOCKS ==================== -->
    <body name="Tower_Alpha" pos="45 35 25"><geom name="Tower_Alpha_geom" type="box" size="14 12 25" material="building_glass"/></body>
    <body name="Tower_Beta" pos="-50 40 30"><geom name="Tower_Beta_geom" type="box" size="16 14 30" material="building_concrete"/></body>
    <body name="Tower_Gamma" pos="55 -45 20"><geom name="Tower_Gamma_geom" type="box" size="12 15 20" material="building_brick"/></body>
    <body name="Tower_Delta" pos="-45 -50 28"><geom name="Tower_Delta_geom" type="box" size="15 15 28" material="building_glass"/></body>
    <body name="Building_North1" pos="0 65 18"><geom name="Building_North1_geom" type="box" size="18 10 18" material="building_concrete"/></body>
    <body name="Building_South1" pos="0 -65 22"><geom name="Building_South1_geom" type="box" size="18 10 22" material="building_brick"/></body>
    <body name="Building_East1" pos="80 0 16"><geom name="Building_East1_geom" type="box" size="12 20 16" material="building_concrete"/></body>
    <body name="Building_West1" pos="-80 0 16"><geom name="Building_West1_geom" type="box" size="12 20 16" material="building_damaged"/></body>

    <!-- Outer Skyscrapers -->
    <body name="Skyscraper_NE" pos="110 90 45"><geom name="Skyscraper_NE_geom" type="box" size="22 20 45" material="building_glass"/></body>
    <body name="Skyscraper_NW" pos="-110 95 40"><geom name="Skyscraper_NW_geom" type="box" size="20 22 40" material="building_concrete"/></body>
    <body name="Skyscraper_SE" pos="105 -95 38"><geom name="Skyscraper_SE_geom" type="box" size="20 20 38" material="building_brick"/></body>
    <body name="Skyscraper_SW" pos="-105 -100 42"><geom name="Skyscraper_SW_geom" type="box" size="24 18 42" material="building_glass"/></body>

    <!-- Collapsed Disaster Ruin Zone -->
    <body name="Rubble_Main" pos="30 -15 3.5" euler="15 10 -25"><geom name="Rubble_Main_geom" type="box" size="12 8 3.5" material="rubble_mat"/></body>
    <body name="Rubble_West" pos="-25 20 2.5" euler="-10 20 15"><geom name="Rubble_West_geom" type="box" size="8 10 2.5" material="rubble_mat"/></body>
    <body name="Rubble_Pile1" pos="35 5 1.5"><geom name="Rubble_Pile1_geom" type="sphere" size="3.5" material="rubble_mat"/></body>
    <body name="Rubble_Pile2" pos="-30 -20 1.8"><geom name="Rubble_Pile2_geom" type="sphere" size="4.0" material="rubble_mat"/></body>

    <!-- Active Fire Zones -->
    <body name="Fire_Zone_1" pos="32 -12 1.0"><geom name="Fire_Flame_1" type="sphere" size="3.5" material="fire_mat" contype="0" conaffinity="0"/></body>
    <body name="Fire_Zone_2" pos="-28 22 1.0"><geom name="Fire_Flame_2" type="sphere" size="3.0" material="fire_mat" contype="0" conaffinity="0"/></body>

    <!-- Target Victims (Survivors awaiting rescue) -->
    <body name="Victim_01" pos="35 -14 0.5" euler="0 90 0"><geom name="Victim_01_torso" type="capsule" fromto="0 0 0 0 0 0.22" size="0.18" rgba="0.85 0.65 0.45 1" contype="0" conaffinity="0"/><geom name="Victim_01_head" type="sphere" pos="0 0 0.35" size="0.15" rgba="0.95 0.75 0.55 1" contype="0" conaffinity="0"/></body>
    <body name="Victim_02" pos="-25 24 0.5" euler="0 90 0"><geom name="Victim_02_torso" type="capsule" fromto="0 0 0 0 0 0.22" size="0.18" rgba="0.85 0.65 0.45 1" contype="0" conaffinity="0"/><geom name="Victim_02_head" type="sphere" pos="0 0 0.35" size="0.15" rgba="0.95 0.75 0.55 1" contype="0" conaffinity="0"/></body>
    <body name="Victim_03" pos="60 10 0.9"><geom name="Victim_03_torso" type="capsule" fromto="0 0 0 0 0 1.2" size="0.18" rgba="0.85 0.65 0.45 1" contype="0" conaffinity="0"/><geom name="Victim_03_head" type="sphere" pos="0 0 1.35" size="0.15" rgba="0.95 0.75 0.55 1" contype="0" conaffinity="0"/></body>
    <body name="Victim_04" pos="-10 50 0.9"><geom name="Victim_04_torso" type="capsule" fromto="0 0 0 0 0 1.2" size="0.18" rgba="0.85 0.65 0.45 1" contype="0" conaffinity="0"/><geom name="Victim_04_head" type="sphere" pos="0 0 1.35" size="0.15" rgba="0.95 0.75 0.55 1" contype="0" conaffinity="0"/></body>
    <body name="Victim_05" pos="15 -55 0.5" euler="0 90 0"><geom name="Victim_05_torso" type="capsule" fromto="0 0 0 0 0 0.22" size="0.18" rgba="0.85 0.65 0.45 1" contype="0" conaffinity="0"/><geom name="Victim_05_head" type="sphere" pos="0 0 0.35" size="0.15" rgba="0.95 0.75 0.55 1" contype="0" conaffinity="0"/></body>
    <body name="Victim_06" pos="-60 -25 0.9"><geom name="Victim_06_torso" type="capsule" fromto="0 0 0 0 0 1.2" size="0.18" rgba="0.85 0.65 0.45 1" contype="0" conaffinity="0"/><geom name="Victim_06_head" type="sphere" pos="0 0 1.35" size="0.15" rgba="0.95 0.75 0.55 1" contype="0" conaffinity="0"/></body>
  </worldbody>
</mujoco>"""
    return xml


def build_swarm_scene_xml(
    num_drones: int = 6,
    base_scene_path: Optional[str] = None,
    output_path: Optional[str] = None,
    compact: bool = False,
) -> str:
    """
    Builds or patches an MJCF XML file with N drones.
    If base_scene_path exists and compact is False, patches the existing disaster city.
    Otherwise, generates the compact high-performance disaster environment.
    """
    if compact or not base_scene_path or not os.path.exists(base_scene_path):
        xml_content = generate_compact_disaster_city(num_drones=num_drones)
    else:
        with open(base_scene_path, "r", encoding="utf-8") as f:
            base_content = f.read()

        # Remove any existing placeholder drones D01, D02, D03, drone_...
        # Pattern matches <body name="D0[1-9]" ... </body> or <body name="drone_[0-9]+" ... </body>
        drone_pattern = re.compile(
            r'<body\s+name=["\'](D0[1-9]|drone_\d+)["\'][^>]*>.*?</body>',
            re.DOTALL
        )
        cleaned_content = drone_pattern.sub("", base_content)

        # Generate N drone elements in safe takeoff formation
        drone_xmls = []
        formation_radius = 10.0 + (num_drones * 1.5)
        for i in range(num_drones):
            angle = 2.0 * math.pi * i / num_drones
            x = formation_radius * math.cos(angle)
            y = formation_radius * math.sin(angle)
            drone_xmls.append(generate_drone_body_xml(i, x, y, init_z=3.0))

        drones_injection = "\n    <!-- ==================== SWARM DRONES (INJECTED) ==================== -->\n" + "\n".join(drone_xmls) + "\n"

        # Insert before the first procedural building, or after safe_zone_marker, or before </worldbody>
        if "<geom name=\"safe_zone_marker\"" in cleaned_content:
            marker = cleaned_content.find("/>", cleaned_content.find("<geom name=\"safe_zone_marker\""))
            if marker != -1:
                insert_pos = marker + 2
                xml_content = cleaned_content[:insert_pos] + drones_injection + cleaned_content[insert_pos:]
            else:
                xml_content = cleaned_content.replace("</worldbody>", drones_injection + "  </worldbody>")
        else:
            xml_content = cleaned_content.replace("</worldbody>", drones_injection + "  </worldbody>")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

    return xml_content
