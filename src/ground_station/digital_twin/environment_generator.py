"""
Procedural 3D Disaster Environment Generator for MuJoCo.
Generates a congested 10 km x 10 km disaster environment with enlarged buildings,
massive disaster rubble fields, giant trees, tall transmission towers, and tight street blocks.
"""

import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
try:
    from src.perception.entities import Fire, Human, ColdZone
except ImportError:
    try:
        from entities import Fire, Human, ColdZone
    except ImportError:
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        from src.perception.entities import Fire, Human, ColdZone


@dataclass
class GeneratedObstacle:
    """Metadata for an obstacle generated in the procedural environment."""
    name: str
    obs_type: str  # 'Building', 'Rubble', 'Tree', 'Pole'
    pos: np.ndarray  # Center position [x, y, z]
    shape: str  # 'box', 'cylinder', 'sphere', 'tree'
    size_params: np.ndarray  # Dimensions for geometric distance calculations
    heading_deg: float = 0.0


class ProceduralEnvironmentGenerator:
    """
    Procedurally creates a congested 10 km x 10 km MuJoCo disaster environment
    with enlarged structures, heavy disaster rubble fields, giant trees, and towers.
    """

    def __init__(
        self,
        env_size: float = 10000.0,
        num_buildings: int = 800,
        num_rubble_zones: int = 50,
        num_rubble_per_zone: int = 30,
        num_trees: int = 900,
        num_poles: int = 350,
        seed: int = 12345,
        safe_radius: float = 100.0,
        shadow_size: int = 2048,
        ambient_temp_c: float = 18.0,
        num_fire_zones: int = 30,
        num_cold_zones: int = 12,
        cold_zone_delta_c: Tuple[float, float] = (-10.0, -4.0),
        num_humans: int = 150,
    ):
        self.env_size = env_size
        self.half_size = env_size / 2.0  # 5,000 m
        self.num_buildings = num_buildings
        self.num_rubble_zones = num_rubble_zones
        self.num_rubble_per_zone = num_rubble_per_zone
        self.num_trees = num_trees
        self.num_poles = num_poles
        self.seed = seed
        self.safe_radius = safe_radius  # 200m x 200m safe box centered at (0, 0)
        self.shadow_size = shadow_size
        self.ambient_temp_c = ambient_temp_c
        self.num_fire_zones = num_fire_zones
        self.num_cold_zones = num_cold_zones
        self.cold_zone_delta_c = cold_zone_delta_c
        self.num_humans = num_humans

        self.rng = np.random.default_rng(seed)
        self.obstacles: List[GeneratedObstacle] = []
        self.fires: List[Fire] = []
        self.humans: List[Human] = []
        self.cold_zones: List[ColdZone] = []

    def local_ambient_temp(self, pos: np.ndarray) -> float:
        """Ambient temperature with Gaussian fire and cold-pocket influence."""
        p = np.asarray(pos, dtype=np.float64)
        temp = self.ambient_temp_c
        for zone in self.cold_zones:
            distance = np.linalg.norm(p[:2] - zone.pos[:2])
            temp += zone.delta_c * np.exp(-0.5 * (distance / max(zone.radius_m, 1.0)) ** 2)
        for fire in self.fires:
            distance = np.linalg.norm(p - fire.pos)
            temp += min(100.0, (fire.temp_c - self.ambient_temp_c) * fire.intensity *
                        np.exp(-0.5 * (distance / max(fire.radius_m * 5.0, 2.0)) ** 2))
        return float(temp)

    def is_in_safe_zone(self, x: float, y: float, clearance: float = 20.0) -> bool:
        """Ensures the 200m x 200m area around origin (0, 0) remains clear."""
        total_margin = self.safe_radius + clearance
        return abs(x) < total_margin and abs(y) < total_margin

    def generate(self) -> Tuple[str, List[GeneratedObstacle]]:
        """
        Generates the 10km x 10km congested disaster environment and exports valid MuJoCo MJCF XML.
        Returns:
          (mjcf_xml_content, list_of_generated_obstacles)
        """
        self.obstacles.clear()
        self.fires.clear()
        self.humans.clear()
        self.cold_zones.clear()
        self.rng = np.random.default_rng(self.seed)

        # 1. District Layout & Disaster Ground-Zero Zones across the 10km x 10km world
        # Prominent districts including an intensive Catastrophic Disaster Zone right next to origin
        urban_districts = [
            # Ground Zero Disaster Zone (starts just east of the takeoff plaza, packed with heavy rubble)
            {"center": (260.0, 160.0), "name": "Disaster Ground Zero", "radius": 750.0, "density": "disaster"},
            # Downtown Metro Core with massive skyscrapers
            {"center": (-380.0, 280.0), "name": "Downtown Central", "radius": 700.0, "density": "high"},
            # West Financial & High-Rise District
            {"center": (-1100.0, -800.0), "name": "West Financial Center", "radius": 850.0, "density": "high"},
            # Harbor & Port Disaster Ruin
            {"center": (850.0, -950.0), "name": "Harbor Industrial Ruins", "radius": 800.0, "density": "disaster"},
            # North Commercial Hub
            {"center": (1400.0, 1200.0), "name": "North Commercial Hub", "radius": 900.0, "density": "high"},
            # Southwest Sub-City
            {"center": (-2200.0, -2000.0), "name": "Southwest Sub-City", "radius": 1000.0, "density": "medium"},
            # East Industrial Basin
            {"center": (2800.0, -2200.0), "name": "East Industrial Basin", "radius": 1100.0, "density": "industrial"},
            # Northwest Tech Sector
            {"center": (-3200.0, 2600.0), "name": "Northwest Sector", "radius": 1100.0, "density": "medium"},
            # Northeast Perimeter Outpost
            {"center": (3500.0, 3200.0), "name": "Northeast Complex", "radius": 1200.0, "density": "disaster"},
        ]

        # Additional procedural outer districts
        num_outer = self.rng.integers(5, 9)
        for d_i in range(num_outer):
            cx = self.rng.uniform(-self.half_size * 0.82, self.half_size * 0.82)
            cy = self.rng.uniform(-self.half_size * 0.82, self.half_size * 0.82)
            if self.is_in_safe_zone(cx, cy, 350.0):
                continue
            d_type = self.rng.choice(["high", "medium", "disaster", "industrial"])
            urban_districts.append({
                "center": (cx, cy),
                "name": f"District_{d_i+10}",
                "radius": self.rng.uniform(600.0, 1200.0),
                "density": d_type,
            })

        # =========================================================================
        # 2. ENLARGED CONGESTED BUILDINGS (800+ Buildings: Mega Skyscrapers, Dense Blocks)
        # =========================================================================
        bldg_xml_lines = []
        bldg_count = 0

        buildings_in_districts = int(self.num_buildings * 0.86)

        for district in urban_districts:
            if bldg_count >= buildings_in_districts:
                break

            dcx, dcy = district["center"]
            d_radius = district["radius"]
            density = district["density"]

            num_blocks = self.rng.integers(3, 7) if density in ("high", "disaster") else self.rng.integers(2, 5)

            for _ in range(num_blocks):
                if bldg_count >= buildings_in_districts:
                    break

                b_dist = self.rng.uniform(0.0, d_radius * 0.85)
                b_ang = self.rng.uniform(0, 2 * math.pi)
                block_x = dcx + b_dist * math.cos(b_ang)
                block_y = dcy + b_dist * math.sin(b_ang)

                block_angle = self.rng.uniform(0, math.pi / 2)
                cos_a, sin_a = math.cos(block_angle), math.sin(block_angle)

                rows = self.rng.integers(2, 4)
                cols = self.rng.integers(2, 5)

                # Tight street gaps (10m to 24m)
                alley_x = self.rng.uniform(10.0, 24.0)
                alley_y = self.rng.uniform(10.0, 24.0)

                cursor_x = 0.0
                for r in range(rows):
                    cursor_y = 0.0
                    row_max_w = 0.0
                    for c in range(cols):
                        if bldg_count >= buildings_in_districts:
                            break

                        if density == "high":
                            cat = self.rng.choice(["skyscraper", "highrise", "midrise", "dense_row"], p=[0.30, 0.35, 0.25, 0.10])
                        elif density == "disaster":
                            cat = self.rng.choice(["damaged", "highrise", "midrise", "skyscraper"], p=[0.45, 0.25, 0.20, 0.10])
                        elif density == "industrial":
                            cat = self.rng.choice(["warehouse", "midrise", "damaged"], p=[0.50, 0.30, 0.20])
                        else:
                            cat = self.rng.choice(["highrise", "midrise", "dense_row", "skyscraper"], p=[0.25, 0.40, 0.25, 0.10])

                        # ENLARGED BUILDING DIMENSIONS:
                        if cat == "skyscraper":
                            w = self.rng.uniform(45.0, 80.0)
                            d = self.rng.uniform(45.0, 80.0)
                            h = self.rng.uniform(120.0, 250.0)  # Mega skyscraper up to 250m
                        elif cat == "highrise":
                            w = self.rng.uniform(35.0, 65.0)
                            d = self.rng.uniform(35.0, 65.0)
                            h = self.rng.uniform(60.0, 120.0)  # High-rise 60-120m
                        elif cat == "midrise":
                            w = self.rng.uniform(28.0, 55.0)
                            d = self.rng.uniform(28.0, 55.0)
                            h = self.rng.uniform(30.0, 65.0)
                        elif cat == "warehouse":
                            w = self.rng.uniform(60.0, 130.0)  # Massive warehouse footprint
                            d = self.rng.uniform(45.0, 95.0)
                            h = self.rng.uniform(16.0, 32.0)
                        elif cat == "damaged":
                            w = self.rng.uniform(25.0, 55.0)
                            d = self.rng.uniform(25.0, 55.0)
                            h = self.rng.uniform(12.0, 42.0)  # Collapsed / fractured building
                        else:  # dense_row
                            w = self.rng.uniform(18.0, 35.0)
                            d = self.rng.uniform(20.0, 38.0)
                            h = self.rng.uniform(14.0, 28.0)

                        local_x = cursor_x + w / 2.0
                        local_y = cursor_y + d / 2.0

                        bx = block_x + (local_x * cos_a - local_y * sin_a)
                        by = block_y + (local_x * sin_a + local_y * cos_a)

                        cursor_y += d + alley_y
                        row_max_w = max(row_max_w, w)

                        if self.is_in_safe_zone(bx, by, max(w, d) / 2.0):
                            continue

                        bldg_count += 1
                        bldg_id = f"Building_{bldg_count:03d}"
                        bz = h / 2.0
                        hx, hy, hz = w / 2.0, d / 2.0, h / 2.0

                        mat_idx = (bldg_count % 3) + 1
                        mat_name = f"building_facade{mat_idx}"
                        if cat == "damaged":
                            mat_name = "building_damaged_mat"
                        elif cat == "skyscraper":
                            mat_name = "building_glass_mat"

                        bldg_xml_lines.append(
                            f'    <body name="{bldg_id}" pos="{bx:.2f} {by:.2f} {bz:.2f}">'
                            f'\n      <geom name="{bldg_id}_geom" type="box" size="{hx:.2f} {hy:.2f} {hz:.2f}" material="{mat_name}"/>'
                            f'\n      <geom name="{bldg_id}_roof" type="box" pos="0 0 {hz + 0.1:.2f}" size="{max(0.2, hx - 0.5):.2f} {max(0.2, hy - 0.5):.2f} 0.1" material="building_roof" contype="0" conaffinity="0"/>'
                            f'\n    </body>'
                        )

                        self.obstacles.append(
                            GeneratedObstacle(
                                name=bldg_id,
                                obs_type="Building",
                                pos=np.array([bx, by, bz], dtype=np.float64),
                                shape="box",
                                size_params=np.array([w, d, h], dtype=np.float64),
                            )
                        )

                    cursor_x += row_max_w + alley_x

        # Generate remaining scattered buildings across 10 km map
        while bldg_count < self.num_buildings:
            bx = self.rng.uniform(-self.half_size * 0.90, self.half_size * 0.90)
            by = self.rng.uniform(-self.half_size * 0.90, self.half_size * 0.90)

            w = self.rng.uniform(25.0, 75.0)
            d = self.rng.uniform(25.0, 75.0)
            h = self.rng.uniform(16.0, 85.0)

            if self.is_in_safe_zone(bx, by, max(w, d) / 2.0):
                continue

            bldg_count += 1
            bldg_id = f"Building_{bldg_count:03d}"
            bz = h / 2.0
            hx, hy, hz = w / 2.0, d / 2.0, h / 2.0
            mat_name = f"building_facade{(bldg_count % 3) + 1}"

            bldg_xml_lines.append(
                f'    <body name="{bldg_id}" pos="{bx:.2f} {by:.2f} {bz:.2f}">'
                f'\n      <geom name="{bldg_id}_geom" type="box" size="{hx:.2f} {hy:.2f} {hz:.2f}" material="{mat_name}"/>'
                f'\n      <geom name="{bldg_id}_roof" type="box" pos="0 0 {hz + 0.1:.2f}" size="{max(0.2, hx - 0.5):.2f} {max(0.2, hy - 0.5):.2f} 0.1" material="building_roof" contype="0" conaffinity="0"/>'
                f'\n    </body>'
            )

            self.obstacles.append(
                GeneratedObstacle(
                    name=bldg_id,
                    obs_type="Building",
                    pos=np.array([bx, by, bz], dtype=np.float64),
                    shape="box",
                    size_params=np.array([w, d, h], dtype=np.float64),
                )
            )

        # =========================================================================
        # 3. MASSIVE DISASTER RUBBLE FIELDS (50 Zones x 30 Pieces = 1,500 Rubble Pieces)
        # =========================================================================
        rubble_xml_lines = []
        rubble_count = 0

        # Focus heavy rubble heavily into the Disaster Ground Zero and collapsed zones
        disaster_centers = [
            (260.0, 160.0),   # Ground Zero right near takeoff plaza
            (320.0, 240.0),
            (210.0, 350.0),
            (-350.0, 420.0),
            (850.0, -950.0),
            (-1100.0, -800.0),
        ]

        rubble_zone_anchors = []
        # First 18 clusters dedicated directly to the prominent disaster zones
        for i in range(min(18, self.num_rubble_zones)):
            base_pt = disaster_centers[i % len(disaster_centers)]
            r_ang = self.rng.uniform(0, 2 * math.pi)
            r_rad = self.rng.uniform(30.0, 250.0)
            rx = base_pt[0] + r_rad * math.cos(r_ang)
            ry = base_pt[1] + r_rad * math.sin(r_ang)
            if not self.is_in_safe_zone(rx, ry, 40.0):
                rubble_zone_anchors.append((rx, ry))

        # Remaining rubble clusters distributed across urban districts and outer terrain
        while len(rubble_zone_anchors) < self.num_rubble_zones:
            if urban_districts and self.rng.random() < 0.70:
                distr = self.rng.choice(urban_districts)
                r_ang = self.rng.uniform(0, 2 * math.pi)
                r_dist = self.rng.uniform(30.0, distr["radius"] * 0.95)
                rcx = distr["center"][0] + r_dist * math.cos(r_ang)
                rcy = distr["center"][1] + r_dist * math.sin(r_ang)
            else:
                rcx = self.rng.uniform(-self.half_size * 0.88, self.half_size * 0.88)
                rcy = self.rng.uniform(-self.half_size * 0.88, self.half_size * 0.88)

            if self.is_in_safe_zone(rcx, rcy, 40.0):
                continue
            rubble_zone_anchors.append((rcx, rcy))

        for rcx, rcy in rubble_zone_anchors:
            cluster_radius = self.rng.uniform(25.0, 75.0)
            for _ in range(self.num_rubble_per_zone):
                ang = self.rng.uniform(0, 2 * math.pi)
                dist = self.rng.uniform(0, cluster_radius)
                rx = rcx + dist * math.cos(ang)
                ry = rcy + dist * math.sin(ang)

                if self.is_in_safe_zone(rx, ry, 10.0):
                    continue

                rubble_count += 1
                rubble_id = f"Rubble_{rubble_count:03d}"

                # ENLARGED RUBBLE PIECES:
                # 55% massive tilted slabs, 25% fallen pillars, 20% giant debris boulders
                shape_choice = self.rng.choice(["box", "cylinder", "sphere"], p=[0.55, 0.25, 0.20])
                mat_name = "rubble_mat" if (rubble_count % 2 == 0) else "rubble_alt_mat"

                if shape_choice == "box":
                    # Giant collapsed slabs: 2.5m to 12.0m wide!
                    sx = self.rng.uniform(2.5, 12.0)
                    sy = self.rng.uniform(2.5, 12.0)
                    sz = self.rng.uniform(1.2, 5.5)
                    rz = sz / 2.0
                    yaw = self.rng.uniform(0, 360)
                    pitch = self.rng.uniform(-25, 25)
                    roll = self.rng.uniform(-25, 25)

                    rubble_xml_lines.append(
                        f'    <body name="{rubble_id}" pos="{rx:.2f} {ry:.2f} {rz:.2f}" euler="{pitch:.1f} {roll:.1f} {yaw:.1f}">'
                        f'\n      <geom name="{rubble_id}_geom" type="box" size="{sx/2:.2f} {sy/2:.2f} {sz/2:.2f}" material="{mat_name}"/>'
                        f'\n    </body>'
                    )
                    self.obstacles.append(
                        GeneratedObstacle(
                            name=rubble_id,
                            obs_type="Rubble",
                            pos=np.array([rx, ry, rz], dtype=np.float64),
                            shape="box",
                            size_params=np.array([sx, sy, sz], dtype=np.float64),
                            heading_deg=yaw,
                        )
                    )

                elif shape_choice == "cylinder":
                    # Fallen concrete pillars & steel conduits: radius 0.8m to 3.5m, length 3m to 15m!
                    radius = self.rng.uniform(0.8, 3.5)
                    length = self.rng.uniform(3.0, 15.0)
                    rz = length / 2.0
                    tilt_pitch = self.rng.uniform(-35, 35)
                    tilt_yaw = self.rng.uniform(0, 360)

                    rubble_xml_lines.append(
                        f'    <body name="{rubble_id}" pos="{rx:.2f} {ry:.2f} {rz:.2f}" euler="{tilt_pitch:.1f} 0 {tilt_yaw:.1f}">'
                        f'\n      <geom name="{rubble_id}_geom" type="cylinder" size="{radius:.2f} {length/2:.2f}" material="{mat_name}"/>'
                        f'\n    </body>'
                    )
                    self.obstacles.append(
                        GeneratedObstacle(
                            name=rubble_id,
                            obs_type="Rubble",
                            pos=np.array([rx, ry, rz], dtype=np.float64),
                            shape="cylinder",
                            size_params=np.array([radius, length], dtype=np.float64),
                        )
                    )

                else:  # sphere
                    # Giant debris mounds & collapsed chunks: radius 1.8m to 7.0m!
                    s_radius = self.rng.uniform(1.8, 7.0)
                    rz = s_radius
                    rubble_xml_lines.append(
                        f'    <body name="{rubble_id}" pos="{rx:.2f} {ry:.2f} {rz:.2f}">'
                        f'\n      <geom name="{rubble_id}_geom" type="sphere" size="{s_radius:.2f}" material="{mat_name}"/>'
                        f'\n    </body>'
                    )
                    self.obstacles.append(
                        GeneratedObstacle(
                            name=rubble_id,
                            obs_type="Rubble",
                            pos=np.array([rx, ry, rz], dtype=np.float64),
                            shape="sphere",
                            size_params=np.array([s_radius], dtype=np.float64),
                        )
                    )

        # =========================================================================
        # 4. ENLARGED TREES (900 Trees: Urban Parks, Boulevards & Giant Forests)
        # =========================================================================
        tree_xml_lines = []
        tree_count = 0

        forest_zones = [
            {"center": (-180.0, 240.0), "radius": 240.0},  # Park immediately near takeoff
            {"center": (1100.0, -800.0), "radius": 450.0},
            {"center": (-1800.0, 1500.0), "radius": 600.0},
            {"center": (2600.0, 2400.0), "radius": 750.0},
            {"center": (-3400.0, -2600.0), "radius": 800.0},
        ]

        for i in range(self.num_trees):
            r_type = self.rng.random()
            if r_type < 0.45 and urban_districts:
                distr = self.rng.choice(urban_districts)
                ang = self.rng.uniform(0, 2 * math.pi)
                r_dist = self.rng.uniform(20.0, distr["radius"])
                tx = distr["center"][0] + r_dist * math.cos(ang)
                ty = distr["center"][1] + r_dist * math.sin(ang)
            elif r_type < 0.80 and forest_zones:
                f_zone = self.rng.choice(forest_zones)
                ang = self.rng.uniform(0, 2 * math.pi)
                r_dist = self.rng.uniform(0, f_zone["radius"])
                tx = f_zone["center"][0] + r_dist * math.cos(ang)
                ty = f_zone["center"][1] + r_dist * math.sin(ang)
            else:
                tx = self.rng.uniform(-self.half_size * 0.90, self.half_size * 0.90)
                ty = self.rng.uniform(-self.half_size * 0.90, self.half_size * 0.90)

            if self.is_in_safe_zone(tx, ty, 20.0):
                continue

            tree_count += 1
            tree_id = f"Tree_{tree_count:03d}"

            # ENLARGED TREE DIMENSIONS:
            # Trunk: height 8m to 24m, radius 0.6m to 2.2m
            # Crown: radius 5m to 16m!
            trunk_h = self.rng.uniform(8.0, 24.0)
            trunk_r = self.rng.uniform(0.6, 2.2)
            crown_r = self.rng.uniform(5.0, 16.0)
            crown_z = trunk_h + crown_r * 0.65

            crown_mat = "tree_crown_mat" if (tree_count % 2 == 0) else "tree_crown_alt_mat"

            tree_xml_lines.append(
                f'    <body name="{tree_id}" pos="{tx:.2f} {ty:.2f} 0">'
                f'\n      <geom name="{tree_id}_trunk" type="cylinder" fromto="0 0 0 0 0 {trunk_h:.2f}" size="{trunk_r:.2f}" material="tree_trunk_mat"/>'
                f'\n      <geom name="{tree_id}_crown" type="sphere" pos="0 0 {crown_z:.2f}" size="{crown_r:.2f}" material="{crown_mat}"/>'
                f'\n    </body>'
            )

            self.obstacles.append(
                GeneratedObstacle(
                    name=tree_id,
                    obs_type="Tree",
                    pos=np.array([tx, ty, crown_z / 2.0], dtype=np.float64),
                    shape="tree",
                    size_params=np.array([trunk_r, trunk_h, crown_r, crown_z], dtype=np.float64),
                )
            )

        # =========================================================================
        # 5. ENLARGED POLES & TRANSMISSION TOWERS (350 Poles/Towers)
        # =========================================================================
        pole_xml_lines = []
        pole_count = 0

        for i in range(self.num_poles):
            if urban_districts and self.rng.random() < 0.70:
                distr = self.rng.choice(urban_districts)
                ang = self.rng.uniform(0, 2 * math.pi)
                r_dist = self.rng.uniform(15.0, distr["radius"] * 1.05)
                px = distr["center"][0] + r_dist * math.cos(ang)
                py = distr["center"][1] + r_dist * math.sin(ang)
            else:
                px = self.rng.uniform(-self.half_size * 0.88, self.half_size * 0.88)
                py = self.rng.uniform(-self.half_size * 0.88, self.half_size * 0.88)

            if self.is_in_safe_zone(px, py, 15.0):
                continue

            pole_count += 1
            pole_id = f"Pole_{pole_count:03d}"

            # ENLARGED POLES / TOWERS:
            # Street lamps: 8-15m, Utility poles: 16-28m, Tall Telecom Towers: 30-70m!
            p_cat = self.rng.choice(["street", "utility", "tower"], p=[0.45, 0.35, 0.20])
            if p_cat == "street":
                pole_h = self.rng.uniform(8.0, 16.0)
                pole_r = self.rng.uniform(0.25, 0.45)
                pole_mat = "pole_mat"
            elif p_cat == "utility":
                pole_h = self.rng.uniform(16.0, 28.0)
                pole_r = self.rng.uniform(0.40, 0.75)
                pole_mat = "pole_wood_mat"
            else:  # massive broadcast / power tower
                pole_h = self.rng.uniform(32.0, 70.0)
                pole_r = self.rng.uniform(0.80, 1.60)
                pole_mat = "pole_mat"

            pole_xml_lines.append(
                f'    <body name="{pole_id}" pos="{px:.2f} {py:.2f} 0">'
                f'\n      <geom name="{pole_id}_geom" type="cylinder" fromto="0 0 0 0 0 {pole_h:.2f}" size="{pole_r:.2f}" material="{pole_mat}"/>'
                f'\n    </body>'
            )

            self.obstacles.append(
                GeneratedObstacle(
                    name=pole_id,
                    obs_type="Pole",
                    pos=np.array([px, py, pole_h / 2.0], dtype=np.float64),
                    shape="cylinder",
                    size_params=np.array([pole_r, pole_h], dtype=np.float64),
                )
            )

        # =========================================================================
        # 6. ASSEMBLE FULL MUJOCO XML (10 km x 10 km)
        # =========================================================================
        # Thermal features share this generator's seeded stream, but remain separate
        # from the obstacle list and therefore from all collision queries.
        building_list = [o for o in self.obstacles if o.obs_type == "Building"]
        rubble_list = [o for o in self.obstacles if o.obs_type == "Rubble"]
        tree_list = [o for o in self.obstacles if o.obs_type == "Tree"]
        fire_xml, human_xml = [], []
        for zone_i in range(self.num_fire_zones):
            count = int(self.rng.integers(1, 5))
            anchor = rubble_list[int(self.rng.integers(len(rubble_list)))].pos if rubble_list else np.array([250., 150., 0.])
            for _ in range(count):
                roll = self.rng.random()
                indoor = roll >= 0.60 and roll < 0.90 and bool(building_list)
                rooftop = roll >= 0.90 and bool(building_list)
                parent = None
                if indoor or rooftop:
                    b = building_list[int(self.rng.integers(len(building_list)))]
                    parent = b.name
                    fx = b.pos[0] + self.rng.uniform(-b.size_params[0] * 0.4, b.size_params[0] * 0.4)
                    fy = b.pos[1] + self.rng.uniform(-b.size_params[1] * 0.4, b.size_params[1] * 0.4)
                    fz = (b.pos[2] + self.rng.uniform(-b.size_params[2] * 0.35, b.size_params[2] * 0.35)) if indoor else b.size_params[2] + 0.5
                else:
                    fx, fy = anchor[0] + self.rng.normal(0, 25), anchor[1] + self.rng.normal(0, 25)
                    fz = 1.0
                if self.is_in_safe_zone(fx, fy, 0.0):
                    fx = 115.0
                cls = int(self.rng.choice(3, p=[0.50, 0.35, 0.15]))
                ranges = [(0.5, 2.0, 300, 500), (2.0, 6.0, 500, 700), (6.0, 15.0, 700, 900)][cls]
                radius, temp = self.rng.uniform(*ranges[:2]), self.rng.uniform(*ranges[2:])
                fire = Fire(f"Fire_{len(self.fires)+1:03d}", np.array([fx, fy, fz]), radius, temp,
                            float(self.rng.uniform(0.35, 1.0)), indoor, parent)
                self.fires.append(fire)
                fire_xml.append(f'    <body name="{fire.id}" pos="{fx:.2f} {fy:.2f} {fz:.2f}"><geom name="{fire.id}_halo" type="sphere" size="{radius*1.8:.2f}" rgba="1 0.12 0.015 0.20" contype="0" conaffinity="0"/><geom name="{fire.id}_core" type="sphere" size="{radius:.2f}" rgba="1 0.42 0.03 0.78" contype="0" conaffinity="0"/></body>')

        # Cold pockets use tree patches as park anchors where possible.
        for i in range(self.num_cold_zones):
            tree = tree_list[int(self.rng.integers(len(tree_list)))] if tree_list else None
            x, y = (tree.pos[0], tree.pos[1]) if tree is not None else (self.rng.uniform(-4000, 4000), self.rng.uniform(-4000, 4000))
            if self.is_in_safe_zone(x, y, 0):
                x += 120
            self.cold_zones.append(ColdZone(f"Cold_{i+1:02d}", np.array([x, y, 0.0]),
                                            float(self.rng.uniform(40, 120)), float(self.rng.uniform(*self.cold_zone_delta_c))))

        # Random victims are kept on open ground or placed on a building's ground floor.
        for i in range(max(0, self.num_humans - 2)):
            indoor = bool(building_list) and self.rng.random() < 0.15
            parent = None
            if indoor:
                b = building_list[int(self.rng.integers(len(building_list)))]
                x = b.pos[0] + self.rng.uniform(-b.size_params[0]*0.4, b.size_params[0]*0.4)
                y = b.pos[1] + self.rng.uniform(-b.size_params[1]*0.4, b.size_params[1]*0.4)
                parent = b.name
            else:
                src = rubble_list[int(self.rng.integers(len(rubble_list)))].pos if rubble_list else np.zeros(3)
                x, y = src[0] + self.rng.normal(0, 40), src[1] + self.rng.normal(0, 40)
            if self.is_in_safe_zone(x, y, 0):
                x = 120.0
            posture = str(self.rng.choice(["standing", "lying"], p=[0.35, 0.65]))
            human = Human(f"Human_{i+1:03d}", np.array([x, y, 0.9 if posture == "standing" else 0.25]), posture,
                          surface_temp_c=float(self.rng.uniform(31.0, 34.0)), indoor=indoor, parent_building=parent)
            self.humans.append(human)
        for hid, x, y in (("Human_Demo_01", 30.0, 0.0), ("Human_Demo_02", 8.0, 5.0)):
            self.humans.append(Human(hid, np.array([x, y, 0.9])))
        for human in self.humans:
            height = 1.25 if human.posture == "standing" else 0.22
            euler = "0 90 0" if human.posture == "lying" else "0 0 0"
            human_xml.append(f'    <body name="{human.id}" pos="{human.pos[0]:.2f} {human.pos[1]:.2f} {human.pos[2]:.2f}" euler="{euler}"><geom name="{human.id}_torso" type="capsule" fromto="0 0 0 0 0 {height:.2f}" size="0.18" rgba="0.78 0.58 0.36 1" contype="0" conaffinity="0"/><geom name="{human.id}_head" type="sphere" pos="0 0 {height+0.15:.2f}" size="0.15" rgba="0.88 0.69 0.50 1" contype="0" conaffinity="0"/></body>')

        xml_content = f"""<mujoco model="congested_disaster_city_10km">
  <compiler angle="degree" coordinate="local"/>
  <option gravity="0 0 0" timestep="0.0166667"/>
  <!-- Allocate 64M stack memory for scene -->
  <size memory="64M"/>

  <default>
    <!-- Static obstacles detect collisions with drone without checking against each other -->
    <geom contype="1" conaffinity="0"/>
  </default>

  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="{self.shadow_size}"/>
    <headlight ambient="0.45 0.45 0.48" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <!-- 30 km far clipping plane for 10 km environment -->
    <map znear="0.1" zfar="30000.0"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.38 0.55 0.75" rgb2="0.14 0.18 0.25" width="512" height="512"/>
    
    <!-- Terrain Texture: 10 km asphalt / disaster concrete ground -->
    <texture name="terrain_tex" type="2d" builtin="checker" width="512" height="512" rgb1="0.22 0.23 0.24" rgb2="0.16 0.17 0.18"/>
    <material name="terrain_mat" texture="terrain_tex" texrepeat="1250 1250" reflectance="0.04" shininess="0.05"/>

    <!-- Drone Materials -->
    <material name="drone_chassis" rgba="0.12 0.12 0.14 1.0" specular="0.4" shininess="0.3"/>
    <material name="drone_arm" rgba="0.28 0.30 0.33 1.0" specular="0.3"/>
    <material name="rotor_front" rgba="0.95 0.25 0.15 0.9" specular="0.5"/>
    <material name="rotor_rear" rgba="0.15 0.65 0.95 0.9" specular="0.5"/>
    <material name="drone_indicator" rgba="0.2 0.9 0.3 1.0"/>
    <material name="comm_sphere_mat" rgba="0.2 0.55 1.0 0.12"/>
    <material name="comm_sphere_d2_mat" rgba="0.9 0.5 0.1 0.10"/>

    <!-- Building Facade Materials (Varied colors: Concrete, Glass, Brick, Steel) -->
    <material name="building_facade1" rgba="0.45 0.47 0.52 1.0" specular="0.2"/>
    <material name="building_facade2" rgba="0.36 0.38 0.40 1.0" specular="0.2"/>
    <material name="building_facade3" rgba="0.50 0.48 0.45 1.0" specular="0.15"/>
    <material name="building_glass_mat" rgba="0.25 0.40 0.55 1.0" specular="0.6" shininess="0.5"/>
    <material name="building_damaged_mat" rgba="0.35 0.32 0.30 1.0" specular="0.1"/>
    <material name="building_roof" rgba="0.20 0.22 0.24 1.0"/>

    <!-- Rubble Materials -->
    <material name="rubble_mat" rgba="0.50 0.46 0.40 1.0" specular="0.1"/>
    <material name="rubble_alt_mat" rgba="0.36 0.34 0.32 1.0" specular="0.1"/>

    <!-- Tree Materials -->
    <material name="tree_trunk_mat" rgba="0.34 0.22 0.14 1.0" specular="0.05"/>
    <material name="tree_crown_mat" rgba="0.16 0.40 0.18 1.0" specular="0.1"/>
    <material name="tree_crown_alt_mat" rgba="0.22 0.46 0.20 1.0" specular="0.1"/>

    <!-- Pole Materials -->
    <material name="pole_mat" rgba="0.52 0.54 0.56 1.0" specular="0.35"/>
    <material name="pole_wood_mat" rgba="0.40 0.30 0.20 1.0" specular="0.1"/>
  </asset>

  <worldbody>
    <!-- Lighting -->
    <light directional="true" pos="500 500 2500" dir="-0.3 -0.3 -1.0" diffuse="0.9 0.9 0.9" specular="0.3 0.3 0.3"/>
    <light directional="true" pos="-1000 -1000 2000" dir="0.4 0.4 -0.8" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>

    <!-- Terrain: 10 km x 10 km ground plane (half-size = 5,000 m) with collision enabled -->
    <geom name="terrain" type="plane" size="{self.half_size:.1f} {self.half_size:.1f} 1" material="terrain_mat"/>

    <!-- Visible 3D Coordinate Triad at Origin (0, 0, 0) -->
    <geom name="axis_origin" type="sphere" pos="0 0 0.2" size="1.2" rgba="1 1 1 1" contype="0" conaffinity="0"/>
    <geom name="axis_x" type="cylinder" fromto="0 0 0.2 40 0 0.2" size="0.7" rgba="0.95 0.15 0.15 1" contype="0" conaffinity="0"/>
    <geom name="axis_y" type="cylinder" fromto="0 0 0.2 0 40 0.2" size="0.7" rgba="0.15 0.85 0.2 1" contype="0" conaffinity="0"/>
    <geom name="axis_z" type="cylinder" fromto="0 0 0.2 0 0 40" size="0.7" rgba="0.2 0.4 0.95 1" contype="0" conaffinity="0"/>

    <!-- Safe Takeoff Plaza (visual circle at 100m radius) -->
    <geom name="safe_zone_marker" type="cylinder" pos="0 0 0.05" size="100.0 0.05" rgba="0.2 0.8 0.4 0.15" contype="0" conaffinity="0"/>

    <!-- ==================== DRONES ==================== -->
    <!-- Primary Drone D01 (Mocap body: direct position control) -->
    <body name="D01" mocap="true" pos="0 0 5">
      <geom name="D01_body" type="cylinder" size="0.9 0.25" material="drone_chassis" contype="0" conaffinity="1"/>
      <geom name="D01_dome" type="sphere" pos="0 0 0.15" size="0.45" material="drone_indicator" contype="0" conaffinity="0"/>
      <geom name="D01_arm1" type="capsule" fromto="-1.5 -1.5 0 1.5 1.5 0" size="0.1" material="drone_arm" contype="0" conaffinity="1"/>
      <geom name="D01_arm2" type="capsule" fromto="-1.5 1.5 0 1.5 -1.5 0" size="0.1" material="drone_arm" contype="0" conaffinity="1"/>
      <geom name="D01_rotor_fl" type="cylinder" pos="1.5 1.5 0.1" size="0.65 0.03" material="rotor_front" contype="0" conaffinity="1"/>
      <geom name="D01_rotor_fr" type="cylinder" pos="1.5 -1.5 0.1" size="0.65 0.03" material="rotor_front" contype="0" conaffinity="1"/>
      <geom name="D01_rotor_bl" type="cylinder" pos="-1.5 1.5 0.1" size="0.65 0.03" material="rotor_rear" contype="0" conaffinity="1"/>
      <geom name="D01_rotor_br" type="cylinder" pos="-1.5 -1.5 0.1" size="0.65 0.03" material="rotor_rear" contype="0" conaffinity="1"/>
      <!-- 100m Communication Radius Sphere (visual only, non-colliding) -->
      <geom name="D01_comm_sphere" type="sphere" size="100.0" material="comm_sphere_mat" contype="0" conaffinity="0"/>
    </body>

    <!-- Standby Drone D02 (can be spawned for 100m inter-drone test) -->
    <body name="D02" mocap="true" pos="0 0 -100">
      <geom name="D02_body" type="cylinder" size="0.9 0.25" material="drone_chassis" contype="0" conaffinity="1"/>
      <geom name="D02_dome" type="sphere" pos="0 0 0.15" size="0.45" rgba="0.9 0.6 0.1 1" contype="0" conaffinity="0"/>
      <geom name="D02_arm1" type="capsule" fromto="-1.5 -1.5 0 1.5 1.5 0" size="0.1" material="drone_arm" contype="0" conaffinity="1"/>
      <geom name="D02_arm2" type="capsule" fromto="-1.5 1.5 0 1.5 -1.5 0" size="0.1" material="drone_arm" contype="0" conaffinity="1"/>
      <geom name="D02_rotor_fl" type="cylinder" pos="1.5 1.5 0.1" size="0.65 0.03" material="rotor_front" contype="0" conaffinity="1"/>
      <geom name="D02_rotor_fr" type="cylinder" pos="1.5 -1.5 0.1" size="0.65 0.03" material="rotor_front" contype="0" conaffinity="1"/>
      <geom name="D02_rotor_bl" type="cylinder" pos="-1.5 1.5 0.1" size="0.65 0.03" material="rotor_rear" contype="0" conaffinity="1"/>
      <geom name="D02_rotor_br" type="cylinder" pos="-1.5 -1.5 0.1" size="0.65 0.03" material="rotor_rear" contype="0" conaffinity="1"/>
      <geom name="D02_comm_sphere" type="sphere" size="100.0" material="comm_sphere_d2_mat" contype="0" conaffinity="0"/>
    </body>

    <!-- Standby Drone D03 -->
    <body name="D03" mocap="true" pos="0 0 -100">
      <geom name="D03_body" type="cylinder" size="0.9 0.25" material="drone_chassis" contype="0" conaffinity="1"/>
      <geom name="D03_dome" type="sphere" pos="0 0 0.15" size="0.45" rgba="0.8 0.2 0.8 1" contype="0" conaffinity="0"/>
      <geom name="D03_arm1" type="capsule" fromto="-1.5 -1.5 0 1.5 1.5 0" size="0.1" material="drone_arm" contype="0" conaffinity="1"/>
      <geom name="D03_arm2" type="capsule" fromto="-1.5 1.5 0 1.5 -1.5 0" size="0.1" material="drone_arm" contype="0" conaffinity="1"/>
      <geom name="D03_rotor_fl" type="cylinder" pos="1.5 1.5 0.1" size="0.65 0.03" material="rotor_front" contype="0" conaffinity="1"/>
      <geom name="D03_rotor_fr" type="cylinder" pos="1.5 -1.5 0.1" size="0.65 0.03" material="rotor_front" contype="0" conaffinity="1"/>
      <geom name="D03_rotor_bl" type="cylinder" pos="-1.5 1.5 0.1" size="0.65 0.03" material="rotor_rear" contype="0" conaffinity="1"/>
      <geom name="D03_rotor_br" type="cylinder" pos="-1.5 -1.5 0.1" size="0.65 0.03" material="rotor_rear" contype="0" conaffinity="1"/>
      <geom name="D03_comm_sphere" type="sphere" size="100.0" rgba="0.8 0.2 0.8 0.08" contype="0" conaffinity="0"/>
    </body>

    <!-- ==================== PROCEDURAL BUILDINGS ({len(bldg_xml_lines)}) ==================== -->
{chr(10).join(bldg_xml_lines)}

    <!-- ==================== PROCEDURAL RUBBLE ({len(rubble_xml_lines)}) ==================== -->
{chr(10).join(rubble_xml_lines)}

    <!-- ==================== PROCEDURAL TREES ({len(tree_xml_lines)}) ==================== -->
{chr(10).join(tree_xml_lines)}

    <!-- ==================== PROCEDURAL POLES ({len(pole_xml_lines)}) ==================== -->
{chr(10).join(pole_xml_lines)}

    <!-- Thermal fires and static human victims; non-colliding -->
{chr(10).join(fire_xml)}
{chr(10).join(human_xml)}

  </worldbody>
</mujoco>
"""
        return xml_content, self.obstacles

    def export_to_file(self, output_path: str = "generated_scene.xml") -> List[GeneratedObstacle]:
        """Generates the environment and saves it to an XML file."""
        xml_content, obstacles = self.generate()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        print(f"[Environment Generator] Exported {len(obstacles)} procedural objects to '{output_path}' (10 km x 10 km, Seed: {self.seed})")
        return obstacles


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate 10km x 10km MuJoCo Disaster City Environment")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for reproducibility")
    parser.add_argument("--buildings", type=int, default=800, help="Number of buildings")
    parser.add_argument("--rubble_zones", type=int, default=50, help="Number of rubble zones")
    parser.add_argument("--rubble_per_zone", type=int, default=30, help="Pieces per rubble zone")
    parser.add_argument("--trees", type=int, default=900, help="Number of trees")
    parser.add_argument("--poles", type=int, default=350, help="Number of poles")
    parser.add_argument("--output", type=str, default="generated_scene.xml", help="Output MJCF XML path")
    args = parser.parse_args()

    gen = ProceduralEnvironmentGenerator(
        env_size=10000.0,
        seed=args.seed,
        num_buildings=args.buildings,
        num_rubble_zones=args.rubble_zones,
        num_rubble_per_zone=args.rubble_per_zone,
        num_trees=args.trees,
        num_poles=args.poles,
    )
    gen.export_to_file(args.output)
