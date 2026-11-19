"""Static thermal world entities generated alongside the obstacle map."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Fire:
    id: str
    pos: np.ndarray
    radius_m: float
    temp_c: float
    intensity: float
    indoor: bool = False
    parent_building: str | None = None


@dataclass
class ColdZone:
    id: str
    pos: np.ndarray
    radius_m: float
    delta_c: float


@dataclass
class Human:
    id: str
    pos: np.ndarray
    posture: str = "standing"
    body_temp_c: float = 36.6
    surface_temp_c: float = 32.5
    indoor: bool = False
    parent_building: str | None = None
    detected_faint: bool = False
    confirmed: bool = False
