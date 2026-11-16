"""
Module: src.safety
Owner: Arkendu Chatterjee
Responsibilities:
- Deterministic Safety Shield (Velocity Obstacles / ORCA / Control Barrier Functions).
- Anti-tunneling continuous swept-path collision math.
- Action filter guaranteeing minimum inter-drone separation (zero-collision guarantee).
- Safety ablation runner: benchmark comparisons of Shield ON vs. Shield OFF.
"""
from .safety_shield import SafetyShield

__all__ = ["SafetyShield"]
