"""
Module: src.rl
Owner: Satwik Choudhury
Responsibilities:
- Multi-Agent Proximal Policy Optimization (MAPPO) with Centralized Training & Decentralized Execution (CTDE).
- Permutation-Invariant Neighbor Self-Attention Encoders.
- Decentralized 3D Actor and Centralized 3D Critic neural network architectures.
- Policy optimization, GAE calculation, and model checkpointing.
"""
from .mappo_agent import Decentralized3DActor, Centralized3DCritic, MAPPOAgent

__all__ = ["Decentralized3DActor", "Centralized3DCritic", "MAPPOAgent"]
