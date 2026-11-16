"""
Module: src.communication
Owner: Chandan Mondal
Responsibilities:
- Simulated P2P Ad-Hoc Mesh Network with realistic physical constraints.
- Range-based signal attenuation, packet loss probability, and transmission latency.
- Structured telemetry message schemas (Protobuf / JSON / MAVLink bridge).
"""
from .comm_mesh import MeshNetworkSimulator

__all__ = ["MeshNetworkSimulator"]
