"""
Module: src.coordination
Owner: Somnath Saha
Responsibilities:
- Decentralized Swarm Intelligence & Autonomous Exploration.
- Distributed Bayesian & Byzantine Consensus for survivor target validation.
- Dynamic Role & Task Allocation (utility auctioning: thermal scanner vs medical kit vs relay drone).
- Self-Healing Manager (heartbeat monitoring & Voronoi sector redistribution upon drone failure).
"""
from .consensus_engine import ConsensusEngine
from .task_allocator import TaskAllocator
from .self_healing import SelfHealingManager

__all__ = ["ConsensusEngine", "TaskAllocator", "SelfHealingManager"]
