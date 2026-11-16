"""
Module: src.dynamics
Owner: Arkendu Chatterjee
Responsibilities:
- Realistic drone flight dynamics: acceleration limits, inertia, aerodynamic drag.
- Environmental disturbances: stochastic 3D wind gusts and turbulence.
- Battery discharge curves and Return-To-Launch (RTL) energy management.
"""
from .flight_dynamics import FlightDynamics

__all__ = ["FlightDynamics"]
