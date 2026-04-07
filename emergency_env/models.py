"""
Emergency Resource Allocation - OpenEnv Models
===============================================
Defines the type-safe data structures used by the environment:
  - Action      : Which ambulance moves in which direction
  - Observation : Full grid snapshot returned after each step/reset
  - State       : Episode-level metadata returned by state()
  - StepResult  : Bundles (observation, reward, done, info)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Direction(IntEnum):
    """Cardinal movement directions for an ambulance."""
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3


class Priority(IntEnum):
    """Emergency request severity levels."""
    NORMAL   = 1   # +10 reward on pickup
    CRITICAL = 2   # +20 reward on pickup; -50 penalty if missed within 20 steps


# ---------------------------------------------------------------------------
# Core model objects
# ---------------------------------------------------------------------------


@dataclass
class EmergencyRequest:
    """A single pending emergency on the grid."""
    request_id: int
    x: int
    y: int
    priority: Priority
    age: int = 0          # Steps since this request was spawned

    def to_dict(self) -> dict:
        return {
            "id":       self.request_id,
            "x":        self.x,
            "y":        self.y,
            "priority": self.priority.value,
            "age":      self.age,
        }


@dataclass
class AmbulanceState:
    """Current position of a single ambulance."""
    ambulance_id: int
    x: int
    y: int

    def to_dict(self) -> dict:
        return {
            "id": self.ambulance_id,
            "x":  self.x,
            "y":  self.y,
        }


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


@dataclass
class EmergencyAction:
    """
    Discrete action: move ambulance `ambulance_id` one step in `direction`.

    Action space (flat integer encoding):
        action = ambulance_id * 4 + direction
        ambulance_id ∈ {0, 1}
        direction    ∈ {UP=0, DOWN=1, LEFT=2, RIGHT=3}
    Total action space size = 2 × 4 = 8
    """
    ambulance_id: int       # 0 or 1
    direction: Direction    # UP / DOWN / LEFT / RIGHT

    @staticmethod
    def from_int(action_int: int) -> "EmergencyAction":
        """Decode a flat integer action into (ambulance_id, direction)."""
        if not (0 <= action_int < 8):
            raise ValueError(f"Action must be in [0, 7]; got {action_int}")
        ambulance_id = action_int // 4
        direction    = Direction(action_int % 4)
        return EmergencyAction(ambulance_id=ambulance_id, direction=direction)

    def to_int(self) -> int:
        """Encode as flat integer."""
        return self.ambulance_id * 4 + int(self.direction)

    def to_dict(self) -> dict:
        return {
            "ambulance_id": self.ambulance_id,
            "direction":    self.direction.name,
        }


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


@dataclass
class EmergencyObservation:
    """
    Full grid snapshot returned by reset() and step().

    Fields
    ------
    ambulances          : List of current ambulance positions.
    pending_requests    : List of all unresolved emergency requests.
    grid_size           : (width, height) of the city grid.
    step_number         : Current step within the episode.
    action_space_size   : Total number of valid discrete actions (always 8).
    """
    ambulances:        List[dict]          # serialised AmbulanceState
    pending_requests:  List[dict]          # serialised EmergencyRequest
    grid_size:         Tuple[int, int]     # (10, 10)
    step_number:       int
    action_space_size: int = 8

    def to_dict(self) -> dict:
        return {
            "ambulances":        self.ambulances,
            "pending_requests":  self.pending_requests,
            "grid_size":         list(self.grid_size),
            "step_number":       self.step_number,
            "action_space_size": self.action_space_size,
        }


# ---------------------------------------------------------------------------
# State (episode metadata)
# ---------------------------------------------------------------------------


@dataclass
class EmergencyState:
    """
    Episode-level metadata returned by state().
    Mirrors the OpenEnv convention for episode_id and step_count.
    """
    episode_id:         int
    step_count:         int
    total_reward:       float
    resolved_count:     int            # emergencies resolved so far
    missed_critical:    int            # critical requests missed (penalty applied)
    ambulance_positions: List[dict]   # current positions
    pending_requests:    List[dict]   # unresolved requests with priorities

    def to_dict(self) -> dict:
        return {
            "episode_id":          self.episode_id,
            "step_count":          self.step_count,
            "total_reward":        self.total_reward,
            "resolved_count":      self.resolved_count,
            "missed_critical":     self.missed_critical,
            "ambulance_positions": self.ambulance_positions,
            "pending_requests":    self.pending_requests,
        }


# ---------------------------------------------------------------------------
# StepResult  (OpenEnv standard return type)
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """
    Standard OpenEnv return object from step().
    Compatible with the (observation, reward, done, info) tuple convention.
    """
    observation: EmergencyObservation
    reward:      float
    done:        bool
    info:        dict = field(default_factory=dict)

    def as_tuple(self):
        """Return the classic RL 4-tuple."""
        return self.observation, self.reward, self.done, self.info
