"""
Emergency Resource Allocation - OpenEnv Package Init
=====================================================
Public API surface of the emergency_env package.
Mirrors the canonical OpenEnv package structure:

    from emergency_env import (
        EmergencyResourceEnv,   # main environment class
        EmergencyAction,        # action dataclass / int encoder
        EmergencyObservation,   # observation dataclass
        EmergencyState,         # state / episode-metadata dataclass
        StepResult,             # (obs, reward, done, info) bundle
        Direction,              # movement enum  UP/DOWN/LEFT/RIGHT
        Priority,               # request priority enum  NORMAL/CRITICAL
    )
"""

from .environment import EmergencyResourceEnv
from .models import (
    AmbulanceState,
    Direction,
    EmergencyAction,
    EmergencyObservation,
    EmergencyRequest,
    EmergencyState,
    Priority,
    StepResult,
)

__all__ = [
    "EmergencyResourceEnv",
    "EmergencyAction",
    "EmergencyObservation",
    "EmergencyState",
    "AmbulanceState",
    "EmergencyRequest",
    "StepResult",
    "Direction",
    "Priority",
]
