"""
Emergency Resource Allocation - OpenEnv Environment (Server-side)
=================================================================
Implements the core Environment base class following the OpenEnv API:

    reset()        → EmergencyObservation
    step(action)   → StepResult  (observation, reward, done, info)
    state()        → EmergencyState

Grid World
----------
  • 10 × 10 city grid  (0-indexed, origin [0,0] = top-left)
  • 2 Ambulances       (start at centre [5, 5])
  • 5 random emergency requests spawned at reset()

Reward Logic
------------
  • +20   on reaching a CRITICAL request
  • +10   on reaching a NORMAL   request
  •  -1   per step (time penalty)
  • -50   when a CRITICAL request is not resolved within MAX_CRITICAL_AGE steps

Episode termination
-------------------
  • All requests have been resolved  OR
  • MAX_STEPS steps have elapsed
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_W          = 10          # grid width  (x-axis)
GRID_H          = 10          # grid height (y-axis)
NUM_AMBULANCES  = 2
NUM_REQUESTS    = 5
START_X         = 5           # ambulance spawn x
START_Y         = 5           # ambulance spawn y
MAX_STEPS       = 100         # hard episode limit
MAX_CRITICAL_AGE = 20         # steps before un-resolved CRITICAL request triggers -50

REWARD_CRITICAL = +20
REWARD_NORMAL   = +10
REWARD_STEP     =  -1
PENALTY_MISSED  = -50


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class EmergencyResourceEnv:
    """
    OpenEnv-compliant environment for Emergency Resource Allocation.

    Usage
    -----
    >>> env = EmergencyResourceEnv()
    >>> obs = env.reset()
    >>> result = env.step(EmergencyAction(ambulance_id=0, direction=Direction.UP))
    >>> obs, reward, done, info = result.as_tuple()
    >>> meta = env.state()
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, seed: Optional[int] = None):
        self._seed      = seed
        self._rng       = random.Random(seed)
        self._episode   = 0
        self._step      = 0
        self._total_reward    = 0.0
        self._resolved_count  = 0
        self._missed_critical = 0

        # Live objects (populated on reset)
        self._ambulances: List[AmbulanceState]   = []
        self._requests:   List[EmergencyRequest] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _spawn_requests(self) -> List[EmergencyRequest]:
        """Generate NUM_REQUESTS non-overlapping emergency requests."""
        occupied = {(START_X, START_Y)}   # ambulance spawn cells
        requests: List[EmergencyRequest] = []
        req_id = 0

        while len(requests) < NUM_REQUESTS:
            x = self._rng.randint(0, GRID_W - 1)
            y = self._rng.randint(0, GRID_H - 1)
            if (x, y) in occupied:
                continue
            occupied.add((x, y))

            # Roughly 40 % of requests are CRITICAL
            priority = (
                Priority.CRITICAL
                if self._rng.random() < 0.4
                else Priority.NORMAL
            )
            requests.append(EmergencyRequest(
                request_id=req_id,
                x=x,
                y=y,
                priority=priority,
            ))
            req_id += 1

        return requests

    def _build_observation(self) -> EmergencyObservation:
        return EmergencyObservation(
            ambulances       = [a.to_dict() for a in self._ambulances],
            pending_requests = [r.to_dict() for r in self._requests],
            grid_size        = (GRID_W, GRID_H),
            step_number      = self._step,
        )

    @staticmethod
    def _move(x: int, y: int, direction: Direction) -> Tuple[int, int]:
        """Apply one-step movement, clamped to grid boundaries."""
        dx, dy = {
            Direction.UP:    ( 0, -1),
            Direction.DOWN:  ( 0, +1),
            Direction.LEFT:  (-1,  0),
            Direction.RIGHT: (+1,  0),
        }[direction]
        return (
            max(0, min(GRID_W - 1, x + dx)),
            max(0, min(GRID_H - 1, y + dy)),
        )

    def _check_pickup(self, ambulance: AmbulanceState) -> float:
        """
        Check whether the ambulance is at the same cell as any pending
        request.  Resolves the request and returns the reward gained.
        """
        reward = 0.0
        to_remove = []
        for req in self._requests:
            if req.x == ambulance.x and req.y == ambulance.y:
                reward += (REWARD_CRITICAL if req.priority == Priority.CRITICAL
                           else REWARD_NORMAL)
                to_remove.append(req)
                self._resolved_count += 1

        for req in to_remove:
            self._requests.remove(req)

        return reward

    def _apply_critical_timeout_penalties(self) -> float:
        """
        For every CRITICAL request that has exceeded MAX_CRITICAL_AGE steps
        without being resolved, apply the -50 penalty and remove it.
        """
        penalty   = 0.0
        to_remove = []
        for req in self._requests:
            if req.priority == Priority.CRITICAL and req.age >= MAX_CRITICAL_AGE:
                penalty -= 50.0
                to_remove.append(req)
                self._missed_critical += 1

        for req in to_remove:
            self._requests.remove(req)

        return penalty

    # ------------------------------------------------------------------
    # Public OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> EmergencyObservation:
        """
        Clear the map, spawn new requests, and return the initial Observation.

        Returns
        -------
        EmergencyObservation
            Initial observation with both ambulances at [5, 5] and 5 fresh
            emergency requests at randomised grid positions.
        """
        self._episode        += 1
        self._step            = 0
        self._total_reward    = 0.0
        self._resolved_count  = 0
        self._missed_critical = 0

        # Spawn ambulances at the centre of the grid
        self._ambulances = [
            AmbulanceState(ambulance_id=i, x=START_X, y=START_Y)
            for i in range(NUM_AMBULANCES)
        ]

        # Spawn 5 random emergency requests
        self._requests = self._spawn_requests()

        return self._build_observation()

    def step(self, action: EmergencyAction | int) -> StepResult:
        """
        Execute one discrete action.

        Parameters
        ----------
        action : EmergencyAction | int
            Either an EmergencyAction dataclass **or** a flat integer in [0, 7]:
                action_int = ambulance_id × 4 + direction
            direction encoding: UP=0, DOWN=1, LEFT=2, RIGHT=3

        Returns
        -------
        StepResult
            Bundles (observation, reward, done, info).
            Also accessible via result.as_tuple().
        """
        if not self._ambulances:
            raise RuntimeError("Call reset() before step().")

        # Accept integer shorthand
        if isinstance(action, int):
            action = EmergencyAction.from_int(action)

        if not (0 <= action.ambulance_id < NUM_AMBULANCES):
            raise ValueError(
                f"ambulance_id must be 0 or 1; got {action.ambulance_id}"
            )

        # ── 1. Age all pending requests ──────────────────────────────────
        for req in self._requests:
            req.age += 1

        # ── 2. Move the chosen ambulance ─────────────────────────────────
        amb = self._ambulances[action.ambulance_id]
        amb.x, amb.y = self._move(amb.x, amb.y, action.direction)

        # ── 3. Step penalty ───────────────────────────────────────────────
        reward = float(REWARD_STEP)

        # ── 4. Check for pickups ──────────────────────────────────────────
        reward += self._check_pickup(amb)

        # ── 5. Apply critical-timeout penalties ───────────────────────────
        reward += self._apply_critical_timeout_penalties()

        # ── 6. Advance step counter ───────────────────────────────────────
        self._step        += 1
        self._total_reward += reward

        # ── 7. Termination check ─────────────────────────────────────────
        done = (not self._requests) or (self._step >= MAX_STEPS)

        info = {
            "step":             self._step,
            "total_reward":     self._total_reward,
            "resolved_total":   self._resolved_count,
            "missed_critical":  self._missed_critical,
            "requests_left":    len(self._requests),
            "action_taken":     action.to_dict(),
        }

        return StepResult(
            observation = self._build_observation(),
            reward      = reward,
            done        = done,
            info        = info,
        )

    def state(self) -> EmergencyState:
        """
        Return full episode metadata dictionary.

        Returns
        -------
        EmergencyState
            Contains episode_id, step_count, total_reward, ambulance positions,
            pending request list, and all other episode-level counters.
        """
        return EmergencyState(
            episode_id          = self._episode,
            step_count          = self._step,
            total_reward        = self._total_reward,
            resolved_count      = self._resolved_count,
            missed_critical     = self._missed_critical,
            ambulance_positions = [a.to_dict() for a in self._ambulances],
            pending_requests    = [r.to_dict() for r in self._requests],
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def action_space_size(self) -> int:
        """Total number of valid discrete actions."""
        return NUM_AMBULANCES * 4   # = 8

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (GRID_W, GRID_H)

    def render(self) -> str:
        """
        ASCII art render of the 10×10 grid.
        Symbols:
            A0, A1 = ambulances
            !!     = CRITICAL request
            ??     = NORMAL   request
            .      = empty cell
        """
        grid = [["." for _ in range(GRID_W)] for _ in range(GRID_H)]

        for req in self._requests:
            grid[req.y][req.x] = ("!!" if req.priority == Priority.CRITICAL
                                   else "??")

        for amb in self._ambulances:
            grid[amb.y][amb.x] = f"A{amb.ambulance_id}"

        header  = f"\n{'=' * 34}\n  City Grid  (Step {self._step})\n{'=' * 34}"
        rows    = "\n".join("  " + " ".join(f"{cell:>2}" for cell in row)
                            for row in grid)
        legend  = (
            "\n  Legend: A0/A1=Ambulance  !!=Critical  ??=Normal  .=Empty"
        )
        return header + "\n" + rows + legend + "\n"
