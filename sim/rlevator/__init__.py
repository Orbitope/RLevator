"""rlevator: simulacrum port of the Unity RLevator elevator-dispatch env.

Semantic port of Assets/ElevatorRL/Runtime/ (Building.cs and friends), S rung,
AS0 primitive actions, Midday @ intensity 1.0. See spec.md for the normative
rules; all constants here mirror spec.md's Constants section exactly.
"""

import os
from enum import IntEnum

# ---- rung selection (env-var; default S = the validated baseline) ----
# Only floors (F) and cars (E) differ across rungs; everything else is shared
# or derived. Mirrors the Unity scale-ladder presets (EXPERIMENT_PLAN.md §3).
_RUNGS = {
    "S": dict(F=8,  E=3),    # S_BuildingConfig.asset
    "M": dict(F=16, E=5),    # M_BuildingConfig.asset
    "L": dict(F=30, E=8),    # L_BuildingConfig.asset (future; needs VDN action factoring)
}
RUNG = os.environ.get("RLEVATOR_RUNG", "S").upper()
if RUNG not in _RUNGS:
    raise ValueError(f"RLEVATOR_RUNG={RUNG!r} not in {list(_RUNGS)}")

# ---- topology / timing ----
F = _RUNGS[RUNG]["F"]    # floors, 0 = lobby
E = _RUNGS[RUNG]["E"]    # cars
CAPACITY = 8            # riders per car
MAX_QUEUE = 12          # waiters per (floor, direction) queue
UNITS_PER_FLOOR = 16    # position sub-units per floor
MAX_POS = UNITS_PER_FLOOR * (F - 1)
TRAVEL_TICKS = 16       # 1.6 s per floor -> 1 unit per 0.1 s tick
DOOR_TICKS = 8          # 0.8 s doors opening OR closing
DWELL_TICKS = 12        # 1.2 s load/unload hold
SUBTICKS_PER_STEP = 5   # decision interval 0.5 s / 0.1 s tick
MAX_WAIT_TICKS = 450    # 45 s abandonment threshold
MAX_DECISIONS = 2048    # episode truncation, in steps
# deterministic car start floors, spread evenly (spec: Reset); [0,4,7] at S
START_FLOORS = [round(i * (F - 1) / (E - 1)) for i in range(E)]
K_MAX = 8               # Poisson count cap per (floor, sub-tick)

N_ACTIONS = 6 ** E      # packed base-6 per-car primitives

# ---- car state encoding (spec: State space) ----
IDLE, MOVING, DOORS_OPENING, DWELLING, DOORS_CLOSING = 0, 1, 2, 3, 4

# ---- traffic (TrafficConfig defaults + V2 baseline) ----
INTENSITY = 1.0
POPULATION = E * CAPACITY * 15   # loadPerSlot=15; 360 at S, 600 at M
BIN_SECONDS = 900.0     # 15-minute profile bins
MIDDAY_BINS = [12, 13, 14, 15, 16, 17, 30, 31, 32, 33, 34, 35]
MAX_SUBTICKS = MAX_DECISIONS * SUBTICKS_PER_STEP   # 10240 (table length)

# ---- reward coefficients (RewardConfig defaults, float64) ----
R_DELIVERED = 10.0
# toward/away movement shaping (±0.4). E10 ablation: RLEVATOR_SHAPING=off zeros
# it — the structured archs exploited this term for higher reward w/o better
# service, so test whether removing it aligns the objective with wait/abandon.
_SHAPING = os.environ.get("RLEVATOR_SHAPING", "on").lower() != "off"
R_TOWARD = 0.4 if _SHAPING else 0.0
R_AWAY = -0.4 if _SHAPING else 0.0
R_REJECTED = -5.0
R_ABANDONED = -8.0
R_IN_CAR = -0.04        # per rider-second
R_IN_QUEUE = -0.12      # per waiting-passenger-second

# ---- 48-bin arrival-rate day profiles (verbatim from PassengerArrivals.cs) ----
AR_IN = [1.8, 1.85, 2.8, 2.9, 3.8, 5.9, 7.3, 7.0, 7.1, 4.85, 3.9, 3.5,
         2.1, 1.6, 1.4, 1.7, 1.6, 1.55, 1.5, 1.45, 1.6, 1.5, 1.65, 1.7,
         2.6, 4.1, 3.8, 4.8, 4.0, 2.8, 1.4, 2.2, 2.1, 1.7, 1.3, 1.6,
         1.5, 1.4, 1.0, 1.1, 1.0, 1.1, 0.9, 0.6, 0.6, 0.6, 0.5, 0.4]

AR_INTER = [0.3, 0.6, 0.6, 0.55, 0.7, 1.2, 2.2, 2.5, 4.5, 3.2, 3.4, 3.95,
            3.5, 3.2, 3.0, 4.3, 3.3, 2.8, 4.0, 3.25, 4.2, 5.5, 5.3, 5.4,
            5.6, 4.6, 4.9, 5.8, 6.0, 3.6, 1.9, 1.9, 2.2, 2.3, 1.9, 2.5,
            2.6, 2.4, 2.3, 1.0, 1.3, 1.5, 1.3, 1.0, 0.7, 0.6, 0.6, 0.6]

AR_OUT = [0.4, 0.4, 0.25, 0.55, 0.4, 0.3, 0.6, 0.7, 0.7, 0.8, 1.3, 1.3,
          1.6, 1.5, 1.4, 1.0, 1.4, 1.8, 4.0, 4.8, 8.2, 5.7, 5.8, 4.2,
          3.3, 2.75, 2.4, 1.2, 1.2, 1.1, 1.4, 2.1, 1.5, 1.3, 1.8, 1.4,
          1.6, 2.7, 2.0, 2.5, 3.4, 6.4, 5.0, 4.6, 3.7, 3.9, 2.85, 2.0]


class Slots(IntEnum):
    """RNG slots — mirrors the table in spec.md."""
    ARRIVAL_COUNT = 0   # per step; index = s * F + f
    DEST = 1            # per step; index = (s * F + f) * K_MAX + k
    # 2, 3 reserved: FLEET_COUNT / FLEET_SHUFFLE (future randomizeActive)
