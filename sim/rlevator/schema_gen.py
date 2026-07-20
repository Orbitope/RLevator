"""Generate schema.json for the active rung (RLEVATOR_RUNG) from the constants.

The state schema encodes per-rung dimensions (E cars, F floors, bounds), so it
must track the rung. Run this before validating a non-S rung:

    RLEVATOR_RUNG=M python -m rlevator.schema_gen   # writes M-dimensioned schema.json
    python -m rlevator.schema_gen                    # restores the S default

Mirrors spec.md's state-space table; keep the two in sync.
"""
from __future__ import annotations

import json
from pathlib import Path

from rlevator import (
    CAPACITY, DOORS_CLOSING, DWELL_TICKS, E, F, MAX_DECISIONS, MAX_POS, MAX_QUEUE,
    MAX_WAIT_TICKS, N_ACTIONS, RUNG,
)


def _arr(n, items):
    return {"type": "array", "minItems": n, "maxItems": n, "items": items}


def _int(lo, hi):
    return {"type": "integer", "minimum": lo, "maximum": hi}


def build_schema() -> dict:
    car_i = _int(-1, F - 1)
    q_i = _int(-1, F - 1)
    state = {
        "type": "object",
        "description": (f"One rlevator state (rung {RUNG}: F={F} floors, E={E} cars, "
                        f"capacity {CAPACITY}, max queue {MAX_QUEUE}). All-integer state; "
                        "no x-atol. Mirrors spec.md's state-space table."),
        "properties": {
            "t": _int(0, MAX_DECISIONS),
            "pos": _arr(E, _int(0, MAX_POS)),
            "target": _arr(E, _int(0, F - 1)),
            "dir": _arr(E, {"type": "integer", "enum": [-1, 1]}),
            "car_state": _arr(E, _int(0, DOORS_CLOSING)),
            "timer": _arr(E, _int(0, DWELL_TICKS)),
            "pending": _arr(E, {"type": "integer", "enum": [0, 3, 4, 5]}),
            "rider_dest": _arr(E, _arr(CAPACITY, car_i)),
            "up_count": _arr(F, _int(0, MAX_QUEUE)),
            "down_count": _arr(F, _int(0, MAX_QUEUE)),
            "up_dest": _arr(F, _arr(MAX_QUEUE, q_i)),
            "down_dest": _arr(F, _arr(MAX_QUEUE, q_i)),
            "up_wait": _arr(F, _arr(MAX_QUEUE, _int(0, MAX_WAIT_TICKS))),
            "down_wait": _arr(F, _arr(MAX_QUEUE, _int(0, MAX_WAIT_TICKS))),
        },
        "required": ["t", "pos", "target", "dir", "car_state", "timer", "pending",
                     "rider_dest", "up_count", "down_count", "up_dest", "down_dest",
                     "up_wait", "down_wait"],
        "additionalProperties": False,
    }
    meta = {
        "type": "object",
        "properties": {
            "env": {"type": "string"}, "format_version": {"type": "string"},
            "simulacrum_version": {"type": "string"}, "seed": {"type": "integer"},
            "source": {"enum": ["reference", "batched"]},
            "instance": {"type": ["integer", "null"]},
            "git_sha": {"type": ["string", "null"]}, "created_at": {"type": "string"},
        },
        "required": ["env", "format_version", "seed", "source", "created_at"],
    }
    step = {
        "type": "object",
        "properties": {
            "t": {"type": "integer"}, "episode": {"type": "integer"},
            "state": {"$ref": "#/$defs/state"}, "action": {"$ref": "#/$defs/action"},
            "reward": {"type": "number"}, "terminated": {"type": "boolean"},
        },
        "required": ["t", "episode", "state", "action", "reward", "terminated"],
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "rlevator state and trajectory schemas",
        "$defs": {
            "state": state,
            "action": {
                "description": ("Packed base-6 per-car primitives: car i's command is "
                                "(a // 6**i) % 6; 0 NOOP, 1 up, 2 down, 3 board-up, "
                                "4 board-down, 5 unload."),
                "type": "integer", "minimum": 0, "maximum": N_ACTIONS - 1,
            },
            "trajectory": {
                "type": "object",
                "properties": {
                    "metadata": meta,
                    "initial": {
                        "type": "object",
                        "properties": {"episode": {"type": "integer"},
                                       "state": {"$ref": "#/$defs/state"}},
                        "required": ["episode", "state"],
                    },
                    "steps": {"type": "array", "items": step},
                },
                "required": ["metadata", "initial", "steps"],
            },
        },
    }


if __name__ == "__main__":
    path = Path(__file__).with_name("schema.json")
    path.write_text(json.dumps(build_schema(), indent=2) + "\n")
    print(f"wrote {path}  (rung {RUNG}: E={E}, F={F}, action<{N_ACTIONS})")
