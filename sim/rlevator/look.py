"""Collective LOOK heuristic, ported from ElevatorHeuristics.CollectiveLook.

Operates on a schema-conformant state dict (so it can drive either
implementation via the harness ScriptedPolicy hook) and returns the packed
base-6 action integer. This is the project's classical baseline: its return
on identical seeds/traffic is the reference point every RL number is stated
against.
"""

from __future__ import annotations

from rlevator import CAPACITY, E, F, IDLE, MOVING, UNITS_PER_FLOOR


def _riders(state: dict, i: int) -> list[int]:
    out = []
    for dest in state["rider_dest"][i]:
        if dest == -1:
            break
        out.append(dest)
    return out


def _floor(state: dict, i: int) -> int:
    # Display floor, same rounding as the observation (spec: State space).
    return (state["pos"][i] + 8) // UNITS_PER_FLOOR


def collective_look(state: dict) -> int:
    """Mirrors ElevatorHeuristics.CollectiveLook pass for pass."""
    act = [0] * E
    target = [0] * E
    has_target = [False] * E

    # 1. cars with riders head to the nearest desired floor (prefer current
    #    direction: floors "ahead" beat floors behind).
    for i in range(E):
        riders = _riders(state, i)
        if not riders:
            continue
        fl = _floor(state, i)
        di = state["dir"][i]
        best = -1
        best_dist = None
        found_ahead = False
        for d in riders:
            ahead = d > fl if di > 0 else d < fl
            dist = abs(d - fl)
            if ahead:
                if not found_ahead or dist < best_dist:
                    best, best_dist, found_ahead = d, dist, True
            elif not found_ahead and (best_dist is None or dist < best_dist):
                best, best_dist = d, dist
        target[i] = best
        has_target[i] = best >= 0

    # 2. assign hall calls to nearest idle empty cars (greedy, no double-claim).
    claimed: set[int] = set()
    for i in range(E):
        if _riders(state, i):
            continue
        fl = _floor(state, i)
        best, bd = -1, None
        for f in range(F):  # v1: every car serves the full building
            if state["up_count"][f] == 0 and state["down_count"][f] == 0:
                continue
            if f in claimed:
                continue
            d = abs(f - fl)
            if bd is None or d < bd:
                bd, best = d, f
        if best >= 0:
            claimed.add(best)
            target[i] = best
            has_target[i] = True

    # 3. resolve each car into a primitive action.
    for i in range(E):
        if state["car_state"][i] != IDLE:  # AtFloor <=> Idle
            act[i] = 0
            continue
        f = _floor(state, i)
        riders = _riders(state, i)
        if f in riders:
            act[i] = 5  # unload
            continue
        rem = CAPACITY - len(riders)
        if rem > 0:
            if not riders:
                if state["up_count"][f] > 0 and (not has_target[i] or target[i] >= f):
                    act[i] = 3
                    continue
                if state["down_count"][f] > 0:
                    act[i] = 4
                    continue
                if state["up_count"][f] > 0:
                    act[i] = 3
                    continue
            else:
                if state["dir"][i] > 0 and state["up_count"][f] > 0:
                    act[i] = 3
                    continue
                if state["dir"][i] < 0 and state["down_count"][f] > 0:
                    act[i] = 4
                    continue
        if not has_target[i]:
            act[i] = 0
            continue
        if target[i] > f:
            act[i] = 1
            continue
        if target[i] < f:
            act[i] = 2
            continue
        if rem > 0 and state["up_count"][f] > 0:
            act[i] = 3
            continue
        if rem > 0 and state["down_count"][f] > 0:
            act[i] = 4
            continue
        act[i] = 0

    # pack base-6 (spec: Actions).
    packed = 0
    for i in range(E):
        packed += act[i] * (6 ** i)
    return packed
