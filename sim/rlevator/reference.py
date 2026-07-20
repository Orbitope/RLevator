"""Readable single-instance reference implementation of rlevator.

Written from spec.md ONLY. Style: dataclass state, explicit ifs/fors, no
vectorization, no premature abstraction, every rule traceable to a spec line.
All randomness via simulacrum.rng scalar draws with slots from rlevator.Slots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from simulacrum import ReferenceEnv, rng

from rlevator import (
    AR_IN, AR_INTER, AR_OUT, BIN_SECONDS, CAPACITY, DOOR_TICKS, DWELL_TICKS,
    DOORS_CLOSING, DOORS_OPENING, DWELLING, E, F, IDLE, INTENSITY, K_MAX,
    MAX_DECISIONS, MAX_POS, MAX_QUEUE, MAX_SUBTICKS, MAX_WAIT_TICKS,
    MIDDAY_BINS, MOVING, POPULATION, R_ABANDONED, R_AWAY, R_DELIVERED,
    R_IN_CAR, R_IN_QUEUE, R_REJECTED, R_TOWARD, START_FLOORS,
    SUBTICKS_PER_STEP, UNITS_PER_FLOOR, Slots,
)


# ---------------------------------------------------------------------------
# spec: The traffic model (deterministic part) — precomputed at init with
# scalar Python float64 arithmetic (never torch ops).
# ---------------------------------------------------------------------------


def _interp_rate(table: list[float], sim: float) -> float:
    # spec: Traffic model — InterpAr then RatePerSecond, exactly as written.
    period = len(MIDDAY_BINS) * BIN_SECONDS
    tmod = sim % period
    idx = min(int(tmod / BIN_SECONDS), len(MIDDAY_BINS) - 1)
    shift = tmod - idx * BIN_SECONDS
    nxt = (idx + 1) % len(MIDDAY_BINS)
    a = table[MIDDAY_BINS[idx]]
    b = table[MIDDAY_BINS[nxt]]
    ar = a + shift * (b - a) / BIN_SECONDS
    return (ar * POPULATION) / 30000.0


def _build_traffic_tables():
    """Per absolute sub-tick u: lam_tick[u][f], Poisson cdf[u][f][k], dest
    cdf dcdf[u][o][d]. spec: Traffic model."""
    lam_tick: list[list[float]] = []
    pcdf: list[list[list[float]]] = []
    dcdf: list[list[list[float]]] = []
    for u in range(MAX_SUBTICKS):
        sim = u * 0.1
        rate_in = _interp_rate(AR_IN, sim)
        rate_inter = _interp_rate(AR_INTER, sim)
        rate_out = _interp_rate(AR_OUT, sim)

        # spec: per-floor origination rates.
        lambda_sec = [0.0] * F
        lambda_sec[0] = rate_in
        per_upper = (rate_out + rate_inter) / (F - 1)
        for f in range(1, F):
            lambda_sec[f] = per_upper

        # spec: destination rows.
        p_lobby = rate_out / (rate_out + rate_inter)
        dest_row = [[0.0] * F for _ in range(F)]
        for d in range(1, F):
            dest_row[0][d] = 1.0 / (F - 1)
        for o in range(1, F):
            dest_row[o][0] = p_lobby
            each = (1.0 - p_lobby) / (F - 2)
            for d in range(1, F):
                if d != o:
                    dest_row[o][d] = each

        # spec: lam_tick[u][f] = (lambda_sec[f] * INTENSITY) * 0.1.
        lam_u = [(lambda_sec[f] * INTENSITY) * 0.1 for f in range(F)]
        lam_tick.append(lam_u)

        # spec: Poisson CDF thresholds per (u, f).
        pcdf_u = []
        for f in range(F):
            lam = lam_u[f]
            term = math.exp(-lam)
            cdf = [term]
            for k in range(1, K_MAX):
                term = term * lam / k
                cdf.append(cdf[k - 1] + term)
            pcdf_u.append(cdf)
        pcdf.append(pcdf_u)

        # spec: destination CDF rows (sequential accumulation in d order).
        dcdf_u = []
        for o in range(F):
            c = 0.0
            row = []
            for d in range(F):
                c = c + dest_row[o][d]
                row.append(c)
            dcdf_u.append(row)
        dcdf.append(dcdf_u)
    return lam_tick, pcdf, dcdf


_TABLES = None


def _traffic_tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = _build_traffic_tables()
    return _TABLES


def _sign(x: int) -> int:
    # spec: Step (d) — sign(0) = +1, mirroring Mathf.Sign.
    return 1 if x >= 0 else -1


@dataclass
class State:
    t: int                              # spec: State space — step counter (RNG key)
    pos: list[int]                      # [E], sub-floor units
    target: list[int]                   # [E]
    dir: list[int]                      # [E], -1 / +1
    car_state: list[int]                # [E], 0..4
    timer: list[int]                    # [E], ticks
    pending: list[int]                  # [E], 0/3/4/5
    rider_dest: list[list[int]]         # [E][CAPACITY], -1 = empty, compact prefix
    up_count: list[int]                 # [F]
    down_count: list[int]               # [F]
    up_dest: list[list[int]]            # [F][MAX_QUEUE], FIFO front at 0
    down_dest: list[list[int]]          # [F][MAX_QUEUE]
    up_wait: list[list[int]]            # [F][MAX_QUEUE]
    down_wait: list[list[int]]          # [F][MAX_QUEUE]


class RlevatorReference(ReferenceEnv):
    def __init__(self) -> None:
        self.lam_tick, self.pcdf, self.dcdf = _traffic_tables()

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int, episode: int = 0) -> State:
        self.seed_episode(seed, episode)
        # spec: Reset — fully deterministic, no RNG slots consumed.
        self.state = State(
            t=0,
            pos=[UNITS_PER_FLOOR * START_FLOORS[i] for i in range(E)],
            target=[START_FLOORS[i] for i in range(E)],
            dir=[1] * E,
            car_state=[IDLE] * E,
            timer=[0] * E,
            pending=[0] * E,
            rider_dest=[[-1] * CAPACITY for _ in range(E)],
            up_count=[0] * F,
            down_count=[0] * F,
            up_dest=[[-1] * MAX_QUEUE for _ in range(F)],
            down_dest=[[-1] * MAX_QUEUE for _ in range(F)],
            up_wait=[[0] * MAX_QUEUE for _ in range(F)],
            down_wait=[[0] * MAX_QUEUE for _ in range(F)],
        )
        return self.state

    # ------------------------------------------------------------------ step
    def step(self, action: int) -> tuple[State, float, bool, dict]:
        st = self.state
        t_pre = st.t

        # spec: Rewards — seven integer accumulators, zeroed each step.
        delivered = 0
        rejected = 0
        abandoned = 0
        toward_units = 0
        away_units = 0
        rider_ticks = 0
        queue_ticks = 0
        self._boarded_waits: list[int] = []  # metrics only (see info below)

        # spec: Step 1 — decode and apply each car's command in order.
        for i in range(E):
            cmd = (action // (6 ** i)) % 6
            self._apply_command(st, i, cmd)

        # spec: Step 2 — five sub-ticks.
        for s in range(SUBTICKS_PER_STEP):
            u = SUBTICKS_PER_STEP * t_pre + s

            # spec: Step 2b — expire waiters with wait >= MAX_WAIT_TICKS.
            for f in range(F):
                abandoned += self._expire_queue(st, f, up=True)
                abandoned += self._expire_queue(st, f, up=False)

            # spec: Step 2c — spawn arrivals.
            for f in range(F):
                x = rng.draw_uniform(self.key, t_pre, Slots.ARRIVAL_COUNT,
                                     index=s * F + f)
                cdf = self.pcdf[u][f]
                n = K_MAX
                for k in range(K_MAX):
                    if x < cdf[k]:
                        n = k
                        break
                for k in range(n):
                    r = rng.draw_uniform(self.key, t_pre, Slots.DEST,
                                         index=(s * F + f) * K_MAX + k)
                    dcdf = self.dcdf[u][f]
                    d = -1
                    for cand in range(F):
                        if r <= dcdf[cand]:
                            d = cand
                            break
                    if d == -1:
                        # spec: float-residue fallback.
                        d = 1 if f == 0 else 0
                    if d == f:
                        # spec: same-floor arrival is discarded.
                        continue
                    if d > f:
                        if st.up_count[f] >= MAX_QUEUE:
                            rejected += 1  # spec: rejected, nothing enqueued
                        else:
                            st.up_dest[f][st.up_count[f]] = d
                            st.up_wait[f][st.up_count[f]] = 0
                            st.up_count[f] += 1
                    else:
                        if st.down_count[f] >= MAX_QUEUE:
                            rejected += 1
                        else:
                            st.down_dest[f][st.down_count[f]] = d
                            st.down_wait[f][st.down_count[f]] = 0
                            st.down_count[f] += 1

            # spec: Step 2d — advance cars.
            for i in range(E):
                cs = st.car_state[i]
                if cs == MOVING:
                    pos_old = st.pos[i]
                    st.pos[i] = pos_old + st.dir[i]
                    for slot in range(CAPACITY):
                        dest = st.rider_dest[i][slot]
                        if dest == -1:
                            break  # compact prefix — nothing after the first -1
                        to_dest = UNITS_PER_FLOOR * dest - pos_old
                        if _sign(to_dest) == st.dir[i]:
                            toward_units += 1
                        else:
                            away_units += 1
                    if st.pos[i] == UNITS_PER_FLOOR * st.target[i]:
                        st.car_state[i] = IDLE
                elif cs == DOORS_OPENING:
                    st.timer[i] -= 1
                    if st.timer[i] == 0:
                        delivered += self._transfer(st, i)
                        st.car_state[i] = DWELLING
                        st.timer[i] = DWELL_TICKS
                elif cs == DWELLING:
                    st.timer[i] -= 1
                    if st.timer[i] == 0:
                        st.car_state[i] = DOORS_CLOSING
                        st.timer[i] = DOOR_TICKS
                elif cs == DOORS_CLOSING:
                    st.timer[i] -= 1
                    if st.timer[i] == 0:
                        st.car_state[i] = IDLE
                        st.pending[i] = 0
                        st.timer[i] = 0
                # Idle: nothing.

            # spec: Step 2e — age queues, accumulate occupancy ticks.
            waiting = 0
            for f in range(F):
                for j in range(st.up_count[f]):
                    st.up_wait[f][j] += 1
                waiting += st.up_count[f]
                for j in range(st.down_count[f]):
                    st.down_wait[f][j] += 1
                waiting += st.down_count[f]
            queue_ticks += waiting
            in_car = 0
            for i in range(E):
                in_car += self._rider_count(st, i)
            rider_ticks += in_car

        # spec: Step 3 — increment the decision counter.
        st.t = t_pre + 1

        # spec: Rewards — pinned float64 evaluation order.
        r_del = R_DELIVERED * delivered
        r_tow = R_TOWARD * (toward_units * 0.0625)
        r_awy = R_AWAY * (away_units * 0.0625)
        r_rej = R_REJECTED * rejected
        r_abn = R_ABANDONED * abandoned
        r_ine = R_IN_CAR * (rider_ticks * 0.1)
        r_inq = R_IN_QUEUE * (queue_ticks * 0.1)
        reward = r_del + r_tow + r_awy + r_rej + r_abn + r_ine + r_inq

        # spec: Termination — step cap only.
        terminated = st.t == MAX_DECISIONS
        # info: observability-only metrics (not state, not compared by the
        # differential test) — used by eval scripts to mirror Unity's
        # delivered/wait bookkeeping.
        info = {
            "delivered": delivered,
            "rejected": rejected,
            "abandoned": abandoned,
            "boarded_waits": self._boarded_waits,
        }
        return st, reward, terminated, info

    # ---------------------------------------------------------------- helpers
    def _apply_command(self, st: State, i: int, cmd: int) -> None:
        # spec: Actions — silently ignored unless the car is Idle.
        if st.car_state[i] != IDLE:
            return
        f = st.pos[i] // UNITS_PER_FLOOR
        if cmd == 1:
            # spec: Actions — up one floor; no-op at the top floor.
            if f < F - 1:
                st.target[i] = f + 1
                st.dir[i] = 1
                st.car_state[i] = MOVING
        elif cmd == 2:
            # spec: Actions — down one floor; no-op at the lobby.
            if f > 0:
                st.target[i] = f - 1
                st.dir[i] = -1
                st.car_state[i] = MOVING
        elif cmd in (3, 4, 5):
            # spec: Actions — open doors with the pending command.
            st.pending[i] = cmd
            st.car_state[i] = DOORS_OPENING
            st.timer[i] = DOOR_TICKS
        # cmd 0: NOOP.

    def _expire_queue(self, st: State, f: int, up: bool) -> int:
        # spec: Step 2b — remove every waiter with wait >= MAX_WAIT_TICKS,
        # compact preserving order.
        dest = st.up_dest[f] if up else st.down_dest[f]
        wait = st.up_wait[f] if up else st.down_wait[f]
        count = st.up_count[f] if up else st.down_count[f]
        kept_dest = []
        kept_wait = []
        removed = 0
        for j in range(count):
            if wait[j] >= MAX_WAIT_TICKS:
                removed += 1
            else:
                kept_dest.append(dest[j])
                kept_wait.append(wait[j])
        if removed:
            for j in range(MAX_QUEUE):
                if j < len(kept_dest):
                    dest[j] = kept_dest[j]
                    wait[j] = kept_wait[j]
                else:
                    dest[j] = -1
                    wait[j] = 0
            if up:
                st.up_count[f] = len(kept_dest)
            else:
                st.down_count[f] = len(kept_dest)
        return removed

    def _rider_count(self, st: State, i: int) -> int:
        n = 0
        for slot in range(CAPACITY):
            if st.rider_dest[i][slot] == -1:
                break
            n += 1
        return n

    def _transfer(self, st: State, i: int) -> int:
        # spec: Step 2d Transfer — keyed on pending, at floor pos // 16.
        f = st.pos[i] // UNITS_PER_FLOOR
        delivered = 0
        if st.pending[i] == 5:
            # unload: remove riders with dest == f, compact preserving order.
            kept = []
            for slot in range(CAPACITY):
                dest = st.rider_dest[i][slot]
                if dest == -1:
                    break
                if dest == f:
                    delivered += 1
                else:
                    kept.append(dest)
            for slot in range(CAPACITY):
                st.rider_dest[i][slot] = kept[slot] if slot < len(kept) else -1
        elif st.pending[i] == 3:
            # board up-queue: FIFO front, greedy to capacity; dir = +1 always.
            n = self._rider_count(st, i)
            while n < CAPACITY and st.up_count[f] > 0:
                st.rider_dest[i][n] = st.up_dest[f][0]
                self._boarded_waits.append(st.up_wait[f][0])  # metrics only
                n += 1
                self._pop_front(st, f, up=True)
            st.dir[i] = 1
        elif st.pending[i] == 4:
            # board down-queue.
            n = self._rider_count(st, i)
            while n < CAPACITY and st.down_count[f] > 0:
                st.rider_dest[i][n] = st.down_dest[f][0]
                self._boarded_waits.append(st.down_wait[f][0])  # metrics only
                n += 1
                self._pop_front(st, f, up=False)
            st.dir[i] = -1
        return delivered

    def _pop_front(self, st: State, f: int, up: bool) -> None:
        # spec: Step 2d Transfer — pop the FIFO front, shifting the rest.
        dest = st.up_dest[f] if up else st.down_dest[f]
        wait = st.up_wait[f] if up else st.down_wait[f]
        count = st.up_count[f] if up else st.down_count[f]
        for j in range(count - 1):
            dest[j] = dest[j + 1]
            wait[j] = wait[j + 1]
        dest[count - 1] = -1
        wait[count - 1] = 0
        if up:
            st.up_count[f] = count - 1
        else:
            st.down_count[f] = count - 1

    # ------------------------------------------------------------ observation
    def observe(self, state: State) -> np.ndarray:
        # spec: Observations — float32[98], exact block order; every ratio is
        # float32(int) / float32(constant).
        out: list[np.float32] = []
        one = np.float32(1.0)
        zero = np.float32(0.0)

        # block 1: carFloor — one-hot(F) of display floor (pos + 8) // 16.
        for i in range(E):
            fl = (state.pos[i] + 8) // UNITS_PER_FLOOR
            for f in range(F):
                out.append(one if f == fl else zero)

        # block 2: carActive — all cars in service in v1.
        for i in range(E):
            out.append(one)

        # block 3: carButtons — any rider wants floor f.
        for i in range(E):
            for f in range(F):
                wants = False
                for slot in range(CAPACITY):
                    dest = state.rider_dest[i][slot]
                    if dest == -1:
                        break
                    if dest == f:
                        wants = True
                        break
                out.append(one if wants else zero)

        # block 4: hallButtons — up then down, interleaved per floor.
        for f in range(F):
            out.append(one if state.up_count[f] > 0 else zero)
            out.append(one if state.down_count[f] > 0 else zero)

        # block 5: hallCallAge — front waiter's wait / 450, up then down.
        for f in range(F):
            if state.up_count[f] > 0:
                w = min(state.up_wait[f][0], MAX_WAIT_TICKS)
                out.append(np.float32(w) / np.float32(MAX_WAIT_TICKS))
            else:
                out.append(zero)
            if state.down_count[f] > 0:
                w = min(state.down_wait[f][0], MAX_WAIT_TICKS)
                out.append(np.float32(w) / np.float32(MAX_WAIT_TICKS))
            else:
                out.append(zero)

        # block 6: carMotion — one-hot(3) of m, then pos / 112.
        for i in range(E):
            if state.car_state[i] == MOVING:
                m = 2 if state.dir[i] > 0 else 0
            else:
                m = 1
            for v in range(3):
                out.append(one if v == m else zero)
            out.append(np.float32(state.pos[i]) / np.float32(MAX_POS))

        # block 7: carLoads — rider count / capacity.
        for i in range(E):
            out.append(np.float32(self._rider_count(state, i)) / np.float32(CAPACITY))

        return np.array(out, dtype=np.float32)

    # -------------------------------------------------------------- serialize
    def to_json(self, state: State) -> dict:
        return {
            "t": int(state.t),
            "pos": [int(v) for v in state.pos],
            "target": [int(v) for v in state.target],
            "dir": [int(v) for v in state.dir],
            "car_state": [int(v) for v in state.car_state],
            "timer": [int(v) for v in state.timer],
            "pending": [int(v) for v in state.pending],
            "rider_dest": [[int(v) for v in row] for row in state.rider_dest],
            "up_count": [int(v) for v in state.up_count],
            "down_count": [int(v) for v in state.down_count],
            "up_dest": [[int(v) for v in row] for row in state.up_dest],
            "down_dest": [[int(v) for v in row] for row in state.down_dest],
            "up_wait": [[int(v) for v in row] for row in state.up_wait],
            "down_wait": [[int(v) for v in row] for row in state.down_wait],
        }

    def from_json(self, obj: dict) -> State:
        return State(
            t=int(obj["t"]),
            pos=[int(v) for v in obj["pos"]],
            target=[int(v) for v in obj["target"]],
            dir=[int(v) for v in obj["dir"]],
            car_state=[int(v) for v in obj["car_state"]],
            timer=[int(v) for v in obj["timer"]],
            pending=[int(v) for v in obj["pending"]],
            rider_dest=[[int(v) for v in row] for row in obj["rider_dest"]],
            up_count=[int(v) for v in obj["up_count"]],
            down_count=[int(v) for v in obj["down_count"]],
            up_dest=[[int(v) for v in row] for row in obj["up_dest"]],
            down_dest=[[int(v) for v in row] for row in obj["down_dest"]],
            up_wait=[[int(v) for v in row] for row in obj["up_wait"]],
            down_wait=[[int(v) for v in row] for row in obj["down_wait"]],
        )
