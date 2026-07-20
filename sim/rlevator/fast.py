"""Batched tensor implementation of rlevator.

Written from spec.md ONLY — not from reference.py. State: int64 tensors with
leading [N] dim. Branching -> torch.where masking; the only Python loops are
over spec constants (SUBTICKS_PER_STEP, K_MAX, E, the two queue directions),
which is compile-safe. Cars transfer sequentially (car order matters when two
cars board the same queue in the same sub-tick — spec: Step 2d).
"""

from __future__ import annotations

import math

import torch

from simulacrum import BatchedEnv, invariant, rng

from rlevator import (
    AR_IN, AR_INTER, AR_OUT, BIN_SECONDS, CAPACITY, CAR_MAX_FLOOR, CAR_MIN_FLOOR,
    DOOR_TICKS, DWELL_TICKS,
    DOORS_CLOSING, DOORS_OPENING, DWELLING, E, F, IDLE, INTENSITY, K_MAX,
    MAX_DECISIONS, MAX_POS, MAX_QUEUE, MAX_SUBTICKS, MAX_WAIT_TICKS,
    MIDDAY_BINS, MOVING, POPULATION, R_ABANDONED, R_AWAY, R_DELIVERED,
    R_IN_CAR, R_IN_QUEUE, R_REJECTED, R_TOWARD, START_FLOORS,
    SUBTICKS_PER_STEP, UNITS_PER_FLOOR, Slots,
)

_UPF = UNITS_PER_FLOOR

# Eval wait-histogram: 64 bins over [0, MAX_WAIT_TICKS], mirroring Unity's
# WaitHistogram (Stats/WaitHistogram.cs) so waitP95 matches bin-for-bin.
_WAIT_BINS = 64


def _build_tables(max_subticks: int = MAX_SUBTICKS) -> tuple[torch.Tensor, torch.Tensor]:
    """spec: Traffic model — Poisson CDF thresholds pcdf[u, f, k] and dest
    CDF rows dcdf[u, o, d], precomputed with scalar Python float64 math
    (never torch ops; torch.exp is not bit-identical to libm).

    ``max_subticks`` defaults to the training horizon (MAX_SUBTICKS = one
    2048-decision episode). Eval passes a longer span (the rate model is
    periodic in ``u``, so extending it is exact, not extrapolation) — see
    RlevatorBatched(max_decisions=...)."""
    nbins = len(MIDDAY_BINS)
    period = nbins * BIN_SECONDS
    pcdf_flat: list[float] = []
    dcdf_flat: list[float] = []
    for u in range(max_subticks):
        sim = u * 0.1
        # spec: InterpAr then RatePerSecond, per component table.
        rates = []
        for table in (AR_IN, AR_INTER, AR_OUT):
            tmod = sim % period
            idx = min(int(tmod / BIN_SECONDS), nbins - 1)
            shift = tmod - idx * BIN_SECONDS
            nxt = (idx + 1) % nbins
            a = table[MIDDAY_BINS[idx]]
            b = table[MIDDAY_BINS[nxt]]
            ar = a + shift * (b - a) / BIN_SECONDS
            rates.append((ar * POPULATION) / 30000.0)
        rate_in, rate_inter, rate_out = rates

        # spec: per-floor origination rates and destination rows.
        per_upper = (rate_out + rate_inter) / (F - 1)
        lambda_sec = [rate_in] + [per_upper] * (F - 1)
        p_lobby = rate_out / (rate_out + rate_inter)
        each = (1.0 - p_lobby) / (F - 2)

        for f in range(F):
            # spec: lam_tick = (lambda_sec[f] * INTENSITY) * 0.1.
            lam = (lambda_sec[f] * INTENSITY) * 0.1
            term = math.exp(-lam)
            c = term
            pcdf_flat.append(c)
            for k in range(1, K_MAX):
                term = term * lam / k
                c = c + term
                pcdf_flat.append(c)

        for o in range(F):
            if o == 0:
                row = [0.0] + [1.0 / (F - 1)] * (F - 1)
            else:
                row = [p_lobby] + [0.0 if d == o else each for d in range(1, F)]
            c = 0.0
            for d in range(F):
                c = c + row[d]
                dcdf_flat.append(c)

    pcdf = torch.tensor(pcdf_flat, dtype=torch.float64).view(max_subticks, F, K_MAX)
    dcdf = torch.tensor(dcdf_flat, dtype=torch.float64).view(max_subticks, F, F)
    return pcdf, dcdf


_TABLES: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _tables(max_subticks: int = MAX_SUBTICKS) -> tuple[torch.Tensor, torch.Tensor]:
    cached = _TABLES.get(max_subticks)
    if cached is None:
        cached = _TABLES[max_subticks] = _build_tables(max_subticks)
    return cached


def _compact_queue(dest: torch.Tensor, wait: torch.Tensor, count: torch.Tensor,
                   keep: torch.Tensor, aQ: torch.Tensor):
    """Remove non-kept occupied slots, preserving order (stable argsort);
    vacated slots become dest=-1, wait=0. keep must be a subset of occupied."""
    order = torch.argsort((~keep).to(torch.int8), dim=-1, stable=True)
    new_count = keep.sum(-1)
    live = aQ.view(1, 1, -1) < new_count.unsqueeze(-1)
    new_dest = torch.where(live, dest.gather(-1, order), torch.full_like(dest, -1))
    new_wait = torch.where(live, wait.gather(-1, order), torch.zeros_like(wait))
    return new_dest, new_wait, new_count


class RlevatorBatched(BatchedEnv):
    def __init__(self, n: int, max_decisions: int | None = None,
                 collect_metrics: bool = False, **kw) -> None:
        super().__init__(n, **kw)
        dev = self.device
        # Horizon: defaults to the training episode length (spec: MAX_DECISIONS);
        # eval passes a longer span (e.g. 7200 = Unity's 3600 s / 0.5 s protocol).
        self._max_decisions = MAX_DECISIONS if max_decisions is None else int(max_decisions)
        # Opt-in eval instrumentation. OFF by default: the training/differential
        # path (and bench.py) is bit-for-bit unchanged when collect_metrics=False.
        self._collect = bool(collect_metrics)
        pcdf, dcdf = _tables(self._max_decisions * SUBTICKS_PER_STEP)
        self.pcdf = pcdf.to(dev)
        self.dcdf = dcdf.to(dev)
        self._start_pos = (torch.tensor(START_FLOORS, dtype=torch.int64, device=dev)
                           * _UPF).unsqueeze(0)                       # [1,E]
        # per-car service bands (spec: zoning); full-building for non-zoned rungs.
        self._car_max = torch.tensor(CAR_MAX_FLOOR, dtype=torch.int64, device=dev).view(1, E)
        self._car_min = torch.tensor(CAR_MIN_FLOOR, dtype=torch.int64, device=dev).view(1, E)
        self._pow6 = torch.tensor([6 ** i for i in range(E)],
                                  dtype=torch.int64, device=dev)      # [E]
        self._aE = torch.arange(E, dtype=torch.int64, device=dev)
        self._aF = torch.arange(F, dtype=torch.int64, device=dev)
        self._aQ = torch.arange(MAX_QUEUE, dtype=torch.int64, device=dev)
        self._aC = torch.arange(CAPACITY, dtype=torch.int64, device=dev)
        # Draw-index tensors per sub-tick (spec: RNG slots).
        self._idx_arrival = [
            (s * F + self._aF).view(1, F) for s in range(SUBTICKS_PER_STEP)
        ]
        self._idx_dest = [
            ((s * F + self._aF).view(F, 1) * K_MAX
             + torch.arange(K_MAX, dtype=torch.int64, device=dev).view(1, K_MAX)
             ).view(1, F, K_MAX)
            for s in range(SUBTICKS_PER_STEP)
        ]
        # spec: float-residue fallback destination — 1 from the lobby, else 0.
        self._dest_fallback = torch.where(
            self._aF == 0,
            torch.ones_like(self._aF),
            torch.zeros_like(self._aF)).view(1, F, 1)

    # ------------------------------------------------------------------ reset
    def _reset_instances(self, mask: torch.Tensor) -> None:
        n, dev = self.n, self.device
        if not hasattr(self, "pos"):
            z_e = torch.zeros(n, E, dtype=torch.int64, device=dev)
            self.pos = z_e.clone()
            self.target = z_e.clone()
            self.dir = z_e.clone()
            self.car_state = z_e.clone()
            self.timer = z_e.clone()
            self.pending = z_e.clone()
            self.rider_dest = torch.zeros(n, E, CAPACITY, dtype=torch.int64, device=dev)
            self.up_count = torch.zeros(n, F, dtype=torch.int64, device=dev)
            self.down_count = torch.zeros(n, F, dtype=torch.int64, device=dev)
            self.up_dest = torch.zeros(n, F, MAX_QUEUE, dtype=torch.int64, device=dev)
            self.down_dest = torch.zeros(n, F, MAX_QUEUE, dtype=torch.int64, device=dev)
            self.up_wait = torch.zeros(n, F, MAX_QUEUE, dtype=torch.int64, device=dev)
            self.down_wait = torch.zeros(n, F, MAX_QUEUE, dtype=torch.int64, device=dev)
            if self._collect:
                # Metrics-only (never in schema/state; not differential-tested):
                # rider_wait carries each boarded rider's frozen queue-wait ticks
                # (parallel to rider_dest) so wait can be tallied at DELIVERY, the
                # way Unity's StatsCollector does. Accumulators below are zeroed at
                # the warmup boundary via reset_metrics().
                self.rider_wait = torch.zeros(n, E, CAPACITY, dtype=torch.int64, device=dev)
                self._m_delivered = torch.zeros(n, dtype=torch.int64, device=dev)
                self._m_abandoned = torch.zeros(n, dtype=torch.int64, device=dev)
                self._m_rejected = torch.zeros(n, dtype=torch.int64, device=dev)
                self._m_wsum = torch.zeros(n, dtype=torch.int64, device=dev)   # wait ticks
                self._m_wcount = torch.zeros(n, dtype=torch.int64, device=dev)
                self._m_wpeak = torch.zeros(n, dtype=torch.int64, device=dev)
                self._m_whist = torch.zeros(n, _WAIT_BINS, dtype=torch.int64, device=dev)

        m1 = mask.unsqueeze(-1)                    # [N,1] vs [N,E] / [N,F]
        m2 = mask.view(-1, 1, 1)                   # vs [N,*,*]
        zero = torch.zeros((), dtype=torch.int64, device=dev)
        one = torch.ones((), dtype=torch.int64, device=dev)
        neg1 = -one

        # spec: Reset — deterministic; no RNG slots consumed.
        self.pos = torch.where(m1, self._start_pos, self.pos)
        self.target = torch.where(m1, self._start_pos // _UPF, self.target)
        self.dir = torch.where(m1, one, self.dir)
        self.car_state = torch.where(m1, zero, self.car_state)   # Idle
        self.timer = torch.where(m1, zero, self.timer)
        self.pending = torch.where(m1, zero, self.pending)
        self.rider_dest = torch.where(m2, neg1, self.rider_dest)
        self.up_count = torch.where(m1, zero, self.up_count)
        self.down_count = torch.where(m1, zero, self.down_count)
        self.up_dest = torch.where(m2, neg1, self.up_dest)
        self.down_dest = torch.where(m2, neg1, self.down_dest)
        self.up_wait = torch.where(m2, zero, self.up_wait)
        self.down_wait = torch.where(m2, zero, self.down_wait)
        if self._collect:
            self.rider_wait = torch.where(m2, zero, self.rider_wait)

    # ------------------------------------------------------------------ step
    def _step_impl(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dev = self.device
        t_pre = self.t
        i64 = torch.int64

        # spec: Rewards — seven integer accumulators, zeroed each step.
        delivered = torch.zeros(self.n, dtype=i64, device=dev)
        rejected = torch.zeros_like(delivered)
        abandoned = torch.zeros_like(delivered)
        toward_units = torch.zeros_like(delivered)
        away_units = torch.zeros_like(delivered)
        rider_ticks = torch.zeros_like(delivered)
        queue_ticks = torch.zeros_like(delivered)

        # spec: Step 1 — decode base-6 digits and apply per-car commands.
        cmds = (actions.unsqueeze(-1) // self._pow6) % 6         # [N,E]
        idle = self.car_state == IDLE
        fl = self.pos // _UPF
        # spec: Actions — up/down bounded by each car's service band (zoning).
        up_ok = idle & (cmds == 1) & (fl < self._car_max)
        dn_ok = idle & (cmds == 2) & (fl > self._car_min)
        door = idle & (cmds >= 3)
        self.target = torch.where(up_ok, fl + 1, torch.where(dn_ok, fl - 1, self.target))
        self.dir = torch.where(up_ok, torch.ones_like(self.dir),
                               torch.where(dn_ok, -torch.ones_like(self.dir), self.dir))
        self.car_state = torch.where(
            up_ok | dn_ok, torch.full_like(self.car_state, MOVING),
            torch.where(door, torch.full_like(self.car_state, DOORS_OPENING),
                        self.car_state))
        self.pending = torch.where(door, cmds, self.pending)
        self.timer = torch.where(door, torch.full_like(self.timer, DOOR_TICKS),
                                 self.timer)

        # spec: Step 2 — five sub-ticks.
        for s in range(SUBTICKS_PER_STEP):
            u = SUBTICKS_PER_STEP * t_pre + s                    # [N]
            pcdf_u = self.pcdf.index_select(0, u)                # [N,F,K]
            dcdf_u = self.dcdf.index_select(0, u)                # [N,F,F]

            # ---- 2b expire ------------------------------------------------
            for up in (True, False):
                dest, wait, count = self._queue(up)
                occ = self._aQ.view(1, 1, -1) < count.unsqueeze(-1)
                exp = occ & (wait >= MAX_WAIT_TICKS)
                abandoned = abandoned + exp.sum((1, 2))
                dest, wait, count = _compact_queue(dest, wait, count, occ & ~exp,
                                                   self._aQ)
                self._set_queue(up, dest, wait, count)

            # ---- 2c spawn -------------------------------------------------
            x = rng.draw_uniform_torch(self.keys.unsqueeze(-1),
                                       t_pre.unsqueeze(-1),
                                       Slots.ARRIVAL_COUNT,
                                       self._idx_arrival[s])      # [N,F]
            n_arr = (x.unsqueeze(-1) >= pcdf_u).sum(-1)           # [N,F]
            r = rng.draw_uniform_torch(self.keys.view(-1, 1, 1),
                                       t_pre.view(-1, 1, 1),
                                       Slots.DEST,
                                       self._idx_dest[s])         # [N,F,K]
            d_raw = (r.unsqueeze(-1) > dcdf_u.unsqueeze(2)).sum(-1)  # [N,F,K]
            d = torch.where(d_raw == F, self._dest_fallback, d_raw)
            not_self = d != self._aF.view(1, F, 1)                # discard d == f

            for k in range(K_MAX):
                act_k = (n_arr > k) & not_self[:, :, k]           # [N,F]
                dk = d[:, :, k]
                goes_up = dk > self._aF.view(1, F)
                for up in (True, False):
                    dest, wait, count = self._queue(up)
                    can = act_k & (goes_up if up else ~goes_up)
                    full = count >= MAX_QUEUE
                    rejected = rejected + (can & full).sum(-1)
                    add = can & ~full
                    at_slot = self._aQ.view(1, 1, -1) == count.unsqueeze(-1)
                    write = add.unsqueeze(-1) & at_slot
                    dest = torch.where(write, dk.unsqueeze(-1), dest)
                    wait = torch.where(write, torch.zeros_like(wait), wait)
                    count = count + add.to(i64)
                    self._set_queue(up, dest, wait, count)

            # ---- 2d advance cars ------------------------------------------
            cs0 = self.car_state
            moving = cs0 == MOVING
            pos_old = self.pos
            self.pos = torch.where(moving, pos_old + self.dir, pos_old)
            occ_r = self.rider_dest != -1
            to_dest = _UPF * self.rider_dest - pos_old.unsqueeze(-1)
            sgn = torch.where(to_dest >= 0, torch.ones_like(to_dest),
                              -torch.ones_like(to_dest))          # sign(0)=+1
            rmask = moving.unsqueeze(-1) & occ_r
            tw = rmask & (sgn == self.dir.unsqueeze(-1))
            toward_units = toward_units + tw.sum((1, 2))
            away_units = away_units + (rmask & ~(sgn == self.dir.unsqueeze(-1))).sum((1, 2))
            arrived = moving & (self.pos == _UPF * self.target)

            in_doors = (cs0 == DOORS_OPENING) | (cs0 == DWELLING) | (cs0 == DOORS_CLOSING)
            self.timer = torch.where(in_doors, self.timer - 1, self.timer)
            open_done = (cs0 == DOORS_OPENING) & (self.timer == 0)
            dwell_done = (cs0 == DWELLING) & (self.timer == 0)
            close_done = (cs0 == DOORS_CLOSING) & (self.timer == 0)

            # Transfers: sequential in car order (queue contention — spec 2d).
            for i in range(E):
                delivered = delivered + self._transfer_car(i, open_done[:, i])

            self.car_state = torch.where(
                arrived, torch.full_like(cs0, IDLE),
                torch.where(open_done, torch.full_like(cs0, DWELLING),
                            torch.where(dwell_done, torch.full_like(cs0, DOORS_CLOSING),
                                        torch.where(close_done, torch.full_like(cs0, IDLE),
                                                    cs0))))
            self.timer = torch.where(open_done, torch.full_like(self.timer, DWELL_TICKS),
                                     torch.where(dwell_done,
                                                 torch.full_like(self.timer, DOOR_TICKS),
                                                 self.timer))
            self.pending = torch.where(close_done, torch.zeros_like(self.pending),
                                       self.pending)

            # ---- 2e age ---------------------------------------------------
            waiting = torch.zeros_like(queue_ticks)
            for up in (True, False):
                dest, wait, count = self._queue(up)
                occ = self._aQ.view(1, 1, -1) < count.unsqueeze(-1)
                wait = torch.where(occ, wait + 1, wait)
                self._set_queue(up, dest, wait, count)
                waiting = waiting + occ.sum((1, 2))
            queue_ticks = queue_ticks + waiting
            rider_ticks = rider_ticks + (self.rider_dest != -1).sum((1, 2))

        # spec: Rewards — pinned float64 evaluation order.
        f64 = torch.float64
        r_del = R_DELIVERED * delivered.to(f64)
        r_tow = R_TOWARD * (toward_units.to(f64) * 0.0625)
        r_awy = R_AWAY * (away_units.to(f64) * 0.0625)
        r_rej = R_REJECTED * rejected.to(f64)
        r_abn = R_ABANDONED * abandoned.to(f64)
        r_ine = R_IN_CAR * (rider_ticks.to(f64) * 0.1)
        r_inq = R_IN_QUEUE * (queue_ticks.to(f64) * 0.1)
        rewards = r_del + r_tow + r_awy + r_rej + r_abn + r_ine + r_inq

        if self._collect:
            # Metrics-only tallies (delivered-wait is accrued in _transfer_car).
            self._m_delivered = self._m_delivered + delivered
            self._m_abandoned = self._m_abandoned + abandoned
            self._m_rejected = self._m_rejected + rejected

        # spec: Termination — post-step counter hits the cap (the horizon;
        # default MAX_DECISIONS, longer under eval).
        terminated = (t_pre + 1) == self._max_decisions
        return rewards, terminated

    # ---------------------------------------------------------------- helpers
    def _queue(self, up: bool):
        if up:
            return self.up_dest, self.up_wait, self.up_count
        return self.down_dest, self.down_wait, self.down_count

    def _set_queue(self, up: bool, dest, wait, count) -> None:
        if up:
            self.up_dest, self.up_wait, self.up_count = dest, wait, count
        else:
            self.down_dest, self.down_wait, self.down_count = dest, wait, count

    def _transfer_car(self, i: int, mask: torch.Tensor) -> torch.Tensor:
        """spec: Step 2d Transfer for car i under mask (its doors just
        finished opening). Returns per-instance delivered counts."""
        n = self.n
        f = self.pos[:, i] // _UPF                                # [N]
        p = self.pending[:, i]
        row = self.rider_dest[:, i, :]                            # view; reassigned, never mutated
        col3 = (self._aE == i).view(1, E, 1)                      # write-back mask
        wrow = self.rider_wait[:, i, :] if self._collect else None

        # unload (pending 5): remove riders with dest == f, keep order.
        unload = mask & (p == 5)
        occ = row != -1
        hit = unload.unsqueeze(-1) & occ & (row == f.unsqueeze(-1))
        delivered = hit.sum(-1)
        if self._collect:
            # Tally delivered riders' frozen queue-wait (ticks) — Unity counts
            # wait at delivery, so this is measured here, not at boarding.
            w_hit = torch.where(hit, wrow, torch.zeros_like(wrow))    # [N,CAP]
            self._m_wsum = self._m_wsum + w_hit.sum(-1)
            self._m_wcount = self._m_wcount + delivered
            self._m_wpeak = torch.maximum(self._m_wpeak, w_hit.amax(-1))
            bins = ((wrow * _WAIT_BINS) // MAX_WAIT_TICKS).clamp(0, _WAIT_BINS - 1)
            self._m_whist = self._m_whist.scatter_add(1, bins, hit.to(torch.int64))
        kept = occ & ~hit
        order = torch.argsort((~kept).to(torch.int8), dim=-1, stable=True)
        kept_n = kept.sum(-1)
        compacted = torch.where(self._aC.view(1, -1) < kept_n.unsqueeze(-1),
                                row.gather(-1, order),
                                torch.full_like(row, -1))
        row = torch.where(unload.unsqueeze(-1), compacted, row)
        if self._collect:
            comp_w = torch.where(self._aC.view(1, -1) < kept_n.unsqueeze(-1),
                                 wrow.gather(-1, order), torch.zeros_like(wrow))
            wrow = torch.where(unload.unsqueeze(-1), comp_w, wrow)

        # board (pending 3 = up queue, 4 = down queue), FIFO greedy.
        for up, pend_val, dirv in ((True, 3, 1), (False, 4, -1)):
            b = mask & (p == pend_val)                            # [N]
            dest, wait, count = self._queue(up)
            fidx3 = f.view(n, 1, 1).expand(n, 1, MAX_QUEUE)
            q_dest = dest.gather(1, fidx3).squeeze(1)             # [N,Q]
            q_wait = wait.gather(1, fidx3).squeeze(1)
            q_cnt = count.gather(1, f.unsqueeze(-1)).squeeze(1)   # [N]

            rc = (row != -1).sum(-1)
            take = torch.minimum(CAPACITY - rc, q_cnt)
            take = torch.where(b, take, torch.zeros_like(take))

            # append queue[0..take) to rider slots rc..rc+take.
            src = (self._aC.view(1, -1) - rc.unsqueeze(-1)).clamp(0, MAX_QUEUE - 1)
            boarded = q_dest.gather(-1, src)
            fill = ((self._aC.view(1, -1) >= rc.unsqueeze(-1))
                    & (self._aC.view(1, -1) < (rc + take).unsqueeze(-1)))
            row = torch.where(fill, boarded, row)
            if self._collect:
                # Carry each boarded rider's frozen queue-wait into the car.
                wrow = torch.where(fill, q_wait.gather(-1, src), wrow)

            # pop the taken prefix off the queue (shift left by take).
            keep_mask = self._aQ.view(1, -1) < (q_cnt - take).unsqueeze(-1)
            shift_idx = (self._aQ.view(1, -1) + take.unsqueeze(-1)).clamp(max=MAX_QUEUE - 1)
            new_q_dest = torch.where(keep_mask, q_dest.gather(-1, shift_idx),
                                     torch.full_like(q_dest, -1))
            new_q_wait = torch.where(keep_mask, q_wait.gather(-1, shift_idx),
                                     torch.zeros_like(q_wait))
            dest = dest.scatter(1, fidx3, new_q_dest.unsqueeze(1))
            wait = wait.scatter(1, fidx3, new_q_wait.unsqueeze(1))
            count = count.scatter(1, f.unsqueeze(-1), (q_cnt - take).unsqueeze(-1))
            self._set_queue(up, dest, wait, count)

            # dir = ±1 even if nobody boarded (spec: Transfer).
            dircol = b.unsqueeze(-1) & (self._aE == i).view(1, E)
            self.dir = torch.where(dircol, torch.full_like(self.dir, dirv), self.dir)

        self.rider_dest = torch.where(col3, row.unsqueeze(1), self.rider_dest)
        if self._collect:
            self.rider_wait = torch.where(col3, wrow.unsqueeze(1), self.rider_wait)
        return delivered

    # -------------------------------------------------------------- eval hooks
    def reset_metrics(self) -> None:
        """Zero the post-warmup accumulators (mirrors Unity's StartEpoch). Call
        once at the warmup boundary; rider_wait state is preserved, so a rider
        who boarded pre-warmup still contributes its full wait at delivery."""
        if not self._collect:
            raise RuntimeError("RlevatorBatched was built without collect_metrics=True")
        z = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self._m_delivered = z.clone()
        self._m_abandoned = z.clone()
        self._m_rejected = z.clone()
        self._m_wsum = z.clone()
        self._m_wcount = z.clone()
        self._m_wpeak = z.clone()
        self._m_whist = torch.zeros(self.n, _WAIT_BINS, dtype=torch.int64, device=self.device)

    def metrics(self) -> dict[str, torch.Tensor]:
        """Per-instance eval accumulators since the last reset_metrics(). Wait
        fields are in ticks (×0.1 → seconds); wait_hist bins mirror Unity's
        64-bin WaitHistogram over [0, MAX_WAIT_TICKS]."""
        return {
            "delivered": self._m_delivered,
            "abandoned": self._m_abandoned,
            "rejected": self._m_rejected,
            "wait_sum_ticks": self._m_wsum,
            "wait_count": self._m_wcount,
            "wait_peak_ticks": self._m_wpeak,
            "wait_hist": self._m_whist,
        }

    # ------------------------------------------------------------ observation
    def observe(self) -> torch.Tensor:
        # spec: Observations — float32[98], exact block order; ratios are
        # float32(int) / float32(constant).
        f32 = torch.float32
        fl = (self.pos + 8) // _UPF                               # display floor
        car_floor = (self._aF.view(1, 1, F) == fl.unsqueeze(-1)).to(f32).reshape(self.n, E * F)
        car_active = torch.ones(self.n, E, dtype=f32, device=self.device)
        car_buttons = (self.rider_dest.unsqueeze(-1)
                       == self._aF.view(1, 1, 1, F)).any(2).to(f32).reshape(self.n, E * F)

        up_btn = self.up_count > 0
        dn_btn = self.down_count > 0
        hall_buttons = torch.stack([up_btn, dn_btn], dim=-1).to(f32).reshape(self.n, 2 * F)

        up_front = torch.where(up_btn,
                               self.up_wait[:, :, 0].clamp(max=MAX_WAIT_TICKS),
                               torch.zeros_like(self.up_count))
        dn_front = torch.where(dn_btn,
                               self.down_wait[:, :, 0].clamp(max=MAX_WAIT_TICKS),
                               torch.zeros_like(self.down_count))
        hall_age = (torch.stack([up_front, dn_front], dim=-1).to(f32)
                    / float(MAX_WAIT_TICKS)).reshape(self.n, 2 * F)

        movingv = self.car_state == MOVING
        m = torch.where(movingv,
                        torch.where(self.dir > 0,
                                    torch.full_like(self.dir, 2),
                                    torch.zeros_like(self.dir)),
                        torch.ones_like(self.dir))
        motion_hot = (torch.arange(3, device=self.device).view(1, 1, 3)
                      == m.unsqueeze(-1)).to(f32)
        posn = (self.pos.to(f32) / float(MAX_POS)).unsqueeze(-1)
        car_motion = torch.cat([motion_hot, posn], dim=-1).reshape(self.n, 4 * E)

        loads = (self.rider_dest != -1).sum(-1).to(f32) / float(CAPACITY)

        return torch.cat([car_floor, car_active, car_buttons, hall_buttons,
                          hall_age, car_motion, loads], dim=-1)

    # -------------------------------------------------------------- serialize
    def state_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "t": self.t,
            "pos": self.pos,
            "target": self.target,
            "dir": self.dir,
            "car_state": self.car_state,
            "timer": self.timer,
            "pending": self.pending,
            "rider_dest": self.rider_dest,
            "up_count": self.up_count,
            "down_count": self.down_count,
            "up_dest": self.up_dest,
            "down_dest": self.down_dest,
            "up_wait": self.up_wait,
            "down_wait": self.down_wait,
        }

    # -------------------------------------------------------------- invariants
    @invariant("bounds")
    def _inv_bounds(self) -> torch.Tensor:
        cars_ok = ((self.pos >= 0) & (self.pos <= MAX_POS)
                   & (self.target >= 0) & (self.target <= F - 1)
                   & (self.dir.abs() == 1)
                   & (self.car_state >= 0) & (self.car_state <= 4)
                   & ((self.pending == 0) | ((self.pending >= 3) & (self.pending <= 5)))
                   ).all(-1)
        return cars_ok & (self.t >= 0) & (self.t <= MAX_DECISIONS)

    @invariant("idle_at_floor")
    def _inv_idle_at_floor(self) -> torch.Tensor:
        return ((self.car_state == MOVING) | (self.pos % _UPF == 0)).all(-1)

    @invariant("moving_consistent")
    def _inv_moving(self) -> torch.Tensor:
        delta = _UPF * self.target - self.pos
        sgn = torch.where(delta >= 0, torch.ones_like(delta), -torch.ones_like(delta))
        ok = (delta != 0) & (self.dir == sgn)
        return (~(self.car_state == MOVING) | ok).all(-1)

    @invariant("timer_state")
    def _inv_timer(self) -> torch.Tensor:
        cs = self.car_state
        still = (cs == IDLE) | (cs == MOVING)
        doors = (cs == DOORS_OPENING) | (cs == DOORS_CLOSING)
        ok = torch.where(still, self.timer == 0,
                         torch.where(doors,
                                     (self.timer >= 1) & (self.timer <= DOOR_TICKS),
                                     (self.timer >= 1) & (self.timer <= DWELL_TICKS)))
        return ok.all(-1)

    @invariant("pending_doors")
    def _inv_pending(self) -> torch.Tensor:
        in_doors = ((self.car_state == DOORS_OPENING) | (self.car_state == DWELLING)
                    | (self.car_state == DOORS_CLOSING))
        return ((self.pending != 0) == in_doors).all(-1)

    @invariant("rider_prefix")
    def _inv_riders(self) -> torch.Tensor:
        occ = self.rider_dest != -1
        prefix = (occ[:, :, 1:] <= occ[:, :, :-1]).all(-1)
        vals = torch.where(occ, (self.rider_dest >= 0) & (self.rider_dest <= F - 1),
                           self.rider_dest == -1)
        return (prefix & vals.all(-1)).all(-1)

    @invariant("queue_shape")
    def _inv_queues(self) -> torch.Tensor:
        ok = torch.ones(self.n, dtype=torch.bool, device=self.device)
        floors = self._aF.view(1, F, 1)
        for dest, wait, count, is_up in (
            (self.up_dest, self.up_wait, self.up_count, True),
            (self.down_dest, self.down_wait, self.down_count, False),
        ):
            occ = self._aQ.view(1, 1, -1) < count.unsqueeze(-1)
            if is_up:
                dest_dir = dest > floors
            else:
                dest_dir = (dest < floors) & (dest >= 0)
            slot_ok = torch.where(occ, dest_dir, (dest == -1) & (wait == 0))
            wait_ok = (wait >= 0) & (wait <= MAX_WAIT_TICKS)
            both = occ[:, :, 1:] & occ[:, :, :-1]
            mono = ~both | (wait[:, :, :-1] >= wait[:, :, 1:])
            ok = ok & (slot_ok & wait_ok).all(-1).all(-1) & mono.all(-1).all(-1)
        return ok

    @invariant("reset_support")
    def _inv_reset(self) -> torch.Tensor:
        fresh = self.t == 0
        empty = ((self.up_count == 0).all(-1) & (self.down_count == 0).all(-1)
                 & (self.rider_dest == -1).all(-1).all(-1)
                 & (self.car_state == IDLE).all(-1)
                 & (self.pos == self._start_pos).all(-1))
        return ~fresh | empty
