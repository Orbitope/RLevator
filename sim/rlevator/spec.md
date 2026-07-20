# rlevator — environment spec

Single source of truth. Both `reference.py` and `fast.py` are written from
this document — never from each other. Every rule below must be traceable in
both implementations.

**Provenance.** This is a *semantic port* of the Unity RLevator simulation
(`Assets/ElevatorRL/Runtime/Building.cs`, `Elevator.cs`, `PassengerArrivals.cs`
at the E15/V2 state of the repo), S rung (`S_BuildingConfig.asset`), AS0
primitive actions, the 7 default observation blocks, fixed fully-in-service
fleet. Deliberate deviations from Unity (all confirmed):

1. **RNG**: simulacrum counter-based named-slot draws replace Unity's two
   never-reseeded .NET `System.Random` streams. Trajectories are NOT
   bit-comparable to Unity; distributions are.
2. **Integer time**: 1 tick = 0.1 s (all Unity durations are multiples of
   0.1 s). One env step = one decision interval = 5 ticks. Unity's 0.02 s
   physics tick is coarsened to 0.1 s; the Poisson process is additive so the
   arrival law is unchanged (rates are interpolated at 0.1 s granularity
   instead of 0.02 s — negligible).
3. **Poisson sampling**: single-uniform inverse-CDF (capped at `K_MAX`)
   replaces Unity's variable-draw Knuth loop. Identical distribution up to
   the truncated tail (P(k > 8) < 1e-20 at these rates).
4. **Traffic constants**: `PATTERN = Midday`, `INTENSITY = 1.0` — the V2
   baseline for rung S (`elev-v2-s-midday-01`). NOT UpPeak: the project's
   own eval harness documents that UpPeak@1.0 saturates rung S (~41%
   rejection), which would make every policy look alike.
5. Metrics-only Unity fields (passenger `id`, `arrivalTime`, in-car
   `age`/`waitTime` carryover, `MaxWaitObserved`, …) are dropped; they do
   not affect dynamics, reward, or observations.
6. Display-floor rounding is round-half-up (`(pos + 8) // 16`) instead of
   Unity's `Mathf.RoundToInt` banker's rounding; the two differ only when a
   moving car sits exactly at a half-floor boundary, and only in the
   observation, never in dynamics (dynamics only read the floor when Idle,
   where `pos` is an exact floor).
7. Reward arithmetic is float64 with a pinned evaluation order (Unity uses
   float32); coefficients are identical.

## Constants

Topology and timing (from `S_BuildingConfig.asset`):

```
F  = 8                # floors, 0 = lobby
E  = 3                # cars
CAPACITY   = 8        # riders per car
MAX_QUEUE  = 12       # waiters per (floor, direction) queue
UNITS_PER_FLOOR = 16  # position sub-units per floor (1 tick of travel each)
MAX_POS    = 112      # = UNITS_PER_FLOOR * (F - 1)
TRAVEL_TICKS = 16     # 1.6 s per floor  -> 1 unit per tick
DOOR_TICKS   = 8      # 0.8 s doors opening OR closing
DWELL_TICKS  = 12     # 1.2 s load/unload hold
SUBTICKS_PER_STEP = 5 # decision interval 0.5 s / 0.1 s tick
MAX_WAIT_TICKS = 450  # 45 s abandonment threshold
MAX_DECISIONS  = 2048 # episode truncation, in steps (matches Unity)
START_FLOORS = [0, 4, 7]  # deterministic car start floors (Unity's spread
                          # formula lo + round(i*(hi-lo)/(E-1)) with
                          # round-half-to-even, evaluated for S)
K_MAX = 8             # Poisson count cap per (floor, sub-tick)
```

Traffic (from `TrafficConfig` defaults + the V2 S-rung baseline):

```
PATTERN    = Midday   # fixed for v1 (see deviation 4)
INTENSITY  = 1.0
POPULATION = 360      # E * CAPACITY * loadPerSlot(15)
BIN_SECONDS = 900.0   # 15-minute profile bins
MIDDAY_BINS = [12,13,14,15,16,17,30,31,32,33,34,35]  # the pattern's slice
                      # of the 48-bin day (paper's "InterFloor")
```

Reward coefficients (float64, from `RewardConfig` defaults):

```
R_DELIVERED = 10.0    R_TOWARD = 0.4      R_AWAY = -0.4
R_REJECTED  = -5.0    R_ABANDONED = -8.0
R_IN_CAR    = -0.04   # per rider-second
R_IN_QUEUE  = -0.12   # per waiting-passenger-second
```

The three 48-bin arrival-rate profiles, transcribed verbatim from
`PassengerArrivals.cs` (read as float64 decimal literals):

```
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
```

## The traffic model (deterministic part)

At absolute sub-tick `u` (0-based within the episode; `u = SUBTICKS_PER_STEP
* t + s` for step `t`, sub-tick `s`), sim time is `sim = u * 0.1` seconds
(float64). The per-floor arrival rates and destination distribution are pure
functions of `u`, computed with **scalar Python float64 arithmetic**
(`math.exp`, plain `*`, `/`, `+` in exactly the order written below). Both
implementations MUST precompute these tables at init with scalar Python
math — the batched implementation may store them as tensors afterwards but
must never recompute them with torch ops (torch's `exp` is not guaranteed
bit-identical to libm). This is what keeps the two implementations
bit-identical despite the transcendental.

Interpolated component rate for a profile table `T` (mirrors Unity
`InterpAr` then `RatePerSecond`):

```
period = len(MIDDAY_BINS) * BIN_SECONDS          # 12 * 900.0 = 10800.0
tmod   = sim % period
idx    = min(int(tmod / BIN_SECONDS), len(MIDDAY_BINS) - 1)
shift  = tmod - idx * BIN_SECONDS
nxt    = (idx + 1) % len(MIDDAY_BINS)
a, b   = T[MIDDAY_BINS[idx]], T[MIDDAY_BINS[nxt]]
ar     = a + shift * (b - a) / BIN_SECONDS
rate   = (ar * POPULATION) / 30000.0             # arrivals/sec
```

Applied to the three tables this yields `rate_in(u)`, `rate_inter(u)`,
`rate_out(u)`. Then (mirrors `LoadPatternAtTime`, `upper = F - 1 = 7`):

```
lambda_sec[0]      = rate_in
per_upper          = (rate_out + rate_inter) / 7.0
lambda_sec[1..7]   = per_upper                             # each
p_lobby            = rate_out / (rate_out + rate_inter)    # denom > 0 always
                                                           # for these tables
dest_row[0][0]     = 0.0
dest_row[0][1..7]  = 1.0 / 7.0                             # each
dest_row[o][0]     = p_lobby                     for o in 1..7
dest_row[o][o]     = 0.0
dest_row[o][d]     = (1.0 - p_lobby) / 6.0       for d in 1..7, d != o
```

Per-sub-tick Poisson rate, mirroring Unity's `lambda[f] * intensity * dt`
left-to-right: `lam_tick[u][f] = (lambda_sec[f] * INTENSITY) * 0.1`.

Poisson CDF thresholds (for the inverse-CDF count draw), per `(u, f)`:

```
term   = math.exp(-lam_tick[u][f])
cdf[0] = term
for k in 1 .. K_MAX - 1:
    term   = term * lam_tick[u][f] / k
    cdf[k] = cdf[k - 1] + term
```

Destination CDF row, per `(u, origin)`: `dcdf[d] = dest_row[origin][0] + ...
+ dest_row[origin][d]` (sequential float64 accumulation in `d` order,
`d = 0 .. F-1`).

Episodes never exceed `MAX_DECISIONS * SUBTICKS_PER_STEP = 10240` sub-ticks
(1024 s), so tables need `u in [0, 10240)`.

## State space

All fields int64. Serialized form: `schema.json` `$defs/state`.

| field | shape | bounds | meaning |
|---|---|---|---|
| `t` | scalar | [0, MAX_DECISIONS] | in-episode step (decision) counter — the RNG key |
| `pos` | [E] | [0, MAX_POS] | car position in sub-floor units (16 = one floor) |
| `target` | [E] | [0, F-1] | commanded destination floor (meaningful while Moving) |
| `dir` | [E] | {-1, +1} | last-serviced direction |
| `car_state` | [E] | {0..4} | 0 Idle, 1 Moving, 2 DoorsOpening, 3 Dwelling, 4 DoorsClosing |
| `timer` | [E] | [0, DWELL_TICKS] | door/dwell countdown in ticks |
| `pending` | [E] | {0, 3, 4, 5} | which action opened the doors (0 = none) |
| `rider_dest` | [E, CAPACITY] | [-1, F-1] | rider destinations, -1 = empty slot, compact prefix |
| `up_count` | [F] | [0, MAX_QUEUE] | up-queue length per floor |
| `down_count` | [F] | [0, MAX_QUEUE] | down-queue length per floor |
| `up_dest` | [F, MAX_QUEUE] | [-1, F-1] | up-queue destinations, FIFO front at index 0, -1 = empty |
| `down_dest` | [F, MAX_QUEUE] | [-1, F-1] | down-queue destinations, same layout |
| `up_wait` | [F, MAX_QUEUE] | [0, MAX_WAIT_TICKS] | ticks waited, aligned with `up_dest`; 0 in empty slots |
| `down_wait` | [F, MAX_QUEUE] | [0, MAX_WAIT_TICKS] | same for the down queues |

A car's rider count is the number of non-(-1) entries in its `rider_dest`
row (they form a compact prefix). A car's display floor is
`floor(c) = (pos[c] + 8) // 16` (integer division, round half up); whenever
`car_state[c] != Moving`, `pos[c]` is an exact multiple of 16 and
`floor(c) = pos[c] // 16`.

## Actions

One integer `a` in `[0, 216)` (= 6^E), the packed base-6 encoding of the
per-car primitives: car `i`'s command is digit `i`, i.e.
`cmd[i] = (a // 6**i) % 6`. Per-car command semantics (Unity
`Building.ApplyAction`):

| cmd | meaning |
|---|---|
| 0 | NOOP |
| 1 | move up one floor |
| 2 | move down one floor |
| 3 | open doors to board the up-queue |
| 4 | open doors to board the down-queue |
| 5 | open doors to unload riders for this floor |

A command is **silently ignored** (no-op, no error) unless the car is Idle
(`car_state == 0`). For an Idle car at floor `f = pos // 16`:

- cmd 1: if `f < F - 1`: `target = f + 1`, `dir = +1`, `car_state = Moving`.
  At the top floor it is a no-op.
- cmd 2: if `f > 0`: `target = f - 1`, `dir = -1`, `car_state = Moving`.
  At the lobby it is a no-op.
- cmd 3, 4, 5: `pending = cmd`, `car_state = DoorsOpening`,
  `timer = DOOR_TICKS`.
- cmd 0: nothing.

Every action value is legal in every state (illegal sub-commands degrade to
no-ops), so uniform random sampling over [0, 216) is safe.

## Step

`step(a)` advances one decision interval (0.5 s):

1. **Apply actions**: decode `cmd[i]` for each car `i = 0 .. E-1` in order
   and apply as above.
2. **Run 5 sub-ticks**, `s = 0 .. 4`, each at absolute sub-tick
   `u = 5 * t_pre + s` (`t_pre` = the step counter before this step; RNG
   draws in sub-ticks use step key `t_pre`). Each sub-tick, in this exact
   order (mirrors Unity `Building.Tick`):

   a. **Rates**: look up `lam_tick[u]`, `cdf[u]`, `dcdf[u]` (deterministic).

   b. **Expire**: for each floor `f` and each queue (up, then down), remove
      every waiter with `wait >= MAX_WAIT_TICKS` (by invariant 7 these form
      a prefix of the queue); compact the queue preserving order. Each
      removal increments the `abandoned` accumulator.

   c. **Spawn**: for each floor `f = 0 .. F-1`:
      - Draw `n` = inverse-CDF Poisson: one uniform
        `x = draw_uniform(key, t_pre, ARRIVAL_COUNT, index = s * F + f)`;
        `n` = smallest `k` with `x < cdf[u][f][k]`, or `K_MAX` if none.
      - For each arrival `k = 0 .. n-1`: draw
        `r = draw_uniform(key, t_pre, DEST, index = (s * F + f) * K_MAX + k)`;
        destination `d` = first index with `r <= dcdf[u][f][d]` scanning
        `d = 0 .. F-1`; if no index satisfies (float residue), fall back to
        `d = 1` if `f == 0` else `d = 0`. If `d == f`, the arrival is
        **discarded** (no passenger, no penalty — mirrors Unity's
        `if (d == f) continue`). Otherwise the passenger joins the up queue
        of `f` if `d > f`, else the down queue, with `wait = 0`, appended at
        the back — unless that queue already holds `MAX_QUEUE` waiters, in
        which case the arrival is **rejected** (increment the `rejected`
        accumulator, nothing enqueued).

   d. **Advance cars**: for each car `i = 0 .. E-1` by `car_state`:
      - Moving: `pos += dir` (one sub-unit). For each rider `r` of the car,
        with `to_dest = 16 * rider_dest - pos_old` (position before this
        move): if `sign(to_dest) == dir` then `toward_units += 1` else
        `away_units += 1`, where `sign(0) = +1` (mirrors `Mathf.Sign`).
        Then if `pos == 16 * target`: `car_state = Idle` (timer stays 0).
      - DoorsOpening: `timer -= 1`; if `timer == 0`: **transfer** (below),
        then `car_state = Dwelling`, `timer = DWELL_TICKS`.
      - Dwelling: `timer -= 1`; if `timer == 0`: `car_state = DoorsClosing`,
        `timer = DOOR_TICKS`.
      - DoorsClosing: `timer -= 1`; if `timer == 0`: `car_state = Idle`,
        `pending = 0`, `timer = 0`.
      - Idle: nothing.

      **Transfer** at floor `f = pos // 16`, by `pending`:
      - 5 (unload): remove every rider with `rider_dest == f`, compacting
        the prefix and preserving the order of the remaining riders; each
        removal increments `delivered`.
      - 3 (board up): while the car has fewer than CAPACITY riders and
        `up_count[f] > 0`, pop the FRONT of the up queue (index 0, shifting
        the rest forward) and append its destination to the car's rider
        prefix. Then `dir = +1` (even if nobody boarded).
      - 4 (board down): same from the down queue; then `dir = -1`.

   e. **Age**: every waiter in every queue gets `wait += 1`;
      `queue_ticks += total number of waiters` (counted AFTER this
      sub-tick's expiry/spawn/boarding); `rider_ticks += total riders in
      cars`.

3. `t = t_pre + 1`.
4. **Reward** (below); **termination**: `terminated = (t == MAX_DECISIONS)`.

## Rewards

Seven integer accumulators, zeroed at the start of every step, summed over
its 5 sub-ticks: `delivered`, `rejected`, `abandoned`, `toward_units`,
`away_units` (rider·sub-units), `rider_ticks`, `queue_ticks`.

**Default reward (EXPERIMENT_PLAN §9.4):** `R_TOWARD = R_AWAY = 0` — the
toward/away movement shaping is OFF by default because it hurt deployed
service quality (policies gamed it for reward). The `r_tow`/`r_awy` terms
below stay in the expression (they just evaluate to 0), so bit-identity is
unaffected. `RLEVATOR_SHAPING=on` restores the Unity-faithful ±0.4 weights.

The step reward is the float64 expression, evaluated in exactly this order
(mirrors Unity `CollectReward`; `0.0625 = 1/16` converts sub-units to
floors, `0.1` converts ticks to seconds):

```
r_del = R_DELIVERED * delivered
r_tow = R_TOWARD    * (toward_units * 0.0625)
r_awy = R_AWAY      * (away_units * 0.0625)
r_rej = R_REJECTED  * rejected
r_abn = R_ABANDONED * abandoned
r_ine = R_IN_CAR    * (rider_ticks * 0.1)
r_inq = R_IN_QUEUE  * (queue_ticks * 0.1)
reward = r_del + r_tow + r_awy + r_rej + r_abn + r_ine + r_inq
```

All inputs are int64 counts, so both implementations produce bit-identical
float64 rewards; no `x-atol` is declared anywhere.

## Termination

`terminated` iff the post-step counter `t == MAX_DECISIONS` (2048). There is
no other terminal condition (the Unity task is continuing; 2048 decisions is
its truncation limit, adopted here as the episode boundary). Random actions
always terminate in exactly `MAX_DECISIONS` steps.

## Reset

Fully deterministic (no RNG slots consumed):

- `t = 0`.
- Car `i`: `pos = 16 * START_FLOORS[i]`, `target = START_FLOORS[i]`,
  `dir = +1`, `car_state = Idle`, `timer = 0`, `pending = 0`, no riders
  (`rider_dest` all -1).
- All queues empty: counts 0, dest arrays all -1, wait arrays all 0.

Different seeds produce different trajectories from the first step's
arrival draws (the episode key feeds every draw).

## Observations

`float32[98]`, written in this exact block order (Unity
`Building.WriteObservation`, blocks 1–7; all cars in service, so the
in-service gates are constant-true). Every ratio is computed as: cast the
int64 to float32, divide by the float32 constant.

| # | block | size | encoding |
|---|---|---|---|
| 1 | carFloor | E×F = 24 | per car: one-hot(F) of `floor(c) = (pos + 8) // 16` |
| 2 | carActive | E = 3 | `1.0` per car |
| 3 | carButtons | E×F = 24 | per car, per floor `f`: `1.0` if any `rider_dest[c][:] == f` else `0.0` |
| 4 | hallButtons | 2F = 16 | per floor `f`: `up_count[f] > 0 ? 1 : 0`, then `down_count[f] > 0 ? 1 : 0` (interleaved) |
| 5 | hallCallAge | 2F = 16 | per floor `f`: up then down: `float32(min(wait_front, 450)) / 450.0f` where `wait_front` is the front waiter's wait (`[f][0]`), `0.0` if the queue is empty |
| 6 | carMotion | 4E = 12 | per car: one-hot(3) of `m` (`m = 2` if Moving and `dir > 0`, `0` if Moving and `dir < 0`, else `1`), then `float32(pos) / 112.0f` |
| 7 | carLoads | E = 3 | per car: `float32(rider_count) / 8.0f` |

(The queue front holds the oldest waiter — invariant 7 — so `wait_front`
equals Unity's max-over-queue `OldestWaitFrac`.)

## Invariants

Hold in EVERY reachable state, including terminal states. Each becomes an
`@invariant` on the batched env.

1. `bounds`: `0 <= pos <= MAX_POS`; `0 <= target <= F-1`;
   `dir in {-1, +1}`; `car_state in {0..4}`; `pending in {0, 3, 4, 5}`;
   `0 <= t <= MAX_DECISIONS`.
2. `idle_at_floor`: `car_state != Moving` implies `pos % 16 == 0`.
3. `moving_consistent`: `car_state == Moving` implies `pos != 16 * target`
   and `dir == sign(16 * target - pos)`.
4. `timer_state`: Idle or Moving implies `timer == 0`; DoorsOpening or
   DoorsClosing implies `1 <= timer <= DOOR_TICKS`; Dwelling implies
   `1 <= timer <= DWELL_TICKS`.
5. `pending_doors`: `pending != 0` iff
   `car_state in {DoorsOpening, Dwelling, DoorsClosing}`.
6. `rider_prefix`: in each `rider_dest` row, non-(-1) entries form a
   compact prefix of length <= CAPACITY, each in `[0, F-1]`; entries after
   the prefix are exactly -1.
7. `queue_shape`: each queue's first `count` dest entries are in
   `[0, F-1]` with `up_dest[f][j] > f` / `down_dest[f][j] < f`; entries at
   index >= count are dest = -1 and wait = 0; occupied waits are in
   `[0, MAX_WAIT_TICKS]` and non-increasing from front to back
   (FIFO: oldest first).
8. `reset_support`: `t == 0` implies all queues empty, all cars rider-less
   and Idle with `pos == 16 * START_FLOORS[i]`.

## RNG slots

| slot | name | used at | distribution |
|---|---|---|---|
| 0 | ARRIVAL_COUNT | step `t`, index `s * F + f` | one uniform → inverse-CDF Poisson(`lam_tick[u][f]`), capped at K_MAX |
| 1 | DEST | step `t`, index `(s * F + f) * K_MAX + k` | one uniform → inverse-CDF categorical over `dcdf[u][f]` |

`s` = sub-tick within the step (0..4), `f` = floor, `k` = arrival number
within that (sub-tick, floor) (0..K_MAX-1), `u = 5t + s`. Reset consumes no
draws. The two decisions use different slots, so their index spaces are
independent. Draws for all `(s, f, k)` may be computed up-front and
discarded where unused (stateless RNG).

Reserved for future variants (not used in v1): slot 2 `FLEET_COUNT`, slot 3
`FLEET_SHUFFLE` (reset-time, step 0) for the `randomizeActive` fleet
mechanic; day-cycle and other rungs change only constants, not slots.
