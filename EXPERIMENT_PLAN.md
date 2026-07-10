# Elevator RL — Research & Experiment Plan

Companion to [`DASHBOARD_PLAN.md`](DASHBOARD_PLAN.md) (the visualization, now built). This
document is the **research framing, baseline, technical prerequisites, and experiment matrix**
for training an RL group-elevator controller and comparing it against the classical heuristic.

---

## 0. Thesis

**Hand-tuned group-control heuristics (collective LOOK + nearest-car dispatch) are near-optimal
in small, unconstrained buildings, but degrade as the dispatch problem grows in scale and
constraint. A learned policy should recover the heuristic in the easy regime and *pull ahead* in
the hard regime — many floors, per-car floor restrictions (elevator banks / zoning), heavy and
asymmetric traffic, and heterogeneous fleets.**

This is explicitly **not** a "beat LOOK on an 8-floor / 3-car building" project — LOOK is very
strong there and the honest expected result is a tie. The interesting claim is about **where the
heuristic's greedy, reactive, per-car-myopic structure breaks down**, and whether RL's capacity
to *anticipate demand* and *coordinate globally* buys a measurable win as the environment scales.

### Why LOOK should degrade with scale/constraint (the mechanism we're probing)
- **Greedy nearest-car dispatch** approximates a combinatorial assignment problem. The
  approximation is tight with few cars/floors and loose as both grow.
- **Reactive, never anticipatory.** LOOK assigns the nearest car to a call that *already exists*.
  It never pre-positions cars for demand it could predict (up-peak → park near lobby; down-peak
  → distribute high). Anticipation is worth more the longer the round-trip (more floors).
- **Per-car myopia under floor restrictions.** With banks/zoning (`floorRange` per car), a
  restricted car may sit idle because its *nearest* call is out of range while a distant in-zone
  call starves. Global coordination matters more as zones multiply.
- **No fairness / tail control.** LOOK optimizes throughput-ish behavior; it has no mechanism to
  bound the *worst* wait, which is where abandonment (`-8` reward) concentrates at scale.

We will report the LOOK↔RL gap **as a function of scale/constraint**, so the headline artifact is
a curve (gap vs. building size / #zones / traffic intensity), not a single number.

---

## 1. Baselines (three implemented)

### 1.1 LOOK + nearest-car — `ElevatorHeuristics.CollectiveLook(Building)`
Three passes per decision:
1. **Rider LOOK:** each occupied car targets the nearest desired floor *ahead in its current
   direction* (no mid-trip reversal).
2. **Nearest-car dispatch:** each empty idle car greedily claims the nearest unclaimed floor with
   a waiting rider; honors `[minFloor, maxFloor]`; `claimed` set prevents double-assignment.
3. **Resolve to primitive action** (unload / board-up / board-down / step / idle).

Use it three ways: (a) headline baseline for every experiment, (b) demonstration source for
optional BC/GAIL warm-start, (c) the sandbox dispatcher for the dashboard.

### 1.2 ETA/ETD — `EtaHeuristic` (stateful, one instance per episode)
The honest strong bar: unlike LOOK, which only ever reassigns *idle, empty* cars to new hall
calls, `EtaHeuristic` lets **any** in-service car — busy or not — claim a new call if its
estimated arrival time beats every other car's, given that car's onboard riders' destinations and
its other already-committed calls (a one-reversal scan-time estimate: continue the current sweep
to the farthest committed stop, reverse if needed, travel to the call, paying a stop penalty —
`doorTime*2 + dwellTime` — per intermediate committed stop). Assignments are **sticky**: a claimed
call stays with its car until served or the car leaves service, matching real destination-dispatch
systems (needed because a car only moves one floor per decision and re-derives everything from
scratch each tick — without stickiness, calls thrash between cars).

Two variants, both implemented (`new EtaHeuristic(numElevators, weightByQueueDepth: bool)`):
- **`ETA` (pure, `weightByQueueDepth=false`):** processes hall calls in floor-index order,
  assigning each independently to its lowest-ETA car.
- **`ETA-Weighted` (`weightByQueueDepth=true`):** processes calls in descending queue-size order
  first, so the busiest call claims the best available car before smaller ones compete.

**Documented finding (paired, same seed, 8fl/3cars/UpPeak/intensity=0.5, 2026-07-09):** pure ETA
is essentially a wash vs. LOOK in aggregate (waitMean/P95 within ~1-3%) but **not** in the
breakdown: it cut wait *p95* by 20–43% on mid-building floors (4–6) while **increasing lobby
(floor 0) abandonment by ~52–150%**. Mechanism: letting busy cars take on extra calls lengthens
their round trip, so they return to "empty and available" less often — starving the lobby, by far
the highest-volume origin under up-peak, in favor of nearer but sparser calls. This is a real,
citable limitation of naive ETA (motivates load-aware dispatch in production systems) and — more
importantly — **direct evidence for why the per-floor/per-window breakdowns in
[STATS_VIZ_REPLAY_PLAN.md](STATS_VIZ_REPLAY_PLAN.md) §1.2/§1.3 are load-bearing**: the episode
aggregate alone reports "roughly a tie," completely hiding the trade-off. Queue-depth weighting is
the mechanism expected to fix this (untested as of this writing — run the comparison in
`EvalHarness` to confirm before relying on `ETA-Weighted` as "the" strong baseline).

**Still optional / not yet built:**
- **Zoned LOOK:** static sectoring under up-peak (assign cars to floor bands). Cheap to add if the
  zoning experiments (E4) want a heuristic-with-zoning-awareness comparator beyond range-aware LOOK.

---

## 2. Technical prerequisites (fix before any training run)

These are correctness/efficiency items found in review of the current `Runtime/` code. Track them
as a checklist; several are one-liners but materially affect learnability.

- [ ] **Truncation vs. termination.** `ElevatorControllerAgent.OnActionReceived` calls
  `EndEpisode()` at `episodeDecisionLimit`. This is a *continuing* task with no terminal state;
  `EndEpisode()` tells PPO the final-state value is 0 (false) and biases the value function at
  exactly the truncation states. Use `EpisodeInterrupted()` (bootstraps the value) for the
  step-limit cutoff; reserve `EndEpisode()` for genuine terminals (none here).
- [ ] **Discount / horizon for a 0.5 s decision cadence.** γ=0.99 ⇒ ~50 s effective horizon ≈ one
  round trip in a small building — too short once buildings get tall. Plan to sweep **γ ∈ {0.99,
  0.995, 0.999}** and raise `time_horizon` (128 → 256–512). Delivered reward lands dozens of
  masked-NOOP decisions after the pivotal choice; credit assignment needs the longer horizon.
- [ ] **Observe wait, not just presence.** Reward penalizes `queueSeconds` and `abandoned`, but
  observations carry only queue *lengths* and binary hall buttons — the policy can't see that a
  floor is about to abandon (−8 incoming). Add an obs block: **normalized age of the oldest hall
  call per floor** (up/down), `min(1, oldestWait / maxWait)`. Real ETA controllers track this.
  Make it a toggle in `ObservationConfig` so it's also an ablation.
- [ ] **Per-car weight sharing (architecture).** Current obs is per-car one-hot blocks + a flat
  MLP: the network must relearn the same logic `numElevators` times, with no sharing, and it's
  sensitive to car ordering. This is the single biggest scalability limiter. Plan a **shared
  per-car encoder** (identical weights applied per car) and/or **attention over cars** as an
  explicit architecture variant (§6). Keep the flat MLP as the baseline arm.
- [ ] **Reward normalization / scale.** `delivered=+10` is large and sparse vs. dense per-second
  penalties; with `normalize: true` already set, still verify value-loss isn't dominated by the
  delivery spikes.
- [x] **Reward decomposition logging — DONE (M0).** `Building` now accumulates lifetime
  `Rw{Total,Delivered,Toward,Away,Rejected,Abandoned,InElevator,InQueue}`; `StatsCollector`/
  `EpisodeStats` surface the per-episode decomposition to CSV. Wire the same fields into
  ML-Agents' `StatsRecorder` for TensorBoard once training starts (STATS_VIZ_REPLAY_PLAN.md §5).
- [x] **Wasted fully-masked decisions — DONE.** `ElevatorControllerAgent.FixedUpdate` called
  `RequestDecision()` on every fixed `decisionInterval` tick regardless of car state; with
  `floorTravelTime` (1.6s) several times `decisionInterval` (0.5s), most ticks had every car
  mid-travel/door-cycle (all 5 non-NOOP actions masked for every car) — a wasted `OnActionReceived`
  round-trip. Fixed: skip `RequestDecision()` when no in-service car is `AtFloor`. Note this only
  removes wasted decisions, not the up-to-`decisionInterval` latency between a car becoming idle
  and the policy next acting on it (true event-driven decisions would remove that too — see §7
  below, this is the same granularity question, one level up). Also means
  `episodeDecisionLimit` now corresponds to a fleet/traffic-dependent amount of sim time.

---

## 3. The scale ladder (environment configurations)

The core independent variable is **problem difficulty**, dialed along four axes. Define a small
set of named `BuildingConfig` presets so every experiment references a rung, not ad-hoc numbers.

| Rung | Floors | Cars | Restrictions | Purpose |
|---|---|---|---|---|
| **S — Small** | 8 | 3 | none (all cars serve all floors) | Sanity / parity regime. Expect RL ≈ LOOK. |
| **M — Mid** | 16 | 5 | none | First scale step; nearest-car starts to fray. |
| **L — Large** | 30 | 8 | none | Tall building; anticipation & horizon matter. |
| **Z — Zoned** | 30 | 8 | 2–3 banks via `floorRange` (low-rise / high-rise / overlap) | The constraint regime — LOOK's per-car myopia should hurt most. |
| **H — Heterogeneous** | 40 | 10 | banks + variable in-service fleet (`randomizeActive`) + mid-episode service changes | The full "messy real building" — the target regime for the thesis. |

Traffic axis (crossed with rungs where affordable): `TrafficPattern ∈ {UpPeak, DownPeak, Lunch,
Midday, Uniform}` × `intensity ∈ {0.5, 1.0, 1.5, 2.0}`. RL's edge should be largest at **UpPeak
and intensity ≥ 1.5** (heaviest, most directional — where zoning/anticipation pay off).

> Floor restrictions are already supported: `BuildingConfig.floorRange` (`Vector2Int[]` per car)
> and `MinFloor/MaxFloor`; LOOK honors them (heuristic line 50) and actions are masked to range
> (`WriteDiscreteActionMask`). Zoned/heterogeneous experiments need config assets, not new code.

---

## 4. Experiment matrix (in priority order)

Each experiment names: the question, the arms, the rung(s), and the primary metric.

### E1 — LOOK baseline characterization *(no training; do this first)*
- **Q:** How does LOOK's avg / p95 / max wait, throughput, and abandonment scale across rungs
  S→H and across all patterns × intensities?
- **Arms:** LOOK (and optionally Zoned-LOOK, ETA) — pure sandbox runs, many seeds.
- **Output:** the reference surface every later result is measured against; also the first
  evidence for/against the thesis (does LOOK's tail wait blow up on Z/H?).

### E2 — RL parity in the easy regime
- **Q:** Can PPO match LOOK on rung **S**? (Necessary floor; if it can't match here, fix the
  setup before scaling.)
- **Arms:** LOOK vs. PPO (flat MLP). Fixed fleet, no restrictions.
- **Metric:** % of LOOK's throughput and wait at convergence. Target: ≈100%.

### E3 — **The headline: RL vs. LOOK as scale/constraint increases**
- **Q:** Does the RL−LOOK performance gap grow along S→M→L→Z→H?
- **Arms:** LOOK, (ETA), PPO-best-architecture, per rung.
- **Metric:** relative improvement in avg wait / p95 wait / abandonment vs. LOOK, plotted **as a
  function of rung**. This curve is the paper/blog thesis figure.
- **Focus:** UpPeak & DownPeak at intensity ≥ 1.5, where the mechanism (§0) predicts the largest
  gap.

### E4 — Zoning / floor-restriction stress *(the core of the thesis)*
- **Q:** With per-car banks (rung **Z**), does RL exploit the structure better than range-aware
  LOOK — e.g. by pre-positioning restricted cars and covering overlap zones?
- **Arms:** range-aware LOOK, Zoned-LOOK, PPO. Vary #zones and overlap.
- **Metric:** wait/abandonment vs. LOOK; **plus behavioral analysis** (do learned cars idle at
  zone-appropriate floors? — visualize on the dashboard).

### E5 — Observation ablations
- **Q:** How much does each obs block matter, especially **oldest-wait-age** (§2) and full-state
  vs. realistic buttons-only?
- **Arms:** toggle `ObservationConfig` blocks — {full state} vs {buttons+queue-len only,
  "realistic"} vs {+ wait-age}. Cheap: config-only, no code.
- **Metric:** convergence performance per obs set; expect wait-age to matter most on Z/H.

### E6 — Architecture: flat MLP vs. shared per-car vs. attention
- **Q:** Does weight sharing / attention over cars unlock the large-fleet rungs (L/H)?
- **Arms:** flat MLP (baseline), shared per-car encoder, attention.
- **Metric:** performance & sample-efficiency on L and H; this is where scalability is won or lost.

### E7 — Fleet-size generalization
- **Q:** Does one policy trained with randomized/curriculum fleet size generalize across fleet
  sizes vs. fixed-fleet specialists?
- **Arms:** curriculum (per `elevator_ppo.yaml`) vs. randomized-active vs. per-size specialists,
  evaluated across in-service counts. Robustness sub-test: mid-episode service changes
  (`serviceChangeProbability > 0`).
- **Metric:** performance vs. specialists across the fleet-size axis; graceful degradation.

### E8 — Stretch
- **Fairness reward:** penalize squared wait (or add explicit max-wait term); measure tail-wait
  reduction vs. throughput cost.
- **Pattern transfer:** train on one pattern, test on another (up→down); measure generalization.
- **BC/GAIL warm-start** from LOOK demonstrations to accelerate E3/E4.

### E9 — Action space granularity: primitive vs. target-floor (semi-MDP)
- **Q:** Does deciding at "pick a target floor" granularity (options-style; see §7) learn faster
  and/or reach a better policy than the current one-floor-at-a-time primitive space?
- **Motivation:** both implemented baselines (LOOK, ETA/ETD) already reason at target-floor
  granularity internally (`target[]`) and only mechanically translate to primitive steps each
  tick — evidence that's the natural unit of a *dispatch* decision, and that forcing RL through the
  primitive interface adds an indirection (re-deriving the same target across several masked/step
  decisions) that a semi-MDP action space would remove.
- **Arms:** A-Primitive (current: noop/up/down/board-up/board-down/unload, one branch per car,
  decided every eligible tick) vs. A-TargetFloor (§7: commit to a target floor, held until arrival
  or an explicit replan action, decided as an option).
- **Metric:** sample efficiency (steps/wall-clock to match LOOK) and asymptotic performance, per
  rung; secondary — does A-TargetFloor's loss of mid-trip redirect flexibility cost anything on
  Z/H (zoned/heterogeneous), where anticipatory re-routing might matter more?

---

## 5. Metrics & reporting

> Full instrumentation, run-comparison, ContentKit-styled visualization, and replay design live
> in [`STATS_VIZ_REPLAY_PLAN.md`](STATS_VIZ_REPLAY_PLAN.md). Summary below.


Log per run (sandbox eval, many seeds, fixed eval traffic tape per rung for comparability):
- **Throughput:** delivered / sim-hour; delivered / decision.
- **Wait:** mean, **p95**, **max** (tail is the story on big buildings); in-car time too.
- **Failures:** abandonment rate, rejection rate.
- **Reward decomposition:** per-term (`delivered/toward/away/rejected/abandoned/inElevator/inQueue`)
  — already separated in `Building.Acc`; surface as TensorBoard custom stats + eval CSV.
- **Behavioral:** car idle-floor distribution, zone adherence, reposition frequency — for E4/E5
  qualitative analysis, rendered on the dashboard for the "emergent strategy" screenshots.

Reporting principle: **every RL number is expressed relative to LOOK on the identical traffic
tape/seed.** The deliverable figures are (1) LOOK-scaling surface (E1), (2) RL−LOOK gap vs. rung
(E3), (3) zoning behavioral panels (E4).

---

## 6. Architecture variants (detail for E6)

- **A0 — Flat MLP (current):** concat all per-car one-hot blocks → 2×256 MLP → MultiDiscrete head
  (one size-6 branch per car). Baseline; ordering-sensitive; O(cars) parameter reuse = none.
- **A1 — Shared per-car encoder:** apply an identical small MLP to each car's slice of the obs
  (car state + its view of hall demand), pool a shared building-context vector, decode each car's
  branch from [car-embedding ⊕ context]. Permutation-friendly, parameter-shared, scales to more
  cars without widening the net.
- **A2 — Attention over cars:** self-attention across car embeddings before decoding, so a car's
  action conditions on what the *others* are doing — directly targets the coordination failure
  mode of greedy dispatch. Highest ceiling, most complex.

ML-Agents path: custom obs layout is fine for A0. A1/A2 likely need either careful obs grouping
with attention-capable network settings, or exporting to a custom trainer. Decide after E2/E3
show whether A0 plateaus.

---

## 7. Action space variants (detail for E9)

The environment currently decides at **primitive** granularity: every eligible tick, each car
picks one of {noop, step up one floor, step down one floor, board-up, board-down, unload}
(`Building.ApplyAction`, cases 1/2 move exactly one floor; the car re-idles and the full dispatch
recomputes from scratch). This is a legitimate, complete low-level action set — matches the
physical constraints exactly, and is what most ML-Agents examples default to — but it is not the
only reasonable choice, and building the two heuristic baselines surfaced concrete evidence about
the alternative.

- **AS0 — Primitive (current).** One MultiDiscrete branch (size 6) per car, decided every tick a
  car is `AtFloor`. Pro: maximal flexibility — a car can be redirected floor-by-floor as new
  information arrives, no separate "replan" mechanism needed. Con: a single conceptual dispatch
  decision ("go serve floor 7") is re-derived across every intermediate arrival, spreading credit
  for one good call across many identical-looking decisions and diluting the learning signal —
  exactly the mechanism flagged in §2's discount/horizon item.
- **AS1 — Target-floor (semi-MDP / options).** A car's action is "commit to target floor f"
  (chosen only when the car is `AtFloor` and has no outstanding commitment); a fixed, deterministic
  sub-controller (literally the existing `Building.ApplyAction` step/board/unload resolution,
  reused, not reimplemented) drives the car floor-by-floor toward f without further policy
  involvement until arrival. The policy is asked for a *new* decision only on arrival (or
  optionally on an explicit "replan" trigger — e.g., a much higher-priority call has appeared —
  to avoid completely losing redirection flexibility). Pro: decisions align 1:1 with what LOOK/ETA
  already treat as the meaningful unit (their own `target[]`), which should sharply cut the
  effective decision count needed to complete one dispatch and ease credit assignment. Con: a
  committed car cannot react to new information until arrival unless the replan trigger is added
  (itself a design choice — what should trigger it, and does it reopen the same thrashing risk
  seen with `EtaHeuristic`'s early non-sticky prototype in §1.2?).

Both are straightforward to implement given what already exists: AS1's "drive toward a committed
target" logic is exactly `EtaHeuristic`/`CollectiveLook`'s existing resolve step, just driven by a
policy-chosen target instead of a heuristically-computed one. Recommended default: build AS0 first
(already in place), get E2/E3 baseline numbers, then build AS1 and run E9 — don't let this block
the M1/M2 milestones below.

---

## 8. Milestones

1. **M0 — Baseline surface (E1).** No training. Produces the yardstick + first thesis evidence.
   *(Instrumentation + LOOK/ETA/ETA-Weighted baselines are implemented; the full sweep across
   rungs/patterns/intensities is the remaining work.)*
2. **M1 — Trainable setup (E2).** Apply remaining §2 fixes (truncation, horizon/γ, wait-age obs);
   PPO matches LOOK on rung S. Gate: parity.
3. **M2 — Scale curve (E3).** Train/eval across S→H; produce the gap-vs-rung headline figure.
4. **M3 — Zoning + obs (E4, E5).** The thesis core + cheap ablations.
5. **M4 — Architecture + generalization (E6, E7).** Unlock large fleets; one-policy-many-fleets.
6. **M5 — Action space (E9).** Build AS1 (target-floor); compare against AS0.
7. **M6 — Stretch + write-up (E8).** Fairness, transfer, emergent-strategy visuals.

---

### Appendix — expectation calibration
Classical result (Crites & Barto, 1996) needed a **10-floor, 4-car** building under **heavy
down-peak** to show clear RL gains over the best heuristics — i.e. the win shows up at scale and
under load, not in toy settings. That is consistent with this plan's thesis and is why the
experiment weight is on rungs **L/Z/H** at **intensity ≥ 1.5**, not on rung S.
