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

## 1. Baseline: collective LOOK + nearest-car (already implemented)

`ElevatorHeuristics.CollectiveLook(Building)` — three passes per decision:
1. **Rider LOOK:** each occupied car targets the nearest desired floor *ahead in its current
   direction* (no mid-trip reversal).
2. **Nearest-car dispatch:** each empty idle car greedily claims the nearest unclaimed floor with
   a waiting rider; honors `[minFloor, maxFloor]`; `claimed` set prevents double-assignment.
3. **Resolve to primitive action** (unload / board-up / board-down / step / idle).

Use it three ways: (a) headline baseline for every experiment, (b) demonstration source for
optional BC/GAIL warm-start, (c) the sandbox dispatcher for the dashboard.

**Optional stronger baselines** (to make an RL win meaningful, not just "beat the weakest thing"):
- **Zoned LOOK:** static sectoring under up-peak (assign cars to floor bands). Cheap to add.
- **Estimated-Time-of-Arrival (ETA) dispatch:** assign each hall call to the car with least
  estimated marginal delay (the core of real ETA/ETD controllers). This is the honest bar for a
  "scale" claim — if RL beats nearest-car but not ETA, say so.

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
  delivery spikes. Consider logging the per-term reward decomposition (the `Acc` struct already
  separates them) to a custom TensorBoard stat.

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

## 7. Milestones

1. **M0 — Baseline surface (E1).** No training. Produces the yardstick + first thesis evidence.
2. **M1 — Trainable setup (E2).** Apply §2 fixes; PPO matches LOOK on rung S. Gate: parity.
3. **M2 — Scale curve (E3).** Train/eval across S→H; produce the gap-vs-rung headline figure.
4. **M3 — Zoning + obs (E4, E5).** The thesis core + cheap ablations.
5. **M4 — Architecture + generalization (E6, E7).** Unlock large fleets; one-policy-many-fleets.
6. **M5 — Stretch + write-up (E8).** Fairness, transfer, emergent-strategy visuals.

---

### Appendix — expectation calibration
Classical result (Crites & Barto, 1996) needed a **10-floor, 4-car** building under **heavy
down-peak** to show clear RL gains over the best heuristics — i.e. the win shows up at scale and
under load, not in toy settings. That is consistent with this plan's thesis and is why the
experiment weight is on rungs **L/Z/H** at **intensity ≥ 1.5**, not on rung S.
