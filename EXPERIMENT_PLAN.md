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

**Documented finding 1 — the lobby/mid-building trade-off (paired, same seed, 8fl/3cars/UpPeak,
intensity=0.5, 2026-07-09):** pure ETA is essentially a wash vs. LOOK in aggregate (waitMean/P95
within ~1-3%) but **not** in the breakdown: it cut wait *p95* by 20–43% on mid-building floors
(4–6) while **increasing lobby (floor 0) abandonment by ~52–150%**. Mechanism: letting busy cars
take on extra calls lengthens their round trip, so they return to "empty and available" less
often — starving the lobby, by far the highest-volume origin under up-peak, in favor of nearer but
sparser calls. This is a real, citable limitation of naive ETA (motivates load-aware dispatch in
production systems) and — more importantly — **direct evidence for why the per-floor/per-window
breakdowns in [STATS_VIZ_REPLAY_PLAN.md](STATS_VIZ_REPLAY_PLAN.md) §1.2/§1.3 are load-bearing**:
the episode aggregate alone reports "roughly a tie," completely hiding the trade-off.

**Documented finding 2 — queue-depth weighting is a no-op under sticky assignment (tested across
the full calibrated S/M/L/Z/H × UpPeak/Lunch × {0.5,1,1.5}× sweep, 90 cells, 2026-07-09):**
`ETA-Weighted` produced **bit-identical results to pure `ETA` in every single cell** — 0/30
(preset×pattern×multiple) triples differ, at building sizes from 8 to 40 floors. Mechanism:
weighting only changes the *processing order* of calls that are simultaneously unclaimed in the
same decision, but under sticky assignment a claimed call only re-enters contention once its floor
is fully served — which happens one floor at a time, not synchronized across floors — so two
different floors genuinely competing for the same single best-fit car essentially never coincides,
at any scale tested. This is a structural property of the sticky-claim design, not a scenario-
specific coincidence (see the two single-rung checks that first surfaced it, both also null,
before the full sweep confirmed it holds everywhere). **Implication:** a version that periodically
re-evaluates ALL calls (not just newly-freed ones) might let weighting matter, at the cost of
reintroducing the thrashing stickiness was built to prevent — an explicit trade-off, not yet built.
Until/unless that's revisited, treat `ETA` and `ETA-Weighted` as the same baseline in practice.

**Documented finding 3 — LOOK is more robust than ETA specifically under zoning.** At a matched
~10%-abandonment calibration point (see §3), LOOK edges out ETA more clearly at Z/H (zoned) than
at S/M/L (unzoned) — e.g. UpPeak abandonRate: S 0.102→0.098 (ETA slightly better), M 0.098→0.092
(ETA slightly better), L 0.100→0.112 (LOOK better), **Z 0.157→0.206, H 0.239→0.295 (LOOK clearly
better)**. This is the opposite direction from finding 1's "ETA wins mid-building, loses lobby" —
under zoning, ETA's willingness to send busy cars on long detours to new calls appears to interact
badly with cross-zone travel distance specifically, whereas LOOK's more conservative "only
reassign empty cars" rule is comparatively more robust. Caveat: only 1-2 seeds so far per cell —
treat as a lead to confirm with more seeds, not yet a settled result.

**Still optional / not yet built:**
- **Zoned LOOK:** static sectoring under up-peak (assign cars to floor bands). Cheap to add if the
  zoning experiments (E4) want a heuristic-with-zoning-awareness comparator beyond range-aware LOOK.

---

## 2. Technical prerequisites (fix before any training run)

These are correctness/efficiency items found in review of the current `Runtime/` code. Track them
as a checklist; several are one-liners but materially affect learnability.

- [x] **Truncation vs. termination — DONE (M1).** `ElevatorControllerAgent.OnActionReceived`
  now calls `EpisodeInterrupted()` (bootstraps the value) at `episodeDecisionLimit` instead of
  `EndEpisode()` (which would tell PPO the final-state value is 0 — false for this continuing
  task, and biases the value function at exactly the truncation states).
- [x] **Discount / horizon — DONE (M1), default only.** `config/elevator_ppo.yaml`:
  `gamma: 0.99→0.995`, `time_horizon: 128→256`. This is the sweep's middle default, not a
  validated final choice — still sweep **γ ∈ {0.99, 0.995, 0.999}** once training starts.
- [x] **Observe wait, not just presence — DONE (M1).** Added `ObservationConfig.hallCallAge`
  (default **on**) + `Building.ObservationSize/WriteObservation`: normalized age of the
  longest-waiting rider per hall queue (up/down per floor), `Mathf.Clamp01(maxWait_in_queue /
  cfg.maxWait)`, via new `Building.OldestWaitFrac`. Toggle off for the "realistic, no wait-age"
  ablation arm (E5).
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

## 3. The scale ladder (environment configurations) — IMPLEMENTED

Five `BuildingConfig` preset assets at `Assets/ElevatorRL/Config/Presets/` (generate/regenerate
via **Tools ▸ Elevator RL ▸ Generate Scale Ladder Presets**), plus a sixth for E7 only:

| Rung | Floors | Cars | Zoning (`floorRange`, all nested at floor 0 — see note) | Fleet |
|---|---|---|---|---|
| **S — Small** | 8 | 3 | none | fixed |
| **M — Mid** | 16 | 5 | none | fixed |
| **L — Large** | 30 | 8 | none | fixed |
| **Z — Zoned** | 30 | 8 | low 2 cars (0–14) / mid 3 cars (0–22) / high 3 cars (0–29) | fixed |
| **H — Heterogeneous** | 40 | 10 | low 3 cars (0–15) / mid 3 cars (0–28) / high 4 cars (0–39) | fixed |
| **H_VarFleet** (E7 only) | 40 | 10 | same as H | `randomizeActive`, min 5, service-change 0.02 |

**Nested, not partitioned, zoning:** `BuildingConfig.floorRange` is a single contiguous `[min,max]`
per car, and since virtually all traffic originates or terminates at the lobby (floor 0), every
car's range starts there — a car excluding floor 0 could never serve lobby-bound/lobby-origin
riders and would permanently strand anyone whose only eligible cars were such a car. Nesting still
creates real scarcity (only high-rise cars ever reach the top) without that failure mode — a
common real design (all banks reach the sky lobby; only some continue further up).

**Car allocation is weighted toward high-rise, not even.** An initial even-ish split (Z 3/3/2, H
4/4/2) left the top-exclusive band structurally under-capacity: 34–68% abandonment even at the
lowest intensity tested, with LOOK/ETA/ETA-Weighted all failing identically — a system that far
past capacity gives no assignment-quality signal at all (nothing to differentiate dispatchers).
Root cause: a handful of full-range "high-rise" cars aren't actually *reserved* for the top floors
— they're generalists competing for abundant nearby lower-floor calls too, so the rare, distant,
long-round-trip (~90–150s) top-floor demand chronically starves under any greedy dispatch. Z/H's
current 2/3/3 and 3/3/4 splits fixed this (see §1.2 finding 3 for the resulting LOOK vs. ETA
numbers) — confirmed by calibration landing on genuine interior crossing points (below) rather
than pinning at a search-range floor.

**H_VarFleet is split out from H.** H originally also carried `randomizeActive`/
`serviceChangeProbability` (variable fleet), but a randomly-selected out-of-service subset isn't
zone-aware — an unlucky draw could gut a specific zone's capacity entirely, and abandonment doesn't
average out gracefully once queues blow past `maxWait` (this made H uncalibratable at *any*
intensity: 26% abandonment even near-zero load). Variable-fleet robustness (E7) and scale/zoning
saturation (E1/E3/E4) are different questions — H is now fixed-fleet like S/M/L/Z; `H_VarFleet` is
the same topology/zoning with variable fleet enabled, for E7 exclusively, run at a fixed,
comfortably-below-saturation intensity (not the calibration approach below).

### Intensity: calibrated per (rung, pattern), not a flat shared value

Two problems with using one flat `intensity` across all rungs, both found and fixed 2026-07-09:

1. **`PassengerArrivals` hub-floor rates don't scale with building size.** Per-floor rates
   naturally scale in total as floor count grows (more floors summed), but a single-point
   aggregate hub (the lobby under UpPeak/Lunch; the top floor under Lunch) is one absolute rate
   that doesn't grow just because the building added floors elsewhere. **Fixed:** hub floors now
   scale by `numFloors / ReferenceFloors` (8, the S rung) — S is unaffected; taller buildings get
   proportionally more hub traffic. DownPeak's `lambda[0]` is deliberately left unscaled (the
   lobby is a destination there, not an origination hub).
2. **Even after that fix, capacity-per-unit-intensity differs enormously by rung** (zoning and
   sheer floor count both matter). A flat intensity value that's comfortable for S is either
   trivial or catastrophic elsewhere.

**Fix: bisection calibration against LOOK, per (preset, pattern).** `EvalHarness.CalibrateIntensity`
finds the intensity where LOOK's `abandonRate ≈ 10%`, then the real sweep evaluates all policies at
{0.5×, 1.0×, 1.5×} of that calibrated base. Menu: **Tools ▸ Elevator RL ▸ Run E1 Sweep - Calibrated**.
Confirmed calibrated bases (UpPeak, 2026-07-09, single seed): S≈1.33, M≈0.41, L≈0.29, **Z≈0.017,
H≈0.009** — Z/H need dramatically less nominal intensity to reach the same *relative* saturation,
which is itself informative: zoning reduces effective capacity per unit of demand far more than
the raw floor/car counts alone would suggest.

Traffic axis (crossed with rungs): `TrafficPattern ∈ {UpPeak, DownPeak, Lunch, Midday, Uniform}` —
currently sweeping UpPeak + Lunch; DownPeak/Midday/Uniform not yet run at full ladder scale.
RL's edge should be largest at **UpPeak, high multiple** (heaviest, most directional — where
zoning/anticipation pay off).

> Floor restrictions are already supported: `BuildingConfig.floorRange` (`Vector2Int[]` per car)
> and `MinFloor/MaxFloor`; LOOK and ETA both honor them, and actions are masked to range
> (`WriteDiscreteActionMask`).

---

## 4. Experiment matrix (in priority order)

Each experiment names: the question, the arms, the rung(s), and the primary metric.

### E1 — LOOK baseline characterization *(no training; do this first)*
- **Q:** How does LOOK's avg / p95 / max wait, throughput, and abandonment scale across rungs
  S→H and across all patterns × intensities?
- **Arms:** LOOK (and optionally Zoned-LOOK, ETA) — pure sandbox runs, many seeds.
- **Output:** the reference surface every later result is measured against; also the first
  evidence for/against the thesis (does LOOK's tail wait blow up on Z/H?).

### E2 — RL parity in the easy regime — **DONE, result: PPO beats both baselines on rung S**
- **Q:** Can PPO match LOOK on rung **S**? (Necessary floor; if it can't match here, fix the
  setup before scaling.)
- **Arms:** LOOK vs. PPO (flat MLP). Fixed fleet, no restrictions.
- **Metric:** % of LOOK's throughput and wait at convergence. Target: ≈100%.
- **Result (2026-07-13, run `elev-e2-s-ppo-01`, 5M steps, S preset — 8 floors/3 cars, UpPeak,
  intensity 0.5, seeds 1–5, 3600s/episode after 300s warmup):** PPO didn't just match LOOK, it beat
  both baselines on every metric, consistently across all 5 seeds (no range overlap):

  | | delivered | waitMean | waitP95 |
  |---|---|---|---|
  | LOOK | 1662–1709 | 16.0–17.0s | ~37s |
  | ETA | 1612–1677 | 16.0–17.5s | ~37s |
  | **PPO** | **1815–1837** | **14.0–15.2s** | **~33s** |

  Full per-seed data: `Runs/20260713-112355-E2-sweep-S-UpPeak/e2_sweep_summary.csv`. Eval method:
  `PpoDispatcher` (`Assets/ElevatorRL/Editor/PpoDispatcher.cs`) runs the trained ONNX directly via
  Sentis/InferenceEngine as an `EvalHarness` `Dispatcher`, reusing the real
  `Building.WriteObservation` and the AS0 action-mask logic — so it's evaluated through the exact
  same Building/StatsCollector loop as LOOK/ETA, not a reimplementation.
  **This is a surprising result relative to §0's thesis** (RL should win at scale/zoning, not on
  small rungs — c.f. Crites & Barto's 10-floor/4-car requirement, Appendix). Worth watching whether
  the RL−LOOK gap *grows* through E3's scale ladder (expected) or whether S was already close to
  saturated/informative enough that PPO's edge here doesn't cleanly separate from the scale-driven
  mechanism §0 describes. Single-run caveat: one training seed so far (`elev-e2-s-ppo-01`); the
  5-seed sweep above is on the *eval* side only, not repeated training runs.

### E3 — **The headline: RL vs. LOOK as scale/constraint increases** — PAUSED after S, M; NEXT: resume ladder (rung L) on E6's winning bignet2 (768×4) recipe, which matches LOOK/ETA at M (see E6)
- **Q:** Does the RL−LOOK performance gap grow along S→M→L→Z→H?
- **Arms:** LOOK, (ETA), PPO-best-architecture, per rung.
- **Metric:** relative improvement in avg wait / p95 wait / abandonment vs. LOOK, plotted **as a
  function of rung**. This curve is the paper/blog thesis figure.
- **Focus:** UpPeak & DownPeak at intensity ≥ 1.5, where the mechanism (§0) predicts the largest
  gap. (Note: the S/M results below use the fixed `SmokeIntensity=0.5` for continuity with E2's
  methodology, not yet each rung's calibrated saturation point per §3 — S≈1.33, M≈0.41. At 0.5, M
  is running *closer to* its own saturation edge than S is to its. This is a real methodology gap
  to close before treating the current S→M trend as the final headline curve.)
- **Result so far — rung M reverses from rung S, but mostly closes with more training** (M preset —
  16 floors/5 cars, UpPeak, intensity 0.5, seeds 1–5, same eval protocol as E2):

  | | delivered | waitMean |
  |---|---|---|
  | LOOK | 2054–2164 | 15.7–17.7s |
  | ETA | 2056–2161 | 17.0–18.1s |
  | PPO @ 5M steps | 1526–1629 | 20.7–21.2s |
  | **PPO @ 10M steps** | **1935–1979** | **19.9–20.5s** |

  At the same 5M-step budget as the S run (`elev-e3-m-ppo-01`), PPO delivered ~25% *fewer*
  passengers and waited ~20% *longer* than both baselines — the opposite direction from S. Initial
  read was a hard architecture ceiling (flat 256×2 MLP, no weight sharing across cars, not scaling
  from 3-car to 5-car coordination) — but checking the reward curve shape first showed M's reward
  was still climbing meaningfully faster than S's had by the same point (S flat by 5M; M gained
  ~4300 reward in its last 2.8M steps), arguing for "hadn't finished training" over "can't learn
  this." Resumed the same run to 10M steps (`resume_training.sh`, same architecture/hyperparameters,
  no changes) and re-evaluated: **closed ~65-70% of the delivered-count gap** (was ~500 behind
  LOOK/ETA, now ~150-200 behind), smaller improvement on wait time. So more training steps clearly
  help a lot — this was NOT primarily an architecture ceiling at 5M — but PPO still hasn't caught
  LOOK/ETA even at 10M (2x the S-rung budget), so there's a real remaining gap, and/or diminishing
  returns are setting in (reward was leveling off near the end of the 10M run, less clearly than a
  hard plateau but less steep than earlier). Full data: `sweep_summary.csv` under
  `Runs/20260713-125134-E3-sweep-M-UpPeak/` (5M) and `Runs/20260713-142044-E3-sweep-M-10M-UpPeak/`
  (10M). Open question for the next step: extend further (e.g. 15-20M) to see if it fully closes,
  or treat the remaining gap as evidence the flat-MLP architecture needs help (E6: shared per-car
  encoder / attention) even once given enough time. **Resolved by E6**: neither new architecture
  helped, but a bigger flat MLP (768×4, "bignet2") closed the gap almost entirely at 10M
  (2110.6/5000 vs LOOK's 2119.2) — see E6 for the full writeup.

- **Rung L, resumed on bignet2's recipe (`elev-e3-l-bignet2-01`) — launched 2026-07-15.** L is 8
  cars / 30 floors (vs M's 5 cars / 16 floors) — a genuinely bigger coordination problem, so the
  same open question applies again: does the capacity that closed the gap at M also scale to L, or
  does L need more? Config `config/elevator_ppo_e3_l_bignet2.yaml` (same 768×4 network as
  bignet2), scene re-pointed at `L_BuildingConfig` via `Point Agent At L Preset` + rebuilt headless
  trainer (caught and fixed a stale-scene bug here: the Editor had `TrainingAttention.unity` open
  from earlier E6-B work, so the preset-pointing menu item silently failed against the wrong
  scene's GameObject hierarchy until `Training.unity` was explicitly loaded first — a reminder to
  always verify `get_scene_info` before scene-mutating menu calls, not just check the on-disk
  `.unity` file). 5M steps first, same cheap-test-first protocol as every prior rung/size decision.

- **Rung L 5M-step result — severe gap, much worse than anything seen at M.** Finished cleanly at
  step 5,000,448, reward -32,984 (roughly flat over the last ~1M steps: -32,140 at 4.64M →
  -32,984 at 5.0M, within noise — a much shallower improvement curve than the sharp early drop
  from -40,000 at 1.18M). Eval sweep
  (`Runs/20260715-050128-E3-sweep-L-bignet2-5M-UpPeak/sweep_summary.csv`, 5 seeds, UpPeak):

  | Policy | delivered (mean/5 seeds) | abandoned (mean) | vs LOOK/ETA |
  |---|---|---|---|
  | LOOK | 2683.0 | 1062.6 | baseline |
  | ETA | 2699.2 | ~1050 | baseline |
  | **PPO (bignet2, 5M)** | **893.2** | **2422.0** | **-67%** |

  This is a dramatically worse shortfall than anything at rung M (worst M result was -34%, for
  the failed attention architecture) — bignet2's 768×4 network at 5M steps is nowhere near
  solving L; abandonment is more than double LOOK/ETA's. L is a genuinely bigger coordination
  problem (8 cars/30 floors vs M's 5 cars/16 floors) and reward was still (slowly) improving at
  the cutoff, so extended to 10M via `config/elevator_ppo_e3_l_bignet2_10m.yaml` (same
  cheap-test-first protocol), **but expectations are set explicitly lower than M's result**: M's
  ~25% gap closed with one 4x-capacity jump at 10M steps; L's ~67% gap is unlikely to close with
  a single step-budget doubling on the same network size and may need iterative extensions
  and/or an even bigger network once this run's trend is visible — a materially different
  (harder) story than rung M's clean resolution.

- **Rung L 10M-step result — meaningful progress, but the gap is far from closed.** Finished
  cleanly at step 10,000,000, reward -27,638 (up from -32,984 at 5M — real improvement, but the
  curve was clearly diminishing: -32,140 at 4.64M → -32,984 at 5.0M → -27,638 at 10.0M, i.e. all
  of the 5M-cutoff's apparent flatness turned out to be temporary, but the second 5M-step half
  only bought about as much improvement as expected from a slowing curve, not an accelerating
  one). Eval sweep
  (`Runs/20260715-072031-E3-sweep-L-bignet2-10M-UpPeak/sweep_summary.csv`, 5 seeds, UpPeak):

  | Policy | delivered (mean/5 seeds) | vs LOOK/ETA |
  |---|---|---|
  | LOOK | 2683.0 | baseline |
  | ETA | 2699.2 | baseline |
  | PPO (bignet2, 5M) | 893.2 | -67% |
  | **PPO (bignet2, 10M)** | **1397.2** | **-48%** |

  Doubling the step budget closed roughly 28% of the remaining gap (67%→48%), a real but far
  slower rate of convergence than M showed (M's 256×2→768×4 capacity jump at a *fixed* 10M-step
  budget closed its ~25% gap almost entirely). **This confirms the two rungs are qualitatively
  different problems, not just a bigger instance of the same one** — L is not simply "M but
  slower to converge on the same recipe"; even 15M cumulative steps on the biggest network tried
  leaves L at roughly half of LOOK/ETA's throughput. Decision point reached: continuing to
  iteratively extend steps on bignet2 is the cheapest next lever (consistent with the pattern so
  far — every extension has bought real, if diminishing, improvement) but is unlikely to close a
  gap this size alone; a bigger network specifically sized for L's larger fleet (8 cars, more
  than M's 5) is the more principled next test once/if further step extensions plateau. **Pausing
  the L investigation here to report this finding rather than committing to another 10M-step run
  or architecture change without a decision on how much further budget to invest** — this is a
  natural checkpoint given how much the L result diverges from M's clean resolution.

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
- **Updated 2026-07-15 (per user request): added a 4th arm — Omniscient — and bumped network size
  for all E5 arms.** Four `ObservationConfig` assets now exist
  (`ObservationConfig_FullState`/`_Realistic`/`_RealisticPlusWaitAge`/`_Omniscient`) with menu
  items (`Tools/Elevator RL/E5 Obs Ablations/...`) to swap the agent's obs config and rebake
  `VectorObservationSize`.
  - **New: `omniscientDestinations` observation block** (`ObservationConfig.cs`,
    `Building.cs:ObservationSize`/`WriteObservation`/`WriteDestHistogram`) — an EXACT destination
    histogram for every waiting and in-car rider (2×F×F for hall queues per origin floor/direction,
    + E×F for car riders), not just presence (`hallButtons`) or count (`queueLengths`). No real
    controller has this before boarding; it exists purely to measure the ceiling — how much is
    left on the table if the policy could see exactly who's going where. On rung M this adds 592
    floats to the 293-float full-state baseline (885 total), confirmed via the baked
    `VectorObservationSize` log matching the hand-computed value exactly.
  - **Network size bumped to 1024 hidden units × 5 layers ("bignet3") for ALL four E5 arms**,
    applied regardless of which obs config — per the user's explicit request, not gated on the
    obs-ablation question itself.
  - **Arm 4 (Omniscient) launched first** as `elev-e5-m-omniscient-01`
    (`config/elevator_ppo_e5_m_omniscient.yaml`), 5M steps — this is the highest-value arm (the
    "how much can we beat the current best by" ceiling test), so it's running ahead of Arms 1-3.
    Arms 1 (full-state), 2 (realistic), 3 (realistic+wait-age) — all re-created at the new 1024×5
    size — queued to follow sequentially.
  - **Arm 4 (Omniscient) — 5M INTERIM, INCONCLUSIVE (NOT a ceiling result). CORRECTED
    2026-07-15: my first write-up here concluded "omniscient info didn't help / came in below the
    baseline" — that conclusion was WRONG and is retracted.** Two mistakes: (1) I compared
    omniscient-at-5M against bignet2-at-**10M** (unfair by exactly 5M steps), and (2) I called the
    5M reward a "plateau" and skipped the extend-to-10M protocol, when it was in fact still
    descending ~700 reward/M-steps at the cutoff — I mistook the ±400-900 noise in the raw
    20K-summary rows for convergence.
    - **The clean diagnostic is training reward, which is OBSERVATION-INDEPENDENT** (computed by
      `Building.CollectReward` from sim state, not from what the agent sees), so omniscient and
      bignet2 are directly comparable on reward at matched steps despite different obs/net. Both are
      the rung-M Training scene, same RewardConfig/traffic. Trajectories are intertwined within
      noise the entire way down:
      | Step | bignet2 (768×4, 254-in) | omniscient (1024×5, 885-in) |
      |------|-------------------------|------------------------------|
      | 1M   | -15,929 | -15,667 |
      | 2M   | -11,778 | -13,034 |
      | 3M   | -9,137  | -9,799  |
      | 4M   | -7,926  | -7,796  |
      | 5M   | **-6,932** | **-7,107** |
      | 10M  | **-5,297** | *(not yet run)* |
      At 5M the two are **tied** (within the noise band). bignet2's entire margin comes from its
      5M→10M improvement (-6,932 → -5,297, +1,635 reward) — the exact stretch omniscient never got
      to run. So the eval deficit (1969 vs bignet2's 2110 delivered) is an **undertraining
      artifact**, not evidence about the value of the observation.
    - **Information-theoretically the omniscient obs is a strict superset of full-state**, so at
      convergence it cannot be worse; a worse-looking result is under-optimization by definition.
      Three factors make omniscient converge *slower per step* than bignet2, so it needs AT LEAST
      the full 10M (likely more) before any verdict: (a) **sparse-input dilution** — of the 592
      added floats, under UpPeak nearly all demand originates at the lobby so the vast majority of
      destination-histogram slots are ~always zero, diluting the 293 informative features ~3.5×;
      (b) `normalize:true` running stats over near-constant-zero features add conditioning noise;
      (c) bigger net (1024×5 vs 768×4) = more params to fit.
    - **NEXT: resume `elev-e5-m-omniscient-01` 5M→10M and re-eval** (it was still climbing — this is
      the extend-to-10M protocol I should have followed). Only after the matched-budget comparison
      (omniscient@10M vs bignet2@10M, and vs the Arm-1 full-state@5M control now training) is the
      value-of-omniscience question actually answerable.
    - **Eval-harness bug found and fixed first:** `EvalHarness.RunSingle`/`RunScaleLadderSweep`
      built the PPO eval `Building` with a **fresh default `ObservationConfig`**
      (`ScriptableObject.CreateInstance<ObservationConfig>()`), not the asset the model was actually
      trained with. This happened to be harmless for all prior E2/E3/E6 sweeps because the default
      field values exactly equal the baseline config those arms used — but it silently broke every
      E5 arm, since none of them use the default flags. First eval attempt on the omniscient model
      showed PPO badly losing to LOOK/ETA (delivered ~633–1209 vs. LOOK's ~2100+) purely because the
      885-float input the model expects was being filled from a config with `queueLengths`,
      `timeOfDay`, `pattern`, and `omniscientDestinations` all false — a garbled/misaligned input,
      not a real performance signal. **Fix:** added `obsConfigAssetPath` to `RunScaleLadderSweep` /
      `obsConfigOverride` to `RunSingle` (`EvalHarness.cs`) so PPO's eval `Building` is built with
      the same `ObservationConfig` asset used in training; wired the omniscient sweep menu item to
      `ObservationConfig_Omniscient.asset`. Re-ran and PPO delivered jumped from ~900 to ~1965-2007
      — confirms this was purely an eval-harness bug, not a bad model. **Any future E5 sweep menu
      item (Arms 1-3) must pass its matching `obsConfigAssetPath` or it will silently repeat this
      bug.**
    - **5M interim eval (5-seed UpPeak sweep, rung M, `Runs/20260715-143735-...`):** PPO
      (omniscient, 1024×5, 5M steps) delivered mean **1969.0** vs. LOOK **2119.2** vs. ETA
      **2109.8**. Per-seed: PPO {1965, 2007, 1965, 1971, 1937} vs. LOOK {2134, 2164, 2122, 2122,
      2054} vs. ETA {2107, 2161, 2103, 2122, 2056}. **This is an undertrained-checkpoint number, not
      a ceiling** — see the reward-trajectory analysis above: at 5M the omniscient policy is tied
      with bignet2-at-5M in (observation-independent) reward, and bignet2 gained +1,635 reward
      over its 5M→10M continuation. The comparison to make is omniscient@10M vs bignet2@10M, which
      requires resuming this run. Do NOT cite 1969-vs-2110 as a finished result.

### E6 — Architecture: flat MLP vs. shared per-car vs. attention — **DONE, result: bigger flat MLP matches LOOK/ETA; both new architectures rejected**
- **Q:** Does weight sharing / attention over cars unlock the large-fleet rungs (L/H)?
- **Arms:** flat MLP (baseline), shared per-car encoder, attention.
- **Metric:** performance & sample-efficiency on L and H; this is where scalability is won or lost.
- **TL;DR (full detail below):** Architecture A (multi-agent parameter sharing) never learned —
  first attempt used the wrong trainer for cooperative multi-agent (fixed), second attempt revealed
  a structural flaw instead (per-car agents can't see other cars' state at all, so coordination is
  impossible regardless of trainer) — rejected, not fixable without eroding the design's point.
  Architecture B (BufferSensor attention) trained fine but lost to the flat-MLP baseline at every
  step budget tried. The winning move was the cheap one: just give the *original* flat MLP more
  capacity. Three sizes at 10M steps trace a clean improvement curve — 256×2: -8%, 512×3: -4%,
  **768×4 ("bignet2"): -0.4% vs LOOK, actually beats ETA (2110.6 vs 2109.8 delivered)** — RL
  **matches the classical heuristics at rung M** for the first time in this project. See the full
  comparison table and updated next-step plan near the end of this section.
- **2026-07-13 — paused E3's S→H ladder here.** M's flat-MLP result (5M steps: PPO badly behind
  LOOK/ETA; 10M steps: closed most but not all of the gap) shows the single-agent/flat-observation
  architecture is running out of information/capacity to coordinate 5 cars, and L (8 cars) would
  only be harder. Rather than keep burning ~80min/5M-step runs on a recipe that may not scale,
  moving to E6 now: design and implement a shared per-car architecture before resuming the ladder.
  Two concrete implementation paths to choose between (not yet decided):
  1. **Multi-agent parameter sharing (ML-Agents-native):** replace the single
     `ElevatorControllerAgent` (one flat MultiDiscrete action, one big concatenated observation)
     with N per-car `Agent` instances — one per elevator — all sharing the same Behavior Name.
     ML-Agents automatically ties their policy weights together when the Behavior Name matches, so
     this gets "shared per-car encoder" for free from the platform, no custom network code. Each
     car agent would observe its own local state (floor, direction, its own queue calls) plus
     whatever shared/global state it needs (hall calls building-wide, other cars' positions), and
     emit one 6-way discrete action for itself. Requires deciding: individual vs. shared/team
     reward (shared reward is the more faithful MDP for a dispatch problem where cars must
     cooperate, but can be a harder credit-assignment target), and reworking who owns
     `Building.Tick()`/decision cadence now that no single agent owns the whole loop.
  2. **Custom shared-encoder network, same single-agent structure:** keep one
     `ElevatorControllerAgent`/one MultiDiscrete action, but replace ML-Agents' default flat MLP
     with a custom network (extending `Unity.MLAgents.Extensions` or a custom `NetworkBody`) that
     runs one shared small encoder over each car's observation slice, then concatenates/pools those
     encodings before the shared trunk — closer to the literal "shared per-car encoder" phrasing in
     this section, but needs custom ML-Agents network code (no longer just YAML config), which is
     more implementation risk/time than option 1.

- **Decision (2026-07-13):** implement BOTH, both pure C# / no custom Python (planning discovery:
  the shared-encoder network is achievable via ML-Agents' built-in BufferSensor attention, avoiding
  the torch/ONNX-export fragility). Plan file: `peaceful-giggling-fern.md`.
  - **Architecture A — multi-agent parameter sharing** (N per-car `ElevatorCarAgent` sharing Behavior
    "ElevatorCar", fixed per-car obs, team reward via `BuildingManager`).
  - **Architecture B — single-agent + BufferSensor attention** (global obs → VectorSensor, per-car
    entities → `BufferSensorComponent` → shared per-car encoder + residual self-attention; keeps the
    E-branch joint action). Behavior "ElevatorAttention".

- **Architecture A result (run `elev-e6a-m-ppo-01`, 5M steps, rung M) — FAILED TO LEARN, but the
  setup was mis-specified.** Training reward stayed flat at ~-21000 the entire run (started -20650,
  ended -21190) — no learning, vs the flat MLP climbing to ~-9500 over the same budget. Eval:
  PPO delivered **0** passengers across all 5 seeds (`Runs/20260713-154736-E3-sweep-M-e6a-UpPeak/`).
  Root cause is almost certainly the **wrong trainer**: this used plain `trainer_type: ppo`, i.e.
  independent PPO agents (shared weights) each treating the broadcast team reward as an individual
  reward, with a decentralized critic that sees only one car's local obs. That's a known-weak setup
  for *cooperative* multi-agent — ML-Agents ships **MA-POCA** (`trainer_type: poca` + a
  `SimpleMultiAgentGroup` with `AddGroupReward`/`GroupEpisodeInterrupted`) specifically for this,
  giving a centralized critic over the whole group. So this result is NOT a fair test of "does
  multi-agent parameter sharing help" — it's "independent PPO on a cooperative task doesn't learn,"
  which is expected. Redoing Architecture A with MA-POCA (agent groups + poca) is the correct test.
  (Dispatcher-vs-policy note: delivered=0 alone could be a dispatcher bug, but the flat *training*
  reward — which uses the real `ElevatorCarAgent`, not the eval dispatcher — is independent evidence
  the policy itself never learned. A per-action histogram in `MultiAgentPpoDispatcher` was added to
  confirm the dispatcher reads actions correctly; run it once the CPU is free of the B training.)

- **Architecture B 5M-step result** (`elev-e6b-m-ppo-01`, rung M) — reward climbed from -26,800
  (step 40k) to -10,835 (step 5,000,192) but was still visibly inching up in the last ~1M steps
  (-11.2k → -10.7k, noisy but not flat), the same "still learning, not a capacity ceiling" curve
  shape as the flat-MLP M run before its 5M→10M extension. -10,835 is worse than the flat MLP's
  ~-9500 at 5M, but far ahead of Architecture A's failed flat -21,000. Per the same
  cheap-test-first logic as E3 (extend steps before touching network size), resumed the identical
  run to 10M steps via `scripts/resume_training.sh` + `config/elevator_ppo_e6b_m_10m.yaml`
  (resumed cleanly from step 5,000,192, snapshot at
  `Runs/training/elev-e6b-m-ppo-01/resume_20260714T011221Z/`).

- **Architecture B 10M-step result — training reward climbed to ~-9,948 (step 10,000,128), touching
  the flat-MLP's range** but the **eval sweep tells a different story**. Built `AttentionDispatcher`
  (`Assets/ElevatorRL/Editor/AttentionDispatcher.cs`) for eval — confirmed the exported ONNX's actual
  input contract directly via `onnx.load` (not assumed): `obs_0` is the BufferSensor entity tensor
  `(batch, 5, 38)`, `obs_1` is the global VectorSensor `(batch, 64)` — BufferSensorComponent registers
  before the Agent's own VectorSensor, so the indices are swapped from CollectObservations' call
  order. Ran the standard 5-seed UpPeak sweep
  (`Runs/20260713-205532-E3-sweep-M-e6b-10M-UpPeak/sweep_summary.csv`):

  | Policy              | delivered (mean/5 seeds) | vs LOOK/ETA |
  |----------------------|---------------------------|-------------|
  | LOOK                 | 2119.2                    | baseline    |
  | ETA                  | 2109.8                    | baseline    |
  | Flat-MLP, 5M steps   | 1590.4                    | -25%        |
  | Flat-MLP, 10M steps  | 1958.4                    | -8%         |
  | **Architecture B (attention), 10M steps** | **1406.2**   | **-34%**    |
  | Architecture A (independent PPO, mis-specified) | 0 | -100% (broken, see above) |

  **Architecture B loses to the flat MLP at equal (10M) step budget — and loses to the flat MLP's
  5M-step result too.** Despite reaching a similar *training* cumulative reward (~-9,948 vs the flat
  MLP's ~+2,500–3,500 eval `rwTotal` at 10M — note training reward and eval `rwTotal` aren't the same
  metric/episode length, but the eval numbers are the fair apples-to-apples comparison), attention
  delivers meaningfully fewer passengers with much higher abandonment (~1000-1100 vs LOOK's ~400-460).
  This is the opposite of the motivating hypothesis for E6 — the "shared-encoder + attention"
  architecture, despite the more sophisticated inductive bias, has not translated into better
  dispatch quality than the plain flat MLP at this size, at least not within the same 10M-step
  budget (attention trains ~2x slower per step, so it has effectively seen less optimization per
  wall-clock than the flat MLP already had going into this comparison).

  **Conclusion so far: neither new architecture beats the flat-MLP baseline at rung M.** Architecture
  A's MA-POCA retrain (queued, not yet run — see below) is still an open question; if it also fails
  to beat the flat MLP, the honest read is that rung M's real bottleneck may be step budget /
  reward-shaping rather than architecture, and the cheaper "flat MLP with more capacity" experiment
  the user raised becomes the more promising next lever — see the check-in discussion in-session
  (2026-07-13) about whether a bigger flat MLP is a fair alternative to a new architecture. Not
  concluding this until Architecture A-poca's result is in.

- **Architecture A MA-POCA retrain, attempt 1 (`elev-e6a-m-poca-01`) — FAILED, second
  mis-specification found.** Rebuilt and trained for 5M steps: eval sweep again showed **PPO
  delivered 0 across all 5 seeds**, identical to the original broken independent-PPO attempt.
  Training logs showed `Mean Reward: 0.000` at every one of 223 summary points across the entire
  run, despite `Mean Group Reward` swinging normally (~-20,000 to -22,000) and hundreds of episodes
  completing (ruling out "no episodes finished yet"). Root cause: the MA-POCA conversion
  (`3416aad`) called only `_group.AddGroupReward()`, never `agent.AddReward()` on the individual
  `ElevatorCarAgent`s. Per ML-Agents' own docs this is correct in principle ("group rewards are
  treated differently... not equivalent to calling AddReward() on each agent"), but in practice the
  individual agents were receiving **zero reward signal of any kind** — every one of their own
  per-agent value/advantage estimates had nothing to learn from, likely starving policy learning
  even with a working centralized critic. Fixed (commit `f1eb88f`) by also crediting each in-service
  car with `AddReward(groupReward / numCars)` alongside the existing group reward. Verified the fix
  with a 100k-step smoke test before committing to a full run: `Mean Reward` now tracks
  `Mean Group Reward / 5` almost exactly (e.g. -4070.8 vs -20354.1/5 = -4070.8), confirming the
  wiring is correct this time. (Process note: also hit two infra snags worth flagging for next
  time — a build ran before an in-flight recompile had actually finished, silently baking stale
  code into the headless player; and Unity's MCP bridge stops processing incoming requests entirely
  when the Editor isn't the frontmost app on macOS, requiring an explicit
  `osascript ... set frontmost of process "Unity" to true` before menu-item calls would go through.)

- **Architecture A MA-POCA retrain, attempt 2 (`elev-e6a-m-poca-01-v2`) — STOPPED at 2.4M/5M
  steps, structural flaw found in the observation design, not the trainer.** With the per-agent
  reward fix confirmed active (`Mean Reward` non-zero and tracking `Mean Group Reward / 5`), the
  training reward was still completely flat over the first 2.4M steps (-4050 to -4250 per-agent,
  ~-20,700 to -22,100 group, noisy but no upward trend at all — contrast the flat MLP's climb from
  ~-20,000 to ~-9,500 over the same budget). Two independently-broken trainer setups (plain PPO,
  then POCA with corrected reward routing) producing the *identical* flat-non-learning signature
  pointed at something upstream of the trainer. Reading `Building.WriteCarObservation`
  (`Assets/ElevatorRL/Runtime/Building.cs:481-540`) found it: **each `ElevatorCarAgent` only
  observes its own car's state (floor/motion/load/buttons) plus building-wide shared info (hall
  calls, queues, time, pattern) — it has zero visibility into any *other* car's position, motion,
  or load.** Elevator dispatch is inherently a coordination problem (which car should answer a hall
  call depends on where the other cars already are), so no trainer can teach coordination an agent
  structurally cannot perceive. This is exactly the concern the user raised at the very start of E6
  ("the architecture doesn't have enough information available to learn such a large building
  setup") — Architecture A's per-car design, specifically, reintroduced that problem even though
  Architecture B (which does give every car full visibility into its peers via BufferSensor
  entities) does not have this flaw and was still tested to a clean result. **Architecture A is not
  being retried further** — fixing it would mean adding all-other-cars' state to each car's
  observation, which starts to erode the "fixed per-car obs size independent of fleet" property
  that was the whole point of the parameter-sharing design, and Architecture B already tested the
  "give every car full peer visibility" idea via a different (better-suited) mechanism and still
  lost to the flat MLP.

- **Pivot: bigger flat MLP (`elev-e3-m-bignet-01`) — RESULT: capacity was indeed a real factor,
  and a cheap one to buy.** `config/elevator_ppo_e3_m_bignet.yaml` — identical to
  `elevator_ppo_e3_m.yaml` except `hidden_units: 256→512`, `num_layers: 2→3`. Training reward
  outpaced the original network at every checkpoint sampled (step 2.0M: -12,300 vs original's
  -14,900; step 4.0M: -7,700 vs -10,800), finishing 5M steps at **-7,290** vs the original's
  **~-9,500**. Eval sweep (5 seeds, UpPeak,
  `Runs/20260714-182608-E3-sweep-M-bignet-UpPeak/sweep_summary.csv`) confirms this in delivered
  count, not just reward:

  | Policy                          | delivered (mean/5 seeds) | vs LOOK/ETA |
  |----------------------------------|---------------------------|-------------|
  | LOOK                             | 2119.2                    | baseline    |
  | ETA                              | 2109.8                    | baseline    |
  | Flat-MLP 256×2, 5M steps         | 1590.4                    | -25%        |
  | **Flat-MLP 512×3 ("bignet"), 5M steps** | **1929.6**          | **-9%**     |
  | Flat-MLP 256×2, 10M steps        | 1958.4                    | -8%         |
  | Architecture B (attention), 10M steps | 1406.2               | -34%        |
  | Architecture A (both attempts)  | 0                          | -100% (structurally broken, see above) |

  **The bigger network hits nearly the same delivered count at 5M steps (1929.6) that the original
  256×2 network needed a full extra 5M steps to reach (1958.4 at 10M)** — i.e. more capacity bought
  roughly the same result in about half the training budget, which is a materially better
  cost/benefit than "just train longer" and dramatically better than either new architecture tried
  in E6. This is the strongest evidence yet that rung M's flat-MLP gap vs. LOOK/ETA was at least
  partly a genuine capacity limit, not solely an inductive-bias/architecture problem — reinforcing
  the user's original instinct when this pivot was proposed.

  **Bignet-10M result — RL nearly closes the gap with LOOK/ETA for the first time in this whole
  investigation.** Extended the identical run to 10M steps via `scripts/resume_training.sh` +
  `config/elevator_ppo_e3_m_bignet_10m.yaml` (resumed cleanly from step 5,000,192, snapshot at
  `Runs/training/elev-e3-m-bignet-01/resume_20260715T012223Z/`). Training reward kept improving but
  with clearly diminishing returns (-7,290 at 5M → -6,700 at 7M → -6,100 at 8.3M → -5,713 at 10M —
  each ~1.5M-step stretch bought roughly half the previous stretch's gain, a textbook
  diminishing-returns curve rather than a hard wall). Eval sweep
  (`Runs/20260714-231218-E3-sweep-M-bignet-10M-UpPeak/sweep_summary.csv`) confirms the reward gain
  translated into real dispatch quality:

  | Policy                          | delivered (mean/5 seeds) | vs LOOK/ETA |
  |----------------------------------|---------------------------|-------------|
  | LOOK                             | 2119.2                    | baseline    |
  | ETA                              | 2109.8                    | baseline    |
  | Flat-MLP 256×2, 5M steps         | 1590.4                    | -25%        |
  | Flat-MLP 512×3 ("bignet"), 5M steps | 1929.6                 | -9%         |
  | Flat-MLP 256×2, 10M steps        | 1958.4                    | -8%         |
  | **Flat-MLP 512×3 ("bignet"), 10M steps** | **2031.8**         | **-4%**     |
  | Architecture B (attention), 10M steps | 1406.2               | -34%        |
  | Architecture A (both attempts)  | 0                          | -100% (structurally broken) |

  **This is the closest any RL policy has come to LOOK/ETA at rung M across the entire E3/E6
  investigation** — within ~4%, versus the original network's ~8% gap at the same 10M-step budget.
  Confirms decisively that (a) the E6 architectures (multi-agent parameter sharing, BufferSensor
  attention) were not the right lever for this problem, and (b) the flat MLP's rung-M shortfall was
  real capacity, closeable with a bigger network and more steps — exactly the user's original
  hypothesis. Given diminishing returns were already visible by 10M, an even-bigger network is
  worth trying as the next increment but should be expected to close some, not necessarily all, of
  the remaining ~4% gap. **Recommended next steps:** (1) try one step bigger (e.g. 768–1024 hidden
  units and/or 4 layers) on rung M to see if the gap keeps shrinking or has plateaued; (2) if it
  does keep closing, resume the E3 scale ladder (L, then Z, then H) directly on the winning
  bigger-flat-MLP recipe rather than either E6 architecture, per the plan's original step 4.

- **E6-bignet2 5M-step result (`elev-e3-m-bignet2-01`, 768 hidden units × 4 layers) — still
  climbing, extended to 10M.** Trailed bignet-1 (512×3) through the middle of training (e.g. step
  3.44M: -8,896 vs bignet-1's -7,976 at the same step — larger network needing more samples to get
  going, as expected), but caught up and passed it by the 5M cutoff: **-6,932 vs bignet-1's -7,290
  at 5M.** Still clearly improving in the last ~1M steps (-7,900 → -6,932, a bigger gain than
  bignet-1 showed at its own 5M mark), so extended to 10M via `scripts/resume_training.sh` +
  `config/elevator_ppo_e3_m_bignet2_10m.yaml` (resumed cleanly from step 5,000,192).

- **E6-bignet2 10M-step result — RL matches LOOK/ETA at rung M for the first time in this entire
  project.** Finished cleanly at step 10,000,000, reward -5,298 (vs bignet-1's final -5,713, and
  consistently ahead of bignet-1 at every equivalent step past 5M — e.g. -6,387 vs -6,871 at
  6.76M). Eval sweep
  (`Runs/20260715-023432-E3-sweep-M-bignet2-10M-UpPeak/sweep_summary.csv`):

  | Policy                                    | delivered (mean/5 seeds) | vs LOOK/ETA |
  |---------------------------------------------|---------------------------|-------------|
  | LOOK                                         | 2119.2                    | baseline    |
  | ETA                                          | 2109.8                    | baseline    |
  | Flat-MLP 256×2, 5M                           | 1590.4                    | -25%        |
  | Flat-MLP 512×3 ("bignet"), 5M                | 1929.6                    | -9%         |
  | Flat-MLP 256×2, 10M                          | 1958.4                    | -8%         |
  | Flat-MLP 512×3 ("bignet"), 10M               | 2031.8                    | -4%         |
  | **Flat-MLP 768×4 ("bignet2"), 10M**          | **2110.6**                | **-0.4%, beats ETA** |
  | Architecture B (attention), 10M              | 1406.2                    | -34%        |
  | Architecture A (both attempts)               | 0                          | -100% (structurally broken) |

  Individual seeds show PPO essentially tied with LOOK on 4/5 seeds (2134/2134, 2161/2161,
  2107/2122, 2095/2122) and ahead of ETA on all 5. Waits run slightly longer on average
  (`waitMean` ~20s vs LOOK/ETA's ~17s) but abandonment is markedly *lower* (290-374 vs LOOK/ETA's
  ~390-466) — the policy is trading a bit of average wait for fewer people giving up entirely,
  which nets out to matching total throughput. **This closes out the capacity-scaling question
  definitively**: three network sizes (256×2, 512×3, 768×4) at 10M steps trace a clean, still-not-
  fully-saturated improvement curve (-8% → -4% → -0.4%), confirming the user's original hypothesis
  that rung M's flat-MLP shortfall was capacity, not architecture.

- **Agreed follow-on sequence (2026-07-14), updated after bignet2's result:** bignet2 essentially
  closed the rung-M gap, so per the plan's original E6 step 4, the natural next move is **resuming
  the E3 scale ladder (rung L, then Z, then H) directly on the bignet2 (768×4) recipe** rather than
  continuing to chase marginal gains at rung M with an even bigger network — rung M is no longer
  the bottleneck. Remaining queued items, now reordered: (1) **resume E3 ladder** on bignet2's
  recipe (rung L next); (2) **E5** — observation ablations (cheap, config-only); (3) **E10** —
  reward shaping ablations (wait-min, in-car-time-min, average-vs-longest-wait via a quadratic wait
  penalty, throughput-only control, abandonment-averse — see E10 below for the full arm list); (4)
  **E11** — revisit Architecture B (attention) with lessons learned, since losing to bignet doesn't
  necessarily mean attention is a dead end for this problem, just that this particular
  setup/budget lost this particular comparison. E5/E10/E11 remain valuable but are no longer
  gating the headline scale-ladder question the way they would have been if bignet2 had plateaued.

### E7 — Fleet-size generalization
- **Q:** Does one policy trained with randomized/curriculum fleet size generalize across fleet
  sizes vs. fixed-fleet specialists?
- **Arms:** curriculum (per `elevator_ppo.yaml`) vs. randomized-active vs. per-size specialists,
  evaluated across in-service counts. Robustness sub-test: mid-episode service changes
  (`serviceChangeProbability > 0`).
- **Metric:** performance vs. specialists across the fleet-size axis; graceful degradation.

### E8 — Stretch
- ~~Fairness reward~~ — promoted to its own section, see E10.
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

### E10 — Reward shaping ablations *(queued: after E6-bignet2, alongside/before E5)*
- **Q:** Does *what* the reward optimizes for (not just how much capacity the policy has) change
  which policy emerges — specifically average-wait vs. tail/longest-wait, and hallway-time vs.
  in-car-time?
- **Current baseline weights** (`RewardConfig.cs`): `delivered=+10`, `movedToward=+0.4`,
  `movedAway=-0.4`, `rejected=-5`, `abandoned=-8`, `inElevator=-0.04`/rider-second,
  `inQueue=-0.12`/passenger-second. `inQueue`/`inElevator` are both **linear** in time — a
  policy that makes 10 people wait 10s each scores identically to one that makes 1 person wait
  100s, i.e. the current reward is already an *average*-wait-style objective; it does not
  distinguish average from tail/longest wait at all.
- **Arms (each a new `RewardConfig` asset + training config, same architecture/protocol as
  whatever's currently winning — bignet-10M as of 2026-07-14 — so reward is the only variable)**:
  1. **Wait-minimizing** — up-weight `inQueue` substantially (e.g. ×3-5), zero or shrink
     `inElevator`. Framing: "get people out of the hallway; a slower ride once boarded is fine."
  2. **In-car-time-minimizing** — the inverse: up-weight `inElevator`, shrink `inQueue`. Framing:
     "once picked up, get them there fast; longer hallway waits are acceptable."
  3. **Average vs. longest wait (fairness)** — replace/augment the linear `inQueue` term with a
     **quadratic** wait penalty: accumulate `sum_i(waitTime_i) * dt` per passenger per tick
     (approximating d/dt(waitTime²)) instead of the current flat `waiting_count * dt`
     (`Building.AgeOccupants`/`AgeQueue`, `Assets/ElevatorRL/Runtime/Building.cs:247-274`) — this
     makes the marginal cost of waiting grow with how long someone's already waited, directly
     penalizing a long tail even if it lowers average wait less. Needs: a new `Acc` field (e.g.
     `queueSecondsWeighted`), the accumulation change in `AgeQueue`, a new `RewardConfig` weight,
     and wiring into `CollectReward()`. Compare mean wait vs. `waitP95`/`waitMax` (already tracked
     in `StatsCollector`/eval CSVs) against the linear baseline — the quadratic arm should trade
     some average-wait for a lower p95/max.
  4. **Throughput-only (control)** — zero `inQueue` and `inElevator` entirely; reward purely on
     `delivered`/`movedToward`/`movedAway`/`rejected`/`abandoned`. Tests how much the continuous
     occupancy penalties (vs. sparse event reward alone) actually matter for shaping useful
     mid-episode behavior — a cheap ablation, not expected to win, but informative.
  5. **Abandonment-averse** — up-weight `abandoned` further and/or add an early-warning term for
     riders approaching `maxWait` (before they actually abandon), testing whether anticipatory
     shaping beats reacting only after the fact.
- **Metric:** delivered count (throughput) + `waitMean` vs. `waitP95`/`waitMax` (average vs. tail)
  per arm, same 5-seed UpPeak sweep protocol as E6. Expect a real throughput/fairness trade-off in
  arm 3, not a free lunch.

### E11 — Improved attention architecture *(queued: after E10, revisiting E6 Architecture B)*
- **Q:** Architecture B (BufferSensor attention) trained without errors but lost to the flat MLP at
  every step budget tried (E6: -34% vs LOOK/ETA at 10M, vs bignet's -4%) — is that a fundamental
  ceiling for attention on this task, or did the *specific* attention setup just need more
  tuning/capacity of its own?
- **Candidate changes to try** (cheap to expensive):
  1. **Bigger attention network_settings** — same capacity bump that worked for the flat MLP
     (hidden_units/num_layers) applied to `elevator_ppo_e6b_m.yaml`'s trunk — attention's per-car
     entity encoder + residual self-attention block sizes are also configurable; untested whether
     they were undersized rather than the mechanism being wrong.
  2. **Longer training relative to its slower per-step cost** — attention trains ~2x slower per
     step than the flat MLP; E6-B's 10M-step budget may still be the *equivalent* of the flat
     MLP's ~5M in wall-clock/gradient-update terms. Worth an extended run once bignet2/E10 establish
     what budget actually saturates performance for this task.
  3. **Positional/identity encoding per car entity** — current per-car entities are permutation-
     invariant by design (`Building.WriteCarEntity`), which may be *discarding* useful information
     (e.g., car index is stable across an episode and correlates with zone assignment in Z-rung
     configs) — worth testing whether adding a car-index feature to each entity helps without
     reintroducing full order-dependence.
  4. **Combine with E10's reward findings** — if reward shaping alone materially changes learned
     behavior, re-run the best attention config against the best reward config before concluding
     architecture was the limiting factor rather than reward signal.
- **Metric:** same 5-seed UpPeak sweep protocol; compare against whatever the current best flat-MLP
  recipe is at the time (bignet2 or later), not just the original E6-B result.

### E12 — Traffic-pattern realism: interfloor / two-way / day-cycle *(NEXT — promoted ahead of E7-E11)*
- **Motivating discovery (2026-07-15):** **every experiment to date (E2-E6) trained AND evaluated on
  UpPeak only** — `Config/TrafficConfig.asset` has `defaultPattern: 0 (UpPeak)`, `useDayCycle: 0`,
  and `EvalHarness.RunScaleLadderSweep` hardcodes `const TrafficPattern pattern =
  TrafficPattern.UpPeak`. UpPeak is a **single-lobby-hub** pattern, which is precisely the case where
  greedy collective control (LOOK) is near-optimal *by construction* — so "RL only matches LOOK/ETA
  at rung M" (E6) was measured on the heuristics' best-case home turf and the *least* favorable
  pattern for a learned policy. This likely explains the whole "parity, not victory" story.
- **Q:** Does the RL-vs-LOOK/ETA gap **invert** (RL opens a real lead) under interfloor and two-way
  traffic, where destinations/origins are distributed building-wide and greedy dispatch leaves the
  most on the table? And does a single day-cycle **generalist** hold up across patterns vs. a real
  building's shifting demand?
- **Grounding (literature, see 2026-07-15 research):**
  - Traffic is a standard 3-way mix incoming:outgoing:interfloor. CIBSE Guide D lunch template
    **45:45:10**, Barney & Al-Sharif **40:40:20**; interfloor is typically **10-30%** of peak demand
    and *dominant* in **residential, hospital, hotel, mixed-use** buildings. UpPeak is a worst-case
    *sizing* scenario, not typical daily operation.
  - The canonical RL-elevator result (**Crites & Barto, NeurIPS 1996**; Sutton & Barto §11.4) trained
    on a **down-peak profile WITH up + interfloor traffic**, 10 floors / 4 cars — RL's demonstrated
    advantage was never on pure up-peak. The 2024 *Adv. Eng. Informatics* "Traffic Pattern-Aware
    Elevator Dispatching via Deep RL" (D3QN/SMDP) makes traffic-pattern-awareness across
    up/down/lunch/interfloor its central thesis.
  - **Destination Control Systems** (Schindler Miconic/PORT, Otis Compass, KONE, TK) take the rider's
    destination at a lobby kiosk/keycard **before boarding** (~25% trip-time / ~30% capacity gains),
    so the E5 `omniscientDestinations` HALL block (2×F×F) is a **realistic DCS signal**, not purely
    hypothetical — and DCS is most valuable exactly in interfloor-heavy buildings. (Tooltip in
    `ObservationConfig.cs` corrected accordingly.) The in-car E×F block remains beyond-DCS ceiling.
- **What's already in place (no new sim code needed for the patterns themselves):**
  `PassengerArrivals.LoadPattern` already implements all five: **UpPeak** (lobby→up, weak interfloor),
  **DownPeak** (up→lobby, weak interfloor), **Lunch** (genuine two-hub two-way: lobby+top),
  **Midday/Uniform** (pure building-wide interfloor — every origin→every dest equally). `useDayCycle`
  already auto-selects the pattern by sim time. The missing pieces are plumbing, not simulation.
- **Arms (decided 2026-07-15: BOTH specialists + generalist, sizes M and L):**
  1. **Per-pattern specialists** — train one policy per {UpPeak (have for M via bignet2), DownPeak,
     Lunch, Interfloor(Midday/Uniform)} × {M, L}, each on its single fixed pattern. Isolates the
     per-pattern RL-vs-heuristic ceiling. Expectation: near-parity on UpPeak/DownPeak (single-hub),
     **RL lead emerges on Lunch and especially Interfloor.**
  2. **Day-cycle generalist** — one policy per size {M, L} trained with `useDayCycle: 1` so it sees
     all patterns within an episode. Tests the realistic single-controller case; compare to the
     specialists (does one policy match N specialists, à la the 2024 unified-model result?).
  - Fold in **E8's "pattern transfer"** here: eval each specialist on the OTHER patterns to measure
    brittleness (e.g. UpPeak-specialist under Interfloor) — the generalist should dominate that
    off-pattern regime.
- **Implementation steps (plumbing):**
  1. **Parametrize the eval harness on pattern:** add a `TrafficPattern` arg to `RunScaleLadderSweep`
     (replace the hardcoded UpPeak) + per-pattern menu items; for the generalist add a full day-cycle
     eval episode (`useDayCycle` on) reporting per-pattern-segment stats. LOOK/ETA are re-run on each
     pattern as the per-pattern baseline (they need no retraining).
  2. **Training configs:** per-pattern `TrafficConfig` assets (or a config field) for the specialists;
     a day-cycle `TrafficConfig` for the generalist. Same 1024×5 net + PPO hypers as bignet2/E5 so
     traffic pattern is the only variable.
  3. **Re-evaluate the E5 omniscient arm here** — its natural proving ground. Run both framings:
     (a) full `omniscientDestinations` (DCS + beyond), (b) hall-only DCS-realistic variant (drop the
     in-car E×F block — needs a small `ObservationConfig`/`Building.WriteObservation` toggle).
- **Metric:** per-pattern 5-seed sweeps, delivered + waitMean + **waitP95/waitMax** (tail matters more
  under interfloor) vs. LOOK/ETA, per rung. **Headline figure:** RL−LOOK gap as a function of traffic
  pattern (expect the gap to swing from ~0 on UpPeak to clearly positive on Interfloor). Secondary:
  generalist-vs-specialists, and omniscient/DCS-obs lift by pattern.
- **Note:** this reframes E3's "scale ladder" headline — the honest thesis figure is now RL−LOOK gap
  over *both* axes (building size AND traffic pattern), not size alone. E10 (avg-vs-tail-wait reward)
  becomes more meaningful after E12 and should follow it.
- **Decisions (2026-07-15):** recipe = **bignet2 (768×4, baseline obs)** so traffic pattern is the
  only variable vs. the UpPeak headline — meaning **UpPeak specialists = reuse existing bignet2**
  (M: `ElevatorController_M_e3_bignet2_10m.onnx`; L: `..._L_e3_bignet2_10m.onnx`), no retrain. The
  traffic pattern is injected at runtime via env params (see infra commit `fdf276e`), so **one
  headless build per size serves all patterns**. Interfloor = **Midday (enum 3)** (uniform arrivals,
  all-to-all destinations). Sizes M and L; both specialists AND a day-cycle generalist per size.
- **EXECUTION QUEUE / STATUS (update after each step; autonomous loop resumes from here):**
  - Infra: runtime override + eval param + 4 YAMLs (`config/elevator_ppo_e12_{downpeak,lunch,interfloor,daycycle}.yaml`) — DONE, committed `fdf276e`. Override verified in headless: `[E12] traffic override active: pattern=Midday` (env param traffic_pattern=3 → Midday), and interfloor reward scale (~-41k@40k) differs from UpPeak (~-26k@40k), confirming the pattern really changed.
  - M build (baseline obs, VectorObservationSize=254, 5 branches) — DONE (`Builds/HeadlessTrainer/RLevatorTrainer.app`, 15:14 mtime).
  - **`elev-e12-m-interfloor-01`: DONE, training complete at 10M steps** (5M then extended per protocol — reward was still climbing at the 5M cutoff: -29,263→-17,277; the last 500K-step delta was ambiguous vs. noise so extended). Final reward -13,652, clearly plateaued over the last 1M steps (oscillating -13,300 to -14,700 with no further trend) — this is convergence. Model copied to `Assets/ElevatorRL/Models/ElevatorController_M_e12_interfloor_10m.onnx`, eval menu item added (`EvalHarness.cs`, `RunE12SweepMInterfloor`, pattern=Midday, obsSize=254).
  - **[RESOLVED 2026-07-15 ~19:33] Unity MCP bridge blocker.** User manually focused/clicked the Unity Editor window; bridge responded immediately after (`get_scene_info` succeeded on the next call). Confirms this was an OS-level focus/App Nap issue, not a stuck dialog — no code or config change needed. Noted for future sessions: if the bridge goes unresponsive (alive, idle, sockets connected, zero requests logged) after the Editor has been backgrounded a long time, ask the user to click the window before spending more time on autonomous diagnosis.
  - **`elev-e12-m-interfloor-01` eval — DONE. Result: SURPRISING, opposite the working hypothesis.** 5-seed sweep, rung M, interfloor (Midday) traffic, `Runs/20260715-193332-E3-sweep-M-e12-interfloor-10M-Midday/sweep_summary.csv`:

    | Policy | delivered (mean) | abandoned (mean) | rejected | waitMean |
    |---|---|---|---|---|
    | LOOK  | 2619.8 | 1298.4 | 0 | 19.3s |
    | ETA   | 2680.8 | 1239.0 | 0 | 19.5s |
    | PPO   | **2241.8** | **1675.8** | 0 | 19.8s |

    **PPO is 14.4% behind LOOK and 16.4% behind ETA** — a clearly worse gap than UpPeak's near-parity (bignet2: -0.4% vs LOOK, actually beat ETA). This directly contradicts the working hypothesis that interfloor/complex traffic is where RL should have the most room to beat greedy heuristics. The gap is driven almost entirely by **abandonment**, not rejection: nobody is turned away in this pattern (`rejected=0` for all three policies — interfloor's distributed demand apparently never saturates any single queue to the reject threshold), but PPO lets ~30% more riders time out waiting than LOOK/ETA does. PPO's reward is even negative (-1800 to -3451) vs. LOOK/ETA's positive 5000-8000 range, computed under the identical reward formula — so this isn't a metric-choice artifact, PPO is genuinely worse on every axis here.
    - **This result should be trusted, not dismissed as undertraining** (learning from the omniscient-arm mistake above): the interfloor model trained cleanly to a converged 10M-step plateau (reward flat over the last 1M steps), matching the exact protocol that produced the winning bignet2 UpPeak result. No step-budget confound this time.
    - **Working hypotheses for why (not yet tested):** (a) Midday's higher, more uniform arrival rate (`lambda=0.15` at every floor vs. UpPeak's `0.05` background + `0.9` lobby spike) may be pushing the whole system into a higher-utilization regime where LOOK's simple "car goes toward nearest unserved call" is closer to optimal and RL's learned policy — tuned on 5M-10M steps of a qualitatively different (hub-and-spoke) traffic distribution's local optima — hasn't found the right coordination strategy for spread-out demand; (b) the flat/joint MultiDiscrete action space (E6's winning architecture) may cope worse with interfloor precisely because there's no single obvious hub to greedily prioritize — more genuinely worth architecture study (E6 revisited, or E9's target-floor semi-MDP) than the flat action space handles well; (c) simple undertrained-relative-to-task-difficulty (10M steps may just not be enough for THIS harder traffic mix, even though the reward curve looks converged — a converged reward doesn't rule out a converged-to-a-worse-local-optimum scenario). Do not resolve this speculatively — gather the other 3 M patterns first (lunch/downpeak/daycycle) to see whether "PPO worse than heuristics" is interfloor-specific or a broader pattern-mismatch story, then revisit.
  - [QUEUED, M, in order] lunch → downpeak → daycycle. Each: 5M first (extend to 10M if still climbing), then eval on its OWN pattern vs LOOK/ETA, document here, commit.
  - [THEN] point agent at L preset (rebake to VectorObservationSize=648, 8 branches) → build L → run interfloor/lunch/downpeak/daycycle at L.
  - Eval reminder: each PPO sweep uses baseline obs (default ObservationConfig = harness default, so no obsConfigAssetPath needed) but MUST pass the matching `pattern` arg to `RunScaleLadderSweep` (add a per-(model,pattern) menu item as each model lands). UpPeak reference numbers already on file (M bignet2: PPO 2110.6 / LOOK 2119.2 / ETA 2109.8). Eval output dir naming fixed (`EvalHarness.cs`) to show the actual pattern instead of a hardcoded "-UpPeak" suffix.
  - Headline so far: RL−LOOK delivered gap per pattern × size — **UpPeak: ~0% (parity/slight win) · Interfloor: -14.4% (clear loss)**. Opposite the hypothesized direction; more patterns needed before drawing a conclusion.

### E13 — Sequence/spatial architectures for the interfloor gap *(IN PROGRESS)*
- **Trigger:** E12 interfloor loss (-14.4% vs LOOK). Hypothesis under test: the flat MLP over a
  single-snapshot, concatenated observation lacks the inductive bias to handle spread-out interfloor
  demand — it can't reason about (a) *temporal* context (how demand is evolving) or (b) *spatial*
  floor-adjacency (a call 1 floor from a car vs. 10 floors away look like arbitrary vector indices).
- **Literature grounding (from the 2024 Traffic-Pattern-Aware D3QN paper's source + Crites&Barto):**
  - The 2024 paper's convolution is **spatial (Conv1d over the FLOOR axis)**, NOT temporal. Neither
    it nor Crites&Barto feed any per-tick history — both are single-snapshot Markovian designs. Their
    "temporal grouping with gradient surgery" is a *multi-task training trick* for unified
    all-patterns training, not a sequence architecture.
  - Both use a **single shared-weight network + a global/team reward** for joint per-car decisions
    (2024: one Q-net outputs `q[car_id*5:(car_id+1)*5]` for all cars; C&B: one shared value fn per
    car; reward accumulates into one controller-level `self.con.reward`). **This is exactly our
    current design** — we are NOT missing "independent per-car agents"; we already tried the more
    decentralized version (E6 Architecture A, per-car parameter-shared agents) and it failed to learn.
  - So the genuine architectural gap vs. us is the **conv-over-floors encoder** + explicit
    arrival-rate channels, not the agent/reward structure.
- **E13a — recurrent memory (LSTM) probe [RUNNING].** Cheapest test of "does temporal history help,"
  even though precedent says the task is Markovian. `elev-e13-m-interfloor-memory-01`
  (`config/elevator_ppo_e13_interfloor_memory.yaml`): identical to the failed interfloor run + ML-Agents
  built-in recurrent policy (`network_settings.memory`, seq_len 64 / mem 128). Pure config change — no
  C#, no new build, runs against the existing M headless build. If it matches the -14.4% baseline →
  the task really is Markovian and temporal memory is a dead end; a clear gain → history matters.
  Launched 2026-07-15 ~20:20. **RESULT: ABANDONED as a negative result at 1.9M/5M steps (killed
  2026-07-15 ~21:15 per user — "not worth the time").** The LSTM tracked *consistently and wideningly
  BEHIND* the plain baseline at matched steps — memory −33.5k @ 1M / −31.7k @ 1.9M vs. baseline
  −29.3k @ 1M / −24.9k @ 2M — i.e. temporal memory made convergence slower, not better, with no sign
  of closing the gap. **Conclusion: the interfloor dispatch task is effectively Markovian** (matches
  Crites&Barto and the 2024 D3QN paper, both single-snapshot); per-tick history is not the missing
  ingredient. Reward-trajectory comparison at matched steps was decisive enough to not spend the full
  5-10M budget. The `RecurrentPpoDispatcher` + eval wiring built for it are kept (cheap, and document
  the recurrent-eval gotcha below) but no model was evaluated. **Pivot: E13b spatial conv is now the
  primary architecture bet.**
  - *(Eval gotcha, retained for reference)* a recurrent policy needs its hidden state threaded through
    each inference step (feed `recurrent_in`, carry `recurrent_out`); the stateless `PpoDispatcher`
    would feed zeroed memory every step and behave degenerately (same failure class as the E5
    obs-config bug) — hence `RecurrentPpoDispatcher` exists if we ever revisit recurrent policies.
- **E13b — floor-axis spatial conv [CODE BUILT, not yet trained].** The native analog of the paper's
  Conv1d-over-floors, done WITHOUT custom torch (avoiding this project's prior onnxscript/ONNX-export
  fragility) by emitting per-floor features as a visual grid so ML-Agents' built-in CNN convolves over
  the floor axis. Implemented this session:
  - `Building.WriteFloorGrid(ObservationWriter)` + `Building.FloorGridFeatures=8` — per-floor state as
    a (channels=1, height=F, width=8) grid: {hallUp, hallDown, upAge, downAge, upQlen, downQlen,
    carsHere, carsWantHere}, same normalizations as the flat obs.
  - `FloorGridSensor : ISensor` (pull-based, reads the live Building each Write) +
    `FloorGridSensorComponent : SensorComponent` — drop the component next to the existing
    `ElevatorControllerAgent` (no agent change; verified Agent.Initialize runs before
    InitializeSensors so `agent.Sim` is live at CreateSensors time). The grid feeds ALONGSIDE the flat
    VectorSensor (ML-Agents concatenates CNN + vector encoder outputs), so flat obs keeps carrying
    car/global state.
  - Editor menu `Tools/Elevator RL/E13 Conv/Add|Remove Floor-Grid Sensor To Agent`; config
    `config/elevator_ppo_e13_interfloor_conv.yaml` with `vis_encode_type: match3` (small CNN, min-res
    5 — OK for F>=5 floors x 8 feats).
  - **Design caveat:** ML-Agents visual encoders are 2D, so the (F x 8) grid gets a 3x3 conv that also
    mixes adjacent (arbitrarily-ordered) features — a minor impurity vs. a true 1D floor conv; the
    floor-axis locality (the point) is captured.
  - **DE-RISK COMPLETE (2026-07-15 ~21:20).** Attached the sensor, rebuilt headless, ran a 30k-step
    smoke train — it (a) trained with no observation-spec error (the visual grid + match3 encoder is
    accepted end-to-end) and (b) exported a conv ONNX cleanly, resolving the "does match3 export to
    ONNX" open risk. `scripts/inspect_onnx.py` on the smoke model confirmed the input contract:
    **`obs_0` = (batch, 1, 16, 8) NCHW visual grid, `obs_1` = (batch, 254) flat vector** (grid takes
    obs_0 exactly as predicted from sensor-name sort). Fixed `ConvDispatcher` from its NHWC guess to
    the confirmed NCHW `(1,1,F,8)` (`FillFloorGrid`'s h-major layout already matches). Ran the conv
    eval sweep against the smoke model: **Sentis multi-input inference runs with no crash** (PPO
    delivered=0 is just the untrained 30k-step policy — its training reward was still at the ~-41k
    init level; the inference PATH is proven). Full conv eval pipeline is now end-to-end validated.
  - **FULL RUN LAUNCHED: `elev-e13-m-interfloor-conv-01`** (2026-07-15 ~21:24,
    `config/elevator_ppo_e13_interfloor_conv.yaml`, 5M, interfloor/Midday). Healthy at 40k
    (reward -40,770, init scale). NOTE the CNN makes it ~2.6x slower than the flat baseline (~300
    steps/s vs ~800), so 5M ≈ 4.7h. **Key comparison (does NOT even need the eval dispatcher):** conv
    training reward vs the baseline interfloor curve at matched steps — baseline hit -29.3k @ 1M,
    -24.9k @ 2M, converged ~-13.7k @ 10M; if conv tracks meaningfully ABOVE that, the floor-adjacency
    inductive bias is helping. Then eval on Midday vs LOOK/ETA (menu `E13 Conv/Run Sweep ...PPO-conv`;
    a good training reward that evals to ~0 would signal a residual ConvDispatcher obs-value bug to
    debug, same class as the E5 bug).
  - **RESULT (2026-07-16 ~01:11): conv beats the flat MLP decisively ON TRAINING REWARD — but see the
    EVAL section below, which does NOT (yet) confirm this on delivered passengers.** Completed 5M steps. On the
    (observation/dispatcher-independent) training-reward axis, the conv led the flat MLP at *every*
    matched step by ~+3-4.5k reward, the whole run:
    | Step | flat MLP | conv | conv adv |
    |------|----------|------|----------|
    | 500k | -34,214 | -30,233 | +3,981 |
    | 1.04M | -28,134 | -24,729 | +3,405 |
    | 1.66M | -26,918 | -22,619 | +4,299 |
    | 2.38M | -24,173 | -19,707 | +4,466 |
    | 3.1M  | -20,947 | -17,277 | +3,670 |
    | 3.9M  | -19,362 | -16,360 | +3,002 |
    | 5M    | **-17,277** | **~-15,100** | +~2,200 |
    Headline framing: **conv @3.1M (-17,277) already equals the flat MLP's FULL 5M value** → ~38%
    fewer samples to reach the same performance; and conv @5M (~-15,100) surpasses the flat MLP's 5M
    ceiling, landing roughly where the flat MLP was at ~7-8M. This is the interfloor gap's first real
    dent from the architecture side (vs. the LSTM which made it worse). Model saved
    `results/elev-e13-m-interfloor-conv-01/`.
  - **EVAL OBTAINED 2026-07-16 (stale-import fix worked) — AND IT COMPLICATES THE WIN. Read this
    before citing the training-reward result above.** A fresh Editor imported the model correctly and
    the conv eval ran properly (PPO delivers ~1,900-2,055, so the earlier `delivered=0` is confirmed
    to have been the stale-import artifact and `ConvDispatcher` runs end-to-end). 5-seed Midday sweep
    (`Runs/20260716-174643-...`):
    | policy | delivered (mean) | abandoned | eval rwTotal | vs LOOK |
    |---|---|---|---|---|
    | LOOK | 2619.8 | 1298 | +5k..+7k | — |
    | ETA | 2680.8 | 1239 | +5.7k..+8.3k | +2.3% |
    | flat MLP **@10M** | 2241.8 | 1676 | -1.8k..-3.5k | -14.4% |
    | **conv @5M** | **1978.4** | **1935** | **-7.9k..-10.7k** | **-24.5%** |
    **The conv DELIVERS 11.7% FEWER than the flat MLP and its eval reward is far worse — the opposite
    of the training-reward story.** Two candidate explanations, not yet separated:
    1. **Unfair budget:** conv=5M vs flat=10M. The flat run's 5M checkpoint was PRUNED
       (`keep_checkpoints: 5` kept only 8.5M-10M), so a matched flat@5M eval is not available without
       retraining — hence the 10M conv extension below is the decisive test.
    2. ~~**A subtle ConvDispatcher grid mismatch**~~ — **RULED OUT 2026-07-16 by direct inspection of
       the actual serialization path** (no need to infer it from the 10M run). Training writes the grid
       via `ObservationWriter[ch,h,w]`, whose flat index is
       `TensorExtensions.Index(n,c,h,w) = n*H*W*C + c*H*W + h*W + w`
       (`com.unity.ml-agents/Runtime/Inference/TensorExtensions.cs:56`), and `SetTarget` builds a
       3-element visual spec as `TensorShape(batch, shape[0], shape[1], shape[2])` = **NCHW**
       (C=1, H=16 floors, W=8 features). So training's `writer[0,f,c]` → flat index **`f*8 + c`** —
       byte-identical to `FillFloorGrid`'s `buf[f*8+c]`, which is exactly how Sentis reads
       ConvDispatcher's `TensorShape(1,1,F,8)` NCHW tensor. **Train and eval observation orderings are
       provably the same; the dispatcher is correct and the eval numbers are TRUSTWORTHY.**
    3. **(new, live hypothesis) Train/eval distribution mismatch → reward-vs-delivered divergence.**
       Training runs at `TrafficConfig.intensity = 1.0` (+ randomizeActive); eval runs at
       `SmokeIntensity = 0.5`, no randomization. So "better training reward" is measured in a
       *heavier-load* regime than the eval. The conv's floor-adjacency bias may specialize to the
       saturated regime and generalize worse to the lighter eval load — which would explain better
       training reward AND worse eval delivered/abandoned simultaneously, with no bug involved. If
       conv@10M still underdelivers, test this by evaluating at intensity 1.0 (matched to training).
    **Decisive test RUNNING: `elev-e13-m-interfloor-conv-01` resumed 5M→10M** (from step 5,000,192,
    `config/elevator_ppo_e13_interfloor_conv_10m.yaml`, ~4.7h). With (2) ruled out, the read is clean:
    conv@10M ≥ flat@10M's 2241.8 → explanation (1) (budget), conv win is real on the headline metric.
    Still ~1,980 despite far better training reward → explanation (3) (train/eval load mismatch), i.e.
    the conv optimizes the intensity-1.0 training regime in a way that does NOT produce more
    deliveries at the intensity-0.5 eval regime — a genuine, publishable finding about
    reward/metric divergence rather than a bug, and a prompt to re-examine the eval intensity choice
    (which all E2-E13 headline numbers share).
    **Interim honest status: the conv's advantage is established ONLY on training reward; it does NOT
    yet translate to delivered passengers, and may not.**
  - *(superseded note)* **[EVAL PENDING — infra-blocked, NOT a result] delivered-passenger sweep not yet obtained.**
    First eval attempts showed PPO delivered=0, but this is a **stale-model-import artifact, not the
    conv policy**: `cp`-ing the new ONNX over the existing eval path left Unity serving the OLD cached
    import (auto-import is Editor-focus-gated), so the sweep evaluated stale/undertrained weights.
    **Fixed** (commit `fb04701`): `RunScaleLadderSweep` now `AssetDatabase.ImportAsset(...,
    ForceUpdate)` before load, and the conv model was copied to a fresh `_5m.onnx` path. Could not
    complete the corrected sweep this session: the Unity MCP bridge went unresponsive again and a
    clean Editor restart hung at startup (past 01:20, licensing stage, 0% CPU). **To finish (one
    clean Editor session):** recompile → run menu `E13 Conv/Run Sweep (...PPO-conv...)` → expect a
    sane delivered count (~2000+, comparable-to-better than the flat MLP's interfloor 2242 given the
    reward win). The training-level WIN above stands independently of this.
  - **NEXT after eval:** (1) extend conv 5M→10M for a converged apples-to-apples vs the flat MLP's
    10M (-13,652) — conv was still climbing at 5M; (2) if the win holds, conv becomes the interfloor
    (and likely general) recipe → re-run the other E12 patterns (Lunch/DownPeak/DayCycle) and rung L
    on the conv architecture; (3) the reused `RecurrentPpoDispatcher`/LSTM path stays shelved.
- **Sequencing (per user 2026-07-15):** architecture work started in parallel now rather than waiting
  for Lunch/DownPeak data; runs sequenced (not concurrent) to avoid the machine oversubscription that
  has been destabilizing the Unity Editor.
- **Editor-vs-training instability (confirmed 2026-07-15):** the Unity Editor cannot reliably serve
  the MCP bridge while a 20-env training run saturates the machine — cheap reads (`get_scene_info`)
  slip through but menu-execute/build/recompile stall the main thread and freeze it (needs a manual
  window-focus or a force-kill+relaunch). **Decision (user): let the E13a memory run finish (~2h),
  then batch ALL Editor-dependent work in one stable session with the machine free.** All
  Editor-INDEPENDENT prep is done and committed ahead of that session: both eval dispatchers
  (`RecurrentPpoDispatcher` for E13a, `ConvDispatcher` for E13b), the recurrent/conv branches +
  menu items in `RunScaleLadderSweep`, the conv sensor/component/menu, and both configs.
- **POST-MEMORY BATCH CHECKLIST (mechanical; minimize Editor exposure):**
  1. `cp results/elev-e13-m-interfloor-memory-01/ElevatorController.onnx
     Assets/ElevatorRL/Models/ElevatorController_M_e13_interfloor_memory.onnx`
  2. Recompile (verify all E13 C# compiles clean — sensor, 2 dispatchers, eval wiring).
  3. **E13a eval:** run menu `Tools/Elevator RL/E13 Conv/Run Sweep (...PPO-LSTM...)`. Sanity-check the
     result isn't degenerate (would signal the `recurrent_in` width ≠ 128 — confirm via onnx.load).
     Document result vs the -14.4% baseline; if memory doesn't help, temporal-history hypothesis is
     dead and E13b (spatial conv) becomes the main bet.
  4. **E13b build+de-risk:** menu Point Agent At Baseline Obs → E13 Conv/Add Floor-Grid Sensor →
     Build Headless Trainer. Then a SHORT conv smoke-train (`config/elevator_ppo_e13_interfloor_conv.yaml`,
     max_steps tiny) to export an ONNX; onnx.load it to confirm obs_0=visual/obs_1=flat + the visual
     tensor layout (NHWC vs NCHW) BEFORE the full run. Fix `ConvDispatcher._visualShape` if needed.
  5. If export/inference OK → full conv train (5M, extend if climbing) → copy ONNX to
     `ElevatorController_M_e13_interfloor_conv.onnx` → run menu `E13 Conv/Run Sweep (...PPO-conv...)`.
  6. Also still-pending from before the freeze (decide if worth it given the memory result): the
     pattern-aware interfloor retrain (`ObservationConfig_PatternAware` + `..._patternaware.yaml`
     already created; needs Point Agent At Pattern-Aware Obs → rebuild → train → eval).

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

**AS1 IMPLEMENTED (mechanics validated headlessly; agent loop pending a Play-mode run).**
- `TargetFloorControl` (runtime) holds the reusable stateless pieces: `ResolveTowardTarget`
  (committed target → one primitive action; services current floor, opportunistic same-direction
  boarding, else steps toward target), `CommitmentFulfilled` (arrived + nothing left to do → time
  to re-query the policy), and `LookTargets` (LOOK-equivalent *target* selector = CollectiveLook
  passes 1–2 returning target floors — the AS1 heuristic/demonstration policy).
- `ElevatorControllerAgent` gains an `ActionSpaceMode { Primitive, TargetFloor }` toggle:
  `Initialize` sizes branches (6 for AS0, `numFloors` for AS1); `FixedUpdate` runs the per-tick
  deterministic driver for committed cars and requests a decision only when a car actually needs a
  new target (sparse, semi-MDP); `OnActionReceived` adopts targets only for asking cars (in-flight
  cars' branches are masked to a single option); `WriteDiscreteActionMask` restricts asking cars to
  in-range floors and locks committed cars; `Heuristic` emits `LookTargets`. Reward is drained per
  (sparse) decision → correct semi-MDP credit attribution.
- **Validation (headless, no Play mode):** a `TargetFloorLook` dispatcher (`LookTargets` +
  `ResolveTowardTarget`, recomputed each tick) reproduces LOOK within a few percent on
  8fl/3cars/UpPeak/i=0.5/seed=1 (delivered 1680→1708, waitP95 37.2→36.4s, abandoned 27→24) —
  slightly better because the resolve boards both directions when idle at a target. Menu:
  **Tools ▸ Elevator RL ▸ Validate AS1**. This confirms the target/resolve logic; it does NOT
  exercise the agent's commitment-persistence / sparse-decision loop (that lives only in the agent
  and needs a Play-mode run — fold into the ML-Agents integration check before the E9 training).

---

## 8. Milestones

1. **M0 — Baseline surface (E1) — DONE for UpPeak/Lunch.** Instrumentation, all three baselines,
   the five (six w/ H_VarFleet) scale-ladder presets, and calibrated sweep infrastructure are
   implemented and producing real data (§1.2 findings 1–3; §3 calibrated intensities). Remaining:
   more seeds per cell (currently 1-2), DownPeak/Midday/Uniform patterns, and committing the
   sweep-runner code + this doc's findings.
2. **M1 — Trainable setup (E2) — DONE.** §2's truncation, horizon/γ default, and wait-age
   observation are all applied. PPO trained on rung S (`elev-e2-s-ppo-01`, headless/parallel
   pipeline — see §2 note below) and evaluated vs LOOK/ETA across 5 seeds: PPO wins outright, not
   just parity (see E2 above). Parity gate cleared and then some.
3. **M2 — Scale curve (E3).** Train/eval across S→H; produce the gap-vs-rung headline figure.
4. **M3 — Zoning + obs (E4, E5).** The thesis core + cheap ablations.
5. **M4 — Architecture + generalization (E6, E7).** Unlock large fleets; one-policy-many-fleets.
6. **M5 — Action space (E9) — AS1 built + mechanics validated; comparison pending training.**
   `TargetFloorControl` + agent `ActionSpaceMode.TargetFloor` implemented, sub-controller validated
   headlessly vs LOOK (§7). Remaining: Play-mode check of the commitment loop, then train AS0 vs
   AS1 and compare sample-efficiency/asymptote.
7. **M6 — Stretch + write-up (E8).** Fairness, transfer, emergent-strategy visuals.

---

### Appendix — expectation calibration
Classical result (Crites & Barto, 1996) needed a **10-floor, 4-car** building under **heavy
down-peak** to show clear RL gains over the best heuristics — i.e. the win shows up at scale and
under load, not in toy settings. That is consistent with this plan's thesis and is why the
experiment weight is on rungs **L/Z/H** at **intensity ≥ 1.5**, not on rung S.
