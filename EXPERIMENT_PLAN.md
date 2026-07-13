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

### E3 — **The headline: RL vs. LOOK as scale/constraint increases** — PAUSED after S, M (see E6)
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
  encoder / attention) even once given enough time. Decision pending — not yet resolved before
  training L.

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

### E6 — Architecture: flat MLP vs. shared per-car vs. attention — **PULLED FORWARD, active now**
- **Q:** Does weight sharing / attention over cars unlock the large-fleet rungs (L/H)?
- **Arms:** flat MLP (baseline), shared per-car encoder, attention.
- **Metric:** performance & sample-efficiency on L and H; this is where scalability is won or lost.
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
  Decision on which path, plus the reward-shaping question in option 1, pending discussion.

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
