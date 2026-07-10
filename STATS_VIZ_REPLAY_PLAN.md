# Elevator RL — Stats, Comparison, Visualization & Replay Strategy

Expands §5 of [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) into an implementable system for
**collecting runtime stats, comparing runs, visualizing effects, and replaying illustrative
episodes** — styled with ContentKit. Chart/color decisions follow the `dataviz` method
(form first, color by job, validated palette) mapped onto the `CKColor` palette.

---

## 0. Principles

- **One canonical record.** Every run — LOOK, ETA, or any RL variant — emits the *same*
  `EpisodeStats` schema. Comparison is only meaningful if the schema is identical.
- **Paired comparison on a fixed traffic tape.** Arrivals are seeded (`PassengerArrivals` →
  one `System.Random`). Running two policies on the *same seed* makes them face *identical*
  demand — a paired A/B with far lower variance than independent runs. This is the backbone of
  every comparison and every replay.
- **Determinism ⇒ cheap replay.** Sim + (LOOK/ETA rule | RL inference) are deterministic given
  `(configs, seed)`. A replay is therefore just that tuple re-simulated — no heavy frame log.
- **Color follows the entity, not its rank** (dataviz non-negotiable): each *policy* owns one
  hue across *every* chart. A filter that drops a policy never repaints the survivors.
- **ContentKit is a muted/filmic palette** — steel & sage read near-gray (validator: chroma
  FAIL), so **every series is always direct-labeled or legended + positionally/texturally
  distinct; hue is never the only cue.** Amber is the brightest, most saturated hue → reserved
  for the protagonist (RL), and it blooms.

---

## 1. Canonical records

Three tidy (long-format) tables at increasing resolution. The aggregate answers "who won"; the
breakdowns answer **"where and when"** — which is where the thesis actually lives (RL's edge is
predicted to concentrate on *specific floors* during *specific times of day*, not uniformly).

### 1.1 `EpisodeStats` — one row per evaluated episode (aggregate)

**Identity / repro:** `runId`, `gitSha`, `policy` (LOOK/ETA/PPO-A0/PPO-A1/…), `modelPath`,
`buildingPreset` (+ config hash), `trafficPreset`, `pattern`, `intensity`, `seed`,
`simSeconds`, `warmupSeconds`.

**Throughput:** `delivered`, `deliveredPerHour`, `deliveredPerDecision`.

**Wait (the story at scale — keep the tail):** `waitMean`, `waitP50`, `waitP95`, `waitMax`;
`rideMean`, `rideP95`. (Percentiles from §2 histogram/reservoir, not a running mean.)

**Failure:** `abandoned`, `rejected`, `abandonRate`, `rejectRate`.

**Fairness:** `waitStd`, `waitTailRatio` = p95/p50, optional `waitGini`.

**Reward decomposition** (already separated in `Building.Acc`): `rwTotal` + the 7 terms
`rwDelivered/Toward/Away/Rejected/Abandoned/InElevator/InQueue`.

**Fleet:** `utilFleetMean`, per-car `util[i]`, `idleFraction`, `inServiceMean` (for variable
fleet).

### 1.2 `FloorStats` — one row per (episode × floor) → `floor_stats.csv`

The per-floor breakdown. Essential for the zoning story: shows *which* floors LOOK starves and
whether RL covers them. Metrics scoped to passengers **originating** at that floor.

`floor`, `carsServing` (how many in-service cars have this floor in range — the restriction
signal), `origins`, `destinations` (arrived-to count), `delivered`, `abandoned`, `rejected`,
`waitMean`, `waitP95`, `waitMax`, `queueLenMean`, `queueLenMax`. Optional up/down split
(`waitP95Up`/`waitP95Dn`) since down-peak concentrates on high-floor down-queues.

### 1.3 `WindowStats` — one row per (episode × time-bucket) → `window_stats.csv`

The temporal breakdown. `pattern` alone is a coarse time-of-day proxy and, crucially, a **day-cycle
episode** (`traffic.useDayCycle`) sweeps up-peak→lunch→down-peak *within one episode* — a single
aggregate row blurs them. Fixed buckets (e.g. `bucketSeconds = 300`) recover the within-day curve
and feed the live sparklines.

`bucketStart`, `activePattern` (resolved per bucket for day-cycle runs), `delivered`,
`deliveredRate`, `waitMean`, `waitP95`, `abandoned`, `rejected`, `fleetUtilMean`, `carsInService`.

### 1.4 `FloorWindowStats` (opt-in) — (episode × floor × bucket) → `floor_window_stats.csv`

The floor×time cross that backs the **heatmap** (§6.2 #4). Larger (≈ floors × buckets rows/
episode), so gated behind an eval flag. `floor`, `bucketStart`, `waitMean`, `waitP95`,
`queueLenMean`, `abandoned`.

> **Time-of-day comparison two ways:** (a) *across* episodes — `pattern` is already a sweep axis,
> so aggregate `EpisodeStats` filtered by pattern answers "does RL win more in up-peak than
> midday?"; (b) *within* an episode — `WindowStats` answers "how does the gap evolve as up-peak
> builds and clears?" Run day-cycle episodes for (b); fixed-pattern episodes for clean (a).

---

## 2. Instrumentation (how we collect it)

Small, decoupled additions to the runtime (research phase — runtime edits are in scope):

1. **`Building` events** (zero cost when unsubscribed) — each carries the `Passenger` (or its
   `origin`/`dest`/`waitTime`) so samples route into per-floor and per-window bins:
   - `event Action<Passenger,float> OnDelivered;` (passenger, rideTime) — fire in `DoTransfer`
     unload (passenger carries `origin`, `dest`, `waitTime`; delivery time = `simTime`).
   - `event Action<Passenger> OnAbandoned;` — fire in `ExpireOne`.
   - `event Action<int> OnRejected;` (floor) — fire in `SpawnArrivals` on full-queue.
2. **`WaitHistogram`** — fixed bins over `[0, maxWait]` (+overflow) for O(1) p50/p95/max without
   storing every passenger. The delivered/abandoned event carries the passenger's **origin
   floor** and delivery **sim-time**, so the collector routes each sample into the right
   per-floor and per-time-bucket histogram — the §1.2–1.4 breakdowns fall out of the same event
   stream at negligible cost (a handful of KB of histograms per episode). Optional bounded
   reservoir (~4k) when exact percentiles are wanted.
3. **`StatsCollector`** (plain C#, runs headless): subscribes to the events, maintains the global
   + per-floor + per-window (+ optional floor×window) histograms and counters, flushes
   `EpisodeStats` / `FloorStats` / `WindowStats` rows at episode end. Used by both the sandbox
   and the eval harness.

> No change to sim *behavior* — events are observational; existing `DeliveredTotal`/`Acc`
> counters stay. The histogram is what upgrades "mean wait" to "p95/max wait," which is where
> the large-building story lives.

---

## 3. Run identity & storage

- **`RunManifest.json`** per run: `runId` (`yyyymmdd-hhmm-<shortsha>`), git sha, policy, model,
  preset names + **config hashes** (hash of serialized config → exact-param traceability),
  sweep axes, code version.
- **Layout:** `Runs/{runId}/manifest.json`, `episodes.csv`, `summary.csv`.
  `Runs/` is git-ignored (raw data); a curated **`Results/`** holds committed `summary.csv`s +
  exported figures for the write-up (small, diff-able, reproducible from manifests).

---

## 4. Eval harness (generating comparable runs)

`EvalHarness` — editor menu **Tools ▸ Elevator RL ▸ Run Eval Sweep** + a `-batchmode` CLI entry
for headless/CI. Iterates the sweep **policies × presets × patterns × intensities × seeds**; per
cell: build sim, run headless at max timescale for fixed `simSeconds` (e.g. 2 simulated hours),
discard `warmupSeconds` transient, collect `EpisodeStats`, append to `episodes.csv`; write
`summary.csv` (mean ± CI over seeds).

- **Fixed traffic tape:** same seed set across policies within a cell ⇒ paired comparison.
- **Baselines first** (matches EXPERIMENT_PLAN E1): LOOK, ETA/ETD, Zoned-LOOK — no training, pure
  sandbox. Produces the yardstick surface before any RL exists.

---

## 5. Training integration (TensorBoard)

In `ElevatorControllerAgent`, push the same metrics as **custom ML-Agents stats** at episode end
via `Academy.Instance.StatsRecorder.Add("Elevator/WaitP95", …)` (and AvgWait, AbandonRate,
DeliveredPerHour, plus the 7 reward terms). Then training curves are directly comparable to eval
CSVs and to each other in TensorBoard. Add a **periodic deterministic eval** on the fixed tape
during training so the reported number is eval-policy, not exploration-noised.

---

## 6. Visualization system (ContentKit-styled)

### 6.1 Validated color roles (run through `dataviz/validate_palette.js`, dark mode)

Policy = entity → **fixed hue, every chart** (CVD separation PASSED for Steel/Sage/Amber):

| Policy | Base (chrome) | Data-mark (bright) | Notes |
|---|---|---|---|
| **LOOK** (naive baseline) | `Steel #6B7A8D` | `SteelBright #9AAABB` | the bar everyone clears |
| **ETA/ETD** (strong baseline) | `Sage #7D9A6A` | `#8FB07A` | honest bar |
| **RL** (protagonist) | `Amber #C49A3C` | `AmberBright #E8C068` | brightest → draws the eye; **blooms** |
| **4th+ variant** (Zoned-LOOK, RL-attention) | — | — | **not a new hue**: texture (45° hatch) on the parent policy's hue, or small-multiples (dataviz: a 4th series folds into texture/facets, never a cycled hue) |

Reserved semantic colors — **never a series**: `Coral #FF5E3A` = failure (abandonment/rejection,
always with icon+label); reward delta uses `AmberBright` (+) / `SteelBright` (−) as the dashboard
already does.

Sequential (heatmaps): **one hue, dark→light** — `Void → Raised → Amber → AmberBright`
(validate monotonic lightness); overlay `Coral` markers on abandonment cells (marker, not ramp).
Diverging (RL−LOOK gap): **Steel (LOOK-favored) ↔ neutral Border-gray at 0 ↔ Amber (RL-favored)**.

Chrome/grid/axes stay ultra-recessive (`Border`, `TextMuted`); **text wears ink tokens
(`TextBright/Primary/Secondary/Muted`), never the series color.** Fonts: JetBrains Mono for
values/labels, Rajdhani for hero numbers — the dashboard's existing conventions.

### 6.2 Chart catalog (form chosen first, per dataviz)

1. **Scale curve — the headline** (change across ordered rungs). Line: x = rung S→M→L→Z→H,
   y = % improvement in p95 wait vs LOOK; one amber line per RL variant, steel zero-reference
   line = LOOK. Direct-labeled endpoints. *This is the thesis figure* (EXPERIMENT_PLAN E3).
2. **Policy comparison** (magnitude by identity). Small-multiple grouped bars: y = p95 wait,
   groups = policy, one facet per rung. Direct value labels in muted ink; 2px surface gap
   between bars.
3. **Wait distribution — ECDF** (distribution, tail-readable). One line per policy on a chosen
   rung; p95 read directly off the curve. Preferred over histogram because the tail *is* the
   point at scale.
4. **Per-floor wait heatmap** (magnitude over floor × time-bucket, from `FloorWindowStats`).
   Sequential amber ramp; exposes which floors starve under zoning *and when* (the E4 story).
   Coral cell markers where abandonment occurred. Show LOOK and RL as two stacked heatmaps on a
   shared scale so the rescued floors read at a glance.
4b. **Per-floor improvement bars** (from `FloorStats`). Diverging bars, y = floor,
   x = RL−LOOK Δ p95 wait per floor (Amber = RL better, Steel = LOOK better, gray at 0). The
   direct evidence that the win concentrates on high/restricted floors, not the lobby.
4c. **Time-of-day gap curve** (from `WindowStats`, day-cycle episode). The scale curve's
   temporal sibling: x = sim time across a full day, y = RL−LOOK p95 gap, with pattern bands
   (up-peak/lunch/down-peak) shaded behind. Shows *when* RL earns its advantage.
5. **Reward decomposition** (part-to-whole over time). Stacked area of the 7 terms across an
   episode, 2px surface gaps between bands — diagnostic for *why* a policy wins/loses.
6. **Live episode sparklines** — reward + avg wait vs sim time, embedded in the running
   dashboard.
7. **Fleet utilization / idle** — per-car bars (extends the dashboard's existing util block).
8. **Leaderboard table** — sortable summary + the required accessible table view of every chart.

All charts: legend present for ≥2 series (none for 1), selective direct labels (never every
point), recessive grid, one y-axis (**never dual-axis**), hover tooltips in the interactive
build. Built as uGUI/TMP widgets reusing `CKUI`/`CKColor`; renderable through the ContentKit
post-process (bloom) camera and capturable with ContentKit Recorder for the write-up.

---

## 7. Replay system

### 7.1 Deterministic scenarios
A **`Scenario`** = `(buildingPreset, trafficPreset, pattern, intensity, seed, policyRef|modelPath)`
— serializable. Because sim + policy are deterministic, re-simulating the tuple reproduces the
episode *exactly*; no per-frame log needed. Transport (play/pause/**STEP**/speed) already exists
in the dashboard; add scrub-to-decision + "jump to flagged moment."

**`ScenarioLibrary`** (ScriptableObject): curated list with `title`, `note`, and a flagged
decision index — the saved "examples that illustrate points."

### 7.2 Auto-flagging illustrative moments
During the eval sweep, detect and auto-save scenarios where the story is visible:
- an **abandonment under LOOK that RL avoided** on the same tape (the cleanest "RL wins" clip);
- a **max-wait outlier** (worst tail episode);
- a **large per-episode RL−LOOK delta** (biggest divergence).
Each writes a `Scenario` + timestamp into the library.

### 7.3 Side-by-side dual-sim (the demo)
Two `Building` instances on the **same seed/tape**, two policies, rendered as two building views
with **synchronized transport** (shared play/pause/step/speed). When the policies diverge on the
same passenger — one delivers, the other lets them abandon — flash the affected floor `Coral`.
ContentKit-styled; record via ContentKit Recorder → the headline video/GIF for the write-up.

---

## 8. Build order

1. **M0 — Instrumentation.** `Building` events + `WaitHistogram` + `EpisodeStats` + `StatsCollector`
   → CSV/JSON. (Unlocks real p95/max wait.)
2. **M1 — Eval harness + baselines.** Fixed-tape sweep; LOOK/ETA/Zoned-LOOK `summary.csv`
   (EXPERIMENT_PLAN E1) — the yardstick, no training required.
3. **M2 — Training stats.** ML-Agents `StatsRecorder` custom stats + periodic deterministic eval.
4. **M3 — Visualization.** ContentKit charts §6.2 (scale curve, comparison bars, ECDF, heatmap,
   leaderboard) with the validated §6.1 roles.
5. **M4 — Replay.** `Scenario`/`ScenarioLibrary`, auto-flagging, side-by-side dual-sim view.
6. **M5 — Deliverables.** Recorded figures + the divergence clip for the write-up.

> Dependency note: M0 gates everything (schema first). M1 can run against the current LOOK/ETA
> with no RL, producing publishable baseline evidence immediately. M3/M4 are the ContentKit-styled
> payoff and should reuse `CKUI`/`CKColor` and the bloom camera already built for the dashboard.
