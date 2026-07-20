# RLevator — the story (writeup / video spine)

An ordered log of hypothesis → measurement → pivot. The **retractions are the point**: this is a story
about how easy it is to draw the wrong conclusion from a flawed benchmark, and how measurement (not
cleverness) keeps you honest. Durable numbers live in `EXPERIMENT_PLAN.md`; this is the through-line.

## Act I — "Can RL beat classical elevator dispatch?"

1. **The setup.** Elevators are dispatched by decades-old heuristics (collective control / "LOOK",
   nearest-car/ETA). Can modern RL (PPO in Unity ML-Agents) beat them? Baselines: LOOK, ETA.
2. **Early promise, then a wall.** RL *beat* the heuristics on the easy rung (S: 3 cars) — but fell
   behind as the building scaled (M: 5 cars, L: 8), and lost *badly* on interfloor traffic once we
   realized we'd only ever tested lobby-up "UpPeak."

## Act II — Chasing the gap (six hypotheses, six measurements, six deaths)

Each "why does RL lose?" hypothesis was killed by a specific number:
3. **Temporal memory?** Added an LSTM. It tracked *behind* the plain baseline. ❌ (task is Markovian.)
4. **Spatial structure?** Added a floor-axis conv. It learned *faster* but plateaued at the *same*
   place — and I'd built it wrong vs. the reference paper (features on the wrong axis, no rate signal).
   Claimed "first architecture to beat the flat MLP," then **retracted** when the advantage eroded to
   noise by 7M steps. ❌
5. **An eval bug?** Traced the ONNX serialization index by hand — dispatcher was correct. ❌
6. **More information?** The "omniscient" destination observation didn't help. ❌
7. **Train/eval regime mismatch?** We trained at one load, evaluated at another. Real, but **refuted**
   as the cause: RL still lost by ~19% when evaluated in its *own* training regime. ❌ (And I had to
   retract a mechanism argument — "deliveries are throughput-capped" — when the data showed they rise
   with load.)
8. **A bad reward?** The decisive test: LOOK beats the trained policy by **28% on the policy's own
   reward**, while leaving ~21% of the fleet idle. So the reward is *fine* — RL was **failing to
   optimize it**. An optimization/search failure, not a design failure. ❌ (reward misspecification)

## Act III — The benchmark itself was rigged

9. **"Review our traffic vs the paper's."** Reading the reference paper's generator was the turn: our
   environment wasn't a *harder* version of theirs — it was a *different, degenerate* one.
   - **13.3× more loaded** per unit of fleet carrying capacity (216 vs 16 arrivals/hr/slot).
   - **Uniform all-to-all** "interfloor" traffic — the least predictable signal possible; nothing to
     exploit.
   - **One flat, constant arrival rate** — no temporal structure to be "smart" about.
   - vs. the paper's **three concurrent components on a time-varying 48-bin day profile**.
   - At 13× overload, the optimal policy collapses toward "sweep continuously" — which is *literally
     what LOOK does*. **We had built a benchmark where the heuristic was near-optimal by construction,
     then spent fifteen experiments concluding RL wasn't clever.**
10. **The rebuild.** Rewrote the generator to match the paper/CIBSE: concurrent components, binned
    time-varying rates, load calibrated to fleet capacity.
11. **The re-baseline (V0).** On the corrected traffic the heuristics finally *disagree* (up to 21%
    wait-time spread) and neither is optimal — a solvable, non-degenerate regime. The question "can RL
    beat the heuristics?" is fair again. **← we are here.**

## Act IV — The payoff

12. **Does RL win now? First data point: yes — with a caveat.** Same PPO recipe that lost for the
    entire project, retrained from scratch on the corrected traffic (rung M, interfloor pattern,
    nominal load): it now delivers *more* passengers than LOOK or ETA in every one of 5 seeds, abandons
    almost nobody (avg 1.0 vs LOOK's 5.6 and ETA's 14.4), scores the *highest* reward of the three, and
    beats both on tail wait (P95). It only loses on *mean* wait, and only to ETA — LOOK is worse than
    PPO there too. **The exact failure signature from Act II (low reward, low utilization) is now
    reversed**: PPO's utilization is the highest of the three, not the lowest — it's running the fleet
    harder, not routing the same trips more cleverly. Same net. Same trainer. Same hyperparameters.
    The only thing that changed was the traffic generator.
13. **Rung S: a near three-way tie — and that's the expected result, not a setback.** Same recipe,
    same pattern, retrained fresh on the small (3-car) fleet: LOOK, ETA, and PPO all land within ~0.8%
    of each other on delivered/reward, and nobody abandons anyone — the small fleet has slack, so
    dispatch quality barely moves the needle. PPO still shaves ~20% off tail wait (P95), but that's the
    only lever left when everyone's served regardless. This is the project's own founding thesis talking:
    RL was never expected to beat LOOK by much where LOOK is already near-optimal (small buildings);
    the bet has always been that the edge appears at scale/constraint. **S as a tied control point makes
    M's win more credible, not less — it isn't a rising tide lifting every rung.**
14. **Rung L: the largest win yet — the trend is monotonic.** Bigger fleet (8 cars/30 floors), same
    recipe, extended to 10M steps because reward was still climbing at 5M (unlike S and M, which had
    both plateaued). PPO beats both heuristics on delivered, reward, AND tail wait by the widest margin
    of any rung so far, and cuts abandonment by ~90% vs LOOK and ~95% vs ETA (1.6 vs 16.4 vs 30.0 riders
    left behind). **S → M → L is now a clean, monotonic, thesis-confirming trend** — tie, then win, then
    bigger win — the first time this project's scale ladder has told a coherent story instead of a
    confound to explain away. The shape of the result already matches the bet made on day one:
    RL doesn't beat LOOK where LOOK is already near-optimal, and wins by more as coordination gets harder.
15. **Stress load complicates the story honestly — and that's the point.** Trained overnight,
    unattended, with zero Editor interaction (a new pre-build-then-train workflow: every rung built to
    its own persistent path first, then training runs chained back-to-back). S at 1.5× load: still a
    wash, confirming the tie isn't a load artifact. M at 1.5× load: **not a repeat of the clean nominal
    win** — PPO still delivers more, cuts tail wait, and cuts abandonment ~90-97%, but LOOK edges ahead
    on total reward and PPO has the *worst* mean wait of the three. Reported plainly rather than
    smoothed over: the M win is real at nominal load and genuinely mixed at stress load.
16. **UpPeak, rung M — the single largest, cleanest win of the entire project.** PPO wins every metric:
    delivered (+1.2%), mean wait **cut in half** (5.79s vs ~12s for both heuristics), tail wait
    (35-41% better), zero abandonment, reward (+6.6%). The twist: **UpPeak was the only pattern this
    project ever tested for most of its history (E1-E11)** — every one of those early runs concluded
    "RL doesn't beat LOOK," on traffic later shown to be degenerate (E15). The exact pattern that
    anchored the project's original pessimism is now, on honest traffic, its strongest result — likely
    because UpPeak's converge-on-one-floor structure is the most learnable/exploitable pattern of all,
    more so than Midday's diffuse interfloor traffic.
17. **The sim leaves Unity — and the port is held to the standard Act III taught us.** After E15
    proved a benchmark can silently be the bug, the sim core was ported to a batched PyTorch env the
    paranoid way: a written spec as the single source of truth, TWO independent implementations
    (readable vs vectorized) differential-tested bit-for-bit at zero tolerance, invariants checked
    batch-wide every step. The battery passed its first full run — and the one scare resolved the
    Act-II way: Python LOOK seemed to deliver 12% more per hour than Unity ("did the generator
    drift?"), until integrating the arrival tables over each eval's *actual* time window showed the
    whole gap was eval windowing; the generator matches Unity to <1.2%, and LOOK's wait distribution
    (5.79 s / P95 15.5 s vs Unity's 5.96 s / 16.9 s) carries over. Hypothesis killed by arithmetic,
    port validated (`EXPERIMENT_PLAN.md` §9).
18. **The payoff is measured, and it's the whole reason: 37×.** The migration was never about
    training speed (that was already headless) — it was about *iteration*, which the Editor gates.
    Held to a real baseline (the actual rung-S headless Unity PPO run: 1,247 steps/s across 20 envs),
    the tensor env does 46,479 end-to-end on the same laptop, CPU-only: a 5M-step run falls from 67
    minutes to under 2. The honest shape of the win: `torch.compile` buys ~3×, batching saturates
    around N≈16k on 5 CPU cores, and — the counterintuitive part — batching *alone*, uncompiled, was
    briefly slower than a plain single-instance loop. Enough that the tempting next move (rewrite for
    JAX, rent cloud TPUs) is explicitly *deferred*: bank the free 37×, escalate to cloud-CUDA only
    when a measured scale run proves it necessary. The catch the win exposed: with Unity out of the
    loop, the *eval* pipeline has to be rebuilt in Python too, or "RL beats LOOK" would be comparing
    across two rulers.
19. **On the new pipeline, RL finally beats LOOK — at M, not S, exactly as promised.** With the
    Python eval rebuilt and matched to Unity, the tensor env trained PPO end-to-end. Two traps had
    to be walked through first, both re-discoveries of this plan's own scars: greedy eval collapsed
    an otherwise-competent policy (a full car parking on a legal no-op — the E13c train/eval-mismatch
    lesson in miniature), fixed by masking board-when-full and annealing entropy to zero; and a
    value-based detour (D3QN) proved robust but delivered ~2× the wait, so PPO — which had learned
    the *better* policy all along — stayed. On rung S the result is a clean **tie** with LOOK
    (waitMean 5.85 vs 5.9 s, zero abandonment) — precisely the thesis's prediction that RL can't beat
    a good heuristic on a small building. Then M: the naive net collapsed (M needs the bigger net —
    E6, rediscovered), but bignet2 at 10M **beat LOOK by 18% on mean wait, 19% on the tail, with a
    quarter the abandonment, at equal throughput.** The first rung where the RL win is real. The
    honest asterisk: it beats Unity's own E6 (which only matched) — likely the board-when-full mask,
    to be attributed before it's leaned on.
20. **The architecture bake-off had a twist ending: it wasn't the architecture.** We built four
    policy nets — flat MLP, floor-conv, shared-per-car, and cross-car attention — expecting the
    coordination-aware ones to pull ahead at scale (8 cars at L). Two of them had to be debugged the
    honest way (conv's first impl never fed a car its own riders' destinations — caught in a 3-min
    learning-check, not a 50-min run, after the lesson that you deep-dive an architecture *before*
    trusting a long run). But when all four were correct and trained: **flat won at every rung.** The
    fancy nets reliably scored *higher shaped-reward* yet *deployed worse* — they were better at
    gaming the objective, not at moving people. That pointed the finger away from the network and at
    the reward. One ablation settled it: deleting the ±0.4 "moved-toward" shaping term — the exact
    thing the fancy nets exploited — made the plain flat policy **strictly better**, beating LOOK at
    M by −25% wait with **zero abandonment**, and at L cutting abandonment from 20 to 3. The whole
    architecture detour resolved to a one-line reward change. The lesson that keeps repeating in this
    project — *the benchmark/objective is where the bug hides, not the model* — held one more time.
    **← we are here.**
- **If it keeps winning — the tradeoffs.** Reward variants (longest-wait vs journey-time vs
  lobby-priority) — note ETA still wins mean wait here, so "RL wins" is already metric-dependent, which
  is exactly what the reward-tradeoff axis is for. And the money question: **is destination-dispatch
  (omniscient info — a big capex + "use an app" UX cost) worth the wait-time it saves?** Let the
  measured seconds decide.
- **How far does multi-agent coordination scale?** Bigger fleets, zoning — where RL *should* win most.

## Lessons (the actual thesis)
- The most dangerous bug is not in the model — it's in the **benchmark**. A flawed environment produces
  confident, wrong conclusions that survive many experiments.
- **Measure the thing that would refute you.** Every hypothesis here died to one number; the fastest
  progress came from killing our own ideas, including several of the assistant's.
