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
    **← we are here — S confirms the easy case is a wash; L (bigger, more coordination-dependent fleet)
    is the rung the thesis says should show the largest edge, and is next.**
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
