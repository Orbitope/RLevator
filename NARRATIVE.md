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

## Act IV — The payoff (planned)
- **Does RL win now?** Re-run the ladder (PPO, then MA-POCA / BC-from-LOOK, targeting the diagnosed
  search failure) on realistic traffic.
- **If it wins — the tradeoffs.** Reward variants (longest-wait vs journey-time vs lobby-priority),
  and the money question: **is destination-dispatch (omniscient info — a big capex + "use an app" UX
  cost) worth the wait-time it saves?** Let the measured seconds decide.
- **How far does multi-agent coordination scale?** Bigger fleets, zoning — where RL *should* win most.

## Lessons (the actual thesis)
- The most dangerous bug is not in the model — it's in the **benchmark**. A flawed environment produces
  confident, wrong conclusions that survive many experiments.
- **Measure the thing that would refute you.** Every hypothesis here died to one number; the fastest
  progress came from killing our own ideas, including several of the assistant's.
