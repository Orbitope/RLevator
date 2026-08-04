# RLevator — promo kit

**Article:** *What's Taking So Long?*
**Slot:** Week 5 — r/reinforcementlearning Tue 1 Sep · r/MachineLearning Thu 3 Sep (gated) · X Wed 2 Sep

> ⚠️ **Decide this before 1 Sep.** `EXPERIMENT_PLAN.md` and `NARRATIVE.md` carry results *newer* than
> the published article — the V2 traffic re-baseline and the Act IV wins. Everything below quotes
> only what the article actually publishes, so the post and the page agree. If you'd rather lead with
> the newer numbers, update the article first: posting stale figures to r/reinforcementlearning and
> correcting in the comments is a bad trade.

## Links

| Where | URL | Status |
|---|---|---|
| Article | `https://orbitope.github.io/rlevator/` | ⚠️ **unverified** — inferred from the confirmed `orbitope.github.io/simulacrum/` pattern, and the repo is `RLevator`, so **check the case**. |
| Repo | `https://github.com/orbitope/rlevator` | goes in a **comment**, never the post body |
| Playable | none | |

## The hook

Elevator dispatch runs on a seventy-year-old heuristic (LOOK) and it is genuinely good. Measured on
identical traffic, five fixed seeds, hour-long episodes:

| Building | Result |
|---|---|
| S · 8 floors / 3 cars | RL ties the heuristic — there was nothing to win |
| M · 16 floors / 5 cars | mean wait **9.07s → 6.86s**, abandoned **3.4 → 0** |
| L · 30 floors / 8 cars | more delivered, abandonment down ~5× |

The twist: **none of it came from a fancier network.** Flat MLP, floor-axis conv, per-car agents and
attention all land within noise of each other. What moved the number was deleting a reward term — a
±0.4-per-floor shaping nudge that looked harmless and turned out to be the most important coefficient
in the project.

Supporting: the whole simulator was rebuilt out of Unity/ML-Agents into a pure tensor environment at
~46,000 agent-steps/s, ~37× the original pipeline, so a 5M-step run drops from hours to ~2 minutes.

---

## Reddit — r/reinforcementlearning · Tue 1 Sep

**Title**

```text
Deleting a reward-shaping term beat every architecture change I tried (learned elevator dispatch vs LOOK)
```

**Body**

```text
Learned dispatch against the classical LOOK heuristic on identical traffic, five fixed seeds,
hour-long episodes, greedy evaluation. On a 16-floor / 5-car building: mean wait 9.07s -> 6.86s and
abandonment 3.4 people -> 0. On the 8-floor building it ties, which is the expected result — a small
fleet has enough slack that dispatch quality barely matters. The part I didn't expect is that flat
MLP, floor-axis conv, per-car agents and attention all landed within noise of each other; what
actually moved it was dropping a +/-0.4 per-floor shaping term I'd assumed was harmless.
```

**Link:** the article.

**Image:** `img/results-M-unshaped.png` — the full method table with LOOK as the baseline row.
Second image if you post a gallery: `img/results-flip.mp4`, the shaped→unshaped flip on the same
building, which is the finding itself rather than a summary of it.

**Top-level comment**

```text
Protocol detail, since it's the thing I'd want to check first: counter-based RNG means the same seed
replays an identical passenger tape for every method, so LOOK and the learned policy are graded on
the same people arriving at the same instants. Evaluation is greedy argmax, matching how a
deterministic dispatcher actually runs.

Code, configs and the experiment log: https://github.com/orbitope/rlevator
```

---

## Reddit — r/MachineLearning · Thu 3 Sep — ⚠️ gated

Post only if the account has real standing by then. If it's still thin, **skip it** — the project
stands fine on r/reinforcementlearning alone.

**Title** (flair `[P]`)

```text
[P] Learned elevator dispatch vs a 70-year-old heuristic: the objective mattered, the architecture didn't
```

**Body**

```text
Elevator dispatch is a well-studied control problem with strong hand-engineered baselines, which
makes it a decent test of whether a learned policy earns its keep. Same traffic, same seeds, same
greedy evaluation for every method. The headline is that the learned policy ties LOOK on a small
building and beats it on larger ones (16 floors: 9.07s -> 6.86s mean wait, 3.4 -> 0 abandoned), but
the useful result is negative: four architectures spanning flat MLP to per-car attention were
separated by less than the gap between two reward functions. The simulator was also rebuilt from
Unity/ML-Agents into a batched tensor env (~37x end-to-end), validated by differential testing
against a readable single-instance reference at zero tolerance.
```

**Image:** `img/results-M-shaped.png` + `img/results-M-unshaped.png` — the same five methods under
the two reward functions. The architecture spread versus the reward spread is visible at a glance.

---

## X — Wed 2 Sep

**Tweet 1** — attach `img/results-M-unshaped.png`

```text
Elevator dispatch has run on the same heuristic since the 1950s, and it's genuinely good. I spent
months trying to beat it with PPO.

16-floor building, identical traffic, 5 seeds:
mean wait 9.07s -> 6.86s
people who gave up waiting: 3.4 -> 0
```
`chars: 242/280`

**Tweet 2** — attach `img/architectures.png`

```text
Here's the part I didn't expect.

None of that came from a better network.

Flat MLP. Floor-axis conv. Per-car agents. Attention. All four landed within noise of each other.
Months of architecture work, separated by nothing.
```
`chars: 224/280`

**Tweet 3** — attach `img/results-flip.mp4`

```text
What moved it was deleting a reward term.

A ±0.4 nudge per floor a loaded car moves toward its rider's destination. Looks harmless. Reads like
free shaping.

It was the most important coefficient in the project — and it was making things worse.
```
`chars: 245/280`

**Tweet 4** — attach `img/what-the-agent-sees.mp4`

```text
Also: in the small building it just ties. Which is the right answer, not a failure — 3 cars have
enough slack that dispatch quality barely matters.

If your baseline is already near-optimal, a win means your benchmark is broken.

https://orbitope.github.io/rlevator/
```
`chars: 253/280`

---

## Asset index — `docs/promo/img/`

All captured from `docs/index.html` itself by `scripts/capture_promo.mjs`, driving the article's own
controls rather than guessing at scroll positions. Nothing is redrawn.

| File | What it is |
|---|---|
| `results-M-unshaped.png` | 16fl/5car, final reward — LOOK 9.07s vs Flat MLP 6.86s, 3.4 vs 0 abandoned |
| `results-M-shaped.png` | same building under the shaped reward — all five methods, the architecture spread |
| `results-flip.mp4` / `.gif` | the shaped→unshaped flip. The finding, not a summary of it. |
| `results-S-unshaped.png` / `results-L-unshaped.png` | the small building tying, the large one pulling ahead |
| `results-M-abandoned.png` | the abandonment metric on its own |
| `what-the-agent-sees.mp4` / `.gif` | the observation vector built up one block at a time |
| `observation-early.png` / `observation-full.png` | stills from that sequence |
| `reward.png`, `reward-selector.png` | the seven-force reward and its coefficient panel |
| `architectures.png` | flat MLP vs conv vs per-car vs attention |
| `og.png` | the article's existing social card (pre-existing, not captured) |

---

## Appendix — the five-week calendar

Warm-up **Wed 5 – Thu 6 Aug**: ordinary commenting, no links, in r/WebGames and r/puzzles. Keep
low-level commenting going in each week's target subs throughout — with a new account this matters
more than any single post.

| Week | Reddit #1 | Reddit #2 | X |
|---|---|---|---|
| 1 | Thu 6 Aug — **Gridlocked** → r/WebGames | Sat 8 Aug — r/puzzles | — (already posted) |
| 2 | Tue 11 Aug — **Hex Truchet** → r/proceduralgeneration | Thu 13 Aug — r/tabletopgamedesign | Wed 12 Aug |
| 3 | Tue 18 Aug — **Simulacrum** → r/reinforcementlearning | Thu 20 Aug — r/MachineLearning `[P]` (gated) | Wed 19 Aug |
| 4 | Tue 25 Aug — **Pushman** → r/Unity3D | Thu 27 Aug — r/gamedev | Wed 26 Aug |
| 5 | Tue 1 Sep — **RLevator** → r/reinforcementlearning | Thu 3 Sep — r/MachineLearning `[P]` (gated) | Wed 2 Sep |

Reddit posts land Tuesday mornings US-Eastern; the second sub is staggered two days so two threads
are never live at once. X threads go Wednesday, a day behind Reddit, so a good comment can be folded
in. **r/MachineLearning is gated on account standing** — skip it if the account is still thin; both
RL projects stand fine on r/reinforcementlearning alone. r/algorithms is deliberately unused: best
topical fit for Gridlocked, but hostile to self-promotion from a new account. Revisit after week 5.

This is the second r/reinforcementlearning post — two full weeks after Simulacrum's, on a genuinely
different topic. Don't compress that gap.

Nothing here posts itself. Reddit and X both punish anything that reads as automated, and the
comment replies are most of the value.
