# RLevator — project instructions

RL elevator dispatch (Unity + ML-Agents 4.0.x) benchmarked against classical heuristics
(LOOK / ETA). The durable experiment record is `EXPERIMENT_PLAN.md`; the current phase is the
**V2 re-baseline** (see the plan file that spawned it) after E15 found the old traffic model was
degenerate and rewrote the generator (`PassengerArrivals`).

## Documenting results (these results feed a future writeup + video — treat every experiment as on-record)

For **every** experiment, record in `EXPERIMENT_PLAN.md`:
- **Claim** — one line.
- **Decisive measurement** — the single number that supports or refutes it, and where it came from
  (the `Runs/<...>/sweep_summary.csv`, TB tag, or CSV column).
- **Reproducibility handle** — run id, model `.onnx`, and the **git commit** (already snapshotted by
  `scripts/start_training.sh`). Every RL number is stated **relative to LOOK on the identical
  seed/traffic**.
- **Verdict.**
- **Story beat** — the hypothesis raised and how the data resolved it, **including retractions and
  corrections**. The retractions are the most valuable narrative content; the video's spine is the
  hypothesis → measurement → pivot arc, not the final table. Keep the arc in `NARRATIVE.md`.

Never overstate: if a result is confounded, undertrained, or a single noisy reading, say so. Prefer
killing your own hypothesis with a measurement over defending it.

## Operational rules (learned the hard way — they override default behavior)

- **Do NOT try to launch the Unity Editor yourself.** Bare `open -a`, `--args`, and direct-exec
  launches hang at licensing (missing Hub session token). **Ask the user to open the project from
  Unity Hub.** Don't burn cycles retrying launches.
- **Editor work and 20-env training CANNOT overlap** — the Editor goes unresponsive under training
  load. Batch Editor work (rebuilds, menu evals) into training gaps; pause training if needed
  (checkpointed, `keep_checkpoints: 5`, resume via `scripts/resume_training.sh`).
- **Unattended/solo execution: pre-build every environment BEFORE launching any training, then run
  training back-to-back with zero further Editor interaction.** Training (`mlagents-learn` against an
  already-built `.app`) never touches the Unity Editor — only *building* and *eval* do. The Editor also
  tends to go unresponsive for long stretches when unattended, so don't interleave "point preset → save
  → rebuild" with each training run; that serializes N training runs behind N Editor round-trips. Instead,
  in one Editor-responsive window: for each rung/config needed, point the agent at its preset, save, and
  build to a **per-rung path** (`Tools/Elevator RL/Build Headless Trainer - S/M/L/Z/H (macOS)` —
  `Builds/HeadlessTrainer_{S,M,L,Z,H}/`, distinct from the shared default path so later builds don't
  overwrite earlier ones). Then launch every training run in the batch, sequentially, purely via
  `scripts/start_training.sh <run-id> <config> Builds/HeadlessTrainer_<rung>/RLevatorTrainer.app 20 <rung>`
  — no Unity calls needed until the whole batch finishes. Only then batch the evals (each needs the
  Editor, but back-to-back eval calls are minutes, not the hours training takes).
- After a force-kill of the Editor, `rm -f Temp/UnityLockfile` before it can relaunch.
- MCP-Unity calls (`recompile_scripts`, `execute_menu_item`) frequently **time out client-side but
  succeed server-side** — verify via `~/Library/Logs/Unity/Editor.log` (grep for the expected log
  line), not the tool's timeout.
- **The headless build BAKES the scene** (sensors, obs config, brain params). After any scene/sim
  change, rebuild before trusting a run. `start_training.sh` now records the scene's wired configs and
  warns on a stale build.
- Commit/push only when the user asks. End commit messages with the Co-Authored-By trailer.

## Key facts

- **LOOK is a policy *inside* our action space** — `ElevatorHeuristics.CollectiveLook` returns `int[E]`
  through the same `Building.ApplyAction` the agent uses. Its reward is the achievable target; if RL
  scores worse, that is an optimization/search failure, not a representability limit (E13f).
- ML-Agents trainers available here: **ppo, sac, poca**; reward signals: extrinsic, GAIL, curiosity,
  RND. DQN/D3QN (the reference paper's algo) is NOT native — custom torch, avoid unless last resort.
- Much of the R/A/C experiment matrix is **already implemented** (per-car agents, attention, OD-conv,
  arrival-rate obs, AS1 target-floor, `Heuristic()`-emits-LOOK) — most arms need only a config +
  rebuild to start.
