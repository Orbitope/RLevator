#!/usr/bin/env bash
# scripts/start_training.sh <run-id> [config-yaml] [env-build-path] [num-envs] [preset-name]
#
# preset-name (default S) selects which Assets/ElevatorRL/Config/Presets/<name>_BuildingConfig.asset
# gets copied into the reproducibility snapshot — this MUST match whatever rung Training.unity's
# ElevatorController is actually pointed at (via Tools > Elevator RL > Point Agent At <X> Preset),
# or the snapshot silently mislabels which building the run used.
#
# With env-build-path set (e.g. Builds/HeadlessTrainer/RLevatorTrainer.app), runs
# headless against that standalone build instead of the live Editor: no window,
# no MCP/Editor-focus dependency, and num-envs parallel instances multiplexed
# into one training run. Omit env-build-path to fall back to Editor-connected
# training (the original single-instance, Play-button-driven mode).
#
# Reproducibility wrapper for mlagents-learn. Before launching training, snapshots
# everything needed to reconstruct exactly what produced a given run's results:
#   - git commit hash + uncommitted diff (the exact C# code — Building.WriteObservation,
#     reward weights, dispatch logic, everything — that was live for this run)
#   - the training config yaml actually used
#   - the ScriptableObject config assets actually wired to the ElevatorController agent
#     (BuildingConfig/RewardConfig/ObservationConfig/TrafficConfig) — these hold VALUES
#     (which observation blocks are on, reward weights, traffic pattern/intensity) that
#     live in Unity assets, invisible to ML-Agents' own results/ folder
#   - the frozen Python environment (pip freeze) — dependency drift is real; this is
#     the whole reason the ml-agents/grpcio install broke earlier this session
#   - the resolved com.unity.ml-agents package version
#
# ML-Agents itself already saves (into results/<run-id>/, not duplicated here):
#   the resolved training config, model checkpoints (.onnx/.pt), and TensorBoard summaries.
#
set -euo pipefail

RUN_ID="${1:?Usage: start_training.sh <run-id> [config-yaml] [env-build-path] [num-envs] [preset-name]}"
CONFIG="${2:-config/elevator_ppo.yaml}"
ENV_BUILD="${3:-}"
NUM_ENVS="${4:-1}"
PRESET="${5:-S}"
VENV="$HOME/mlagents-venv"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAP_DIR="$PROJECT_ROOT/Runs/training/$RUN_ID"

if [ -d "$SNAP_DIR" ]; then
  echo "[start_training] ERROR: $SNAP_DIR already exists — pick a unique run-id (or pass --force to mlagents-learn separately if you intend to resume)." >&2
  exit 1
fi
mkdir -p "$SNAP_DIR/configs"

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$SNAP_DIR/started_at_utc.txt"

git -C "$PROJECT_ROOT" rev-parse HEAD > "$SNAP_DIR/git_commit.txt"
git -C "$PROJECT_ROOT" status --short > "$SNAP_DIR/git_status.txt"
git -C "$PROJECT_ROOT" diff > "$SNAP_DIR/git_diff_uncommitted.patch"
if [ -s "$SNAP_DIR/git_diff_uncommitted.patch" ]; then
  echo "[start_training] WARNING: uncommitted changes present — captured as a patch," \
       "but committing before real (non-sanity) runs is strongly recommended." >&2
fi

cp "$PROJECT_ROOT/$CONFIG" "$SNAP_DIR/training_config.yaml"
"$VENV/bin/pip" freeze > "$SNAP_DIR/requirements_frozen.txt" 2>/dev/null || true
grep -A6 '"com.unity.ml-agents"' "$PROJECT_ROOT/Packages/packages-lock.json" > "$SNAP_DIR/ml_agents_package_version.txt" || true

# Config ScriptableObjects wired to ElevatorController in Assets/Scenes/Training.unity
cp "$PROJECT_ROOT/Assets/ElevatorRL/Config/Presets/${PRESET}_BuildingConfig.asset" "$SNAP_DIR/configs/BuildingConfig.asset"
cp "$PROJECT_ROOT/Assets/ElevatorRL/Config/RewardConfig.asset"      "$SNAP_DIR/configs/"
cp "$PROJECT_ROOT/Assets/ElevatorRL/Config/ObservationConfig.asset" "$SNAP_DIR/configs/"
cp "$PROJECT_ROOT/Assets/ElevatorRL/Config/TrafficConfig.asset"     "$SNAP_DIR/configs/"

echo "[start_training] Reproducibility snapshot written to $SNAP_DIR"
cd "$PROJECT_ROOT"

if [ -n "$ENV_BUILD" ]; then
  echo "$ENV_BUILD" > "$SNAP_DIR/env_build_path.txt"
  echo "[start_training] Launching headless: mlagents-learn $CONFIG --run-id=$RUN_ID --env=$ENV_BUILD --num-envs=$NUM_ENVS --no-graphics"
  exec "$VENV/bin/mlagents-learn" "$CONFIG" --run-id="$RUN_ID" \
    --env="$ENV_BUILD" --num-envs="$NUM_ENVS" --no-graphics --timeout-wait=300
else
  echo "[start_training] Launching: mlagents-learn $CONFIG --run-id=$RUN_ID --timeout-wait=300"
  # --timeout-wait generous (default is 60s) since bringing the Unity Editor to the front to
  # press Play can take a while (see mcp-unity focus notes elsewhere in this project).
  exec "$VENV/bin/mlagents-learn" "$CONFIG" --run-id="$RUN_ID" --timeout-wait=300
fi
