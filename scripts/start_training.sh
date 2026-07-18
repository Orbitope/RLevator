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

# --- Wired-config audit (guards the silent-misconfiguration bug class that has bitten 3x:
# E5's obs-config eval bug, E13b's stale model import, and the grid sensor still attached when
# launching E14a). The fixed-path copies above record what the assets CONTAIN, but not which assets
# the scene is actually WIRED to — those can differ (they did for every E5 run: the snapshot held
# the baseline ObservationConfig while the scene used ObservationConfig_FullState/etc.). This block
# records the truth: the guid the scene references, resolved to an asset path, plus which optional
# sensor components are attached. Also copies the ACTUAL wired obs config, not just the default one.
SCENE="$PROJECT_ROOT/Assets/Scenes/Training.unity"
{
  echo "# What Assets/Scenes/Training.unity ACTUALLY references at launch time (guid -> asset)."
  echo "# CAVEAT: if ENV_BUILD was pre-built earlier and the scene has since been re-pointed at a"
  echo "# different rung/config (the batch pre-build-everything-then-train workflow), this block"
  echo "# reflects the CURRENT scene, NOT necessarily what's baked into ENV_BUILD's binary. The"
  echo "# reproducibility snapshot's BuildingConfig copy below is unaffected -- it uses the explicit"
  echo "# preset-name arg, not live scene state -- but treat THIS printed block as informational only"
  echo "# in that workflow, not as proof of what ENV_BUILD is running."
  for key in buildingConfig rewardConfig observationConfig trafficConfig; do
    guid=$(grep -m1 "  $key:" "$SCENE" | grep -oE 'guid: [0-9a-f]+' | awk '{print $2}')
    if [ -n "$guid" ]; then
      meta=$(grep -rl "guid: $guid" "$PROJECT_ROOT/Assets" --include="*.meta" 2>/dev/null | head -1)
      asset="${meta%.meta}"
      echo "$key: guid=$guid asset=${asset#$PROJECT_ROOT/}"
      if [ "$key" = "observationConfig" ] && [ -f "$asset" ]; then
        cp "$asset" "$SNAP_DIR/configs/ObservationConfig.WIRED.asset"
      fi
    fi
  done
  echo "# Optional sensor components attached to the scene (instances by script guid):"
  for s in FloorGridSensorComponent FloorODSensorComponent; do
    g=$(grep -m1 '^guid:' "$PROJECT_ROOT/Assets/ElevatorRL/Runtime/$s.cs.meta" 2>/dev/null | awk '{print $2}')
    [ -n "$g" ] && echo "$s: $(grep -c "$g" "$SCENE") instance(s)"
  done
} > "$SNAP_DIR/configs/wired_configs.txt"
echo "[start_training] Wired-config audit:"
sed 's/^/[start_training]   /' "$SNAP_DIR/configs/wired_configs.txt" | grep -v '^#' || true

# --- Stale-build guard: the headless build BAKES the scene (sensors, obs config, brain params).
# If the scene was edited after the build, the run silently uses the OLD configuration.
if [ -n "$ENV_BUILD" ]; then
  BUILD_BIN=$(ls "$PROJECT_ROOT/$ENV_BUILD"/Contents/MacOS/* 2>/dev/null | head -1)
  if [ -n "$BUILD_BIN" ] && [ "$SCENE" -nt "$BUILD_BIN" ]; then
    echo "[start_training] WARNING: Training.unity is NEWER than the headless build — the build is" \
         "likely STALE and bakes an older scene. Rebuild before trusting this run." >&2
  fi
fi

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
