#!/usr/bin/env bash
# scripts/resume_training.sh <run-id> <config-yaml> <env-build-path> <num-envs>
#
# Resumes an existing run-id from its last checkpoint under a (typically bumped-max_steps)
# config, via mlagents-learn --resume. Reproducibility snapshot from the original
# start_training.sh run stays in Runs/training/<run-id>/ untouched; this appends a
# resume_<timestamp>/ subfolder recording what changed for the extension (new config, git state
# at resume time, refreshed pip freeze) rather than overwriting the original run's record.
set -euo pipefail

RUN_ID="${1:?Usage: resume_training.sh <run-id> <config-yaml> <env-build-path> <num-envs>}"
CONFIG="${2:?Usage: resume_training.sh <run-id> <config-yaml> <env-build-path> <num-envs>}"
ENV_BUILD="${3:?Usage: resume_training.sh <run-id> <config-yaml> <env-build-path> <num-envs>}"
NUM_ENVS="${4:-1}"
VENV="$HOME/mlagents-venv"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAP_DIR="$PROJECT_ROOT/Runs/training/$RUN_ID"

if [ ! -d "$SNAP_DIR" ]; then
  echo "[resume_training] ERROR: $SNAP_DIR does not exist — this isn't an existing run, use start_training.sh instead." >&2
  exit 1
fi
if [ ! -d "$PROJECT_ROOT/results/$RUN_ID" ]; then
  echo "[resume_training] ERROR: results/$RUN_ID not found — no checkpoint to resume from." >&2
  exit 1
fi

RESUME_DIR="$SNAP_DIR/resume_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RESUME_DIR"

git -C "$PROJECT_ROOT" rev-parse HEAD > "$RESUME_DIR/git_commit.txt"
git -C "$PROJECT_ROOT" status --short > "$RESUME_DIR/git_status.txt"
git -C "$PROJECT_ROOT" diff > "$RESUME_DIR/git_diff_uncommitted.patch"
cp "$PROJECT_ROOT/$CONFIG" "$RESUME_DIR/training_config.yaml"
"$VENV/bin/pip" freeze > "$RESUME_DIR/requirements_frozen.txt" 2>/dev/null || true
echo "$ENV_BUILD" > "$RESUME_DIR/env_build_path.txt"

echo "[resume_training] Resume snapshot written to $RESUME_DIR"
echo "[resume_training] Launching: mlagents-learn $CONFIG --run-id=$RUN_ID --resume --env=$ENV_BUILD --num-envs=$NUM_ENVS --no-graphics"
cd "$PROJECT_ROOT"
exec "$VENV/bin/mlagents-learn" "$CONFIG" --run-id="$RUN_ID" --resume \
  --env="$ENV_BUILD" --num-envs="$NUM_ENVS" --no-graphics --timeout-wait=300
