#!/usr/bin/env bash
# scripts/batch_v2_queue1.sh — chained V2 training queue, no Unity Editor interaction required.
# Each run only needs its already-built .app (Builds/HeadlessTrainer_{S,M,L}/); training itself
# never touches the Editor. Runs strictly sequentially (each needs the full machine).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "[batch] 1/3: elev-v2-s-midday-stress-01 (S, Midday, intensity 1.5)"
./scripts/start_training.sh elev-v2-s-midday-stress-01 config/elevator_ppo_v2_s_midday_stress.yaml Builds/HeadlessTrainer_S/RLevatorTrainer.app 20 S

echo "[batch] 2/3: elev-v2-m-midday-stress-01 (M, Midday, intensity 1.5)"
./scripts/start_training.sh elev-v2-m-midday-stress-01 config/elevator_ppo_v2_m_midday_stress.yaml Builds/HeadlessTrainer_M/RLevatorTrainer.app 20 M

echo "[batch] 3/3: elev-v2-m-uppeak-01 (M, UpPeak, intensity 1.0)"
./scripts/start_training.sh elev-v2-m-uppeak-01 config/elevator_ppo_v2_m_uppeak.yaml Builds/HeadlessTrainer_M/RLevatorTrainer.app 20 M

echo "[batch] all 3 runs complete"
