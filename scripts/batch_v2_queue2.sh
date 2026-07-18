#!/usr/bin/env bash
# scripts/batch_v2_queue2.sh — chained V2 training queue #2, no Unity Editor interaction required.
# Pattern + intensity are both env-param overrides now (ElevatorControllerAgent.cs), so this queue
# reuses the S/M/L builds from queue1 unchanged -- no rebuild needed at all for this batch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "[batch2] 1/4: elev-v2-m-downpeak-01 (M, DownPeak, intensity 1.0)"
./scripts/start_training.sh elev-v2-m-downpeak-01 config/elevator_ppo_v2_m_downpeak.yaml Builds/HeadlessTrainer_M/RLevatorTrainer.app 20 M

echo "[batch2] 2/4: elev-v2-m-lunch-01 (M, Lunch, intensity 1.0)"
./scripts/start_training.sh elev-v2-m-lunch-01 config/elevator_ppo_v2_m_lunch.yaml Builds/HeadlessTrainer_M/RLevatorTrainer.app 20 M

echo "[batch2] 3/4: elev-v2-l-midday-stress-01 (L, Midday, intensity 1.5)"
./scripts/start_training.sh elev-v2-l-midday-stress-01 config/elevator_ppo_v2_l_midday_stress.yaml Builds/HeadlessTrainer_L/RLevatorTrainer.app 20 L

echo "[batch2] 4/4: elev-v2-m-daycycle-01 (M, day-cycle generalist, intensity 1.0)"
./scripts/start_training.sh elev-v2-m-daycycle-01 config/elevator_ppo_v2_m_daycycle.yaml Builds/HeadlessTrainer_M/RLevatorTrainer.app 20 M

echo "[batch2] all 4 runs complete"
