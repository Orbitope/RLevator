"""Throughput benchmark for the tensor env vs the headless Unity trainer.

Why this file exists (EXPERIMENT_PLAN.md §9): the migration off Unity is justified
by *iteration wall-clock*, and this is the reproducible measurement of that claim.
Run it after any change that could affect step cost (fast.py, obs size, rung size).

    <simulacrum venv>/python -m rlevator.bench --env-scaling
    <simulacrum venv>/python -m rlevator.bench --train-loop
    <simulacrum venv>/python -m rlevator.bench            # both

BASELINE — headless Unity, rung S, PPO (config/elevator_ppo_e2_s.yaml)
    run results/elev-e2-s-ppo-01: 5,000,192 agent-steps in 4010.7 s wall
    across --num-envs=20 --no-graphics  =>  1,247 agent-steps/sec total.

REFERENCE RESULT (Apple M-series, 11 logical / 5 perf cores, torch 2.13.0,
CPU only, 5 threads, 2026-07-18):
    env-only, compiled:   256->40k  1024->77k  4096->98k  16384->102k  65536->102k /s
    env-only, eager:      256->15k  1024->19k  4096->25k  16384->31k          /s
    end-to-end (env + 256x2 PPO):  N=512->36k  N=2048->46.5k  N=8192->43k     /s
  => end-to-end 46,479 steps/s at N=2048  =  37x the headless-Unity 1,247/s;
     a 5M-step rung-S run drops from 66.8 min (Unity) to ~1.8 min.
  Findings: torch.compile is worth ~3x; batching saturates ~N=4k-16k at ~100k/s
  (compute-bound on 5 CPU cores); learner cost pulls the end-to-end sweet spot
  down to N~2048. GPU (CUDA) is the next lever if bigger rungs / many concurrent
  experiments make even ~2 min/run the bottleneck.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from rlevator import N_ACTIONS
from rlevator.fast import RlevatorBatched

OBS = 98

# Headless-Unity rung-S PPO baseline (results/elev-e2-s-ppo-01).
UNITY_STEPS_PER_SEC = 5_000_192 / 4010.704786791932  # ~1247


def bench_env(n: int, steps: int, compile_: bool) -> float:
    """instance-steps/sec stepping the env with random actions (no learner)."""
    env = RlevatorBatched(n, compile=compile_, emit_final_states=False)
    env.reset(torch.arange(1, n + 1, dtype=torch.int64))
    g = torch.Generator().manual_seed(0)
    acts = torch.randint(0, N_ACTIONS, (steps, n), generator=g)
    warm = min(15, steps // 4)
    for t in range(warm):
        env.step(acts[t])
    t0 = time.time()
    for t in range(warm, steps):
        env.step(acts[t])
    return n * (steps - warm) / (time.time() - t0)


def env_scaling() -> None:
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}")
    print(f"Unity headless baseline: {UNITY_STEPS_PER_SEC:,.0f} agent-steps/sec (20 envs)\n")
    print("=== ENV-ONLY, EAGER ===")
    print(f"{'N':>7} {'inst-steps/s':>14} {'xUnity':>8}")
    for n in (1, 64, 256, 1024, 4096, 16384):
        r = bench_env(n, 120, False)
        print(f"{n:>7} {r:>14,.0f} {r / UNITY_STEPS_PER_SEC:>7.0f}x")
    print("\n=== ENV-ONLY, COMPILED ===")
    print(f"{'N':>7} {'inst-steps/s':>14} {'xUnity':>8}")
    for n in (256, 1024, 4096, 16384, 65536):
        r = bench_env(n, 80, True)
        print(f"{n:>7} {r:>14,.0f} {r / UNITY_STEPS_PER_SEC:>7.0f}x")


class ACNet(nn.Module):
    """256x2 actor-critic matching config/elevator_ppo_e2_s.yaml."""

    def __init__(self, hidden: int = 256, layers: int = 2):
        super().__init__()
        seq, d = [], OBS
        for _ in range(layers):
            seq += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.body = nn.Sequential(*seq)
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def train_loop(n: int, rollout_len: int, iters: int = 4,
               num_epoch: int = 3, minibatch: int = 2048,
               gamma: float = 0.995, lam: float = 0.95, clip: float = 0.2) -> tuple[float, float]:
    """End-to-end agent-steps/sec: compiled-env rollout + PPO update. Compute
    throughput only (this loop is NOT tuned to converge). iter 0 is discarded as
    compile warmup."""
    env = RlevatorBatched(n, compile=True, emit_final_states=False)
    env.reset(torch.arange(1, n + 1, dtype=torch.int64))
    net = ACNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    obs = env.observe().to(torch.float32)

    total_steps, t_start = 0, None
    for it in range(iters + 1):
        if it == 1:
            t_start = time.time()
        O, A, LP, V, R, D = [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(rollout_len):
                logits, val = net(obs)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                O.append(obs); A.append(a); LP.append(dist.log_prob(a)); V.append(val)
                nobs, rew, term, _ = env.step(a)
                obs = nobs.to(torch.float32)
                R.append(rew.to(torch.float32)); D.append(term.to(torch.float32))
        if it >= 1:
            total_steps += rollout_len * n
        with torch.no_grad():
            _, last_v = net(obs)
        O = torch.stack(O); A = torch.stack(A); LP = torch.stack(LP)
        V = torch.stack(V); R = torch.stack(R); D = torch.stack(D)
        adv = torch.zeros_like(R); gae = torch.zeros(n)
        for t in reversed(range(rollout_len)):
            nextv = last_v if t == rollout_len - 1 else V[t + 1]
            delta = R[t] + gamma * nextv * (1 - D[t]) - V[t]
            gae = delta + gamma * lam * (1 - D[t]) * gae
            adv[t] = gae
        ret = adv + V
        bO = O.reshape(-1, OBS); bA = A.reshape(-1); bLP = LP.reshape(-1)
        bAdv = adv.reshape(-1); bRet = ret.reshape(-1)
        bAdv = (bAdv - bAdv.mean()) / (bAdv.std() + 1e-8)
        B = bO.shape[0]
        for _ in range(num_epoch):
            perm = torch.randperm(B)
            for s in range(0, B, minibatch):
                idx = perm[s:s + minibatch]
                logits, val = net(bO[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(bA[idx]) - bLP[idx]).exp()
                surr = torch.min(ratio * bAdv[idx],
                                 torch.clamp(ratio, 1 - clip, 1 + clip) * bAdv[idx])
                loss = -surr.mean() + 0.5 * (val - bRet[idx]).pow(2).mean() \
                    - 0.005 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); opt.step()
    dt = time.time() - t_start
    return total_steps / dt, dt


def training() -> None:
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}")
    print(f"Unity headless baseline: {UNITY_STEPS_PER_SEC:,.0f} agent-steps/sec (20 envs)\n")
    print(f"{'N':>7} {'rollout':>8} {'steps/s (env+learner)':>22} {'xUnity':>8}")
    for n, rl in ((512, 128), (2048, 128), (8192, 64)):
        sps, dt = train_loop(n, rl)
        print(f"{n:>7} {rl:>8} {sps:>22,.0f} {sps / UNITY_STEPS_PER_SEC:>7.0f}x   ({dt:.1f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-scaling", action="store_true")
    ap.add_argument("--train-loop", action="store_true")
    args = ap.parse_args()
    both = not (args.env_scaling or args.train_loop)
    if args.env_scaling or both:
        env_scaling()
    if args.train_loop or both:
        print()
        training()
