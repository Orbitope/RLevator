"""PPO trainer on the tensor env — the Editor-free replacement for mlagents-learn.

Hyperparameters mirror config/elevator_ppo_e2_s.yaml (256x2 net, gamma 0.995,
lambda 0.95, clip 0.2, 3 epochs, lr 3e-4 linear, entropy 5e-3, normalize obs).
MultiDiscrete policy with AS0 action masking (policy.py). Trains on the native
2048-decision (1024 s) episode; eval.py scores the checkpoint at Unity's 3600 s
protocol against LOOK.

    <venv>/python -m rlevator.train --steps 3_000_000 --out models/ppo_s.pt
    <venv>/python -m rlevator.eval --policy PPO --ckpt models/ppo_s.pt --seeds 1-5 \
        --compare Runs/…-V2-S-midday-Midday/sweep_summary.csv
"""
from __future__ import annotations

import argparse
import os
import time
from collections import deque

import torch

from rlevator import N_ACTIONS
from rlevator.fast import RlevatorBatched
from rlevator.policy import E, OBS, legal_mask, make_net, pack_actions


def train(steps: int, out: str, n: int = 1024, rollout: int = 128,
          epochs: int = 3, minibatch: int = 4096, lr: float = 3e-4,
          gamma: float = 0.995, lam: float = 0.95, clip: float = 0.2,
          ent_coef: float = 5e-3, vf_coef: float = 0.5, seed: int = 1,
          hidden: int = 256, layers: int = 2, ent_frac: float = 0.6,
          arch: str = "flat") -> None:
    torch.manual_seed(seed)
    env = RlevatorBatched(n, compile=True)
    env.reset(torch.arange(1, n + 1, dtype=torch.int64))
    net = make_net(arch, hidden=hidden, layers=layers)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)

    updates = max(1, steps // (n * rollout))
    recent = deque(maxlen=20)          # per-update mean step-reward (sync-free)
    step_count = 0
    t0 = time.time()
    EP_LEN = 2048                      # decisions per episode (proxy scaling)
    best = float("-inf")               # keep the best-by-proxy checkpoint, not the

    def save():                        # (possibly collapsed) final one
        tmp = out + ".tmp"
        torch.save({"net": net.state_dict(), "hidden": hidden, "layers": layers,
                    "arch": arch}, tmp)
        os.replace(tmp, out)

    for up in range(updates):
        frac = 1.0 - up / updates                          # linear schedules
        for g in opt.param_groups:
            g["lr"] = lr * frac
        # Anneal entropy fully to 0 by 60% of training (then hold at 0) so the
        # greedy mode (argmax) sharpens onto the learned distribution. Unity
        # evals deterministically; a still-diffuse policy whose argmax is a
        # degenerate no-op fails greedy despite being competent sampled. Linear
        # decay left it too diffuse at 5M — this drives entropy to 0 earlier.
        ent_now = ent_coef * max(0.0, 1.0 - up / (ent_frac * updates))

        O, M, A, LP, V, R, D = [], [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(rollout):
                obs = env.observe().to(torch.float32)
                mask = legal_mask(env)
                net.norm.update(obs)
                logits, val = net(obs, mask)
                dist = net.dist(logits)
                a = dist.sample()                           # [N,E]
                lp = dist.log_prob(a).sum(-1)               # [N]
                _, rew, done, _ = env.step(pack_actions(a))
                rew = rew.to(torch.float32); done = done.to(torch.float32)
                O.append(obs); M.append(mask); A.append(a); LP.append(lp)
                V.append(val); R.append(rew); D.append(done)
            obs = env.observe().to(torch.float32)
            _, last_v = net(obs, legal_mask(env))
        step_count += n * rollout

        O = torch.stack(O); M = torch.stack(M); A = torch.stack(A)
        LP = torch.stack(LP); V = torch.stack(V); R = torch.stack(R); D = torch.stack(D)
        adv = torch.zeros_like(R); gae = torch.zeros(n)
        for t in reversed(range(rollout)):
            nextv = last_v if t == rollout - 1 else V[t + 1]
            delta = R[t] + gamma * nextv * (1 - D[t]) - V[t]
            gae = delta + gamma * lam * (1 - D[t]) * gae
            adv[t] = gae
        ret = adv + V
        recent.append(R.mean().item() * EP_LEN)            # episode-return proxy
        if len(recent) >= 5:                               # keep best-by-proxy (skip warmup)
            mret_now = sum(recent) / len(recent)
            if mret_now > best:
                best = mret_now
                save()

        bO = O.reshape(-1, OBS); bM = M.reshape(-1, *M.shape[2:]); bA = A.reshape(-1, E)
        bLP = LP.reshape(-1); bAdv = adv.reshape(-1); bRet = ret.reshape(-1)
        bAdv = (bAdv - bAdv.mean()) / (bAdv.std() + 1e-8)
        B = bO.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(B)
            for s in range(0, B, minibatch):
                idx = perm[s:s + minibatch]
                logits, val = net(bO[idx], bM[idx])
                dist = net.dist(logits)
                lp = dist.log_prob(bA[idx]).sum(-1)
                ratio = (lp - bLP[idx]).exp()
                a1 = ratio * bAdv[idx]
                a2 = torch.clamp(ratio, 1 - clip, 1 + clip) * bAdv[idx]
                ploss = -torch.min(a1, a2).mean()
                vloss = (val - bRet[idx]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = ploss + vf_coef * vloss - ent_now * ent
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        if up % 10 == 0 or up == updates - 1:
            mret = sum(recent) / len(recent) if recent else float("nan")
            sps = step_count / (time.time() - t0)
            print(f"upd {up:4d}/{updates}  steps {step_count:>9,}  "
                  f"ep_ret~ {mret:8.1f}  {sps:,.0f} steps/s")

    if not os.path.exists(out):        # tiny run that never hit the best-save path
        save()
    print(f"\nbest-by-proxy checkpoint at {out}  (net {hidden}x{layers}, best ep_ret~ "
          f"{best:.0f}); eval greedy vs LOOK to compare")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--out", default="models/ppo_s.pt")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ent-frac", type=float, default=0.6,
                    help="fraction of training over which entropy anneals to 0")
    ap.add_argument("--arch", default="flat", help="flat|conv|percar|attn")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    train(args.steps, args.out, n=args.n, rollout=args.rollout, seed=args.seed,
          hidden=args.hidden, layers=args.layers, ent_frac=args.ent_frac, arch=args.arch)
