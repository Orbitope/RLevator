"""D3QN (Double + Dueling DQN) trainer on the tensor env — the reference paper's
algorithm family, and a value-based policy whose greedy eval (argmax Q) has no
diffuse-mode brittleness (the trap that made PPO's argmax park full cars).

Rung S (E=3) has a joint action space of only 6^E = 216, so this is exact
joint-action D3QN: the packed base-6 action IS the Q-head index. Per-car legal
masks (policy.legal_mask, = Unity's WriteDiscreteActionMask) combine into a
joint mask; illegal joint actions are -inf for both argmax and the Double target.
Scale to rung L/Z/H by factoring Q per car (VDN/QMIX) — see EXPERIMENT_PLAN.

    <venv>/python -m rlevator.dqn --steps 3_000_000 --out models/dqn_s.pt
    <venv>/python -m rlevator.eval --policy D3QN --ckpt models/dqn_s.pt --seeds 1-5 --compare Runs/…/sweep_summary.csv
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn

from rlevator import E, N_ACTIONS
from rlevator.fast import RlevatorBatched
from rlevator.policy import OBS, RunningNorm, legal_mask

# Joint action j in [0, N_ACTIONS) decodes to per-car digits (base-6); j is
# exactly the env's packed action, so no separate packing is needed.
_DIGITS = torch.stack([(torch.arange(N_ACTIONS) // (6 ** i)) % 6 for i in range(E)], dim=1)  # [216,E]


def joint_mask(percar: torch.Tensor) -> torch.Tensor:
    """[N,E,6] per-car legal mask -> [N,216] joint legal mask (AND across cars)."""
    dig = _DIGITS.to(percar.device)
    jm = torch.ones(percar.shape[0], N_ACTIONS, dtype=torch.bool, device=percar.device)
    for i in range(E):
        jm &= percar[:, i, :].index_select(1, dig[:, i])
    return jm


class DuelingQ(nn.Module):
    def __init__(self, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.norm = RunningNorm(OBS)
        seq, d = [], OBS
        for _ in range(layers):
            seq += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.body = nn.Sequential(*seq)
        self.v = nn.Linear(hidden, 1)
        self.a = nn.Linear(hidden, N_ACTIONS)

    def forward(self, obs: torch.Tensor, jmask: torch.Tensor) -> torch.Tensor:
        h = self.body(self.norm(obs))
        v = self.v(h)
        a = self.a(h)
        a = a - (a * jmask).sum(1, keepdim=True) / jmask.sum(1, keepdim=True).clamp(min=1)
        return (v + a).masked_fill(~jmask, -1e9)


class Replay:
    """Preallocated ring buffer. Stores per-car masks (18 bools) not joint masks."""

    def __init__(self, cap: int, dev):
        self.cap, self.dev, self.size, self.ptr = cap, dev, 0, 0
        self.obs = torch.zeros(cap, OBS, device=dev)
        self.nobs = torch.zeros(cap, OBS, device=dev)
        self.pcm = torch.zeros(cap, E, 6, dtype=torch.bool, device=dev)
        self.npcm = torch.zeros(cap, E, 6, dtype=torch.bool, device=dev)
        self.act = torch.zeros(cap, dtype=torch.int64, device=dev)
        self.rew = torch.zeros(cap, device=dev)
        self.done = torch.zeros(cap, device=dev)

    def push(self, obs, pcm, act, rew, nobs, npcm, done):
        b = obs.shape[0]
        idx = (self.ptr + torch.arange(b, device=self.dev)) % self.cap
        self.obs[idx] = obs; self.nobs[idx] = nobs
        self.pcm[idx] = pcm; self.npcm[idx] = npcm
        self.act[idx] = act; self.rew[idx] = rew; self.done[idx] = done
        self.ptr = (self.ptr + b) % self.cap
        self.size = min(self.size + b, self.cap)

    def sample(self, b):
        i = torch.randint(0, self.size, (b,), device=self.dev)
        return (self.obs[i], self.pcm[i], self.act[i], self.rew[i],
                self.nobs[i], self.npcm[i], self.done[i])


def train(steps: int, out: str, n: int = 256, buffer: int = 300_000,
          batch: int = 512, lr: float = 3e-4, gamma: float = 0.995,
          target_every: int = 2000, grad_per_step: int = 4, learn_start: int = 20_000,
          eps_start: float = 1.0, eps_end: float = 0.05, eps_frac: float = 0.4,
          seed: int = 1) -> None:
    torch.manual_seed(seed)
    dev = torch.device("cpu")
    env = RlevatorBatched(n)                     # eager: exploration/replay path, not compiled
    env.reset(torch.arange(1, n + 1, dtype=torch.int64))
    q = DuelingQ(); qt = DuelingQ(); qt.load_state_dict(q.state_dict())
    for p in qt.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(q.parameters(), lr=lr, eps=1e-5)
    buf = Replay(buffer, dev)

    obs = env.observe().to(torch.float32)
    pcm = legal_mask(env)
    env_steps = 0
    grad_steps = 0
    losses = []
    t0 = time.time()
    total_env_iters = steps // n

    def save(path):
        tmp = path + ".tmp"
        torch.save({"net": q.state_dict()}, tmp)
        os.replace(tmp, path)

    for it in range(total_env_iters):
        eps = max(eps_end, eps_start + (eps_end - eps_start) * (it / (eps_frac * total_env_iters)))
        with torch.no_grad():
            jm = joint_mask(pcm)
            q.norm.update(obs)
            qv = q(obs, jm)
            greedy = qv.argmax(1)
            rnd = torch.rand(n, N_ACTIONS, device=dev).masked_fill(~jm, -1.0).argmax(1)
            explore = torch.rand(n, device=dev) < eps
            act = torch.where(explore, rnd, greedy)
        nobs, rew, done, _ = env.step(act)
        nobs = nobs.to(torch.float32); rew = rew.to(torch.float32); done = done.to(torch.float32)
        npcm = legal_mask(env)
        buf.push(obs, pcm, act, rew, nobs, npcm, done)
        obs, pcm = nobs, npcm
        env_steps += n

        if buf.size >= learn_start:
            for _ in range(grad_per_step):
                bo, bpcm, ba, br, bno, bnpcm, bd = buf.sample(batch)
                bjm = joint_mask(bpcm); bnjm = joint_mask(bnpcm)
                with torch.no_grad():
                    a_star = q(bno, bnjm).argmax(1)                       # Double: online picks
                    q_next = qt(bno, bnjm).gather(1, a_star[:, None]).squeeze(1)
                    y = br + gamma * q_next * (1 - bd)
                qsa = q(bo, bjm).gather(1, ba[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(qsa, y)               # Huber
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
                grad_steps += 1
                if grad_steps % target_every == 0:                       # hard target update
                    qt.load_state_dict(q.state_dict())
                losses.append(loss.item())

        if it % 200 == 0 or it == total_env_iters - 1:
            ml = sum(losses[-500:]) / max(1, len(losses[-500:])) if losses else float("nan")
            sps = env_steps / (time.time() - t0)
            print(f"it {it:5d}  steps {env_steps:>9,}  eps {eps:.3f}  "
                  f"loss {ml:7.3f}  buf {buf.size:>7,}  {sps:,.0f} steps/s", flush=True)
        if it > 0 and it % 4000 == 0:                    # periodic atomic checkpoint
            save(out)

    save(out)
    print(f"\nsaved {out}  (eval greedy vs LOOK ~251 delivered / 5.9s waitMean)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--out", default="models/dqn_s.pt")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    train(args.steps, args.out, n=args.n, seed=args.seed)
