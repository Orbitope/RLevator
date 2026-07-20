"""Headless Python eval harness — mirrors Unity's EvalHarness + StatsCollector.

Produces the SAME sweep_summary.csv columns
(policy,seed,delivered,waitMean,waitP95,waitMax,abandoned,rejected,util,rwTotal)
under the SAME protocol (3600 s episode, 300 s warmup, per-decision sampling,
64-bin WaitHistogram), so a tensor-trained policy is comparable to the Unity
Runs/ CSVs on identical seeds/traffic — the plan's "every RL number relative to
LOOK on the identical traffic" rule (EXPERIMENT_PLAN.md §5), now Editor-free.

Metric definitions transcribed from Assets/ElevatorRL/Runtime/Stats/
StatsCollector.cs + WaitHistogram.cs:
  * everything is gated post-warmup (reset_metrics at the boundary);
  * wait* are over DELIVERED passengers' frozen queue-wait (rider_wait);
  * util = per-decision mean car Load (riders/capacity) over in-service cars,
    averaged across decisions;
  * rwTotal = reward accumulated warmup->end.

    <simulacrum venv>/python -m rlevator.eval --policy LOOK --seeds 1-5 \
        --out /tmp/py_S_midday.csv \
        --compare Runs/20260717-174511-E3-sweep-V2-S-midday-Midday/sweep_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import math

import torch

from rlevator import CAPACITY, E, MAX_WAIT_TICKS, N_ACTIONS
from rlevator.fast import _WAIT_BINS, RlevatorBatched
from rlevator.look import collective_look
from rlevator.policy import legal_mask, make_net, pack_actions

DECISION_SECONDS = 0.5
EVAL_SECONDS = 3600.0
WARMUP_SECONDS = 300.0
EVAL_DECISIONS = round(EVAL_SECONDS / DECISION_SECONDS)      # 7200
WARMUP_DECISIONS = round(WARMUP_SECONDS / DECISION_SECONDS)  # 600
WAIT_BINW_S = (MAX_WAIT_TICKS * 0.1) / _WAIT_BINS            # 45/64 s, matches Unity


def _percentile(hist: list[int], n: int, peak_s: float, p: float = 0.95) -> float:
    """Bit-for-bit port of WaitHistogram.Percentile (linear interp in-bin)."""
    if n == 0:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return peak_s
    target = math.ceil(p * n)
    cum = 0
    for i, c in enumerate(hist):
        prev = cum
        cum += c
        if cum >= target:
            lo = i * WAIT_BINW_S
            frac = (target - prev) / c if c > 0 else 0.0
            est = lo + frac * WAIT_BINW_S
            return est if est < peak_s else peak_s
    return peak_s


def _scripted_actions(env: RlevatorBatched) -> torch.Tensor:
    """One packed base-6 action per instance from the single-instance LOOK port."""
    pos = env.pos.tolist()
    rd = env.rider_dest.tolist()
    di = env.dir.tolist()
    cs = env.car_state.tolist()
    uc = env.up_count.tolist()
    dc = env.down_count.tolist()
    out = [
        collective_look({"pos": pos[j], "rider_dest": rd[j], "dir": di[j],
                         "car_state": cs[j], "up_count": uc[j], "down_count": dc[j]})
        for j in range(env.n)
    ]
    return torch.tensor(out, dtype=torch.int64, device=env.device)


def _ppo_actions_fn(ckpt: str, sample: bool = False):
    blob = torch.load(ckpt, map_location="cpu")
    net = make_net(blob.get("arch", "flat"), hidden=blob.get("hidden", 256),
                   layers=blob.get("layers", 2))
    net.load_state_dict(blob["net"])
    net.eval()

    @torch.no_grad()
    def act(env: RlevatorBatched) -> torch.Tensor:
        logits, _ = net(env.observe().to(torch.float32), legal_mask(env))
        a = net.dist(logits).sample() if sample else logits.argmax(-1)
        return pack_actions(a)
    return act


def _dqn_actions_fn(ckpt: str):
    from rlevator.dqn import DuelingQ, joint_mask
    net = DuelingQ()
    net.load_state_dict(torch.load(ckpt, map_location="cpu")["net"])
    net.eval()

    @torch.no_grad()
    def act(env: RlevatorBatched) -> torch.Tensor:
        q = net(env.observe().to(torch.float32), joint_mask(legal_mask(env)))
        return q.argmax(-1)                            # greedy argmax Q; index == packed action
    return act


def run_eval(policy: str, seeds: list[int], ckpt: str | None = None,
             sample: bool = False) -> list[dict]:
    """Run one eval episode per seed (batched) and return per-seed metric rows."""
    n = len(seeds)
    env = RlevatorBatched(n, max_decisions=EVAL_DECISIONS, collect_metrics=True)
    env.reset(torch.tensor(seeds, dtype=torch.int64))

    if policy == "LOOK":
        action_fn = _scripted_actions
    elif policy == "PPO":
        if not ckpt:
            raise ValueError("--ckpt required for policy PPO")
        action_fn = _ppo_actions_fn(ckpt, sample=sample)
    elif policy == "D3QN":
        if not ckpt:
            raise ValueError("--ckpt required for policy D3QN")
        action_fn = _dqn_actions_fn(ckpt)
    else:
        raise ValueError(f"unknown policy {policy!r}")

    util_sum = torch.zeros(n, dtype=torch.float64)
    rw_sum = torch.zeros(n, dtype=torch.float64)
    samples = 0
    for t in range(EVAL_DECISIONS):
        if t == WARMUP_DECISIONS:
            env.reset_metrics()
            util_sum.zero_(); rw_sum.zero_(); samples = 0
        actions = action_fn(env)
        _, reward, _, _ = env.step(actions)
        if t >= WARMUP_DECISIONS:
            load = (env.rider_dest != -1).sum(-1).to(torch.float64) / CAPACITY  # [n,E]
            util_sum += load.mean(dim=1)
            rw_sum += reward.to(torch.float64)
            samples += 1

    m = {k: v.tolist() for k, v in env.metrics().items()}
    rows = []
    for j, seed in enumerate(seeds):
        wc = m["wait_count"][j]
        peak_s = m["wait_peak_ticks"][j] * 0.1
        rows.append({
            "policy": policy, "seed": seed,
            "delivered": m["delivered"][j],
            "waitMean": (m["wait_sum_ticks"][j] / wc * 0.1) if wc else 0.0,
            "waitP95": _percentile(m["wait_hist"][j], wc, peak_s),
            "waitMax": peak_s,
            "abandoned": m["abandoned"][j],
            "rejected": m["rejected"][j],
            "util": util_sum[j].item() / samples if samples else 0.0,
            "rwTotal": rw_sum[j].item(),
        })
    return rows


COLUMNS = ["policy", "seed", "delivered", "waitMean", "waitP95", "waitMax",
           "abandoned", "rejected", "util", "rwTotal"]


def _fmt(row: dict) -> dict:
    return {
        "policy": row["policy"], "seed": row["seed"], "delivered": row["delivered"],
        "waitMean": f"{row['waitMean']:.2f}", "waitP95": f"{row['waitP95']:.2f}",
        "waitMax": f"{row['waitMax']:.2f}", "abandoned": row["abandoned"],
        "rejected": row["rejected"], "util": f"{row['util']:.3f}",
        "rwTotal": f"{row['rwTotal']:.0f}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="LOOK", help="LOOK or PPO")
    ap.add_argument("--ckpt", default=None, help="policy checkpoint (required for PPO)")
    ap.add_argument("--seeds", default="1-5", help="e.g. 1-5 or 1,2,3")
    ap.add_argument("--out", default=None, help="write sweep_summary.csv here")
    ap.add_argument("--compare", default=None, help="Unity sweep_summary.csv to diff against")
    args = ap.parse_args()

    if "-" in args.seeds and "," not in args.seeds:
        a, b = args.seeds.split("-"); seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    rows = run_eval(args.policy, seeds, ckpt=args.ckpt)

    print("  ".join(f"{c:>9}" for c in COLUMNS))
    for r in rows:
        f = _fmt(r)
        print("  ".join(f"{f[c]:>9}" for c in COLUMNS))

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow(_fmt(r))
        print(f"\nwrote {args.out}")

    if args.compare:
        import statistics
        uni = {}
        with open(args.compare) as fh:
            for r in csv.DictReader(fh):
                if r["policy"] == args.policy:
                    uni.setdefault(int(r["seed"]), r)
        print(f"\n=== Python vs Unity ({args.compare}), policy={args.policy} ===")
        keys = ["delivered", "waitMean", "waitP95", "waitMax", "abandoned", "rejected", "util"]
        print(f"{'metric':>10} {'py(mean)':>10} {'unity(mean)':>12} {'delta%':>8}")
        for k in keys:
            py = statistics.mean(float(r[k]) for r in rows)
            un = statistics.mean(float(uni[s][k]) for s in seeds if s in uni) if uni else float("nan")
            d = (py - un) / un * 100 if un else float("nan")
            print(f"{k:>10} {py:>10.2f} {un:>12.2f} {d:>+7.1f}%")


if __name__ == "__main__":
    main()
