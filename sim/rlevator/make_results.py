"""Generate docs/results.json for the article's Results viz — from REAL evals.

Re-runs the eval harness (rlevator.eval.run_eval, seeds 1-5, Unity protocol:
3600 s episode / 300 s warmup / greedy+masked) against the saved checkpoints,
one (rung, reward) cell at a time. Numbers are regenerated, not hand-copied,
so the site's table is reproducible from the repo + checkpoints.

Each checkpoint appears only in the cell it was TRAINED in (its objective);
LOOK appears in every cell (behavior is reward-independent; only its reward
total differs with the shaping env). Cells with no trained policy just carry
LOOK. RLEVATOR_RUNG / RLEVATOR_SHAPING are process-wide (read at import), so
the parent shells out one child per cell.

    <simulacrum venv>/python -m rlevator.make_results          # orchestrate all cells
    ... --cell M unshaped                                       # (internal) one cell -> JSON on stdout
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import date
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parents[1]
DOCS = SIM_DIR.parent / "docs"

SEEDS = [1, 2, 3, 4, 5]

# (rung, reward) -> {method_key: checkpoint (relative to sim/) or None for LOOK}
# Only checkpoints trained under that cell's objective are listed in it.
CELLS: dict[tuple[str, str], dict[str, str | None]] = {
    ("S", "shaped"): {"LOOK": None, "flat": "models/ppo_s_fixed.pt",
                      "D3QN": "models/dqn_s_10m.pt"},
    ("S", "unshaped"): {"LOOK": None},
    ("M", "shaped"): {"LOOK": None, "flat": "models/ppo_m_bignet.pt",
                      "conv": "models/ppo_m_conv.pt", "percar": "models/ppo_m_percar.pt",
                      "attention": "models/ppo_m_attn.pt"},
    ("M", "unshaped"): {"LOOK": None, "flat": "models/ppo_m_noshaping.pt",
                        "attention": "models/ppo_m_attn_noshaping.pt"},
    ("L", "shaped"): {"LOOK": None, "flat": "models/ppo_l_bignet2.pt",
                      "attention": "models/ppo_l_attn.pt"},
    ("L", "unshaped"): {"LOOK": None, "flat": "models/ppo_l_noshaping.pt"},
}


def run_cell(rung: str, reward: str) -> dict:
    """Child mode: env vars already set; eval every method in this cell."""
    from rlevator.eval import run_eval  # import AFTER env vars are applied

    out: dict[str, dict] = {}
    for method, ckpt in CELLS[(rung, reward)].items():
        if ckpt is None:
            rows = run_eval("LOOK", SEEDS)
        elif method == "D3QN":
            rows = run_eval("D3QN", SEEDS, ckpt=str(SIM_DIR / ckpt))
        else:
            rows = run_eval("PPO", SEEDS, ckpt=str(SIM_DIR / ckpt))
        out[method] = {
            "throughput": round(statistics.mean(r["delivered"] for r in rows), 1),
            "avgWait": round(statistics.mean(r["waitMean"] for r in rows), 2),
            "waitP95": round(statistics.mean(r["waitP95"] for r in rows), 2),
            "abandoned": round(statistics.mean(r["abandoned"] for r in rows), 1),
            "reward": round(statistics.mean(r["rwTotal"] for r in rows)),
        }
        print(f"  {rung}/{reward}/{method}: {out[method]}", file=sys.stderr)
    return out


def orchestrate() -> None:
    results: dict[str, dict[str, dict]] = {}
    for (rung, reward) in CELLS:
        print(f"== cell {rung}/{reward} ==", file=sys.stderr)
        env = dict(os.environ, RLEVATOR_RUNG=rung,
                   RLEVATOR_SHAPING=("on" if reward == "shaped" else "off"))
        proc = subprocess.run(
            [sys.executable, "-m", "rlevator.make_results", "--cell", rung, reward],
            cwd=SIM_DIR, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"cell {rung}/{reward} failed:\n{proc.stderr[-3000:]}")
        print(proc.stderr, file=sys.stderr, end="")
        results.setdefault(rung, {})[reward] = json.loads(proc.stdout)

    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         cwd=SIM_DIR, capture_output=True, text=True).stdout.strip()
    blob = {
        "meta": {
            "seeds": SEEDS,
            "protocol": "3600 s episode, 300 s warmup, greedy + masked inference, "
                        "means over seeds; identical traffic per seed across methods",
            "generated": date.today().isoformat(),
            "git": git or None,
        },
        "results": results,
    }
    DOCS.mkdir(exist_ok=True)
    path = DOCS / "results.json"
    path.write_text(json.dumps(blob, indent=1) + "\n")
    print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", nargs=2, metavar=("RUNG", "REWARD"))
    args = ap.parse_args()
    if args.cell:
        rung, reward = args.cell
        assert os.environ.get("RLEVATOR_RUNG") == rung, "env/args mismatch"
        json.dump(run_cell(rung, reward), sys.stdout)
    else:
        orchestrate()
