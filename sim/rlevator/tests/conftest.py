import sys
from pathlib import Path

import pytest

ENV_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENV_ROOT.parent))

pytest_plugins = ["simulacrum.harness.plugin"]


@pytest.fixture
def harness_config():
    from simulacrum.harness import (
        DiscreteActionSampler, HarnessConfig, ScriptedPolicy,
    )

    from rlevator.fast import RlevatorBatched
    from rlevator.look import collective_look
    from rlevator.reference import RlevatorReference

    from rlevator import N_ACTIONS, RUNG

    # LOOK mean return over 20 episodes (base_seed 1000), pinned per rung — the
    # deterministic guard against silent dynamics/reward drift. Values are for
    # the DEFAULT reward (no toward/away shaping, §9.4). (Shaped values, pre-§9.4:
    # S 837.17 / M 1479.77 / L 2548.30 — used if RLEVATOR_SHAPING=on.)
    look_expected = {
        "S": 728.9308000000067,
        "M": 1112.9284000000084,
        "L": 1490.7670000000003,
        "Z": 11.847799999999847,   # zoning overloads LOOK at intensity 1.0 (§E4)
    }

    return HarnessConfig(
        name="rlevator",
        root=ENV_ROOT,
        reference_factory=RlevatorReference,
        batched_factory=lambda n, debug=False: RlevatorBatched(n, debug=debug),
        # Training shape: compiled step core (bit-checked against eager by
        # the parity test), no per-step terminal JSON.
        benchmark_factory=lambda n, debug=False: RlevatorBatched(
            n, debug=debug, compile=True, emit_final_states=False),
        action_sampler=DiscreteActionSampler(n_actions=N_ACTIONS),  # 6**E = 216
        scripted_policies=[
            # LOOK is the project's classical baseline (Unity CollectiveLook
            # port). Expected return measured on these exact seeds; the run
            # is deterministic, so tol only needs to absorb nothing — it
            # guards against silent dynamics/reward drift.
            ScriptedPolicy(
                name="look",
                policy=lambda state, t: collective_look(state),
                expected_return=look_expected[RUNG],
                tol=1.0,
            ),
        ],
    )
