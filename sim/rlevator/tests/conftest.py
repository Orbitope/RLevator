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
    # deterministic guard against silent dynamics/reward drift. Add a rung's
    # value once measured (RLEVATOR_RUNG=<r> pytest reports the actual mean).
    look_expected = {
        "S": 837.1720500000081,
        "M": 1479.7684,
        "L": 2548.296999999994,
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
