"""Actor-critic net + AS0 action masking, shared by train.py and eval.py.

Architecture mirrors config/elevator_ppo_e2_s.yaml: a 256x2 MLP trunk feeding a
MultiDiscrete head (E independent size-6 branches, one per car) and a value head.
Observation normalization (config normalize: true) rides on a running mean/std.

Action masking is the Python port of ElevatorControllerAgent.WriteDiscreteActionMask
(AS0 / primitive), v1 assumptions (all cars in service, full [0,F-1] range):
  non-idle car -> only NOOP; idle car at floor f -> NOOP always, up iff f<F-1,
  down iff f>0, board-up/down iff that hall queue non-empty, unload iff a rider
  wants f. Illegal actions are silently no-ops in the env, but masking removes the
  wasted probability mass and is what lets PPO learn in a reasonable step budget.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rlevator import CAPACITY, E, F, IDLE, UNITS_PER_FLOOR

# Observation width (spec: Observations, blocks 1-7), rung-dependent:
# carFloor(E*F)+carActive(E)+carButtons(E*F)+hallButtons(2F)+hallCallAge(2F)
# +carMotion(4E)+carLoads(E). 98 at S, 254 at M.
OBS = 2 * E * F + 6 * E + 4 * F
N_CAR_ACTIONS = 6
_UPF = UNITS_PER_FLOOR


def legal_mask(env) -> torch.Tensor:
    """[N,E,6] bool legal-action mask for the current batched state."""
    n, dev = env.n, env.device
    fl = env.pos // _UPF                                        # [N,E] display floor (idle => exact)
    idle = env.car_state == IDLE                               # [N,E]
    up_cnt = env.up_count.gather(1, fl)                        # [N,E] queue at each car's floor
    dn_cnt = env.down_count.gather(1, fl)
    wants = (env.rider_dest == fl.unsqueeze(-1)).any(-1)        # [N,E] any rider wants this floor
    # board only if the car has free capacity. Unity's mask omits this check, so
    # a full car's "board" is a legal no-op — the exact attractor PPO's greedy
    # argmax parked on. LOOK never boards a full car, so masking it is fair and
    # removes the trap (see rlevator-rl-algorithm decision).
    free = (env.rider_dest != -1).sum(-1) < CAPACITY           # [N,E]

    mask = torch.zeros(n, E, N_CAR_ACTIONS, dtype=torch.bool, device=dev)
    mask[..., 0] = True                                        # NOOP always legal
    mask[..., 1] = idle & (fl < F - 1)                        # up
    mask[..., 2] = idle & (fl > 0)                            # down
    mask[..., 3] = idle & (up_cnt > 0) & free                 # board up (only if room)
    mask[..., 4] = idle & (dn_cnt > 0) & free                 # board down (only if room)
    mask[..., 5] = idle & wants                                # unload
    return mask


def pack_actions(per_car: torch.Tensor) -> torch.Tensor:
    """[N,E] per-car actions in {0..5} -> [N] packed base-6 integer (spec: Actions)."""
    pow6 = (6 ** torch.arange(E, device=per_car.device)).view(1, E)
    return (per_car * pow6).sum(-1)


class RunningNorm(nn.Module):
    """Observation normalizer (running mean/var), like ML-Agents normalize: true."""

    def __init__(self, dim: int = OBS):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        bmean = x.mean(0)
        bvar = x.var(0, unbiased=False)
        bn = torch.tensor(float(x.shape[0]))
        tot = self.count + bn
        delta = bmean - self.mean
        self.mean += delta * bn / tot
        m_a = self.var * self.count
        m_b = bvar * bn
        self.var = (m_a + m_b + delta.pow(2) * self.count * bn / tot) / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / (self.var.sqrt() + 1e-8)).clamp(-5.0, 5.0)


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.norm = RunningNorm(OBS)
        seq, d = [], OBS
        for _ in range(layers):
            seq += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        self.body = nn.Sequential(*seq)
        self.pi = nn.Linear(hidden, E * N_CAR_ACTIONS)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        """Returns (logits [N,E,6] masked, value [N])."""
        h = self.body(self.norm(obs))
        logits = self.pi(h).view(-1, E, N_CAR_ACTIONS)
        logits = logits.masked_fill(~mask, -1e9)
        return logits, self.v(h).squeeze(-1)

    @staticmethod
    def dist(logits: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=logits)


# ---------------------------------------------------------------------------
# Alternative architectures (same interface as ActorCritic: .norm, .dist,
# forward(obs, mask) -> (logits [N,E,6] masked, value [N])). All swap in via
# make_net(); the env/eval/mask/reward are untouched, so switching arch is a
# pure policy-side change (honors "no observation/reward tweaks" — the conv
# reshapes the EXISTING obs, it does not add channels).
# ---------------------------------------------------------------------------

# Observation block layout, mirroring fast.py observe() exactly. Offsets in F/E.
_OFF = {}
_o = 0
for _name, _sz in (("carFloor", E * F), ("carActive", E), ("carButtons", E * F),
                   ("hallButtons", 2 * F), ("hallAge", 2 * F), ("carMotion", 4 * E),
                   ("carLoads", E)):
    _OFF[_name] = (_o, _o + _sz)
    _o += _sz


def _split_obs(o: torch.Tensor) -> dict:
    """Flat obs [N, OBS] -> reshaped floor/car blocks (layout = observe())."""
    s = lambda k: o[:, _OFF[k][0]:_OFF[k][1]]
    return {
        "carFloor": s("carFloor").reshape(-1, E, F),        # [N,E,F]
        "carButtons": s("carButtons").reshape(-1, E, F),    # [N,E,F]
        "hallButtons": s("hallButtons").reshape(-1, F, 2),  # [N,F,2] (up,dn per floor)
        "hallAge": s("hallAge").reshape(-1, F, 2),          # [N,F,2]
        "carMotion": s("carMotion").reshape(-1, E, 4),      # [N,E,4] (3 one-hot + pos)
        "carActive": s("carActive"),                        # [N,E]
        "carLoads": s("carLoads"),                          # [N,E]
    }


class _DistMixin:
    @staticmethod
    def dist(logits: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=logits)


class ConvAC(nn.Module, _DistMixin):
    """Floor-grid conv done right: Conv1d over the FLOOR axis (weights shared
    across floors — the point for L's 30 floors) produces a per-floor embedding
    [N, C, F]. Each car then reads the conv column AT ITS OWN FLOOR (spatially
    local features) + a global pooled floor context + its own scalars, and
    decodes its action from that per-car vector. No giant flatten; the
    action-relevant per-car signal isn't drowned. Observation content unchanged."""

    def __init__(self, hidden: int = 256, conv_ch: int = 64, layers: int = 2):
        super().__init__()
        self.norm = RunningNorm(OBS)
        fch = 2 * E + 4                                       # carFloor+carButtons+2 hall
        self.conv = nn.Sequential(
            nn.Conv1d(fch, conv_ch, 3, padding=1), nn.ReLU(),
            nn.Conv1d(conv_ch, conv_ch, 3, padding=1), nn.ReLU())
        # per-car head sees: conv local (at its floor) + conv pooled context
        # + its scalars + its OWN view (floor one-hot + rider destinations). The
        # own-view is essential — without carButtons_i a car can't see where its
        # riders want to go, and can't learn to deliver.
        head_in = conv_ch + conv_ch + 6 + 2 * F
        dec, d = [], head_in
        for _ in range(layers):
            dec += [nn.Linear(d, hidden), nn.ReLU()]; d = hidden
        self.dec = nn.Sequential(*dec)
        self.pi = nn.Linear(hidden, N_CAR_ACTIONS)            # per-car (applied to each car)
        self.v = nn.Sequential(nn.Linear(conv_ch, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, obs, mask):
        b = _split_obs(self.norm(obs))                        # normalized features
        car_fl = _split_obs(obs)["carFloor"].argmax(-1)       # floor idx from RAW one-hot
        grid = torch.cat([b["carFloor"], b["carButtons"],
                          b["hallButtons"].transpose(1, 2), b["hallAge"].transpose(1, 2)],
                         dim=1)                                # [N, 2E+4, F]
        cmap = self.conv(grid)                                # [N, C, F] per-floor embedding
        pooled = cmap.mean(-1)                                # [N, C] global floor context
        local = cmap.gather(2, car_fl.unsqueeze(1).expand(-1, cmap.size(1), -1)) \
            .transpose(1, 2)                                  # [N, E, C] conv col at car's floor
        scal = torch.cat([b["carActive"].unsqueeze(-1), b["carMotion"],
                          b["carLoads"].unsqueeze(-1)], dim=-1)  # [N,E,6]
        own = torch.cat([b["carFloor"], b["carButtons"]], dim=-1)  # [N,E,2F] own floor+dests
        pc = torch.cat([local, pooled.unsqueeze(1).expand(-1, E, -1), scal, own], dim=-1)
        logits = self.pi(self.dec(pc)).masked_fill(~mask, -1e9)   # [N,E,6]
        return logits, self.v(pooled).squeeze(-1)


def _per_car_own_ctx(b: dict):
    """Per-car own-view [N,E,2F+6] and shared building context [N,4F]."""
    own = torch.cat([b["carFloor"], b["carButtons"], b["carMotion"],
                     b["carActive"].unsqueeze(-1), b["carLoads"].unsqueeze(-1)], dim=-1)
    ctx = torch.cat([b["hallButtons"].flatten(1), b["hallAge"].flatten(1)], dim=-1)
    return own, ctx


class PerCarAC(nn.Module, _DistMixin):
    """A1 — shared per-car encoder: one encoder applied to every car's
    [own-view ⊕ building-context]; decode each car's action from
    [car-embedding ⊕ pooled context]. Permutation-shared, fleet-size-independent."""

    def __init__(self, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.norm = RunningNorm(OBS)
        in_dim = (2 * F + 6) + 4 * F                          # own + context
        enc, d = [], in_dim
        for _ in range(layers):
            enc += [nn.Linear(d, hidden), nn.ReLU()]; d = hidden
        self.enc = nn.Sequential(*enc)
        self.dec = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, N_CAR_ACTIONS))
        self.v = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, obs, mask):
        b = _split_obs(self.norm(obs))
        own, ctx = _per_car_own_ctx(b)
        ctx_e = ctx.unsqueeze(1).expand(-1, E, -1)
        emb = self.enc(torch.cat([own, ctx_e], dim=-1))       # [N,E,H]
        pooled = emb.mean(1)                                  # [N,H]
        dec_in = torch.cat([emb, pooled.unsqueeze(1).expand(-1, E, -1)], dim=-1)
        logits = self.dec(dec_in).masked_fill(~mask, -1e9)    # [N,E,6]
        return logits, self.v(pooled).squeeze(-1)


class AttnAC(nn.Module, _DistMixin):
    """A2 — attention over cars: shared per-car embedding, then self-attention
    across the E car tokens so each car conditions on the others (targets the
    coordination failure directly), decode per car."""

    def __init__(self, hidden: int = 256, layers: int = 2, heads: int = 4):
        super().__init__()
        self.norm = RunningNorm(OBS)
        in_dim = (2 * F + 6) + 4 * F
        enc, d = [], in_dim
        for _ in range(layers):
            enc += [nn.Linear(d, hidden), nn.ReLU()]; d = hidden
        self.enc = nn.Sequential(*enc)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ln = nn.LayerNorm(hidden)
        self.dec = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, N_CAR_ACTIONS))
        self.v = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, obs, mask):
        b = _split_obs(self.norm(obs))
        own, ctx = _per_car_own_ctx(b)
        emb = self.enc(torch.cat([own, ctx.unsqueeze(1).expand(-1, E, -1)], dim=-1))  # [N,E,H]
        att, _ = self.attn(emb, emb, emb)
        emb = self.ln(emb + att)                              # [N,E,H]
        logits = self.dec(emb).masked_fill(~mask, -1e9)
        return logits, self.v(emb.mean(1)).squeeze(-1)


def make_net(arch: str = "flat", hidden: int = 256, layers: int = 2):
    """Policy/value net by name. All share the ActorCritic interface."""
    if arch == "flat":
        return ActorCritic(hidden=hidden, layers=layers)
    if arch == "conv":
        return ConvAC(hidden=hidden, layers=layers)
    if arch == "percar":
        return PerCarAC(hidden=hidden, layers=layers)
    if arch == "attn":
        return AttnAC(hidden=hidden, layers=layers)
    raise ValueError(f"unknown arch {arch!r} (flat|conv|percar|attn)")
