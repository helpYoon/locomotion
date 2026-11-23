"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env)."""

from __future__ import annotations

import torch

from src import landing
from src.mdp import GAIT_DIM, state_dim


def fields(num_j: int, xi_dim: int, a_dim: int | None = None, vel: bool = False, land: bool = False,
           gait: bool = False) -> dict:
    out = {
        "s": (state_dim(num_j), torch.float32),
        "s1": (state_dim(num_j), torch.float32),
        "a": (num_j, torch.float32),
        "a_prev": (a_dim or num_j, torch.float32),
        "xi": (xi_dim, torch.float32),
        "r": ((), torch.float32),
        "f": ((), torch.long),
        "ep_step": ((), torch.int32),
        "done": ((), torch.bool),
        "cut": ((), torch.bool),
    }
    if vel:  # --vel_frac: the velocity command in force during the transition, and whether it was read from the clip
        out["cmd"] = (3, torch.float32)
        out["cmd_ref"] = ((), torch.bool)
    if gait:  # --sym_weight: the velocity worlds' gait memory at the transition's start state (src/mdp.py GAIT_DIM)
        out["gait"] = (GAIT_DIM, torch.float32)
    if land:  # --land_frac: the landing command state in force, and whether it follows a clip's label
        out["land"] = (landing.STATE_DIM, torch.float32)
        out["lab"] = ((), torch.bool)
    return out


class ReplayBuffer:
    def __init__(self, T: int, N: int, device: str, num_j: int, xi_dim: int, a_dim: int | None = None,
                 vel: bool = False, land: bool = False, gait: bool = False):  # fmt: skip
        self.T, self.N, self.device = T, N, device
        self.fields = fields(num_j, xi_dim, a_dim, vel, land, gait)
        self.data = {}
        for k, (shape, dt) in self.fields.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0
        # use_device_ptr (train.py's CUDA graphs): a device copy of ``ptr``, so that sampling captured in a CUDA graph
        # follows the write head
        self.ptr_t = None
        # and a device copy of ``size``: sampling bounds follow it (uniform draws scaled by the fill level instead of
        # randint with a host bound), so the captured updates stay valid while the buffer fills
        self.size_t = None

    start_split = None  # train.py --start_prac_frac: (n0, p)
    env_lo = 0  # train.py --vel_frac, --land_frac: envs [0, env_lo) follow a command; sample_starts uses the rest

    def use_device_ptr(self):
        self.ptr_t = torch.full((), self.ptr, dtype=torch.long, device=self.device)
        self.size_t = torch.full((), self.size, dtype=torch.long, device=self.device)

    def _randint(self, n, B: int) -> torch.Tensor:
        """Uniform integers in [0, n) with n a host int, or (after use_device_ptr) a device tensor."""
        if self.size_t is None:
            return torch.randint(0, n, (B,), device=self.device)
        return torch.minimum((torch.rand(B, device=self.device) * n).long(), n - 1)

    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)
        if self.ptr_t is not None:
            self.ptr_t.fill_(self.ptr)
        if self.size_t is not None:
            self.size_t.fill_(self.size)

    def add_device(self, tr: dict):
        """add() at the device-side write head (capturable: train.py's captured collection). Advances ptr_t / size_t
        on the device; the host-side ptr / size are advanced by advance_host() after each replay."""
        row = self.ptr_t.view(1)
        for k in self.fields:
            self.data[k].index_copy_(0, row, tr[k].unsqueeze(0).to(self.data[k].dtype))  # add() casts the same way
        self.ptr_t.copy_((self.ptr_t + 1) % self.T)
        self.size_t.copy_(torch.clamp(self.size_t + 1, max=self.T))

    def advance_host(self):
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        ptr = self.ptr_t if self.ptr_t is not None else self.ptr
        size = self.size_t if self.size_t is not None else self.size
        return (ptr - size + age) % self.T

    def sample_windows(self, B: int, k: int):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        t0 = self._randint((self.size_t if self.size_t is not None else self.size) - k + 1, B)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in ("s", "s1", "a", "a_prev", "xi", "cut", "f")}
        # transition j is valid if the state stayed continuous (no episode end, no push) after 0..j-1
        d = out["cut"].float()
        mask = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        out["mask"] = mask
        return out

    def sample_starts(self, B: int, hist: int):
        """Start points for imagination plus the indices of their observation history (oldest first)."""
        age = self._randint(self.size_t if self.size_t is not None else self.size, B)
        if self.start_split is None:
            env = torch.randint(self.env_lo, self.N, (B,), device=self.device)
        else:  # train.py --start_prac_frac: share p of the starts from envs [env_lo, n0), the rest from [n0, N)
            n0, p = self.start_split
            lo = self.env_lo
            e0 = lo + (torch.rand(B, device=self.device) * (n0 - lo)).long().clamp(max=n0 - lo - 1)
            e1 = n0 + (torch.rand(B, device=self.device) * (self.N - n0)).long().clamp(max=self.N - n0 - 1)
            pick0 = torch.rand(B, device=self.device) < p
            env = torch.where(pick0, e0, e1)
        return self.starts_at(age, env, hist)

    def sample_vel_starts(self, B: int, hist: int, n_vel: int):
        """sample_starts over the velocity envs [0, n_vel) (uniform), each start with its command, the command's
        source and its episode step (and its gait memory, where the buffer keeps it)."""
        age = self._randint(self.size_t if self.size_t is not None else self.size, B)
        env = (torch.rand(B, device=self.device) * n_vel).long().clamp(max=n_vel - 1)
        return self.starts_at(age, env, hist, extra=("cmd", "cmd_ref", "ep_step") + (("gait",) if "gait" in self.data else ()))

    def sample_land_starts(self, B: int, hist: int, n_vel: int, n_land: int):
        """sample_starts over the landing envs [n_vel, n_vel + n_land) (uniform), each start with its command state,
        whether it follows a clip's label and its episode step."""
        age = self._randint(self.size_t if self.size_t is not None else self.size, B)
        env = n_vel + (torch.rand(B, device=self.device) * n_land).long().clamp(max=n_land - 1)
        return self.starts_at(age, env, hist, extra=("land", "lab", "ep_step"))

    def starts_at(self, age: torch.Tensor, env: torch.Tensor, hist: int, extra: tuple[str, ...] = ()):
        """Start points at the transitions (age, env) (age 0 = oldest) plus their observation history (oldest first);
        ``extra``: further fields of the start transitions."""
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))  # stay inside the episode
        back = torch.minimum(back, age.unsqueeze(-1))  # stay inside the buffer
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {k: self.data[k][row, env] for k in ("s", "a_prev", "xi", "f") + extra}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev", "xi", "f")}
        return start, hist_d

