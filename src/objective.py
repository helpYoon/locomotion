"""Actor objective over an imagined horizon: discounted rewards with survival weights, closed by the critic."""

from __future__ import annotations

import torch


def actor_objective(r: torch.Tensor, d: torch.Tensor, v_boot: torch.Tensor, gamma: float):
    """SHAC objective J = sum_{t<H} gamma^t c_t r_t + gamma^H c_H V(z_H), c_t = prod_{k<t} (1 - d_k).
    r, d: (H, B); v_boot: (B,) critic value of the last imagined state. Returns (J, c_t, c_H).
    The survival products are a loop of multiplications (the same products as cumprod, whose backward tests for zeros
    on the host, which a CUDA graph cannot capture)."""
    H, B = r.shape
    acc, cs = torch.ones_like(d[0]), []
    for t in range(H):
        acc = acc * (1 - d[t])
        cs.append(acc)
    alive = torch.stack(cs)
    c_t = torch.cat((torch.ones(1, B, device=r.device), alive[:-1]), 0)
    disc = gamma ** torch.arange(H, device=r.device, dtype=torch.float32).unsqueeze(-1)
    J = (disc * c_t * r).sum(0)
    c = alive[-1]
    boot = (gamma**H) * c * v_boot
    J = J + boot
    return J, c_t, c
