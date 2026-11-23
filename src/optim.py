"""Optimizers: Muon (hidden matrices) + Adam.

``MuonAdam``: Muon (Jordan et al. 2024; torch.optim.Muon) on the hidden weight matrices of the given MLPs, Adam on
their first and last Linear and on all vectors. Muon's step is rescaled to the update RMS of Adam
(adjust_lr_fn='match_rms_adamw', Liu et al. 2025), so the Adam learning rate is reused unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _linears(m: nn.Module) -> list[nn.Linear]:
    return [x for x in m.modules() if isinstance(x, nn.Linear)]


class MuonAdam:
    """Muon on hidden matrices of ``mlps`` (all Linear layers except each MLP's first and last), Adam elsewhere.

    ``capturable`` (train.py's CUDA graphs): the learning rate is a 0-dim device tensor (set it with ``fill_``), Adam is
    the capturable fused kernel, and the Muon step is done here with the same arithmetic as torch.optim.Muon but with
    the tensor learning rate (torch.optim.Muon reads a tensor lr on the host, so a captured step would freeze it)."""

    def __init__(self, model: nn.Module, mlps: list[nn.Module], lr: float, betas=(0.9, 0.95), fused: bool = True,
                 capturable: bool = False):  # fmt: skip
        hidden = {id(l.weight) for m in mlps for l in _linears(m)[1:-1]}
        muon = [p for p in model.parameters() if id(p) in hidden]
        adam = [p for p in model.parameters() if id(p) not in hidden and p.requires_grad]
        self.capturable = capturable
        if capturable:
            dev = muon[0].device
            self.muon = torch.optim.Muon(muon, lr=torch.tensor(float(lr), device=dev), weight_decay=0.0, adjust_lr_fn="match_rms_adamw")
            self.adam = torch.optim.Adam(adam, lr=torch.tensor(float(lr), device=dev), betas=betas, fused=True, capturable=True)
        else:
            self.muon = torch.optim.Muon(muon, lr=lr, weight_decay=0.0, adjust_lr_fn="match_rms_adamw")
            self.adam = torch.optim.Adam(adam, lr=lr, betas=betas, fused=fused)

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adam.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none), self.adam.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _muon_step_tensor_lr(self, ok: torch.Tensor | None = None):
        from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

        for g in self.muon.param_groups:
            assert g["weight_decay"] == 0.0
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.muon.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                buf = st["momentum_buffer"]
                if ok is None:
                    buf.lerp_(p.grad, 1 - g["momentum"])
                else:
                    buf.copy_(torch.where(ok, buf.lerp(p.grad, 1 - g["momentum"]), buf))
                update = p.grad.lerp(buf, g["momentum"]) if g["nesterov"] else buf
                update = _zeropower_via_newtonschulz(update, g["ns_coefficients"], g["ns_steps"], g["eps"])
                ratio = _adjust_lr(1.0, g["adjust_lr_fn"], p.shape)
                d = update.to(p.dtype) * (g["lr"] * -ratio)  # fp32 product (a bf16 update x 0-dim tensor stays bf16)
                p.add_(d if ok is None else torch.where(ok, d, 0.0))

    @torch.no_grad()
    def step(self, ok: torch.Tensor | None = None):
        """``ok`` (with ``capturable``): optional 0-dim bool device tensor; where it is false the whole step is skipped
        without a host read (parameters, momenta, Adam moments and step counts unchanged). The gradients must be finite
        (zero them first). A gated step always uses the tensor-lr Muon arithmetic, capturable or not."""
        if ok is not None:
            self._muon_step_tensor_lr(ok)
            assert self.adam.defaults["fused"], "needs the fused Adam (fused=True; capturable=True implies it)"
            self.adam.found_inf = (~ok).float()  # the fused kernel skips the step and undoes the step count
            try:
                self.adam.step()
            finally:
                self.adam.found_inf = None
            return
        if self.capturable:
            self._muon_step_tensor_lr()
        else:
            self.muon.step()
        self.adam.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}
