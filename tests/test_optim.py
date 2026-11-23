"""src/optim.py's MuonAdam: the Muon/Adam parameter split, and the capturable optimizer against the eager one."""

import pytest
import torch

from src.networks import Policy
from src.optim import MuonAdam


def test_muonadam_split():
    """MuonAdam puts exactly the hidden matrices on Muon (2 per MLP), the rest on Adam."""
    torch.manual_seed(0)
    pi = Policy(20, 30, 5, hist=3, enc_hidden=(64, 32, 16), dec_hidden=(64, 32, 16))
    oa = MuonAdam(pi, [pi.enc, pi.dec], lr=1e-3, fused=False)
    n_muon = sum(len(g["params"]) for g in oa.muon.param_groups)
    n_adam = sum(len(g["params"]) for g in oa.adam.param_groups)
    assert n_muon == 4 and n_muon + n_adam == len(list(pi.parameters())), f"muon {n_muon}, adam {n_adam}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the capturable optimizer and the CUDA graph need CUDA")
@pytest.mark.filterwarnings("ignore:Please use the new API settings:UserWarning")
def test_capturable_matches_eager():
    """capturable MuonAdam (tensor lr, own Muon step, capturable fused Adam) == MuonAdam, eagerly and inside a CUDA graph.
    Muon orthogonalizes in bf16, so a 1e-7 difference flips bf16 roundings after a few steps; the yardstick is the
    spread of plain MuonAdam itself under a 1e-6 relative lr perturbation."""
    torch.manual_seed(0)
    dev = "cuda"
    tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        pis = [Policy(20, 30, 5, hist=3, enc_hidden=(64, 48, 16), dec_hidden=(64, 32, 16)).to(dev) for _ in range(4)]
        for q in pis[1:]:
            q.load_state_dict(pis[0].state_dict())
        init = [p.detach().clone() for p in pis[0].parameters()]
        mk = lambda q, cap: MuonAdam(q, [q.enc, q.dec], lr=1e-3, capturable=cap)  # noqa: E731
        opts = [mk(pis[0], False), mk(pis[1], True), mk(pis[2], True), mk(pis[3], False)]
        xo, xg = torch.randn(256, 3, 20, device=dev), torch.randn(256, 30, device=dev)
        lrs = [1e-3 * (1 - 0.02 * i) for i in range(20)]

        def stepf(k):
            opts[k].zero_grad(set_to_none=True)
            (pis[k](xo, xg) ** 2).mean().backward()
            opts[k].step()

        graph = None
        for i, lr in enumerate(lrs):
            for k in range(4):
                lr_k = lr * (1 + 1e-6) if k == 3 else lr
                for gr in opts[k].param_groups:
                    gr["lr"].fill_(lr_k) if torch.is_tensor(gr["lr"]) else gr.__setitem__("lr", lr_k)
            stepf(0), stepf(1), stepf(3)
            if i < 3:  # warm-up on a side stream, then capture on the next step and replay
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    stepf(2)
                torch.cuda.current_stream().wait_stream(side)
            else:
                if graph is None:
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        stepf(2)
                graph.replay()

        def rel(k):  # max over tensors of |p_0 - p_k| / |total change of p_0|
            return max(float((p.detach() - q.detach()).norm() / (p.detach() - p0).norm().clamp(min=1e-12))
                       for p, q, p0 in zip(pis[0].parameters(), pis[k].parameters(), init))  # fmt: skip

        floor = rel(3)
        assert rel(1) < max(3 * floor, 1e-4), (
            f"capturable MuonAdam != MuonAdam (20 steps, changing lr): max diff / total change {rel(1):.1e} (noise floor {floor:.1e})")
        assert rel(2) < max(3 * floor, 1e-4), f"capturable MuonAdam in a CUDA graph != MuonAdam: {rel(2):.1e}"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
