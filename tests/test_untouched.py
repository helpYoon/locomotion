"""The t1 and t2 robots compute what they computed before the t2_measured work (2026-11-20): a short rollout of one
existing policy of each in the training simulator on the CPU equals a stored one. Skipped where a checkpoint or its
reference is missing (both live under logs/, which git ignores).

Write the references (from a tree known to be good): CUDA_VISIBLE_DEVICES= .venv/bin/python tests/test_untouched.py --write"""

import os
import sys

import numpy as np
import pytest
import torch

torch.set_num_threads(4)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = [("logs/t2_vel_stand_s3/policy/ckpt_013000.pt", "logs/regression/t2_s3.npz"),
         ("logs/t1_vel_ankle_sharp_x2_s1/policy/ckpt_013000.pt", "logs/regression/t1_sharp.npz")]  # fmt: skip
CMDS = [[0.5, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.8], [0.0, 0.0, 0.0]]  # one world each
STEPS = 20
TOL = 0.0  # two runs of --write were bitwise equal (2026-11-20)


def rollout(ckpt: str) -> np.ndarray:
    """Joint and root positions (STEPS, 4, J + 3) of 4 worlds under CMDS from the standing reset."""
    from utils.velocity_rig import Rig  # here: it starts Warp

    rig = Rig(os.path.join(REPO, ckpt), len(CMDS), dev="cpu")
    rig.reset()
    out = []
    for _ in range(STEPS):
        rig.step(CMDS)
        S = rig.env.state()
        out.append(torch.cat((S[4], S[0]), -1).numpy().copy())
    return np.stack(out)


@pytest.mark.parametrize("ckpt,ref", CASES)
def test_rollout_unchanged(ckpt, ref):
    if torch.cuda.is_available():
        pytest.skip("run with CUDA_VISIBLE_DEVICES= (the references are CPU rollouts)")
    if not (os.path.exists(os.path.join(REPO, ckpt)) and os.path.exists(os.path.join(REPO, ref))):
        pytest.skip("checkpoint or reference missing")
    np.testing.assert_allclose(rollout(ckpt), np.load(os.path.join(REPO, ref))["x"], rtol=0, atol=TOL)


if __name__ == "__main__":
    assert sys.argv[1:] == ["--write"], __doc__
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    for ckpt, ref in CASES:
        os.makedirs(os.path.dirname(ref), exist_ok=True)
        np.savez(ref, x=rollout(ckpt))
        print("wrote", ref)
