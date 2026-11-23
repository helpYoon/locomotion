"""utils/rotations.py against scipy, plus gradchecks of its differentiable functions."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from utils import rotations as r

N = 5000
SC = Rotation.random(N, random_state=0)


def wxyz(rot):
    return np.roll(rot.as_quat(), 1, -1)  # scipy is xyzw


def err(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


@pytest.mark.parametrize("dt, tol", [(torch.float64, 1e-9), (torch.float32, 2e-5)])
def test_against_scipy(dt, tol):
    def T(x):
        return torch.tensor(np.asarray(x), dtype=dt)

    M, q = SC.as_matrix(), wxyz(SC)
    rv = SC.as_rotvec()
    assert err(r.quat_to_mat(T(q)), M) < tol
    qq = r.mat_to_quat(T(M)).numpy()
    assert err(np.abs((qq * q).sum(-1)), 1) < tol and (qq[:, 0] >= 0).all()
    angle = np.linalg.norm(rv, axis=-1)
    assert err(r.axis_angle_to_mat(T(rv / angle[:, None]), T(angle)), M) < tol
    assert err(r.so3_exp(T(rv)), M) < tol
    assert err(r.so3_log(T(M)), rv) < 10 * tol
    assert err(r.quat_to_rotvec(T(-3 * q)), rv) < 10 * tol  # non-unit, w < 0
    u = np.random.default_rng(2).normal(size=(N, 3))
    assert err((r.skew(T(rv)) @ T(u)[..., None])[..., 0], np.cross(rv, u)) < tol
    yaw = np.arctan2(M[:, 1, 0], M[:, 0, 0])
    # 1e-5 in float64: yaw_mat normalizes the projected x axis with 1e-9 under the square root
    assert err(r.yaw_mat(T(M)), Rotation.from_euler("z", yaw[:, None]).as_matrix()) < max(tol, 1e-5)
    assert err(r.mat_to_6d(T(M)), M[:, :, :2].reshape(N, 6)) < tol


_g = torch.Generator().manual_seed(0)
_V = (torch.rand(6, 3, generator=_g, dtype=torch.float64) - 0.5) * 4
_Q = torch.nn.functional.normalize(torch.rand(6, 4, generator=_g, dtype=torch.float64) - 0.5, dim=-1)
_R, _R2 = r.so3_exp(_V), r.so3_exp(_V.flip(0))
_AXIS = torch.tensor([0.0, 0.6, 0.8], dtype=torch.float64)
GRADCHECK = {
    "quat_to_mat": (r.quat_to_mat, (_Q,)),
    "mat_to_quat": (r.mat_to_quat, (_R,)),
    "axis_angle_to_mat": (lambda a: r.axis_angle_to_mat(_AXIS, a), (_V[:, 0],)),
    "so3_exp": (r.so3_exp, (_V,)),
    "so3_log": (r.so3_log, (_R,)),
    "quat_to_rotvec": (r.quat_to_rotvec, (_Q,)),
    "rot_angle_sq": (r.rot_angle_sq, (_R, _R2)),
    "yaw_mat": (r.yaw_mat, (_R,)),
}


@pytest.mark.parametrize("name", GRADCHECK)
def test_gradcheck(name):
    fn, args = GRADCHECK[name]
    assert torch.autograd.gradcheck(fn, tuple(a.clone().requires_grad_() for a in args), eps=1e-6, atol=1e-5)
