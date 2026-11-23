"""robot/kinematics.py against MuJoCo's own forward kinematics and body velocities."""

import mujoco
import numpy as np
import torch

from robot.kinematics import Kinematics
from robot.t1 import MJCF_PATH, SPEC
from utils.rotations import quat_to_mat


def test_matches_mujoco():
    m = mujoco.MjModel.from_xml_path(MJCF_PATH)
    d = mujoco.MjData(m)
    kin = Kinematics(SPEC, "cpu")
    rng = np.random.default_rng(0)
    qpos, qvel, want = [], [], []
    for _ in range(8):
        d.qpos[:3] = rng.normal(size=3) * 10
        d.qpos[3:7] = rng.normal(size=4)
        d.qpos[3:7] /= np.linalg.norm(d.qpos[3:7])
        d.qpos[7:] = rng.uniform(m.jnt_range[1:, 0], m.jnt_range[1:, 1])
        d.qvel[:] = rng.normal(size=m.nv)
        mujoco.mj_forward(m, d)
        vel = np.zeros((m.nbody - 1, 6))  # [angular, linear] at the body origin, world frame
        for b in range(1, m.nbody):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, vel[b - 1], 0)
        qpos.append(d.qpos.copy()), qvel.append(d.qvel.copy())
        want.append((d.xpos[1:].copy(), d.xmat[1:].reshape(-1, 3, 3).copy(), vel[:, 3:], vel[:, :3]))

    # one batched call: the consumers call forward on (B, ...) states
    qpos, qvel = torch.tensor(np.array(qpos), dtype=torch.float32), torch.tensor(np.array(qvel), dtype=torch.float32)
    R = quat_to_mat(qpos[:, 3:7])
    got = kin.forward(qpos[:, :3], R, qvel[:, :3], (R @ qvel[:, 3:6, None])[..., 0], qpos[:, 7:], qvel[:, 6:],
                      all_bodies=True)
    for g, w in zip(got, zip(*want)):
        np.testing.assert_allclose(g, np.stack(w), atol=2e-5)
