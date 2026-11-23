"""CPU self-check of the velocity path's mirror (src/mirror.py Mirror.obs / cmd_n, policy_mirror_error): the mirrored
observation is the observation of the mirrored robot, the mirrors undo themselves, the mirror error is the same number
for a situation and its mirror image, and it trains the velocity encoder alone.

    CUDA_VISIBLE_DEVICES= pytest tests/test_vel_mirror.py
"""

import pytest
import torch

from robot.kinematics import Kinematics
from robot.spec import get_robot
from src.dr import XiLayout
from src.mdp import MDP
from src.mirror import Mirror, policy_mirror_error
from src.motion import MotionLib
from src.networks import Policy
from utils.rotations import so3_exp

ROBOTS = ["t1", "t2_measured"]
B, HIST = 16, 2


def setup(name):
    """The MDP of utils/velocity_rig.py (a one-clip standing library) and the robot's Mirror, on the CPU."""
    torch.manual_seed(0)
    robot = get_robot(name)
    kin = Kinematics(robot, "cpu")
    q0 = torch.tensor(robot.default_joint_pos_list())
    T = 10
    quat = torch.tensor([1.0, 0, 0, 0]).repeat(T, 1)
    stand = (torch.tensor(robot.robot_cfg().init_state.pos).repeat(T, 1), quat, q0.repeat(T, 1))
    motion = MotionLib.from_clips(["stand"], *([x] for x in stand), kin, "cpu")
    layout = XiLayout(robot, "cpu")
    mdp = MDP(kin, motion, q0, layout.dim, robot.sensing)
    mdp.feet, mdp.contact_z = [robot.tracked_bodies.index(b) for b in robot.feet], robot.contact_z
    mdp.enable_velocity(robot)
    return robot, mdp, Mirror(robot, layout.slices, "cpu"), layout


def random_state(robot, n):
    J = robot.num_joints
    q0 = torch.tensor(robot.default_joint_pos_list())
    p = torch.randn(n, 3) * 0.3 + torch.tensor([0.0, 0.0, 0.7])
    R = so3_exp(torch.randn(n, 3) * 0.3)
    return p, R, torch.randn(n, 3), torch.randn(n, 3), q0 + 0.2 * torch.randn(n, J), torch.randn(n, J)


@pytest.mark.parametrize("name", ROBOTS)
def test_obs_mirror_matches_mirrored_state(name):
    robot, mdp, M, layout = setup(name)
    J = robot.num_joints
    S = random_state(robot, B)
    a_prev = torch.randn(B, 2 * J)  # two action blocks, as the world model's inputs carry
    xi = torch.rand(B, layout.dim) * 0.2 - 0.1
    f = torch.zeros(B, dtype=torch.long)
    up = torch.ones(B, dtype=torch.bool)
    bias, tilt = torch.randn(B, 3) * 0.05, torch.randn(B, 2) * 0.05
    bias_m = torch.stack((-bias[:, 0], bias[:, 1], -bias[:, 2]), -1)  # a pseudo-vector: -M b
    tilt_m = torch.stack((-tilt[:, 0], tilt[:, 1]), -1)  # (roll, pitch) -> (-roll, pitch)
    for sensor, sensor_m in ((None, None), ((bias, tilt), (bias_m, tilt_m))):
        o = mdp.obs(S, a_prev, xi, f, up, sensor=sensor)
        o_m = mdp.obs(M.state(S), M.joints(a_prev), M.xi(xi), f, up, sensor=sensor_m)
        assert o.shape[-1] == 3 * J + 12
        assert torch.allclose(M.obs(o), o_m, atol=1e-5), (M.obs(o) - o_m).abs().max(0).values


@pytest.mark.parametrize("name", ROBOTS)
def test_mirrors_are_involutions(name):
    robot, mdp, M, _ = setup(name)
    o = torch.randn(B, HIST, mdp.obs_dim)
    assert torch.equal(M.obs(M.obs(o)), o)
    assert not torch.equal(M.obs(o), o)
    c = torch.randn(B, 3)
    assert torch.equal(M.cmd_n(M.cmd_n(c)), c)
    assert torch.equal(M.cmd_n(torch.tensor([0.4, 0.3, -1.0])), torch.tensor([0.4, -0.3, 1.0]))


def small_policy(robot, mdp):
    pi = Policy(mdp.obs_dim, 8, robot.num_joints, hist=HIST, enc_hidden=(32,), dec_hidden=(32,), vel=True)
    torch.nn.init.normal_(pi.dec[-1].weight, std=0.5)  # the constructor scales the last layer by 0.01
    return pi


@pytest.mark.parametrize("name", ROBOTS)
def test_mirror_error_is_mirror_invariant(name):
    robot, mdp, M, _ = setup(name)
    pi = small_policy(robot, mdp)
    o, c = torch.randn(B, HIST, mdp.obs_dim), torch.rand(B, 3) * 2 - 1
    e = policy_mirror_error(pi, M, c, o)
    assert e.shape == (B,) and (e >= 0).all() and (e > 0).any()
    assert torch.allclose(e, policy_mirror_error(pi, M, M.cmd_n(c), M.obs(o)), atol=1e-6)


@pytest.mark.parametrize("name", ROBOTS)
def test_mirror_error_trains_only_the_velocity_encoder(name):
    robot, mdp, M, _ = setup(name)
    pi = small_policy(robot, mdp)
    policy_mirror_error(pi, M, torch.rand(B, 3) * 2 - 1, torch.randn(B, HIST, mdp.obs_dim)).mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in pi.enc_vel.parameters())
    assert all(p.grad is None for p in (*pi.dec.parameters(), *pi.enc.parameters()))


@pytest.mark.parametrize("name", ROBOTS)
def test_symmetric_command_box_check(name):
    from src.mirror import assert_mirrorable_box

    _, mdp, _, _ = setup(name)
    assert_mirrorable_box(mdp.cmd_lo, mdp.cmd_hi)
    with pytest.raises(AssertionError):
        assert_mirrorable_box(torch.tensor([-1.0, -0.8, -2.8]), torch.tensor([1.4, 0.7, 2.8]))
