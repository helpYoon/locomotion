"""CPU self-check of the step-symmetry term (train.py --sym_weight; docs/superpowers/specs/2026-11-22-step-symmetry-term-
design.md): the gait memory's swing detector, touchdown counting, start condition and penalty (src/mdp.py gait_*), the
velocity critic's memory features and the buffer's gait field, on synthetic walks.

    CUDA_VISIBLE_DEVICES= pytest tests/test_gait_sym.py
"""

import pytest
import torch

from robot.kinematics import Kinematics
from robot.spec import get_robot
from src.buffer import ReplayBuffer
from src.dr import XiLayout
from src.mdp import GAIT_DIM, GAIT_FEAT, MDP, VelocityCfg, gait_reset, state_dim, swing_step
from src.motion import CONTROL_DT, MotionLib
from utils.rotations import so3_exp

DT = CONTROL_DT
T_P, D_P, LAST, VALID = 1, 2, 3, 4  # memory columns (src/mdp.py GAIT_DIM's layout)


@pytest.fixture(scope="module")
def mdp():
    robot = get_robot("t2_measured")
    kin = Kinematics(robot, "cpu")
    q0 = torch.tensor(robot.default_joint_pos_list())
    T = 10
    stand = (torch.tensor(robot.robot_cfg().init_state.pos).repeat(T, 1), torch.tensor([1.0, 0, 0, 0]).repeat(T, 1), q0.repeat(T, 1))
    motion = MotionLib.from_clips(["stand"], *([x] for x in stand), kin, "cpu")
    m = MDP(kin, motion, q0, XiLayout(robot, "cpu").dim, robot.sensing)
    m.feet, m.contact_z = [robot.tracked_bodies.index(b) for b in robot.feet], robot.contact_z
    m.enable_velocity(robot, VelocityCfg(w_sym=-1.0))
    m.robot = robot
    return m


ON = torch.tensor([True])


def steps(times, leads, feet=None, swing=10, n0=20, tail=5):
    """A synthetic walk of one row, heading +x (R = I): step i by foot feet[i] (default i % 2), a swing of ``swing``
    control steps at 0.5 m/s ending at its touchdown, times[i] [s] after the previous touchdown (times[0]: unused), and
    at the touchdown the landing foot leads[i] ahead of the other. Returns foot_v (T, 1, 2, 3), foot_xy (T, 1, 2, 2),
    R (T, 1, 3, 3) and the touchdown indices."""
    feet = feet or [i % 2 for i in range(len(times))]
    taus = [n0 + swing]
    for t in times[1:]:
        taus.append(taus[-1] + round(t / DT))
    T = taus[-1] + tail
    v = torch.zeros(T, 1, 2, 3)
    xy = torch.zeros(T, 1, 2, 2)
    xy[:, 0, 0, 1], xy[:, 0, 1, 1] = 0.1, -0.1
    for tau, k, d in zip(taus, feet, leads):
        v[tau - swing : tau, 0, k, 0] = 0.5
        xy[tau:, 0, k, 0] = xy[tau, 0, 1 - k, 0] + d
    return v, xy, torch.eye(3).expand(T, 1, 3, 3).clone(), taus


def roll(mdp, v, xy, R, gate=ON, mem0=None, settled=None):
    settled = torch.ones(v.shape[:2], dtype=torch.bool) if settled is None else settled
    return mdp.gait_rollout(torch.zeros(v.shape[1], GAIT_DIM) if mem0 is None else mem0, v, xy, R, gate, settled)


def scored_values(pen, scored):
    return pen[scored]


def test_even_gait_scores_zero(mdp):
    pen, scored, mems = roll(mdp, *steps([0.6] * 8, [0.3] * 8)[:3])
    assert mems.shape[0] == pen.shape[0] + 1 and mems.shape[-1] == GAIT_DIM
    assert int(scored.sum()) == 8 - 2
    assert float(pen.abs().max()) < 1e-9  # float rounding of 0.02 s sums and of positions


def test_length_limp_value(mdp):
    pen, scored, _ = roll(mdp, *steps([0.6] * 8, [0.2, 0.1] * 4)[:3])
    assert int(scored.sum()) == 6
    x = scored_values(pen, scored)
    assert torch.allclose(x, torch.full_like(x, (0.1 / 0.3) ** 2), atol=1e-5)


def test_time_limp_value(mdp):
    pen, scored, _ = roll(mdp, *steps([1.2, 0.8] * 4, [0.3] * 8)[:3])
    assert int(scored.sum()) == 6
    x = scored_values(pen, scored)
    assert torch.allclose(x, torch.full_like(x, (0.4 / 2.0) ** 2), atol=1e-5)


def test_floor(mdp):
    pen, scored, _ = roll(mdp, *steps([0.6] * 8, [0.05, 0.03] * 4)[:3])
    x = scored_values(pen, scored)
    assert torch.allclose(x, torch.full_like(x, (0.02 / 0.2) ** 2), atol=1e-5)


def test_same_foot_twice_is_ignored(mdp):
    v, xy, R, taus = steps([0.6] * 6, [0.3, 0.25, 0.3, 0.3, 0.3, 0.3], feet=[0, 0, 1, 0, 1, 0])
    pen, scored, mems = roll(mdp, v, xy, R)
    h = taus[1]  # the second left touchdown: state h is mems[h + 1]
    assert not scored[h].any()
    assert torch.equal(mems[h + 1][:, T_P : VALID + 1], mems[h][:, T_P : VALID + 1])
    assert float(mems[h + 1][0, LAST]) == 1.0


def test_simultaneous_touchdowns_count_once(mdp):
    T = 30
    v = torch.zeros(T, 1, 2, 3)
    v[5:15, 0, :, 0] = 0.5  # both feet swing and land on step 15
    xy = torch.zeros(T, 1, 2, 2)
    pen, scored, mems = roll(mdp, v, xy, torch.eye(3).expand(T, 1, 3, 3).clone())
    assert float(mems[16][0, LAST]) == 1.0  # left, from a fresh memory
    assert float(mems[16][0, 0]) == 0.0  # one counted touchdown reset the timer
    assert not scored.any()


def test_long_step_not_scored(mdp):
    v, xy, R, taus = steps([0.6, 0.6, 0.6, 2.2, 0.6, 0.6], [0.3] * 6)
    pen, scored, _ = roll(mdp, v, xy, R)
    assert [bool(scored[t, 0]) for t in taus] == [False, False, True, False, False, True]


def test_resting_feet_never_score(mdp):
    T = 500
    xy = torch.randn(1, 1, 2, 2).expand(T, 1, 2, 2).clone()
    pen, scored, mems = roll(mdp, torch.zeros(T, 1, 2, 3), xy, torch.eye(3).expand(T, 1, 3, 3).clone())
    assert not scored.any() and float(pen.abs().max()) == 0.0
    assert torch.equal(mems[:, :, 1:], mems[:1, :, 1:].expand_as(mems[:, :, 1:]))


def ref_swings(sp, v_on=0.15, v_off=0.08, min_len=3):
    """Session scratch gait/gait.py swings, on the speeds themselves: the touchdown (swing end) indices."""
    out, s, on = [], 0, False
    for t, x in enumerate(sp):
        if not on and x > v_on:
            s, on = t, True
        elif on and x < v_off:
            if t - s >= min_len:
                out.append(t)
            on = False
    return out


def test_swings_match_reference(mdp):
    c = mdp.vel
    g = torch.Generator().manual_seed(0)
    levels = torch.tensor([0.0, 0.05, 0.1, 0.12, 0.2, 0.5])
    for _ in range(20):
        seg = []
        while len(seg) < 300:
            seg += [float(levels[torch.randint(6, (), generator=g)])] * int(torch.randint(1, 16, (), generator=g))
        sp = torch.tensor(seg[:300])
        on, n, tds = torch.zeros(1, dtype=torch.bool), torch.zeros(1), []
        for t in range(300):
            on, n, td = swing_step(on, n, sp[t : t + 1], c.swing_on, c.swing_off, c.swing_min)
            if bool(td):
                tds.append(t)
        assert tds == ref_swings(sp.tolist(), c.swing_on, c.swing_off, c.swing_min)


def test_reset_resample(mdp):
    v, xy, R, taus = steps([0.6] * 10, [0.3] * 10)
    cut = taus[5] - 2  # foot 1 is 8 steps into its swing to touchdown 5
    pen1, sc1, m1 = roll(mdp, v[:cut], xy[:cut], R[:cut])
    mem = gait_reset(m1[-1], torch.tensor([True]), full=False)
    assert float(mem[0, 0]) == 0 and float(mem[0, LAST]) == 0 and float(mem[0, VALID]) == 0
    assert torch.equal(mem[:, 5:], m1[-1][:, 5:])  # the swing state is kept
    pen2, sc2, m2 = roll(mdp, v[cut:], xy[cut:], R[cut:], mem0=mem)
    assert [bool(sc2[t - cut, 0]) for t in taus[5:8]] == [False, False, True]
    assert float(gait_reset(m1[-1], torch.tensor([True]), full=True).abs().sum()) == 0.0
    assert torch.equal(gait_reset(m1[-1], torch.tensor([False]), full=True), m1[-1])


def test_unsettled_touchdowns_not_counted(mdp):
    # a training reset's recovery shuffle (reset velocity noise) is not a step: touchdowns count from the episode's
    # sym_settle-th step on, and the first two counted after it are unscored, as after any reset
    v, xy, R, taus = steps([0.6] * 8, [0.3, 0.1] * 4)
    K = taus[3]  # touchdowns 0-2 come before the episode has settled
    pen, scored, mems = roll(mdp, v, xy, R, settled=(torch.arange(v.shape[0]) >= K).unsqueeze(-1))
    assert float(mems[K, 0, LAST]) == 0.0 and float(mems[K, 0, VALID]) == 0.0
    assert int(scored.sum()) == 8 - 3 - 2
    assert not scored[: taus[5]].any() and bool(scored[taus[5], 0])


def test_gate(mdp):
    on = [(0.4, 0, 0), (0.4, 0.1, 0), (0.4, 0, 0.3), (0.4, 0, -0.3)]
    off = [(0.4, 0.2, 0), (-0.4, 0, 0), (0.1, 0, 0), (0.4, 0, 0.6), (0, 0, 0)]
    c = torch.tensor(on + off, dtype=torch.float32)
    held = torch.zeros(len(c), dtype=torch.bool)
    assert mdp.gait_gate(c, held).tolist() == [True] * len(on) + [False] * len(off)
    assert not mdp.gait_gate(c, ~held).any()  # rows reading the clip's command


def test_mirrored_walk_scores_the_same(mdp):
    v, xy, R, _ = steps([1.2, 0.8] * 4, [0.2, 0.1] * 4)
    pen, scored, _ = roll(mdp, v, xy, R)
    flip = torch.tensor([1.0, -1.0])
    vm, xym = v.flip(2) * torch.tensor([1.0, -1.0, 1.0]), xy.flip(2) * flip
    pen_m, scored_m, _ = roll(mdp, vm, xym, R)
    assert torch.equal(scored, scored_m) and torch.allclose(pen, pen_m, atol=1e-6)


def test_fresh_memory_finite(mdp):
    g = torch.Generator().manual_seed(1)
    H, B = 40, 16
    v = (torch.rand(H, B, 2, 3, generator=g) * 0.6).requires_grad_(True)
    xy = torch.randn(H, B, 2, 2, generator=g).requires_grad_(True)
    R = so3_exp(torch.randn(H, B, 3, generator=g) * 0.3).requires_grad_(True)
    pen, scored, mems = mdp.gait_rollout(torch.zeros(B, GAIT_DIM), v, xy, R, torch.ones(B, dtype=torch.bool),
                                         torch.ones(H, B, dtype=torch.bool))  # fmt: skip
    assert torch.isfinite(pen).all() and torch.isfinite(mems).all()
    pen.sum().backward()
    for x in (v, xy, R):
        assert x.grad is None or torch.isfinite(x.grad).all()


def test_gradient_only_through_feet_positions(mdp):
    for leads, nonzero in (([0.2, 0.1] * 4, True), ([0.3] * 8, False)):
        v, xy, R, _ = steps([0.6] * 8, leads)
        v, xy, R = v.requires_grad_(True), xy.requires_grad_(True), R.requires_grad_(True)
        pen, _, _ = roll(mdp, v, xy, R)
        pen.sum().backward()
        assert (float(xy.grad.abs().sum()) > 1e-3) == nonzero  # even: float rounding only (~1e-6)
        for x in (v, R):
            assert x.grad is None or float(x.grad.abs().sum()) == 0.0


def test_rollout_equals_steps(mdp):
    v, xy, R, _ = steps([1.2, 0.8] * 4, [0.2, 0.1] * 4)
    pen, scored, mems = roll(mdp, v, xy, R)
    mem = torch.zeros(1, GAIT_DIM)
    for h in range(v.shape[0]):
        p, s, mem = mdp.gait_step(mem, v[h], xy[h], R[h], ON, ON)
        assert torch.equal(p, pen[h]) and torch.equal(s, scored[h]) and torch.equal(mem, mems[h + 1])


def test_critic_features(mdp):
    robot = mdp.robot
    J, B = robot.num_joints, 4
    S = (torch.randn(B, 3), so3_exp(torch.randn(B, 3) * 0.3), torch.randn(B, 3), torch.randn(B, 3),
         torch.randn(B, J), torch.randn(B, J))  # fmt: skip
    cmd, xi_n, a_prev = torch.randn(B, 3), torch.randn(B, XiLayout(robot, "cpu").dim), torch.randn(B, 2 * J)
    base = mdp.velocity_critic_features(S, cmd, xi_n, a_prev)
    mem = torch.zeros(B, GAIT_DIM)
    mem[:, 0], mem[:, T_P], mem[:, D_P], mem[:, LAST], mem[:, VALID] = 5.0, 0.6, 0.3, -1.0, 1.0
    gf = mdp.gait_features(mem, torch.tensor([True, False, True, False]))
    assert gf.shape == (B, GAIT_FEAT)
    assert gf[:, 0].tolist() == [2.0] * B and gf[:, 3].tolist() == [-1.0] * B and gf[:, 4].tolist() == [1.0, 0.0, 1.0, 0.0]
    full = mdp.velocity_critic_features(S, cmd, xi_n, a_prev, gait_f=gf)
    assert torch.equal(full[:, : base.shape[-1]], base) and full.shape[-1] == base.shape[-1] + GAIT_FEAT


def test_buffer_gait_field(mdp):
    J, xi_dim = mdp.robot.num_joints, XiLayout(mdp.robot, "cpu").dim
    T, N, n_vel = 8, 4, 4
    assert "gait" not in ReplayBuffer(T, N, "cpu", J, xi_dim, a_dim=2 * J, vel=True).data
    buf = ReplayBuffer(T, N, "cpu", J, xi_dim, a_dim=2 * J, vel=True, gait=True)
    env_id = torch.arange(N, dtype=torch.float32)
    for t in range(T):
        buf.add({"s": torch.zeros(N, state_dim(J)), "s1": torch.zeros(N, state_dim(J)), "a": torch.zeros(N, J),
                 "a_prev": torch.zeros(N, 2 * J), "xi": env_id.unsqueeze(-1).expand(-1, xi_dim), "r": torch.zeros(N),
                 "f": torch.zeros(N, dtype=torch.long), "ep_step": torch.full((N,), t), "done": torch.zeros(N, dtype=torch.bool),
                 "cut": torch.zeros(N, dtype=torch.bool), "cmd": torch.zeros(N, 3), "cmd_ref": torch.zeros(N, dtype=torch.bool),
                 "gait": (env_id * 10 + t).unsqueeze(-1).expand(-1, GAIT_DIM)})  # fmt: skip
    s, _ = buf.sample_vel_starts(16, 2, n_vel)
    assert s["gait"].shape == (16, GAIT_DIM)
    assert torch.equal(s["gait"][:, 0], s["xi"][:, 0] * 10 + s["ep_step"].float())
