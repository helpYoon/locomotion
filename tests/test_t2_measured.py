"""robot/t2_measured.py: the T2 as a real one was measured (Booster's walking gain table, joint friction, waist and
ankle damping x 0.69) and the randomization built around it (docs/superpowers/specs/2026-11-20-t2-measured-retrain-
design.md). CPU only: run with CUDA_VISIBLE_DEVICES= ."""

import numpy as np
import pytest
import torch

torch.set_num_threads(4)

from robot import t2_measured  # noqa: E402
from robot.spec import check_robot_numbers, get_robot, robot_numbers  # noqa: E402
from robot.t2 import SPEC as T2  # noqa: E402
from robot.t2_measured import SPEC  # noqa: E402

# booster_deploy 7bb1462 tasks/locomotion/robots/t2/__init__.py (the walking task), copied; head and wrists: Booster's
# default table (booster_deploy/robots/t2.py), as the third party's wrists
KP = [18.214394] * 2 + [67.1133, 54.0854, 54.0854, 54.0854, 18.214394, 18.214394, 18.214394] * 2 \
    + [155.9961, 206.9129, 110.9343] + [110.9343] * 4 + [141.7042, 52.6468] + [110.9343] * 4 + [141.7042, 52.6468]  # fmt: skip
KD_SENT = [1.449455] * 2 + [2.1362, 1.7215, 1.7215, 1.7215, 1.449455, 1.449455, 1.449455] * 2 \
    + [1.7215, 1.7215, 3.5311] + [3.5311] * 4 + [1.7215, 1.7215] + [3.5311] * 4 + [1.7215, 1.7215]  # fmt: skip
LINKAGE = (16, 17, 23, 24, 29, 30)  # waist pitch / roll and the ankles
KD_SIM_LINKAGE = [3.426018, 4.544264, 3.112137, 1.156241, 3.112137, 1.156241]  # sent x linkage ratio x 0.69
# BarryLi159159/T2_supermimic f737ef1 src/supermimic/robots/t2.py JOINT_FRICTION_V4 [N m]; head: the wrists' value
FRICTION = [0.3, 0.3] + [0.6, 0.63, 0.47, 0.4, 0.3, 0.3, 0.3] * 2 + [2.6, 3.3, 0.52] \
    + [0.95, 0.65, 0.6, 0.72, 2.4, 1.4] * 2  # fmt: skip


def test_gains():
    kp, kd, lim = SPEC.pd_gains()
    np.testing.assert_allclose(kp, KP, rtol=1e-9)
    serial = [i for i in range(31) if i not in LINKAGE]
    np.testing.assert_allclose(np.array(kd)[serial], np.array(KD_SENT)[serial], rtol=1e-9)
    np.testing.assert_allclose(np.array(kd)[list(LINKAGE)], KD_SIM_LINKAGE, atol=1e-6)
    np.testing.assert_allclose(SPEC.sent_gains[0], KP, rtol=1e-9)
    np.testing.assert_allclose(SPEC.sent_gains[1], KD_SENT, rtol=1e-9)
    assert lim == T2.pd_gains()[2]  # effort limits unchanged
    np.testing.assert_allclose(SPEC.action_scale_list(), 0.25 * np.array(lim) / np.array(kp), rtol=1e-12)


def test_friction_in_the_compiled_model():
    m = SPEC.robot_cfg().spec_fn().compile()
    got = [m.dof_frictionloss[m.jnt_dofadr[m.joint(n).id]] for n in SPEC.joint_names]
    np.testing.assert_allclose(got, FRICTION, rtol=1e-9)


def test_everything_else_is_the_t2():
    for f in ("mjcf_path", "joint_names", "tracked_bodies", "default_joint_pos", "delay_groups", "delay_range",
              "eval_delay", "mirror_perm", "mirror_sign", "velocity", "feet", "contact_z", "com_body", "com_range"):  # fmt: skip
        assert getattr(SPEC, f) == getattr(T2, f), f
    assert T2.sent_gains is None and get_robot("t2_measured") is SPEC
    assert not T2.robot_cfg().spec_fn().compile().dof_frictionloss.any()  # the t2 itself has no joint friction


def test_robot_numbers_guard():
    n = robot_numbers(SPEC)
    check_robot_numbers(SPEC, n)
    check_robot_numbers(SPEC, None)  # no record (an older checkpoint): fine
    with pytest.raises(ValueError, match="kp"):
        check_robot_numbers(T2, n)  # another robot's numbers
    bad = dict(n, friction=[0.0] + n["friction"][1:])
    with pytest.raises(ValueError, match="friction"):
        check_robot_numbers(SPEC, bad)


def test_a_joint_without_a_friction_value_is_an_error():
    with pytest.raises(KeyError):
        t2_measured.friction_of("left_toe_joint")


# ---------------------------------------------------------------- randomization config, xi layout, mirror
from src.dr import XiLayout  # noqa: E402
from src.mirror import Mirror  # noqa: E402


def test_xi_layout():
    for name, dim in (("t1", 38), ("t2", 40)):  # today's sizes: unchanged
        L = XiLayout(get_robot(name), "cpu")
        assert L.dim == dim and list(L.slices) == ["friction", "joint_offset", "com", "delay"]
    L = XiLayout(SPEC, "cpu")
    assert L.dim == 134
    assert list(L.slices) == ["friction", "joint_offset", "com", "delay", "joint_friction", "kp", "kd", "trunk_mass"]
    assert (L.slices["joint_offset"].start, L.slices["joint_offset"].stop) == (1, 32)
    assert [L.slices[k].stop - L.slices[k].start for k in ("joint_friction", "kp", "kd", "trunk_mass")] == [31, 31, 31, 1]
    c, s = L.center, L.scale
    assert c[L.slices["friction"]].item() == pytest.approx(0.95) and s[L.slices["friction"]].item() == pytest.approx(0.65)
    assert torch.all(c[L.slices["joint_friction"]] == 1) and torch.all(s[L.slices["joint_friction"]] == 0.5)
    assert torch.all(c[L.slices["kp"]] == 1) and torch.allclose(s[L.slices["kd"]], torch.tensor(0.1))
    assert c[L.slices["trunk_mass"]].item() == 0 and s[L.slices["trunk_mass"]].item() == 5


def test_mirror_of_the_new_groups():
    L = XiLayout(SPEC, "cpu")
    M = Mirror(SPEC, L.slices, "cpu")
    xi = torch.rand(3, L.dim) + 0.5
    m = M.xi(xi)
    torch.testing.assert_close(M.xi(m), xi)  # twice = identity
    li, ri = SPEC.joint_names.index("left_knee_pitch_joint"), SPEC.joint_names.index("right_knee_pitch_joint")
    for g in ("joint_friction", "kp", "kd"):
        a = L.slices[g].start
        assert torch.equal(m[:, a + ri], xi[:, a + li]) and torch.equal(m[:, a + li], xi[:, a + ri])  # swapped, no sign
    w = L.slices["joint_friction"].start + SPEC.joint_names.index("waist_roll_joint")
    assert torch.equal(m[:, w], xi[:, w])  # a centre joint's factor keeps its sign (its joint offset flips)
    t = L.slices["trunk_mass"].start
    assert torch.equal(m[:, t], xi[:, t])


# ---------------------------------------------------------------- the simulator applies friction, gains and masses
from src.dr import ramp_at  # noqa: E402

NOMINAL_KP, NOMINAL_KD = (torch.tensor(x) for x in SPEC.pd_gains()[:2])
FRICTION_TABLE = torch.tensor(FRICTION)


def make_env(robot_name: str, worlds: int, eval_mode: bool):
    """A TrackingSim on the CPU whose library is one clip of the robot standing in its default pose (as
    utils/velocity_rig.Rig builds its own), without a policy."""
    from robot.kinematics import Kinematics
    from src.env import TrackingSim, make_sim
    from src.mdp import MDP
    from src.motion import MotionLib

    assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES= (CPU simulator tests)"
    robot, dev, T = get_robot(robot_name), "cpu", 100
    kin = Kinematics(robot, dev)
    stand = (torch.tensor(robot.robot_cfg().init_state.pos).repeat(T, 1), torch.tensor([1.0, 0, 0, 0]).repeat(T, 1),
             torch.tensor(robot.default_joint_pos_list()).repeat(T, 1))  # fmt: skip
    motion = MotionLib.from_clips(["stand"], *([x] for x in stand), kin, dev)
    mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
    mdp.feet, mdp.contact_z = [robot.tracked_bodies.index(b) for b in robot.feet], robot.contact_z
    mdp.enable_velocity(robot)
    return TrackingSim(make_sim(robot), worlds, robot, motion, mdp, dev, eval_mode=eval_mode)


def gains(env) -> tuple[torch.Tensor, torch.Tensor]:
    """The simulator's kp and kd (worlds, J) in joint order, from the actuators."""
    kp, kd = torch.zeros(env.N, env.J), torch.zeros(env.N, env.J)
    for act in env.robot.actuators:
        kp[:, act.target_ids], kd[:, act.target_ids] = act.stiffness, act.damping
    return kp, kd


def per_world(env, field: str) -> torch.Tensor:
    x = getattr(env.sim.model, field)
    return x if x.shape[0] == env.N and x.dim() == 2 else x.reshape(-1, x.shape[-1])[:1].expand(env.N, -1)


def frictionloss(env) -> torch.Tensor:
    return per_world(env, "dof_frictionloss")[:, env.robot.indexing.joint_v_adr.reshape(-1)]


def body_id(env, body: str) -> int:
    return int(env.robot.indexing.body_ids[env.robot.body_names.index(body)])


def body_mass(env, body: str) -> torch.Tensor:
    return per_world(env, "body_mass")[:, body_id(env, body)]


def mass0(env, body: str) -> float:
    return float(env.sim.get_default_field("body_mass")[body_id(env, body)])


def all_worlds(env):
    return torch.arange(env.N), torch.zeros(env.N, dtype=torch.long)


def test_ramp_at():
    assert ramp_at(0, 0) == 1.0 and ramp_at(5, 0) == 1.0  # off: full, and inert
    assert ramp_at(0, 2000) == 0.0 and ramp_at(500, 2000) == 0.25
    assert ramp_at(2000, 2000) == 1.0 and ramp_at(5000, 2000) == 1.0  # a resume past the ramp


def test_eval_mode_is_the_measured_robot():  # an xi left at zero would multiply the reward's gains by 0
    env = make_env("t2_measured", 4, eval_mode=True)
    L = env.layout
    for g in ("joint_friction", "kp", "kd"):
        assert torch.all(env.xi[:, L.slices[g]] == 1.0), g
    assert torch.all(env.xi[:, L.slices["trunk_mass"]] == 0.0)
    kp, kd = gains(env)
    torch.testing.assert_close(kp, NOMINAL_KP.expand(4, -1))
    torch.testing.assert_close(kd, NOMINAL_KD.expand(4, -1))
    torch.testing.assert_close(frictionloss(env), FRICTION_TABLE.expand(4, -1))
    torch.testing.assert_close(body_mass(env, "trunk"), torch.full((4,), mass0(env, "trunk")))
    env.reset(*all_worlds(env))
    assert torch.all(env.xi[:, L.slices["kp"]] == 1.0)


def test_training_mode_draws_inside_the_ranges_and_xi_matches_the_simulator():
    env = make_env("t2_measured", 64, eval_mode=False)
    L = env.layout
    env.reset(*all_worlds(env))
    f, kpf, kdf = (env.xi[:, L.slices[g]] for g in ("joint_friction", "kp", "kd"))
    assert f.min() >= 0.5 and f.max() <= 1.5 and f.std() > 0.2
    assert kpf.min() >= 0.9 and kpf.max() <= 1.1 and kdf.min() >= 0.9 and kdf.max() <= 1.1 and kpf.std() > 0.04
    assert not torch.equal(kpf, kdf)  # a factor of its own each
    kp, kd = gains(env)
    torch.testing.assert_close(kp, NOMINAL_KP * kpf)
    torch.testing.assert_close(kd, NOMINAL_KD * kdf)
    torch.testing.assert_close(frictionloss(env), FRICTION_TABLE * f)
    add = env.xi[:, L.slices["trunk_mass"]].squeeze(-1)
    assert add.min() >= -5 and add.max() <= 5 and add.std() > 1.5
    torch.testing.assert_close(body_mass(env, "trunk"), mass0(env, "trunk") + add)
    r = body_mass(env, "left_knee_pitch_link") / mass0(env, "left_knee_pitch_link")
    assert r.min() >= 0.95 and r.max() <= 1.05 and r.std() > 0.01
    fr = env.xi[:, L.slices["friction"]]
    assert fr.min() >= 0.3 and fr.max() <= 1.6 and fr.max() > 1.2  # the wider ground-friction range


def test_ramp_zero_leaves_the_gains_nominal():
    env = make_env("t2_measured", 16, eval_mode=False)
    env.mdp.ramp.fill_(0.0)
    env.reset(*all_worlds(env))
    L = env.layout
    assert torch.all(env.xi[:, L.slices["kp"]] == 1.0) and torch.all(env.xi[:, L.slices["kd"]] == 1.0)
    assert env.xi[:, L.slices["joint_friction"]].std() > 0.2  # friction is not ramped
    env.mdp.ramp.fill_(0.5)
    env.reset(*all_worlds(env))
    kpf = env.xi[:, L.slices["kp"]]
    assert kpf.min() >= 0.95 and kpf.max() <= 1.05 and kpf.std() > 0.02


def test_gains_survive_both_reset_paths():  # mjlab's scene reset must not leave a world on defaults xi does not show
    env = make_env("t2_measured", 16, eval_mode=False)
    L = env.layout
    env.reset(torch.arange(8), torch.zeros(8, dtype=torch.long))  # the non-masked path, half the worlds
    kp, _ = gains(env)
    torch.testing.assert_close(kp, NOMINAL_KP * env.xi[:, L.slices["kp"]])
    mask = torch.arange(16) % 2 == 0
    before = env.xi.clone()
    env.reset_masked(mask)
    kp, kd = gains(env)
    torch.testing.assert_close(kp, NOMINAL_KP * env.xi[:, L.slices["kp"]])
    torch.testing.assert_close(kd, NOMINAL_KD * env.xi[:, L.slices["kd"]])
    torch.testing.assert_close(frictionloss(env), FRICTION_TABLE * env.xi[:, L.slices["joint_friction"]])
    assert torch.equal(env.xi[~mask], before[~mask])  # the others are untouched
    assert not torch.equal(env.xi[mask][:, L.slices["kp"]], before[mask][:, L.slices["kp"]])  # the reset ones redrew


def test_reward_torque_equals_the_simulator_torque():
    env = make_env("t2_measured", 16, eval_mode=False)
    env.reset(*all_worlds(env))
    torch.manual_seed(0)
    env.robot.write_joint_velocity_to_sim(0.5 * torch.randn(16, 31))
    a = 0.3 * torch.randn(16, 31)
    S = env.state()
    tau_est = env.mdp.pd_torque(S[4], S[5], a, env.xi)  # the expression velocity_reward uses
    env.robot.set_joint_position_target(env.default_q + env.xi[:, 1:32] + env.alpha * a)
    env.scene.write_data_to_sim()
    env.sim.step()
    tau_sim = env.sim.data.qfrc_actuator[:, env.robot.indexing.joint_v_adr.reshape(-1)]
    assert tau_est.abs().max() < 60  # away from every torque limit that matters here
    torch.testing.assert_close(tau_est, tau_sim, atol=1e-4, rtol=0)
    nominal = torch.clamp(env.mdp.kp * (env.default_q + env.xi[:, 1:32] + env.alpha * a - S[4]) - env.mdp.kd * S[5],
                          -env.mdp.tau_lim, env.mdp.tau_lim)  # fmt: skip
    assert (tau_est - nominal).abs().max() > 0.1  # the randomized gains matter


def test_the_t2_draws_nothing_new():
    env = make_env("t2", 8, eval_mode=False)
    assert env.layout.dim == 40 and not env.sim.mj_model.dof_frictionloss.any()
    env.reset(*all_worlds(env))
    kp, kd = gains(env)
    torch.testing.assert_close(kp, torch.tensor(T2.pd_gains()[0]).expand(8, -1))
    S = env.state()
    a = torch.zeros(8, 31)
    want = torch.clamp(env.mdp.kp * (env.default_q + env.xi[:, 1:32] - S[4]) - env.mdp.kd * S[5], -env.mdp.tau_lim, env.mdp.tau_lim)
    assert torch.equal(env.mdp.pd_torque(S[4], S[5], a, env.xi), want)  # the expression in the file before


# ---------------------------------------------------------------- sensor errors and ramped noise
from utils.rotations import so3_exp, yaw_mat  # noqa: E402

J = 31


def an_mdp_and_a_random_state(robot_name: str, n: int, tilt: float = 0.2):
    """(mdp, S, a_prev, xi, f): an MDP on the CPU without a simulator and n random states, the root tilted by up to
    ``tilt`` rad (0: upright at a random heading), xi at the layout's centre."""
    from robot.kinematics import Kinematics
    from src.mdp import MDP
    from src.motion import MotionLib

    robot, dev, T = get_robot(robot_name), "cpu", 100
    kin = Kinematics(robot, dev)
    q0 = torch.tensor(robot.default_joint_pos_list())
    stand = (torch.tensor(robot.robot_cfg().init_state.pos).repeat(T, 1), torch.tensor([1.0, 0, 0, 0]).repeat(T, 1), q0.repeat(T, 1))
    motion = MotionLib.from_clips(["stand"], *([x] for x in stand), kin, dev)
    L = XiLayout(robot, dev)
    mdp = MDP(kin, motion, q0, L.dim, robot.sensing)
    torch.manual_seed(0)
    Jn = robot.num_joints
    R = yaw_mat(so3_exp(torch.tensor([0.0, 0.0, 3.0]) * torch.rand(n, 1))) @ so3_exp(tilt * (2 * torch.rand(n, 3) - 1) * torch.tensor([1.0, 1.0, 0.0]))
    S = (torch.rand(n, 3), R, torch.randn(n, 3), torch.randn(n, 3), q0 + 0.1 * torch.randn(n, Jn), torch.randn(n, Jn))
    return mdp, S, torch.randn(n, 2 * Jn), L.center.repeat(n, 1), torch.zeros(n, dtype=torch.long)


UP = torch.ones(8, dtype=torch.bool)  # rows that follow no reference: the velocity rows


def test_sensor_none_and_zero_agree():
    mdp, S, a_prev, xi, f = an_mdp_and_a_random_state("t2_measured", 8)
    z = (torch.zeros(8, 3), torch.zeros(8, 2))
    torch.testing.assert_close(mdp.obs(S, a_prev, xi, f, UP, sensor=z), mdp.obs(S, a_prev, xi, f, UP), atol=1e-6, rtol=0)
    torch.testing.assert_close(mdp.obs(S, a_prev, xi, f, sensor=z), mdp.obs(S, a_prev, xi, f), atol=1e-6, rtol=0)


def test_gyro_bias_and_tilt():
    mdp, S, a_prev, xi, f = an_mdp_and_a_random_state("t2_measured", 8, tilt=0.0)
    b, t = torch.full((8, 3), 0.02), torch.tensor([[0.035, 0.0]]).expand(8, 2)
    o0, o1 = mdp.obs(S, a_prev, xi, f, UP), mdp.obs(S, a_prev, xi, f, UP, sensor=(b, t))
    torch.testing.assert_close(o1[:, 2 * J : 2 * J + 3] - o0[:, 2 * J : 2 * J + 3], b, atol=1e-6, rtol=0)  # gyro: + bias
    g0, g1 = o0[:, 2 * J + 3 : 2 * J + 6], o1[:, 2 * J + 3 : 2 * J + 6]
    torch.testing.assert_close(g0, torch.tensor([0.0, 0.0, -1.0]).expand(8, 3), atol=1e-6, rtol=0)
    ang = torch.acos((g0 * g1).sum(-1).clamp(-1, 1))
    torch.testing.assert_close(ang, torch.full((8,), 0.035), atol=1e-4, rtol=0)  # gravity turned by the tilt
    assert not torch.allclose(o1[:, -6:], o0[:, -6:], atol=1e-3)  # the orientation term sees the same IMU
    assert torch.equal(o1[:, : 2 * J], o0[:, : 2 * J]) and torch.equal(o1[:, 2 * J + 6 : -6], o0[:, 2 * J + 6 : -6])
    # a history: the row's offsets broadcast over its steps
    Sh = tuple(x.unsqueeze(1).repeat(1, 5, *[1] * (x.dim() - 1)) for x in S)
    oh = mdp.obs(Sh, a_prev.unsqueeze(1).repeat(1, 5, 1), xi.unsqueeze(1).repeat(1, 5, 1), f.unsqueeze(1).repeat(1, 5),
                 UP.unsqueeze(-1), sensor=(b.unsqueeze(1), t.unsqueeze(1)))  # fmt: skip
    torch.testing.assert_close(oh[:, 3], o1, atol=1e-6, rtol=0)


def test_noise_grows_with_the_ramp():
    mdp, *_ = an_mdp_and_a_random_state("t2_measured", 8)
    o = torch.zeros(20000, mdp.obs_dim)
    mdp.ramp.fill_(0.0)
    n0 = mdp.add_noise(o).abs().amax(0)
    mdp.ramp.fill_(1.0)
    n1 = mdp.add_noise(o).abs().amax(0)
    idx = [0, J, 2 * J, 2 * J + 3, mdp.obs_dim - 1]  # joint position, joint velocity, gyro, gravity, orientation
    torch.testing.assert_close(n0[idx], torch.tensor([0.01, 0.5, 0.2, 0.05, 0.05]), rtol=0.02, atol=0)
    torch.testing.assert_close(n1[idx], torch.tensor([0.02, 1.0, 0.4, 0.10, 0.10]), rtol=0.02, atol=0)
    assert n1[2 * J + 6 : 3 * J + 6].abs().max() == 0  # the previous action carries no noise


def test_draw_sensor_follows_the_ramp_and_is_absent_on_the_t2():
    mdp, *_ = an_mdp_and_a_random_state("t2_measured", 8)
    mdp.ramp.fill_(0.0)
    b, t = mdp.draw_sensor((4096,))
    assert b.abs().max() == 0 and t.abs().max() == 0
    mdp.ramp.fill_(1.0)
    b, t = mdp.draw_sensor((4096,))
    assert b.shape == (4096, 3) and t.shape == (4096, 2)
    assert 0.019 < b.abs().max() <= 0.02 and 0.033 < t.abs().max() <= 0.035
    assert abs(b.std().item() - 0.02 / 3**0.5) < 0.001  # uniform
    mdp2, *_ = an_mdp_and_a_random_state("t2", 8)
    torch.manual_seed(0)
    assert mdp2.draw_sensor((16,)) is None
    x = torch.rand(1)
    torch.manual_seed(0)
    assert torch.equal(torch.rand(1), x)  # no draw was made
    o = torch.zeros(4, mdp2.obs_dim)
    torch.manual_seed(1)
    a = mdp2.add_noise(o)
    mdp2.ramp.fill_(0.3)
    torch.manual_seed(1)
    assert torch.equal(mdp2.add_noise(o), a)  # the ramp does nothing there


def test_the_simulator_observation_carries_the_worlds_offsets():
    env = make_env("t2_measured", 16, eval_mode=False)
    env.reset(*all_worlds(env))
    b, t = env.randomizer.sensor()
    assert b.shape == (16, 3) and b.abs().max() <= 0.02 and b.std() > 0.005 and t.shape == (16, 2) and t.abs().max() <= 0.035
    S = env.state()
    clean = env.mdp.obs(S, env.a_prev, env.xi, env.f, None)
    env.mdp.obs_noise.zero_()  # the offsets alone
    o = env.observe()
    torch.testing.assert_close(o[:, 2 * J : 2 * J + 3] - clean[:, 2 * J : 2 * J + 3], b, atol=1e-6, rtol=0)
    keep = b.clone()
    env.reset_masked(torch.arange(16) < 8)
    b2, _ = env.randomizer.sensor()
    assert torch.equal(b2[8:], keep[8:]) and not torch.equal(b2[:8], keep[:8])  # redrawn per episode, for the reset worlds
    assert make_env("t2_measured", 4, eval_mode=True).randomizer.sensor() is None
    assert make_env("t2", 4, eval_mode=False).randomizer.sensor() is None


# ---------------------------------------------------------------- training script, guard wiring, export metadata
# train.py starts training on import: read from the sources (as tests/test_schedule.py)
import ast  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SRC = (REPO / "src" / "train.py").read_text()
S3 = REPO / "logs" / "t2_vel_stand_s3" / "policy" / "ckpt_013000.pt"


def add_argument(path: str, flag: str) -> dict:
    """The keywords of ``flag``'s add_argument call in a script, literals evaluated."""
    for node in ast.walk(ast.parse((REPO / path).read_text())):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == flag):  # fmt: skip
            return {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg in ("default", "choices", "type") and not isinstance(k.value, ast.Name)}
    raise KeyError(flag)


def test_train_flag_and_choices():
    assert add_argument("src/train.py", "--dr_ramp_iters")["default"] == 0
    for path in ("src/train.py", "src/evaluate.py", "utils/record_clips.py", "utils/render_gif.py", "utils/check_clips.py"):
        assert add_argument(path, "--robot")["choices"] == ["t1", "t2", "t2_measured"], path
    assert '--robot "${ROBOT:-t2}"' in (REPO / "utils" / "t2_run.sh").read_text()


def test_train_wires_the_ramp_and_the_imagined_sensor_errors():
    assert "mdp.ramp.fill_(ramp_at(it, args.dr_ramp_iters))" in SRC  # every iteration, into the device scalar
    assert "mdp.ramp.fill_(ramp_at(0, args.dr_ramp_iters))" in SRC  # before the simulator's first draws
    assert "args.robot_numbers = robot_numbers(robot)" in SRC
    assert SRC.index("args.robot_numbers = robot_numbers(robot)") < SRC.index('json.dump(vars(args)')  # saved with them
    body = {n.name: ast.get_source_segment(SRC, n) for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef)}
    for fn in ("_imag_step", "_imag_step_v"):
        assert "sensor=sens" in body[fn], fn
    assert "mdp.draw_sensor(" in body["policy_update"] and "sensor=" in body["policy_update"]
    assert body["policy_update"].count("sens)") >= 2  # passed to both imagined steps
    if "_land_imag_step" in body:  # the landing rollout (T1 only, where the tree has it) passes none
        assert "sensor=" not in body["_land_imag_step"]
    for col in ("dr_ramp", "kp_std", "gyro_std"):
        assert col in SRC


def test_rig_refuses_another_robots_numbers(tmp_path):
    if not S3.exists():
        pytest.skip("checkpoint missing")
    from utils.velocity_rig import Rig

    ck = torch.load(S3, map_location="cpu", weights_only=False)
    assert "robot_numbers" not in ck["args"]  # an older checkpoint: no record
    Rig(str(S3), 1, dev="cpu")  # loads as before
    ck["args"] = dict(ck["args"], robot_numbers=robot_numbers(T2))
    torch.save(ck, tmp_path / "same.pt")
    Rig(str(tmp_path / "same.pt"), 1, dev="cpu")  # its own numbers: fine
    ck["args"] = dict(ck["args"], robot_numbers=robot_numbers(SPEC))
    torch.save(ck, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="robot numbers differ"):
        Rig(str(tmp_path / "other.pt"), 1, dev="cpu")


def test_evaluate_and_export_check_the_numbers_and_export_writes_the_sent_gains():
    ev, ex = (REPO / "src" / "evaluate.py").read_text(), (REPO / "utils" / "export_onnx.py").read_text()
    assert "check_robot_numbers(robot, _run_args.get(\"robot_numbers\"))" in ev
    assert "kp_sent" in ex and "kd_sent" in ex and "robot.sent_gains" in ex


# ---------------------------------------------------------------- final review fixes
def test_the_simulator_is_compared_with_xi():
    """DomainRandomizer.sim_error: how far the simulator's gains and joint friction are from what xi says (the probe
    gates it on the GPU, where the writes happen inside a captured graph)."""
    env = make_env("t2_measured", 16, eval_mode=False)
    env.reset(*all_worlds(env))
    gain_err, fric_err = env.randomizer.sim_error(env.xi)
    assert gain_err < 1e-5 and fric_err < 1e-5
    env.robot.actuators[20].stiffness[3, 0] *= 1.5  # one world's hip gain no longer what xi says
    assert env.randomizer.sim_error(env.xi)[0] > 0.1
    env.reset(*all_worlds(env))
    env.sim.model.dof_frictionloss[5] *= 2.0
    assert env.randomizer.sim_error(env.xi)[1] > 0.1
    assert make_env("t2", 4, eval_mode=False).randomizer.sim_error(None) == (0.0, 0.0)  # nothing to compare


def test_train_logs_the_simulator_error_and_the_probe_gates_it():
    for col in ("gain_err", "fric_err"):
        assert col in SRC
    probe = REPO / "tacc" / "t2m_probe.sh"  # the cluster scripts are not tracked
    if probe.exists():
        text = probe.read_text()
        assert 'gate("sim_gain"' in text and 'gate("sim_friction"' in text and 'gate("eval"' in text
        assert "src.evaluate" in text


def test_record_clips_checks_the_numbers():
    assert "check_robot_numbers(" in (REPO / "utils" / "record_clips.py").read_text()


def test_check_velocity_refuses_the_t1():
    import subprocess
    import sys

    t1 = REPO / "logs" / "t1_vel_ankle_sharp_x2_s1" / "policy" / "ckpt_013000.pt"
    if not t1.exists():
        pytest.skip("checkpoint missing")
    r = subprocess.run([sys.executable, "-m", "utils.check_velocity", "--ckpt", str(t1), "--conditions", "nominal"],
                       cwd=REPO, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, timeout=300)  # fmt: skip
    assert r.returncode == 1 and "T2" in r.stderr and "Traceback" not in r.stderr


def test_export_can_run_on_the_cpu():
    """utils/export_onnx.py --device: the export and its self-check without the GPU (default: the GPU, as before)."""
    assert add_argument("utils/export_onnx.py", "--device")["default"] == "cuda:0"
    assert "Rig(args.ckpt, dev=args.device)" in (REPO / "utils" / "export_onnx.py").read_text()


def test_play_velocity_can_run_on_the_cpu():
    """utils/play_velocity.py --device: keyboard control without the GPU (default: the GPU, as before)."""
    assert add_argument("utils/play_velocity.py", "--device")["default"] == "cuda:0"
    assert "Rig(args.ckpt, args.worlds, dev=args.device)" in (REPO / "utils" / "play_velocity.py").read_text()
