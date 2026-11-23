"""CPU self-check of the velocity-command interface's parts (train.py --vel_frac): the velocity rows' observation, the
velocity encoder and the frozen decoder, the reference command, the velocity reward and fall termination, and the
imagination starts.

    pytest tests/test_velocity.py

It does not run the simulator or a training step; those need a GPU run of train.py --vel_frac.
"""

import dataclasses
import math

import torch

from src.buffer import ReplayBuffer
from src.dr import XiLayout
from robot.kinematics import Kinematics
from src.mdp import MDP, VelocityCfg, state_dim, upright_rel
from src.motion import CONTROL_DT, MotionLib
from src.networks import Policy
from robot.spec import get_robot
from utils.rotations import mat_to_6d, so3_exp


def test_velocity_interface():
    torch.manual_seed(0)
    dev = "cpu"
    robot = get_robot("t1")
    J = robot.num_joints
    kin = Kinematics(robot, dev)
    q0 = torch.tensor(robot.default_joint_pos_list())

    def clip(T: int, speed: float, yaw_rate: float):
        """A standing pose gliding at ``speed`` along its heading while turning at ``yaw_rate``."""
        t = torch.arange(T) * CONTROL_DT
        yaw = yaw_rate * t
        vel = speed * torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros(T)), -1)
        p = torch.cumsum(vel * CONTROL_DT, 0) + torch.tensor([0.0, 0.0, 0.688])
        quat = torch.stack((torch.cos(yaw / 2), torch.zeros(T), torch.zeros(T), torch.sin(yaw / 2)), -1)
        return p, quat, q0.repeat(T, 1)

    clips = [clip(100, 1.0, 0.0), clip(100, 0.6, 0.5)]
    motion = MotionLib.from_clips(["straight", "circle"], *zip(*clips), kin, dev)
    xi_dim = XiLayout(robot, dev).dim
    mdp = MDP(kin, motion, q0, xi_dim, robot.sensing)
    mdp.feet, mdp.contact_z = [robot.tracked_bodies.index(b) for b in robot.feet], robot.contact_z
    mdp.enable_velocity(robot)
    cfg: VelocityCfg = mdp.vel

    # ---- reference command: the clips' own velocities
    motion.set_ref_cmd()
    assert torch.allclose(motion.ref_cmd[50], torch.tensor([1.0, 0.0, 0.0]), atol=1e-3), motion.ref_cmd[50]
    # 5e-3: the test clip's Euler integration leaves 0.6 sin(0.5 dt / 2) = 0.003 m/s sideways
    assert torch.allclose(motion.ref_cmd[150], torch.tensor([0.6, 0.0, 0.5]), atol=5e-3), motion.ref_cmd[150]

    # ---- commands
    assert torch.allclose(mdp.normalize_cmd(mdp.cmd_lo), -torch.ones(3)) and torch.allclose(mdp.normalize_cmd(mdp.cmd_hi), torch.ones(3))
    c = mdp.sample_commands(20000)
    zero = (c == 0).all(-1).float().mean()
    assert ((c >= mdp.cmd_lo) | (c == 0)).all() and ((c <= mdp.cmd_hi) | (c == 0)).all() and abs(zero - cfg.zero_cmd_prob) < 0.01, zero

    # ---- velocity reward: the ceiling, then one term at a time
    f = torch.tensor([50])
    S = motion.ref_state(f)
    xi, a = torch.zeros(1, xi_dim), torch.zeros(1, J)  # a = 0 holds the default pose: no torque
    a_prev = torch.zeros(1, 2 * J)
    high, still, on = torch.ones(1, 2), torch.zeros(1, 2, 3), torch.ones(1, dtype=torch.bool)
    k = cfg.scale * CONTROL_DT
    reward = lambda cmd, a=a, z=high, v=still, v0=still, v1=still, on=on: mdp.velocity_reward(S, cmd, a, a_prev, xi, z, v, v0, v1, on)  # noqa: E731
    cmd = torch.tensor([[1.0, 0.0, 0.0]])
    top = reward(cmd)
    assert abs(top - cfg.max_step_reward) < 1e-5 and abs(cfg.max_step_reward - 0.1) < 1e-9, top
    assert abs(reward(cmd + torch.tensor([0.5, 0.0, 0.0])) - (top - k * 5 * (1 - math.exp(-1)))) < 1e-5  # lin vel kernel
    assert abs(reward(cmd + torch.tensor([0.0, 0.0, 0.5])) - (top - k * 5 * (1 - math.exp(-1)))) < 1e-5  # yaw rate kernel
    # --track_sharp (off by default): that share of each tracking term is paid by a kernel four times narrower
    assert cfg.track_sharp == 0
    cfg.track_sharp = 0.75
    assert abs(reward(cmd) - top) < 1e-6  # the same ceiling
    for off_cmd in (torch.tensor([0.5, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.5])):  # 0.5 off: the wide kernel pays e^-1, the narrow e^-4
        assert abs(reward(cmd + off_cmd) - (top - k * 5 * (1 - 0.25 * math.exp(-1) - 0.75 * math.exp(-4)))) < 1e-5
    cfg.track_sharp = 0.0
    moving = torch.tensor([[[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    assert abs(reward(cmd, v=moving) - (top - k * 0.001 * 0.1 / CONTROL_DT**2)) < 1e-6  # jerk of one foot
    assert abs(reward(cmd, v=moving, on=~on) - top) < 1e-6  # no jerk on an episode's first two steps
    down = torch.full((1, 2), robot.contact_z - 0.05)
    assert abs(reward(cmd, z=down, v=moving, v0=moving, v1=moving) - (top - k * 0.25 * 0.1)) < 1e-6  # slide of the foot in contact
    push = torch.zeros(1, J)
    push[0, 20] = 0.01
    _, terms = mdp.velocity_reward(S, cmd, push, a_prev, xi, high, still, still, still, on, terms=True)
    assert abs(terms["torque"] - mdp.kp[20] * mdp.alpha[20] * 0.01) < 1e-4 and abs(terms["rate"] - 0.01) < 1e-5, terms

    # ---- standing command: zero, or below both thresholds (a clip that stands still has a small command, not zero)
    slow = torch.tensor([[0.05, -0.05, 0.1]])
    cmds = torch.cat((torch.zeros(1, 3), slow, torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.0, 0.3]])))
    assert mdp.standing(cmds).tolist() == [True, True, False, False]

    # ---- stance term (off by default): at a standing command, the whole pose's distance from the default pose
    stop = torch.zeros(1, 3)
    bent = S[4].clone()
    bent[0, [3, 10, 16, 19, 20, 27]] += 0.1  # two shoulders, the waist, a hip yaw, a knee, an ankle 0.1 rad off: 0.06 rad^2
    S_bent = (*S[:4], bent, S[5])
    vr = lambda S_, cmd_: mdp.velocity_reward(S_, cmd_, a, a_prev, xi, high, still, still, still, on)  # noqa: E731
    off = [vr(S, stop), vr(S_bent, stop), vr(S_bent, cmd), vr(S_bent, slow)]
    assert cfg.w_stance == 0
    cfg.w_stance = -4.0
    assert abs(vr(S, stop) - off[0]) < 1e-7  # the default pose: no penalty
    assert abs(vr(S_bent, stop) - (off[1] - k * 4 * (1 - math.exp(-0.06 / cfg.stance_scale)))) < 1e-6
    assert abs(vr(S_bent, cmd) - off[2]) < 1e-7  # a walking command: no penalty
    assert abs(vr(S_bent, slow) - (off[3] - k * 4 * (1 - math.exp(-0.06 / cfg.stance_scale)))) < 1e-6  # a standing clip's command
    far = S[4] + 1.0
    assert vr((*S[:4], far, S[5]), stop) > vr((*S[:4], far, S[5]), cmd + 9) - k * 4 - 1e-6  # bounded by |w_stance|
    cfg.w_stance = 0.0

    # ---- ankle term (off by default): at a standing command, the ankle pitch motors' torque
    ia = [list(robot.joint_names).index(f"{s}_Ankle_Pitch") for s in ("Left", "Right")]
    load = torch.zeros(1, J)
    load[0, ia] = 0.5  # a setpoint 0.5 action units off the pose on both ankles: kp x alpha x 0.5 N m each
    tau2 = float(((mdp.kp[ia] * mdp.alpha[ia] * 0.5) ** 2).sum())
    ar = lambda cmd_: mdp.velocity_reward(S, cmd_, load, a_prev, xi, high, still, still, still, on)  # noqa: E731
    off = [ar(stop), ar(slow), ar(cmd)]
    assert cfg.w_ankle == 0
    cfg.w_ankle = -4.0
    pen = k * 4 * (1 - math.exp(-tau2 / cfg.ankle_scale))
    assert pen > k * 0.4  # the test load is on the kernel's slope, not at its foot
    assert abs(ar(stop) - (off[0] - pen)) < 1e-6 and abs(ar(slow) - (off[1] - pen)) < 1e-6
    assert abs(ar(cmd) - off[2]) < 1e-7  # a walking command: no penalty
    # its zero point (--ankle_target) and roll share (--ankle_roll_weight), both off by default
    assert cfg.ankle_target == 0 and cfg.ankle_roll == 0
    tau_a = (mdp.kp[ia] * mdp.alpha[ia] * 0.5).tolist()
    cfg.ankle_target = tau_a[0]  # the load is what is asked for: no penalty
    assert abs(tau_a[0] - tau_a[1]) < 1e-6 and abs(ar(stop) - off[0]) < 1e-6
    cfg.ankle_target = tau_a[0] / 2  # half of it: a quarter of the squared torque
    assert abs(ar(stop) - (off[0] - k * 4 * (1 - math.exp(-tau2 / 4 / cfg.ankle_scale)))) < 1e-6
    cfg.ankle_target = 0.0
    ir = [list(robot.joint_names).index(f"{s}_Ankle_Roll") for s in ("Left", "Right")]
    load_r = load.clone()
    load_r[0, ir] = 0.5
    tau2_r = float(((mdp.kp[ir] * mdp.alpha[ir] * 0.5) ** 2).sum())
    rr = lambda cmd_: mdp.velocity_reward(S, cmd_, load_r, a_prev, xi, high, still, still, still, on)  # noqa: E731
    cfg.w_ankle = 0.0
    off_r = [rr(stop), rr(cmd)]
    cfg.w_ankle = -4.0
    assert abs(rr(stop) - (off_r[0] - pen)) < 1e-6  # roll share 0: the roll load is free, as before
    cfg.ankle_roll = 2.0
    assert abs(rr(stop) - (off_r[0] - k * 4 * (1 - math.exp(-(tau2 + 2.0 * tau2_r) / cfg.ankle_scale)))) < 1e-6
    assert abs(rr(cmd) - off_r[1]) < 1e-7  # a walking command: no penalty
    cfg.ankle_roll = 0.0
    cfg.w_ankle = 0.0

    # ---- rest-pose term (off by default): at a standing command, the named joints' distance from the robot's rest pose
    assert cfg.w_rest == 0 and cfg.rest_pose == ()
    pose = (("Left_Hip_Yaw", 0.1), ("Right_Hip_Yaw", -0.1), ("Waist", 0.0))
    mdp.enable_velocity(robot, dataclasses.replace(cfg, rest_pose=pose))
    cfg = mdp.vel
    jn = list(robot.joint_names)
    at = S[4].clone()
    at[0, [jn.index(n) for n, _ in pose]] = torch.tensor([v for _, v in pose])
    away = at.clone()
    away[0, jn.index("Left_Hip_Yaw")] += 0.2  # 0.04 rad^2 off the pose
    away[0, jn.index("Left_Knee_Pitch")] += 0.3  # not one of the pose's joints: free
    S_at, S_away = (*S[:4], at, S[5]), (*S[:4], away, S[5])
    off = [vr(S_at, stop), vr(S_away, stop), vr(S_away, cmd)]
    cfg.w_rest = -2.0
    assert abs(vr(S_at, stop) - off[0]) < 1e-7  # the rest pose: no penalty
    assert abs(vr(S_away, stop) - (off[1] - k * 2 * (1 - math.exp(-0.04 / cfg.rest_scale)))) < 1e-6
    assert abs(vr(S_away, cmd) - off[2]) < 1e-7  # a walking command: no penalty
    cfg.w_rest = 0.0

    # ---- Table A.6 terms (off by default), at every command: joint limits, sideways lean, roll and pitch rates
    assert cfg.w_limit == cfg.w_lean == cfg.w_tilt_rate == 0
    yawed = torch.tensor([[[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]]) @ S[1]  # the heading frame is not the world's
    rot = lambda ax, ang: so3_exp(ang * torch.eye(3)[ax])  # noqa: E731
    side, fwd = yawed @ rot(0, 0.3), yawed @ rot(1, 0.3)  # about the heading x the trunk tips sideways, about y forward
    spin = torch.zeros(1, 3)
    spin[0] = yawed[0] @ torch.tensor([0.3, 0.4, 2.0])  # roll 0.3, pitch 0.4, yaw 2.0 rad/s in the heading frame
    over = S[4].clone()
    over[0, 20] = mdp.soft_hi[20] + 0.1  # one joint 0.1 rad past its soft limit
    cases = {"lean": ((S[0], side, *S[2:]), math.sin(0.3) ** 2), "fwd": ((S[0], fwd, *S[2:]), 0.0),
             "tilt_rate": ((S[0], yawed, S[2], spin, *S[4:]), 0.25), "limit": ((*S[:4], over, S[5]), 0.1)}
    for name, (S_, want) in cases.items():
        _, t = mdp.velocity_reward(S_, cmd, a, a_prev, xi, high, still, still, still, on, terms=True)
        assert abs(t["lean" if name == "fwd" else name] - want) < 1e-5, (name, t)
    base = [vr(S_, cmd) for S_, _ in cases.values()]
    cfg.w_limit, cfg.w_lean, cfg.w_tilt_rate = -5.0, -10.0, -0.125
    for (name, (S_, want)), b in zip(cases.items(), base):
        w = {"lean": 10.0, "fwd": 10.0, "tilt_rate": 0.125, "limit": 5.0}[name]
        assert abs(vr(S_, cmd) - (b - k * w * want)) < 1e-6, name
    cfg.w_limit = cfg.w_lean = cfg.w_tilt_rate = 0.0

    # ---- step length (off by default): a touchdown pays w_step x ((lead - l) / l)^2, l = v T(v) / 2 of the cycle law
    assert cfg.w_step == 0 and "step" in mdp.velocity_reward(S, cmd, a, a_prev, xi, high, still, still, still, on, terms=True)[1]
    cfg.step_cycle = (1.12, 0.42)
    ell = lambda v: v * 1.12 * max(v, cfg.step_v_min) ** -0.42 / 2  # noqa: E731
    cz = robot.contact_z
    left_lands = torch.tensor([[cz - 0.05, cz + 0.05]])  # the left foot down now, the right one in the air
    up2 = torch.full((1, 2), cz + 0.05)  # both feet in the air one step earlier
    S_y = (S[0], yawed, *S[2:])  # heading forward = world +y: the lead is read in the heading frame

    def feet(lead):  # the left foot `lead` ahead of the right one along the heading, 0.2 m to its left
        return torch.tensor([[[-0.1, lead], [0.1, 0.0]]])

    def step_term(v, lead, z=left_lands, z0=up2, on_=on, c=None):
        c = torch.tensor([[v, 0.0, 0.0]]) if c is None else c
        t = mdp.velocity_reward(S_y, c, a, a_prev, xi, z, still, still, still, on_, terms=True, foot_xy=feet(lead), foot_z0=z0)[1]
        return float(t["step"])

    assert abs(ell(0.5) - 0.3746) < 1e-4 and abs(ell(0.15) - 0.15 * 1.12 * 0.2**-0.42 / 2) < 1e-9
    assert step_term(0.5, ell(0.5)) < 1e-3  # the law's step: free (the right foot's 5 cm leaves 4.5e-5 of a touchdown)
    assert abs(step_term(0.5, ell(0.5) / 2) - 0.25) < 1e-3  # half the step: ((l/2 - l) / l)^2
    assert abs(step_term(0.5, -ell(0.5)) - 4.0) < 2e-3  # landing behind the other foot
    assert abs(step_term(0.15, ell(0.15) / 2) - 0.25) < 1e-3  # under step_v_min the cycle time holds at T(step_v_min)
    down2 = torch.full((1, 2), cz - 0.05)
    assert step_term(0.5, 0.0, z=down2, z0=down2) < 1e-3  # no foot lands: no term, whatever the lead
    assert step_term(0.5, 0.0, on_=~on) < 1e-6  # an episode's first steps (stale previous heights)
    assert step_term(cfg.step_v_max + 0.1, 0.0) < 1e-6  # past the law's fitted speeds
    for c_ in ([0.0, 0.0, 0.0], [0.05, 0.0, 1.0], [0.0, 0.5, 0.0], [-0.5, 0.0, 0.0]):  # standing, turning, sideways, back
        assert step_term(0.0, 0.0, c=torch.tensor([c_])) < 1e-6, c_
    l5 = ell(0.5)
    lu = 0.8 * l5 / 2 + 0.6 * 0.2  # the lead along the command (0.8, 0.6): the 0.2 m side offset counts there
    assert abs(step_term(0.0, l5 / 2, c=torch.tensor([[0.4, 0.3, 0.0]])) - 0.64 * ((lu - l5) / l5) ** 2) < 1e-3  # share 0.64
    base = vr(S_y, torch.tensor([[0.5, 0.0, 0.0]]))
    cfg.w_step = -20.0
    got = float(mdp.velocity_reward(S_y, torch.tensor([[0.5, 0.0, 0.0]]), a, a_prev, xi, left_lands, still, still, still, on,
                                    foot_xy=feet(ell(0.5) / 2), foot_z0=up2))  # fmt: skip
    assert abs(got - (base - k * 20 * 0.25)) < 1e-4, (got, base)
    cfg.w_step, cfg.step_cycle = 0.0, (0.0, 0.0)

    # ---- fall termination
    p, R, *rest = S
    tilt = lambda deg: torch.tensor([[1.0, 0, 0], [0, math.cos(math.radians(deg)), -math.sin(math.radians(deg))],
                                     [0, math.sin(math.radians(deg)), math.cos(math.radians(deg))]]).unsqueeze(0)  # noqa: E731
    at = lambda z, R=R: (torch.tensor([[0.0, 0.0, z]]), R, *rest)  # noqa: E731
    assert not mdp.fall_termination(at(0.33)) and mdp.fall_termination(at(0.31))
    assert not mdp.fall_termination(at(0.688, tilt(60))) and mdp.fall_termination(at(0.688, tilt(80)))
    assert mdp.soft_fall_termination(at(0.688)) < 1e-3 and abs(mdp.soft_fall_termination(at(0.32)) - 0.5) < 1e-3

    # ---- observation of a velocity row: the reference entries show an upright reference at the robot's own heading
    yaw = torch.tensor([[[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]])
    up = torch.ones(1, dtype=torch.bool)
    turned = (p, yaw @ R, *rest)  # a quarter turn away from the clip's heading
    o_clip, o_up = mdp.obs(turned, a_prev, xi, f), mdp.obs(turned, a_prev, xi, f, up)
    eye6 = mat_to_6d(torch.eye(3)).unsqueeze(0)
    assert torch.equal(o_up[:, :-6], o_clip[:, :-6]) and torch.allclose(o_up[:, -6:], eye6, atol=1e-6)
    assert not torch.allclose(o_clip[:, -6:], eye6, atol=1e-2)  # the clip's reference: the heading difference shows
    assert torch.equal(mdp.obs(turned, a_prev, xi, f, ~up), o_clip)
    o_tilt = mdp.obs(at(0.688, yaw @ tilt(20)), a_prev, xi, f, up)
    assert torch.allclose(o_tilt[:, -6:], mat_to_6d(tilt(-20)), atol=1e-6)  # the robot's tilt alone

    # ---- the same six entries from the gravity direction alone, as the deploy export rebuilds them (no reference there)
    pitch = lambda rad: torch.tensor([[math.cos(rad), 0, math.sin(rad)], [0, 1.0, 0], [-math.sin(rad), 0, math.cos(rad)]]).unsqueeze(0)  # noqa: E731
    turn = lambda rad: torch.tensor([[math.cos(rad), -math.sin(rad), 0], [math.sin(rad), math.cos(rad), 0], [0, 0, 1.0]]).unsqueeze(0)  # noqa: E731
    for yaw_, roll_, pitch_ in ((0.0, 0.0, 0.0), (0.7, 20.0, 0.3), (-2.5, -35.0, -0.6), (3.0, 10.0, 0.9)):
        o_any = mdp.obs(at(0.688, turn(yaw_) @ tilt(roll_) @ pitch(pitch_)), a_prev, xi, f, up)
        assert torch.allclose(upright_rel(o_any[:, 2 * J + 3 : 2 * J + 6]), o_any[:, -6:], atol=1e-5), (yaw_, roll_, pitch_)
    assert mdp.obs_dim == 3 * J + 6 + 6  # joints, joint speeds, gyro, gravity, previous action; then the six entries

    # ---- policy: the motion path is the default one; the velocity path goes through the same decoder, weights frozen
    obs_dim, goal_dim, hist = mdp.obs_dim, motion.goal_dim(10), 10
    oh, g = torch.randn(4, hist, obs_dim), torch.randn(4, goal_dim)
    pi = Policy(obs_dim, goal_dim, J, hist=hist, vel=True)
    dec = lambda z: pi.dec(torch.cat((z, pi.obs_norm(oh).flatten(-2)), -1))  # noqa: E731
    assert torch.equal(pi(oh, g), dec(torch.tanh(pi.enc(pi.goal_norm(g)))))
    assert pi.enc_vel[0].in_features == 3 + obs_dim * hist
    cn = mdp.normalize_cmd(mdp.sample_commands(4))
    act_vel = lambda o: pi.decode_frozen(pi.encode_vel(cn, o), o)  # noqa: E731
    assert torch.allclose(act_vel(oh), dec(pi.encode_vel(cn, oh)), atol=1e-6)
    act_vel(oh).sum().backward()
    grads = lambda net: [w.grad for w in net.parameters()]  # noqa: E731
    assert all(x is not None and x.abs().sum() > 0 for x in grads(pi.enc_vel))  # its return trains its encoder only
    assert all(x is None for x in grads(pi.dec) + grads(pi.enc))

    # ---- imagination starts: motion rows from the motion envs, velocity rows from the velocity envs
    T, N, n_vel = 8, 10, 4
    buf = ReplayBuffer(T, N, dev, J, xi_dim, a_dim=2 * J, vel=True)
    env_id = torch.arange(N, dtype=torch.float32)
    for t in range(T):
        buf.add({"s": torch.zeros(N, state_dim(J)), "s1": torch.zeros(N, state_dim(J)), "a": torch.zeros(N, J),
                 "a_prev": torch.zeros(N, 2 * J), "xi": env_id.unsqueeze(-1).expand(-1, xi_dim), "r": torch.zeros(N),
                 "f": torch.zeros(N, dtype=torch.long), "ep_step": torch.full((N,), t), "done": torch.zeros(N, dtype=torch.bool),
                 "cut": torch.zeros(N, dtype=torch.bool), "cmd": env_id.unsqueeze(-1).expand(-1, 3), "cmd_ref": env_id % 2 == 0})  # fmt: skip
    buf.env_lo = n_vel
    for split in (None, (7, 0.5)):
        buf.start_split = split
        s, h = buf.sample_starts(2000, 3)
        assert s["xi"][:, 0].min() == n_vel and s["xi"][:, 0].max() == N - 1 and "cmd" not in s
    s, h = buf.sample_vel_starts(2000, 3, n_vel)
    assert s["xi"][:, 0].min() == 0 and s["xi"][:, 0].max() == n_vel - 1
    assert torch.equal(s["cmd"][:, 0], s["xi"][:, 0]) and torch.equal(s["cmd_ref"], s["xi"][:, 0] % 2 == 0) and s["ep_step"].max() == T - 1
    assert h["s"].shape == (2000, 3, state_dim(J))

    print("velocity interface checks passed")
