"""Run an ONNX export of a velocity policy (utils/export_onnx.py's output) in plain MuJoCo, on the robot it was trained
for, with the keyboard (sim-to-sim: MuJoCo's own engine in place of the training simulator's MuJoCo-Warp). For the T2
robots (robot/t2.py, robot/t2_measured.py); the T1 has utils/play_mujoco.py.

usage (repo root; the physics and the policy run on the CPU, and CUDA_VISIBLE_DEVICES= keeps the libraries this
imports from opening the GPU at all):
    CUDA_VISIBLE_DEVICES= .venv/bin/python -m utils.play_native --onnx logs/t2m_vel_s2/policy/ckpt_013000.onnx
    CUDA_VISIBLE_DEVICES= .venv/bin/python -m utils.play_native --onnx logs/t2m_vel_s2/policy/ckpt_013000.onnx --script
        # --script: no viewer, a scripted drive with commanded against actual velocity
    ... --steptest [--trials 4] [--variant V]
        # the step test over the command box (672 commands, 5 s each, random per-group lags): falls and tracking error
    ... --stand [--start reset|handover] [--prev_action hold|zeros] [--variant V]
        # still within 3 s of the start and of each stop, no shaking; then the last setpoint frozen for 2 s
--variant (VARIANTS below) changes one thing from training: the gains, the sensors' age, pushes, joint friction, or how
the firmware applies the gains on the waist and ankle linkages.

Keys (as utils/play_velocity.py): up / down vx +-0.1 m/s, a / d vy +-0.1 m/s, left / right yaw rate +-0.2 rad/s (capped
at the policy's command box), space stops, backspace resets the robot. It starts standing and resets itself after a
fall.

The hardware settings are training's by construction, not copied:
- the model is the one the training simulator compiles (the same mjlab scene and solver options), so the measured T2
  has its joint friction, armature, masses and contact geometry;
- the gains, torque-speed curve, effort limits, default pose and action scale are read from the same robot spec
  (the waist and ankle joints carry the joint-side damping the real linkage gives the kd sent);
- every setpoint is applied --delay physics steps late (default: the robot's evaluation delay, 4 steps = 20 ms); the
  step test and the stand check draw one lag per actuator group, as training does.
The robot is the one whose gains, action scale and default pose equal those recorded in the ONNX file; a file that
matches none is refused. What differs from training: MuJoCo's engine, and this file's own observation and motor code
(what a robot-side runner has to get right). No randomization, noise or sensor errors, as an evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from collections import deque

import mujoco
import numpy as np

CONTROL_DT, DECIMATION = 0.02, 4  # src/motion.py, src/env.py: 50 Hz control over 4 physics steps of 5 ms
ROBOTS = ("t2_measured", "t2")
SCRIPT = [(2, (0, 0, 0)), (4, (0.5, 0, 0)), (4, (1.0, 0, 0)), (4, (-0.5, 0, 0)), (4, (0, 0.4, 0)), (4, (0, 0, 1.0)),
          (4, (0.6, 0, -1.0)), (3, (0, 0, 0))]  # utils/velocity_rig.py's  # fmt: skip
STEADY = 2.0  # --script: the velocity is averaged over each segment's last STEADY seconds
# What the robot might do differently from training (--variant; docs/t2_kpkd_risk_2026-11-20.md for the H cases):
# gains_x*: every kp and kd scaled; latency: the sensors one control step old; push: 0.5 m/s kicks while standing;
# no_friction: no joint friction; H1f1: the linkage damps at Booster's rule (factor 1) instead of the measured 0.69;
# H2: kp per motor too; H3: kd at the joint; H4: both (_swapped: the other motor order).
VARIANTS = ("nominal", "gains_x0.8", "gains_x1.2", "latency", "push", "no_friction", "H1f1", "H2", "H2_swapped", "H3", "H4",
            "H4_swapped")  # fmt: skip
HYPOTHESES = {"H1f1": ("H1", 0, 1.0), "H2": ("H2", 0, None), "H2_swapped": ("H2", 1, None), "H3": ("H3", 0, None),
              "H4": ("H4", 0, None), "H4_swapped": ("H4", 1, None)}  # variant: (case, motor order, damping factor or the robot's)  # fmt: skip
# motor = J @ joint for the two-motor linkages, at the neutral pose (Booster's T2_31dof_parallel.xml)
WAIST_J, ANKLE_J = [[-1.203, -1.382], [1.203, -1.382]], [[-1.143, 0.695], [1.146, 0.698]]
LINKAGE_PAIRS = (("waist_pitch_joint", "waist_roll_joint", WAIST_J), ("left_ankle_pitch_joint", "left_ankle_roll_joint", ANKLE_J),
                 ("right_ankle_pitch_joint", "right_ankle_roll_joint", [[-1.143, -0.695], [1.146, -0.698]]))  # fmt: skip
MOTOR_MAX = 60.0  # N m per linkage motor
# --stand: utils/check_velocity.py's stop cases: 3 s at the command, then zero
STOP_CASES = ((0, 0, 0), (0.8, 0, 0), (0.4, 0, 0), (-0.6, 0, 0), (-0.3, 0, 0), (0, 0.4, 0), (0, -0.4, 0), (0, 0, 1.0),
              (0, 0, -1.0), (0.5, 0, 0.5))  # fmt: skip
T_CMD, T_ALL, T_PREPARE, T_STAND, T_FROZEN = 3.0, 10.0, 2.0, 5.0, 2.0  # s
# Booster's T2 prepare gains (booster_deploy robots/t2.py, prepare_state): what holds the robot at the prepare pose,
# with no balance control, until the policy takes over
PREPARE_KP = [40.0] * 2 + [70.0, 70.0, 70.0, 70.0, 55.0, 70.0, 55.0] * 2 + [500.0, 500.0, 400.0] + [500.0, 350.0, 180.0, 500.0, 500.0, 500.0] * 2
PREPARE_KD = [0.65] * 2 + [1.5] * 14 + [3.0] * 15


def load_metadata(onnx_path: str) -> dict:
    import onnx

    return {p.key: json.loads(p.value) for p in onnx.load(onnx_path, load_external_data=False).metadata_props}


def robot_of(meta: dict, name: str | None = None):
    """The robot spec whose gains, action scale, default pose and joints are the file's (``name``: that one, or exit)."""
    from robot.spec import get_robot

    def differs(robot) -> str | None:
        kp, kd, _ = robot.pd_gains()
        for key, mine in (("joint_names", list(robot.joint_names)), ("kp", kp), ("kd", kd),
                          ("action_scale", robot.action_scale_list()), ("default_q", robot.default_joint_pos_list())):  # fmt: skip
            theirs = meta[key]
            if len(theirs) != len(mine) or (key == "joint_names" and theirs != mine):
                return key
            if key != "joint_names" and not np.allclose(theirs, mine, rtol=1e-6, atol=1e-9):
                return key
        return None

    for n in ROBOTS if name is None else (name,):
        robot = get_robot(n)
        bad = differs(robot)
        if bad is None:
            return robot
        if name is not None:
            sys.exit(f"the file's {bad} is not robot {name!r}'s: it was exported for another robot's numbers "
                     f"(drop --robot to pick the robot from the file)")  # fmt: skip
    sys.exit(f"the file's gains, action scale or default pose match none of {', '.join(ROBOTS)}")


def build_model(robot) -> mujoco.MjModel:
    """The MuJoCo model the training simulator compiles for ``robot``: its mjlab scene (the robot under "robot/", the
    ground plane, a motor per joint) with src/env.py's solver options."""
    from mjlab.scene import Scene

    from src.env import make_scene_cfg, make_sim

    m = Scene(make_scene_cfg(robot, 1), device="cpu").compile()
    make_sim(robot).mujoco.apply(m)
    return m


def motor_torque(q, qd, target, kp, kd, effort, v_max, denom) -> np.ndarray:
    """robot/t1.py's BoosterPdActuator: the PD torque toward ``target``, limited to the effort limit below the knee
    speed and falling linearly to zero at v_max (denom = v_max - v_knee)."""
    lim = np.clip(effort * (v_max - np.abs(qd)) / denom, 0.0, effort)
    return np.clip(kp * (target - q) - kd * qd, -lim, lim)


def pair_torque(case: str, order: int, e, qd, J, kp, kd_sent: float, factor: float):
    """Joint torques (pitch, roll) of a two-motor linkage pair under one way the firmware might apply the gains, and
    the largest motor torque asked. e, qd: the pair's setpoint error and speed; J: motor = J @ joint; kp: the pair's
    kp sent; kd_sent: the kd sent to both; factor: the share of Booster's motor-side damping the linkage gives.
    H1: kp at the joint, kd per motor. H2: kp per motor too. H3: kd at the joint. H4: kp per motor, kd at the joint.
    order 1 gives each motor the other joint's kp. Each motor is limited to MOTOR_MAX."""
    kpm = kp if order == 0 else kp[::-1]
    if case == "H1":
        tau = kp * e - J.T @ (factor * kd_sent * (J @ qd))
    elif case == "H2":
        tau = J.T @ (kpm * (J @ e) - factor * kd_sent * (J @ qd))
    elif case == "H3":
        tau = kp * e - kd_sent * qd
    elif case == "H4":
        tau = J.T @ (kpm * (J @ e)) - kd_sent * qd
    else:
        raise ValueError(f"unknown case {case!r}")
    motor = np.linalg.solve(J.T, tau)
    return J.T @ np.clip(motor, -MOTOR_MAX, MOTOR_MAX), float(np.abs(motor).max())


class NativeSim:
    """One robot in plain MuJoCo, stepped as the training simulator steps it: every physics step the motor law on the
    setpoint each actuator group wrote its lag of physics steps earlier (the oldest one until that many exist), written
    to the motors. ``delay``: None (the robot's evaluation delay), one lag for all, or one per robot.delay_groups."""

    def __init__(self, robot, delay=None, variant: str = "nominal"):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; one of {VARIANTS}")
        self.robot, self.variant, self.m = robot, variant, build_model(robot)
        self.d = mujoco.MjData(self.m)
        m, names = self.m, list(robot.joint_names)
        jid = [m.joint(f"robot/{n}").id for n in names]
        self.qadr, self.dadr = np.array([m.jnt_qposadr[j] for j in jid]), np.array([m.jnt_dofadr[j] for j in jid])
        self.jrange = m.jnt_range[jid].copy()  # (joints, 2)
        self.root_q, self.root_v = int(m.jnt_qposadr[m.joint("robot/world_joint").id]), int(m.jnt_dofadr[m.joint("robot/world_joint").id])
        act_joint = [int(m.actuator_trnid[i, 0]) for i in range(m.nu)]
        self.act = np.array([act_joint.index(j) for j in jid])  # each joint's motor
        kp, kd, effort = robot.pd_gains()
        self.kp, self.kd, self.effort = np.array(kp), np.array(kd), np.array(effort)
        if variant.startswith("gains_x"):
            self.kp, self.kd = self.kp * float(variant[7:]), self.kd * float(variant[7:])
        if variant == "no_friction":
            self.m.dof_frictionloss[:] = 0.0
        self.pairs = []  # (pitch joint, roll joint, J, kp sent, kd sent, case, order, factor) under a gain hypothesis
        if variant in HYPOTHESES:
            if robot.sent_gains is None:
                raise ValueError(f"{variant}: {robot.name} has no table of gains sent")
            from robot.t2_measured import DAMPING_FACTOR

            case, order, factor = HYPOTHESES[variant]
            for a, b, J in LINKAGE_PAIRS:
                ia, ib = names.index(a), names.index(b)
                self.pairs.append((ia, ib, np.array(J), np.array(robot.sent_gains[0])[[ia, ib]], robot.sent_gains[1][ia], case, order,
                                   DAMPING_FACTOR if factor is None else factor))  # fmt: skip
        v_max, v_knee = np.zeros(len(names)), np.zeros(len(names))
        import re

        for a in robot.robot_cfg().articulation.actuators:  # the torque-speed curve, per joint
            for i, n in enumerate(names):
                if any(re.fullmatch(p, n) for p in a.target_names_expr):
                    v_max[i], v_knee[i] = a.v_max, a.v_knee
        self.v_max, self.denom = v_max, np.maximum(v_max - np.minimum(v_knee, v_max), 1e-6)
        self.default_q, self.scale = np.array(robot.default_joint_pos_list()), np.array(robot.action_scale_list())
        self.init_pos = np.array(robot.robot_cfg().init_state.pos)
        self.set_lags(delay)
        from src.mdp import VelocityCfg

        cfg = dataclasses.replace(VelocityCfg(), **robot.velocity)
        self.fall_height, self.fall_tilt = cfg.fall_height, cfg.fall_tilt
        self.reset()

    def reset(self):
        """Standing in the default pose at the spec's start height, at rest (the training evaluation's start)."""
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.root_q : self.root_q + 3] = self.init_pos
        self.d.qpos[self.root_q + 3 : self.root_q + 7] = (1.0, 0.0, 0.0, 0.0)
        self.d.qpos[self.qadr] = self.default_q
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.setpoints = deque(maxlen=self.delay + 1)
        self.peak_motor = 0.0  # the largest linkage motor torque asked since the reset (gain hypotheses only)

    def set_lags(self, lags=None):
        """The setpoint lags in physics steps, from the next reset: None (the robot's evaluation delay), one int for
        every joint, or one per robot.delay_groups (as training draws them)."""
        groups = self.robot.delay_groups
        if lags is None or np.ndim(lags) == 0:
            lags = [self.robot.eval_delay if lags is None else int(lags)] * len(groups)
        if len(lags) != len(groups):
            raise ValueError(f"{len(lags)} lags for {len(groups)} delay groups")
        self.lag = np.zeros(len(self.robot.joint_names), dtype=int)
        for lag, joints in zip(lags, groups):
            self.lag[list(joints)] = int(lag)
        self.delay, self.one_lag = int(self.lag.max()), bool((self.lag == self.lag[0]).all())

    def delayed(self, target: np.ndarray) -> np.ndarray:
        """Record this physics step's setpoint; the one to apply: per joint, the one its group wrote its lag of steps
        ago, or the oldest so far."""
        self.setpoints.append(target)
        if self.one_lag:
            return self.setpoints[0]
        n = len(self.setpoints)
        return np.stack(self.setpoints)[n - 1 - np.minimum(self.lag, n - 1), np.arange(len(target))]

    def torque(self, applied: np.ndarray) -> np.ndarray:
        """The joint torques toward the setpoint ``applied`` at the current state: the training motor law, and under a
        gain hypothesis pair_torque on the three linkage pairs."""
        q, qd = self.d.qpos[self.qadr], self.d.qvel[self.dadr]
        tau = motor_torque(q, qd, applied, self.kp, self.kd, self.effort, self.v_max, self.denom)
        for ia, ib, J, kp, kd_sent, case, order, factor in self.pairs:
            tau[[ia, ib]], peak = pair_torque(case, order, applied[[ia, ib]] - q[[ia, ib]], qd[[ia, ib]], J, kp, kd_sent, factor)
            self.peak_motor = max(self.peak_motor, peak)
        return tau

    def step(self, action: np.ndarray):
        """One control step: the policy's action held for DECIMATION physics steps."""
        self.hold(self.default_q + self.scale * action)

    def hold(self, target: np.ndarray):
        """One control step toward the joint setpoint ``target``."""
        for _ in range(DECIMATION):
            self.d.ctrl[self.act] = self.torque(self.delayed(target))
            mujoco.mj_step(self.m, self.d)

    def push(self, rng: np.random.Generator):
        """A 0.5 m/s kick to the root in a random horizontal direction."""
        ang = rng.uniform(0.0, 2.0 * np.pi)
        self.d.qvel[self.root_v : self.root_v + 2] += 0.5 * np.array([np.cos(ang), np.sin(ang)])

    def rotation(self) -> np.ndarray:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.d.qpos[self.root_q + 3 : self.root_q + 7])
        return R.reshape(3, 3)

    def velocity(self) -> np.ndarray:
        """(vx, vy) of the root in its heading frame and the yaw rate about the world's z."""
        R = self.rotation()
        yaw = np.arctan2(R[1, 0], R[0, 0])
        v = self.d.qvel[self.root_v : self.root_v + 3]  # world frame
        w = R @ self.d.qvel[self.root_v + 3 : self.root_v + 6]  # the free joint's angular velocity is in the root frame
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], w[2]])

    def fallen(self) -> bool:
        return bool(self.d.qpos[self.root_q + 2] < self.fall_height or 1.0 - self.rotation()[2, 2] > self.fall_tilt)


def observation(sim: NativeSim, prev_action: np.ndarray) -> np.ndarray:
    """The policy's observation of the simulator's state, in the ONNX file's layout: joint positions minus the default
    pose, joint speeds, the root's angular velocity in its own frame, the gravity direction in the root frame
    (R^T (0, 0, -1)) and the previous action."""
    d = sim.d
    gyro = d.qvel[sim.root_v + 3 : sim.root_v + 6]
    return np.concatenate((d.qpos[sim.qadr] - sim.default_q, d.qvel[sim.dadr], gyro, -sim.rotation()[2], prev_action)).astype(np.float32)


class Policy:
    """The ONNX file with its observation history (the first observation repeated at a reset) and previous action.
    obs_latency 1: the sensor entries (all but the previous action) are the ones read at the previous call."""

    def __init__(self, onnx_path: str, meta: dict, obs_latency: int = 0):
        self.obs_latency = obs_latency
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self.hist_len = int(self.session.get_inputs()[0].shape[1])
        self.num_j = len(meta["joint_names"])
        self.reset()

    def reset(self):
        self.hist, self.prev, self.sensors = None, np.zeros(self.num_j, dtype=np.float32), None

    def __call__(self, sim: NativeSim, cmd) -> np.ndarray:
        o = observation(sim, self.prev)
        if self.obs_latency:
            n = 2 * self.num_j + 6
            old, self.sensors = (o[:n] if self.sensors is None else self.sensors), o[:n].copy()
            o = np.concatenate((old, o[n:]))
        self.hist = np.repeat(o[None], self.hist_len, 0) if self.hist is None else np.concatenate((self.hist[1:], o[None]))
        (a,) = self.session.run(None, {"obs_hist": self.hist[None], "cmd": np.asarray([cmd], dtype=np.float32)})
        self.prev = a[0].astype(np.float32)
        return self.prev.astype(np.float64)


def describe(sim: NativeSim) -> str:
    m, r = sim.m, sim.robot
    fl = m.dof_frictionloss[sim.dadr]
    return (f"robot {r.name}: joint friction {fl.min():g}-{fl.max():g} N m, kp {sim.kp.min():.4g}-{sim.kp.max():.4g}, "
            f"waist pitch / ankle roll kd {sim.kd[list(r.joint_names).index('waist_pitch_joint')]:.4g} / "
            f"{sim.kd[list(r.joint_names).index('left_ankle_roll_joint')]:.4g} (joint side), setpoints "
            f"{sim.lag.min() * 5}{'' if sim.one_lag else f'-{sim.delay * 5}'} ms late, variant {sim.variant}, "
            f"{m.opt.timestep * 1000:g} ms physics steps")  # fmt: skip


def run_script(onnx_path: str, robot_name: str | None = None, delay: int | None = None, quiet: bool = False,
               variant: str = "nominal"):
    """SCRIPT from standing; each segment's mean (vx, vy, yaw rate) over its last STEADY seconds, and the falls."""
    meta = load_metadata(onnx_path)
    sim = NativeSim(robot_of(meta, robot_name), delay, variant)
    pi = Policy(onnx_path, meta, obs_latency=int(variant == "latency"))
    out = (lambda *a, **k: None) if quiet else print
    out(describe(sim))
    out("  commanded (vx, vy, yaw)    actual, mean of the last 2 s    fell")
    rows, falls, t0 = [], 0, time.perf_counter()
    for dur, cmd in SCRIPT:
        seg, fell = [], False
        for _ in range(round(dur / CONTROL_DT)):
            sim.step(pi(sim, cmd))
            seg.append(sim.velocity())
            if sim.fallen():
                fell, falls = True, falls + 1
                sim.reset(), pi.reset()
        tail = np.mean(seg[-round(STEADY / CONTROL_DT) :], 0)
        rows.append(tuple(float(x) for x in tail))
        out(f"  {cmd[0]:+5.2f} {cmd[1]:+5.2f} {cmd[2]:+5.2f}          {tail[0]:+5.2f} {tail[1]:+5.2f} {tail[2]:+5.2f}"
            f"               {'yes' if fell else 'no'}")  # fmt: skip
    out(f"{falls} falls; {sum(d for d, _ in SCRIPT)} s simulated in {time.perf_counter() - t0:.1f} s")
    return rows, falls


def steptest_commands(low, high, n: int = 15) -> np.ndarray:
    """The step test's commands: per axis n values across the command box, the one nearest zero set to zero (the T2
    box's vx axis has no zero of its own, and without one there are no pure-axis commands), in the three planes
    vx-vy, vx-yaw, vy-yaw, the standing command left out (utils/play_mujoco.py's grid and order)."""
    axes = [np.linspace(low[i], high[i], n) for i in range(3)]
    for x in axes:
        x[np.abs(x).argmin()] = 0.0
    cmds = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = np.meshgrid(axes[i], axes[j], indexing="ij")
        c = np.zeros((n * n, 3))
        c[:, i], c[:, j] = a.ravel(), b.ravel()
        cmds += [x for x in c if np.linalg.norm(x) > 1e-6]
    return np.array(cmds)


def run_name(onnx_path: str) -> str:
    """logs/<run>/policy/<file>.onnx -> <run>; any other file -> its name without the extension."""
    parent, stem = os.path.split(os.path.abspath(onnx_path))
    return os.path.basename(os.path.dirname(parent)) if os.path.basename(parent) == "policy" else os.path.splitext(stem)[0]


def command_type(cmd) -> str:
    nz = [i for i in range(3) if abs(cmd[i]) > 1e-9]
    if len(nz) != 1:
        return "mixed"
    return ("forward" if cmd[0] > 0 else "backward") if nz[0] == 0 else ("sideways", "yaw")[nz[0] - 1]


def draw_lags(robot, rng: np.random.Generator) -> np.ndarray:
    """One lag per actuator group, uniform over the robot's delay range (as training draws them per episode)."""
    return rng.integers(robot.delay_range[0], robot.delay_range[1] + 1, len(robot.delay_groups))


def run_steptest(onnx_path: str, robot_name: str | None = None, variant: str = "nominal", trials: int = 4, seed: int = 0,
                 commands=None, out_dir: str = "logs/native", quiet: bool = False) -> dict:  # fmt: skip
    """The step test (utils/play_mujoco.py's, on this robot): per command, ``trials`` rollouts of 5 s from standing,
    each with its own draw of the per-group lags; a rollout's error is |mean over the last 2 s of (velocity - command)|
    / |command|, nan if it fell. Per command the median over the trials that did not fall; then the median over
    commands. Written to <out_dir>/steptest_<robot>_<run>_<variant>.json."""
    meta = load_metadata(onnx_path)
    robot = robot_of(meta, robot_name)
    sim, pi = NativeSim(robot, variant=variant), Policy(onnx_path, meta, obs_latency=int(variant == "latency"))
    cmds = steptest_commands(meta["cmd_low"], meta["cmd_high"]) if commands is None else np.asarray(commands, dtype=float)
    steps, steady, t0 = round(5.0 / CONTROL_DT), round(2.0 / CONTROL_DT), time.perf_counter()
    per_cmd = []
    for k, cmd in enumerate(cmds):
        rng, errs = np.random.default_rng([seed, k]), []
        for _ in range(trials):
            sim.set_lags(draw_lags(robot, rng))
            sim.reset(), pi.reset()
            err, fell = np.zeros(3), False
            for t in range(steps):
                sim.step(pi(sim, cmd))
                if sim.fallen():
                    fell = True
                    break
                if t >= steps - steady:
                    err += sim.velocity() - cmd
            errs.append(np.nan if fell else float(np.linalg.norm(err / steady) / np.linalg.norm(cmd)))
        e = np.array(errs)
        per_cmd.append([*cmd.tolist(), None if np.isnan(e).all() else float(np.nanmedian(e)), int(np.isnan(e).sum())])
        if not quiet and (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(cmds)} commands, {sum(r[4] for r in per_cmd)} falls, {time.perf_counter() - t0:.0f} s", file=sys.stderr, flush=True)
    ok = [r[3] for r in per_cmd if r[3] is not None]
    by_type = {}
    for name in ("forward", "backward", "sideways", "yaw", "mixed"):
        rows = [r for r in per_cmd if command_type(r[:3]) == name]
        good = [r[3] for r in rows if r[3] is not None]
        by_type[name] = {"commands": len(rows), "median_rel_error": float(np.median(good)) if good else None, "falls": sum(r[4] for r in rows)}
    res = {"onnx": onnx_path, "iteration": meta.get("iteration"), "robot": robot.name, "variant": variant, "trials": trials, "seed": seed,
           "commands": len(cmds), "median_rel_error": float(np.median(ok)) if ok else None,
           "fall_frac": sum(r[4] for r in per_cmd) / (len(cmds) * trials), "cmds_all_fell": sum(r[3] is None for r in per_cmd),
           "by_type": by_type, "per_command": per_cmd}  # fmt: skip
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"steptest_{robot.name}_{run_name(onnx_path)}_{variant}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    if not quiet:
        print({k: v for k, v in res.items() if k != "per_command"}, "->", out)
    return res


def hold_action(q, default_q, scale) -> np.ndarray:
    """The action whose setpoint is the pose q: the previous action a robot-side runner starts with."""
    return (np.asarray(q) - default_q) / scale


def still_after(qd_rms: np.ndarray, v_root: np.ndarray, start: int) -> float:
    """Seconds after step ``start`` until still for good (utils/check_velocity.py's rule): the first 0.5 s block from
    which every block has joint speed RMS below 0.01 rad/s and root planar speed below 0.02 m/s; nan if never."""
    ok = [(qd_rms[b : b + 25].max() < 0.01 or np.sqrt((qd_rms[b : b + 25] ** 2).mean()) < 0.01) and v_root[b : b + 25].max() < 0.02
          for b in range(start, len(qd_rms) - 24, 25)]  # fmt: skip
    return next((0.5 * j for j in range(len(ok)) if all(ok[j:])), float("nan"))


def reversals(q_deg: np.ndarray) -> int:
    """Direction reversals of one joint's trace, counting only moves above 1e-3 degrees per step."""
    d = np.diff(q_deg)
    d = np.sign(d[np.abs(d) > 1e-3])
    return int((d[1:] != d[:-1]).sum())


def start_policy(sim: NativeSim, pi: Policy, start: str, prev_action: str) -> bool:
    """Bring the robot to where the policy takes over; False if it fell before that. start "reset": the standing reset.
    "handover": Booster's prepare phase, T_PREPARE s at the default pose under PREPARE_KP / PREPARE_KD in joint space
    with no policy (and no gain hypothesis: how Booster's own gains reach the linkage is not this tool's question).
    prev_action: what the policy sees as its previous action at its first step, "hold" (the action that holds the
    pose it starts from) or "zeros"."""
    sim.reset(), pi.reset()
    if start == "handover":
        kp, kd, pairs = sim.kp, sim.kd, sim.pairs
        sim.kp, sim.kd, sim.pairs = np.array(PREPARE_KP), np.array(PREPARE_KD), []
        for _ in range(round(T_PREPARE / CONTROL_DT)):
            sim.hold(sim.default_q)
        sim.kp, sim.kd, sim.pairs = kp, kd, pairs
        if sim.fallen():
            return False
    elif start != "reset":
        raise ValueError(f"start {start!r}: reset or handover")
    if prev_action == "hold":
        pi.prev = hold_action(sim.d.qpos[sim.qadr], sim.default_q, sim.scale).astype(np.float32)
    elif prev_action != "zeros":
        raise ValueError(f"prev_action {prev_action!r}: hold or zeros")
    return True


def tilt_deg(sim: NativeSim) -> float:
    return float(np.degrees(np.arccos(np.clip(sim.rotation()[2, 2], -1.0, 1.0))))


def run_stand(onnx_path: str, robot_name: str | None = None, variant: str = "nominal", start: str = "reset",
              prev_action: str = "hold", seed: int = 0, out_dir: str = "logs/native", quiet: bool = False) -> dict:  # fmt: skip
    """The stand check: each of STOP_CASES for T_CMD s, then the zero command up to T_ALL s, from ``start`` (see
    start_policy), each with its own draw of the per-group lags. Per case: a fall; seconds until still for good after
    the stop (and, for the standing case, after the start); the most direction reversals of a waist or ankle joint in
    the last 2 s (shaking); the ankle torques at rest; setpoints beyond the joint ranges. Passed: no fall, still
    within 3 s of the start and of every stop, at most one reversal. Then the exit hand-over: T_STAND s standing, and
    the last setpoint frozen for T_FROZEN s with no policy (what the motors hold until Booster's walking controller
    takes over): the trunk's tilt over time, reported and not part of passed. Written to
    <out_dir>/stand_<robot>_<run>_<variant>_<start>_<prev_action>.json."""
    meta = load_metadata(onnx_path)
    robot = robot_of(meta, robot_name)
    sim, pi = NativeSim(robot, variant=variant), Policy(onnx_path, meta, obs_latency=int(variant == "latency"))
    names = list(robot.joint_names)
    link = [names.index(n) for pair in LINKAGE_PAIRS for n in pair[:2]]
    ankles = [names.index(n) for n in ("left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint")]
    n_cmd, n_all, tail = round(T_CMD / CONTROL_DT), round(T_ALL / CONTROL_DT), round(2.0 / CONTROL_DT)
    cases = []
    for k, cmd in enumerate(STOP_CASES):
        rng = np.random.default_rng([seed, k])
        sim.set_lags(draw_lags(robot, rng))
        row = {"case": list(cmd), "lags": sim.lag[[g[0] for g in robot.delay_groups]].tolist(), "fell": not start_policy(sim, pi, start, prev_action)}
        Q, QD, XY, TAU, beyond = [], [], [], [], []
        for t in range(n_all if not row["fell"] else 0):
            zero = t >= n_cmd or not any(cmd)
            if variant == "push" and zero and t % round(2.0 / CONTROL_DT) == 0 and t > 0:
                sim.push(rng)
            a = pi(sim, (0.0, 0.0, 0.0) if t >= n_cmd else cmd)
            target = sim.default_q + sim.scale * a
            beyond.append(float(np.maximum(sim.jrange[:, 0] - target, target - sim.jrange[:, 1]).max()))
            sim.step(a)
            if sim.fallen():
                row["fell"] = True
                break
            Q.append(sim.d.qpos[sim.qadr].copy()), QD.append(sim.d.qvel[sim.dadr].copy())
            XY.append(sim.d.qpos[sim.root_q : sim.root_q + 2].copy()), TAU.append(sim.d.ctrl[sim.act][ankles].copy())
        if row["fell"]:
            row.update(still_start=None, still_stop=None, reversals=None, ankle_torque=None)
        else:
            Q, QD, XY = np.array(Q), np.array(QD), np.array(XY)
            qd_rms, v_root = np.sqrt((QD**2).mean(1)), np.linalg.norm(np.gradient(XY, axis=0), axis=1) / CONTROL_DT
            nan = lambda x: None if np.isnan(x) else float(x)  # noqa: E731
            row.update(still_start=nan(still_after(qd_rms, v_root, 0)) if not any(cmd) else None,
                       still_stop=nan(still_after(qd_rms, v_root, n_cmd)),
                       reversals=max(reversals(np.degrees(Q[-tail:, j])) for j in link),
                       ankle_torque=np.mean(TAU[-tail:], 0).round(2).tolist())  # fmt: skip
        b = np.array(beyond) if beyond else np.zeros(1)
        row.update(beyond_range=[float((b > 0).mean()), float(max(b.max(), 0.0))], peak_motor=float(sim.peak_motor))
        cases.append(row)
    # the exit hand-over: the last setpoint frozen
    sim.set_lags(draw_lags(robot, np.random.default_rng([seed, len(STOP_CASES)])))
    frozen = {"tilt_deg": [], "fell_at": None}
    if start_policy(sim, pi, start, prev_action):
        for _ in range(round(T_STAND / CONTROL_DT)):
            a = pi(sim, (0.0, 0.0, 0.0))
            sim.step(a)
        target = sim.default_q + sim.scale * a
        for t in range(1, round(T_FROZEN / CONTROL_DT) + 1):
            sim.hold(target)
            if frozen["fell_at"] is None and sim.fallen():
                frozen["fell_at"] = round(t * CONTROL_DT, 2)
            if t in (10, 25, 50, 100):
                frozen["tilt_deg"].append(round(tilt_deg(sim), 2))
    ok = lambda x: x is not None and x <= 3.0  # noqa: E731
    passed = all(not c["fell"] and ok(c["still_stop"]) and c["reversals"] <= 1 for c in cases) and ok(cases[0]["still_start"])
    res = {"onnx": onnx_path, "iteration": meta.get("iteration"), "robot": robot.name, "variant": variant, "start": start,
           "prev_action": prev_action, "seed": seed, "passed": bool(passed), "cases": cases, "frozen": frozen}  # fmt: skip
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"stand_{robot.name}_{run_name(onnx_path)}_{variant}_{start}_{prev_action}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    if not quiet:
        print(describe(sim).split(", setpoints")[0] + f"; variant {variant}, start {start}, previous action {prev_action}")
        print("  first 3 s cmd | fell | still after start / stop s | reversals | ankle torque N m: pitch L / R, roll L / R | "
              "setpoints beyond range: share, most rad | most motor N m")
        for c in cases:
            f2 = lambda x: " -- " if x is None else f"{x:4.1f}"  # noqa: E731
            at = c["ankle_torque"] or [float("nan")] * 4
            print(f"  {c['case'][0]:+.1f} {c['case'][1]:+.1f} {c['case'][2]:+.1f} | {'yes' if c['fell'] else 'no':3} | {f2(c['still_start'])} / "
                  f"{f2(c['still_stop'])} | {c['reversals']} | {at[0]:+6.1f} {at[1]:+6.1f} {at[2]:+6.1f} {at[3]:+6.1f} | "
                  f"{c['beyond_range'][0]:.3f} {c['beyond_range'][1]:.3f} | {c['peak_motor']:.0f}")  # fmt: skip
        print(f"  frozen setpoint after {T_STAND:g} s standing: tilt at 0.2 / 0.5 / 1 / 2 s {frozen['tilt_deg']} deg, "
              f"fell at {frozen['fell_at']} s")
        print(f"  {'PASSED' if passed else 'FAILED'} (no fall, still within 3 s of the start and of each stop, no shaking) -> {out}")
    return res


def run_viewer(onnx_path: str, robot_name: str | None = None, delay: int | None = None):
    import glfw  # a mujoco dependency
    import mujoco.viewer

    meta = load_metadata(onnx_path)
    sim = NativeSim(robot_of(meta, robot_name), delay)
    pi = Policy(onnx_path, meta)
    print(describe(sim), flush=True)
    # the robot's own MJCF for display (the simulated model has neither meshes nor visual geoms), posed as the simulator
    vm = mujoco.MjModel.from_xml_path(sim.robot.mjcf_path)
    vd = mujoco.MjData(vm)
    v_root = int(vm.jnt_qposadr[0])  # the floating base
    v_qadr = [int(vm.jnt_qposadr[vm.joint(n).id]) for n in sim.robot.joint_names]

    def show():
        vd.qpos[v_root : v_root + 7] = sim.d.qpos[sim.root_q : sim.root_q + 7]
        vd.qpos[v_qadr] = sim.d.qpos[sim.qadr]
        mujoco.mj_forward(vm, vd)

    cmd, want_reset = [0.0, 0.0, 0.0], [False]
    keys = {glfw.KEY_UP: (0, 0.1), glfw.KEY_DOWN: (0, -0.1), glfw.KEY_A: (1, 0.1), glfw.KEY_D: (1, -0.1),
            glfw.KEY_LEFT: (2, 0.2), glfw.KEY_RIGHT: (2, -0.2)}  # fmt: skip
    lo, hi = meta["cmd_low"], meta["cmd_high"]

    def key(k):  # the viewer's UI thread; plain float writes under the GIL
        if k in keys:
            i, dv = keys[k]
            cmd[i] = min(max(round(cmd[i] + dv, 2), lo[i]), hi[i])
        elif k in (glfw.KEY_SPACE, glfw.KEY_ENTER):
            cmd[:] = [0.0, 0.0, 0.0]
        elif k == glfw.KEY_BACKSPACE:
            cmd[:] = [0.0, 0.0, 0.0]
            want_reset[0] = True

    show()
    with mujoco.viewer.launch_passive(vm, vd, key_callback=key) as v:
        v.cam.type, v.cam.distance = mujoco.mjtCamera.mjCAMERA_TRACKING, 3.0
        v.cam.trackbodyid = int(vm.jnt_bodyid[0])  # the floating base
        start, tick = time.perf_counter(), 0
        while v.is_running():
            sim.step(pi(sim, list(cmd)))
            act = sim.velocity()
            if want_reset[0] or sim.fallen():
                sim.reset(), pi.reset()
                want_reset[0] = False
            show()
            v.set_texts((None, None, "up/down vx | a/d vy | left/right yaw | space stop | backspace reset",
                         f"cmd vx {cmd[0]:+.1f} vy {cmd[1]:+.1f} yaw {cmd[2]:+.1f}   "
                         f"act vx {act[0]:+.2f} vy {act[1]:+.2f} yaw {act[2]:+.2f}"))  # fmt: skip
            v.sync()
            tick += 1
            time.sleep(max(0.0, start + tick * CONTROL_DT - time.perf_counter()))  # wall-clock pacing


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True, help="an export of utils/export_onnx.py")
    p.add_argument("--robot", default=None, choices=ROBOTS, help="default: the robot whose numbers the file records")
    p.add_argument("--delay", type=int, default=None, help="physics steps (5 ms) each setpoint is applied late (default: the robot's evaluation delay)")
    p.add_argument("--script", action="store_true", help="no viewer: run SCRIPT and print commanded vs actual velocity")
    p.add_argument("--variant", default="nominal", choices=VARIANTS, help="a difference from training (not for the viewer; see VARIANTS)")
    p.add_argument("--steptest", action="store_true", help="no viewer: the step test over the command box, random per-group lags")
    p.add_argument("--trials", type=int, default=4, help="--steptest: lag draws per command")
    p.add_argument("--seed", type=int, default=0, help="--steptest, --stand: the lag draws' seed")
    p.add_argument("--stand", action="store_true", help="no viewer: the stand check (still within 3 s of the start and of each stop)")
    p.add_argument("--start", default="reset", choices=("reset", "handover"), help="--stand: the standing reset, or Booster's prepare hand-over")
    p.add_argument("--prev_action", default="hold", choices=("hold", "zeros"), help="--stand: the policy's previous action at its first step")
    a = p.parse_args()
    if a.stand:
        run_stand(a.onnx, a.robot, a.variant, a.start, a.prev_action, a.seed)
    elif a.steptest:
        run_steptest(a.onnx, a.robot, a.variant, a.trials, a.seed)
    elif a.script:
        run_script(a.onnx, a.robot, a.delay, variant=a.variant)
    else:
        run_viewer(a.onnx, a.robot, a.delay)
