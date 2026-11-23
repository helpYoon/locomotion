# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mujoco==3.12.0",
#     "torch==2.11.0",
#     "numpy==2.5.4",
#     "onnxruntime==1.31.1",
#     "glfw==2.11.0",
#     "imageio-ffmpeg==0.6.0",
#     "matplotlib==3.11.2",
#     "pillow==13.0.0",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Run an ONNX export of a velocity-command interface (deploy/'s format, utils/export_onnx.py's output) in native MuJoCo
(sim-to-sim): a video, the step test, or a viewer. A standalone script: it brings its own environment (the header
above, locked in utils/play_mujoco.py.lock), imports nothing from this repo and reads only the ONNX file and assets/t1/.

video    a scripted command sequence from standing, rendered to an MP4 with a plot of commanded vs actual velocity.
steptest the step test (from standing, a fixed command for 5 s, error over the last 2 s) on a grid.
viewer   keyboard teleop in the MuJoCo viewer (needs a display), humanoid_nmpc's keys: up / down vx +-0.1 m/s,
         a / d vy +-0.1 m/s, left / right yaw rate +-0.2 rad/s (capped at the command box), space stops,
         backspace resets the robot; it starts at rest and resets itself after a fall.
landtest rollouts of an ONNX export of the landing interface (utils/export_landing.py), each from the default stance
         with one foot-landing command or two, or (--walk_onnx) from where a walk and a stop under a velocity export
         leave the robot. Started by utils/landing_native.py, which writes the commands and scores the recorded
         rollouts; not meant to be run by hand.
landvideo one rollout of the hand-over, rendered to an MP4: a walk and a stop under a velocity export (--walk_onnx),
         then the landing export (--onnx) takes over and is sent LAND_VIDEO's two commands.
Outputs are named with _onnx; no checkpoint and no motion data are needed.

The robot is a snapshot of the training model the old SuperMimic env simulated (Isaac Lab on Newton), with each
joint's PD gains, torque-speed curve, delay range and action scale (T1 order) beside it: the old play_mujoco's export
mode (tag supermimic-frozen-2026-11-15) wrote both to assets/t1/mujoco/, where they are tracked, and robot/t1.py reads
the same table. The controller is that env's: every physics step an explicit PD torque toward the setpoint its
actuator group recorded lag steps earlier (lags 2-8 physics steps, drawn per episode; the oldest setpoint until that
many exist), clipped by the motor's torque-speed curve, applied as a generalized force (Newton's joint_f ->
qfrc_applied); a CONTROL_DT control step; the noise-free observation with its history, the first observation repeated.

usage (from the repo root; the model files and logs/mujoco/ are relative paths):
       uv run utils/play_mujoco.py steptest --onnx logs/land_r4_d10_al2_s3/ckpt_004000_both.onnx
       uv run utils/play_mujoco.py viewer --onnx logs/land_r4_d10_al2_s3/ckpt_004000_both.onnx
       MUJOCO_GL=egl uv run utils/play_mujoco.py video --onnx logs/land_r4_d10_al2_s3/ckpt_004000_both.onnx
"""

from __future__ import annotations

import argparse
import json
import os

import mujoco
import numpy as np
import torch

OUT_DIR = "logs/mujoco"
# the old export mode's snapshot of the training model, tracked; a .mjb loads only under the MuJoCo version that wrote
# it (3.12.0)
MODEL_DIR = "assets/t1/mujoco"
MODEL_PATH = os.path.join(MODEL_DIR, "t1_newton.mjb")
ACT_PATH = os.path.join(MODEL_DIR, "t1_actuators.npz")
# the T1's MJCF (meshes, lights, checker floor), which the old training converted to USD
MJCF_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets", "t1", "t1_description", "mjcf", "t1.xml")  # fmt: skip
# MjOption fields MuJoCo-Warp keeps under the same name (it stores impratio as impratio_invsqrt; it has no noslip)
OPT_FIELDS = ("timestep", "tolerance", "ls_tolerance", "ccd_tolerance", "gravity", "wind", "magnetic", "density",
              "viscosity", "integrator", "cone", "solver", "iterations", "ls_iterations", "ccd_iterations",
              "disableflags", "enableflags")  # fmt: skip
INIT_HEIGHT = 0.688  # the old T1_CFG.init_state.pos[2], the step test's start
CONTROL_DT = 0.02  # [s], the policy's step
DECIMATION = 4  # physics steps (the mjb's 5 ms timestep) per control step
# command box, vx, vy [m/s] in the heading frame, yaw rate [rad/s]: training's (the T1 library's 0.1-99.9 percentiles)
CMD_LOW, CMD_HIGH = (-1.05, -0.8, -2.8), (1.4, 0.8, 2.8)
# training's fall termination: root height [m] below FALL_HEIGHT, or 1 - R_zz above FALL_TILT (about 73 deg)
FALL_HEIGHT, FALL_TILT = 0.32, 0.7
T1_JOINT_NAMES = [
    "AAHead_yaw", "Head_pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
    "Waist",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]  # fmt: skip
# video: (duration [s], command (vx, vy) [m/s] in the heading frame and yaw rate [rad/s])
SCRIPT = (
    (2.0, (0.0, 0.0, 0.0)), (4.0, (0.5, 0.0, 0.0)), (4.0, (1.0, 0.0, 0.0)), (3.0, (0.0, 0.5, 0.0)),
    (3.0, (0.0, -0.5, 0.0)), (4.0, (0.4, 0.0, 1.0)), (3.0, (0.0, 0.0, -1.5)), (3.0, (-0.6, 0.0, 0.0)),
    (2.0, (0.0, 0.0, 0.0)),
)  # fmt: skip


def joint_addresses(m: mujoco.MjModel, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """qpos and qvel addresses of the named hinge joints (Newton names them by their path, joined with "_")."""
    all_names = [m.joint(i).name for i in range(m.njnt)]
    ids = []
    for n in names:
        (i,) = [i for i, s in enumerate(all_names) if s == n or s.endswith("_" + n)]  # exactly one match
        ids.append(i)
    return m.jnt_qposadr[ids], m.jnt_dofadr[ids]


def root_addresses(m: mujoco.MjModel) -> tuple[int, int, int]:
    """The free joint's qpos and qvel addresses and its body."""
    free = next(i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
    return m.jnt_qposadr[free], m.jnt_dofadr[free], m.jnt_bodyid[free]


class Visual:
    """The MJCF (meshes, lights, checker floor) posed like a simulated T1, for rendering and the viewer."""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(MJCF_PATH)
        self.d = mujoco.MjData(self.m)
        self.qadr, _ = joint_addresses(self.m, T1_JOINT_NAMES)
        self.root_q, _, self.root_body = root_addresses(self.m)

    def pose(self, sim: "T1Mujoco"):
        self.d.qpos[self.root_q : self.root_q + 7] = sim.d.qpos[sim.root_q : sim.root_q + 7]
        self.d.qpos[self.qadr] = sim.d.qpos[sim.qadr]
        mujoco.mj_forward(self.m, self.d)


class T1Mujoco:
    """One T1 in native MuJoCo under the training env's actuator model, states as (p, R, v, w, q, qd) tuples
    (numpy)."""

    def __init__(self, seed: int = 0, model: str = "newton"):
        """model: "newton", the exported training model, or "mjcf", the MJCF (MJCF_PATH) with the training model's
        solver options: the same robot, contacts and self-collision pairs, so mostly a check of the export."""
        self.m = mujoco.MjModel.from_binary_path(MODEL_PATH)
        if model == "mjcf":
            opt, self.m = self.m.opt, mujoco.MjModel.from_xml_path(MJCF_PATH)
            for name in OPT_FIELDS + ("impratio",):
                setattr(self.m.opt, name, getattr(opt, name))
        self.d = mujoco.MjData(self.m)
        self.qadr, self.vadr = joint_addresses(self.m, T1_JOINT_NAMES)
        self.root_q, self.root_v, self.root_body = root_addresses(self.m)
        act = np.load(ACT_PATH)
        self.kp, self.kd, self.tau_max = act["kp"], act["kd"], act["tau_max"]
        self.v_max, self.denom = act["v_max"], np.maximum(act["v_max"] - act["v_knee"], 1e-6)
        self.scale, self.default_q = act["action_scale"], act["default_q"]
        self.min_delay, self.max_delay, self.group = act["min_delay"], act["max_delay"], act["group"]
        self.substeps = DECIMATION
        self.rng = np.random.default_rng(seed)
        groups = range(self.group.max() + 1)
        self.group_lo, self.group_hi = (np.array([x[self.group == g][0] for g in groups]) for x in (self.min_delay, self.max_delay))
        self.joints = np.arange(len(self.qadr))

    def reset(self, lags: np.ndarray | None = None):
        """Stand at the default pose, INIT_HEIGHT, at rest (the step test's start); lags per actuator group, drawn
        uniformly from each group's range if None."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[self.root_q : self.root_q + 7] = (0.0, 0.0, INIT_HEIGHT, 1.0, 0.0, 0.0, 0.0)
        d.qpos[self.qadr] = self.default_q
        mujoco.mj_forward(m, d)
        self.group_lags = self.rng.integers(self.group_lo, self.group_hi + 1) if lags is None else np.asarray(lags)
        self.lag = self.group_lags[self.group]
        self.setpoints = []  # one per physics step since reset, newest last

    def state(self):
        d, r, v = self.d, self.root_q, self.root_v
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, d.qpos[r + 3 : r + 7])
        R = R.reshape(3, 3)
        # a free joint's qvel: the body origin's world linear velocity, then the angular velocity in the body frame
        return (d.qpos[r : r + 3].copy(), R, d.qvel[v : v + 3].copy(), R @ d.qvel[v + 3 : v + 6],
                d.qpos[self.qadr].copy(), d.qvel[self.vadr].copy())  # fmt: skip

    def step(self, action: np.ndarray):
        """One control step toward the joint setpoint of action (T1 order)."""
        m, d = self.m, self.d
        setpoint = self.default_q + self.scale * action
        for _ in range(self.substeps):
            self.setpoints.append(setpoint)  # as Isaac Lab's delay buffers, one record per physics step
            del self.setpoints[: -int(self.max_delay.max()) - 1]
            idx = np.maximum(len(self.setpoints) - 1 - self.lag, 0)
            target = np.stack(self.setpoints)[idx, self.joints]
            q, qd = d.qpos[self.qadr], d.qvel[self.vadr]
            tau = self.kp * (target - q) - self.kd * qd
            limit = np.clip(self.tau_max * (self.v_max - np.abs(qd)) / self.denom, 0.0, self.tau_max)
            d.qfrc_applied[:] = 0.0
            d.qfrc_applied[self.vadr] = np.clip(tau, -limit, limit)
            mujoco.mj_step(m, d)


class VelocityPolicy:
    """An ONNX export of a velocity interface under onnxruntime on CPU, fed T1Mujoco states: its input shape and
    metadata give the history length and iteration."""

    def __init__(self, onnx: str):
        import onnxruntime as ort

        self.default_q = torch.tensor(np.load(ACT_PATH)["default_q"])  # float32, the training env's default pose
        self.xi_joint_offset = torch.zeros(len(self.default_q))  # eval: no joint calibration offset
        opts = ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = torch.get_num_threads(), 1  # 1 in step test workers
        self.session = ort.InferenceSession(onnx, opts, providers=["CPUExecutionProvider"])
        shapes = {x.name: x.shape for x in self.session.get_inputs()}
        self.hist_len = shapes["obs_hist"][1]
        # a file that also holds the landing interface (utils/export_landing.py --both): velocity mode
        self.rest = {k: np.zeros(shapes[k], np.float32) for k in ("mode", "land_cmd", "state") if k in shapes}
        self.iteration = json.loads(self.session.get_modelmeta().custom_metadata_map["iteration"])

    def reset(self):
        self.hist, self.prev_action = None, torch.zeros(len(self.default_q))

    @torch.no_grad()
    def __call__(self, state, cmd) -> np.ndarray:
        p, R, v, w, q, qd = (torch.as_tensor(x, dtype=torch.float32) for x in state)
        # the noise-free policy observation: joint_pos, joint_vel, ang_vel_b (R^T w), gravity_b (R^T (0, 0, -1), minus
        # R's last row), prev_action
        o = torch.cat((q - self.default_q - self.xi_joint_offset, qd, (R.mT * w[..., None, :]).sum(-1), -R[..., 2, :],
                       self.prev_action), -1)  # fmt: skip
        self.hist = o[None].repeat(self.hist_len, 1) if self.hist is None else torch.cat((self.hist[1:], o[None]))
        cmd = torch.as_tensor(cmd, dtype=torch.float32)[None]
        a = self.session.run(["action"], {"obs_hist": self.hist[None].numpy(), "vel_cmd" if self.rest else "cmd": cmd.numpy(), **self.rest})[0]
        a = torch.from_numpy(a[0])
        self.prev_action = a
        return a.numpy().astype(np.float64)


class LandingPolicy(VelocityPolicy):
    """An ONNX export of a landing interface (utils/export_landing.py): VelocityPolicy's observation, a command
    (go, foot, x, y, yaw) and the model's own state, fed back from its previous output (zeros at a reset). A file
    with a mode input (--both) also holds the velocity interface: walk (vx, vy, yaw rate) then runs that instead."""

    def __init__(self, onnx: str):
        super().__init__(onnx)
        meta = self.session.get_modelmeta().custom_metadata_map
        self.both = "mode" in {x.name for x in self.session.get_inputs()}
        self.terms, self.clock_max = json.loads(meta["state_terms"]), json.loads(meta["clock_range"])[1]

    def reset(self):
        super().reset()
        self.state = np.zeros((1, len(self.terms)), np.float32)

    @torch.no_grad()
    def __call__(self, state, cmd, walk=None) -> np.ndarray:
        p, R, v, w, q, qd = (torch.as_tensor(x, dtype=torch.float32) for x in state)
        o = torch.cat((q - self.default_q - self.xi_joint_offset, qd, (R.mT * w[..., None, :]).sum(-1), -R[..., 2, :],
                       self.prev_action), -1)  # fmt: skip  VelocityPolicy's observation
        self.hist = o[None].repeat(self.hist_len, 1) if self.hist is None else torch.cat((self.hist[1:], o[None]))
        feed = {"obs_hist": self.hist[None].numpy(), "state": self.state}
        if self.both:
            feed.update(mode=np.array([[float(walk is None)]], np.float32), land_cmd=np.asarray([cmd], np.float32),
                        vel_cmd=np.asarray([np.zeros(3) if walk is None else walk], np.float32))  # fmt: skip
        else:
            assert walk is None, "this export holds no velocity interface: pass a velocity export as --walk_onnx"
            feed["cmd"] = np.asarray([cmd], dtype=np.float32)
        a, self.state = self.session.run(None, feed)
        self.prev_action = torch.from_numpy(a[0])
        return a[0].astype(np.float64)


def heading_velocity(state) -> np.ndarray:
    """(vx, vy) in the heading frame and the world yaw rate, as the velocity reward and the step test read them."""
    p, R, v, w, *_ = state
    yaw = np.arctan2(R[1, 0], R[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], w[2]])


def fallen(state) -> bool:
    """Training's fall termination (FALL_HEIGHT, FALL_TILT)."""
    return bool(state[0][2] < FALL_HEIGHT or 1 - state[1][2, 2] > FALL_TILT)


def video(args):
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont

    sim, pi, vis = T1Mujoco(args.seed, args.model), VelocityPolicy(args.onnx), Visual()
    sim.reset(), pi.reset()
    stem = os.path.join(OUT_DIR, f"walk_{pi.iteration:06d}_{args.model}_lags{''.join(map(str, sim.group_lags))}_onnx")
    os.makedirs(OUT_DIR, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(stem + ".mp4", (args.width, args.height), fps=args.fps)  # H.264, yuv420p
    writer.send(None)
    font = ImageFont.load_default(size=args.height // 30)
    vis.m.vis.global_.offwidth, vis.m.vis.global_.offheight = args.width, args.height
    renderer = mujoco.Renderer(vis.m, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, vis.root_body
    cam.distance, cam.elevation, cam.azimuth = 3.0, -15.0, 135.0
    log, t, every = [], 0, max(1, round(1 / (args.fps * CONTROL_DT)))
    for duration, cmd in SCRIPT:
        for _ in range(round(duration / CONTROL_DT)):
            state = sim.state()
            sim.step(pi(state, cmd))
            state = sim.state()
            log.append((t * CONTROL_DT, *cmd, *heading_velocity(state), state[0][2], fallen(state)))
            if t % every == 0:
                vis.pose(sim)
                renderer.update_scene(vis.d, camera=cam)
                img = Image.fromarray(renderer.render())
                act = heading_velocity(state)
                ImageDraw.Draw(img).text(
                    (8, 6), f"t {t * CONTROL_DT:4.1f} s  cmd vx {cmd[0]:+.1f} vy {cmd[1]:+.1f} yaw {cmd[2]:+.1f}\n"
                    f"         act vx {act[0]:+.2f} vy {act[1]:+.2f} yaw {act[2]:+.2f}", fill=(255, 255, 255), font=font)  # fmt: skip
                writer.send(np.asarray(img))
            t += 1
    writer.close()
    log = np.array(log, dtype=np.float64)
    np.save(stem + ".npy", log)
    plot(log, stem + ".png", f"ckpt {pi.iteration}, MuJoCo ({args.model} model), actuator lags {sim.group_lags.tolist()}")
    print(f"wrote {stem}.mp4/.png/.npy; fell: {bool(log[:, -1].any())}; lags {sim.group_lags.tolist()}")
    for (duration, cmd), seg in zip(SCRIPT, np.split(log, np.cumsum([round(d / CONTROL_DT) for d, _ in SCRIPT])[:-1])):
        tail = seg[len(seg) // 2 :]  # the second half of each segment: settled
        print(f"cmd {cmd}: mean actual {np.round(tail[:, 4:7].mean(0), 3).tolist()}, min height {seg[:, 7].min():.3f}")


def plot(log: np.ndarray, path: str, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, ("vx [m/s]", "vy [m/s]", "yaw rate [rad/s]"))):
        ax.plot(log[:, 0], log[:, 4 + i], color="#2a78d6", lw=1.2, label="actual")
        ax.plot(log[:, 0], log[:, 1 + i], color="#52514e", lw=1.2, ls="--", label="command")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].set_title(title)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


_worker = {}


def _init_worker(args):
    torch.set_num_threads(1)
    _worker["sim"], _worker["pi"] = T1Mujoco(args.seed, args.model), VelocityPolicy(args.onnx)


def _command_errors(k: int, cmd: np.ndarray, args) -> list[float]:
    """The step test's relative errors of --trials rollouts of command k (nan where it fell), each with actuator
    lags drawn from a generator seeded by (--seed, k)."""
    sim, pi = _worker["sim"], _worker["pi"]
    rng, steps, steady = np.random.default_rng([args.seed, k]), round(5.0 / CONTROL_DT), round(2.0 / CONTROL_DT)
    errs = []
    for _ in range(args.trials):
        sim.reset(rng.integers(sim.group_lo, sim.group_hi + 1))
        pi.reset()
        err, fell = np.zeros(3), False
        for t in range(steps):
            sim.step(pi(sim.state(), cmd))
            state = sim.state()
            fell |= fallen(state)
            if t >= steps - steady:
                err += heading_velocity(state) - cmd
        errs.append(np.nan if fell else float(np.linalg.norm(err / steady) / np.linalg.norm(cmd)))
    return errs


def step_test(args):
    """The step test on a --grid grid of the command box (per plane, the standing command left out), --trials random
    lag draws per command: per command the median relative error over trials that didn't fall, then the median."""
    import multiprocessing as mp

    axes = [np.linspace(CMD_LOW[i], CMD_HIGH[i], args.grid) for i in range(3)]
    for x in axes:
        x[np.abs(x) < 1e-6] = 0.0
    cmds = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = np.meshgrid(axes[i], axes[j], indexing="ij")
        c = np.zeros((args.grid**2, 3))
        c[:, i], c[:, j] = a.ravel(), b.ravel()
        cmds += [x for x in c if np.linalg.norm(x) > 1e-6]
    _init_worker(args)  # once here first: a worker whose setup raises is respawned forever
    with mp.get_context("spawn").Pool(args.jobs, initializer=_init_worker, initargs=(args,)) as pool:
        errs = pool.starmap(_command_errors, [(k, c, args) for k, c in enumerate(cmds)], chunksize=2)
    per_cmd = [[*c.tolist(), None if np.isnan(e).all() else float(np.nanmedian(e)), int(np.isnan(e).sum())]
               for c, e in zip(cmds, map(np.array, errs))]  # fmt: skip
    ok = [r[3] for r in per_cmd if r[3] is not None]
    iteration = _worker["pi"].iteration
    # "ckpt": the old tool's field (its checkpoint mode), kept so the JSON compares with its results
    res = {"iteration": iteration, "ckpt": None, "onnx": args.onnx, "model": args.model, "grid": args.grid,
           "trials": args.trials, "seed": args.seed, "median_rel_error": float(np.median(ok)) if ok else None,
           "fall_frac": sum(r[4] for r in per_cmd) / (len(cmds) * args.trials),
           "cmds_all_fell": sum(r[3] is None for r in per_cmd), "per_command": per_cmd}  # fmt: skip
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"steptest_{iteration:06d}_{args.model}_grid{args.grid}_seed{args.seed}_onnx.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print({k: v for k, v in res.items() if k != "per_command"}, "->", out)


def _init_land_worker(args):
    torch.set_num_threads(1)
    _worker["sim"], _worker["pi"] = T1Mujoco(args.seed, args.model), LandingPolicy(args.onnx)
    _worker["vel"] = _walk_policy(args)


def _walk_policy(args) -> VelocityPolicy | None:
    """The velocity export of --walk_onnx; None without one, or when it is --onnx itself (a --both export, which
    walks through its own velocity interface)."""
    same = args.walk_onnx and os.path.abspath(args.walk_onnx) == os.path.abspath(args.onnx)
    return VelocityPolicy(args.walk_onnx) if args.walk_onnx and not same else None


def _walker(pi: LandingPolicy, vel: VelocityPolicy | None):
    """state, velocity command -> action: vel's, or pi's own velocity interface."""
    return vel if vel else lambda s, v: pi(s, np.zeros(5), v)


def _land_rollout(k: int, swing: np.ndarray, c: np.ndarray, walk: np.ndarray | None, args) -> tuple:
    """Rollout k of landing commands, from standing in the default pose, with actuator lags drawn from a generator
    seeded by (--seed, k), or --lag on every group. swing (n,): each command's sole to move, 0 left or 1 right; c
    (n, 3): its landing pose in the other sole's frame. The first command is sent after one second of standing (the
    model's SETTLE and some hold), each next one a second after the deadline of the one before; the rollout ends
    --hold_s seconds after the last command's longest clock.
    walk (vx, vy [m/s] in the heading frame, yaw rate [rad/s], seconds walked, seconds stood), if given: the
    velocity export (--walk_onnx) first walks on that command and then stands on the zero command; the landing model
    takes over its observation history and last action with a state of zeros, and states and steps count from there.
    Returns per state (state j is after j control steps) the root's position and wxyz quaternion, the joints, the
    joints' speed RMS [rad/s], the root's speed [m/s] and the fall test; then per command the control step it was
    sent on and the clock the model gave it [control steps]; the lags; and whether the robot fell during the walk."""
    sim, pi = _worker["sim"], _worker["pi"]
    lags = np.random.default_rng([args.seed, k]).integers(sim.group_lo, sim.group_hi + 1)
    sim.reset(lags if args.lag is None else np.full_like(lags, args.lag))
    pi.reset()
    walk_fell = False
    if walk is not None:
        vel, (n_walk, n_stop) = _worker["vel"], (round(x / CONTROL_DT) for x in walk[3:])
        step = _walker(pi, vel)
        if vel:
            vel.reset()
        for j in range(n_walk + n_stop):
            s = sim.state()
            walk_fell |= fallen(s)
            sim.step(step(s, walk[:3] if j < n_walk else np.zeros(3)))
        if vel:
            pi.hist, pi.prev_action = vel.hist, vel.prev_action  # the hand-over between two files
    left, one_s = pi.terms.index("steps_left"), round(1.0 / CONTROL_DT)
    steps = len(swing) * (one_s + pi.clock_max) + round(args.hold_s / CONTROL_DT)
    rec, sent, due = [], [], one_s
    for j in range(steps + 1):
        s = sim.state()
        rec.append((s[0], sim.d.qpos[sim.root_q + 3 : sim.root_q + 7].copy(), s[4], np.sqrt((s[5] ** 2).mean()),
                    np.linalg.norm(s[2]), fallen(s)))  # fmt: skip
        if j == steps:
            break
        i = len(sent)
        go = j == due and i < len(swing)
        a = pi(s, (1.0, swing[i], *c[i]) if go else np.zeros(5))
        if go:  # the model's state now holds the steps left after this one
            clock = int(pi.state[0, left]) + 1
            assert clock > 1, f"rollout {k}: command {i + 1} was not taken, the model was not holding"
            sent.append((j, clock))
            due = j + clock + one_s
        sim.step(a)
    return [np.array(x) for x in zip(*rec)], np.array(sent, int).reshape(-1, 2), sim.group_lags, walk_fell


def land_test(args):
    """The rollouts of --commands (an .npz of swing (K, n) and c (K, n, 3): K rollouts of n landing commands each,
    written by utils/landing_native.py), each from the default stance; they go to --out (.npz), which
    utils/landing_native.py scores with the training simulator's scoring. The file may also hold walk (K, 5), each
    rollout's walk and stop before the landing model takes over (_land_rollout; needs --walk_onnx), and k (K,), the
    rollout numbers the lag draws are seeded by (0 .. K-1 if absent)."""
    import multiprocessing as mp

    cmds = np.load(args.commands)
    K = len(cmds["swing"])
    ids, walk = (cmds[n] if n in cmds else d for n, d in (("k", range(K)), ("walk", [None] * K)))
    jobs = [(int(k), s, c, w, args) for k, s, c, w in zip(ids, cmds["swing"], cmds["c"], walk)]
    _init_land_worker(args)  # once here first: a worker whose setup raises is respawned forever
    with mp.get_context("spawn").Pool(args.jobs, initializer=_init_land_worker, initargs=(args,)) as pool:
        res = pool.starmap(_land_rollout, jobs, chunksize=2)
    names = ("root_p", "root_quat", "q", "qd_rms", "root_speed", "fell")
    rec = {n: np.stack([r[0][i] for r in res], 1) for i, n in enumerate(names)}  # (states, rollouts, ...)
    sent = np.stack([r[1] for r in res])  # (rollouts, commands, 2)
    np.savez(args.out, t_cmd=sent[..., 0], n0=sent[..., 1], lags=np.stack([r[2] for r in res]),
             walk_fell=np.array([r[3] for r in res]),
             **{n: x if x.dtype == bool else x.astype(np.float32) for n, x in rec.items()})  # fmt: skip
    print(f"{len(res)} rollouts{' after a walk and a stop' * ('walk' in cmds)}, {sent.shape[1]} landing "
          f"command{'s' * (sent.shape[1] != 1)} each, {rec['fell'].any(0).sum()} falls -> {args.out}")


# landvideo: the walk (vx, vy [m/s], yaw rate [rad/s]; seconds walked, seconds stood), then the two commands: the
# left sole moved 20 cm forward from where it stands, then the right sole brought beside it (the default stance's
# width, 21.25 cm)
LAND_VIDEO = ((0.5, 0.0, 0.0), 3.0, 3.0, 0.20, (0.0, -0.2125, 0.0))


def land_video(args):
    """LAND_VIDEO in one rollout, each command a second after the hold began; the landing errors printed are the
    model's own reading of the stance it landed in (its state, from the joints) against the command."""
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont

    sim, pi, vel, vis = T1Mujoco(args.seed, args.model), LandingPolicy(args.onnx), _walk_policy(args), Visual()
    sim.reset(None if args.lag is None else np.full(len(sim.group_lo), args.lag))
    pi.reset()
    if vel:
        vel.reset()
    walk = _walker(pi, vel)
    stem = os.path.join(OUT_DIR, f"land_walk_{pi.iteration:06d}_{args.model}_lags{''.join(map(str, sim.group_lags))}_onnx")
    os.makedirs(OUT_DIR, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(stem + ".mp4", (args.width, args.height), fps=args.fps)
    writer.send(None)
    font = ImageFont.load_default(size=args.height // 30)
    vis.m.vis.global_.offwidth, vis.m.vis.global_.offheight = args.width, args.height
    renderer = mujoco.Renderer(vis.m, args.height, args.width)
    for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION):
        renderer.scene.flags[flag] = 0  # a software renderer draws them with artifacts
    cam = mujoco.MjvCamera()
    cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, vis.root_body
    cam.distance, cam.elevation, cam.azimuth = 2.2, -20.0, 135.0  # closer than video's: the feet
    t, every, fell = [0], max(1, round(1 / (args.fps * CONTROL_DT))), [False]

    def tick(act, label: str):
        sim.step(act(sim.state()))
        fell[0] |= fallen(sim.state())
        if t[0] % every == 0:
            vis.pose(sim)
            renderer.update_scene(vis.d, camera=cam)
            img = Image.fromarray(renderer.render())
            ImageDraw.Draw(img).text((8, 6), f"t {t[0] * CONTROL_DT:4.1f} s  {label}", fill=(255, 255, 255), font=font)
            writer.send(np.asarray(img))
        t[0] += 1

    cmd, walk_s, stop_s, ahead, beside = LAND_VIDEO
    for _ in range(round(walk_s / CONTROL_DT)):
        tick(lambda s: walk(s, cmd), f"velocity policy: walk vx {cmd[0]:+.1f} vy {cmd[1]:+.1f} yaw {cmd[2]:+.1f}")
    for _ in range(round(stop_s / CONTROL_DT)):
        tick(lambda s: walk(s, np.zeros(3)), "velocity policy: stop")
    if vel:
        pi.hist, pi.prev_action = vel.hist, vel.prev_action  # the hand-over between two files
    left, xyz, one_s = pi.terms.index("steps_left"), [pi.terms.index(k) for k in ("x", "y", "yaw")], round(1 / CONTROL_DT)
    for i, name in enumerate(("left foot 20 cm forward", "right foot beside it")):
        for _ in range(one_s):
            tick(lambda s: pi(s, np.zeros(5)), "landing policy: hold" + ", hand-over done" * (i == 0))
        # after SETTLE the state reads the left sole in the right sole's frame
        c = pi.state[0, xyz] + (ahead, 0.0, 0.0) if i == 0 else np.array(beside)
        tick(lambda s: pi(s, (1.0, float(i), *c)), f"landing policy: {name}")
        assert pi.state[0, left] > 0, "the command was not taken"
        while pi.state[0, left] > 0:
            tick(lambda s: pi(s, np.zeros(5)), f"landing policy: {name}, {int(pi.state[0, left])} steps left")
        tick(lambda s: pi(s, np.zeros(5)), "landing policy: hold")  # the stance is latched on this tick's state
        e = pi.state[0, xyz] - c
        print(f"{name}: landed {100 * np.hypot(e[0], e[1]):.2f} cm and {np.degrees(abs(e[2])):.2f} deg from the command "
              "(the model's own reading)")
    for _ in range(3 * one_s):
        tick(lambda s: pi(s, np.zeros(5)), "landing policy: hold")
    writer.close()
    print(f"wrote {stem}.mp4 ({t[0] * CONTROL_DT:.1f} s); fell: {fell[0]}; lags {sim.group_lags.tolist()}")


def viewer(args):
    import time

    import glfw  # a mujoco dependency
    import mujoco.viewer

    sim, pi, vis = T1Mujoco(args.seed, args.model), VelocityPolicy(args.onnx), Visual()
    sim.reset(), pi.reset()
    vis.pose(sim)
    cmd, reset = np.zeros(3), [False]
    step = {glfw.KEY_UP: (0, 0.1), glfw.KEY_DOWN: (0, -0.1), glfw.KEY_A: (1, 0.1), glfw.KEY_D: (1, -0.1),
            glfw.KEY_LEFT: (2, 0.2), glfw.KEY_RIGHT: (2, -0.2)}  # fmt: skip
    lo, hi = np.array(CMD_LOW, np.float32), np.array(CMD_HIGH, np.float32)  # float32, as the old tool's torch box

    def key(k):  # the viewer's UI thread; plain float writes under the GIL
        if k in step:
            i, dv = step[k]
            cmd[i] = np.clip(round(cmd[i] + dv, 2), lo[i], hi[i])
        elif k in (glfw.KEY_SPACE, glfw.KEY_ENTER):
            cmd[:] = 0.0
        elif k == glfw.KEY_BACKSPACE:
            cmd[:], reset[0] = 0.0, True

    with mujoco.viewer.launch_passive(vis.m, vis.d, key_callback=key) as v:
        v.cam.type, v.cam.trackbodyid, v.cam.distance = mujoco.mjtCamera.mjCAMERA_TRACKING, vis.root_body, 3.0
        start, tick = time.perf_counter(), 0
        while v.is_running():
            if reset[0] or fallen(sim.state()):
                sim.reset(), pi.reset()
                reset[0] = False
            sim.step(pi(sim.state(), cmd))
            act = heading_velocity(sim.state())
            vis.pose(sim)
            v.set_texts((None, None, "up/down vx | a/d vy | left/right yaw | space stop | backspace reset",
                         f"cmd vx {cmd[0]:+.1f} vy {cmd[1]:+.1f} yaw {cmd[2]:+.1f}   "
                         f"act vx {act[0]:+.2f} vy {act[1]:+.2f} yaw {act[2]:+.2f}"))  # fmt: skip
            v.sync()
            tick += 1
            time.sleep(max(0.0, start + tick * CONTROL_DT - time.perf_counter()))  # wall-clock pacing


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("video", "steptest", "viewer", "landtest", "landvideo"))
    p.add_argument("--onnx", required=True, help="the ONNX export to run (deploy/'s format; landtest: a landing one)")
    p.add_argument("--seed", type=int, default=0, help="actuator lag draws")
    p.add_argument("--model", choices=("newton", "mjcf"), default="newton", help="see T1Mujoco")
    p.add_argument("--fps", type=int, default=50, help="video: a divisor of the 50 Hz control rate")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--grid", type=int, default=15, help="step test: commands per axis and plane (15: the paper's 672)")
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--jobs", type=int, default=12, help="step test, landtest: worker processes")
    p.add_argument("--commands", help="landtest: the commands (.npz), from utils/landing_native.py")
    p.add_argument("--out", help="landtest: the rollouts written (.npz)")
    p.add_argument("--hold_s", type=float, default=3.5, help="landtest: seconds of hold after the last clock")
    p.add_argument("--lag", type=int, help="landtest: this lag [physics steps] on every actuator group instead of a "
                   "draw per command (5: the training simulator's evaluation delay)")  # fmt: skip
    p.add_argument("--walk_onnx", help="landvideo: the velocity export walked with. landtest: a velocity export; each rollout walks and stops under it (the "
                   "commands file's walk) before the landing model takes over")  # fmt: skip
    args = p.parse_args()
    {"video": video, "steptest": step_test, "viewer": viewer, "landtest": land_test, "landvideo": land_video}[args.mode](args)


if __name__ == "__main__":
    main()
