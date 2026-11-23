"""The T2 velocity policy as it runs on the robot: the ONNX file, its observation and history, the checks that the
task sends what the file was trained with, and the stops. No booster_deploy import: only numpy and onnxruntime, both in
booster_deploy's own environment on the robot's computer (a Jetson Thor), so this repository's tests run the same code
(tests/test_booster_t2.py compares it with utils/play_native.py). The model runs on the CPU, one thread.

Observation per frame (99), as training's and the file's ``obs_terms``: joint positions minus the default pose (31),
joint speeds (31), the trunk's angular velocity in its own frame (3), the gravity direction in the trunk frame,
R^T (0, 0, -1) (3), and the previous action (31). The model takes the last 10 frames, newest last (the first frame
repeated at a start), and the command (vx, vy [m/s] in the heading frame, yaw rate [rad/s]). Joint setpoint =
default pose + action_scale * action.
"""

from __future__ import annotations

import json
import platform
import time
from typing import NamedTuple

import numpy as np
import onnxruntime as ort

# The six waist and ankle joints are each driven by two motors through a linkage. The kd sent reaches those motors;
# at the joint it acts as kd x the linkage's armature ratio (Booster's coefficients) x DAMPING_FACTOR (a real T2 was
# measured to give 0.69 of Booster's rule). The file's ``kd`` is that joint-side value: never the one to send.
LINKAGE_RATIO = {"waist_pitch_joint": 2.884254, "waist_roll_joint": 3.825669, "left_ankle_pitch_joint": 2.620008,
                 "left_ankle_roll_joint": 0.9734019, "right_ankle_pitch_joint": 2.620008, "right_ankle_roll_joint": 0.9734019}  # fmt: skip
DAMPING_FACTOR = 0.69
FALL_GRAVITY_Z = -0.5  # gravity z in the trunk frame above this (about 60 degrees of tilt): fallen (Booster's own rule)
STALE_STEPS = 3  # this many bit-identical sensor states in a row (60 ms): the state has stopped updating
STALE_IMU_STEPS = 5  # this many bit-identical IMU readings in a row (100 ms) while the joints still change: the IMU
# has stopped updating, and with it the fall test below, which reads it (0 turns this stop off)
# Both rest on a live sensor never repeating itself exactly. dry_run.py measures the longest such run on the robot.
MIN_STEP_S = 0.010  # a call sooner than this after the last step is the loop catching up after a stall
OBS_TERMS = [["joint_pos", 31], ["joint_vel", 31], ["ang_vel_b", 3], ["gravity_b", 3], ["prev_action", 31]]
REQUIRED = ("joint_names", "default_q", "action_scale", "control_dt", "obs_terms", "kp", "kd", "kp_sent", "kd_sent")


class Step(NamedTuple):
    target: np.ndarray  # joint setpoints, in the file's joint order
    stop: bool = False  # the task must stop driving the motors (target is then the current pose)
    reason: str | None = None  # "fall", "stale" (the state or the IMU stopped updating) or "invalid" (not a number)
    compute_s: float = 0.0
    skipped: bool = False  # a catch-up call: the previous setpoint again, nothing advanced


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """(w, x, y, z) from the IMU's roll, pitch, yaw: the formula booster_deploy uses on /low_state."""
    cy, sy, cr, sr, cp, sp = np.cos(yaw * 0.5), np.sin(yaw * 0.5), np.cos(roll * 0.5), np.sin(roll * 0.5), np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    return np.array([cy * cr * cp + sy * sr * sp, cy * sr * cp - sy * cr * sp, cy * cr * sp + sy * sr * cp, sy * cr * cp - cy * sr * sp])


def gravity_b(quat_wxyz) -> np.ndarray:
    """The gravity direction in the trunk frame, R^T (0, 0, -1): (0, 0, -1) upright, x > 0 pitched forward."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64) / np.linalg.norm(quat_wxyz)
    return np.array([-2.0 * (x * z - w * y), -2.0 * (y * z + w * x), -(1.0 - 2.0 * (x * x + y * y))])


def deadzone(x: float, cap: float, dz: float = 0.1) -> float:
    """A stick within dz of its cap's range of centre gives exactly zero (training's standing command)."""
    return 0.0 if abs(x) < dz * cap else float(x)


def _first_diff(a, b, rtol: float):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return 0
    bad = np.flatnonzero(np.abs(a - b) > rtol * np.maximum(np.abs(b), 1e-9))
    return int(bad[0]) if len(bad) else None


def check_metadata(meta: dict, *, joint_names, joint_stiffness, joint_damping, default_joint_pos, policy_dt,
                   prepare_joint_pos=None) -> None:  # fmt: skip
    """Raise ValueError unless what the task is about to send and assume is what the file was trained with: joint
    order, default pose (and the prepare pose, when given), control rate, observation layout, and the gains sent."""
    for key in REQUIRED:
        if key not in meta:
            raise ValueError(f"the ONNX file records no {key}: not an export this task can run (utils/export_onnx.py of a t2_measured policy)")
    names = list(meta["joint_names"])
    if list(joint_names) != names:
        raise ValueError("joint_names: the task's joint order differs from the file's")
    for label, pose in (("default_joint_pos", default_joint_pos), ("prepare_state.joint_pos (the prepare pose)", prepare_joint_pos)):
        if pose is not None and (i := _first_diff(pose, meta["default_q"], 1e-6)) is not None:
            raise ValueError(f"{label} differs from the file's default_q at {names[i]}")
    if abs(float(policy_dt) - float(meta["control_dt"])) > 1e-9:
        raise ValueError(f"control_dt: the file was trained at {meta['control_dt']} s per step, the task runs at {policy_dt} s")
    if [list(t) for t in meta["obs_terms"]] != OBS_TERMS:
        raise ValueError(f"obs_terms: the file's observation layout {meta['obs_terms']} is not the one this policy code builds")
    if len(meta["action_scale"]) != len(names) or min(meta["action_scale"]) <= 0:
        raise ValueError("action_scale: the file's action scale is not one positive number per joint")
    for label, sent, key in (("joint_stiffness", joint_stiffness, "kp_sent"), ("joint_damping", joint_damping, "kd_sent")):
        if (i := _first_diff(sent, meta[key], 1e-6)) is not None:
            raise ValueError(f"{label} differs from the file's {key} at {names[i]}: the task would send {list(sent)[i]:g}, the file was "
                             f"trained for {meta[key][i]:g} (the file's kd is the joint-side damping in training, never the value to send)")  # fmt: skip
    if (i := _first_diff(meta["kp"], meta["kp_sent"], 1e-6)) is not None:
        raise ValueError(f"kp: the file's training kp differs from its kp_sent at {names[i]}")
    joint_kd = [kd * LINKAGE_RATIO[n] * DAMPING_FACTOR if n in LINKAGE_RATIO else kd for n, kd in zip(names, meta["kd_sent"])]
    if (i := _first_diff(meta["kd"], joint_kd, 1e-4)) is not None:
        raise ValueError(f"kd: the file's training kd at {names[i]} ({meta['kd'][i]:g}) is not kd_sent (x the linkage ratio x "
                         f"{DAMPING_FACTOR} on the waist and ankles: {joint_kd[i]:g}): another robot model than this task assumes")  # fmt: skip


class T2Policy:
    def __init__(self, onnx_path, *, joint_names, joint_stiffness, joint_damping, default_joint_pos, policy_dt: float = 0.02,
                 prepare_joint_pos=None, prev_action: str = "hold"):  # fmt: skip
        """prev_action: what the model sees as its previous action at the first step: "hold", the action whose
        setpoint is the pose the robot starts in (as most of training's resets and Booster's own policy), or "zeros"."""
        if prev_action not in ("hold", "zeros"):
            raise ValueError(f"prev_action {prev_action!r}: hold or zeros")
        self.path, self.prev_action = str(onnx_path), prev_action
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")  # no busy-waiting beside the robot's own services
        try:
            self.session = ort.InferenceSession(self.path, opts, providers=["CPUExecutionProvider"])
        except Exception as e:
            raise RuntimeError(f"onnxruntime {ort.__version__} cannot load {self.path}: {e}") from e
        meta = {}
        for k, v in self.session.get_modelmeta().custom_metadata_map.items():
            try:
                meta[k] = json.loads(v)
            except ValueError:
                meta[k] = v
        check_metadata(meta, joint_names=joint_names, joint_stiffness=joint_stiffness, joint_damping=joint_damping,
                       default_joint_pos=default_joint_pos, policy_dt=policy_dt, prepare_joint_pos=prepare_joint_pos)  # fmt: skip
        shapes = {i.name: list(i.shape) for i in self.session.get_inputs()}
        n_obs = sum(n for _, n in OBS_TERMS)
        if list(shapes) != ["obs_hist", "cmd"] or shapes["obs_hist"][::2] != [1, n_obs] or shapes["cmd"] != [1, 3]:
            raise ValueError(f"the file's inputs {shapes} are not obs_hist (1, history, {n_obs}) and cmd (1, 3)")
        self.hist_len = int(shapes["obs_hist"][1])
        self.joint_names = list(meta["joint_names"])
        self.default_q = np.asarray(meta["default_q"], dtype=np.float64)
        self.action_scale = np.asarray(meta["action_scale"], dtype=np.float64)
        self.cmd_low, self.cmd_high = (np.asarray(meta.get(k, [np.nan] * 3), dtype=np.float64) for k in ("cmd_low", "cmd_high"))
        self.iteration = meta.get("iteration")
        self.session.run(None, {"obs_hist": np.zeros((1, self.hist_len, n_obs), np.float32), "cmd": np.zeros((1, 3), np.float32)})  # warm-up
        self.longest_same = self.longest_same_imu = 0  # the longest runs of identical readings seen (kept over resets)
        self.reset()

    def describe(self) -> str:
        return (f"{platform.machine()}, Python {platform.python_version()}, numpy {np.__version__}, onnxruntime {ort.__version__}; "
                f"{self.path}, iteration {self.iteration}, previous action at start: {self.prev_action}")  # fmt: skip

    def reset(self) -> None:
        self.hist, self._prev = None, None  # the (history, 99) frames, newest last; the previous action
        self._last_now, self._last_target = None, None
        self._last_key, self._same, self._last_imu, self._same_imu = None, 0, None, 0
        self.skipped = 0  # catch-up calls since the reset
        self.bad_commands = 0  # commands that were not numbers since the reset (taken as zero)

    def step(self, q, qd, gyro, quat_wxyz, cmd, now: float) -> Step:
        """One control step from the newest robot state. q, qd: joint positions and speeds in the file's joint order;
        gyro: the trunk's angular velocity in its own frame; quat_wxyz: the trunk's attitude; cmd: (vx, vy, yaw rate);
        now: a monotonic clock, seconds."""
        if self._last_now is not None and now - self._last_now < MIN_STEP_S:
            # Booster's loop runs missed steps back to back after a stall. Training's frames are always 20 ms apart,
            # so those calls repeat the last setpoint and leave the history alone.
            self.skipped += 1
            return Step(self._last_target, skipped=True)
        q, qd, gyro, quat = (np.asarray(x, dtype=np.float64) for x in (q, qd, gyro, quat_wxyz))
        if not all(np.isfinite(x).all() for x in (q, qd, gyro, quat)) or np.linalg.norm(quat) < 0.5:
            return Step(np.where(np.isfinite(q), q, self.default_q), stop=True, reason="invalid")
        imu = (gyro.tobytes(), quat.tobytes())
        key = (q.tobytes(), qd.tobytes(), *imu)
        self._same = self._same + 1 if key == self._last_key else 1
        self._same_imu = self._same_imu + 1 if imu == self._last_imu else 1
        self._last_key, self._last_imu, self._last_now = key, imu, now
        self.longest_same, self.longest_same_imu = max(self.longest_same, self._same), max(self.longest_same_imu, self._same_imu)
        if self._same >= STALE_STEPS or (STALE_IMU_STEPS and self._same_imu >= STALE_IMU_STEPS):
            return Step(q.copy(), stop=True, reason="stale")  # a live sensor is never this still: it has stopped arriving
        g = gravity_b(quat)
        if g[2] > FALL_GRAVITY_Z:
            return Step(q.copy(), stop=True, reason="fall")
        t0 = time.perf_counter()
        cmd = np.asarray(cmd, dtype=np.float64).reshape(3)
        if not np.isfinite(cmd).all():  # a broken remote: stand still
            self.bad_commands += 1
            cmd = np.zeros(3)
        if np.isfinite(self.cmd_low).all() and np.isfinite(self.cmd_high).all():
            cmd = np.clip(cmd, self.cmd_low, self.cmd_high)  # the model was never trained outside its command box
        if self.hist is None:
            hold = (q - self.default_q) / self.action_scale
            self._prev = (hold if self.prev_action == "hold" else np.zeros_like(hold)).astype(np.float32)
        o = np.concatenate((q - self.default_q, qd, gyro, g, self._prev)).astype(np.float32)
        self.hist = np.repeat(o[None], self.hist_len, 0) if self.hist is None else np.concatenate((self.hist[1:], o[None]))
        (a,) = self.session.run(None, {"obs_hist": self.hist[None], "cmd": np.asarray([cmd], dtype=np.float32)})
        if not np.isfinite(a).all():
            return Step(q.copy(), stop=True, reason="invalid")
        self._prev = a[0].astype(np.float32)
        self._last_target = self.default_q + self.action_scale * self._prev
        return Step(self._last_target, compute_s=time.perf_counter() - t0)
