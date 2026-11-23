"""Passive dry run of the T2 velocity task on the robot: this task's observation and policy on live /low_state data.
It publishes nothing and changes no mode, so the robot stays in whatever mode it is in (Booster's own standing or
walking mode, or damping on a hoist). Run it before the first powered run, from the booster_deploy folder, in the
environment booster_deploy runs in (its .venv, with the robot's ROS 2 sourced):

    python -m tasks.bfm_t2.dry_run

Every half second it prints one block; stop with Ctrl-C for the summary. Let it run for at least a minute with the
robot standing still (the summary fails a shorter run). What to look at:
- gravity (IMU): (0, 0, -1) when the trunk is upright; x grows when the robot is tilted forward; y goes negative when
  its right side goes down. Next to it, with both soles flat on level ground, the gravity the leg joints imply: the
  two should agree within about 2 degrees (training covered a constant IMU tilt up to that).
- gyro: near zero at rest (training covered a bias up to 0.02 rad/s); z > 0 when the robot is turned counter-clockwise.
- joints: move one joint by hand and see that joint, and only it, change, with the expected sign.
- command: (0, 0, 0) with the sticks released. A pad whose yaw axis is a trigger reads as full yaw: fix that first.
- action: the policy's output on this state; compare the standing values with the simulated ones in the README.
- compute: the model's time per step. The summary's bar: 99% of steps under 5 ms (of the 20 ms step).
Every stop a real run would make is printed when it happens: a tilt beyond about 60 degrees (fall), a state or an
IMU that stops updating (stale), a reading that is not a number (invalid). With the robot standing still there must be
no stale or invalid one, and the summary fails if there is: those stops fire on three bit-identical readings of the
whole state in a row, or five of the IMU alone (policy.STALE_STEPS, policy.STALE_IMU_STEPS), which assumes a live
sensor never repeats itself exactly. The summary gives the longest such runs it saw; if this robot's readings do
repeat, that stop would fire by mistake in a powered run and drop a standing robot.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from . import DEFAULT_Q, KD_SENT, KP_SENT, ONNX_NAME, ROBOT, VX_BACKWARD_MAX, VX_FORWARD_MAX, VY_MAX, VYAW_MAX
from .policy import STALE_IMU_STEPS, STALE_STEPS, T2Policy, deadzone, gravity_b, quat_from_euler_xyz

STEP_S, COMPUTE_BAR_MS = 0.02, 5.0
MIN_STEPS = 3000  # a minute: a shorter run says little about how often the readings repeat


class DryRun:
    """The policy on a stream of robot states, with the numbers the summary needs. No ROS in here."""

    def __init__(self, onnx_path: str | None = None):
        path = onnx_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ONNX_NAME)
        self.policy = T2Policy(path, joint_names=ROBOT.joint_names, joint_stiffness=KP_SENT, joint_damping=KD_SENT,
                               default_joint_pos=DEFAULT_Q, policy_dt=STEP_S, prepare_joint_pos=DEFAULT_Q)  # fmt: skip
        self.messages, self.first_msg, self.last_msg = 0, None, None
        self.compute, self.period, self.gyro, self.qd_gap = [], [], [], 0.0
        self.stops = {}  # would-be stops, by reason
        self._q, self._now = None, None

    def on_message(self, now: float) -> None:
        """Count a /low_state message (for its rate)."""
        self.messages += 1
        self.first_msg, self.last_msg = (now if self.first_msg is None else self.first_msg), now

    def on_state(self, q, qd, rpy, gyro, cmd, now: float) -> dict:
        """One policy step on the newest state (call it every 20 ms). rpy, gyro: the IMU's, as /low_state gives them."""
        q, qd, gyro = (np.asarray(x, dtype=np.float64) for x in (q, qd, gyro))
        quat = quat_from_euler_xyz(*(float(x) for x in rpy))
        qd_fd = np.zeros_like(q)
        if self._q is not None and now > self._now:
            qd_fd = (q - self._q) / (now - self._now)
            self.period.append(now - self._now)
            self.qd_gap = max(self.qd_gap, float(np.abs(qd - qd_fd).max()))
        self._q, self._now = q.copy(), now
        self.gyro.append(gyro)
        s = self.policy.step(q, qd, gyro, quat, cmd, now)
        if s.stop:
            self.stops[s.reason] = self.stops.get(s.reason, 0) + 1
            self.policy.reset()
        elif not s.skipped:
            self.compute.append(s.compute_s)
        return {"gravity_b": gravity_b(quat), "gyro": gyro, "dq": q - self.policy.default_q, "qd": qd, "qd_fd": qd_fd,
                "action": (np.asarray(s.target) - self.policy.default_q) / self.policy.action_scale, "cmd": tuple(cmd),
                "compute_s": s.compute_s, "stop": s.reason if s.stop else None}  # fmt: skip

    def summary(self) -> str:
        c = np.array(self.compute) * 1000 if self.compute else np.array([np.nan])
        p = np.array(self.period) * 1000 if self.period else np.array([np.nan])
        span = (self.last_msg - self.first_msg) if self.messages > 1 else 0.0
        rate = (self.messages - 1) / span if span > 0 else float("nan")
        p99 = float(np.percentile(c, 99))
        g = np.mean(self.gyro, 0) if self.gyro else np.full(3, np.nan)
        fall, stale, invalid = (self.stops.get(k, 0) for k in ("fall", "stale", "invalid"))
        why = []
        if not self.compute or not p99 < COMPUTE_BAR_MS:
            why.append(f"99% of steps must compute in under {COMPUTE_BAR_MS:g} ms of the {STEP_S * 1000:g} ms step")
        if stale or invalid:
            why.append("a powered run would have stopped on a frozen or invalid reading and dropped the robot")
        if len(self.compute) < MIN_STEPS:
            why.append(f"too short: {len(self.compute)} policy steps, run at least {MIN_STEPS} ({MIN_STEPS * STEP_S:g} s) with the "
                       "robot standing still")  # fmt: skip
        return "\n".join((
            f"dry run on {self.policy.describe()}",
            f"/low_state at {rate:.0f} Hz ({self.messages} messages); {len(self.compute)} policy steps, {self.policy.skipped} catch-up calls",
            f"would-be stops: fall {fall}, stale {stale}, invalid {invalid}; longest run of identical readings: whole state "
            f"{self.policy.longest_same} (a real run stops at {STALE_STEPS}), IMU alone {self.policy.longest_same_imu} (stops at "
            f"{STALE_IMU_STEPS})",
            f"compute per step: median {np.median(c):.2f} ms, 99% {p99:.2f} ms, worst {c.max():.2f} ms",
            f"step period: median {np.median(p):.1f} ms, 1%-99% {np.percentile(p, 1):.1f}-{np.percentile(p, 99):.1f} ms",
            f"mean gyro {g[0]:+.4f} {g[1]:+.4f} {g[2]:+.4f} rad/s; largest gap between a joint's speed and its finite-differenced "
            f"position {self.qd_gap:.2f} rad/s",
            "FAIL: " + "; ".join(why) if why else
            f"PASS: 99% of steps compute in under {COMPUTE_BAR_MS:g} ms, no frozen or invalid reading in {len(self.compute)} steps",
        ))  # fmt: skip


def leg_gravity(model, q, joint_names, prefix: str = "") -> np.ndarray:
    """The gravity direction in the trunk frame that the leg joints imply when both soles are flat on level ground:
    with the model's joints at q and the trunk unrotated, the mean over the two feet (the bodies of the ankle-roll
    joints) of the foot frame's -z axis. Compare it with the IMU's gravity."""
    import mujoco

    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            a = int(model.jnt_qposadr[j])
            d.qpos[a + 3 : a + 7] = (1.0, 0.0, 0.0, 0.0)
    for n, v in zip(joint_names, q):
        d.qpos[model.jnt_qposadr[model.joint(prefix + n).id]] = v
    mujoco.mj_kinematics(model, d)
    g = np.zeros(3)
    for n in ("left_ankle_roll_joint", "right_ankle_roll_joint"):
        g -= d.xmat[model.jnt_bodyid[model.joint(prefix + n).id]].reshape(3, 3)[:, 2]
    return g / np.linalg.norm(g)


def main() -> None:
    import threading

    import rclpy
    from booster_interface.msg import LowState
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    run = DryRun()
    print(run.policy.describe(), flush=True)
    names, latest, lock = list(ROBOT.joint_names), {}, threading.Lock()

    try:  # the legs' gravity needs mujoco and Booster's T2 model; both come with booster_deploy
        import mujoco
        from booster_assets import BOOSTER_ASSETS_DIR

        model = mujoco.MjModel.from_xml_path(ROBOT.mjcf_path.replace("{BOOSTER_ASSETS_DIR}", str(BOOSTER_ASSETS_DIR)))
    except Exception as e:
        model = None
        print(f"the legs' gravity is unavailable ({e})", flush=True)
    try:  # the command as booster_deploy reads it: normalized sticks, scaled by this task's caps
        from booster_deploy.utils.remote_control_service import RemoteControlService

        rclpy.init()
        remote = RemoteControlService()
    except Exception as e:
        remote = None
        print(f"the remote cannot be read ({e}); the command is taken as zero", flush=True)
        if not rclpy.ok():
            rclpy.init()

    def command():
        if remote is None:
            return (0.0, 0.0, 0.0)
        vx, vy, vyaw = remote.get_vx_cmd(), remote.get_vy_cmd(), remote.get_vyaw_cmd()
        cap = VX_FORWARD_MAX if vx >= 0 else VX_BACKWARD_MAX
        return (deadzone(vx * cap, cap), deadzone(vy * VY_MAX, VY_MAX), deadzone(vyaw * VYAW_MAX, VYAW_MAX))

    def on_low_state(msg):
        now = time.monotonic()
        run.on_message(now)
        with lock:
            latest.update(rpy=np.array(msg.imu_state.rpy, dtype=np.float64), gyro=np.array(msg.imu_state.gyro, dtype=np.float64),
                          q=np.array([m.q for m in msg.motor_state_serial][: len(names)], dtype=np.float64),
                          qd=np.array([m.dq for m in msg.motor_state_serial][: len(names)], dtype=np.float64))  # fmt: skip

    node = rclpy.create_node("bfm_t2_dry_run")  # a subscriber only: nothing is published, no mode is changed
    node.create_subscription(LowState, "/low_state", on_low_state,
                             QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))  # fmt: skip
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=lambda: [executor.spin_once(timeout_sec=0.1) for _ in iter(rclpy.ok, False)], daemon=True).start()

    fmt = lambda x: " ".join(f"{v:+.2f}" for v in x)  # noqa: E731
    next_step, step, stops = time.monotonic(), 0, 0
    print("waiting for /low_state ... (Ctrl-C ends the run and prints the summary)", flush=True)
    try:
        while rclpy.ok():
            if time.monotonic() < next_step:
                time.sleep(0.0005)
                continue
            next_step += STEP_S
            with lock:
                state = dict(latest)
            if not state:
                continue
            r = run.on_state(state["q"], state["qd"], state["rpy"], state["gyro"], command(), time.monotonic())
            step += 1
            if r["stop"] is not None:
                stops += 1
                if stops <= 10 or stops % 100 == 0:  # every one at first, then a reminder
                    print(f"*** step {step}: A REAL RUN WOULD STOP HERE ({r['stop']}); would-be stops so far: {stops}", flush=True)
            if step % 25 == 0:
                legs = "unavailable" if model is None else fmt(leg_gravity(model, state["q"], names))
                print(f"--- step {step}\n"
                      f"gravity (IMU) {fmt(r['gravity_b'])}   from the legs, if both soles are flat: {legs}\n"
                      f"gyro {fmt(r['gyro'])} rad/s   command {fmt(r['cmd'])}   compute {r['compute_s'] * 1000:.2f} ms\n"
                      f"joint - default [rad] {fmt(r['dq'])}\njoint speed [rad/s]   {fmt(r['qd'])}\n"
                      f"action                {fmt(r['action'])}", flush=True)  # fmt: skip
    except KeyboardInterrupt:
        pass
    finally:
        print("\n" + run.summary(), flush=True)
        if remote is not None:
            remote.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
