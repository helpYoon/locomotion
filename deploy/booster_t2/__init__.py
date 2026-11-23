"""The T2 velocity policy (this repository's robot ``t2_measured``, run ``t2al1_vel_s1``) as a booster_deploy task.

Install on the robot's computer: copy this folder to ``<booster_deploy>/tasks/bfm_t2/`` and the ONNX file to
``<booster_deploy>/tasks/bfm_t2/models/t2al1_vel_s1_013000.onnx``. Then, from the booster_deploy folder:

    python -m tasks.bfm_t2.dry_run                 # passive: reads /low_state, sends nothing (do this first)
    python scripts/deploy.py --task bfm_t2_vel     # the robot: X = Custom mode and the prepare pose, A = start
    python scripts/deploy.py --task bfm_t2_vel --mujoco

What this task sets, and why (the repository's README, "Deploy on the robot"):
- Gains sent: KP_SENT / KD_SENT, the table the policy was trained for (Booster's T2 walking table). They must equal
  the ``kp_sent`` / ``kd_sent`` recorded in the ONNX file; the file's ``kd`` is training's joint-side damping on the
  waist and ankles and is never sent.
- Default pose and prepare pose: the policy's own (a slight crouch), so "standing" preparation ends where it starts.
- Stops: a fall (tilt beyond about 60 degrees), a robot state or an IMU that stops updating, or a reading that is not
  a number. The stiffness is zeroed first, so the last command the motors keep is damping only. A command that is not
  a number is taken as zero, and a command outside the trained box is held at its edge.
- Exit: Booster switches the robot to one mode at the end of every run. Here that is Booster's walking mode after a
  normal stop (Ctrl-C, a cancelled start) and damping after one of the stops above; see ExitMode.
- The remote's caps are first-session values well inside the trained command box.
Everything is checked against the ONNX file when this module is imported (before the robot is put in Custom mode) and
again when the policy is built; a mismatch raises. So does starting this task without the file, and a booster_deploy
whose controller lacks a member this task reads (it was written against booster_deploy 7bb1462).
"""

from __future__ import annotations

import importlib.util
import multiprocessing
import os
import sys
import time

import numpy as np
import torch
from booster_deploy.controllers import base_controller
from booster_deploy.controllers.base_controller import Policy
from booster_deploy.controllers.controller_cfg import (
    BoosterRobotControllerCfg,
    ControllerCfg,
    MujocoControllerCfg,
    PolicyCfg,
    PrepareStateCfg,
    VelocityCommandCfg,
)
from booster_deploy.robots.t2 import T2_31DOF_CFG
from booster_deploy.utils.registry import register_task

from .policy import T2Policy, deadzone

TASK_NAME = "bfm_t2_vel"
ONNX_NAME = "models/t2al1_vel_s1_013000.onnx"
# Joint order (31): head yaw, pitch | left arm: shoulder pitch, roll, elbow pitch, yaw, wrist pitch, yaw, roll | right
# arm | waist pitch, roll, yaw | left leg: hip pitch, roll, yaw, knee, ankle pitch, roll | right leg
_ARM_KP = [67.1133, 54.0854, 54.0854, 54.0854, 18.214394, 18.214394, 18.214394]
_ARM_KD = [2.1362, 1.7215, 1.7215, 1.7215, 1.449455, 1.449455, 1.449455]
_LEG_KP = [110.9343, 110.9343, 110.9343, 110.9343, 141.7042, 52.6468]
_LEG_KD = [3.5311, 3.5311, 3.5311, 3.5311, 1.7215, 1.7215]
KP_SENT = [18.214394, 18.214394] + _ARM_KP * 2 + [155.9961, 206.9129, 110.9343] + _LEG_KP * 2
KD_SENT = [1.449455, 1.449455] + _ARM_KD * 2 + [1.7215, 1.7215, 3.5311] + _LEG_KD * 2
_ARM_Q = [0.2, -1.3, 0.0, -0.5, 0.0, 0.0, 0.0]
_LEG_Q = [-0.2, 0.0, 0.0, 0.4, -0.2, 0.0]
DEFAULT_Q = [0.0, 0.0] + _ARM_Q + [0.2, 1.3, 0.0, 0.5, 0.0, 0.0, 0.0] + [0.0, 0.0, 0.0] + _LEG_Q * 2
# First-session caps on the remote [m/s, m/s, rad/s]; the trained box is vx +1.90 / -1.39, vy +-1.14, yaw +-2.78.
VX_FORWARD_MAX, VX_BACKWARD_MAX, VY_MAX, VYAW_MAX = 0.5, 0.3, 0.3, 1.0
PERIOD_WARN_S = 0.025  # a loop period above this (20 ms expected) is reported

# ---- the exit mode, split by how the run ended
# Booster reads cfg.booster.exit_mode once, in its main process, when a run ends for any reason, as
# ``cfg.booster.exit_mode.strip().lower()``, and switches the robot to that mode. One setting cannot tell a normal stop
# (the robot stands: hand it to Booster's walking controller) from a fall (damping). So the setting is a str whose
# strip() answers by a flag the policy sets before it stops the controller. The flag is made here, at import in the
# main process, and Booster's policy process is a fork of it, so both see the same one (as with Booster's own events).
_FELL = multiprocessing.Event()


class ExitMode(str):
    """"damping" once the policy has flagged a fall or a dead sensor in this run, else "walking". Its plain value is
    "damping": what a booster_deploy that stopped calling strip() would get (check_exit_hook refuses that at import)."""

    def strip(self, chars=None) -> str:
        return "damping" if _FELL.is_set() else "walking"


def check_exit_hook() -> None:
    """Refuse to load on a booster_deploy that does not read the exit mode through .strip(): there the split above
    would silently not happen. Reads the source as text (importing that module needs ROS 2)."""
    spec = importlib.util.find_spec("booster_deploy.controllers.booster_robot_controller")
    text = open(spec.origin).read() if spec is not None and spec.origin else ""
    if "booster.exit_mode.strip()" not in text:
        raise ValueError("exit_mode: this booster_deploy does not read cfg.booster.exit_mode with .strip() in "
                         "controllers/booster_robot_controller.py, so this task's exit (walking after a normal stop, damping "
                         "after a fall) would not work as designed; see ExitMode in tasks/bfm_t2/__init__.py")  # fmt: skip


class BfmT2Policy(Policy):
    clock = staticmethod(time.monotonic)  # a class attribute so a test can step time

    def __init__(self, cfg: PolicyCfg, controller):
        super().__init__(cfg, controller)
        r = controller.robot.cfg
        path = cfg.checkpoint_path if os.path.isabs(cfg.checkpoint_path) else os.path.join(self.task_path, cfg.checkpoint_path)
        self.policy = T2Policy(path, joint_names=r.joint_names, joint_stiffness=r.joint_stiffness, joint_damping=r.joint_damping,
                               default_joint_pos=r.default_joint_pos, policy_dt=controller.cfg.policy_dt,
                               prepare_joint_pos=r.prepare_state.joint_pos)  # fmt: skip
        print(f"[bfm_t2] {self.policy.describe()}", flush=True)
        self.reset()

    def reset(self) -> None:
        self.policy.reset()
        _FELL.clear()  # a new run
        self._last, self._warned, self._compute, self._period = None, -1e9, [], []

    def inference(self) -> torch.Tensor:
        """One control step: joint position targets in the robot's joint order."""
        ctl = self.controller
        d, vc = ctl.robot.data, ctl.vel_command
        # the command arrives in m/s and rad/s (stick x cap); a released stick must be exactly zero
        cmd = (0.0, 0.0, 0.0) if vc is None else (
            deadzone(vc.lin_vel_x, vc.vx_forward_max if vc.lin_vel_x >= 0 else vc.vx_backward_max),
            deadzone(vc.lin_vel_y, vc.vy_max), deadzone(vc.ang_vel_yaw, vc.vyaw_max))  # fmt: skip
        now = self.clock()
        s = self.policy.step(d.joint_pos.cpu().numpy(), d.joint_vel.cpu().numpy(), d.root_ang_vel_b.cpu().numpy(),
                             d.root_quat_w.cpu().numpy(), cmd, now)  # fmt: skip
        if s.stop:
            # The controller reads joint_stiffness every step and sends this step's command before it stops, so the
            # command the motors keep until Booster switches mode is: hold where you are, damping only.
            ctl.robot.joint_stiffness.zero_()
            _FELL.set()
            why = {"fall": "the robot fell (tilt beyond about 60 degrees)", "stale": "the robot state or its IMU stopped updating",
                   "invalid": "a reading or the model's output is not a number"}[s.reason]  # fmt: skip
            print(f"\n[bfm_t2] STOP: {why}. Stiffness zeroed; exit mode: damping.", flush=True)
            ctl.stop()
        elif s.skipped:
            if now - self._warned > 1.0:
                self._warned = now
                print(f"[bfm_t2] the loop is catching up after a stall: {self.policy.skipped} calls skipped so far", flush=True)
        else:
            self._note(now, s.compute_s)
            if not np.isfinite(cmd).all() and now - self._warned > 1.0:
                self._warned = now
                print(f"[bfm_t2] the command {cmd} is not a number: taken as zero", flush=True)
        return torch.from_numpy(np.asarray(s.target, dtype=np.float32))

    def _note(self, now: float, compute_s: float) -> None:
        """Compute time and loop period: a line every 5 s, and a warning (at most one a second) on a late step."""
        if self._last is not None:
            self._period.append(now - self._last)
            if now - self._last > PERIOD_WARN_S and now - self._warned > 1.0:
                self._warned = now
                print(f"[bfm_t2] late step: {(now - self._last) * 1000:.1f} ms since the last one (20 ms expected)", flush=True)
        self._last = now
        self._compute.append(compute_s)
        if len(self._compute) >= 250:
            c, p = np.array(self._compute) * 1000, np.array(self._period) * 1000
            print(f"[bfm_t2] compute median {np.median(c):.2f} ms, worst {c.max():.2f} ms; period {p.min():.1f}-{p.max():.1f} ms; "
                  f"{self.policy.skipped} catch-up calls skipped", flush=True)  # fmt: skip
            self._compute, self._period = [], []


ROBOT = T2_31DOF_CFG.replace(
    prepare_mode="standing",  # X: hold, Custom mode, 1 s into the prepare pose; A: the policy starts from its own pose
    joint_stiffness=KP_SENT,
    joint_damping=KD_SENT,
    default_joint_pos=DEFAULT_Q,
    prepare_state=PrepareStateCfg(stiffness=T2_31DOF_CFG.prepare_state.stiffness, damping=T2_31DOF_CFG.prepare_state.damping,
                                  joint_pos=DEFAULT_Q),  # fmt: skip
)
CFG = ControllerCfg(
    policy_dt=0.02,
    robot=ROBOT,
    vel_command=VelocityCommandCfg(vx_max=VX_FORWARD_MAX, vx_forward_max=VX_FORWARD_MAX, vx_backward_max=VX_BACKWARD_MAX,
                                   vy_max=VY_MAX, vyaw_max=VYAW_MAX),  # fmt: skip
    policy=PolicyCfg(constructor=BfmT2Policy, checkpoint_path=ONNX_NAME),
    mujoco=MujocoControllerCfg(init_pos=[0.0, 0.0, 0.985]),  # trunk height with the soles on the floor in the default pose
    booster=BoosterRobotControllerCfg(exit_mode=ExitMode("damping")),
)


def check_interface() -> None:
    """Refuse to load on a booster_deploy whose controller lacks a member this task reads while running. (The classes
    and config fields used above already fail at import when renamed; these would only fail at the first step, with
    the robot in Custom mode. On the robot there is no test suite to catch a different booster_deploy.)"""
    need = {base_controller.RobotData: ("joint_pos", "joint_vel", "root_ang_vel_b", "root_quat_w"),
            base_controller.BoosterRobot: ("cfg", "data", "joint_stiffness", "joint_damping"),
            base_controller.BaseController: ("cfg", "robot", "vel_command", "policy")}  # fmt: skip
    missing = [f"{c.__name__}.{n}" for c, names in need.items() for n in names if n not in getattr(c, "__annotations__", {})]
    missing += [] if hasattr(base_controller.BaseController, "stop") else ["BaseController.stop"]
    command = base_controller.VelocityCommand(CFG.vel_command)
    missing += [f"VelocityCommand.{n}" for n in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw", "vx_forward_max", "vx_backward_max",
                                                 "vy_max", "vyaw_max") if not hasattr(command, n)]  # fmt: skip
    if missing:
        raise ValueError(f"this booster_deploy's controllers/base_controller.py has no {', '.join(missing)}: the task {TASK_NAME} "
                         "was written against booster_deploy 7bb1462 and would fail at its first step here")  # fmt: skip


def check_file() -> None:
    """Everything this task sends and assumes against what the ONNX file records. Without the file nothing can be
    checked: starting this task then stops here, before Custom mode; any other start only gets a warning."""
    path = os.path.join(os.path.dirname(__file__), ONNX_NAME)
    if not os.path.exists(path):
        message = f"{path} is missing, so nothing this task sends was checked against the policy file"
        if any(TASK_NAME in arg for arg in sys.argv):
            raise FileNotFoundError(f"{message}; copy the ONNX file there before starting {TASK_NAME}")
        print(f"[bfm_t2] WARNING: {message}. Do not start {TASK_NAME} until it is there.", flush=True)
        return
    T2Policy(path, joint_names=ROBOT.joint_names, joint_stiffness=ROBOT.joint_stiffness, joint_damping=ROBOT.joint_damping,
             default_joint_pos=ROBOT.default_joint_pos, policy_dt=CFG.policy_dt, prepare_joint_pos=ROBOT.prepare_state.joint_pos)  # fmt: skip
    print(f"[bfm_t2] checked against {path}: joint order, poses, rate, observation layout and gains agree", flush=True)


check_exit_hook()
check_interface()
check_file()
register_task(TASK_NAME, CFG)
