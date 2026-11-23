"""Booster T1 (29 DoF) as the old SuperMimic code simulated it (its robot/t1.py, robot/actuator.py and src/env.py, at
tag supermimic-frozen-2026-11-15).

Explicit Booster PD (4 Hz natural frequency, damping ratio 1.5 on the armature) with the torque-speed clip, the
sysid armature / viscous damping / Coulomb friction of the MJCF, self-collisions on, and per-group position-target
delays of U{2..8} physics steps (applied by TrackingSim from ``delay_groups``, not by mjlab). Per-joint gains,
limits, action scales and the default pose come from the table that code exported, assets/t1/mujoco/t1_actuators.npz.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.actuator import IdealPdActuator, IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from robot.spec import RobotSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_PATH = os.path.join(ROOT, "assets", "t1", "t1_description", "mjcf", "t1.xml")
_TAB = dict(np.load(os.path.join(ROOT, "assets", "t1", "mujoco", "t1_actuators.npz")))

T1_JOINT_NAMES = [
    "AAHead_yaw", "Head_pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw", "Left_Wrist_Pitch",
    "Left_Wrist_Yaw", "Left_Hand_Roll",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw", "Right_Wrist_Pitch",
    "Right_Wrist_Yaw", "Right_Hand_Roll",
    "Waist",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]  # fmt: skip

TRACKED_BODIES = [
    "Trunk", "H2", "AL2", "AL3", "AL4", "left_hand_link", "AR2", "AR3", "AR4", "right_hand_link",
    "Hip_Roll_Left", "Hip_Yaw_Left", "Shank_Left", "Ankle_Cross_Left", "left_foot_link",
    "Hip_Roll_Right", "Hip_Yaw_Right", "Shank_Right", "Ankle_Cross_Right", "right_foot_link",
]  # fmt: skip

# left-right joint mirror for src/mirror.py: each joint's partner and sign (the old SuperMimic code's tables)
MIRROR_PERM = (0, 1, 9, 10, 11, 12, 13, 14, 15, 2, 3, 4, 5, 6, 7, 8, 16, 23, 24, 25, 26, 27, 28, 17, 18, 19, 20, 21, 22)
MIRROR_SIGN = (-1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1)


@dataclass(kw_only=True)
class BoosterPdActuatorCfg(IdealPdActuatorCfg):
    """Ideal PD whose torque limit falls linearly from effort_limit at |qd| = v_knee to 0 at |qd| = v_max."""

    v_max: float
    v_knee: float

    def build(self, entity, target_ids, target_names):
        return BoosterPdActuator(self, entity, target_ids, target_names)


class BoosterPdActuator(IdealPdActuator):
    # inherits IdealPdActuator.compute, so mjlab fuses all of them into one group
    param_names = ("stiffness", "damping", "force_limit", "v_max", "denom")

    @staticmethod
    def control_law(params, cmd):
        torque = params["stiffness"] * (cmd.position_target - cmd.pos) + params["damping"] * (cmd.velocity_target - cmd.vel)
        torque = torque + cmd.effort_target
        tau_max = params["force_limit"]
        lim = (tau_max * (params["v_max"] - cmd.vel.abs()) / params["denom"]).clamp(min=0.0).clamp(max=tau_max)
        return torch.clip(torque, -lim, lim)

    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        n = data.nworld
        self.v_max = torch.full((n, len(self._target_names)), self.cfg.v_max, device=device)
        self.denom = torch.full_like(self.v_max, max(self.cfg.v_max - min(self.cfg.v_knee, self.cfg.v_max), 1e-6))


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(MJCF_PATH)
    for a in list(spec.actuators):  # torque <general>s: mjlab adds its own motors
        spec.delete(a)
    for k in list(spec.keys):
        spec.delete(k)
    for s in list(spec.sensors):
        spec.delete(s)
    for g in list(spec.geoms):
        if g.name == "ground" or (g.contype == 0 and g.conaffinity == 0):  # mjlab adds the terrain; visuals unused
            spec.delete(g)
    for m in list(spec.meshes):
        spec.delete(m)
    for g in spec.geoms:
        g.priority = 1  # robot friction (the sim context) wins over the terrain's
    return spec


def _actuators() -> tuple[BoosterPdActuatorCfg, ...]:
    t = _TAB
    return tuple(
        BoosterPdActuatorCfg(target_names_expr=(n,), stiffness=float(t["kp"][i]), damping=float(t["kd"][i]),
                             effort_limit=float(t["tau_max"][i]), v_max=float(t["v_max"][i]), v_knee=float(t["v_knee"][i]))
        for i, n in enumerate(T1_JOINT_NAMES)
    )  # fmt: skip


DEFAULT_JOINT_POS = {n: float(v) for n, v in zip(T1_JOINT_NAMES, _TAB["default_q"]) if v != 0.0}


def get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.688), joint_pos=DEFAULT_JOINT_POS, joint_vel={".*": 0.0}),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=_actuators(), soft_joint_pos_limit_factor=0.9),
    )


# groups in bfm_wm's xi order: arms, waist, legs, feet, head
_G = _TAB["group"]
assert (_TAB["min_delay"] == 2).all() and (_TAB["max_delay"] == 8).all()

SPEC = RobotSpec(
    name="t1",
    mjcf_path=MJCF_PATH,
    joint_names=tuple(T1_JOINT_NAMES),
    tracked_bodies=tuple(TRACKED_BODIES),
    default_joint_pos=DEFAULT_JOINT_POS,
    action_scale={n: float(v) for n, v in zip(T1_JOINT_NAMES, _TAB["action_scale"])},
    com_body="Trunk",
    com_range=((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05)),
    robot_cfg=get_robot_cfg,
    feet=("left_foot_link", "right_foot_link"),
    contact_z=0.0533,
    mirror_perm=MIRROR_PERM,
    mirror_sign=MIRROR_SIGN,
    delay_groups=tuple(tuple(int(j) for j in np.nonzero(_G == g)[0]) for g in range(5)),
    delay_range=(2, 8),
    eval_delay=5,
)
