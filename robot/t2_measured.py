"""Booster T2 as a real one was measured: robot/t2.py's model with Booster's T2 walking gain table, Coulomb friction
on every joint and the waist and ankle damping the robot actually applies.

THE NUMBERS BELOW ARE THE ONE PLACE TO EDIT when our own robot's measurements arrive (the gantry check); a run saves
them with its arguments and a checkpoint refuses to load on different ones (robot.spec.check_robot_numbers).

Sources (numbers copied, their code never run):
- gains: booster_deploy 7bb1462 tasks/locomotion/robots/t2/__init__.py, the walking task's table (natural frequency
  10 Hz, damping ratio 1.0 on each motor's armature), the table the measurements below were taken under. Its head and
  wrist values belong to Booster's walking policy; here they are booster_deploy's default table's (18.214394 /
  1.449455), as the third party's wrists.
- friction and the damping factor: github.com/BarryLi159159/T2_supermimic f737ef1 src/supermimic/robots/t2.py
  (JOINT_FRICTION_V4, PARALLEL_KD_FACTOR_V4), measured on a real T2 on 2026-10-28 by their Booster_T2 project. Their
  head is welded: the head joints take the wrists' friction (same B4522 motor).

The robot applies the kd sent to a waist or ankle (linkage) joint on the motor side: the joint-side damping is the
sent kd x the linkage's armature ratio x DAMPING_FACTOR (Booster's rule is 1.0; six axes measured 0.64-0.78). So the
simulated kd of those six joints differs from the kd sent, and ``SPEC.sent_gains`` carries the table to send.
Design: docs/superpowers/specs/2026-11-20-t2-measured-retrain-design.md.
"""

from __future__ import annotations

import dataclasses
import re

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from robot import t2
from robot.t1 import BoosterPdActuatorCfg
from src.dr import DomainRandCfg
from src.sensing import SensingCfg

_ARM_KP = [67.1133, 54.0854, 54.0854, 54.0854, 18.214394, 18.214394, 18.214394]
_ARM_KD = [2.1362, 1.7215, 1.7215, 1.7215, 1.449455, 1.449455, 1.449455]
_LEG_KP = [110.9343, 110.9343, 110.9343, 110.9343, 141.7042, 52.6468]
_LEG_KD = [3.5311, 3.5311, 3.5311, 3.5311, 1.7215, 1.7215]
# joint order (robot.t2.T2_JOINT_NAMES): head 2, left arm 7, right arm 7, waist pitch / roll / yaw, left leg 6, right leg 6
KP_SENT = [18.214394, 18.214394] + _ARM_KP * 2 + [155.9961, 206.9129, 110.9343] + _LEG_KP * 2
KD_SENT = [1.449455, 1.449455] + _ARM_KD * 2 + [1.7215, 1.7215, 3.5311] + _LEG_KD * 2
DAMPING_FACTOR = 0.69
LINKAGE = ("waist_pitch_joint", "waist_roll_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
           "right_ankle_pitch_joint", "right_ankle_roll_joint")  # fmt: skip
# Coulomb friction at the joint [N m], left / right averaged, one measurement per joint (MuJoCo frictionloss)
FRICTION = {
    r".*_hip_yaw_joint": 0.6, r".*_hip_roll_joint": 0.65, r".*_hip_pitch_joint": 0.95, r".*_knee_pitch_joint": 0.72,
    r"waist_yaw_joint": 0.52, r"waist_roll_joint": 3.3, r"waist_pitch_joint": 2.6,
    r".*_ankle_pitch_joint": 2.4, r".*_ankle_roll_joint": 1.4,
    r".*_shoulder_pitch_joint": 0.6, r".*_shoulder_roll_joint": 0.63, r".*_elbow_pitch_joint": 0.47,
    r".*_elbow_yaw_joint": 0.4, r".*_wrist_.*_joint": 0.3, r"aa_head_yaw_joint": 0.3, r"head_pitch_joint": 0.3,
}  # fmt: skip

# The third party's randomization around those numbers (their envs/tracking_env_cfg.py V4_*): ground friction, a
# factor per joint on the friction above, +-10 % on every kp and kd, link masses, mass on the trunk. Encoder offsets,
# the trunk's centre-of-mass offset and the 0-40 ms delay stay robot.t2's (theirs: 0-20 ms around a measured 10 ms)
DR = DomainRandCfg(friction_range=(0.3, 1.6), joint_friction_scale=(0.5, 1.5), gain_scale=0.1,
                   link_mass_scale=(0.95, 1.05), trunk_mass_add=(-5.0, 5.0))  # fmt: skip

# their sensor errors (V4_GYRO_BIAS_RAD_S, V4_IMU_TILT_RAD: 2 deg of mounting error, V4_NOISE_SCALE_TO), ramped in
SENSING = SensingCfg(gyro_bias=0.02, imu_tilt=0.035, noise_scale_to=2.0)

assert len(KP_SENT) == len(KD_SENT) == len(t2.T2_JOINT_NAMES)


def friction_of(name: str) -> float:
    """A joint's measured friction [N m]: exactly one pattern of FRICTION must match."""
    hits = [v for pat, v in FRICTION.items() if re.fullmatch(pat, name)]
    if len(hits) != 1:
        raise KeyError(f"{len(hits)} friction values for joint {name!r}")
    return hits[0]


def _linkage_ratio(name: str) -> float:
    """Armature ratio of a waist or ankle joint's linkage (robot.t2.PARALLEL, Booster's coefficients)."""
    link, axis = t2.JOINT_MOTOR[name.removeprefix("left_").removeprefix("right_").removesuffix("_joint")]
    return t2.PARALLEL[link][3][axis]


def sim_kd(name: str) -> float:
    """The joint-side damping the simulator gives a joint: the kd sent, through the linkage where there is one."""
    kd = KD_SENT[t2.T2_JOINT_NAMES.index(name)]
    return kd * _linkage_ratio(name) * DAMPING_FACTOR if name in LINKAGE else kd


KP = dict(zip(t2.T2_JOINT_NAMES, KP_SENT))
ACTION_SCALE = {n: 0.25 * e / KP[n] for n, (e, _, _, _) in t2.MOTOR.items()}  # Booster's rule, as robot.t2


def get_spec() -> mujoco.MjSpec:
    """robot.t2's model with every hinge joint's measured friction, compiled in: evaluation, the rig and the export
    then run on it without a write, and the randomization scales it per world (the simulator tests each world's
    value when it builds the friction rows)."""
    spec = t2.get_spec()
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            j.frictionloss = friction_of(j.name)
    return spec


def _actuators() -> tuple[BoosterPdActuatorCfg, ...]:
    return tuple(
        BoosterPdActuatorCfg(target_names_expr=(n,), stiffness=KP[n], damping=sim_kd(n), effort_limit=e, v_max=v, v_knee=k)
        for n, (e, v, k, _) in t2.MOTOR.items()
    )


def get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, t2.INIT_Z), joint_pos=t2.DEFAULT_JOINT_POS, joint_vel={".*": 0.0}),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=_actuators(), soft_joint_pos_limit_factor=0.9),
    )


SPEC = dataclasses.replace(
    t2.SPEC,
    name="t2_measured",
    action_scale=ACTION_SCALE,
    robot_cfg=get_robot_cfg,
    sent_gains=(tuple(KP_SENT), tuple(KD_SENT)),
    dr=DR,
    sensing=SENSING,
)
