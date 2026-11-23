"""Booster T2 (31 DoF) with Booster's Revo2 hands locked open, simulated with Booster's T2 recipe (booster_train
83037fd assets/robots/{actuator.py, booster_t2.py}; booster_deploy 7bb1462 robots/t2.py).

Explicit PD at 8 Hz natural frequency and damping ratio 2.0 on each joint's (reflected) armature, with the torque-speed
clip of robot.t1's BoosterPdActuator; action scale 0.25 E / Kp; self-collisions on; per-group position-target delays of
U{0..8} physics steps (applied by TrackingSim from ``delay_groups``, not by mjlab). The model is
utils/build_t2_mjcf.py's t2.xml: Booster's Revo2-hand MJCF with the fingers welded open, one box per hand, and the base
T2's ankle pitch range. Default pose: the T1-style crouch (hip pitch -0.2, knee 0.4, ankle pitch -0.2) with Booster's
arms.
"""

from __future__ import annotations

import math
import os

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from robot.spec import RobotSpec
from robot.t1 import BoosterPdActuatorCfg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_PATH = os.path.join(ROOT, "assets", "t2", "t2_description", "mjcf", "t2.xml")

T2_JOINT_NAMES = [
    "aa_head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint",
    "waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_pitch_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_pitch_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
]  # fmt: skip

# the T1's 20 tracked bodies mapped by role (robot/t1.py TRACKED_BODIES)
TRACKED_BODIES = [
    "trunk", "head_pitch_link",
    "left_shoulder_roll_link", "left_elbow_pitch_link", "left_elbow_yaw_link", "left_wrist_roll_link",
    "right_shoulder_roll_link", "right_elbow_pitch_link", "right_elbow_yaw_link", "right_wrist_roll_link",
    "left_hip_roll_link", "left_hip_yaw_link", "left_knee_pitch_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_hip_yaw_link", "right_knee_pitch_link", "right_ankle_pitch_link", "right_ankle_roll_link",
]  # fmt: skip

# left-right joint mirror for src/mirror.py: the T1's tables with the waist pitch (+1) and roll (-1) inserted
MIRROR_PERM = (0, 1, 9, 10, 11, 12, 13, 14, 15, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18,
               25, 26, 27, 28, 29, 30, 19, 20, 21, 22, 23, 24)  # fmt: skip
MIRROR_SIGN = (-1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1,
               1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1)  # fmt: skip

# Booster's T2 motors (actuator.py:421-467): effort [N m], v_max and v_knee [rpm], rotor armature [kg m^2]
MOTORS = {"B4522": (22.0, 177, 122, 0.007209), "B6030": (74.0, 224, 110, 0.017),
          "B6025": (60.0, 227, 123, 0.0137), "B7629": (135.0, 200, 59, 0.0281)}  # fmt: skip
# the parallel waist and ankles on a B6025 (actuator.py:470-491): effort, v_max, v_knee and armature ratios of the
# serial (pitch, roll) joint
PARALLEL = {"waist": ((2.3, 2.4), (0.5, 0.4), (0.5, 0.4), (2.884254, 3.825669)),
            "ankle": ((2.2, 1.2), (0.7, 1.0), (0.7, 1.0), (2.620008, 0.9734019))}  # fmt: skip
# joint (side and "_joint" dropped) -> motor, or (linkage, 0 = pitch / 1 = roll) (booster_t2.py:44-130)
JOINT_MOTOR = {
    "aa_head_yaw": "B4522", "head_pitch": "B4522",
    "shoulder_pitch": "B6030", "shoulder_roll": "B6025", "elbow_pitch": "B6025", "elbow_yaw": "B6025",
    "wrist_pitch": "B4522", "wrist_yaw": "B4522", "wrist_roll": "B4522",
    "waist_pitch": ("waist", 0), "waist_roll": ("waist", 1), "waist_yaw": "B7629",
    "hip_pitch": "B7629", "hip_roll": "B7629", "hip_yaw": "B7629", "knee_pitch": "B7629",
    "ankle_pitch": ("ankle", 0), "ankle_roll": ("ankle", 1),
}  # fmt: skip
OMEGA, ZETA = 2 * math.pi * 8.0, 2.0  # Booster's T2 natural frequency (8 Hz) and damping ratio (actuator.py:150-165)


def _motor(name: str) -> tuple[float, float, float, float]:
    """Effort [N m], v_max, v_knee [rad/s] and armature [kg m^2] of a joint (ParallelJointWrapperCfg.__post_init__)."""
    m = JOINT_MOTOR[name.removeprefix("left_").removeprefix("right_").removesuffix("_joint")]
    rpm = 2 * math.pi / 60
    if isinstance(m, str):
        e, v, k, a = MOTORS[m]
        return e, v * rpm, k * rpm, a
    (er, vr, kr, ar), i = PARALLEL[m[0]], m[1]
    e, v, k, a = MOTORS["B6025"]
    return er[i] * e, vr[i] * v * rpm, kr[i] * k * rpm, ar[i] * a


MOTOR = {n: _motor(n) for n in T2_JOINT_NAMES}
KP = {n: a * OMEGA**2 for n, (_, _, _, a) in MOTOR.items()}
KD = {n: 2 * ZETA * a * OMEGA for n, (_, _, _, a) in MOTOR.items()}
ACTION_SCALE = {n: 0.25 * e / KP[n] for n, (e, _, _, _) in MOTOR.items()}  # booster_t2.py:133-144

# Booster's arms (booster_t2.py:34-40) and the T1-style crouch (the user's choice over Booster's straight legs)
DEFAULT_JOINT_POS = {
    "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": -1.3, "left_elbow_yaw_joint": -0.5,
    "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": 1.3, "right_elbow_yaw_joint": 0.5,
    "left_hip_pitch_joint": -0.2, "left_knee_pitch_joint": 0.4, "left_ankle_pitch_joint": -0.2,
    "right_hip_pitch_joint": -0.2, "right_knee_pitch_joint": 0.4, "right_ankle_pitch_joint": -0.2,
}  # fmt: skip
INIT_Z = 0.985  # trunk height standing in DEFAULT_JOINT_POS, lowest sole corner on z = 0 (0.9757), + 9 mm as the T1's 0.688

# booster_t2.py's actuator groups in bfm_wm's xi order: arms, waist, legs, feet, head
DELAY_GROUPS = (tuple(range(2, 16)), (16, 17, 18), (19, 20, 21, 22, 25, 26, 27, 28), (23, 24, 29, 30), (0, 1))

# The T2's VelocityCfg values (src/mdp.py; RobotSpec.velocity): the command box is the 0.1-99.9 percentiles of
# MotionLib.ref_cmd over data/t2 (11,900 clips, 2026-11-19; utils/check_clips.py); fall_height is the T1's 0.32 / 0.688
# at INIT_Z; ankle_scale (24 N m)^2 is a mass-scaled placeholder for the T1's (20 N m)^2.
# rest_pose (mdp.VelocityCfg.w_rest's joints and angles [rad], inert while --rest_weight is 0): the medians of the
# 2,959 still standing frames of data/t2_untwist with flat soles and the feet side by side, made left-right symmetric
# (hip roll +3.2 / -3.6 deg, hip yaw +2.3 / -2.3, waist pitch -3.1, waist roll and yaw -0.01)
# step_cycle (mdp.VelocityCfg.w_step's cycle law T(v) = 1.107 v^-0.555 s, inert while --step_weight is 0): fitted
# 2026-11-22 to the step term's own reading of data/t2_untwist's walking clips, so the clips score about zero: at the
# 6,591 forward touchdowns whose stride over cycle time is within 20% of the clip's command and whose trunk heading is
# within 15 deg of the travel (no run/jog/jump/hop/skip/leap clips), the landing foot's lead along the command in the
# trunk's heading frame, per 0.1 m/s bin of the command (0.33-1.29 m/s), as T = 2 lead / v; within 3% of every bin.
# The clips lead 0.33 m at 0.33 m/s, 0.42 at 0.52, 0.56 at 1.0. (The strides themselves give T = 1.163 v^-0.544, the
# cycle time proper; a first fit over all strides, 1.139 v^-0.362, kept missed swings at slow speeds and read the
# leads 11-15% short there.) The trunk is the T2's root: its heading wanders from the travel at slow speeds
VELOCITY_OVERRIDES = {"cmd_low": (-1.39, -1.14, -2.78), "cmd_high": (1.90, 1.14, 2.78), "fall_height": 0.46,
                      "ankle_scale": 576.0, "step_cycle": (1.107, 0.555),
                      "rest_pose": (("left_hip_roll_joint", 0.059), ("right_hip_roll_joint", -0.059),
                                    ("left_hip_yaw_joint", 0.040), ("right_hip_yaw_joint", -0.040),
                                    ("waist_pitch_joint", -0.053), ("waist_roll_joint", 0.0),
                                    ("waist_yaw_joint", 0.0))}  # fmt: skip


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(MJCF_PATH)
    # MuJoCo's defaults in place of the vendor <option>: src/env.py make_sim sets these, and mjlab's attach warns otherwise
    spec.option.timestep, spec.option.impratio, spec.option.tolerance = 0.002, 1.0, 1e-8
    for a in list(spec.actuators):  # torque <motor>s: mjlab adds its own
        spec.delete(a)
    for k in list(spec.keys):
        spec.delete(k)
    for s in list(spec.sensors):
        spec.delete(s)
    for g in list(spec.geoms):
        if g.name == "floor" or (g.contype == 0 and g.conaffinity == 0):  # mjlab adds the terrain; visuals unused
            spec.delete(g)
    used = {g.meshname for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}
    for m in list(spec.meshes):  # the head and waist colliders are meshes: keep those
        if m.name not in used:
            spec.delete(m)
    for g in spec.geoms:
        g.priority = 1  # robot friction (the sim context) wins over the terrain's
    return spec


def _actuators() -> tuple[BoosterPdActuatorCfg, ...]:
    return tuple(
        BoosterPdActuatorCfg(target_names_expr=(n,), stiffness=KP[n], damping=KD[n], effort_limit=e, v_max=v, v_knee=k)
        for n, (e, v, k, _) in MOTOR.items()
    )


def get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, INIT_Z), joint_pos=DEFAULT_JOINT_POS, joint_vel={".*": 0.0}),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=_actuators(), soft_joint_pos_limit_factor=0.9),
    )


SPEC = RobotSpec(
    name="t2",
    mjcf_path=MJCF_PATH,
    joint_names=tuple(T2_JOINT_NAMES),
    tracked_bodies=tuple(TRACKED_BODIES),
    default_joint_pos=DEFAULT_JOINT_POS,
    action_scale=ACTION_SCALE,
    com_body="trunk",
    com_range=((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05)),
    robot_cfg=get_robot_cfg,
    feet=("left_ankle_roll_link", "right_ankle_roll_link"),
    contact_z=0.0425,  # sole bottom 0.032506 below the ankle roll link + 1 cm, the T1's rule (0.04325 + 0.01 = 0.0533)
    mirror_perm=MIRROR_PERM,
    mirror_sign=MIRROR_SIGN,
    delay_groups=DELAY_GROUPS,
    delay_range=(0, 8),
    eval_delay=4,
    velocity=VELOCITY_OVERRIDES,
)
