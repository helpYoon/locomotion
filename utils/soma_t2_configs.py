"""Write soma-retargeter's configs for the booster_t2 targets that utils/make_booster_t2.py registers, without the Robot
Configurator GUI (route A of the T2 port plan).

Everything starts from soma's tuned Booster T1 configs, with the T1 links mapped to their T2 analogs (LINKS):
- scaler (one per target): human_scale_ratio 0.75 or 0.80; reference_joint_q = the T1's calibration pose (legs and
  waist 0, shoulder roll -/+1.2, elbow yaw -/+1.55, wrist pitch -0.3) at the robot's default root; human_joint_offsets
  computed the Configurator's way (_compute_offsets, robot_config_generator.py:1396-1434): soma's own
  _compute_local_offset from each scaled SOMA zero-pose joint to its mapped link, at that pose, no gizmo offsets;
- retargeter (one per target): the T1's IK weights, iterations and smooth masks; collision_groups dropped (soma never
  reads them);
- post-processing (shared): the T1's, effectors renamed in the same order (ik_limbs indices unchanged).
Self-checks: the same computation reproduces soma's shipped T1 offsets, and Newton's revolute order (the CSV column
order) equals the vendor MJCF's.

usage (soma's venv, CPU): cd <soma-retargeter>
    CUDA_VISIBLE_DEVICES= .venv/bin/python <bfm_wm>/utils/soma_t2_configs.py .
"""

import copy
import json
import os
import sys
import xml.etree.ElementTree as ET

import newton
import numpy as np
import warp as wp

ROOT = os.path.abspath(sys.argv[1])
sys.path.insert(0, ROOT)  # app/ is not an installed package
from app.tools.robot_config_generator import _compute_local_offset  # noqa: E402
import soma_retargeter.io.bvh as bvh_utils  # noqa: E402
import soma_retargeter.pipelines.utils as pipeline_utils  # noqa: E402
from soma_retargeter.animation.skeleton import SkeletonInstance  # noqa: E402
from soma_retargeter.robotics import robot_registry  # noqa: E402
from soma_retargeter.utils import newton_utils  # noqa: E402
from soma_retargeter.utils.space_conversion_utils import FacingDirectionType, SpaceConverter  # noqa: E402

T1_CFG = os.path.join(ROOT, "soma_retargeter", "assets", "robotics", "booster", "t1", "configs")
DST = os.path.join(ROOT, "local-robots", "booster_t2", "configs")
VENDOR_XML = os.path.expanduser("~/booster_assets/robots/T2/T2_31dof_Revo2Hand/T2_31dof_Revo2Hand.xml")
SCALES = {"booster_t2_075": 0.75, "booster_t2_080": 0.80}  # = make_booster_t2.TARGETS
LINKS = {  # soma's T1 link -> T2 link, by role
    "Waist": "waist_yaw_link", "Trunk": "trunk", "H2": "head_pitch_link",
    "AL1": "left_shoulder_pitch_link", "AL2": "left_shoulder_roll_link", "AL3": "left_elbow_pitch_link",
    "AL4": "left_elbow_yaw_link", "left_hand_link": "left_wrist_roll_link",
    "AR1": "right_shoulder_pitch_link", "AR2": "right_shoulder_roll_link", "AR3": "right_elbow_pitch_link",
    "AR4": "right_elbow_yaw_link", "right_hand_link": "right_wrist_roll_link",
    "Hip_Roll_Left": "left_hip_roll_link", "Shank_Left": "left_knee_pitch_link",
    "left_foot_link": "left_ankle_roll_link",
    "Hip_Roll_Right": "right_hip_roll_link", "Shank_Right": "right_knee_pitch_link",
    "right_foot_link": "right_ankle_roll_link",
}
CALIB = {"left_shoulder_roll_joint": -1.2, "left_elbow_yaw_joint": -1.55, "left_wrist_pitch_joint": -0.3,
         "right_shoulder_roll_joint": 1.2, "right_elbow_yaw_joint": 1.55, "right_wrist_pitch_joint": -0.3,
         # legs calibrated straight: the T2 leg is longest at knee 0.15, not 0 (T1-style zero legs put the knee
         # under its 0.129 soft limit in 18-21% of pilot frames, with one-frame jumps in 31-32 of 42 clips; this
         # gave 0.00% and 0 clips at scale 0.75, 2026-11-19 pilot)
         **{f"{s}_{j}": v for s in ("left", "right")
            for j, v in (("hip_pitch_joint", -0.075), ("knee_pitch_joint", 0.15), ("ankle_pitch_joint", -0.075))}}

wp.set_device("cpu")
robot_registry.register_robots_path(os.path.join(ROOT, "local-robots"))
skeleton, zero_pose = bvh_utils.load_bvh(pipeline_utils.get_source_zero_pose_asset_path(pipeline_utils.SourceType.SOMA))


def load(name: str) -> dict:
    with open(os.path.join(T1_CFG, name)) as f:
        return json.load(f)


def save(name: str, data: dict) -> None:
    with open(os.path.join(DST, name), "w") as f:
        json.dump(data, f, indent=4)
    print("wrote", os.path.join(DST, name))


def offsets(target: str, scale: float, joint_q: np.ndarray, match: dict) -> dict:
    """The Configurator's offsets: SOMA zero pose (facing +x, scaled) -> each mapped link at joint_q (default root)."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_world(pipeline_utils.create_robot_builder(target), wp.transform_identity())
    model = builder.finalize()
    q = joint_q.copy()
    q[:7] = model.joint_q.numpy()[:7]
    wp.copy(model.joint_q, wp.array(q, dtype=wp.float32))
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    body_q = state.body_q.numpy()
    body = {newton_utils.get_name_from_label(label): i for i, label in enumerate(model.body_label)}

    z_rot = wp.transform(wp.vec3(0, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.radians(90.0)))
    facing = SpaceConverter(FacingDirectionType.MUJOCO).transform(wp.transform_identity())
    soma = SkeletonInstance(skeleton, (1, 1, 0), wp.mul(z_rot, facing))
    soma.set_local_transforms(zero_pose.get_local_transforms(0))
    soma.scale = scale
    soma_tx = soma.compute_global_transforms()
    out = {}
    for joint, m in match.items():
        tx = _compute_local_offset(wp.transform(*soma_tx[skeleton.joint_index(joint)]),
                                   wp.transform(*body_q[body[m["t_body"]]]))
        out[joint] = [[round(float(v), 4) for v in tx.p], [round(float(v), 4) for v in tx.q]]  # as the GUI saves
    return out


t1_scaler = load("soma_to_booster_t1_scaler_config.json")
t1_retargeter = load("soma_to_booster_t1_retargeter_config.json")
t1_post = load("booster_t1_post_processing_config.json")

# the method reproduces soma's tuned T1 offsets (to the GUI's 4-decimal rounding)
mine = offsets("booster_t1", t1_scaler["human_scale_ratio"], np.array(t1_scaler["reference_joint_q"], np.float32),
               t1_retargeter["ik_match_table"])
for joint, (t, q) in t1_scaler["human_joint_offsets"].items():
    dt = np.abs(np.subtract(mine[joint][0], t)).max()
    dq = np.abs(np.subtract(mine[joint][1], q)).max()
    assert dt < 2e-4 and dq < 2e-4, (joint, mine[joint], t, q)
print("T1 offsets reproduced (max |dt|, |dq| <= 2e-4)")

# one floating base (trunk) and the 31 revolute joints in the vendor MJCF's order (= the CSV columns)
builder = pipeline_utils.create_robot_builder(next(iter(SCALES)))
model = builder.finalize()
names = [newton_utils.get_name_from_label(label) for label in model.joint_label]
free = [n for n, t in zip(names, model.joint_type.numpy()) if t == newton.JointType.FREE]
revolute = newton_utils.get_filtered_joint_names(builder, [newton.JointType.REVOLUTE])
mjcf = [j.get("name") for j in ET.parse(VENDOR_XML).getroot().iter("joint") if j.get("name") in revolute]
assert len(free) == 1 and newton_utils.get_name_from_label(model.body_label[0]) == "trunk", free
assert len(revolute) == 31 and revolute == mjcf, revolute
print(f"Newton: 1 free joint on trunk, 31 revolute joints in MJCF order; default root z {model.joint_q.numpy()[2]:.4f}")

ref_q = model.joint_q.numpy().copy()
coord = {n: int(s) for n, s in zip(names, model.joint_q_start.numpy())}
for joint, value in CALIB.items():
    ref_q[coord[joint]] = value

retargeter = copy.deepcopy(t1_retargeter)
retargeter.pop("collision_groups")  # never read; only self_penetration_weight is, and it is 0
retargeter["post_processing"]["robot_config"] = "configs/booster_t2_post_processing_config.json"
retargeter["smooth_joint_filter_objective_body_masks"] = {
    LINKS[k]: v for k, v in t1_retargeter["smooth_joint_filter_objective_body_masks"].items()}
for m in retargeter["ik_match_table"].values():
    m["t_body"], m["r_body"] = LINKS[m["t_body"]], LINKS[m["r_body"]]

post = copy.deepcopy(t1_post)
post["limb_stabilizer"]["effectors"] = {LINKS[k]: v for k, v in t1_post["limb_stabilizer"]["effectors"].items()}

os.makedirs(DST, exist_ok=True)
save("booster_t2_post_processing_config.json", post)
for target, scale in SCALES.items():
    save(f"soma_to_{target}_scaler_config.json", {
        "human_type": "soma",
        "human_root": "Hips",
        "robot_type": target,
        "robot_root": LINKS[t1_scaler["robot_root"]],
        "human_scale_ratio": scale,
        "reference_joint_q": [round(float(v), 10) for v in ref_q],
        "human_joint_offsets": offsets(target, scale, ref_q, retargeter["ik_match_table"]),
    })
    save(f"soma_to_{target}_retargeter_config.json",
         {**retargeter, "human_robot_scaler_config": f"configs/soma_to_{target}_scaler_config.json"})
