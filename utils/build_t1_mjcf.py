"""Build assets/t1/t1_description/mjcf/t1.xml from Booster's T1_29dof.urdf and sysid.txt.

Frames, masses, inertia, collision shapes, joint ranges and effort limits are the vendor URDF's (booster_assets
robots/T1/T1_29dof.urdf at 12a9751, copied to urdf/T1_29dof.urdf with its meshes). Coulomb friction (sysid's dynamic
column; MuJoCo has one frictionloss acting at all speeds), viscous damping and armature are sysid.txt's, except the
armature of the head's DM4310 motor and the parallel ankles, which are Booster's (booster_train actuator.py, as
robot/actuator.py did at tag supermimic-frozen-2026-11-15). sysid is 23-DoF, so the wrists take the row of their motor
class in the URDF: Wrist_Pitch / Hand_Roll the head's (7 N·m, 41.89 rad/s), Wrist_Yaw the elbow's (E4310). Names are
bfm_wm's, from Booster's earlier CamelCase release.

usage (from the repo root): uv run python utils/build_t1_mjcf.py
"""

import os
import re

import mujoco
import numpy as np

DESC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "t1", "t1_description")
URDF = os.path.join(DESC, "urdf", "T1_29dof.urdf")
SYSID = os.path.join(DESC, "..", "config", "sysid.txt")
OUT = os.path.join(DESC, "mjcf", "t1.xml")

_ARM = {"shoulder_pitch": "Shoulder_Pitch", "shoulder_roll": "Shoulder_Roll", "elbow_pitch": "Elbow_Pitch",
        "elbow_yaw": "Elbow_Yaw", "wrist_pitch": "Wrist_Pitch", "wrist_yaw": "Wrist_Yaw", "wrist_roll": "Hand_Roll"}
_ARM_LINK = dict(zip(_ARM, ("1", "2", "3", "4", "5", "6", None)))
_LEG = {"hip_pitch": ("Hip_Pitch", "Hip_Pitch"), "hip_roll": ("Hip_Roll", "Hip_Roll"), "hip_yaw": ("Hip_Yaw", "Hip_Yaw"),
        "knee_pitch": ("Knee_Pitch", "Shank"), "ankle_pitch": ("Ankle_Pitch", "Ankle_Cross"), "ankle_roll": ("Ankle_Roll", None)}
JOINTS = {"aahead_yaw_joint": "AAHead_yaw", "aahead_pitch_joint": "Head_pitch", "waist_yaw_joint": "Waist"}
BODIES = {"trunk": "Trunk", "aahead_yaw_link": "H1", "aahead_pitch_link": "H2", "waist_yaw_link": "Waist"}
for side, S in (("left", "Left"), ("right", "Right")):
    for k, v in _ARM.items():
        JOINTS[f"{side}_{k}_joint"] = f"{S}_{v}"
        BODIES[f"{side}_{k}_link"] = f"A{S[0]}{_ARM_LINK[k]}" if _ARM_LINK[k] else f"{side}_hand_link"
    for k, (j, b) in _LEG.items():
        JOINTS[f"{side}_{k}_joint"] = f"{S}_{j}"
        BODIES[f"{side}_{k}_link"] = f"{b}_{S}" if b else f"{side}_foot_link"
WRIST_ROW = {"Wrist_Pitch": "AAHead_yaw", "Hand_Roll": "AAHead_yaw", "Wrist_Yaw": "Left_Elbow_Yaw"}  # sysid has no wrists
# Booster's armatures where sysid's differ: DM4310 (head, wrist pitch/roll) and the E4315 behind the parallel ankle
BOOSTER_ARMATURE = {"AAHead_yaw": 0.0018, "Head_pitch": 0.0018, "Wrist_Pitch": 0.0018, "Hand_Roll": 0.0018,
                    "Ankle_Pitch": 2.0 * 0.0339552, "Ankle_Roll": 0.6 * 0.0339552}  # fmt: skip
HOME_ARMS = (0.5, -1.0, 0.0, -1.4, 0.0, 0.0, 0.0)  # left arm; mirrored on the right


def sysid() -> dict[str, tuple[float, float, float]]:
    """joint -> (armature, Coulomb (dynamic) friction, viscous damping) from sysid.txt."""
    txt = open(SYSID).read()
    consts = {k: float(v) for k, v in re.findall(r"^(ARMATURE_\d+) = ([\d.]+)", txt, re.M)}

    def lst(name):
        items = [x.strip() for x in re.search(rf"{name}=\[(.*?)\]", txt, re.S).group(1).split(",") if x.strip()]
        return [x.strip('"') for x in items] if name == "dof_names" else [consts.get(x) or float(x) for x in items]

    cols = zip(lst("dof_armature_list"), lst("dof_joint_dynamic_friction_list"), lst("dof_joint_viscous_friction_list"))
    return dict(zip(lst("dof_names"), cols))


def build() -> mujoco.MjSpec:
    src = open(URDF).read()
    floating = ('<mujoco><compiler meshdir="{}" discardvisual="false"/></mujoco>\n<link name="world"/>'
                '<joint name="floating_base_joint" type="floating"><parent link="world"/><child link="trunk"/></joint>')
    src, n = re.subn(r"(<robot\s+name=\"[^\"]*\">)", lambda mt: mt.group(1) + floating.format(DESC), src, count=1)
    assert n == 1
    spec = mujoco.MjSpec.from_string(src)
    spec.modelname = "t1"
    for old, new in BODIES.items():
        spec.body(old).name = new
    rows = sysid()
    for old, new in JOINTS.items():
        j = spec.joint(old)
        j.name = new
        j.armature, j.frictionloss, viscous = rows.get(new) or rows[WRIST_ROW[new.split("_", 1)[1]]]
        j.damping = [viscous, 0, 0]  # linear term only
        j.armature = BOOSTER_ARMATURE.get(new.removeprefix("Left_").removeprefix("Right_"), j.armature)
    for j in spec.joints[1:]:  # motors in joint (qpos) order, the order controllers index ctrl in
        a = spec.add_actuator(name=j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrllimited=True)
        a.ctrlrange = j.actfrcrange  # the URDF's effort limit
    for g in spec.geoms:  # the URDF's colors are translucent; colliders go in group 3, hidden by default (USD: guide)
        g.rgba = [*g.rgba[:3], 1.0]
        if g.contype or g.conaffinity:
            g.group = 3
    for mesh in spec.meshes:
        mesh.file = os.path.basename(mesh.file)
    spec.modelfiledir, spec.compiler.meshdir = os.path.dirname(OUT), "../meshes"

    # scene, IMU and home keyframe, as in the hand-written t1.xml this replaces
    spec.add_texture(type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=512)  # fmt: skip
    spec.add_texture(name="texplane", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.15, 0.2], width=512, height=512, mark=mujoco.mjtMark.mjMARK_CROSS,
                     markrgb=[0.8, 0.8, 0.8])  # fmt: skip
    mat = spec.add_material(name="matplane", reflectance=0.3, texrepeat=[1, 1], texuniform=True)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "texplane"
    w = spec.worldbody
    w.add_light(type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.4] * 3, specular=[0.1] * 3, pos=[0, 0, 5], dir=[0, 0, -1], castshadow=False)
    w.add_light(type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.6] * 3, specular=[0.2] * 3, pos=[0, 0, 4], dir=[0, 0, -1])
    w.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1], material="matplane")
    spec.body("Trunk").add_site(name="imu", size=[0.01] * 3)
    spec.add_sensor(name="orientation", type=mujoco.mjtSensor.mjSENS_FRAMEQUAT, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="imu-angular-velocity", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname="imu", noise=5e-4, cutoff=34.9)  # fmt: skip
    spec.add_sensor(name="imu-linear-acceleration", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu", noise=1e-2, cutoff=157)  # fmt: skip

    # stand the trunk so the lowest collision geom touches the ground with straight legs and the home arm pose
    joints = np.r_[0, 0, HOME_ARMS, np.array(HOME_ARMS) * [1, -1, 1, -1, 1, -1, -1], 0, np.zeros(12)]
    m = spec.compile()
    assert m.nbody == 31 and (m.body_jntnum[1:] == 1).all(), "every body but the world should have one joint"
    d = mujoco.MjData(m)
    d.qpos[3:7], d.qpos[7:] = [1, 0, 0, 0], joints
    mujoco.mj_kinematics(m, d)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    aabb_z = lambda g: (d.geom_xmat[g].reshape(3, 3)[2] @ (m.geom_aabb[g, :3] + corners * m.geom_aabb[g, 3:]).T).min()  # noqa: E731
    low = min(d.geom_xpos[g, 2] + aabb_z(g) for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_bodyid[g])
    z = round(-low, 4)
    spec.body("Trunk").pos = [0, 0, z]
    spec.add_key(name="home", qpos=np.r_[0, 0, z, 1, 0, 0, 0, joints])
    return spec


if __name__ == "__main__":
    spec = build()
    with open(OUT, "w") as f:
        f.write("<!-- Generated by utils/build_t1_mjcf.py from urdf/T1_29dof.urdf and config/sysid.txt; edit those, not this. -->\n")
        f.write(spec.to_xml())
    print(f"wrote {OUT}")
