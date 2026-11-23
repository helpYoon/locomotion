"""t1.xml, robot/t1.py's joint names and t1_actuators.npz against Booster's T1_29dof.urdf (read here independently of
MuJoCo's importer) and sysid.txt."""

import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from robot.t1 import MJCF_PATH, T1_JOINT_NAMES
from utils.build_t1_mjcf import BODIES, BOOSTER_ARMATURE, DESC, JOINTS, URDF, WRIST_ROW, sysid

ROOT = ET.parse(URDF).getroot()
M = mujoco.MjModel.from_xml_path(MJCF_PATH)
vec = lambda e, k: np.array(((e.get(k) if e is not None else None) or "0 0 0").split(), dtype=float)  # noqa: E731
URDF_JOINTS = {j.get("name"): j for j in ROOT.iter("joint")}
MOVING = {n: j for n, j in URDF_JOINTS.items() if j.get("type") != "fixed"}
COLLIDERS = [g for g in range(M.ngeom) if M.geom_bodyid[g] and (M.geom_contype[g] or M.geom_conaffinity[g])]
NSIZE = {"box": 3, "cylinder": 2, "sphere": 1}  # MuJoCo sizes: half-extents, (radius, half-length), radius
MJ_KIND = {int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
           int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere"}  # fmt: skip
quat = lambda q: Rotation.from_quat(q, scalar_first=True).as_matrix()  # noqa: E731


def origin(e):
    o = e.find("origin")
    return vec(o, "xyz"), Rotation.from_euler("xyz", vec(o, "rpy")).as_matrix()


def mj_colliders(b):
    """Body b's colliders as (kind, size, position, rotation) in its frame."""
    return [(MJ_KIND[int(M.geom_type[g])], M.geom_size[g, : NSIZE[MJ_KIND[int(M.geom_type[g])]]], M.geom_pos[g],
             quat(M.geom_quat[g])) for g in COLLIDERS if M.geom_bodyid[g] == b]  # fmt: skip


def assert_same_shapes(got, want, err):
    """Same kinds in order, sizes and positions, and orientations up to the shape's symmetry (a box's 180-degree
    flips, a cylinder's axis direction)."""
    assert [k for k, *_ in got] == [k for k, *_ in want], err
    for (kind, size, p, R), (_, size_w, p_w, R_w) in zip(got, want):
        np.testing.assert_allclose(np.r_[size, p], np.r_[size_w, p_w], atol=1e-6, err_msg=err)
        rel = R_w.T @ R
        if kind == "box":
            np.testing.assert_allclose(np.abs(rel), np.eye(3), atol=1e-5, err_msg=err)
        elif kind == "cylinder":
            np.testing.assert_allclose(abs(rel[2, 2]), 1.0, atol=1e-9, err_msg=err)


def test_names_and_order():
    assert [M.joint(j).name for j in range(1, M.njnt)] == T1_JOINT_NAMES
    assert sorted(JOINTS[n] for n in MOVING) == sorted(T1_JOINT_NAMES)
    assert [M.actuator(i).name for i in range(M.nu)] == T1_JOINT_NAMES  # motors in joint order, one per joint
    assert (M.actuator_trnid[:, 0] == np.arange(1, M.njnt)).all() and (M.actuator_ctrlrange == M.jnt_actfrcrange[1:]).all()
    assert M.actuator_ctrllimited.all() and (M.actuator_gear == [1, 0, 0, 0, 0, 0]).all()


def test_forward_kinematics():
    """Every body's world pose at random joint angles, from the URDF's joint origins and axes vs MuJoCo's."""
    d = mujoco.MjData(M)
    rng = np.random.default_rng(0)
    children = {}
    for j in URDF_JOINTS.values():
        children.setdefault(j.find("parent").get("link"), []).append(j)
    for _ in range(4):
        q = {n: rng.uniform(*M.joint(JOINTS[n]).range) for n in MOVING}
        d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        for n, v in q.items():
            d.qpos[M.jnt_qposadr[M.joint(JOINTS[n]).id]] = v
        mujoco.mj_kinematics(M, d)
        stack = [("trunk", np.zeros(3), np.eye(3))]
        while stack:
            link, p, R = stack.pop()
            if link in BODIES:
                b = M.body(BODIES[link]).id
                np.testing.assert_allclose(d.xpos[b], p, atol=1e-9, err_msg=link)
                np.testing.assert_allclose(d.xmat[b].reshape(3, 3), R, atol=1e-9, err_msg=link)
            for j in children.get(link, []):
                t, Rj = origin(j)
                if j.get("name") in q:
                    Rj = Rj @ Rotation.from_rotvec(vec(j.find("axis"), "xyz") * q[j.get("name")]).as_matrix()
                stack.append((j.find("child").get("link"), p + R @ t, R @ Rj))


def test_inertia():
    """Mass, CoM and inertia tensor of every body (the massless camera link is dropped)."""
    total = 0.0
    for link in ROOT.iter("link"):
        i = link.find("inertial")
        if i is None:
            continue
        mass = float(i.find("mass").get("value"))
        total += mass
        if link.get("name") not in BODIES:
            continue
        b = M.body(BODIES[link.get("name")]).id
        c, R = origin(i)
        I = {k: float(v) for k, v in i.find("inertia").attrib.items()}
        I = np.array([[I["ixx"], I["ixy"], I["ixz"]], [I["ixy"], I["iyy"], I["iyz"]], [I["ixz"], I["iyz"], I["izz"]]])
        Rm = quat(M.body_iquat[b])
        np.testing.assert_allclose(M.body_mass[b], mass, rtol=1e-6, err_msg=link.get("name"))
        # t1.xml stores the CoM, principal moments and axes to ~6 significant digits
        np.testing.assert_allclose(M.body_ipos[b], c, atol=1e-6, err_msg=link.get("name"))
        np.testing.assert_allclose(Rm @ np.diag(M.body_inertia[b]) @ Rm.T, R @ I @ R.T, atol=1e-5 * np.abs(I).max(),
                                   err_msg=link.get("name"))  # fmt: skip
    np.testing.assert_allclose(M.body_mass.sum(), total, rtol=1e-6)


def test_collision_shapes():
    """Each body's colliders are its URDF collision primitives, in order; nothing else collides (no mesh)."""
    n = 0
    for link in ROOT.iter("link"):
        want = []
        for c in link.iter("collision"):
            (s,) = list(c.find("geometry"))
            size = {"box": lambda: vec(s, "size") / 2, "cylinder": lambda: [float(s.get("radius")), float(s.get("length")) / 2],
                    "sphere": lambda: [float(s.get("radius"))]}[s.tag]()  # fmt: skip
            want.append((s.tag, size, *origin(c)))
        if want:
            assert_same_shapes(mj_colliders(M.body(BODIES[link.get("name")]).id), want, link.get("name"))
        n += len(want)
    assert n == len(COLLIDERS)


def test_joint_limits_friction_and_motors():
    """Ranges and efforts are the URDF's, friction sysid's, armature sysid's but Booster's for the DM4310 and the
    ankles; t1_actuators.npz's torque and speed limits agree, with the URDF's velocities except Booster's parallel
    ankle."""
    assert M.jnt_limited[1:].all() and M.jnt_actfrclimited[1:].all()
    rows = sysid()
    tab = np.load(os.path.join(DESC, "..", "mujoco", "t1_actuators.npz"))
    idx = {n: i for i, n in enumerate(T1_JOINT_NAMES)}
    ankle_vel = {"Ankle_Pitch": 0.6 * 12.15, "Ankle_Roll": 12.15}  # Booster's E4315 through the linkage
    for n, j in MOVING.items():
        name, lim = JOINTS[n], j.find("limit")
        mj = M.joint(name)
        dof = M.jnt_dofadr[mj.id]
        effort, vel = float(lim.get("effort")), float(lim.get("velocity"))
        np.testing.assert_allclose(mj.range, [float(lim.get("lower")), float(lim.get("upper"))], err_msg=name)
        np.testing.assert_allclose(M.jnt_actfrcrange[mj.id], [-effort, effort], err_msg=name)
        key = name.removeprefix("Left_").removeprefix("Right_")
        arm, friction, viscous = rows.get(name) or rows[WRIST_ROW[key]]
        arm = BOOSTER_ARMATURE.get(key, arm)
        np.testing.assert_allclose([M.dof_armature[dof], M.dof_frictionloss[dof], M.dof_damping[dof]], [arm, friction, viscous],
                                   rtol=1e-6, err_msg=name)  # t1.xml prints ~6 significant digits  # fmt: skip
        np.testing.assert_allclose([tab["tau_max"][idx[name]], tab["v_max"][idx[name]]], [effort, ankle_vel.get(key, vel)],
                                   rtol=1e-6, err_msg=name)  # float32  # fmt: skip
