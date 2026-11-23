"""robot/t2.py and utils/build_t2_mjcf.py's t2.xml: the simulated model, FK, mirror, Booster's T2 gains, delay groups
and the default stance (CPU only)."""

import dataclasses

import mujoco
import numpy as np
import pytest
import torch

from robot.kinematics import Kinematics
from robot.t2 import ACTION_SCALE, INIT_Z, MJCF_PATH, MOTOR, PARALLEL, SPEC, T2_JOINT_NAMES, VELOCITY_OVERRIDES, get_spec
from utils.rotations import quat_to_mat

M = mujoco.MjModel.from_xml_path(MJCF_PATH)
SIDES = ("left", "right")

# booster_deploy 7bb1462 booster_deploy/robots/t2.py:33-64, copied
DEPLOY_KP = [18.214394, 18.214394,
             42.952518, 34.614676, 34.614676, 34.614676, 18.214394, 18.214394, 18.214394,
             42.952518, 34.614676, 34.614676, 34.614676, 18.214394, 18.214394, 18.214394,
             99.837519, 132.424295, 70.997986,
             70.997986, 70.997986, 70.997986, 70.997986, 90.690730, 33.693992,
             70.997986, 70.997986, 70.997986, 70.997986, 90.690730, 33.693992]  # fmt: skip
DEPLOY_KD = [1.449455, 1.449455,
             3.418053, 2.754549, 2.754549, 2.754549, 1.449455, 1.449455, 1.449455,
             3.418053, 2.754549, 2.754549, 2.754549, 1.449455, 1.449455, 1.449455,
             2.754549, 2.754549, 5.649840,
             5.649840, 5.649840, 5.649840, 5.649840, 2.754549, 2.754549,
             5.649840, 5.649840, 5.649840, 5.649840, 2.754549, 2.754549]  # fmt: skip
DEPLOY_EFFORT = [22., 22.,
                 74., 60., 60., 60., 22., 22., 22.,
                 74., 60., 60., 60., 22., 22., 22.,
                 138., 144., 135.,
                 135., 135., 135., 135., 132., 72.,
                 135., 135., 135., 135., 132., 72.]  # fmt: skip
DEPLOY_Q0 = [0, 0,
             0.2, -1.3, 0.0, -0.5, 0.0, 0.0, 0.0,
             0.2, 1.3, 0.0, 0.5, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # fmt: skip
LINKAGE = (16, 17, 23, 24, 29, 30)  # waist pitch/roll and the ankles: the robot gets the motor-side Kd there
CROUCH = {19: -0.2, 22: 0.4, 23: -0.2, 25: -0.2, 28: 0.4, 29: -0.2}  # the user's T1-style legs (hip, knee, ankle pitch)


def test_model():
    """t2.xml: the 31 body joints in Booster's deploy order with Booster's armature, no finger joints or motors, one
    box per hand, the base ankle pitch range; get_spec keeps only the colliders (21) and their 2 meshes."""
    assert [M.joint(j).name for j in range(1, M.njnt)] == T2_JOINT_NAMES and M.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert [M.actuator(i).name for i in range(M.nu)] == T2_JOINT_NAMES
    np.testing.assert_allclose(M.dof_armature[6:], [MOTOR[n][3] for n in T2_JOINT_NAMES], rtol=1e-12)
    np.testing.assert_allclose(M.jnt_actfrcrange[1:, 1], [MOTOR[n][0] for n in T2_JOINT_NAMES], rtol=1e-12)
    for side in SIDES:
        np.testing.assert_array_equal(M.joint(f"{side}_ankle_pitch_joint").range, [-0.645771823, 0.331612558])
        assert M.geom(f"{side}_hand_collision").type[0] == mujoco.mjtGeom.mjGEOM_BOX
    m = get_spec().compile()
    assert m.nbody == 1 + 32 + 22 and (m.body_jntnum[1:] <= 1).all()  # world, trunk + 31 links, 22 welded finger links
    assert m.njnt == 32 and m.nu == 0 and m.nsensor == 0 and m.nmesh == 2
    assert m.ngeom == 21 and (m.geom_contype == 1).all() and (m.geom_conaffinity == 1).all() and (m.geom_priority == 1).all()
    assert "floor" not in [m.geom(g).name for g in range(m.ngeom)]
    assert sum(m.geom_type == mujoco.mjtGeom.mjGEOM_MESH) == 2  # head and waist; each hand is one box
    assert abs(m.body_mass.sum() - 42.24) < 0.01


def test_kinematics_matches_mujoco():
    """robot/kinematics.py (float32 torch) against MuJoCo on 50 random states: every moving body's pose and velocity."""
    d = mujoco.MjData(M)
    kin = Kinematics(SPEC, "cpu")
    moving = [b for b in range(1, M.nbody) if M.body_jntnum[b]]
    assert len(moving) == 32 and kin.num_bodies == 32
    assert [moving[i] for i in kin.tracked_idx] == [M.body(n).id for n in SPEC.tracked_bodies]
    rng = np.random.default_rng(0)
    qpos, qvel, want = [], [], []
    for _ in range(50):
        d.qpos[:3] = rng.uniform(-1, 1, size=3)
        d.qpos[3:7] = rng.normal(size=4)
        d.qpos[3:7] /= np.linalg.norm(d.qpos[3:7])
        d.qpos[7:] = rng.uniform(M.jnt_range[1:, 0], M.jnt_range[1:, 1])
        d.qvel[:] = rng.normal(size=M.nv)
        mujoco.mj_forward(M, d)
        vel = np.zeros((len(moving), 6))  # [angular, linear] at the body origin, world frame
        for i, b in enumerate(moving):
            mujoco.mj_objectVelocity(M, d, mujoco.mjtObj.mjOBJ_XBODY, b, vel[i], 0)
        qpos.append(d.qpos.copy()), qvel.append(d.qvel.copy())
        want.append((d.xpos[moving].copy(), d.xmat[moving].reshape(-1, 3, 3).copy(), vel[:, 3:], vel[:, :3]))
    qpos, qvel = torch.tensor(np.array(qpos), dtype=torch.float32), torch.tensor(np.array(qvel), dtype=torch.float32)
    R = quat_to_mat(qpos[:, 3:7])
    got = kin.forward(qpos[:, :3], R, qvel[:, :3], (R @ qvel[:, 3:6, None])[..., 0], qpos[:, 7:], qvel[:, 6:], all_bodies=True)
    for (g, w), atol in zip(zip(got, zip(*want)), (1e-5, 1e-5, 1e-4, 1e-4)):
        np.testing.assert_allclose(g, np.stack(w), atol=atol)


def test_mirror():
    """Mirroring a random pose (q[perm] * sign) reflects every tracked body across the sagittal plane onto its
    partner, and each joint's range is its partner's mirror."""
    kin = Kinematics(SPEC, "cpu")
    perm, sign = list(SPEC.mirror_perm), torch.tensor(SPEC.mirror_sign, dtype=torch.float32)
    assert sorted(perm) == list(range(31)) and all(perm[perm[i]] == i for i in range(31))
    rng = np.random.default_rng(1)
    q = torch.tensor(rng.uniform(M.jnt_range[1:, 0], M.jnt_range[1:, 1], size=(20, 31)), dtype=torch.float32)
    z, eye = torch.zeros(20, 3), torch.eye(3).expand(20, 3, 3)
    p, R, _, _ = kin.forward(z, eye, z, z, q, torch.zeros_like(q))
    pm, Rm, _, _ = kin.forward(z, eye, z, z, q[:, perm] * sign, torch.zeros_like(q))
    S = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
    names = list(SPEC.tracked_bodies)
    partner = [names.index(n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")) for n in names]
    np.testing.assert_allclose(pm[:, partner], p @ S, atol=1e-5)
    np.testing.assert_allclose(Rm[:, partner], S @ R @ S, atol=1e-5)
    lo, hi = M.jnt_range[1:, 0], M.jnt_range[1:, 1]
    s = np.array(SPEC.mirror_sign)
    np.testing.assert_allclose(np.minimum(lo[perm] * s, hi[perm] * s), lo)
    np.testing.assert_allclose(np.maximum(lo[perm] * s, hi[perm] * s), hi)


def test_booster_gains():
    """Kp, effort and the arm/head/waist default equal booster_deploy's literals; its Kd too on the serial joints, and on
    the 6 linkage joints the motor-side Kd times the armature ratio; action scale 0.25 E / Kp."""
    kp, kd, lim = SPEC.pd_gains()
    np.testing.assert_array_almost_equal(kp, DEPLOY_KP, decimal=6)
    np.testing.assert_allclose(lim, DEPLOY_EFFORT, rtol=1e-12)
    serial = [i for i in range(31) if i not in LINKAGE]
    np.testing.assert_array_almost_equal(np.array(kd)[serial], np.array(DEPLOY_KD)[serial], decimal=6)
    ratio = [PARALLEL["waist"][3][0], PARALLEL["waist"][3][1]] + [PARALLEL["ankle"][3][0], PARALLEL["ankle"][3][1]] * 2
    np.testing.assert_allclose(np.array(kd)[list(LINKAGE)], np.array(DEPLOY_KD)[list(LINKAGE)] * ratio, atol=1e-5)
    np.testing.assert_allclose(SPEC.action_scale_list(), 0.25 * np.array(lim) / np.array(kp), rtol=1e-12)
    assert list(ACTION_SCALE) == T2_JOINT_NAMES
    q0 = SPEC.default_joint_pos_list()
    assert q0 == [CROUCH.get(i, v) for i, v in enumerate(DEPLOY_Q0)]
    # torque-speed curve of Appendix A (rad/s): the B7629 hips, the waist and ankle linkages
    np.testing.assert_allclose([MOTOR[n][1:3] for n in ("left_hip_pitch_joint", "waist_pitch_joint", "right_ankle_pitch_joint")],
                               [(20.9440, 6.1785), (11.8857, 6.4403), (16.6400, 9.0164)], atol=1e-4)  # fmt: skip


def test_delay_groups():
    """Booster's five actuator groups, in xi order (arms, waist, legs, feet, head), partition the 31 joints."""
    g = SPEC.delay_groups
    assert sorted(j for grp in g for j in grp) == list(range(31))
    names = [[T2_JOINT_NAMES[j] for j in grp] for grp in g]
    assert all(any(k in n for k in ("shoulder", "elbow", "wrist")) for n in names[0])
    assert all(n.startswith("waist") for n in names[1]) and all(("hip" in n or "knee" in n) for n in names[2])
    assert all("ankle" in n for n in names[3]) and all("head" in n for n in names[4])
    assert SPEC.delay_range == (0, 8) and SPEC.eval_delay == 4


def test_default_stance():
    """Upright in the default pose with the lowest sole corner on the floor: soles level within 4 mm, the CoM ahead of
    both ankles, only the two soles touching (no self-contact), INIT_Z = that height + 9 mm, contact_z = sole depth
    + 1 cm."""
    d = mujoco.MjData(M)
    d.qpos[:7], d.qpos[7:] = [0, 0, 0, 1, 0, 0, 0], SPEC.default_joint_pos_list()
    mujoco.mj_kinematics(M, d)
    corners = np.array([[sx, sy, -1] for sx in (-1, 1) for sy in (-1, 1)])
    sole = np.concatenate([d.geom_xpos[g] + (corners * M.geom_size[g]) @ d.geom_xmat[g].reshape(3, 3).T
                           for g in (M.geom(f"{s}_ankle_collision").id for s in SIDES)])  # fmt: skip
    assert np.ptp(sole[:, 2]) < 0.004
    d.qpos[2] = -sole[:, 2].min()
    mujoco.mj_forward(M, d)
    assert abs(INIT_Z - (d.qpos[2] + 0.009)) < 0.001
    assert all(d.subtree_com[1, 0] > d.xpos[M.body(f"{s}_ankle_pitch_link").id, 0] for s in SIDES)
    assert {(M.geom(c.geom1).name, M.geom(c.geom2).name) for c in d.contact[: d.ncon]} == {("floor", f"{s}_ankle_collision") for s in SIDES}
    for s in SIDES:
        g = M.geom(f"{s}_ankle_collision").id
        assert abs(SPEC.contact_z - (M.geom_size[g, 2] - M.geom_pos[g, 2] + 0.01)) < 1e-4
    assert SPEC.feet == ("left_ankle_roll_link", "right_ankle_roll_link") and all(f in SPEC.tracked_bodies for f in SPEC.feet)


def test_joint_matching_keeps_t1_indices_and_finds_t2s():
    """src/mdp.py's ankle and leg matches: the T1's indices of the old T1-only expressions, and the T2's joints."""
    from robot.spec import get_robot
    from src.mdp import ankle_pitch_joints, leg_joints

    t1 = get_robot("t1").joint_names
    assert ankle_pitch_joints(t1) == [i for i, n in enumerate(t1) if n.endswith("Ankle_Pitch")] == [21, 27]
    assert leg_joints(t1) == [i for i, n in enumerate(t1) if any(k in n for k in ("Hip", "Knee", "Ankle"))] == list(range(17, 29))
    assert [T2_JOINT_NAMES[i] for i in ankle_pitch_joints(T2_JOINT_NAMES)] == ["left_ankle_pitch_joint", "right_ankle_pitch_joint"]
    assert leg_joints(T2_JOINT_NAMES) == list(range(19, 31))
    assert all(n.split("_")[1] in ("hip", "knee", "ankle") for n in np.array(T2_JOINT_NAMES)[leg_joints(T2_JOINT_NAMES)])


def test_velocity_task_takes_each_robots_values():
    """enable_velocity puts RobotSpec.velocity over the cfg it is given (train.py's flags kept); the T1 has none."""
    from robot.spec import get_robot
    from src.dr import XiLayout
    from src.mdp import MDP, VelocityCfg
    from src.motion import MotionLib

    assert set(VELOCITY_OVERRIDES) <= {f.name for f in dataclasses.fields(VelocityCfg)} and SPEC.velocity is VELOCITY_OVERRIDES
    for robot in (SPEC, get_robot("t1")):
        kin = Kinematics(robot, "cpu")
        q0 = torch.tensor(robot.default_joint_pos_list())
        p, quat = torch.tensor([0.0, 0.0, 1.0]).repeat(10, 1), torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(10, 1)
        motion = MotionLib.from_clips(["stand"], [p], [quat], [q0.repeat(10, 1)], kin, "cpu")
        mdp = MDP(kin, motion, q0, XiLayout(robot, "cpu").dim, robot.sensing)
        mdp.feet = [robot.tracked_bodies.index(b) for b in robot.feet]
        mdp.enable_velocity(robot, VelocityCfg(w_ankle=-4.0, track_sharp=0.75))
        assert mdp.vel.w_ankle == -4.0 and mdp.vel.track_sharp == 0.75
        if robot is SPEC:
            assert (mdp.vel.cmd_low, mdp.vel.cmd_high) == ((-1.39, -1.14, -2.78), (1.90, 1.14, 2.78))
            assert (mdp.vel.fall_height, mdp.vel.ankle_scale) == (0.46, 576.0)
            rest = [robot.joint_names[i] for i in mdp.rest_idx.tolist()]  # the joints a stance leaves free, mirrored
            assert rest == [n for n, _ in VELOCITY_OVERRIDES["rest_pose"]] and len(rest) == 7
            assert all(("hip_roll" in n or "hip_yaw" in n or "waist" in n) for n in rest)
            pose = dict(VELOCITY_OVERRIDES["rest_pose"])
            assert all(pose[f"left_hip_{a}_joint"] == -pose[f"right_hip_{a}_joint"] for a in ("roll", "yaw"))
            assert [robot.joint_names[i] for i in mdp.ankle_roll_idx.tolist()] == ["left_ankle_roll_joint", "right_ankle_roll_joint"]
        else:
            assert mdp.vel == VelocityCfg(w_ankle=-4.0, track_sharp=0.75)
        assert torch.equal(mdp.cmd_lo, torch.tensor(mdp.vel.cmd_low)) and torch.equal(mdp.cmd_hi, torch.tensor(mdp.vel.cmd_high))
        assert len(mdp.ankle_idx) == 2
        if robot is SPEC:  # the landing command's sole box is still the T1's: refused, not trained on silently
            with pytest.raises(AssertionError, match="T1-only"):
                mdp.enable_landing(robot)
        else:
            mdp.enable_landing(robot)
            assert mdp.leg_idx.tolist() == list(range(17, 29))


def test_untwist_turns_the_trunk_and_leaves_the_pelvis_and_legs():
    """utils/edit_t2_motions.py point 4: the trunk turns by UNTWIST about its own axes; the pelvis and feet stay."""
    from utils import edit_t2_motions as E
    from utils.rotations import so3_exp

    rng = np.random.default_rng(0)
    q = {n: rng.uniform(-0.3, 0.3, 40) for n in T2_JOINT_NAMES}
    lo = M.jnt_range[M.joint("waist_roll_joint").id][0]
    q["waist_roll_joint"][0] = lo  # the turn takes roll down: this frame clips, and the trunk gives instead
    pos, R = rng.uniform(-1, 1, (40, 3)), so3_exp(torch.from_numpy(rng.normal(0, 0.6, (40, 3)))).numpy()
    waist, pos2, R2, clipped = E.untwist(q, pos, R)

    def fk(q, pos, R):
        d, out = mujoco.MjData(M), []
        for qp in E.qpos_of(q, pos, R):
            d.qpos[:] = qp
            mujoco.mj_kinematics(M, d)
            out.append(np.c_[d.xpos, d.xmat][[E.PELVIS, *E.FEET]])
        return np.array(out)

    assert np.abs(fk(q | waist, pos2, R2) - fk(q, pos, R)).max() < 1e-9
    C = so3_exp(torch.from_numpy(E.UNTWIST)).numpy()
    assert clipped[0] and waist["waist_roll_joint"][0] == lo and not clipped[1:].any()
    assert np.abs(R2[1:] - R[1:] @ C).max() < 1e-9 and np.abs(R2[0] - R[0] @ C).max() > 1e-4
    a = np.radians(E.euler_xyz(R2))  # the CSV's extrinsic xyz Euler angles give the rotation back
    assert np.abs(E.rot(2, a[:, 2]) @ E.rot(1, a[:, 1]) @ E.rot(0, a[:, 0]) - R2).max() < 1e-12
