"""CPU checks of the landing command's parts (src/landing.py and its wiring).    pytest tests/test_landing.py"""

import json
import math
import os

import mujoco
import numpy as np
import pytest
import torch

from robot.kinematics import Kinematics
from robot.t1 import MJCF_PATH, SPEC
from src import landing
from src.landing import LandingCfg, compose, relative
from utils.rotations import quat_to_mat

FEET = [SPEC.tracked_bodies.index(b) for b in SPEC.feet]
KIN = Kinematics(SPEC, "cpu")


def test_sole_geometry_matches_mujoco():
    m = mujoco.MjModel.from_xml_path(MJCF_PATH)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    for _ in range(4):
        d.qpos[:3], d.qpos[3:7] = rng.normal(size=3), rng.normal(size=4)
        d.qpos[3:7] /= np.linalg.norm(d.qpos[3:7])
        d.qpos[7:] = rng.uniform(m.jnt_range[1:, 0], m.jnt_range[1:, 1])
        d.qvel[:] = rng.normal(size=m.nv)
        mujoco.mj_forward(m, d)
        qpos, qvel = torch.tensor(d.qpos, dtype=torch.float32)[None], torch.tensor(d.qvel, dtype=torch.float32)[None]
        R = quat_to_mat(qpos[:, 3:7])
        out = KIN.forward(qpos[:, :3], R, qvel[:, :3], (R @ qvel[:, 3:6, None])[..., 0], qpos[:, 7:], qvel[:, 6:])
        s = landing.soles(*(x[:, FEET] for x in out))
        for k, body in enumerate(SPEC.feet):
            g = next(g for g in range(m.ngeom)
                     if m.geom_bodyid[g] == m.body(body).id and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX)
            X, hx, hy, hz = d.geom_xmat[g].reshape(3, 3), *m.geom_size[g]
            centre = d.geom_xpos[g] + X @ [0, 0, -hz]  # the box's bottom face
            low = min((centre + X @ [sx * hx, sy * hy, 0])[2] for sx in (-1, 1) for sy in (-1, 1))
            vel = np.zeros(6)  # [angular, linear] at the geom's centre, world frame
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, g, vel, 0)
            np.testing.assert_allclose(s["pose"][0, k, :2], centre[:2], atol=2e-5)
            np.testing.assert_allclose(s["pose"][0, k, 2], math.atan2(X[1, 0], X[0, 0]), atol=2e-5)
            np.testing.assert_allclose(s["h"][0, k], low, atol=2e-5)
            np.testing.assert_allclose(s["zc"][0, k], centre[2], atol=2e-5)
            np.testing.assert_allclose(s["tilt"][0, k], math.atan2(math.hypot(X[2, 0], X[2, 1]), X[2, 2]), atol=2e-5)
            v_centre = vel[3:] + np.cross(vel[:3], centre - d.geom_xpos[g])
            np.testing.assert_allclose(s["v"][0, k], v_centre[:2], atol=2e-4)
            np.testing.assert_allclose(s["wz"][0, k], vel[2], atol=2e-4)


def test_frames():
    P, c = torch.tensor([[0.3, -0.2, 2.5]]), torch.tensor([[0.1, 0.2125, -0.4]])
    assert torch.allclose(relative(compose(P, c), P), c, atol=1e-6)
    # the left sole beside the right one
    stay = compose(torch.tensor([[0.0, -0.10625, 0.0]]), torch.tensor([[0.0, 0.2125, 0.0]]))
    assert torch.allclose(stay, torch.tensor([[0.0, 0.10625, 0.0]]), atol=1e-7)
    # P's x axis is the world's y
    turned = compose(torch.tensor([[1.0, 2.0, math.pi / 2]]), torch.tensor([[0.3, 0.0, 0.0]]))
    assert torch.allclose(turned, torch.tensor([[1.0, 2.3, math.pi / 2]]), atol=1e-6)
    back = relative(torch.tensor([[0.0, 0.0, 3.1]]), torch.tensor([[0.0, 0.0, -3.1]]))  # 6.2 rad apart wraps to -0.083
    assert abs(float(back[0, 2]) - (6.2 - 2 * math.pi)) < 1e-6


def test_sole_rel_on_flat_soles():
    q0 = torch.tensor(SPEC.default_joint_pos_list())[None]
    z3 = torch.zeros(1, 3)
    out = KIN.forward(z3, torch.eye(3)[None], z3, z3, q0, torch.zeros_like(q0))
    s = landing.soles(*(x[:, FEET] for x in out))
    for swing in (0, 1):
        r5 = landing.sole_rel(out[0][:, FEET], out[1][:, FEET], torch.tensor([float(swing)]))
        rel = relative(s["pose"][:, swing], s["pose"][:, 1 - swing])
        assert torch.allclose(r5[:, :2], rel[:, :2], atol=1e-4) and abs(float(r5[0, 2])) < 2e-3
        assert torch.allclose(r5[:, 3:], torch.stack((torch.cos(rel[:, 2]), torch.sin(rel[:, 2])), -1), atol=1e-4)


def test_cfg_ceiling():
    assert LandingCfg().max_step_reward == pytest.approx(0.1) and LandingCfg().bonus_reward == pytest.approx(1.25)


def _soles(T, moves=(), width=0.2125):
    """Sole arrays of a built clip: both soles at rest side by side; moves = (sole, f0, f1, dx, dy, dyaw, rise), a
    minimum-jerk displacement between frames f0 and f1 with a parabolic rise."""
    xy = np.zeros((T, 2, 2), np.float32)
    xy[:, 0, 1], xy[:, 1, 1] = width / 2, -width / 2
    yaw, zc, t = np.zeros((T, 2), np.float32), np.zeros((T, 2), np.float32), np.arange(T)
    for k, f0, f1, dx, dy, dyaw, rise in moves:
        s = np.clip((t - f0) / (f1 - f0), 0, 1)
        b = 10 * s**3 - 15 * s**4 + 6 * s**5
        xy[:, k, 0] += b * dx
        xy[:, k, 1] += b * dy
        yaw[:, k] += b * dyaw
        zc[:, k] += rise * 4 * s * (1 - s)
    return xy, yaw, zc


def _tables(T, moves=()):
    return landing.label_tables(*_soles(T, moves), np.array([0]), np.array([T - 1]))


STEP = (0, 60, 85, 0.30, 0.0, 0.0, 0.04)  # the left sole 30 cm forward in 0.5 s, rising 4 cm


def test_label_rule_step_from_a_stand():
    t = _tables(160, [STEP])
    assert len(t["steps"]) == 1
    _, swing, c0, dl, chained = t["steps"][0]
    assert (swing, chained) == (0, 0) and 60 <= c0 + 25 <= 64 and 85 <= dl <= 89  # lift-off just after frame 60
    assert (t["kind"][c0:dl] == 2).all() and (t["kind"][:c0] == 1).all() and (t["kind"][dl:] == 1).all()
    assert t["n"][c0] == dl - c0 and t["n"][dl - 1] == 1 and (t["swing"][c0:dl] == 0).all()
    np.testing.assert_allclose(t["c"][c0], [0.30, 0.2125, 0.0], atol=2e-3)  # the landing, in the right sole's frame
    assert len(t["label_reset"]) == (c0 - 25) - max(c0 - 75, 0) + 1
    assert t["label_reset"].max() == c0 - 25
    assert len(t["free_reset"]) == max(0, c0 - 50) + max(0, (160 - dl) - 50)
    assert t["ahead"][0] == 160 and t["ahead"][159] == 1


def test_label_rule_rejects():
    slide = _tables(160, [(0, 60, 85, 0.30, 0.0, 0.0, 0.0)])  # no rise
    assert len(slide["steps"]) == 0 and (slide["kind"][64:82] == 0).all() and slide["ahead"][70] == 0
    assert len(_tables(160, [STEP, (1, 62, 87, 0.30, 0.0, 0.0, 0.04)])["steps"]) == 0  # both soles move at once
    assert len(_tables(160, [(0, 60, 70, 0.045, 0.0, 0.0, 0.04)])["steps"]) == 0  # travel under 5 cm
    assert len(_tables(160, [STEP, (1, 40, 100, 0.04, 0.0, 0.0, 0.0)])["steps"]) == 0  # the planted sole creeps 3.7 cm
    assert len(_tables(87, [STEP])["steps"]) == 0  # the deadline would be the clip's last frame
    assert (_tables(59)["kind"] == 0).all()  # a clip under 60 frames: nothing, hold included


def test_label_rule_chain_and_clip_borders():
    # the right sole lifts 8 frames after the first deadline
    t = _tables(200, [STEP, (1, 92, 117, 0.30, 0.0, 0.0, 0.04)])
    (_, s1, c1, d1, ch1), (_, s2, c2, d2, ch2) = t["steps"]
    assert (s1, ch1, s2, ch2) == (0, 0, 1, 1) and c2 == d1 and t["n"][c2] == d2 - c2
    assert (t["kind"][c1:d2] == 2).all()
    np.testing.assert_allclose(t["c"][c2], [0.0, -0.2125, 0.0], atol=2e-3)  # it lands beside the left sole
    _, _, c0, dl, _ = _tables(160, [STEP])["steps"][0]
    a, b = _soles(160, [STEP]), _soles(160)  # a standing clip, then the clip with the step: frames 0-159 and 160-319
    two = landing.label_tables(*(np.concatenate(x) for x in zip(b, a)), np.array([0, 160]), np.array([159, 319]))
    assert two["steps"][0, 0] == 1 and two["steps"][0, 2] == 160 + c0
    assert two["ahead"][159] == 1 and two["ahead"][160] == 160
    assert len(two["free_reset"]) == 110 + max(0, c0 - 50) + max(0, 160 - dl - 50)
    assert (two["label_reset"] >= 160).all()


LIB = "data/t1/lib_all.npz"


@pytest.mark.skipif(not os.path.exists(LIB), reason="needs the packed library")
def test_library_label_counts():
    from src.motion import MotionLib

    m = MotionLib.load(KIN, "cpu", "data/t1/keep.txt", LIB, "fp32")
    m.set_land_labels(FEET)
    st = m.land_steps
    counts = (len(st), len(np.unique(st[:, 0])), int((st[:, 4] == 0).sum()), int((st[:, 4] == 1).sum()))
    assert counts == (7516, 2128, 2631, 4885)
    # the rule thresholds a float speed, so a frame at the threshold can flip between machines; the step counts cannot
    # hide such a change
    assert int((m.land_kind == 2).sum()) == 306624
    assert abs(int((m.land_kind == 1).sum()) - 1199343) <= 20
    assert abs(len(m.land_label_reset) - 44021) <= 20 and abs(len(m.land_free_reset) - 413937) <= 20


def _hold_state(B=1, swing=1.0, width=0.2125):
    """The command state and sole poses of B robots holding the home stance (left sole at +width/2)."""
    pose = torch.tensor([[[0.0, width / 2, 0.0], [0.0, -width / 2, 0.0]]]).repeat(B, 1, 1)
    C = torch.zeros(B, landing.STATE_DIM)
    C[:, landing.SWING], C[:, 6:], C[:, 3:6] = swing, pose.flatten(1), pose[:, int(swing)]
    return C, pose


def _s(pose, h=None, v=None, zc=None, wz=None):
    B = pose.shape[0]
    h = torch.zeros(B, 2) if h is None else h
    return {"pose": pose, "h": h, "zc": h if zc is None else zc, "tilt": torch.zeros(B, 2),
            "v": torch.zeros(B, 2, 2) if v is None else v, "wz": torch.zeros(B, 2) if wz is None else wz}  # fmt: skip


Z, ONE = torch.zeros(1), torch.tensor([True])
step = lambda C, pose0, s1: landing.land_step(LandingCfg(), C, pose0, s1, Z, Z, Z, Z, Z)  # noqa: E731  no regularizers
RIGHT_15 = torch.tensor([[0.15, -0.2125, 0.0]])  # the right sole 15 cm forward, in the left sole's frame


def test_phases_and_exact_rewards():
    cfg = LandingCfg()
    C, pose = _hold_state()
    r, C1, _ = step(C, pose, _s(pose))
    assert float(r) == pytest.approx(0.1) and torch.equal(C1, C)  # HOLD at the ceiling: 0.125 x 0.02 x (30 + 5 + 5)
    Cs = landing.land_reset(cfg, C, pose + 0.01, ONE, torch.tensor([0.0]))
    assert (Cs[0, landing.M], Cs[0, landing.N], Cs[0, landing.SWING]) == (25, 0, 0)
    assert torch.allclose(Cs[0, 6:], (pose + 0.01).flatten())
    assert torch.equal(landing.land_issue(Cs, pose, ONE, torch.tensor([1.0]), RIGHT_15, torch.tensor([40.0])), Cs)
    for _ in range(25):  # SETTLE pays survival and stillness
        r, Cs, _ = step(Cs, pose, _s(pose))
        assert float(r) == pytest.approx(0.0875)
    assert Cs[0, landing.M] == 0 and torch.allclose(Cs[0, 6:], pose.flatten())  # anchors latched where SETTLE ends
    Ci = landing.land_issue(Cs, pose, ONE, torch.tensor([1.0]), RIGHT_15, torch.tensor([40.0]))
    assert (Ci[0, landing.SWING], Ci[0, landing.N]) == (1, 40)
    assert torch.allclose(Ci[0, 3:6], torch.tensor([0.15, -0.10625, 0.0]), atol=1e-6)
    assert torch.equal(landing.land_issue(Ci, pose, ONE, torch.tensor([0.0]), RIGHT_15, torch.tensor([30.0])), Ci)
    landed = pose.clone()
    landed[0, 1, 0] = 0.15
    Cd = Ci.clone()
    Cd[0, landing.N] = 1  # the clock's last control step, already on the target
    r, C2, tm = step(Cd, landed, _s(landed))
    assert bool(tm["deadline"]) and float(tm["G"]) == pytest.approx(1.0)
    assert float(r) == pytest.approx(0.0875 + 1.25)
    assert C2[0, landing.N] == 0 and torch.allclose(C2[0, 6:], landed.flatten())  # HOLD on the landed stance


def test_settle_charges_sole_travel():
    cfg = LandingCfg()
    C, pose = _hold_state()
    Cs = landing.land_reset(cfg, C, pose, ONE, torch.tensor([0.0]))
    lifted = torch.tensor([[0.03, 0.0]])
    v = torch.zeros(1, 2, 2)
    v[0, 0, 0] = 1.0
    r, _, _ = step(Cs, pose, _s(pose, zc=lifted, v=v))  # the gate is off: the travel term is the whole charge
    assert float(r) == pytest.approx(0.0875 - 0.05, abs=1e-4)
    r, _, _ = step(Cs, pose, _s(pose, zc=lifted, wz=torch.tensor([[2.0, 0.0]])))
    assert float(r) == pytest.approx(0.0875 - 20 * landing.YAW_M * 2 * 0.125 * 0.02, abs=1e-4)
    r_moving, _, _ = step(C, pose, _s(pose, zc=lifted, v=v))  # HOLD: no such term
    r_still, _, _ = step(C, pose, _s(pose, zc=lifted))
    assert abs(float(r_moving) - float(r_still)) < 1e-6
    moved = pose.clone()
    moved[0, 0, 0] += 0.10
    for _ in range(25):
        _, Cs, _ = step(Cs, pose, _s(moved))
    assert Cs[0, landing.M] == 0 and torch.allclose(Cs[0, 6:], moved.flatten())


def test_each_reward_term_alone():
    """Spec 12: from rest at the ceiling of each phase, one regularizer or stance input at a time moves the reward by
    the design's amount per control step (weight x 0.125 x 0.02) in the phases it acts in, and by nothing in the
    others."""
    cfg = LandingCfg()
    C, pose = _hold_state()
    phases = (C.clone(), C.clone(), C)  # SETTLE, STEP (the clock running, no deadline) and HOLD
    phases[0][:, landing.M], phases[1][:, landing.N] = 5.0, 5.0

    def reward(C, rate=0.0, torque=0.0, jerk=0.0, ankle_t2=0.0, still_v=0.0, tilt=0.0):
        s1 = _s(pose)
        s1["tilt"] = torch.tensor([[tilt, 0.0]])  # the left sole tilted
        x = lambda v: torch.full((1,), v)  # noqa: E731
        return float(landing.land_step(cfg, C, pose, s1, x(rate), x(torque), x(jerk), x(ankle_t2), x(still_v))[0])

    table = [  # the input, then the change of the reward in SETTLE, STEP and HOLD
        ({"rate": 1.0}, (-0.00125, -0.00125, -0.00125)),
        ({"torque": 100.0}, (-0.000125, -0.000125, -0.000125)),
        ({"jerk": 100.0}, (-0.00025, -0.00025, -0.00025)),
        ({"ankle_t2": 400.0}, (0.0, 0.0, -0.00632)),  # -4 x (1 - exp(-1)) x 0.0025
        ({"still_v": 0.3}, (-0.00790, 0.0, -0.00790)),  # -5 x (1 - exp(-1)) x 0.0025
        ({"tilt": 0.1}, (0.0, 0.0, -0.00790)),
    ]
    for kw, change in table:
        for name, C, want in zip(("SETTLE", "STEP", "HOLD"), phases, change):
            assert reward(C, **kw) - reward(C) == pytest.approx(want, abs=2e-5), (name, kw)


def _run(moves, target=0.15, planted=0.0, T=140):
    """Total reward of a built trajectory: the home stance, a right-foot command with clock 40 on state 0 for a
    landing `target` m ahead; moves = (sole, j0, j1, dx, apex), minimum-jerk displacements in x between states j0
    and j1; the planted (left) sole creeps `planted` m over states 25 to 40."""
    C, pose = _hold_state()
    cmd = torch.tensor([[target, -0.2125, 0.0]])
    C = landing.land_issue(C, pose, ONE, torch.tensor([1.0]), cmd, torch.tensor([40.0]))
    j = torch.arange(T + 1, dtype=torch.float32)
    x, h = torch.zeros(T + 1, 2), torch.zeros(T + 1, 2)
    for k, j0, j1, dx, apex in moves:
        s = ((j - j0) / (j1 - j0)).clamp(0, 1)
        x[:, k] += dx * (10 * s**3 - 15 * s**4 + 6 * s**5)
        h[:, k] += apex * 4 * s * (1 - s)
    x[:, 0] += planted * ((j - 25) / 15).clamp(0, 1)
    total, prev = 0.0, pose
    for i in range(1, T + 1):
        p = pose.clone()
        p[0, :, 0] = x[i]
        v = torch.zeros(1, 2, 2)
        v[0, :, 0] = (x[i] - x[i - 1]) / 0.02
        r, C, _ = step(C, prev, _s(p, h[i][None], v))
        total, prev = total + float(r), p
    return total


def test_reward_orders():
    lift = lambda d: [(1, 25, 40, d, 0.03)]  # noqa: E731
    on_target, beyond, short, still = _run(lift(0.15)), _run(lift(0.17)), _run(lift(0.10)), _run([])
    assert on_target > beyond > short > still
    # 40 STEP steps, the bonus, progress less drag
    assert on_target == pytest.approx(140 * 0.1 - 40 * 0.0125 + 1.25 + 0.088, abs=0.02)
    assert on_target - _run([(1, 25, 40, 0.15, 0.0)]) > 0.25  # dragged there: drag, and no progress (0.27)
    assert on_target - _run([(1, 27, 42, 0.15, 0.03)]) > 0.3  # still in the air at the deadline
    assert on_target - _run(lift(0.15), planted=0.03) > 0.5  # the planted sole shuffles 3 cm
    assert on_target - _run(lift(0.15) + [(1, 65, 85, -0.15, 0.03)]) > 0.3  # steps back during the hold


def test_natural_step_is_not_dragging():
    """A step's sole is pitched with a corner at the floor while its centre is high: no drag, full progress. A sole
    sliding flat on the floor drags and earns no progress; a flat sole lifted 3 cm is a clean step."""
    C, pose = _hold_state()
    cmd = torch.tensor([[0.15, -0.2125, 0.0]])
    C = landing.land_issue(C, pose, ONE, torch.tensor([1.0]), cmd, torch.tensor([40.0]))
    c30, s30 = math.cos(math.radians(30)), math.sin(math.radians(30))
    pitched = torch.tensor([[c30, 0.0, s30], [0.0, 1.0, 0.0], [-s30, 0.0, c30]])  # the toe points down
    sole_c = torch.tensor(landing.SOLE_C)
    x0, y0 = 0.05, -0.10625

    def terms(R, zc):
        """drag, progress and the soles' h, zc of one step: the right sole's centre at (x0, y0, zc) moving 1 m/s in
        +x."""
        centre = torch.tensor([[0.0, 0.10625, 0.0], [x0, y0, zc]])
        Rs = torch.stack((torch.eye(3), R))
        bp = (centre - Rs @ sole_c)[None]
        s1 = landing.soles(bp, Rs[None], torch.tensor([[[0.0] * 3, [1.0, 0.0, 0.0]]]), torch.zeros(1, 2, 3))
        pose0 = s1["pose"].clone()
        pose0[0, 1, 0] -= 0.02
        _, _, tm = step(C, pose0, s1)
        return float(tm["drag"]), float(tm["progress"]), s1

    zc_pitch = s30 * landing.SOLE_HX  # the toe corner is at height 0
    drag, progress, s1 = terms(pitched, zc_pitch)
    assert abs(float(s1["h"][0, 1])) < 1e-5 and float(s1["zc"][0, 1]) > 0.05
    assert drag < 0.01 and progress > 0.019
    drag, progress, _ = terms(torch.eye(3), 0.0)
    assert drag > 0.95 and progress < 0.001
    drag, progress, _ = terms(torch.eye(3), 0.03)
    assert drag < 0.01 and progress > 0.019

    def turning(zc):
        """drag of a step in which the right sole turns 2 rad/s about its own centre, at height zc, not moving."""
        centre = torch.tensor([[0.0, 0.10625, 0.0], [x0, y0, zc]])
        bp = (centre - sole_c)[None]
        bw = torch.tensor([[[0.0] * 3, [0.0, 0.0, 2.0]]])
        off = landing._sole_offset(torch.eye(3).expand(1, 2, 3, 3))
        bv = -torch.cross(bw, off, dim=-1)  # the link's origin moves so that the sole's centre stays put
        s1 = landing.soles(bp, torch.eye(3).expand(1, 2, 3, 3), bv, bw)
        assert float(s1["v"].abs().max()) < 1e-6
        return float(step(C, s1["pose"].clone(), s1)[2]["drag"])

    assert abs(turning(0.0) - 0.9656 * landing.YAW_M * 2.0) < 0.01
    assert turning(0.03) < 0.01


def _moved(C, pose, rot=0.0, shift=(0.0, 0.0), mirror=False):
    """The command state and sole poses seen in another world frame, or mirrored left-right (soles and swing
    swapped)."""

    def tf(P):
        x, y, a = P.unbind(-1)
        if mirror:
            y, a = -y, -a
        c, s = math.cos(rot), math.sin(rot)
        return torch.stack((c * x - s * y + shift[0], s * x + c * y + shift[1], landing.wrap(a + rot)), -1)

    T, A, p, C2 = tf(C[:, 3:6]), tf(C[:, 6:].view(-1, 2, 3)), tf(pose), C.clone()
    if mirror:
        A, p, C2[:, landing.SWING] = A.flip(1), p.flip(1), 1 - C[:, landing.SWING]
    C2[:, 3:6], C2[:, 6:] = T, A.flatten(1)
    return C2, p


def test_reward_is_frame_and_side_invariant():
    C, pose = _hold_state()
    cmd = torch.tensor([[0.15, -0.2125, 0.3]])
    C = landing.land_issue(C, pose, ONE, torch.tensor([1.0]), cmd, torch.tensor([1.0]))
    end = pose.clone()
    end[0, 1], end[0, 0, 0] = torch.tensor([0.12, -0.10, 0.25]), 0.01  # 3 cm short, 0.05 rad off; planted 1 cm off
    r0, _, t0 = step(C, pose, _s(end))
    assert 0.2 < float(t0["G"]) < 0.9
    for kw in ({"rot": 3.0, "shift": (2.0, -1.0)}, {"mirror": True}, {"rot": -3.1, "mirror": True}):
        (C2, p0), (_, p1) = _moved(C, pose, **kw), _moved(C, end, **kw)
        r, _, t = step(C2, p0, _s(p1))
        assert float(r) == pytest.approx(float(r0), abs=1e-5)
        assert float(t["G"]) == pytest.approx(float(t0["G"]), abs=1e-5)


def test_gradient_is_finite_on_the_target():
    C, pose = _hold_state()
    cmd = torch.tensor([[0.0, -0.2125, 0.0]])
    C = landing.land_issue(C, pose, ONE, torch.tensor([1.0]), cmd, torch.tensor([1.0]))
    p1 = pose.clone().requires_grad_(True)  # every distance is exactly zero
    step(C, pose, _s(p1))[0].sum().backward()
    assert torch.isfinite(p1.grad).all()


def test_reset_clears_a_running_clock():
    C, pose = _hold_state(B=2)
    both = torch.tensor([True, True])
    C = landing.land_issue(C, pose, both, torch.ones(2), RIGHT_15.repeat(2, 1), torch.full((2,), 40.0))
    C2 = landing.land_reset(LandingCfg(), C, pose, torch.tensor([True, False]), torch.zeros(2))
    assert (C2[0, landing.N], C2[0, landing.M]) == (0, 25) and torch.equal(C2[1], C[1])


def _rect_gap(x, y, a):
    """Distance between the planted sole's rectangle (at the origin) and a landed one at (x, y, a), on boundary
    points."""

    def border(x, y, a):
        t = np.linspace(-1, 1, 81)
        long_, short = np.full_like(t, landing.SOLE_HX), np.full_like(t, landing.SOLE_HY)
        pts = np.concatenate(
            [np.stack((t * long_, s * short), 1) for s in (-1, 1)]
            + [np.stack((s * long_, t * short), 1) for s in (-1, 1)]
        )
        return pts @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]) + [x, y]

    A, B = border(0, 0, 0), border(x, y, a)
    return np.linalg.norm(A[:, None] - B[None], axis=-1).min()


def test_sampler():
    cfg = LandingCfg()
    torch.manual_seed(0)
    B = 20000
    _, pose = _hold_state(B)
    u = torch.rand(B, 3 + 3 * cfg.candidates)
    swing, c, n0, ok = landing.land_sample(cfg, pose, u)
    sg = 1 - 2 * swing  # +1 for a left swing: mirror y and yaw onto the left
    x, y, a, e = c[ok, 0], (sg * c[:, 1])[ok], (sg * c[:, 2])[ok], 1e-5
    assert ok.float().mean() > 0.95 and 0.45 < swing.mean() < 0.55
    assert (x.abs() <= cfg.box_x + e).all() and (y >= cfg.box_y[0] - e).all() and (y <= cfg.box_y[1] + e).all()
    assert (a >= cfg.box_yaw[0] - e).all() and (a <= cfg.box_yaw[1] + e).all()
    travel = torch.hypot(x, y - 0.2125) + landing.YAW_M * a.abs()
    assert (travel >= 0.05 - e).all() and (travel <= 0.60 + e).all()
    assert 0.30 < (travel < 0.15).float().mean() < 0.50  # the near half keeps short commands common
    assert ((n0 - n0.round()).abs() < 1e-6).all() and (n0[ok] >= 35).all() and (n0[ok] <= 67).all()
    for i in torch.nonzero(ok)[:200, 0]:
        assert _rect_gap(float(c[i, 0]), float(sg[i] * c[i, 1]), float(sg[i] * c[i, 2])) >= 0.02 - 1e-3
    u2 = u.clone()
    u2[:, 0] = 1 - u[:, 0]  # the same draw for the other foot is the mirror image
    s2, c2, n2, ok2 = landing.land_sample(cfg, pose, u2)
    assert torch.equal(s2, 1 - swing) and torch.equal(ok2, ok) and torch.equal(n2[ok], n0[ok])
    assert torch.allclose(c2[ok], c[ok] * torch.tensor([1.0, -1.0, -1.0]), atol=1e-6)


def test_sampler_from_a_stance_outside_the_box():
    cfg = LandingCfg()
    torch.manual_seed(1)
    _, pose = _hold_state(4000, width=0.70)  # a landing that went wide, 70 cm between the soles
    pose[:, 0, 0] = 0.50  # and the left sole half a metre ahead
    swing, c, n0, ok = landing.land_sample(cfg, pose, torch.rand(4000, 3 + 3 * cfg.candidates))
    sg = 1 - 2 * swing
    assert ok.any() and (c[ok, 0].abs() <= cfg.box_x + 1e-5).all()
    assert ((sg * c[:, 1])[ok] <= cfg.box_y[1] + 1e-5).all()


def test_sampler_sends_the_first_candidate_that_passes():
    cfg = LandingCfg()
    _, pose = _hold_state()
    lo = torch.tensor([-cfg.box_x, cfg.box_y[0], cfg.box_yaw[0]])
    hi = torch.tensor([cfg.box_x, cfg.box_y[1], cfg.box_yaw[1]])
    draw = lambda c: ((torch.tensor(c) - lo) / (hi - lo)).tolist()  # noqa: E731  the anywhere draw that lands on c
    stay, ahead, behind = draw([0.0, 0.2125, 0.0]), draw([0.15, 0.2125, 0.0]), draw([-0.15, 0.2125, 0.0])
    # a left swing, an anywhere draw, no clock jitter; candidate 0 is the stance now (travel 0: fails), candidate 1
    # lands 15 cm ahead and the other six 15 cm behind (all pass)
    u = torch.tensor([[0.2, 0.9, 0.5] + stay + ahead + behind * (cfg.candidates - 2)])
    swing, c, n0, ok = landing.land_sample(cfg, pose, u)
    assert (float(swing), bool(ok), float(n0)) == (0.0, True, 47.0)  # n0 = 44 + 20 x 0.15
    assert torch.allclose(c, torch.tensor([[0.15, 0.2125, 0.0]]), atol=1e-6)


def test_features():
    cfg = LandingCfg()
    C, pose = _hold_state()
    rel5 = torch.tensor([[0.02, -0.21, 0.01, 1.0, 0.0]])
    f = landing.land_features(cfg, C, rel5)  # holding: the target is the stance held
    assert torch.allclose(f[0], torch.tensor([0, 1, 0.0, -0.425, 1, 0, 0, 0.04, -0.42, 0.02, 1, 0]), atol=1e-6)
    Ci = landing.land_issue(C, pose, ONE, torch.tensor([1.0]), torch.tensor([[0.15, -0.25, 0.3]]), torch.tensor([48.0]))
    f = landing.land_features(cfg, Ci, rel5)
    assert torch.allclose(f[0, :7], torch.tensor([0, 1, 0.3, -0.5, math.cos(0.3), math.sin(0.3), 0.8]), atol=1e-6)
    assert torch.equal(landing.land_features(LandingCfg(derived=False), Ci, rel5)[0, 7:], torch.zeros(5))
    x = landing.land_critic_features(Ci, _s(pose))
    want = torch.tensor(
        [0, 1, 0.96, 0, 0, -0.15, 0.0375, math.sin(-0.3), math.cos(-0.3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    assert x.shape == (1, landing.CRITIC_DIM) and torch.allclose(x[0], want, atol=1e-6)


def test_issue_latches_constants():
    C, pose = _hold_state()
    p = pose.clone().requires_grad_(True)
    Ci = landing.land_issue(C, p, ONE, torch.tensor([1.0]), RIGHT_15, torch.tensor([40.0]))
    assert not Ci.requires_grad  # the anchor and the target are constants of the reward


def test_no_host_constants_in_landing():
    """src/landing.py runs inside captured CUDA graphs: a tensor built from Python data is a host-to-device copy."""
    import inspect

    src = inspect.getsource(landing)
    for banned in ("torch.tensor(", "as_tensor(", "new_tensor("):
        assert banned not in src, banned


def _mdp():
    from src.dr import XiLayout
    from src.mdp import MDP
    from src.motion import MotionLib

    q0 = torch.tensor(SPEC.default_joint_pos_list())
    stand = (
        torch.tensor(SPEC.robot_cfg().init_state.pos).repeat(100, 1),
        torch.tensor([1.0, 0, 0, 0]).repeat(100, 1),
        q0.repeat(100, 1),
    )
    motion = MotionLib.from_clips(["stand"], *([x] for x in stand), KIN, "cpu")
    mdp = MDP(KIN, motion, q0, XiLayout(SPEC, "cpu").dim, SPEC.sensing)
    mdp.feet, mdp.contact_z = FEET, SPEC.contact_z
    mdp.enable_velocity(SPEC)
    mdp.enable_landing(SPEC)
    return mdp, motion


def test_mdp_landing_on_a_standing_robot():
    mdp, motion = _mdp()
    J, f = SPEC.num_joints, torch.tensor([50])
    S = motion.ref_state(f)
    sole, fv = mdp.sole_kin(S)
    assert sole["pose"].shape == (1, 2, 3) and fv.shape == (1, 2, 3) and len(mdp.leg_idx) == 12
    # the default pose stands on flat soles near the floor
    assert sole["h"].abs().max() < 0.02 and sole["tilt"].max() < 0.05
    C = landing.land_reset(mdp.land, torch.zeros(1, landing.STATE_DIM), sole["pose"], ONE, torch.tensor([0.0]))
    C[:, landing.M] = 0  # holding the stance it stands in
    a, a_prev, xi = torch.zeros(1, J), torch.zeros(1, 2 * J), torch.zeros(1, mdp.xi_dim)
    r, C1, pose1, fv1, tm = mdp.land_step(S, C, sole["pose"], a, a_prev, xi, fv, fv, ONE)
    # the ceiling: the soles are flat and on their anchors, the robot still
    assert float(r) == pytest.approx(0.1, abs=1e-4) and torch.equal(C1, C) and torch.equal(pose1, sole["pose"])
    o = mdp.obs(S, a_prev, xi, f, ONE)
    feat = mdp.land_features(C, o)
    rel = relative(sole["pose"][:, 0], sole["pose"][:, 1])
    # the observed joints give the stance
    assert feat.shape == (1, 12) and torch.allclose(feat[0, 7:9], rel[0, :2] / 0.5, atol=1e-3)
    assert mdp.land_critic_features(S, C, xi, a_prev).shape == (1, 71 + mdp.xi_dim + 2 * J + landing.CRITIC_DIM)


def test_mdp_land_features_away_from_the_default_pose():
    """The 5 derived inputs are sole_rel of the joints as the observation reports them (mdp.obs: the true joints less
    the encoder offset xi carries), whatever the root's pose."""
    from src.dr import XiLayout
    from utils.rotations import so3_exp

    mdp, _ = _mdp()
    torch.manual_seed(0)
    B, J = 6, SPEC.num_joints
    q = torch.clamp(mdp.default_q + 0.3 * torch.randn(B, J), KIN.q_lo, KIN.q_hi)
    z3 = torch.zeros(B, 3)
    S = (torch.tensor([0.3, -0.2, 0.68]).repeat(B, 1), so3_exp(0.3 * torch.randn(B, 3)), z3, z3, q, torch.zeros(B, J))
    xi = torch.zeros(B, mdp.xi_dim)
    offset = 0.01 * (2 * torch.randint(0, 2, (B, J)) - 1)  # 0.01 rad on every joint, either way
    xi[:, XiLayout(SPEC, "cpu").slices["joint_offset"]] = offset
    o = mdp.obs(S, torch.zeros(B, 2 * J), xi, torch.zeros(B, dtype=torch.long), torch.ones(B, dtype=torch.bool))
    C = torch.zeros(B, landing.STATE_DIM)
    C[:, landing.SWING] = (torch.arange(B) % 2).float()
    feat = mdp.land_features(C, o)
    bp, bR, _, _ = KIN.forward(z3, torch.eye(3).expand(B, 3, 3), z3, z3, q - offset, torch.zeros(B, J))
    want = landing.sole_rel(bp[:, FEET], bR[:, FEET], C[:, landing.SWING])
    assert torch.allclose(torch.cat((feat[:, 7:10] * 0.5, feat[:, 10:]), -1), want, atol=1e-5)


def test_policy_and_buffer():
    from src.buffer import ReplayBuffer
    from src.mdp import state_dim
    from src.networks import Policy, policy_kwargs

    J, hist = SPEC.num_joints, 10
    obs_dim = 3 * J + 12
    pi = Policy(obs_dim, 730, J, hist=hist, vel=True, land=True)
    assert pi.enc_land[0].in_features == landing.CMD_DIM + obs_dim * hist and pi.enc_land[-1].out_features == 32
    assert Policy(obs_dim, 730, J, vel=True).enc_land is None
    assert policy_kwargs({"vel_frac": 0.125, "land_frac": 0.75}, SPEC) == {"ref_skip": None, "vel": True, "land": True}
    assert policy_kwargs({"vel_frac": 0.5}, SPEC)["land"] is False
    oh, cf = torch.randn(4, hist, obs_dim), torch.randn(4, landing.CMD_DIM)
    pi.decode_frozen(pi.encode_land(cf, oh), oh).sum().backward()
    grads = lambda net: [w.grad for w in net.parameters()]  # noqa: E731
    assert all(g is not None and g.abs().sum() > 0 for g in grads(pi.enc_land))
    assert all(g is None for g in grads(pi.dec) + grads(pi.enc) + grads(pi.enc_vel))
    T, N, xi_dim = 6, 10, 38
    buf = ReplayBuffer(T, N, "cpu", J, xi_dim, a_dim=2 * J, vel=True, land=True)
    env_id = torch.arange(N, dtype=torch.float32)
    for t in range(T):
        buf.add({"s": torch.zeros(N, state_dim(J)), "s1": torch.zeros(N, state_dim(J)), "a": torch.zeros(N, J),
                 "a_prev": torch.zeros(N, 2 * J), "xi": env_id.unsqueeze(-1).expand(-1, xi_dim), "r": torch.zeros(N),
                 "f": torch.zeros(N, dtype=torch.long), "ep_step": torch.full((N,), t),
                 "done": torch.zeros(N, dtype=torch.bool), "cut": torch.zeros(N, dtype=torch.bool),
                 "cmd": torch.zeros(N, 3), "cmd_ref": torch.zeros(N, dtype=torch.bool),
                 "land": env_id.unsqueeze(-1).expand(-1, 12), "lab": env_id % 2 == 0})  # fmt: skip
    s, h = buf.sample_land_starts(2000, 3, 2, 5)  # envs 2 .. 6
    assert s["xi"][:, 0].min() == 2 and s["xi"][:, 0].max() == 6 and torch.equal(s["land"][:, 0], s["xi"][:, 0])
    assert torch.equal(s["lab"], s["xi"][:, 0] % 2 == 0) and h["s"].shape == (2000, 3, state_dim(J))
    assert "land" not in ReplayBuffer(T, N, "cpu", J, xi_dim, a_dim=2 * J, vel=True).fields


BASE = "logs/t1_vel_ankle_sharp_x2_s1/policy/ckpt_013000.pt"


@pytest.mark.skipif(not os.path.exists(BASE), reason="needs the deployed run's checkpoint")
def test_base_checkpoint_loads_and_stays_frozen():
    from src.networks import Policy
    from src.optim import MuonAdam

    J = SPEC.num_joints
    pi = Policy(3 * J + 12, 730, J, hist=10, vel=True, land=True)
    miss = pi.load_state_dict(torch.load(BASE, map_location="cpu", weights_only=False)["pi"], strict=False)
    assert not miss.unexpected_keys and miss.missing_keys and all(k.startswith("enc_land.") for k in miss.missing_keys)
    for net in (pi.enc, pi.dec, pi.enc_vel):
        for p in net.parameters():
            p.requires_grad_(False)
    base = {k: v.clone() for k, v in pi.state_dict().items() if not k.startswith("enc_land.")}
    before = [p.clone() for p in pi.enc_land.parameters()]
    opt = MuonAdam(pi, [pi.enc_land], lr=1e-3, fused=False)
    for _ in range(3):
        oh, cf = torch.randn(8, 10, 3 * J + 12), torch.randn(8, landing.CMD_DIM)
        opt.zero_grad()
        pi.decode_frozen(pi.encode_land(cf, oh), oh).pow(2).mean().backward()
        opt.step()
    sd = pi.state_dict()
    assert all(torch.equal(sd[k], v) for k, v in base.items())  # the deployed tensors, bit for bit
    assert all(not torch.equal(p, b) for p, b in zip(pi.enc_land.parameters(), before))


def test_test_grid_and_score():
    from utils.landing_score import command_grid, score

    cfg = LandingCfg()
    _, pose = _hold_state()
    g = command_grid(cfg, pose[0])
    K = len(g["swing"])
    assert K > 150 and 0.4 < g["swing"].mean() < 0.6 and g["travel"].max() <= 0.60 and g["travel"].min() >= 0.05
    sg = 1 - 2 * g["swing"]
    assert (g["c"][:, 0].abs() <= cfg.box_x + 1e-5).all() and (sg * g["c"][:, 1] >= cfg.box_y[0] - 1e-5).all()
    # the pure turns: 30 and 45 degrees, both feet
    assert int((g["direction"] == -1).sum()) == 4 and (g["n0"] == g["n0"].round()).all()
    # two built rollouts of the same right-foot command: a clean one, and one that slides and then falls
    T, t_cmd, n0 = 300, 50, 45
    cmd = {k: v[:2].clone() for k, v in g.items()}
    cmd["swing"][:], cmd["n0"][:] = 1.0, float(n0)
    cmd["c"][:] = torch.tensor([0.15, -0.2125, 0.0])
    log = {"pose": pose.repeat(T, 2, 1, 1).clone(), "h": torch.zeros(T, 2, 2),
           "fell": torch.zeros(T, 2, dtype=torch.bool), "qd_rms": torch.zeros(T, 2),
           "root_speed": torch.zeros(T, 2), "t_cmd": t_cmd,
           "target": torch.tensor([[0.15, -0.10625, 0.0]] * 2), "anchor": pose.repeat(2, 1, 1)}  # fmt: skip
    log["zc"] = log["h"].clone()  # flat soles
    s = ((torch.arange(T) - (t_cmd + 25)) / 15.0).clamp(0, 1)
    b = 10 * s**3 - 15 * s**4 + 6 * s**5
    log["pose"][:, 0, 1, 0], log["pose"][:, 1, 1, 0] = 0.15 * b, 0.11 * b  # the second lands 4 cm short
    log["h"][:, 0, 1] = 0.03 * 4 * s * (1 - s)  # only the first lifts
    log["zc"] = log["h"].clone()
    log["fell"][200:, 1] = True
    out = score(log, cmd, cfg)
    per = out["per"]
    assert per["arrived"].tolist() == [True, True]
    assert per["lifted"].tolist() == [True, False] and per["fell"].tolist() == [False, True]
    assert float(per["err_cm"][0]) < 0.1 and float(per["err_cm"][1]) == pytest.approx(4.0, abs=0.1)
    assert float(per["apex_cm"][0]) == pytest.approx(3.0, abs=0.05)
    assert float(per["floor_cm"][1]) == pytest.approx(11.0, abs=0.3)
    assert float(per["travel_cm"][0]) == pytest.approx(15.0, abs=0.3)
    assert float(per["travel_cm"][1]) == pytest.approx(11.0, abs=0.3)
    assert float(per["scuff_cm"][1]) == pytest.approx(11.0, abs=0.3) and float(per["scuff_cm"][0]) < 1.0
    assert 0 < out["summary"]["scuff_share"] < 1
    assert math.isfinite(out["summary"]["apex_cm_median"]) and math.isfinite(out["summary"]["floor_cm_median"])
    assert per["still_ok"].tolist() == [True, True] and out["summary"]["falls"] == 1
    assert per["stance_ok"].tolist() == [True, True] and out["summary"]["stance_ok"] == 1.0
    # the natural step: the lowest corner stays on the floor while the centre rises 3 cm
    nat = {**log, "pose": log["pose"][:, :1], "h": torch.zeros(T, 1, 2), "zc": log["zc"][:, :1],
           "fell": log["fell"][:, :1], "qd_rms": log["qd_rms"][:, :1], "root_speed": log["root_speed"][:, :1],
           "target": log["target"][:1], "anchor": log["anchor"][:1]}  # fmt: skip
    sn = score(nat, {k: v[:1] for k, v in cmd.items()}, cfg)
    pn = sn["per"]
    assert pn["lifted"].tolist() == [True] and float(pn["apex_cm"][0]) == pytest.approx(3.0, abs=0.05)
    assert float(pn["floor_cm"][0]) < 2.0 and float(pn["scuff_cm"][0]) == pytest.approx(15.0, abs=0.3)
    assert out["summary"]["pass"] is False and sn["summary"]["pass"] is True
    # the left sole is turned by 1.4 rad before the command: the stance is not held. The rig latches the anchors on
    # the soles of state t_cmd, so the planted sole is kept and the stance line alone fails
    turned = {**nat, "pose": nat["pose"].clone()}
    turned["pose"][10:, 0, 0, 2] = 1.4
    turned["anchor"] = turned["pose"][t_cmd].clone()
    ts = score(turned, {k: v[:1] for k, v in cmd.items()}, cfg)
    assert ts["per"]["stance_ok"].tolist() == [False] and ts["per"]["planted_ok"].tolist() == [True]
    assert float(ts["per"]["stance_rad"][0]) == pytest.approx(1.4, abs=1e-3) and float(ts["per"]["stance_cm"][0]) < 0.1
    assert ts["summary"]["pass"] is False
    bad = pose.clone()
    bad[0, 0, 2] = 1.4
    with pytest.raises(AssertionError):
        command_grid(cfg, bad[0])
    from utils.landing_score import jsonable

    keys = ("pose", "h", "zc", "fell", "qd_rms", "root_speed")
    cut = lambda n: {**log, **{k: log[k][:n] for k in keys}}  # noqa: E731  the log's first n states
    log2 = cut(t_cmd + n0 + 175)  # the rig's 3.5 s hold: still_s resolves to 3 s
    assert score(log2, cmd, cfg)["per"]["still_ok"].tolist() == [True, True]
    with pytest.raises(AssertionError, match="--hold_s"):  # one state too short for the 3 s still window
        score(cut(t_cmd + n0 + 174), cmd, cfg)
    summ = out["summary"]
    json.dumps(jsonable(summ), allow_nan=False)  # empty travel and direction groups are nan before the cleaning


def test_score_library():
    from utils.landing_score import jsonable, score_library

    # five commands: the swing sole's error at the deadline, the command's place in its episode and its travel; five
    # episodes whose SETTLE ended, one of which moved 10 cm before its anchors latched; four that ended, one by a fall
    log = {"err": torch.tensor([0.01, 0.03, 0.08, 0.02, 0.04]), "yaw": torch.tensor([0.0, 0.05, 0.0, 0.2, 0.01]),
           "place": torch.tensor([1, 1, 1, 2, 5]), "travel": torch.tensor([0.08, 0.12, 0.25, 0.50, 0.17]),
           "moved": torch.tensor([0.0, 0.01, 0.10, 0.005, 0.015]), "turned": torch.tensor([0.0, 0.02, 0.0, 0.05, 0.08]),
           "falls": 1, "episodes": 4}  # fmt: skip
    s = score_library(log)
    deg = math.degrees
    want = {"commands": 5, "episodes": 4, "falls": 1, "stance_kept": 0.8, "moved_cm_median": 1.0,
            "moved_cm_p95": 1.5 + 0.8 * 8.5, "arrived": 0.6, "err_cm_median": 3.0, "err_cm_p95": 4.0 + 0.8 * 4.0,
            "err_deg_median": deg(0.01), "err_deg_p95": deg(0.05) + 0.8 * (deg(0.2) - deg(0.05))}  # fmt: skip
    assert {k: s[k] for k in want} == pytest.approx(want, abs=1e-4) and s["pass"] is False
    nan = math.nan
    # n, arrived, the median and the 95th percentile of the error [cm]
    place = {"first": (3, 2 / 3, 3.0, 3.0 + 0.9 * 5.0), "second": (1, 0.0, 2.0, 2.0), "third": (0, nan, nan, nan),
             "fourth or later": (1, 1.0, 4.0, 4.0)}  # fmt: skip
    travel = {"5-10 cm": (1, 1.0, 1.0, 1.0), "10-15 cm": (1, 1.0, 3.0, 3.0), "15-20 cm": (1, 1.0, 4.0, 4.0),
              "20-30 cm": (1, 0.0, 8.0, 8.0), "30-45 cm": (0, nan, nan, nan),
              "45-61 cm": (1, 0.0, 2.0, 2.0)}  # fmt: skip
    for got, groups in ((s["by_place"], place), (s["by_travel"], travel)):
        assert list(got) == list(groups)
        for name, g in groups.items():
            row = tuple(got[name][k] for k in ("n", "arrived", "err_cm_median", "err_cm_p95"))
            assert row == pytest.approx(g, abs=1e-4, nan_ok=True), name
    json.dumps(jsonable(s), allow_nan=False)
    # without the episode that moved and the fall: the stance is kept everywhere and the part passes
    kept = {**log, "moved": log["moved"][[0, 1, 3, 4]], "turned": log["turned"][[0, 1, 3, 4]], "falls": 0}
    s = score_library(kept)
    assert s["stance_kept"] == 1.0 and s["pass"] is True
    # either condition alone fails the part: only the fall removed, or only the moved episode
    assert score_library({**log, "falls": 0})["pass"] is False
    assert score_library({**kept, "falls": 1})["pass"] is False


def test_library_file(tmp_path):
    """A checkpoint's library paths: the saved one where it exists, else data/<robot>/<its name> of the repository."""
    from utils.landing_score import library_file

    saved, local = tmp_path / "node" / "keep.txt", tmp_path / "repo" / "data" / "t1" / "keep.txt"
    saved.parent.mkdir(parents=True)
    saved.write_text("clip")
    assert library_file(str(saved), "t1", tmp_path / "repo") == str(saved)
    saved.unlink()
    with pytest.raises(FileNotFoundError, match="--no_library") as e:
        library_file(str(saved), "t1", tmp_path / "repo")
    assert str(saved) in str(e.value) and str(local) in str(e.value)
    local.parent.mkdir(parents=True)
    local.write_text("clip")
    assert library_file(str(saved), "t1", tmp_path / "repo") == str(local)
