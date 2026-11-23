"""utils/play_native.py: an ONNX velocity policy in plain MuJoCo on the robot it was trained for. Its model, motor law
and observation are checked against the training simulator's (CPU only: run with CUDA_VISIBLE_DEVICES= )."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(4)

from robot.spec import get_robot  # noqa: E402
from test_t2_measured import all_worlds, make_env  # noqa: E402  (the CPU training-simulator fixture)
from utils import play_native as N  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ONNX = REPO / "deploy" / "t2al1_vel_s1_013000.onnx"  # the measured robot's policy, the file for the robot
ONNX_T2 = REPO / "logs" / "t2_vel_stand_s3" / "policy" / "ckpt_013000.onnx"  # the earlier t2 policy


def test_the_model_is_the_training_simulators():
    robot = get_robot("t2_measured")
    m, t = N.build_model(robot), make_env("t2_measured", 1, eval_mode=True).sim.mj_model
    for f in ("timestep", "integrator", "cone", "solver", "iterations", "ls_iterations", "tolerance", "impratio"):
        assert getattr(m.opt, f) == getattr(t.opt, f), f
    for f in ("dof_frictionloss", "dof_armature", "dof_damping", "body_mass", "body_inertia", "geom_friction", "geom_size",
              "actuator_ctrlrange", "jnt_range", "qpos0"):  # fmt: skip
        np.testing.assert_array_equal(getattr(m, f), getattr(t, f), err_msg=f)
    assert (m.nq, m.nv, m.nu) == (38, 37, 31) and m.dof_frictionloss.max() == pytest.approx(3.3)


def test_the_motor_law_is_the_training_actuators():
    from mjlab.actuator.actuator import ActuatorCmd
    from robot.t1 import BoosterPdActuator

    sim = N.NativeSim(get_robot("t2_measured"))
    g = torch.Generator().manual_seed(0)
    q, target = torch.randn(64, 31, generator=g) * 0.5, torch.randn(64, 31, generator=g) * 0.5
    qd = torch.randn(64, 31, generator=g) * 12.0  # past the knee of the torque-speed curve on many joints
    f = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32).expand(64, 31)  # noqa: E731
    params = {"stiffness": f(sim.kp), "damping": f(sim.kd), "force_limit": f(sim.effort), "v_max": f(sim.v_max), "denom": f(sim.denom)}
    cmd = ActuatorCmd(position_target=target, velocity_target=torch.zeros_like(q), effort_target=torch.zeros_like(q), pos=q, vel=qd)
    want = BoosterPdActuator.control_law(params, cmd)
    got = N.motor_torque(q.numpy(), qd.numpy(), target.numpy(), sim.kp, sim.kd, sim.effort, sim.v_max, sim.denom)
    np.testing.assert_allclose(got, want.numpy(), atol=1e-4, rtol=1e-6)
    raw = sim.kp * (target.numpy() - q.numpy()) - sim.kd * qd.numpy()
    assert (np.abs(raw - got) > 1e-6).mean() > 0.05  # the torque and speed limits are exercised, not only the PD term


def test_the_observation_is_trainings():
    env = make_env("t2_measured", 1, eval_mode=True)
    env.reset(*all_worlds(env))
    torch.manual_seed(0)
    for _ in range(5):  # off the default pose, moving
        a = 0.5 * torch.randn(1, 31)
        env.step(a)
    S = env.state()
    want = env.mdp.obs(S, env.a_prev, env.xi, env.f, torch.ones(1, dtype=torch.bool))[0, :99].numpy()
    sim = N.NativeSim(get_robot("t2_measured"))
    sim.d.qpos[:], sim.d.qvel[:] = env.sim.data.qpos[0].numpy(), env.sim.data.qvel[0].numpy()
    got = N.observation(sim, a[0].numpy())
    np.testing.assert_allclose(got, want, atol=1e-5, rtol=0)
    assert got.shape == (99,) and abs(got[62:65]).max() > 0.01  # the gyro entries are not trivially zero


def test_the_delay_applies_the_setpoint_written_lag_steps_earlier():
    sim = N.NativeSim(get_robot("t2_measured"), delay=2)
    sim.reset()
    seen = []
    for k in range(5):
        seen.append(sim.delayed(np.full(31, float(k)))[0])
    assert seen == [0.0, 0.0, 0.0, 1.0, 2.0]  # the oldest one until 2 older exist (TrackingSim._delayed)


def test_it_picks_the_robot_from_the_file_and_refuses_a_mismatch():
    if not (ONNX.exists() and ONNX_T2.exists()):
        pytest.skip("exports missing")
    assert N.robot_of(N.load_metadata(str(ONNX))).name == "t2_measured"
    assert N.robot_of(N.load_metadata(str(ONNX_T2))).name == "t2"
    with pytest.raises(SystemExit, match="t2_measured"):
        N.robot_of(N.load_metadata(str(ONNX_T2)), "t2_measured")  # another robot's gains


def test_it_walks_as_in_the_training_simulator():
    """The scripted drive in plain MuJoCo: no fall, and each segment's velocity near the training simulator's
    (utils/play_velocity.py --script on the same policy, 2026-11-21)."""
    if not ONNX.exists():
        pytest.skip("export missing")
    rows, falls = N.run_script(str(ONNX), quiet=True)
    training = [(0.02, 0.00, -0.01), (0.43, 0.00, 0.00), (0.98, -0.01, -0.01), (-0.45, 0.01, 0.00), (-0.04, 0.28, 0.01),
                (0.00, -0.01, 0.97), (0.60, 0.02, -1.00), (0.00, 0.00, -0.01)]  # fmt: skip
    assert falls == 0
    np.testing.assert_allclose(np.array(rows), np.array(training), atol=0.12)


def test_the_viewer_loop_and_the_keys(monkeypatch):
    """run_viewer with a stand-in for the window: the keys change the command, the command stays inside the policy's
    box, space stops, backspace resets the robot."""
    if not ONNX.exists():
        pytest.skip("export missing")
    import glfw
    import mujoco.viewer

    seen = {"texts": [], "ticks": 0, "sims": []}

    class Window:
        cam = type("Cam", (), {})()

        def __init__(self, key):
            self.key = key

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def is_running(self):
            seen["ticks"] += 1
            t = seen["ticks"]
            presses = {2: [glfw.KEY_UP] * 3 + [glfw.KEY_A, glfw.KEY_LEFT], 4: [glfw.KEY_UP] * 40, 6: [glfw.KEY_SPACE],
                       8: [glfw.KEY_DOWN], 9: [glfw.KEY_BACKSPACE]}  # fmt: skip
            for k in presses.get(t, []):
                self.key(k)
            return t <= 10

        def set_texts(self, texts):
            seen["texts"].append(texts[3])

        def sync(self):
            pass

    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda m, d, key_callback: Window(key_callback))
    monkeypatch.setattr(N.time, "sleep", lambda s: None)
    N.run_viewer(str(ONNX))
    t = seen["texts"]
    assert len(t) == 10
    assert t[0].startswith("cmd vx +0.0 vy +0.0 yaw +0.0")
    assert t[1].startswith("cmd vx +0.3 vy +0.1 yaw +0.2")  # three ups, one a, one left
    assert t[3].startswith("cmd vx +1.9 vy +0.1 yaw +0.2")  # forty more ups: capped at the box (1.90 m/s)
    assert t[5].startswith("cmd vx +0.0 vy +0.0 yaw +0.0")  # space
    assert t[7].startswith("cmd vx -0.1")  # down
    assert t[8].startswith("cmd vx +0.0 vy +0.0 yaw +0.0")  # backspace: stopped and reset
    assert "act vx" in t[8]


# ---- per-group delays and variants (docs/superpowers/specs/2026-11-20-t2-deploy-design.md, section 1)
def test_each_group_gets_its_own_lag():
    robot = get_robot("t2_measured")
    sim = N.NativeSim(robot, delay=(0, 2, 4, 6, 8))
    sim.reset()
    out = [sim.delayed(np.full(31, float(k))) for k in range(10)][-1]  # after 10 writes: k = 9
    for lag, joints in zip((0, 2, 4, 6, 8), robot.delay_groups):
        assert np.all(out[list(joints)] == 9 - lag)
    one = N.NativeSim(robot, delay=3)
    one.reset()
    assert np.all([one.delayed(np.full(31, float(k))) for k in range(6)][-1] == 2)
    with pytest.raises(ValueError):
        N.NativeSim(robot, delay=(1, 2))  # not one lag per group


def test_pair_torque_cases():
    J, kp = np.array(N.ANKLE_J), np.array([141.7042, 52.6468])
    e, qd = np.array([0.1, 0.0]), np.array([0.0, 0.0])
    t1, _ = N.pair_torque("H1", 0, e, qd, J, kp, 1.7215, 0.69)
    np.testing.assert_allclose(t1, kp * e, atol=1e-9)  # kp at the joint
    t2, _ = N.pair_torque("H2", 0, e, qd, J, kp, 1.7215, 0.69)
    np.testing.assert_allclose(t2, (J.T @ np.diag(kp) @ J) @ e, atol=1e-9)  # kp per motor
    assert abs(t2[0] / t1[0] - 1.79) < 0.02 and abs(t2[1]) > 5  # 1.8x stiffer, and coupled
    qd = np.array([1.0, 0.0])
    d1, _ = N.pair_torque("H1", 0, 0 * e, qd, J, kp, 1.7215, 0.69)
    np.testing.assert_allclose(d1, -0.69 * 1.7215 * (J.T @ J) @ qd, atol=1e-9)  # kd per motor, at the factor
    d3, _ = N.pair_torque("H3", 0, 0 * e, qd, J, kp, 1.7215, 0.69)
    np.testing.assert_allclose(d3, -1.7215 * qd, atol=1e-9)  # kd at the joint
    big, peak = N.pair_torque("H2", 0, np.array([2.0, 0.0]), 0 * qd, J, kp, 1.7215, 0.69)
    assert peak > 60 and np.abs(np.linalg.solve(J.T, big)).max() <= 60 + 1e-9  # each motor limited


def test_variants_change_what_they_say():
    robot = get_robot("t2_measured")
    base = N.NativeSim(robot)
    assert set(N.VARIANTS) >= {"nominal", "gains_x0.8", "gains_x1.2", "latency", "push", "no_friction", "H1f1", "H2", "H3", "H4"}
    hi = N.NativeSim(robot, variant="gains_x1.2")
    np.testing.assert_allclose(hi.kp, 1.2 * base.kp)
    np.testing.assert_allclose(hi.kd, 1.2 * base.kd)
    assert N.NativeSim(robot, variant="no_friction").m.dof_frictionloss.max() == 0 and base.m.dof_frictionloss.max() > 3
    with pytest.raises(ValueError):
        N.NativeSim(robot, variant="nope")
    # a hypothesis changes the six linkage joints' torques and nothing else
    h2 = N.NativeSim(robot, variant="H2")
    target = base.default_q + 0.05
    link = [list(robot.joint_names).index(n) for p in N.LINKAGE_PAIRS for n in p[:2]]
    rest = [j for j in range(31) if j not in link]
    tb, th = base.torque(target), h2.torque(target)  # both at the standing reset
    assert np.abs(tb[link] - th[link]).max() > 1.0 and h2.peak_motor > 0 and base.peak_motor == 0
    np.testing.assert_array_equal(th[rest], tb[rest])


def test_push_and_latency():
    sim = N.NativeSim(get_robot("t2_measured"))
    sim.reset()
    sim.push(np.random.default_rng(0))
    assert abs(np.linalg.norm(sim.d.qvel[sim.root_v : sim.root_v + 2]) - 0.5) < 1e-9
    if ONNX.exists():
        sim.reset()
        meta = N.load_metadata(str(ONNX))
        pi = N.Policy(str(ONNX), meta, obs_latency=1)
        a0 = pi(sim, (0, 0, 0))
        sim.step(a0)
        sensors = N.observation(sim, a0)[:68].copy()  # read after the first step
        a1 = pi(sim, (0, 0, 0))
        sim.step(a1)
        pi(sim, (0, 0, 0))
        np.testing.assert_allclose(pi.hist[-1][:68], sensors, atol=1e-6)  # the sensors of one call earlier
        np.testing.assert_allclose(pi.hist[-1][68:], a1, atol=1e-6)  # with the current previous action


# ---- the step test
def test_steptest_grid():
    c = N.steptest_commands((-1.39, -1.14, -2.78), (1.90, 1.14, 2.78))
    assert c.shape == (672, 3) and np.linalg.norm(c, axis=1).min() > 1e-6
    for i in range(3):
        assert 0.0 in c[:, i] and len(np.unique(c[:, i])) == 15
    pure_vx = c[(c[:, 1] == 0) & (c[:, 2] == 0)]
    assert len(np.unique(pure_vx[:, 0])) == 14  # a pure vx line exists (the value nearest zero snapped to it)


def test_steptest_on_a_few_commands(tmp_path):
    if not ONNX.exists():
        pytest.skip("export missing")
    cmds = np.array([[0.5, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 1.0]])
    r = N.run_steptest(str(ONNX), trials=2, commands=cmds, out_dir=str(tmp_path), quiet=True)
    assert r["fall_frac"] == 0 and r["cmds_all_fell"] == 0 and len(r["per_command"]) == 3
    assert r["per_command"][0][3] < 0.3 and r["per_command"][2][3] < 0.15  # vx 0.5 and yaw 1.0 tracked
    assert r["by_type"]["forward"]["falls"] == 0 and r["by_type"]["yaw"]["median_rel_error"] < 0.15
    assert N.run_name("logs/t2al1_vel_s1/policy/ckpt_013000.onnx") == "t2al1_vel_s1" and N.run_name(str(ONNX)) == "t2al1_vel_s1_013000"
    assert (tmp_path / "steptest_t2_measured_t2al1_vel_s1_013000_nominal.json").exists()
    again = N.run_steptest(str(ONNX), trials=2, commands=cmds, out_dir=str(tmp_path), quiet=True)
    assert again["per_command"] == r["per_command"]  # the lag draws are seeded


# ---- the stand check
def test_hold_action():
    q0, s = np.array([0.1, -0.2]), np.array([0.25, 0.5])
    np.testing.assert_allclose(N.hold_action(np.array([0.2, -0.2]), q0, s), [0.4, 0.0])


def test_prepare_gains_are_boosters():
    assert len(N.PREPARE_KP) == len(N.PREPARE_KD) == 31
    assert N.PREPARE_KP[16:19] == [500.0, 500.0, 400.0] and N.PREPARE_KP[19:25] == [500.0, 350.0, 180.0, 500.0, 500.0, 500.0]
    assert N.PREPARE_KD[0] == 0.65 and N.PREPARE_KD[2] == 1.5 and N.PREPARE_KD[30] == 3.0
    vendor = Path.home() / "booster_deploy" / "booster_deploy" / "robots" / "t2.py"
    if vendor.exists():  # the vendor's file, read as text: its T2 prepare_state lists
        import ast

        src = vendor.read_text()
        block = src[src.index("prepare_state=PrepareStateCfg(") :]
        lists = [ast.literal_eval(block[block.index(k) + len(k) : block.index("]", block.index(k)) + 1]) for k in ("stiffness=", "damping=")]
        assert lists == [N.PREPARE_KP, N.PREPARE_KD]


def test_the_stop_cases_are_check_velocitys():
    import ast

    src = (REPO / "utils" / "check_velocity.py").read_text()
    a = src.index("REST = [")
    assert [tuple(c) for c in ast.literal_eval(src[a + 7 : src.index("]", a) + 1])] == list(N.STOP_CASES)
    assert N.STOP_CASES[0] == (0, 0, 0) and len(N.STOP_CASES) == 10


def test_stand_check_passes_from_both_starts(tmp_path):  # the hand-over with the hold action is the start the robot uses
    if not ONNX.exists():
        pytest.skip("export missing")
    for start, prev in (("reset", "zeros"), ("handover", "hold")):
        r = N.run_stand(str(ONNX), start=start, prev_action=prev, out_dir=str(tmp_path), quiet=True)
        assert r["passed"], (start, [(c["case"], c["fell"], c["still_stop"], c["reversals"]) for c in r["cases"]])
        assert len(r["cases"]) == 10 and all(not c["fell"] for c in r["cases"])
        assert all(9 < abs(t) < 15 for t in r["cases"][0]["ankle_torque"][:2])  # about 11.5 N m at each ankle pitch at rest
        assert len(r["frozen"]["tilt_deg"]) == 4 and r["frozen"]["tilt_deg"][0] < 3  # 0.2 s on a frozen setpoint: still upright
        assert (tmp_path / f"stand_t2_measured_t2al1_vel_s1_013000_nominal_{start}_{prev}.json").exists()
