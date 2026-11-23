"""deploy/booster_t2: the T2 velocity policy as a booster_deploy task. policy.py (numpy and onnxruntime only, what runs
on the robot's computer) is tested directly and against utils/play_native.py; the booster_deploy glue runs against
stand-in modules (tests/standins), whose names are compared with Booster's source as text. Vendor code is never
imported. CPU only: run with CUDA_VISIBLE_DEVICES= ."""

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from robot import t2_measured
from utils import play_native as N

REPO = Path(__file__).resolve().parents[1]
TASK = REPO / "deploy" / "booster_t2"
POLICY_PY = TASK / "policy.py"
ONNX = REPO / "deploy" / "t2al1_vel_s1_013000.onnx"  # the measured robot's policy, the file for the robot
SPEC = t2_measured.SPEC
KW = dict(joint_names=list(SPEC.joint_names), joint_stiffness=list(t2_measured.KP_SENT), joint_damping=list(t2_measured.KD_SENT),
          default_joint_pos=SPEC.default_joint_pos_list())  # fmt: skip
needs_onnx = pytest.mark.skipif(not ONNX.exists(), reason="export missing")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def P():
    return load(POLICY_PY, "bfm_t2_policy")


def rpy_of(R: np.ndarray):
    """The xyz Euler angles (roll, pitch, yaw) of a rotation matrix: what the robot's IMU reports as rpy."""
    return np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(R[2, 0]), np.arctan2(R[1, 0], R[0, 0])


# ---------------------------------------------------------------- policy.py
def test_only_numpy_and_onnxruntime_at_module_level():
    tree = ast.parse(POLICY_PY.read_text())
    mods = {n.names[0].name.split(".")[0] for n in tree.body if isinstance(n, ast.Import)} | {
        n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom)
    }
    assert mods <= {"numpy", "onnxruntime", "json", "platform", "sys", "time", "typing", "__future__", "collections"}


def test_quaternion_and_gravity_conventions(P):
    np.testing.assert_allclose(P.quat_from_euler_xyz(0, 0, 0), [1, 0, 0, 0], atol=1e-12)
    g = P.gravity_b(P.quat_from_euler_xyz(0.0, 0.2, 0.0))
    assert g[0] > 0.19 and abs(g[1]) < 1e-9 and g[2] < -0.97  # a forward pitch: gravity x > 0
    assert P.gravity_b(P.quat_from_euler_xyz(0.2, 0.0, 0.0))[1] < -0.19  # a roll, right side down
    rng = np.random.default_rng(0)
    for _ in range(20):  # against MuJoCo's own quaternion-to-matrix, on random attitudes
        r, p, y = rng.uniform(-1.2, 1.2, 3)
        quat, R = P.quat_from_euler_xyz(r, p, y), np.zeros(9)
        import mujoco

        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        np.testing.assert_allclose(rpy_of(R), (r, p, y), atol=1e-9)
        np.testing.assert_allclose(P.gravity_b(quat), -R[2], atol=1e-12)  # training's gravity term: -R[2, :]


@needs_onnx
def test_equal_actions_with_the_plain_mujoco_player(P):
    meta = N.load_metadata(str(ONNX))
    sim, ref = N.NativeSim(N.robot_of(meta)), N.Policy(str(ONNX), meta)
    pol = P.T2Policy(str(ONNX), prev_action="zeros", **KW)
    for i in range(25):
        a = ref(sim, (0.5, 0.0, 0.2))
        q, qd = sim.d.qpos[sim.qadr], sim.d.qvel[sim.dadr]
        quat = P.quat_from_euler_xyz(*rpy_of(sim.rotation()))  # as the robot gives it: rpy from the IMU
        s = pol.step(q, qd, sim.d.qvel[sim.root_v + 3 : sim.root_v + 6], quat, (0.5, 0.0, 0.2), now=0.02 * i)
        assert not s.stop and not s.skipped
        np.testing.assert_allclose(s.target, sim.default_q + sim.scale * a, atol=1e-5)
        sim.step(a)
    # The design asks for 1e-6; the float32 rpy round trip leaves a few 1e-7 in the observation, so this allows 1e-5.
    assert np.abs(a).max() > 0.5 and "onnxruntime" in pol.describe()


@needs_onnx
def test_hold_action_at_the_first_step(P):
    pol = P.T2Policy(str(ONNX), **KW)
    assert pol.hist is None
    q = pol.default_q + 0.05
    pol.step(q, np.zeros(31), np.zeros(3), np.array([1.0, 0, 0, 0]), (0, 0, 0), now=0.0)
    assert pol.hist.shape == (10, 99)
    np.testing.assert_allclose(pol.hist[-1][-31:], 0.05 / pol.action_scale, atol=1e-6)
    np.testing.assert_allclose(pol.hist[0], pol.hist[-1])  # the first observation repeated


@needs_onnx
def test_refusals_name_the_reason(P):  # the wrong gains must not reach the motors
    meta = N.load_metadata(str(ONNX))
    for change, word in ((dict(joint_names=KW["joint_names"][::-1]), "joint_names"),
                         (dict(default_joint_pos=[0.0] * 31), "default"),
                         (dict(joint_damping=meta["kd"]), "kd_sent"),  # training's joint-side kd
                         (dict(joint_stiffness=[0.8 * k for k in KW["joint_stiffness"]]), "kp_sent"),  # another gain table
                         (dict(policy_dt=0.01), "control_dt"),
                         (dict(prepare_joint_pos=[0.0] * 31), "prepare")):  # fmt: skip
        with pytest.raises(ValueError, match=word):
            P.T2Policy(str(ONNX), **{**KW, **change})
    assert set(meta) >= {"kp_sent", "kd_sent"} and meta["kd"] != meta["kd_sent"]  # the two kd tables do differ
    args = dict(joint_names=KW["joint_names"], joint_stiffness=KW["joint_stiffness"], joint_damping=KW["joint_damping"],
                default_joint_pos=KW["default_joint_pos"], policy_dt=0.02)  # fmt: skip
    P.check_metadata(meta, **args)
    for key, word in (("kp_sent", "kp_sent"), ("obs_terms", "obs_terms"), ("action_scale", "action_scale")):
        bad = {k: v for k, v in meta.items() if k != key}
        with pytest.raises(ValueError, match=word):  # e.g. an export from before the gains to send were recorded
            P.check_metadata(bad, **args)
    old = dict(meta, kd=[1.1 * x for x in meta["kd"]])
    with pytest.raises(ValueError, match="kd"):  # a file trained with other damping than the table implies
        P.check_metadata(old, **args)


def test_a_file_onnxruntime_cannot_load(P, tmp_path):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not a model")
    with pytest.raises(RuntimeError, match="onnxruntime"):
        P.T2Policy(str(bad), **KW)


@needs_onnx
def test_fall_stale_and_catch_up(P):  # a frozen sensor stops the task; a stalled loop does not rush the history
    pol = P.T2Policy(str(ONNX), **KW)
    q, z3, up = pol.default_q.copy(), np.zeros(3), np.array([1.0, 0, 0, 0])
    s = pol.step(q, np.zeros(31), z3, P.quat_from_euler_xyz(0, 1.2, 0), (0, 0, 0), now=0.0)
    assert s.stop and s.reason == "fall" and np.array_equal(s.target, q)
    pol.reset()
    a = pol.step(q, np.zeros(31), z3, up, (0, 0, 0), now=0.00)
    n_hist, n_prev = pol.hist.copy(), pol._prev.copy()
    b = pol.step(q + 1e-4, np.zeros(31), z3, up, (0, 0, 0), now=0.005)  # 5 ms later: a catch-up call
    assert b.skipped and np.array_equal(b.target, a.target) and pol.skipped == 1
    assert np.array_equal(pol.hist, n_hist) and np.array_equal(pol._prev, n_prev)  # nothing advanced
    c = pol.step(q, np.zeros(31), z3, up, (0, 0, 0), now=0.02)
    assert not c.skipped and not c.stop and not np.array_equal(pol.hist, n_hist)  # a real step advances the history
    pol.reset()
    out = [pol.step(q, np.zeros(31), z3, up, (0, 0, 0), now=0.02 * i) for i in range(3)]
    assert [o.stop for o in out] == [False, False, True] and out[2].reason == "stale"
    pol.reset()
    out = [pol.step(q + 1e-7 * i, np.zeros(31), np.array([1e-6 * i, 0, 0]), up, (0, 0, 0), now=0.02 * i) for i in range(8)]
    assert not any(o.stop for o in out)  # a quiet but live sensor is not a frozen one


@needs_onnx
def test_a_frozen_imu_with_live_joints_stops(P):  # the fall test reads the IMU: frozen upright, it would never fire
    pol = P.T2Policy(str(ONNX), **KW)
    q, up = pol.default_q.copy(), np.array([1.0, 0, 0, 0])
    out = [pol.step(q + 1e-4 * i, np.full(31, 1e-3 * i), np.zeros(3), up, (0, 0, 0), now=0.02 * i) for i in range(P.STALE_IMU_STEPS)]
    assert [o.stop for o in out] == [False] * (P.STALE_IMU_STEPS - 1) + [True] and out[-1].reason == "stale"
    assert P.STALE_IMU_STEPS == 5 and pol.longest_same == 1 and pol.longest_same_imu == 5


@needs_onnx
def test_a_bad_command_never_reaches_the_model(P):
    q, z, z3, up = np.array(DEFAULT_Q), np.zeros(31), np.zeros(3), np.array([1.0, 0, 0, 0])
    stand = P.T2Policy(str(ONNX), **KW).step(q, z, z3, up, (0, 0, 0), now=0.0)
    pol = P.T2Policy(str(ONNX), **KW)
    s = pol.step(q, z, z3, up, (np.nan, 0.0, np.inf), now=0.0)  # a broken remote reads as "stand still"
    assert not s.stop and np.all(np.isfinite(s.target)) and pol.bad_commands == 1
    np.testing.assert_allclose(s.target, stand.target)
    edge = P.T2Policy(str(ONNX), **KW).step(q, z, z3, up, tuple(pol.cmd_high * [1, 0, 0]), now=0.0)
    far = P.T2Policy(str(ONNX), **KW).step(q, z, z3, up, (9.0, 0, 0), now=0.0)  # outside the trained box: held at its edge
    np.testing.assert_allclose(far.target, edge.target)
    assert np.abs(edge.target - stand.target).max() > 1e-3


@needs_onnx
def test_a_model_output_that_is_not_a_number_stops(P):
    pol = P.T2Policy(str(ONNX), **KW)

    class Broken:
        def run(self, *a, **k):
            return (np.full((1, 31), np.nan, np.float32),)

    pol.session = Broken()
    q = np.array(DEFAULT_Q)
    s = pol.step(q, np.zeros(31), np.zeros(3), np.array([1.0, 0, 0, 0]), (0, 0, 0), now=0.0)
    assert s.stop and s.reason == "invalid" and np.array_equal(s.target, q)


def test_the_operators_roll_sign_matches_the_code(P):  # the dry run's notes and the README tell the operator what to expect
    assert P.gravity_b(P.quat_from_euler_xyz(0.2, 0.0, 0.0))[1] < 0  # roll +0.2: the right side goes down
    for text in ((TASK / "dry_run.py").read_text(), (REPO / "README.md").read_text()):
        assert "y goes negative when its right side goes down" in " ".join(text.split())


@needs_onnx
def test_an_invalid_state_stops(P):  # a NaN or an all-zero attitude must not reach the model as an observation
    pol = P.T2Policy(str(ONNX), **KW)
    q, z3, up = pol.default_q.copy(), np.zeros(3), np.array([1.0, 0, 0, 0])
    bad_q = q.copy()
    bad_q[3] = np.nan
    for args in ((bad_q, np.zeros(31), z3, up), (q, np.zeros(31), z3, np.zeros(4)), (q, np.zeros(31), np.array([0.0, np.inf, 0.0]), up)):
        pol.reset()
        s = pol.step(*args, (0, 0, 0), now=0.0)
        assert s.stop and s.reason == "invalid" and np.all(np.isfinite(s.target))


def test_deadzone(P):
    assert P.deadzone(0.04, 0.5) == 0.0 and P.deadzone(-0.04, 0.5) == 0.0 and P.deadzone(0.06, 0.5) == 0.06


# ---------------------------------------------------------------- the booster_deploy glue, against the stand-ins
import copy  # noqa: E402
import importlib  # noqa: E402
import itertools  # noqa: E402
import multiprocessing  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402

STANDINS = REPO / "tests" / "standins" / "booster_deploy"
VENDOR = Path.home() / "booster_deploy" / "booster_deploy"
OLD_T2_ONNX = REPO / "logs" / "t2_vel_stand_s3" / "policy" / "ckpt_013000.onnx"  # the earlier policy: other gains, no kp_sent
DEFAULT_Q = SPEC.default_joint_pos_list()
EXIT_LINE = 'exit_mode = "damping" if self._safety_abort else self.cfg.booster.exit_mode.strip().lower()'


@pytest.fixture
def load_task(tmp_path):
    """load_task(onnx=ONNX, exit_line=None): deploy/booster_t2 copied to <tmp>/root/tasks/bfm_t2 (with ``onnx``, unless
    None, as its models/t2al1_vel_s1_013000.onnx), imported with a copy of the stand-in booster_deploy package first on
    sys.path (``exit_line``, when given, replaces Booster's exit line in the copy's booster_robot_controller.py).
    Returns the task module and the stand-in registry module."""
    saved_path, saved_mods = list(sys.path), dict(sys.modules)

    def purge():
        for name in [m for m in sys.modules if m.split(".")[0] in ("booster_deploy", "tasks")]:
            del sys.modules[name]

    def loader(onnx=ONNX, exit_line=None, edits=()):
        purge()
        root = tmp_path / f"run{len(list(tmp_path.iterdir()))}"
        shutil.copytree(STANDINS, root / "vendor" / "booster_deploy")
        if exit_line is not None:
            edits = [("controllers/booster_robot_controller.py", EXIT_LINE, exit_line), *edits]
        for rel, old, new in edits:  # a booster_deploy that differs from the one the task was written against
            f = root / "vendor" / "booster_deploy" / rel
            assert f.read_text().count(old) >= 1, (rel, old)
            f.write_text(f.read_text().replace(old, new))
        shutil.copytree(TASK, root / "root" / "tasks" / "bfm_t2", ignore=shutil.ignore_patterns("models", "__pycache__"))
        (root / "root" / "tasks" / "__init__.py").write_text("")
        if onnx is not None:
            (root / "root" / "tasks" / "bfm_t2" / "models").mkdir()
            (root / "root" / "tasks" / "bfm_t2" / "models" / "t2al1_vel_s1_013000.onnx").symlink_to(onnx)
        sys.path[:0] = [str(root / "vendor"), str(root / "root")]
        return importlib.import_module("tasks.bfm_t2"), importlib.import_module("booster_deploy.utils.registry")

    yield loader
    purge()
    sys.path[:] = saved_path
    sys.modules.update({k: v for k, v in saved_mods.items() if k not in sys.modules})


def fake_controller(cfg):
    """The stand-in BaseController built from a ControllerCfg (robot, vel_command, policy), with set_state(q, qd, gyro,
    quat) writing robot.data's tensors, .stopped, and the policy's clock replaced by one advancing 20 ms per call."""
    import torch

    base = importlib.import_module("booster_deploy.controllers.base_controller")

    class Fake(base.BaseController):
        def set_state(self, q, qd, gyro, quat):
            d = self.robot.data
            for t, v, n in ((d.joint_pos, q, 31), (d.joint_vel, qd, 31), (d.root_ang_vel_b, gyro, 3), (d.root_quat_w, quat, 4)):
                t.copy_(torch.as_tensor(np.broadcast_to(np.asarray(v, dtype=np.float32), (n,)).copy()))

        @property
        def stopped(self):
            return not self.is_running

    ctl = Fake(cfg)
    ticks = itertools.count()
    ctl.policy.clock = lambda: 0.02 * next(ticks)
    ctl.start()
    return ctl


@needs_onnx
def test_the_task_registers_with_the_design_values(load_task):
    task, reg = load_task()
    cfg = reg.get_task("bfm_t2_vel")
    assert cfg.policy_dt == 0.02 and cfg.robot.prepare_mode == "standing" and isinstance(cfg.booster.exit_mode, task.ExitMode)
    assert (cfg.vel_command.vx_forward_max, cfg.vel_command.vx_backward_max, cfg.vel_command.vy_max, cfg.vel_command.vyaw_max) == (0.5, 0.3, 0.3, 1.0)
    assert cfg.mujoco.init_pos == [0.0, 0.0, 0.985]
    assert cfg.robot.joint_stiffness == list(t2_measured.KP_SENT) and cfg.robot.joint_damping == list(t2_measured.KD_SENT)
    assert cfg.robot.default_joint_pos == cfg.robot.prepare_state.joint_pos == DEFAULT_Q
    assert cfg.robot.joint_names == list(SPEC.joint_names)
    assert cfg.robot.prepare_state.stiffness[16:19] == [500.0, 500.0, 400.0]  # Booster's prepare gains, kept
    meta = N.load_metadata(str(ONNX))
    assert cfg.robot.joint_stiffness == meta["kp_sent"] and cfg.robot.joint_damping == meta["kd_sent"]
    lo, hi = meta["cmd_low"], meta["cmd_high"]  # the remote's caps stay inside the trained command box
    assert -0.3 >= lo[0] and 0.5 <= hi[0] and 0.3 <= hi[1] and 1.0 <= hi[2]


@needs_onnx
def test_inference_returns_targets_in_robot_order(load_task, P):
    task, reg = load_task()
    ctl = fake_controller(reg.get_task("bfm_t2_vel"))
    ctl.set_state(q=DEFAULT_Q, qd=0, gyro=0, quat=(1, 0, 0, 0))
    ctl.vel_command.lin_vel_x, ctl.vel_command.lin_vel_y = 0.02, -0.02  # inside the deadzone
    out = ctl.policy.inference()
    assert out.shape == (31,) and not ctl.stopped
    zero = P.T2Policy(str(ONNX), **KW).step(np.array(DEFAULT_Q), np.zeros(31), np.zeros(3), np.array([1.0, 0, 0, 0]), (0, 0, 0), now=0.0)
    np.testing.assert_allclose(out.numpy(), zero.target, atol=1e-6)  # the deadzone made the command zero
    ctl = fake_controller(reg.get_task("bfm_t2_vel"))
    ctl.set_state(q=DEFAULT_Q, qd=0, gyro=0, quat=(1, 0, 0, 0))
    ctl.vel_command.lin_vel_x, ctl.vel_command.ang_vel_yaw = -0.2, 0.5  # backward: its own cap (0.3), so outside the deadzone
    out = ctl.policy.inference()
    moving = P.T2Policy(str(ONNX), **KW).step(np.array(DEFAULT_Q), np.zeros(31), np.zeros(3), np.array([1.0, 0, 0, 0]), (-0.2, 0, 0.5), now=0.0)
    np.testing.assert_allclose(out.numpy(), moving.target, atol=1e-6)
    assert np.abs(moving.target - zero.target).max() > 1e-3


@needs_onnx
def test_a_fall_zeroes_the_stiffness_and_stops(load_task, P):
    task, reg = load_task()
    ctl = fake_controller(reg.get_task("bfm_t2_vel"))
    ctl.set_state(q=DEFAULT_Q, qd=0, gyro=0, quat=P.quat_from_euler_xyz(0, 1.2, 0))
    out = ctl.policy.inference()
    assert ctl.stopped and float(ctl.robot.joint_stiffness.abs().max()) == 0.0
    np.testing.assert_allclose(out.numpy(), DEFAULT_Q, atol=1e-6)  # the last command: the current pose, damping only


@needs_onnx
def test_the_exit_is_walking_unless_the_policy_flagged_a_fall(load_task, P):
    task, reg = load_task()
    cfg = reg.get_task("bfm_t2_vel")
    read = lambda c: c.booster.exit_mode.strip().lower()  # noqa: E731  as Booster reads it at the end of a run
    assert read(cfg) == "walking" and str(cfg.booster.exit_mode) == "damping"  # a normal stop, a cancelled start; the plain value
    ctl = fake_controller(cfg)
    ctl.set_state(q=DEFAULT_Q, qd=0, gyro=0, quat=(1, 0, 0, 0))
    ctl.policy.inference()
    assert read(cfg) == "walking" and read(copy.deepcopy(cfg)) == "walking"  # an ordinary step; a copied config
    ctl.set_state(q=DEFAULT_Q, qd=0, gyro=0, quat=P.quat_from_euler_xyz(0, 1.2, 0))
    ctl.policy.inference()
    assert ctl.stopped and read(cfg) == "damping" and read(copy.deepcopy(cfg)) == "damping"
    ctl.policy.reset()
    assert read(cfg) == "walking"  # a new run starts clean


@needs_onnx
def test_a_frozen_sensor_stop_also_ends_in_damping(load_task):
    task, reg = load_task()
    cfg = reg.get_task("bfm_t2_vel")
    ctl = fake_controller(cfg)
    ctl.set_state(q=DEFAULT_Q, qd=0, gyro=0, quat=(1, 0, 0, 0))
    stops = []
    for _ in range(3):  # three bit-identical states, 20 ms apart
        ctl.policy.inference()
        stops.append(ctl.stopped)
    assert stops == [False, False, True] and cfg.booster.exit_mode.strip() == "damping"
    assert float(ctl.robot.joint_stiffness.abs().max()) == 0.0


@needs_onnx
@pytest.mark.filterwarnings("ignore:.*multi-threaded.*:DeprecationWarning")  # pytest has threads; the child only sets the flag
def test_the_flag_crosses_a_fork(load_task):
    task, reg = load_task()
    cfg = reg.get_task("bfm_t2_vel")
    p = multiprocessing.get_context("fork").Process(target=task._FELL.set)  # Booster's policy process is a fork of the reader
    p.start()
    p.join(10)
    assert p.exitcode == 0 and cfg.booster.exit_mode.strip() == "damping"


@needs_onnx
def test_import_refuses_a_booster_that_reads_the_exit_another_way(load_task):
    with pytest.raises(ValueError, match="exit_mode"):
        load_task(exit_line="exit_mode = self.cfg.booster.exit_mode.lower()")


@needs_onnx
def test_import_refuses_a_booster_whose_controller_lost_a_member(load_task):  # on the robot there is no test suite
    for rel, old, new, word in (("controllers/base_controller.py", "    root_quat_w: torch.Tensor\n", "    root_quat: torch.Tensor\n", "root_quat_w"),
                                ("controllers/base_controller.py", "self.vx_forward_max =", "self.vx_fwd_max =", "vx_forward_max"),
                                ("controllers/base_controller.py", "    joint_stiffness: torch.Tensor\n", "    stiffness: torch.Tensor\n", "joint_stiffness")):  # fmt: skip
        with pytest.raises(ValueError, match=word):
            load_task(edits=[(rel, old, new)])


def test_a_missing_file_is_loud_and_stops_the_tasks_own_start(load_task, capsys, monkeypatch):
    load_task(onnx=None)  # another task is being started (or the list is asked for): load, but say so
    out = capsys.readouterr().out
    assert "WARNING" in out and "t2al1_vel_s1_013000.onnx" in out and "nothing" in out
    monkeypatch.setattr(sys, "argv", ["scripts/deploy.py", "--task", "bfm_t2_vel"])
    with pytest.raises(FileNotFoundError, match="t2al1_vel_s1_013000.onnx"):  # this task: stop before Custom mode
        load_task(onnx=None)
    if ONNX.exists():
        load_task()
        assert "checked" in capsys.readouterr().out


def test_import_refuses_a_file_for_other_gains(load_task):  # the wrong gains must not reach the motors
    if not OLD_T2_ONNX.exists():
        pytest.skip("the earlier export is missing")
    with pytest.raises(ValueError, match="kp_sent"):
        load_task(onnx=OLD_T2_ONNX)


def test_the_task_loads_without_ros_and_without_the_file(load_task):
    task, reg = load_task(onnx=None)  # e.g. Booster's MuJoCo mode before the file is copied
    assert "bfm_t2_vel" in reg.list_tasks() and "rclpy" not in sys.modules
    for name in ("dry_run", "policy"):  # Booster's deploy.py imports every module under tasks/
        importlib.import_module(f"tasks.bfm_t2.{name}")
    assert "rclpy" not in sys.modules and "mujoco" not in [m for m in sys.modules if m not in SEEN_MODULES]


SEEN_MODULES = set(sys.modules)  # what this test file's own imports loaded (mujoco among them)

VENDOR_NAMES = {
    "controllers/controller_cfg.py": {
        "RobotCfg": ["name", "prepare_mode", "joint_names", "joint_stiffness", "joint_damping", "default_joint_pos", "effort_limit",
                     "mjcf_path", "prepare_state"],
        "PrepareStateCfg": ["stiffness", "damping", "joint_pos"],
        "ControllerCfg": ["policy_dt", "robot", "vel_command", "policy", "mujoco", "booster"],
        "PolicyCfg": ["constructor", "checkpoint_path"],
        "VelocityCommandCfg": ["vx_max", "vx_forward_max", "vx_backward_max", "vy_max", "vyaw_max"],
        "MujocoControllerCfg": ["init_pos"],
        "BoosterRobotControllerCfg": ["exit_mode"],
    },
    "controllers/base_controller.py": {
        "Policy": ["reset", "inference", "task_path", "cfg", "controller"],
        "RobotData": ["joint_pos", "joint_vel", "root_ang_vel_b", "root_quat_w"],
        "BoosterRobot": ["cfg", "data", "joint_stiffness", "joint_damping"],
        "VelocityCommand": ["lin_vel_x", "lin_vel_y", "ang_vel_yaw", "vx_forward_max", "vx_backward_max", "vy_max", "vyaw_max"],
        "BaseController": ["cfg", "robot", "vel_command", "policy", "start", "stop"],
    },
    "utils/remote_control_service.py": {"RemoteControlService": ["get_vx_cmd", "get_vy_cmd", "get_vyaw_cmd"]},  # the dry run, on the robot
}  # fmt: skip


def class_members(tree: ast.Module, cls: str) -> set:
    """Names a class defines: annotated fields, methods, and attributes its methods assign on self."""
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    names = {f.name for f in node.body if isinstance(f, ast.FunctionDef)}
    for s in ast.walk(node):
        targets = [s.target] if isinstance(s, ast.AnnAssign) else s.targets if isinstance(s, ast.Assign) else []
        for t in targets:
            for e in t.elts if isinstance(t, ast.Tuple) else [t]:
                if isinstance(e, ast.Name):
                    names.add(e.id)
                elif isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id == "self":
                    names.add(e.attr)
    return names


def robot_cfg_literals(path: Path) -> dict:
    call = next(n.value for n in ast.parse(path.read_text()).body if isinstance(n, ast.Assign) and n.targets[0].id == "T2_31DOF_CFG")
    kw = {k.arg: k.value for k in call.keywords}
    out = {k: ast.literal_eval(kw[k]) for k in ("joint_names", "effort_limit")}
    out.update({f"prepare_{k.arg}": ast.literal_eval(k.value) for k in kw["prepare_state"].keywords})
    return out


def test_standins_match_the_vendor_source():  # a renamed field in Booster's code would pass here and fail on the robot
    """Every class and member the task uses exists under the same name in ~/booster_deploy, read as text, and in the
    stand-ins; the stand-in T2_31DOF_CFG's joint names, effort limits and prepare state equal the vendor's; and the
    vendor reads the exit mode and starts its policy process the way the split exit relies on."""
    for rel, classes in VENDOR_NAMES.items():
        for root in (STANDINS, VENDOR):
            if not (root / rel).exists():
                assert root is STANDINS and rel.endswith("remote_control_service.py") or root is VENDOR, (root, rel)
                continue  # no clone here (or: the stand-ins have no remote)
            tree = ast.parse((root / rel).read_text())
            for cls, members in classes.items():
                assert set(members) <= class_members(tree, cls), (str(root), rel, cls, set(members) - class_members(tree, cls))
    if not VENDOR.exists():
        pytest.skip("no booster_deploy clone to compare with")
    assert robot_cfg_literals(STANDINS / "robots" / "t2.py") == robot_cfg_literals(VENDOR / "robots" / "t2.py")
    for fn in ("register_task", "get_task", "list_tasks"):
        assert f"def {fn}(" in (VENDOR / "utils" / "registry.py").read_text()
    portal = (VENDOR / "controllers" / "booster_robot_controller.py").read_text()
    assert EXIT_LINE in portal and EXIT_LINE in (STANDINS / "controllers" / "booster_robot_controller.py").read_text()
    assert "mp.Process(" in portal and "get_context" not in portal and "set_start_method" not in portal  # the default: fork
    quat = (VENDOR / "utils" / "isaaclab" / "math.py").read_text()
    for line in ("qw = cy * cr * cp + sy * sr * sp", "qx = cy * sr * cp - sy * cr * sp", "qy = cy * cr * sp + sy * sr * cp",
                 "qz = sy * cr * cp - cy * sr * sp"):  # fmt: skip
        assert line in quat  # the formula policy.py copies


# ---------------------------------------------------------------- the passive dry run
@needs_onnx
def test_dry_run_summary(load_task):
    load_task()
    dry_run = importlib.import_module("tasks.bfm_t2.dry_run")
    d = dry_run.DryRun(str(ONNX))
    rng = np.random.default_rng(0)
    for i in range(dry_run.MIN_STEPS):
        for k in range(10):
            d.on_message(0.02 * i + 0.002 * k)  # /low_state at 500 Hz
        r = d.on_state(np.array(DEFAULT_Q) + 1e-4 * rng.standard_normal(31), np.zeros(31), (0.0, 0.05, 0.0), 1e-3 * rng.standard_normal(3),
                       (0, 0, 0), now=0.02 * i)  # fmt: skip
    assert r["gravity_b"][0] > 0.04 and len(r["action"]) == 31 and r["stop"] is None
    s = d.summary()
    assert s.splitlines()[-1].startswith("PASS") and platform.machine() in s and "onnxruntime" in s and "500 Hz" in s
    assert "longest run of identical readings: whole state 1" in s and "IMU alone 1" in s
    assert "rclpy" not in sys.modules


@needs_onnx
def test_dry_run_verdict_fails_on_a_would_be_stop_or_a_short_run(load_task):
    load_task()
    dry_run = importlib.import_module("tasks.bfm_t2.dry_run")
    assert dry_run.MIN_STEPS == 3000  # a minute
    rng = np.random.default_rng(0)
    live = lambda d, i: d.on_state(np.array(DEFAULT_Q) + 1e-4 * rng.standard_normal(31), np.zeros(31), (0, 0, 0),  # noqa: E731
                                   1e-3 * rng.standard_normal(3), (0, 0, 0), now=0.02 * i)  # fmt: skip
    short = dry_run.DryRun(str(ONNX))
    for i in range(60):
        live(short, i)
    assert short.summary().splitlines()[-1].startswith("FAIL") and "too short" in short.summary()
    frozen = dry_run.DryRun(str(ONNX))
    for i in range(dry_run.MIN_STEPS):
        live(frozen, i)
    hits = [frozen.on_state(np.array(DEFAULT_Q), np.zeros(31), (0, 0, 0), np.zeros(3), (0, 0, 0), now=0.02 * (dry_run.MIN_STEPS + i))["stop"]
            for i in range(3)]  # fmt: skip
    assert hits == [None, None, "stale"] and frozen.stops == {"stale": 1}
    s = frozen.summary()
    assert s.splitlines()[-1].startswith("FAIL") and "stale 1" in s and "whole state 3" in s  # compute time alone would say PASS


@needs_onnx
def test_dry_run_reports_a_stop_and_goes_on(load_task):
    load_task()
    dry_run = importlib.import_module("tasks.bfm_t2.dry_run")
    d = dry_run.DryRun(str(ONNX))
    r = d.on_state(np.array(DEFAULT_Q), np.zeros(31), (0.0, 1.2, 0.0), np.zeros(3), (0, 0, 0), now=0.0)
    assert r["stop"] == "fall"
    r = d.on_state(np.array(DEFAULT_Q) + 0.001, np.zeros(31), (0.0, 0.0, 0.0), np.zeros(3), (0, 0, 0), now=0.02)
    assert r["stop"] is None and abs(r["qd_fd"][0] - 0.05) < 1e-6  # 0.001 rad in 20 ms


@needs_onnx
def test_leg_gravity_follows_the_ankles(load_task):
    load_task()
    dry_run = importlib.import_module("tasks.bfm_t2.dry_run")
    m, names = N.build_model(SPEC), list(SPEC.joint_names)
    q = np.array(DEFAULT_Q)
    np.testing.assert_allclose(dry_run.leg_gravity(m, q, names, prefix="robot/"), [0, 0, -1], atol=0.02)  # the default pose: soles level
    for n in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
        q[names.index(n)] += 0.1
    g = dry_run.leg_gravity(m, q, names, prefix="robot/")
    assert abs(abs(g[0]) - np.sin(0.1)) < 0.01 and abs(g[1]) < 0.01  # both ankles pitched 0.1 rad: a 0.1 rad lean


def test_dry_run_module_level_imports():
    tree = ast.parse((TASK / "dry_run.py").read_text())
    mods = {n.names[0].name.split(".")[0] for n in tree.body if isinstance(n, ast.Import)} | {
        (n.module or ".").split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.level == 0
    }
    assert mods <= {"numpy", "platform", "sys", "time", "__future__", "os"}  # ROS 2, Booster's remote and mujoco only inside functions


# ---------------------------------------------------------------- deploy/ holds the robot's policy
def test_deploy_holds_the_t2_policy():
    files = sorted(p.name for p in (REPO / "deploy").glob("*.onnx"))
    assert files == ["t2al1_vel_s1_013000.onnx"]  # exactly one ONNX: the robot's
    meta = N.load_metadata(str(REPO / "deploy" / files[0]))
    assert N.robot_of(meta).name == "t2_measured" and meta["kp_sent"] == list(t2_measured.KP_SENT)
    assert (REPO / "deploy" / "booster_t2" / "__init__.py").exists()
    logged = REPO / "logs" / "t2al1_vel_s1" / "policy" / "ckpt_013000.onnx"  # git-ignored
    if logged.exists():
        assert (REPO / "deploy" / files[0]).read_bytes() == logged.read_bytes()  # the export of logs/t2al1_vel_s1


# ---------------------------------------------------------------- the robot's Python may be 3.10
def old_python_problems(path: Path) -> list:
    """Lines whose f-strings need Python 3.12 (PEP 701): a string inside a replacement field that reuses the quote of
    an enclosing f-string, or a backslash or line break inside a replacement field. booster_deploy asks for 3.10+ and
    imports every file of the task folder at start-up, so one such line stops everything on the robot."""
    import io
    import tokenize

    bad, stack = [], []  # stack: (quote character, triple-quoted, depth of open braces) per enclosing f-string
    for tok in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
        if tok.type == tokenize.FSTRING_START:
            quote = tok.string.lstrip("rRfFbBuU")
            if any(q == quote[0] for q, _, _ in stack):
                bad.append((tok.start[0], "nested f-string with the same quote"))
            stack.append([quote[0], len(quote) == 3, 0])
        elif tok.type == tokenize.FSTRING_END:
            stack.pop()
        elif stack:
            top = stack[-1]
            if tok.type == tokenize.OP and tok.string in "{}":
                top[2] += 1 if tok.string == "{" else -1
            elif top[2] > 0:  # inside a replacement field
                if tok.type == tokenize.STRING and any(q == tok.string.lstrip("rRbBuU")[0] for q, _, _ in stack):
                    bad.append((tok.start[0], "a string reusing the f-string's quote"))
                if "\\" in tok.string or (tok.type in (tokenize.NL, tokenize.NEWLINE) and not top[1]):
                    bad.append((tok.start[0], "a backslash or line break in a replacement field"))
    return bad


def test_the_robot_side_files_parse_on_python_3_10(tmp_path):
    for name in ("__init__.py", "policy.py", "dry_run.py"):
        assert old_python_problems(TASK / name) == [], name
        ast.parse((TASK / name).read_text(), feature_version=(3, 10))
    probe = tmp_path / "probe.py"  # the check itself: these two lines are 3.12-only
    probe.write_text("x = {'a': 1}\nprint(f'{x['a']}')\nprint(f\"{f\"{x}\"}\")\n")
    assert [line for line, _ in old_python_problems(probe)] == [2, 3]
