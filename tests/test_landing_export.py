"""CPU checks of the landing command's ONNX export (utils/export_landing.py) and of the native-MuJoCo landing test's
scoring side (utils/landing_native.py).    pytest tests/test_landing_export.py"""

import json
import math
import os
import shutil
import subprocess

import numpy as np
import pytest
import torch
from test_landing import FEET, KIN, _mdp  # the trainer's kinematics and its MDP on the CPU

from robot.t1 import SPEC
from src import landing
from src.landing import LandingCfg, relative
from src.mdp import upright_rel
from utils.rotations import so3_exp

J = SPEC.num_joints
Q0 = torch.tensor(SPEC.default_joint_pos_list())
REC = "logs/land_r4_d10_al2_s3/landing_test_final.npz"  # the trainer's recording of its own landing test
WALK = "logs/land_r4_d10_al2_s3/ckpt_004000_both.onnx"  # the T1's walking policy (deploy/ holds the T2's)


def _net(pi=None, cfg=None):
    from utils.export_landing import LandNet

    return LandNet(pi, KIN, FEET, Q0, cfg or LandingCfg(), 20.0)


def _moving_states(T, B, seed=0):
    """T states of B robots whose root wanders, turns and tilts while every joint moves: the soles go anywhere."""
    g = torch.Generator().manual_seed(seed)
    ramp = torch.linspace(0, 1, T)[:, None, None]
    a, b = (torch.randn(1, B, n, generator=g) for n in (3, 3))
    p = torch.tensor([0.0, 0.0, 0.68]) + ramp * a + 0.3 * torch.sin(5 * ramp + b)
    yaw = torch.zeros(T, B, 3)
    yaw[..., 2] = 2.5 * ramp[..., 0] * torch.randn(1, B, generator=g) + torch.randn(1, B, generator=g)
    tilt = 0.25 * torch.sin(4 * ramp + torch.randn(1, B, 3, generator=g)) * torch.tensor([1.0, 1.0, 0.0])
    R = so3_exp(yaw) @ so3_exp(tilt)
    q = Q0 + 0.25 * torch.sin(6 * ramp + 6 * torch.rand(1, B, J, generator=g)) * torch.rand(1, B, J, generator=g)
    q = torch.clamp(q, KIN.q_lo, KIN.q_hi)
    z = torch.zeros(T, B, 3)
    return [(p[t], R[t], z[t], z[t], q[t], torch.zeros(B, J)) for t in range(T)]


def test_command_state_follows_the_trainer():
    """The exported command state sees only joint angles and the gravity direction. Its 12 encoder inputs and its
    clocks must be the trainer's, whose command state holds world poses (landing.land_reset / land_issue /
    MDP.land_step / MDP.land_features): through SETTLE, a command refused in SETTLE, a step of each foot (the second
    sent on the state the first one's deadline ends in), a command refused during a step, and the hold."""
    mdp, _ = _mdp()
    cfg, net = mdp.land, _net()
    B, T = 2, 150
    S = _moving_states(T + 1, B)
    xi, a0, ap = torch.zeros(B, mdp.xi_dim), torch.zeros(B, J), torch.zeros(B, 2 * J)
    f, up = torch.zeros(B, dtype=torch.long), torch.ones(B, dtype=torch.bool)
    first = torch.tensor([0.0, 1.0])  # the commanded sole shown before the first command (training drew it at random)
    pose = mdp.sole_kin(S[0])[0]["pose"]
    C = landing.land_reset(cfg, torch.zeros(B, landing.STATE_DIM), pose, up, first)
    state = torch.zeros(B, 8)
    state[:, 1] = first
    # (tick, commanded sole of each robot, landing pose in the other sole's frame); a tick of None: the state after
    # the previous command's deadline, per robot
    sends = [(10, 1 - first, torch.tensor([[0.10, 0.25, 0.0], [0.10, -0.25, 0.0]])),  # in SETTLE: refused
             (30, 1 - first, torch.tensor([[0.18, -0.24, -0.2], [-0.12, 0.30, 0.3]])),
             (40, first, torch.tensor([[0.0, 0.30, 0.0], [0.0, -0.30, 0.0]])),  # during the step: refused
             (None, first, torch.tensor([[-0.05, 0.22, 0.35], [0.25, -0.18, 0.1]]))]  # fmt: skip
    deadline = torch.full((B,), -1)
    seen = {"settle": 0, "step": 0, "hold": 0, "taken": 0}
    for t in range(T):
        pose = mdp.sole_kin(S[t])[0]["pose"]
        o = mdp.obs(S[t], ap, xi, f, up)
        cmd = torch.zeros(B, 5)
        for tick, swing, c in sends:
            arrive = deadline == t if tick is None else torch.full((B,), tick == t)
            if not arrive.any():
                continue
            cur = relative(landing._pick(pose, swing), landing._pick(pose, 1 - swing)).double()
            travel = (c[:, :2] - cur[:, :2]).norm(dim=-1) + landing.YAW_M * landing.wrap(c[:, 2] - cur[:, 2]).abs()
            n0 = torch.tensor([float(min(max(round(cfg.clock_base + cfg.clock_per_m * x), cfg.clock_range[0]),
                                         cfg.clock_range[1])) for x in travel.tolist()])  # fmt: skip
            hold = landing.phase(C)[2]
            C = landing.land_issue(C, pose, arrive, swing, c, n0)
            took = arrive & hold
            seen["taken"] += int(took.sum())
            if tick == 30:
                deadline = torch.where(took, t + n0.long(), deadline)
            cmd = torch.where(arrive[:, None], torch.cat((torch.ones(B, 1), swing[:, None], c), -1), cmd)
        want = mdp.land_features(C, o)
        got, state = net.command(o, cmd, state)
        assert torch.allclose(got, want, atol=2e-5), (t, (got - want).abs().max())
        for name, mask in zip(("settle", "step", "hold"), landing.phase(C)):
            seen[name] += int(mask.sum())
        C = mdp.land_step(S[t + 1], C, pose, a0, ap, xi, torch.zeros(B, 2, 3), torch.zeros(B, 2, 3), up)[1]
        assert torch.equal(state[:, 1:4], C[:, :3]), (t, state[:, 1:4], C[:, :3])  # commanded sole, clock, settle clock
    assert seen["taken"] == 2 * B and seen["settle"] == 25 * B and seen["step"] > 80 * B and seen["hold"] > 10 * B
    # the hold after the second step: the state reads the stance the robot landed in
    last = t - 5
    pose = mdp.sole_kin(S[int(deadline.max()) + 60])[0]["pose"]  # any state: only the latched one may be read
    A = C[:, 6:].view(B, 2, 3)
    landed = relative(landing._pick(A, C[:, 0]), landing._pick(A, 1 - C[:, 0]))
    assert last > 0 and torch.equal(state[:, 2], torch.zeros(B)) and torch.allclose(state[:, 5:8], landed, atol=2e-5)


def test_clock_is_the_test_grids():
    """A command's clock is looked up inside the model from the step's size: for every command of the landing test's
    grid, sent from the default stance, it is the grid's own (the sampler's rule without jitter)."""
    from utils.landing_score import command_grid

    cfg, net = LandingCfg(), _net()
    S = (torch.tensor([[0.0, 0.0, 0.688]]), torch.eye(3)[None], torch.zeros(1, 3), torch.zeros(1, 3), Q0[None],
         torch.zeros(1, J))  # fmt: skip
    pose = landing.soles(*(x[:, FEET] for x in KIN.forward(*S)))["pose"][0]
    grid = command_grid(cfg, pose)
    K = len(grid["swing"])
    assert K == 226
    o = torch.zeros(K, 3 * J + 12)
    o[:, 2 * J + 5] = -1.0  # upright: the gravity direction (0, 0, -1)
    o[:, 3 * J + 6], o[:, 3 * J + 9] = 1.0, 1.0  # and the six reference-orientation entries of an upright robot
    hold = torch.zeros(K, 8)
    hold[:, 0] = 1.0  # started, no clock running
    cmd = torch.cat((torch.ones(K, 1), grid["swing"][:, None], grid["c"]), -1)
    _, state = net.command(o, cmd, hold)
    assert torch.equal(state[:, 2] + 1, grid["n0"])  # the steps left after the command's own tick
    assert torch.equal(state[:, 1], grid["swing"]) and torch.allclose(state[:, 5:8], grid["c"])


def test_onnx_reproduces_the_module(tmp_path):
    """The ONNX file under onnxruntime against the module it was exported from, fed the same inputs with each one's
    own state fed back: SETTLE, a command, a refused one, the step, the deadline and the hold (a branch frozen when
    the graph was traced would show in one of them)."""
    import onnxruntime as ort

    from src.networks import Policy
    from utils.export_landing import STATE_TERMS, export

    torch.manual_seed(0)
    hist = 10
    pi = Policy(3 * J + 12, 16, J, hist=hist, enc_hidden=(32,), dec_hidden=(64,), vel=True, land=True)
    with torch.no_grad():  # a decoder that answers, and normalizers that are not the identity
        pi.dec[-1].weight.normal_(0, 0.5)
        pi.obs_norm.mean.normal_(0, 0.1)
        pi.obs_norm.var.uniform_(0.5, 2.0)
    net = _net(pi).eval()
    path = str(tmp_path / "land.onnx")
    export(net, hist, path, {"iteration": 7})
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    assert json.loads(meta["iteration"]) == 7
    assert {x.name: x.shape for x in session.get_inputs()} == {
        "obs_hist": [1, hist, 3 * J + 6], "cmd": [1, 5], "state": [1, len(STATE_TERMS)]}  # fmt: skip
    assert [x.name for x in session.get_outputs()] == ["action", "next_state"]
    n = STATE_TERMS.index("steps_left")
    s_t = torch.zeros(1, len(STATE_TERMS))
    s_o = s_t.numpy()
    clock, phases = [], set()
    for t in range(95):
        o = 0.3 * torch.randn(1, hist, 3 * J + 6)
        g = torch.tensor([0.0, 0.0, -1.0]) + 0.2 * torch.randn(3)
        o[..., 2 * J + 3 : 2 * J + 6] = g / g.norm()
        cmd = torch.zeros(1, 5)
        if t in (5, 30, 40):  # in SETTLE (refused), in the hold (taken), during the step (refused)
            cmd = torch.tensor([[1.0, 1.0, 0.15, -0.25, 0.1]])
        with torch.no_grad():
            a_t, s_t = net(o, cmd, s_t)
        a_o, s_o = session.run(None, {"obs_hist": o.numpy(), "cmd": cmd.numpy(), "state": s_o})
        assert np.abs(a_o - a_t.numpy()).max() < 1e-4 and np.abs(s_o - s_t.numpy()).max() < 1e-5, t
        clock.append(float(s_o[0, n]))
        phases.add("settle" if t < 25 else "step" if clock[-1] > 0 else "hold")
    assert float(a_t.abs().max()) > 0.1  # the comparison is not of zeros
    assert clock[29] == 0 and clock[30] > 38 and clock[41] == clock[30] - 11 and clock[-1] == 0
    assert phases == {"settle", "step", "hold"}


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv: utils/play_mujoco.py runs in its own environment")
def test_native_rollouts_carry_the_commands(tmp_path):
    """utils/play_mujoco.py's landtest on a small random export, two rollouts of two commands each: a rollout is
    recorded from the reset, the first command goes out one second in and the second one second after the first
    one's deadline, and the clock reported for each is the one the model gives that command on the state recorded at
    its tick (so the commands, their feet and poses, the state fed back and the recording's time index all arrive as
    sent)."""
    from src.networks import Policy
    from utils.export_landing import STATE_TERMS, export
    from utils.landing_score import REPO
    from utils.rotations import quat_to_mat

    torch.manual_seed(0)
    pi = Policy(3 * J + 12, 16, J, hist=10, enc_hidden=(32,), dec_hidden=(64,), vel=True, land=True)
    net = _net(pi).eval()
    onnx_path, cmds, rec = (str(tmp_path / n) for n in ("land.onnx", "commands.npz", "rollouts.npz"))
    export(net, 10, onnx_path, {"iteration": 7})
    swing = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # (rollouts, commands)
    c = torch.tensor([[[0.20, 0.2125, 0.0], [0.0, -0.2125, 0.0]], [[0.05, -0.30, -0.4], [0.1, 0.25, 0.2]]])
    np.savez(cmds, swing=swing.numpy(), c=c.numpy())
    subprocess.run(["uv", "run", "utils/play_mujoco.py", "landtest", "--onnx", onnx_path, "--commands", cmds,
                    "--out", rec, "--jobs", "2", "--hold_s", "0.2", "--lag", "5"], check=True, cwd=REPO)  # fmt: skip
    z = np.load(rec)
    assert z["lags"].shape == (2, 5) and (z["lags"] == 5).all()  # --lag: every actuator group's lag, not a draw
    T = 2 * (50 + 62) + 10 + 1  # per command a second and the longest clock, then the hold; and the reset state
    assert z["root_p"].shape == (T, 2, 3) and z["root_quat"].shape == (T, 2, 4)
    assert z["q"].shape == (T, 2, J) and all(z[k].shape == (T, 2) for k in ("fell", "qd_rms", "root_speed"))
    assert np.allclose(z["q"][0], Q0.numpy(), atol=1e-6) and np.allclose(z["root_quat"][0], [1, 0, 0, 0])
    assert np.allclose(z["root_p"][0], [0, 0, 0.688]) and not z["fell"][0].any() and z["qd_rms"][0].max() == 0
    t_cmd, n0 = z["t_cmd"], z["n0"]  # (rollouts, commands): the tick a command was sent on, and its clock
    assert (t_cmd[:, 0] == 50).all() and (t_cmd[:, 1] == 50 + n0[:, 0] + 50).all()
    assert np.abs(z["q"][50] - Q0.numpy()).max() > 1e-3  # the robot has moved by the first command
    hold = torch.zeros(2, len(STATE_TERMS))
    hold[:, 0] = 1.0
    for i in range(2):  # the model's clock on the recorded state each command was sent on
        q = torch.tensor(z["q"][t_cmd[:, i], [0, 1]])
        R = quat_to_mat(torch.tensor(z["root_quat"][t_cmd[:, i], [0, 1]]))
        o = torch.zeros(2, 3 * J + 12)
        o[:, :J], o[:, 2 * J + 3 : 2 * J + 6] = q - Q0, -R[:, 2]
        o[:, 3 * J + 6 :] = upright_rel(-R[:, 2])
        _, state = net.command(o, torch.cat((torch.ones(2, 1), swing[:, i, None], c[:, i]), -1), hold)
        assert (n0[:, i] == state[:, STATE_TERMS.index("steps_left")].numpy() + 1).all() and n0.min() >= 40
    assert len(set(n0.ravel().tolist())) > 1  # the commands are not all the same size


def test_second_commands_and_their_view():
    """The two-step test's second command brings the other sole back beside the one that moved (the default stance,
    seen from the moved sole), and is kept only where its travel is one the sampler sends; its rollouts are read with
    every second command on the same state."""
    from utils.landing_native import at_command, follow_commands

    q = math.pi / 4
    pose = torch.tensor([[0.05, 0.10625, 0.0], [0.05, -0.10625, 0.0]])  # the reset stance: left, right sole
    # the left sole 20 cm forward; the right sole turned 45 degrees toes out on the spot; the left sole 2 cm forward
    grid = {"swing": torch.tensor([0.0, 1.0, 0.0]),
            "c": torch.tensor([[0.20, 0.2125, 0.0], [0.0, -0.2125, -q], [0.02, 0.2125, 0.0]])}  # fmt: skip
    f = follow_commands(LandingCfg(), pose, grid)
    assert f["swing"].tolist() == [1.0, 0.0, 1.0]
    home = torch.tensor([[0.0, -0.2125, 0.0], [0.0, 0.2125, 0.0], [0.0, -0.2125, 0.0]])
    assert torch.allclose(f["c"], home, atol=1e-6)
    turned = math.hypot(0.2125 * math.sin(q), 0.2125 * (1 - math.cos(q))) + landing.YAW_M * q
    assert f["travel"].tolist() == pytest.approx([0.20, turned, 0.02], abs=1e-5)
    assert f["ok"].tolist() == [True, True, False]  # 2 cm is under the smallest travel the sampler sends

    T = 300
    z = {"root_p": torch.arange(T).float()[:, None, None].repeat(1, 2, 3).numpy(),  # every entry its state's index
         "t_cmd": np.array([[50, 148], [50, 151]]), "fell": np.zeros((T, 2), bool)}  # fmt: skip
    z["fell"][60, 1] = True  # one state of the second rollout, during its first step
    for k in ("root_quat", "q", "qd_rms", "root_speed"):
        z[k] = z["root_p"]
    first, second = at_command(z, 0), at_command(z, 1)
    assert first["root_p"].shape == (T, 2, 3) and torch.equal(first["root_p"], torch.as_tensor(z["root_p"]))
    assert second["root_p"].shape == (T - 101, 2, 3)  # cut to the rollout whose second command came last
    assert second["root_p"][50, :, 0].tolist() == [148, 151] and second["root_p"][0, :, 0].tolist() == [98, 101]
    assert second["fell"][0].tolist() == [False, True] and first["fell"][59:61, 1].tolist() == [False, True]


def test_stance_kept_after_the_deadline():
    """After a landing the robot is to stand in the stance it landed in: the stance is kept when neither sole leaves
    its pose at the deadline by more than 2 cm or 0.10 rad during the 3 s after it."""
    from utils.landing_native import kept_after

    T, dl = 260, torch.tensor([100, 100, 100, 100, 95])
    pose = torch.zeros(T, 5, 2, 3)
    pose[:, :, 0, 1], pose[:, :, 1, 1] = 0.10625, -0.10625
    pose[:100, 0, 1, 0] = -0.30  # the right sole lands at the deadline and stays: kept
    pose[130:, 1, 0, 0] = 0.30  # the left (planted) sole steps 30 cm away half a second after the deadline
    pose[130:140, 2, 1, 2] = 0.15  # the right sole turns 0.15 rad for a moment and comes back
    pose[120:, 3, 0, 0] = 0.015  # a 1.5 cm shift: inside the tolerance
    pose[246:, 4, 0, 0] = 0.30  # a step 3.02 s after its deadline: outside the window
    assert kept_after(pose, dl).tolist() == [True, False, False, True, True]


@pytest.mark.skipif(not os.path.exists(REC), reason="needs the trainer's recording of the best checkpoint's test")
def test_sole_log_is_the_rigs():
    """The native test records root poses and joints; the scoring side rebuilds the soles, the target and the planted
    anchor from them. On the training simulator's own recording that must give what its rig logged."""
    from utils.landing_native import sole_log

    z = np.load(REC)
    K = 40
    swing, c = torch.tensor(z["swing"][:K]), torch.tensor(z["c"][:K])
    log = sole_log(*(torch.tensor(z[k][:, :K]) for k in ("root_p", "root_quat", "q")), swing, c, int(z["t_cmd"]))
    # the recording's soles are the GPU rig's (it allows TF32 matmuls): measured up to 0.43 mm and 0.0009 rad from
    # the CPU's kinematics on the same states. A wrong quaternion order, foot or state index is centimetres off
    for k in ("pose", "h", "zc"):
        assert np.abs(log[k].numpy() - z[k][:, :K]).max() < 1.5e-3, k
    planted = (np.arange(K), 1 - z["swing"][:K].astype(int))
    assert np.abs(log["target"].numpy() - z["target"][:K]).max() < 1.5e-3
    assert np.abs(log["anchor"].numpy()[planted] - z["anchor"][:K][planted]).max() < 1.5e-3
    assert log["t_cmd"] == 50 and math.isclose(float(log["pose"][0, 0, 0, 1]), 0.10625, abs_tol=1e-4)


def test_commands_move_with_the_stance():
    """After a walk the robot stands in whatever stance the stop left. Each command of the grid is then the same move
    (foot, distance, direction, turn) from that stance, kept only where the sampler would send it from there."""
    from utils.landing_native import stance_commands
    from utils.landing_score import command_grid

    cfg = LandingCfg()
    home = torch.tensor([[0.05, 0.10625, 0.0], [0.05, -0.10625, 0.0]])  # the default stance: left, right sole
    grid = command_grid(cfg, home)
    K = len(grid["swing"])
    same, keep = stance_commands(cfg, grid, home.expand(K, 2, 3))
    assert keep.all() and all(torch.equal(same[k], grid[k]) for k in grid)
    # the left sole 10 cm ahead and turned 0.1 rad toes out
    stance = torch.tensor([[0.15, 0.10625, 0.1], [0.05, -0.10625, 0.0]])
    moved, keep = stance_commands(cfg, grid, stance.expand(K, 2, 3))
    sw = grid["swing"][keep].long()
    cur = lambda p: relative(p[sw], p[1 - sw])  # noqa: E731  each kept row's swing sole in its planted sole's frame
    assert torch.allclose(moved["c"] - grid["c"][keep], cur(stance) - cur(home), atol=1e-6)
    for k in ("swing", "n0", "travel", "direction", "turn"):  # the move itself, and so its clock, is unchanged
        assert torch.equal(moved[k], grid[k][keep]), k
    # the left sole 30 cm straight ahead would land 40 cm ahead of the right one: outside the box (33 cm)
    row = lambda s, d, k: int(((grid["swing"] == s) & (grid["travel"] == d) & (grid["direction"] == k)  # noqa: E731
                               & (grid["turn"] == 0)).nonzero())  # fmt: skip
    assert not keep[row(0, 0.30, 0)] and keep[row(0, 0.20, 0)] and keep[row(0, 0.30, 4)] and 150 < keep.sum() < K


def test_walks_are_dealt_evenly():
    """Every rollout of the hand-over test walks one of WALKS for one of WALK_S seconds and then stands --stop_s
    under the zero command; the pairs are dealt out evenly, in an order the seed draws."""
    from utils.landing_native import WALK_S, WALKS, walks

    w = walks(226, 0, 2.0)
    assert w.shape == (226, 5) and (w[:, 4] == 2.0).all() and not torch.equal(w, walks(226, 1, 2.0))
    pairs = [tuple(round(x, 4) for x in r) for r in w[:, :4].tolist()]
    want = {(*v, s) for v in WALKS for s in WALK_S}
    assert set(pairs) == want and len(want) == 32
    assert {pairs.count(p) for p in want} == {7, 8}  # 226 rollouts over 32 pairs


def _tiny_exports(tmp_path, same_latent: bool):
    """A small random policy's landing export and velocity export (utils/export_onnx.py's DeployNet; that file is a
    script and cannot be imported). same_latent: both encoders answer the same constant latent, so that the two
    files give the same action on the same observation history."""
    import onnx

    from src.networks import Policy, run_mlp
    from utils.export_landing import BothNet, export

    torch.manual_seed(0)
    pi = Policy(3 * J + 12, 16, J, hist=10, enc_hidden=(32,), dec_hidden=(64,), vel=True, land=True)
    with torch.no_grad():
        pi.dec[-1].weight.normal_(0, 0.05)  # a decoder that answers its observations
        if same_latent:
            for enc in (pi.enc_vel, pi.enc_land):
                enc[-1].weight.zero_()
                enc[-1].bias.copy_(torch.linspace(-1, 1, len(enc[-1].bias)))

    class Vel(torch.nn.Module):
        def forward(self, obs_hist, cmd):
            o = torch.cat((obs_hist, upright_rel(obs_hist[..., 2 * J + 3 : 2 * J + 6])), -1)
            z = pi.encode_vel(cmd, o)
            return torch.clamp(run_mlp(pi.dec, torch.cat((z, pi.obs_norm(o).flatten(-2)), -1)), -20.0, 20.0)

    land, vel = str(tmp_path / "land.onnx"), str(tmp_path / "vel.onnx")
    export(_net(pi).eval(), 10, land, {"iteration": 7})
    example = (torch.zeros(1, 10, 3 * J + 6), torch.zeros(1, 3))
    example[0][..., 2 * J + 5] = -1.0
    torch.onnx.export(Vel().eval(), example, vel, input_names=["obs_hist", "cmd"], output_names=["action"],
                      opset_version=17, dynamo=True, external_data=False)  # fmt: skip
    model = onnx.load(vel)
    onnx.helper.set_model_props(model, {"iteration": "7"})
    onnx.save(model, vel)
    both = str(tmp_path / "both.onnx")  # the command box (-1, 1): Vel's raw command
    export(BothNet(pi, KIN, FEET, Q0, LandingCfg(), 20.0, -torch.ones(3), torch.ones(3)).eval(), 10, both,
           {"iteration": 7})  # fmt: skip
    return land, vel, both


def _landtest(tmp_path, name, onnx_path, *more, **arrays):
    """utils/play_mujoco.py's landtest on the commands file made of arrays; the recording."""
    from utils.landing_score import REPO

    cmds, rec = str(tmp_path / f"{name}_commands.npz"), str(tmp_path / f"{name}.npz")
    np.savez(cmds, **arrays)
    subprocess.run(["uv", "run", "utils/play_mujoco.py", "landtest", "--onnx", onnx_path, "--commands", cmds,
                    "--out", rec, "--jobs", "2", "--hold_s", "0.2", *more], check=True, cwd=REPO)  # fmt: skip
    return np.load(rec)


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv: utils/play_mujoco.py runs in its own environment")
def test_handover_carries_the_history(tmp_path):
    """A rollout may begin with a walk and a stop under a velocity export; the landing model then takes over the
    observation history and the last action, and the recording starts at the hand-over. With two exports that give the
    same action on the same history, 20 control steps under the velocity one followed by the landing one must
    therefore be the landing one's own rollout from its 20th state on; the rollout ids pick the lag draws."""
    land, vel, _ = _tiny_exports(tmp_path, same_latent=True)
    swing = np.array([[0.0], [1.0]], np.float32)
    c = np.array([[[0.20, 0.2125, 0.0]], [[0.05, -0.30, -0.4]]], np.float32)
    plain = _landtest(tmp_path, "plain", land, swing=swing, c=c)
    # (vx, vy, yaw rate, seconds walked, seconds stood); the rollouts in the other order
    walk = np.array([[0.0, 0.2, 0.5, 0.4, 0.0], [0.3, 0.0, 0.0, 0.2, 0.2]], np.float32)
    after = _landtest(tmp_path, "after", land, "--walk_onnx", vel, swing=swing[::-1], c=c[::-1], walk=walk,
                      k=np.array([1, 0]))  # fmt: skip
    assert len(set(map(tuple, plain["lags"]))) == 2 and (after["lags"] == plain["lags"][::-1]).all()
    assert (after["t_cmd"] == 50).all() and after["q"].shape == plain["q"].shape and not after["walk_fell"].any()
    assert np.abs(plain["q"][20] - plain["q"][0]).max() > 1e-2  # the robot moves during those 20 steps
    for k in ("q", "root_p", "root_quat", "qd_rms", "root_speed"):
        assert np.abs(after[k][:30] - plain[k][20:50, ::-1]).max() < 1e-4, k


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv: utils/play_mujoco.py runs in its own environment")
@pytest.mark.skipif(not os.path.exists(WALK), reason="needs the T1 walking export in logs/ (not in the repository)")
def test_walk_follows_its_command_and_stops(tmp_path):
    """The walk before the hand-over is the velocity export's, on the command of the rollout's walk row and then on
    the zero command: under the deployed policy the robot is, at the hand-over, where 2 s of that command take it,
    and at rest if it then stood for 2 s. No landing command is sent (the hand-over test's first pass)."""
    from utils.landing_score import REPO
    from utils.rotations import quat_to_mat

    land, _, _ = _tiny_exports(tmp_path, same_latent=False)
    deployed = str(REPO / WALK)
    walk = np.array([[0.5, 0.0, 0.0, 2.0, 0.0], [0.0, 0.4, 0.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0, 0.0],
                     [0.5, 0.0, 0.0, 2.0, 2.0]], np.float32)  # fmt: skip
    z = _landtest(tmp_path, "walk", land, "--walk_onnx", deployed, "--lag", "5", swing=np.zeros((4, 0), np.float32),
                  c=np.zeros((4, 0, 3), np.float32), walk=walk)  # fmt: skip
    assert z["t_cmd"].shape == (4, 0) and z["q"].shape == (11, 4, J) and not z["walk_fell"].any()
    p, speed = z["root_p"][0], z["root_speed"][0]  # at the hand-over
    yaw = torch.atan2(*quat_to_mat(torch.tensor(z["root_quat"][0]))[:, [1, 0], 0].T).numpy()
    assert 0.5 < p[0, 0] < 1.1 and abs(p[0, 1]) < 0.2 and abs(yaw[0]) < 0.3 and speed[0] > 0.3  # forward, walking
    assert 0.4 < p[1, 1] < 0.9 and abs(p[1, 0]) < 0.2 and abs(yaw[1]) < 0.3  # to the left
    assert 1.2 < yaw[2] < 2.2 and np.linalg.norm(p[2, :2]) < 0.3  # turned to the left on the spot
    assert 0.6 < p[3, 0] < 1.3 and speed[3] < 0.05 and z["qd_rms"][0, 3] < 0.05  # walked, then stood


def test_one_file_holds_both_interfaces(tmp_path):
    """The combined export (BothNet) under onnxruntime: with mode 1 it is the landing module on its fed-back state,
    with mode 0 the velocity encoder on the command mapped from its box, answering a state of zeros, so that the
    first landing tick after a walk is a reset's (SETTLE again)."""
    import onnxruntime as ort

    from src.networks import Policy, run_mlp
    from utils.export_landing import STATE_TERMS, BothNet, export

    torch.manual_seed(0)
    hist = 10
    pi = Policy(3 * J + 12, 16, J, hist=hist, enc_hidden=(32,), dec_hidden=(64,), vel=True, land=True)
    with torch.no_grad():
        pi.dec[-1].weight.normal_(0, 0.5)
        pi.obs_norm.mean.normal_(0, 0.1)
    lo, hi = torch.tensor([-1.0, -0.5, -2.0]), torch.tensor([1.5, 0.5, 2.0])
    land = _net(pi).eval()
    path = str(tmp_path / "both.onnx")
    export(BothNet(pi, KIN, FEET, Q0, LandingCfg(), 20.0, lo, hi).eval(), hist, path, {"iteration": 7})
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    S = len(STATE_TERMS)
    assert {x.name: x.shape for x in session.get_inputs()} == {
        "obs_hist": [1, hist, 3 * J + 6], "mode": [1, 1], "vel_cmd": [1, 3], "land_cmd": [1, 5], "state": [1, S]}
    assert [x.name for x in session.get_outputs()] == ["action", "next_state"]
    meta = session.get_modelmeta().custom_metadata_map
    assert json.loads(meta["cmd_low"]) == lo.tolist() and json.loads(meta["state_terms"]) == list(STATE_TERMS)
    settle = STATE_TERMS.index("settle_left")
    s_t, s_o, gap = torch.zeros(1, S), np.zeros((1, S), np.float32), 0.0
    for t in range(80):  # 40 ticks of landing (a command at 30), 10 of walking, 30 of landing
        o = 0.3 * torch.randn(1, hist, 3 * J + 6)
        g = torch.tensor([0.0, 0.0, -1.0]) + 0.2 * torch.randn(3)
        o[..., 2 * J + 3 : 2 * J + 6] = g / g.norm()
        walking = 40 <= t < 50
        v = torch.tensor([[0.7, -0.2, 1.1]])
        cmd = torch.tensor([[1.0, 1.0, 0.15, -0.25, 0.1]]) if t == 30 else torch.zeros(1, 5)
        with torch.no_grad():
            if walking:
                full = torch.cat((o, upright_rel(o[..., 2 * J + 3 : 2 * J + 6])), -1)
                z = pi.encode_vel((2 * v - hi - lo) / (hi - lo), full)
                a_t = torch.clamp(run_mlp(pi.dec, torch.cat((z, pi.obs_norm(full).flatten(-2)), -1)), -20.0, 20.0)
                s_t = torch.zeros(1, S)
                gap = max(gap, float((a_t - land(o, cmd, s_t)[0]).abs().max()))
            else:
                a_t, s_t = land(o, cmd, s_t)
        a_o, s_o = session.run(None, {"obs_hist": o.numpy(), "mode": np.array([[float(not walking)]], np.float32),
                                      "vel_cmd": v.numpy(), "land_cmd": cmd.numpy(), "state": s_o})  # fmt: skip
        assert np.abs(a_o - a_t.numpy()).max() < 1e-4 and np.abs(s_o - s_t.numpy()).max() < 1e-5, t
        if t in (31, 50):  # a step runs; the first landing tick after the walk starts SETTLE again
            assert (s_o[0, STATE_TERMS.index("steps_left")] > 0) == (t == 31)
            assert (s_o[0, settle] == LandingCfg().settle_steps - 1) == (t == 50)
    assert gap > 0.1  # the two interfaces answer differently: the comparison tells which one ran


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv: utils/play_mujoco.py runs in its own environment")
def test_native_handover_inside_one_file(tmp_path):
    """landtest takes the combined export as both --onnx and --walk_onnx: the rollouts are those of the landing
    export after a walk under the velocity export."""
    land, vel, both = _tiny_exports(tmp_path, same_latent=False)
    arrays = dict(swing=np.array([[0.0], [1.0]], np.float32),
                  c=np.array([[[0.20, 0.2125, 0.0]], [[0.05, -0.30, -0.4]]], np.float32),
                  walk=np.array([[0.0, 0.2, 0.5, 0.4, 0.1], [0.3, 0.0, 0.0, 0.2, 0.2]], np.float32), k=np.array([0, 1]))  # fmt: skip
    two = _landtest(tmp_path, "two", land, "--walk_onnx", vel, **arrays)
    one = _landtest(tmp_path, "one", both, "--walk_onnx", both, **arrays)
    assert np.abs(two["q"][0] - np.asarray(SPEC.default_joint_pos_list())).max() > 1e-2  # the walk moved the robot
    for k in ("q", "root_p", "root_quat", "t_cmd", "n0"):
        assert np.abs(one[k] - two[k]).max() < 1e-4, k
