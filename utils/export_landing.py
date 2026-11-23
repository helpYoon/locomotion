"""Export a --land_frac checkpoint's landing interface to ONNX with the command state inside the model, so that a
caller (utils/play_mujoco.py's landtest, a planner's executor, the robot's runner) sends a foot and a landing pose and
keeps no kinematics, clock or anchors of its own.

usage: python -m utils.export_landing --ckpt logs/land_r4_d10_al2_s3/ckpt_004000.pt   # writes .../ckpt_004000_land.onnx
       python -m utils.export_landing --ckpt logs/land_r4_d10_al2_s3/ckpt_004000.pt --both   # .../ckpt_004000_both.onnx

Inputs, all float32:
  obs_hist (1, hist, 3 J + 6)  what a robot measures, as the velocity export's (utils/export_onnx.py): joint positions
                               and speeds, gyro, gravity direction, previous action; newest last.
  cmd (1, 5)                   go, foot, x, y, yaw. go > 0.5: a command arrives on this tick. foot: the sole to move
                               (0 left, 1 right). x, y [m], yaw [rad]: where its centre lands and how it is turned, in
                               the other (planted) sole's frame as it stands on this tick.
  state (1, 8)                 zeros at a reset, then the previous tick's next_state (STATE_TERMS).
Outputs: action (1, J), clamped as training's, and next_state (1, 8).

--both writes one file that also holds the checkpoint's velocity interface (utils/export_onnx.py's), on the one
decoder. Its inputs are obs_hist, state, the command above as land_cmd, and
  mode (1, 1)                  above 0.5: the landing interface. Else the velocity interface: the action follows
                               vel_cmd and next_state is zeros, so the first landing tick after it is a reset's.
  vel_cmd (1, 3)               vx, vy [m/s] in the heading frame and the yaw rate [rad/s]; read in velocity mode only.
A runner stands on the zero velocity command for a second or more before it switches mode to landing.

The command state is the trainer's (src/landing.py: SETTLE after a reset, STEP while a clock runs, HOLD), rebuilt from
the observation alone. The trainer keeps both soles' world poses (the anchors), but its encoder reads them only as one
sole's pose in the other's frame when they were last latched, and the joint angles with the gravity direction give
that. So the model needs no estimate of where the robot is. A command is taken only in HOLD (else ignored, not queued);
its clock is looked up from the step's size (LandingCfg.clock_*, the sampler's rule without its jitter). The state
tells the caller whether its command was taken (steps_left above 0 in the next_state of the tick it was sent on), when
the step is over (steps_left back at 0) and, from the tick after, the stance the robot landed in.

The file is written only once it reproduces the checkpoint: the training simulator (evaluation conditions, one world)
is driven through SCRIPT by the ONNX model under onnxruntime, and on every control step its action is compared with
PyTorch's on the trainer's own command state and encoder inputs (env.land_command), and its clocks with the trainer's.
That simulator runs on the CPU here (--device), with CUDA hidden from the process: the export touches no GPU.

Then, from the repo root, the result is tested in native MuJoCo:
    python -m utils.landing_native --onnx logs/<run>/ckpt_004000_land.onnx
"""

import argparse
import copy
import json
import math
import os

import torch

from src import landing
from src.landing import SOLE_C, SOLE_HX, SOLE_HY, YAW_M, LandingCfg, _dist, _pick, relative, wrap
from src.mdp import upright_rel
from src.networks import run_mlp

# next_state / state: 1 once started; the commanded sole (0 left, 1 right); control steps left on the clock (0: no
# step runs); control steps of SETTLE left; 1 when the stance is to be read from the next observation; then, while a
# clock runs, the commanded landing pose, else the stance held (the commanded sole's pose in the other sole's frame)
STATE_TERMS = ("started", "foot", "steps_left", "settle_left", "latch", "x", "y", "yaw")
# the self-check's commands, one after another: control steps of HOLD before the command (0: on the state the previous
# deadline ends in), the foot (0 left, 1 right), and the move of that foot's sole from where it stands, as
# utils/landing_score.py's grid: forward, outward [m], turn (toes out positive) [deg]
SCRIPT = ((25, 0, 0.20, 0.0, 0.0), (0, 1, 0.30, 0.0, 0.0), (50, 0, 0.10, 0.08, 15.0), (50, 1, 0.0, 0.0, 30.0),
          (50, 0, -0.10, -0.06, -10.0))  # fmt: skip
END_HOLD = 100  # control steps of hold after the last deadline
# --both: the velocity commands its self-check sends, (vx, vy) [m/s] in the heading frame and the yaw rate [rad/s]
VEL_SCRIPT = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.4, 0.0, 1.0), (-0.6, 0.0, 0.0), (1.0, 0.0, 0.0))


def clock(cfg: LandingCfg, cur: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """The control steps (B,) a command is given, looked up from the step's size as the sampler does
    (landing.land_sample) without its jitter: cur (B, 3) the pose of the sole to move and c (B, 3) its landing pose,
    both in the planted sole's frame."""
    travel = _dist(c[:, :2] - cur[:, :2]) + YAW_M * torch.sqrt(wrap(c[:, 2] - cur[:, 2]) ** 2 + 1e-12)
    return torch.clamp(torch.round(cfg.clock_base + cfg.clock_per_m * travel), *cfg.clock_range)


class LandNet(torch.nn.Module):
    """a = clamp(dec(enc_land(f, o), o)) and the next state, with o = [obs, upright_rel(obs's gravity direction)] and
    f the 12 command inputs of the command state (``command``). kin: the robot's kinematics on the CPU; feet: its
    tracked-body indices of the left and the right foot link; clip: training's action clamp."""

    def __init__(self, pi, kin, feet: list[int], default_q: torch.Tensor, cfg: LandingCfg, clip: float):
        super().__init__()
        self.pi, self.kin, self.feet, self.cfg, self.clip = pi, kin, feet, cfg, clip
        self.J = len(default_q)
        self.register_buffer("default_q", default_q.clone())

    def command(self, o: torch.Tensor, cmd: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One tick of the command state on the observation o (B, 3 J + 12): the 12 inputs of enc_land (B, 12) and the
        next state (B, 8). In the trainer's order: the stance latched by the tick before (or by a reset) is read off
        this observation, a command that arrives in HOLD is taken, the features are built, the clocks advance."""
        cfg, J = self.cfg, self.J
        # the soles with the root at the origin and its heading taken out (rows of H^T R: the columns of the six
        # reference-orientation entries R^T H and their cross product): the floor's plane is kept, so one sole's
        # pose in the other's frame is the world's
        c0, c1 = o[:, 3 * J + 6 :: 2], o[:, 3 * J + 7 :: 2]
        ax, ay, az, bx, by, bz = (c[:, i] for c in (c0, c1) for i in range(3))
        # written out: torch.cross exports as an opset-18 Split, which has no down-conversion to 17
        R = torch.stack((c0, c1, torch.stack((ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx), -1)), -2)
        q, z = o[:, :J] + self.default_q, torch.zeros_like(c0)
        bp, bR, bv, bw = (x[:, self.feet] for x in self.kin.forward(z, R, z, z, q, torch.zeros_like(q)))
        pose = landing.soles(bp, bR, bv, bw)["pose"]
        rel = lambda w: relative(_pick(pose, w), _pick(pose, 1 - w))  # noqa: E731  sole w in the other sole's frame

        foot, n, m, t = state[:, 1], state[:, 2], state[:, 3], state[:, 5:8]
        fresh = state[:, 0] < 0.5
        m = torch.where(fresh, torch.full_like(m, cfg.settle_steps), m)
        t = torch.where((fresh | (state[:, 4] > 0.5))[:, None], rel(foot), t)
        w, c = (cmd[:, 1] > 0.5).to(o.dtype), cmd[:, 2:5]
        take = (cmd[:, 0] > 0.5) & (m < 0.5) & (n < 0.5)  # landing.land_issue: in HOLD only
        n = torch.where(take, clock(cfg, rel(w), c), n)
        foot, t = torch.where(take, w, foot), torch.where(take[:, None], c, t)
        # the trainer's command state with the planted anchor at the origin: its target and the commanded sole's
        # anchor are then both t, so the phases and the features are landing.phase's and landing.land_features' own
        left = (foot < 0.5)[:, None]
        zero = torch.zeros_like(t)
        C = torch.cat((foot[:, None], n[:, None], m[:, None], t, torch.where(left, t, zero),
                       torch.where(left, zero, t)), -1)  # fmt: skip
        feat = landing.land_features(cfg, C, landing.sole_rel(bp, bR, foot))
        settle, stepping, _ = landing.phase(C)
        latch = (settle & (m < 1.5)) | (stepping & (n < 1.5))  # landing.land_step: SETTLE's last step, the deadline
        col = lambda x: x.to(o.dtype)[:, None]  # noqa: E731
        return feat, torch.cat((torch.ones_like(col(n)), col(foot), col(n) - col(stepping), col(m) - col(settle),
                                col(latch), t), -1)  # fmt: skip

    def full(self, obs_hist: torch.Tensor) -> torch.Tensor:
        """The policy's observation history: obs_hist with the six reference-orientation entries."""
        return torch.cat((obs_hist, upright_rel(obs_hist[..., 2 * self.J + 3 : 2 * self.J + 6])), -1)

    def act(self, z: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        a = run_mlp(self.pi.dec, torch.cat((z, self.pi.obs_norm(o).flatten(-2)), -1))  # decode_frozen, no stop-gradient
        return torch.clamp(a, -self.clip, self.clip)

    def forward(self, obs_hist: torch.Tensor, cmd: torch.Tensor, state: torch.Tensor):
        o = self.full(obs_hist)
        feat, nxt = self.command(o[:, -1], cmd, state)
        return self.act(self.pi.encode_land(feat, o), o), nxt


class BothNet(LandNet):
    """LandNet and the velocity interface (utils/export_onnx.py's DeployNet) in front of the one decoder: mode (B, 1)
    above 0.5 picks LandNet's latent and next state, else enc_vel's on vel_cmd (B, 3), mapped from the command box
    cmd_lo..cmd_hi, with a next state of zeros (the landing interface then starts as after a reset)."""

    def __init__(self, pi, kin, feet, default_q, cfg, clip, cmd_lo: torch.Tensor, cmd_hi: torch.Tensor):
        super().__init__(pi, kin, feet, default_q, cfg, clip)
        self.register_buffer("cmd_lo", cmd_lo.clone())
        self.register_buffer("cmd_hi", cmd_hi.clone())

    def forward(self, obs_hist, mode, vel_cmd, land_cmd, state):
        o, land = self.full(obs_hist), mode > 0.5
        feat, nxt = self.command(o[:, -1], land_cmd, state)
        v = (vel_cmd - (self.cmd_hi + self.cmd_lo) / 2) / ((self.cmd_hi - self.cmd_lo) / 2)  # DeployNet's
        z = torch.where(land, self.pi.encode_land(feat, o), self.pi.encode_vel(v, o))
        return self.act(z, o), torch.where(land, nxt, torch.zeros_like(nxt))


def export(net: LandNet, hist: int, path: str, props: dict) -> None:
    """Write net to path as utils/export_onnx.py writes the velocity interface (opset 17 and its IR version, which
    older onnxruntime and TensorRT read), with props (JSON-encoded) as the model's metadata, and with the state's
    layout and the clock's range, which a caller reads (utils/play_mujoco.py's landtest); a BothNet's inputs and
    its velocity command box besides."""
    import onnx

    J = net.J
    props = {**props, "state_terms": list(STATE_TERMS), "clock_range": list(net.cfg.clock_range)}
    example = (torch.zeros(1, hist, 3 * J + 6), torch.zeros(1, 5), torch.zeros(1, len(STATE_TERMS)))
    names = ["obs_hist", "cmd", "state"]
    if isinstance(net, BothNet):
        props.update(cmd_low=net.cmd_lo.tolist(), cmd_high=net.cmd_hi.tolist())
        example = (example[0], torch.ones(1, 1), torch.zeros(1, 3), *example[1:])
        names = ["obs_hist", "mode", "vel_cmd", "land_cmd", "state"]
    example[0][..., 2 * J + 5] = -1.0  # upright: the gravity direction (0, 0, -1)
    torch.onnx.export(net, example, path, input_names=names, opset_version=17, dynamo=True,
                      output_names=["action", "next_state"], external_data=False)  # fmt: skip
    model = onnx.load(path)
    assert [(o.domain, o.version) for o in model.opset_import] == [("", 17)], model.opset_import
    model.ir_version = onnx.helper.find_min_ir_version_for(list(model.opset_import))
    onnx.helper.set_model_props(model, {k: json.dumps(v) for k, v in props.items()})
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--both", action="store_true", help="one file with the velocity interface too (mode input)")
    parser.add_argument("--tol", type=float, default=1e-4, help="largest action difference from PyTorch that passes")
    parser.add_argument("--device", default="cpu", help="where the self-check's simulator runs (one world)")
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before Warp starts: no GPU that another job may be using

    import numpy as np
    import onnxruntime as ort
    from utils.landing_rig import Rig  # after the arguments: it starts Warp
    from utils.landing_score import TOL
    from utils.velocity_rig import ACTION_CLIP

    import src.networks as networks
    from robot.kinematics import Kinematics
    from src.env import PHYSICS_DT
    from src.landing import _separated
    from src.motion import CONTROL_DT

    rig = Rig(args.ckpt, 1, args.device)
    robot, mdp, env, pi, cfg, dev = rig.robot, rig.mdp, rig.env, rig.pi, rig.cfg, rig.dev
    J = robot.num_joints
    OBS = 3 * J + 6
    assert mdp.obs_dim == OBS + 6, "the observation is not [joints, speeds, gyro, gravity, previous action, 6 entries]"
    networks.AMP_DTYPE = None  # the exported model is float32 (training ran under autocast)
    parts = (copy.deepcopy(pi).cpu(), Kinematics(robot, "cpu"), mdp.feet, mdp.default_q.cpu(), cfg, ACTION_CLIP)
    net = (BothNet(*parts, mdp.cmd_lo.cpu(), mdp.cmd_hi.cpu()) if args.both else LandNet(*parts)).eval()
    path = os.path.splitext(args.ckpt)[0] + ("_both.onnx" if args.both else "_land.onnx")
    tmp = path + ".tmp"  # until the check passes
    kp, kd, _ = robot.pd_gains()
    lag = robot.delay_range  # physics steps
    props = {
        "ckpt": args.ckpt, "iteration": rig.iteration, "control_dt": CONTROL_DT,
        "trainer": "train.py --land_frac (the command state, the foot kinematics and the six reference-orientation "
                   "entries are inside this model)",
        "obs_hist": f"(1, {rig.hist_len}, {OBS}) float32, newest last; at a reset the first observation repeated",
        "obs_terms": [["joint_pos", J], ["joint_vel", J], ["ang_vel_b", 3], ["gravity_b", 3], ["prev_action", J]],
        "obs_terms_doc": "joint_pos: q - default_q [rad]; joint_vel [rad/s]; ang_vel_b: root (IMU) angular velocity in "
                         "the root frame [rad/s]; gravity_b: R^T (0, 0, -1), the unit gravity direction in the root "
                         "frame, (0, 0, -1) upright; prev_action: the previous action (this model's raw output), zeros "
                         "at a reset",
        "cmd": "(1, 5) float32: go, foot, x, y, yaw. go > 0.5: a command arrives on this tick; it is taken only while "
               "the robot holds (steps_left and settle_left both 0 in the state passed in), else ignored, not queued. "
               "It was taken if next_state's steps_left is above 0 on this tick; its deadline has passed when "
               "steps_left is back at 0. foot: the sole to move, 0 left, 1 right. x, y [m], yaw [rad]: where that sole's centre lands and how "
               "it is turned, in the other (planted) sole's frame as it stands on this tick (x along its heading, y "
               "to its left). Not clipped here; trained within box_x, box_y, box_yaw (the landed stance, y and yaw "
               "with the sign changed for the right sole), travel and sole_gap. Zeros when there is no command",
        "state": f"(1, {len(STATE_TERMS)}) float32: zeros at a reset, then the previous tick's next_state "
                 "(state_terms)",
        "state_terms_doc": "started: 1 after the first tick. foot: the commanded sole (before the first command the "
                           "value passed in at the reset: training drew it at random). steps_left: control steps to "
                           "the deadline, 0 when no step runs. settle_left: control steps left of the "
                           f"{cfg.settle_steps} after a reset in which no command is taken. latch: 1 when the stance "
                           "is read from the next observation (after SETTLE and after a deadline). x, y, yaw: while "
                           "steps_left > 0 the commanded landing pose, else the stance held, the commanded sole's "
                           "pose in the other sole's frame as the joints and the gravity direction give it; after a "
                           "deadline it is the landed stance from the first tick with latch 0",
        "action": f"(1, {J}) float32, within +-{ACTION_CLIP:g}; joint setpoint = default_q + action_scale * action, "
                  f"tracked by PD with kp, kd (training's), which applied each setpoint {lag[0] * PHYSICS_DT * 1e3:g}-"
                  f"{lag[1] * PHYSICS_DT * 1e3:g} ms late",
        "box_x": cfg.box_x, "box_y": cfg.box_y, "box_yaw": cfg.box_yaw, "travel": cfg.travel,
        "travel_doc": f"distance + {YAW_M} m x |yaw change| between the sole's pose now and its landing pose [m]",
        "sole_gap": cfg.sole_gap, "settle_steps": cfg.settle_steps,
        "clock": f"steps_left of a command = clamp(round({cfg.clock_base:g} + {cfg.clock_per_m:g} x travel), "
                 f"{cfg.clock_range[0]}, {cfg.clock_range[1]}) control steps",
        "sole": "a sole's pose = the centre of the foot link's collision-box bottom face, sole_centre in the link's "
                "frame, projected on the floor, and the link's heading; sole_half: its half length and half width [m]",
        "sole_centre": list(SOLE_C), "sole_half": [SOLE_HX, SOLE_HY],
        "joint_names": list(robot.joint_names), "default_q": robot.default_joint_pos_list(),
        "action_scale": robot.action_scale_list(), "kp": list(kp), "kd": list(kd),
    }  # fmt: skip
    if args.both:
        props["land_cmd"] = props.pop("cmd")
        props["mode"] = ("(1, 1) float32: above 0.5 the landing interface (land_cmd, state), else the velocity "
                         "interface (vel_cmd), which answers a next_state of zeros: the first landing tick after it "
                         "is a reset's, so no command is taken for settle_steps. Stand on the zero velocity command "
                         "for a second or more before switching to landing")
        props["vel_cmd"] = ("(1, 3) float32: vx, vy [m/s] in the heading frame, yaw rate (world z) [rad/s]; not "
                            "clipped here, trained within cmd_low..cmd_high; read in velocity mode only")
    export(net, rig.hist_len, tmp, props)

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(tmp, opts, providers=["CPUExecutionProvider"])
    rig.reset()
    state = np.zeros((1, len(STATE_TERMS)), np.float32)
    state[0, 1] = float(env.land_C[0, landing.SWING])  # the commanded sole this reset drew
    lo, hi = (-cfg.box_x, cfg.box_y[0], cfg.box_yaw[0]), (cfg.box_x, cfg.box_y[1], cfg.box_yaw[1])
    moves, landed = list(SCRIPT), []
    hist, held, left, diff, fell, clocks = None, 0, END_HOLD, 0.0, False, True
    diff_vel, cleared, tick = 0.0, True, 0  # --both: the velocity interface on the same observations
    ones = torch.ones(1, dtype=torch.bool, device=dev)
    while left > 0:
        o = env.observe()
        x = o.unsqueeze(1)
        hist = x.repeat(1, rig.hist_len, 1) if hist is None else torch.cat((hist[:, 1:], x), 1)
        cmd = np.zeros((1, 5), np.float32)
        hold = bool(landing.phase(env.land_C)[2])
        if hold and moves and held == moves[0][0]:  # a command arrives on this state
            _, foot, dx, dy, turn = moves.pop(0)
            swing = torch.full((1,), float(foot), device=dev)
            mirror = torch.tensor([1.0, 1.0 - 2 * foot, 1.0 - 2 * foot], device=dev)
            cur = relative(_pick(env.land_pose, swing), _pick(env.land_pose, 1 - swing))
            cand = cur * mirror + torch.tensor([dx, dy, math.radians(turn)], device=dev)
            c = cand * mirror
            n0 = clock(cfg, cur, c)
            travel = float(_dist(c[:, :2] - cur[:, :2]) + YAW_M * wrap(c[:, 2] - cur[:, 2]).abs())
            inside = all(lo[i] - TOL <= float(cand[0, i]) <= hi[i] + TOL for i in range(3))
            assert inside and bool(_separated(cand, cfg.sole_gap)) and cfg.travel[0] <= travel <= cfg.travel[1], \
                f"the script's command {len(landed) + 1} is outside what the sampler sends: {cand.tolist()}"
            env.land_C.copy_(landing.land_issue(env.land_C, env.land_pose, ones, swing, c, n0))
            cmd[0] = (1.0, foot, *c[0].tolist())
            landed.append([foot, dx, dy, turn, int(n0), None, None])
        held = held + 1 if hold else 0
        cf = env.land_command(o)  # the trainer's 12 inputs of its own command state
        with torch.no_grad():
            ref = torch.clamp(pi.decode_frozen(pi.encode_land(cf, hist), hist).float(), -ACTION_CLIP, ACTION_CLIP)
        feed = {"obs_hist": hist[:, :, :OBS].cpu().numpy(), "state": state}
        if args.both:  # velocity mode first, on a command that changes every half second; its action is not applied
            v = torch.tensor([VEL_SCRIPT[tick // 25 % len(VEL_SCRIPT)]], device=dev)
            tick += 1
            with torch.no_grad():
                ref_v = pi.decode_frozen(pi.encode_vel(mdp.normalize_cmd(v), hist), hist).float()
            a_v, s_v = session.run(None, {**feed, "mode": np.zeros((1, 1), np.float32), "vel_cmd": v.cpu().numpy(),
                                          "land_cmd": cmd})  # fmt: skip
            diff_vel = max(diff_vel, float((torch.from_numpy(a_v).to(dev) - ref_v.clamp(-ACTION_CLIP, ACTION_CLIP)).abs().max()))
            cleared &= not s_v.any()
            feed.update(mode=np.ones((1, 1), np.float32), vel_cmd=np.zeros((1, 3), np.float32), land_cmd=cmd)
        else:
            feed["cmd"] = cmd
        a, state = session.run(None, feed)
        a = torch.from_numpy(a).to(dev)
        diff = max(diff, float((a - ref).abs().max()))
        out = env.step(a)  # the ONNX model's action drives the simulator
        clocks &= state[0, 1:4].tolist() == env.land_C[0, :3].tolist()  # commanded sole, clock, SETTLE's clock
        fell |= bool(mdp.fall_termination(env.state())[0])
        if bool(out["land_terms"]["deadline"][0]):
            landed[-1][5:] = float(out["land_terms"]["e"][0]), float(out["land_terms"]["a"][0])
        if not moves and bool(landing.phase(env.land_C)[2]):
            left -= 1
    print(f"the script driven by onnxruntime in the training simulator ({dev}): largest action difference from PyTorch "
          f"in float32 {diff:.2e}, the command state's clocks equal on every control step: {clocks}, fell: {fell}")
    for foot, dx, dy, turn, n0, e, ang in landed:
        print(f"  {('left', 'right')[foot]} sole {100 * dx:+.0f} cm forward, {100 * dy:+.0f} cm outward, {turn:+.0f} "
              f"deg: clock {n0} steps, landed {100 * e:.2f} cm and {math.degrees(ang):.2f} deg from the target")
    if args.both:
        print(f"velocity mode on the same {tick} observations, {len(VEL_SCRIPT)} commands in turn: largest action "
              f"difference from PyTorch {diff_vel:.2e}, next_state zeros on every step: {cleared}")
    assert diff_vel < args.tol and cleared, f"velocity mode does not reproduce the checkpoint; left at {tmp}"
    assert diff < args.tol and clocks and not fell, f"the ONNX model does not reproduce the checkpoint; left at {tmp}"
    os.replace(tmp, path)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
