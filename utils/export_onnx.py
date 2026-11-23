"""Export a --vel_frac checkpoint's velocity interface to ONNX in the repo's deploy format (the --onnx input of
utils/play_mujoco.py), so the robot-side runner and the MuJoCo tools take the file as they take deploy/'s.

usage: python -m utils.export_onnx --ckpt logs/t1_vel_ankle_x2_s1/policy/ckpt_013000.pt    # writes .../ckpt_013000.onnx

Inputs obs_hist (1, hist, 3 J + 6) and cmd (1, 3), output action (1, J), all float32. The observation is what a robot
measures: joint positions and speeds, gyro, gravity direction, previous action. The six reference-orientation entries
the policy also reads are rebuilt inside the model from the gravity direction (mdp.upright_rel), and the action is
clamped as training's. The model's metadata lists the observation terms, joint order, default pose, action scale and
gains.

The file is written only once it reproduces the checkpoint: the training simulator (evaluation conditions) is driven
through SCRIPT by the ONNX model under onnxruntime, and each action is compared with PyTorch's on the observation
training builds (mdp.obs).

Then, from the repo root, the result runs in native MuJoCo (utils/play_mujoco.py brings its own environment):
    uv run utils/play_mujoco.py steptest --onnx logs/<run>/policy/ckpt_013000.onnx
"""

import argparse
import copy
import json
import os

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--tol", type=float, default=1e-4, help="largest action difference from PyTorch that passes")
parser.add_argument("--device", default="cuda:0",
                    help="where the checkpoint and the self-check's simulator run; cpu (with CUDA_VISIBLE_DEVICES= ) "
                         "needs no GPU")
args = parser.parse_args()

import numpy as np
import onnx
import onnxruntime as ort
import torch
from utils.velocity_rig import ACTION_CLIP, SCRIPT, Rig  # after the arguments: it starts Warp

import src.networks as networks
from src.env import PHYSICS_DT
from src.mdp import upright_rel
from src.motion import CONTROL_DT
from src.networks import run_mlp


class DeployNet(torch.nn.Module):
    """a = clamp(dec(enc_vel(normalize(cmd), o), o)) with o = [obs, upright_rel(obs's gravity direction)]; obs_hist
    (B, hist, 3 J + 6) newest last, cmd (B, 3) as (vx, vy) [m/s] in the heading frame and the yaw rate [rad/s]."""

    def __init__(self, pi, cmd_lo: torch.Tensor, cmd_hi: torch.Tensor, num_joints: int):
        super().__init__()
        self.pi, self.g0 = pi, 2 * num_joints + 3  # the gravity direction's first entry
        self.register_buffer("cmd_mid", (cmd_hi + cmd_lo) / 2)
        self.register_buffer("cmd_half", (cmd_hi - cmd_lo) / 2)

    def forward(self, obs_hist: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        pi = self.pi
        o = torch.cat((obs_hist, upright_rel(obs_hist[..., self.g0 : self.g0 + 3])), -1)
        z = pi.encode_vel((cmd - self.cmd_mid) / self.cmd_half, o)
        a = run_mlp(pi.dec, torch.cat((z, pi.obs_norm(o).flatten(-2)), -1))  # decode_frozen without the stop-gradient
        return torch.clamp(a, -ACTION_CLIP, ACTION_CLIP)


rig = Rig(args.ckpt, dev=args.device)
robot, mdp, J = rig.robot, rig.mdp, rig.robot.num_joints
OBS = 3 * J + 6
assert mdp.obs_dim == OBS + 6, "the observation is not [joints, joint speeds, gyro, gravity, previous action, 6 entries]"
networks.AMP_DTYPE = None  # the exported model is float32 (training ran under autocast)
net = DeployNet(copy.deepcopy(rig.pi).cpu(), mdp.cmd_lo.cpu(), mdp.cmd_hi.cpu(), J).eval()  # the rig's stays on the GPU
path = os.path.splitext(args.ckpt)[0] + ".onnx"
tmp = path + ".tmp"  # until the check passes
example = (torch.zeros(1, rig.hist_len, OBS), torch.zeros(1, 3))
example[0][..., 2 * J + 5] = -1.0  # upright: the gravity direction (0, 0, -1)
# opset 17 and its IR version, as exported at tag supermimic-frozen-2026-11-15: older onnxruntime and TensorRT read them
torch.onnx.export(net, example, tmp, input_names=["obs_hist", "cmd"], output_names=["action"], opset_version=17,
                  dynamo=True, external_data=False)  # fmt: skip
kp, kd, _ = robot.pd_gains()
lag = robot.delay_range  # physics steps
model = onnx.load(tmp)
assert [(o.domain, o.version) for o in model.opset_import] == [("", 17)], model.opset_import
model.ir_version = onnx.helper.find_min_ir_version_for(list(model.opset_import))
onnx.helper.set_model_props(model, {k: json.dumps(v) for k, v in {
    "ckpt": args.ckpt, "iteration": rig.iteration, "control_dt": CONTROL_DT,
    "trainer": "blinkmimic2 train.py --vel_frac (the six reference-orientation entries are rebuilt inside this model)",
    "obs_hist": f"(1, {rig.hist_len}, {OBS}) float32, newest last; at a reset the first observation repeated",
    "obs_terms": [["joint_pos", J], ["joint_vel", J], ["ang_vel_b", 3], ["gravity_b", 3], ["prev_action", J]],
    "obs_terms_doc": "joint_pos: q - default_q [rad]; joint_vel [rad/s]; ang_vel_b: root (IMU) angular velocity in "
                     "the root frame [rad/s]; gravity_b: R^T (0, 0, -1), the unit gravity direction in the root "
                     "frame, (0, 0, -1) upright; prev_action: the previous action (this model's raw output), zeros "
                     "at a reset",
    "cmd": "(1, 3) float32: vx, vy [m/s] in the heading frame, yaw rate (world z) [rad/s]; not clipped here, "
           "trained within cmd_low..cmd_high",
    "cmd_low": mdp.cmd_lo.tolist(), "cmd_high": mdp.cmd_hi.tolist(),
    "action": f"(1, {J}) float32, within +-{ACTION_CLIP:g}; joint setpoint = default_q + action_scale * action, "
              f"tracked by PD with kp, kd (training's), which applied each setpoint {lag[0] * PHYSICS_DT * 1e3:g}-"
              f"{lag[1] * PHYSICS_DT * 1e3:g} ms late",
    "joint_names": list(robot.joint_names), "default_q": robot.default_joint_pos_list(),
    "action_scale": robot.action_scale_list(), "kp": list(kp), "kd": list(kd),
    # where the gains to send differ from training's simulated ones (robot/t2_measured.py: the waist and ankle kd)
    **({} if robot.sent_gains is None else {"kp_sent": list(robot.sent_gains[0]), "kd_sent": list(robot.sent_gains[1]),
                                            "gains_doc": "send kp_sent / kd_sent to the robot; kp / kd are the joint-side "
                                                         "gains the training simulator used"}),
}.items()})  # fmt: skip
onnx.checker.check_model(model)
onnx.save(model, tmp)

opts = ort.SessionOptions()
opts.inter_op_num_threads = 1
session = ort.InferenceSession(tmp, opts, providers=["CPUExecutionProvider"])
amp = torch.bfloat16 if rig.args.get("amp", "none") == "bf16" else None  # the autocast training ran its networks under
rig.reset()
diff, diff_amp, fell, speeds = 0.0, 0.0, False, []
for duration, cmd in SCRIPT:
    for _ in range(round(duration / CONTROL_DT)):
        hist = rig.observe()  # training's observation, the six entries included
        networks.AMP_DTYPE = None
        ref = rig.act(cmd)  # PyTorch in float32: what the export must reproduce
        networks.AMP_DTYPE = amp
        ref_amp = rig.act(cmd) if amp is not None else ref
        (a,) = session.run(None, {"obs_hist": hist[:, :, :OBS].cpu().numpy(), "cmd": np.asarray([cmd], dtype=np.float32)})
        a = torch.from_numpy(a).to(ref.device)
        diff, diff_amp = max(diff, float((a - ref).abs().max())), max(diff_amp, float((a - ref_amp).abs().max()))
        vel, down = rig.apply(a)  # the ONNX model's action drives the simulator
        fell |= down
    speeds.append(vel)
print(f"the script driven by onnxruntime in the training simulator: largest action difference from PyTorch in float32 "
      f"{diff:.2e} (from PyTorch under training's autocast: {diff_amp:.2e}, rounding), fell: {fell}")
for (_, cmd), vel in zip(SCRIPT, speeds):
    print(f"  cmd {cmd}: velocity at the segment's end {np.round(vel, 2).tolist()}")
assert diff < args.tol and not fell, f"the ONNX model does not reproduce the checkpoint; left at {tmp}"
os.replace(tmp, path)
print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
