"""Drive a --vel_frac checkpoint's velocity interface with the keyboard in the MuJoCo viewer (needs a display). The
robot runs in the training simulator under evaluation conditions: no randomization, noise or pushes, the fixed
evaluation delay. The simulator is built from the checkpoint's own robot, so a policy of the measured T2
(robot/t2_measured.py) runs on that robot's gains, joint friction and waist and ankle damping.

usage: python -m utils.play_velocity --ckpt logs/t1_vel_s1/policy/ckpt_006500.pt
       python -m utils.play_velocity --ckpt logs/t1_vel_s1/policy/ckpt_006500.pt --script   # no viewer, see SCRIPT
       CUDA_VISIBLE_DEVICES= python -m utils.play_velocity --ckpt logs/t2m_vel_s2/policy/ckpt_013000.pt --device cpu
           # without the GPU: one world runs at about real time on the CPU (a control step takes about 20 ms)

Keys (utils/play_mujoco.py's viewer): up / down vx +-0.1 m/s, a / d vy +-0.1 m/s, left / right yaw rate
+-0.2 rad/s (capped at the command box), space stops, backspace resets the robot. It starts standing and resets itself
after a fall.
"""

import argparse
import time

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--script", action="store_true", help="no viewer: run SCRIPT and print commanded vs actual velocity")
parser.add_argument("--worlds", type=int, default=1, help="identical copies simulated (world 0 is shown)")
parser.add_argument("--device", default="cuda:0", help="where the policy and the simulator run; cpu (with CUDA_VISIBLE_DEVICES= ) needs no GPU")
args = parser.parse_args()

import torch
from utils.velocity_rig import SCRIPT, Rig  # after the arguments: it starts Warp

from src.motion import CONTROL_DT
from utils.rotations import mat_to_quat

STEADY = 2.0  # --script: the actual velocity is averaged over each segment's last STEADY seconds

rig = Rig(args.ckpt, args.worlds, dev=args.device)
env, mdp, robot, N = rig.env, rig.mdp, rig.robot, rig.N
reset, step = rig.reset, rig.step  # step(cmd): one control step; world 0's velocity and whether it has fallen


def run_script():
    reset()
    t0, falls = time.perf_counter(), 0
    print("  commanded (vx, vy, yaw)    actual, mean of the last 2 s    fell")
    for dur, cmd in SCRIPT:
        rows, fell = [], False
        for _ in range(round(dur / CONTROL_DT)):
            act, down = step(cmd)
            rows.append(act)
            if down:
                fell, falls = True, falls + 1
                reset()
        tail = torch.tensor(rows[-round(STEADY / CONTROL_DT) :]).mean(0).tolist()
        print(f"  {cmd[0]:+5.2f} {cmd[1]:+5.2f} {cmd[2]:+5.2f}          {tail[0]:+5.2f} {tail[1]:+5.2f} {tail[2]:+5.2f}"
              f"               {'yes' if fell else 'no'}")  # fmt: skip
    sim_s = sum(d for d, _ in SCRIPT)
    print(f"{falls} falls; {sim_s} s simulated in {time.perf_counter() - t0:.1f} s with {N} world(s)")


def visual():
    """The robot's own MJCF (the simulated model has neither meshes nor visual geoms) with its data, and a function
    that poses it as world 0."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(robot.mjcf_path)
    d = mujoco.MjData(m)
    root = int(m.jnt_qposadr[0])  # the floating base
    qadr = [int(m.jnt_qposadr[m.joint(n).id]) for n in robot.joint_names]

    def show():
        p, R, _, _, q, _ = env.state()
        d.qpos[root : root + 3] = p[0].cpu().numpy()
        d.qpos[root + 3 : root + 7] = mat_to_quat(R[0]).cpu().numpy()
        d.qpos[qadr] = q[0].cpu().numpy()
        mujoco.mj_forward(m, d)

    return m, d, show


def run_viewer():
    import glfw  # a mujoco dependency
    import mujoco
    import mujoco.viewer

    m, d, show = visual()
    cmd, want_reset = [0.0, 0.0, 0.0], [False]
    keys = {glfw.KEY_UP: (0, 0.1), glfw.KEY_DOWN: (0, -0.1), glfw.KEY_A: (1, 0.1), glfw.KEY_D: (1, -0.1),
            glfw.KEY_LEFT: (2, 0.2), glfw.KEY_RIGHT: (2, -0.2)}  # fmt: skip
    lo, hi = mdp.cmd_lo.tolist(), mdp.cmd_hi.tolist()

    def key(k):  # the viewer's UI thread; plain float writes under the GIL
        if k in keys:
            i, dv = keys[k]
            cmd[i] = min(max(round(cmd[i] + dv, 2), lo[i]), hi[i])
        elif k in (glfw.KEY_SPACE, glfw.KEY_ENTER):
            cmd[:] = [0.0, 0.0, 0.0]
        elif k == glfw.KEY_BACKSPACE:
            cmd[:] = [0.0, 0.0, 0.0]
            want_reset[0] = True

    reset()
    show()
    with mujoco.viewer.launch_passive(m, d, key_callback=key) as v:
        v.cam.type, v.cam.distance = mujoco.mjtCamera.mjCAMERA_TRACKING, 3.0
        v.cam.trackbodyid = int(m.jnt_bodyid[0])  # the floating base
        start, tick = time.perf_counter(), 0
        while v.is_running():
            act, down = step(list(cmd))
            if want_reset[0] or down:
                reset()
                want_reset[0] = False
            show()
            v.set_texts((None, None, "up/down vx | a/d vy | left/right yaw | space stop | backspace reset",
                         f"cmd vx {cmd[0]:+.1f} vy {cmd[1]:+.1f} yaw {cmd[2]:+.1f}   "
                         f"act vx {act[0]:+.2f} vy {act[1]:+.2f} yaw {act[2]:+.2f}"))  # fmt: skip
            v.sync()
            tick += 1
            time.sleep(max(0.0, start + tick * CONTROL_DT - time.perf_counter()))  # wall-clock pacing


run_script() if args.script else run_viewer()
