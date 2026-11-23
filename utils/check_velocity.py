"""Stop, walking-speed and stance checks of a velocity policy in the training simulator on the CPU, under a set of
physics conditions. Every (condition, case) is one world of one process; 10 s per world.

usage (repo root, the GPU hidden, at most 4 threads, under a memory cap):
    CUDA_VISIBLE_DEVICES= systemd-run --user --scope -q -p MemoryMax=16G .venv/bin/python -m utils.check_velocity \\
        --ckpt logs/t2m_vel_s1/policy/ckpt_013000.pt [--conditions nominal,friction_hi] [--out checks.npz]

Cases (22 per condition): 10 stop cases (3 s at a command, 7 s at zero) and 12 speed cases (0.25 / 0.5 / 0.9 of the
robot's command box on vx+, vx-, vy+, yaw+, held 10 s, the speed averaged over 6-10 s).

The T2 robots only (the measures name the T2's joints). Conditions: nominal (the robot's evaluation delay) and
delay10 (10 ms on every group) for both. For the measured one (robot/t2_measured.py: joint friction in the model and
its randomization in xi) also the edges of that randomization: friction_lo / friction_hi
(joint friction x 0.5 / x 1.5), gains_lo / gains_hi (kp and kd x 0.9 / x 1.1), mass_lo / mass_hi (trunk -5 / +5 kg);
and two outside it: no_friction, booster_damping (the waist and ankle kd / 0.69: Booster's rule). Default: all the
robot supports.

Per stop case: falls; seconds from the stop until still for good (0.5 s blocks: joint speed RMS < 0.01 rad/s and root
planar speed < 0.02 m/s); joint speed RMS and the largest peak-to-peak of the six waist and ankle joints over the last
2 s; and the stance over the last 2 s: each foot's toe-out against the trunk heading, stance width, waist yaw, heading
change over the last 4 s, trunk pitch (+ = forward), and the torque the actuators apply at ankle pitch and ankle roll
(joint side). Per speed case: the mean velocity in the heading frame and reached / commanded on the commanded axis.
"""

import argparse
import math
import sys
import time

import numpy as np
import torch

torch.set_num_threads(4)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ckpt", required=True, help="a --vel_frac policy checkpoint")
parser.add_argument("--conditions", default=None, help="comma-separated (default: all the robot supports)")
parser.add_argument("--out", default=None, help="write the rollouts and the per-case numbers here (.npz)")
args = parser.parse_args()
if torch.cuda.is_available():
    sys.exit("check_velocity runs on the CPU: start it with CUDA_VISIBLE_DEVICES= (and a memory cap, see --help)")

from utils.rotations import mv3, yaw_mat  # noqa: E402
from utils.velocity_rig import Rig  # noqa: E402  (it starts Warp)

REST = [(0, 0, 0), (0.8, 0, 0), (0.4, 0, 0), (-0.6, 0, 0), (-0.3, 0, 0), (0, 0.4, 0), (0, -0.4, 0), (0, 0, 1.0),
        (0, 0, -1.0), (0.5, 0, 0.5)]  # fmt: skip
DT, T_CMD, T_ALL = 0.02, 3.0, 10.0
STEPS = round(T_ALL / DT)
LINKAGE = ("waist_pitch_joint", "waist_roll_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
           "right_ankle_pitch_joint", "right_ankle_roll_joint")  # fmt: skip
# condition -> (joint friction factor, kp and kd factor, waist and ankle kd factor, trunk mass added [kg], delay [steps] or None)
CONDITIONS = {
    "nominal": (1.0, 1.0, 1.0, 0.0, None), "delay10": (1.0, 1.0, 1.0, 0.0, 2),
    "friction_lo": (0.5, 1.0, 1.0, 0.0, None), "friction_hi": (1.5, 1.0, 1.0, 0.0, None),
    "gains_lo": (1.0, 0.9, 1.0, 0.0, None), "gains_hi": (1.0, 1.1, 1.0, 0.0, None),
    "mass_lo": (1.0, 1.0, 1.0, -5.0, None), "mass_hi": (1.0, 1.0, 1.0, 5.0, None),
    "no_friction": (0.0, 1.0, 1.0, 0.0, None), "booster_damping": (1.0, 1.0, 1 / 0.69, 0.0, None),
}  # fmt: skip
ANY_ROBOT = ("nominal", "delay10")

# the robot and its command box come with the checkpoint: one world first, to read them
ck_args = torch.load(args.ckpt, map_location="cpu", weights_only=False)["args"]
from robot.spec import get_robot  # noqa: E402

robot = get_robot(ck_args["robot"])
if not {*LINKAGE, "waist_yaw_joint"} <= set(robot.joint_names):  # before the simulator is built
    sys.exit(f"check_velocity measures the T2's joints (its waist and ankle linkages): robot {robot.name!r} is not a T2")
from src.dr import XiLayout  # noqa: E402

has_friction = "joint_friction" in XiLayout(robot, "cpu").slices  # a measured robot: friction in the model, factors in xi
conds = args.conditions.split(",") if args.conditions else [c for c in CONDITIONS if has_friction or c in ANY_ROBOT]
for c in conds:
    if c not in CONDITIONS:
        sys.exit(f"unknown condition {c!r}: one of {', '.join(CONDITIONS)}")
    if c not in ANY_ROBOT and not has_friction:
        sys.exit(f"condition {c!r} needs a measured robot's joint friction and randomization (robot/t2_measured.py), "
                 f"and robot {robot.name!r} has none: it supports {', '.join(ANY_ROBOT)}")  # fmt: skip

worlds = [(c, kind, k) for c in conds for kind, n in (("rest", len(REST)), ("track", 12)) for k in range(n)]
N = len(worlds)
rig = Rig(args.ckpt, N, dev="cpu")
env, sim, mdp = rig.env, rig.env.sim, rig.mdp
names = list(robot.joint_names)
J = len(names)
HI, LO = mdp.cmd_hi.tolist(), mdp.cmd_lo.tolist()
TRACK = [(f"{ax}{s} {f}", tuple((HI if s == "+" else LO)[i] * f if i == k else 0.0 for i in range(3)))
         for ax, s, k in (("vx", "+", 0), ("vx", "-", 0), ("vy", "+", 1), ("yaw", "+", 2)) for f in (0.25, 0.5, 0.9)]  # fmt: skip
case_cmd = lambda kind, k: REST[k] if kind == "rest" else TRACK[k][1]  # noqa: E731
print(f"{args.ckpt}: robot {robot.name}, iteration {rig.iteration}; {N} worlds = {len(conds)} conditions x 22 cases", flush=True)

# ---- the conditions, applied per world in the simulator and written to xi
col = lambda i: torch.tensor([CONDITIONS[c][i] for c, _, _ in worlds], dtype=torch.float32)  # noqa: E731
f_fric, f_gain, f_kdl, m_add = col(0), col(1), col(2), col(3)
L = env.layout.slices
if has_friction:
    from mjlab.managers.event_manager import _DERIVED_FIELDS, RecomputeLevel

    dof_adr = env.robot.indexing.joint_v_adr.reshape(-1)
    base = sim.get_default_field("dof_frictionloss").float()
    sim.expand_model_fields(tuple(sorted({"dof_frictionloss", "body_mass", *_DERIVED_FIELDS[RecomputeLevel.set_const]})))
    sim.model.dof_frictionloss[:] = base.unsqueeze(0) * f_fric.unsqueeze(1)
    trunk = int(env.robot.indexing.body_ids[env.robot.body_names.index(robot.com_body)])
    mass = sim.get_default_field("body_mass").float().repeat(N, 1)
    mass[:, trunk] += m_add
    sim.model.body_mass[:] = mass
    sim.recompute_constants(RecomputeLevel.set_const)
    for act in env.robot.actuators:  # in place: the actuators' gains are views into their fused group's tensors
        (n,) = (x.split("/")[-1] for x in act._target_names)
        act.stiffness[:, 0] *= f_gain
        act.damping[:, 0] *= f_gain * (f_kdl if n in LINKAGE else 1.0)
    for g, f in (("joint_friction", f_fric), ("kp", f_gain), ("kd", f_gain), ("trunk_mass", m_add)):
        if g in L:
            env.xi[:, L[g]] = f.unsqueeze(1)
    if "kd" in L:
        lk = [names.index(n) for n in LINKAGE]
        env.xi[:, L["kd"].start + torch.tensor(lk)] *= f_kdl.unsqueeze(1)
lag = [CONDITIONS[c][4] for c, _, _ in worlds]
rig.reset()
for w, d in enumerate(lag):  # after the reset: the delay lives in xi and is read every step
    if d is not None:
        env.xi[w, L["delay"]] = float(d)

# ---- the rollout
lk = [names.index(n) for n in LINKAGE]
ank = [names.index(f"{s}_ankle_{a}_joint") for a in ("pitch", "roll") for s in ("left", "right")]  # pitch L R, roll L R
waist_yaw = names.index("waist_yaw_joint")
q_adr = env.robot.indexing.joint_v_adr.reshape(-1)
cmd_at = lambda i: [list(map(float, case_cmd(kind, k))) if (kind == "track" or i * DT < T_CMD) else [0.0, 0.0, 0.0]  # noqa: E731
                    for _, kind, k in worlds]  # fmt: skip
P, V, Q, QD, F, YAW, PITCH, TOE, WIDTH, TAU = ([] for _ in range(10))
t0 = time.perf_counter()
for i in range(STEPS):
    rig.observe()
    rig.apply(rig.act(cmd_at(i)))
    S = env.state()
    R = S[1]
    Ry = yaw_mat(R)
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    bp, bR, _, _ = mdp.kin.forward(*S)
    fp, fR = bp[:, mdp.feet_idx], bR[:, mdp.feet_idx]  # (N, 2, ...): left, right
    fyaw = torch.atan2(fR[..., 1, 0], fR[..., 0, 0]) - yaw.unsqueeze(1)
    fyaw = torch.atan2(torch.sin(fyaw), torch.cos(fyaw)) * torch.tensor([1.0, -1.0])  # toe-out: + = outward, both feet
    v = mv3(Ry.transpose(-1, -2), S[2])
    P.append(S[0].numpy().copy()), V.append(torch.stack((v[:, 0], v[:, 1], S[3][:, 2]), 1).numpy().copy())
    Q.append(S[4].numpy().copy()), QD.append(S[5].numpy().copy()), F.append(mdp.fall_termination(S).numpy().copy())
    YAW.append(yaw.numpy().copy()), PITCH.append(torch.asin((-R[:, 2, 0]).clamp(-1, 1)).numpy().copy())
    TOE.append(fyaw.numpy().copy())
    WIDTH.append(mv3(Ry.transpose(-1, -2), fp[:, 0] - fp[:, 1])[:, 1].abs().numpy().copy())
    TAU.append(sim.data.qfrc_actuator[:, q_adr][:, ank].numpy().copy())
    if (i + 1) % 100 == 0:
        print(f"  step {i + 1}/{STEPS}: {time.perf_counter() - t0:.0f} s", flush=True)
P, V, Q, QD, F, YAW, PITCH, TOE, WIDTH, TAU = map(np.array, (P, V, Q, QD, F, YAW, PITCH, TOE, WIDTH, TAU))

# ---- per case, per condition
n_end, n6 = round(8.0 / DT), round(6.0 / DT)
wrap = lambda a: math.atan2(math.sin(a), math.cos(a))  # noqa: E731
rows = {}
for ci, c in enumerate(conds):
    ids = [w for w, (cc, _, _) in enumerate(worlds) if cc == c]
    rest, track = ids[: len(REST)], ids[len(REST) :]
    fr, kg, kdl, ma, dl = CONDITIONS[c]
    print(f"\n### {c} (joint friction x{fr:g}, kp and kd x{kg:g}, waist and ankle kd x{kdl:.3g}, trunk {ma:+g} kg, "
          f"delay {'the evaluation one' if dl is None else f'{dl * 5} ms'})")  # fmt: skip
    print("  stop: first 3 s cmd | still after s | joint speed RMS | linkage p-p deg (worst joint) | toe-out L / R deg | "
          "width mm | waist yaw deg | heading drift deg | trunk pitch deg | ankle torque N m: pitch L / R, roll L / R | fall")
    stills = []
    for w in rest:
        qd = np.sqrt((QD[:, w] ** 2).mean(1))
        vr = np.linalg.norm(np.gradient(P[:, w, :2], axis=0), axis=1) / DT
        ok = [(qd[b : b + 25].max() < 0.01 or np.sqrt((qd[b : b + 25] ** 2).mean()) < 0.01) and vr[b : b + 25].max() < 0.02
              for b in range(round(T_CMD / DT), STEPS - 24, 25)]  # fmt: skip
        still = next((0.5 * j for j in range(len(ok)) if all(ok[j:])), float("nan"))
        stills.append(still)
        ptp = np.degrees(np.ptp(Q[n_end:, w][:, lk], 0))
        toe, tau = np.degrees(TOE[n_end:, w].mean(0)), TAU[n_end:, w].mean(0)
        r = dict(still=still, qd_rms=float(np.sqrt((qd[n_end:] ** 2).mean())), ptp=float(ptp.max()), toe_l=float(toe[0]),
                 toe_r=float(toe[1]), width=float(1000 * WIDTH[n_end:, w].mean()),
                 waist_yaw=float(np.degrees(Q[n_end:, w, waist_yaw].mean())), drift=float(np.degrees(wrap(YAW[-1, w] - YAW[n6, w]))),
                 pitch=float(np.degrees(PITCH[n_end:, w].mean())), tau=[float(x) for x in tau], fall=int(F[:, w].any()))  # fmt: skip
        rows[(c, "rest", worlds[w][2])] = r
        cm = case_cmd("rest", worlds[w][2])
        print(f"  {cm[0]:+.1f} {cm[1]:+.1f} {cm[2]:+.1f} | {still:4.1f} | {r['qd_rms']:.4f} | {r['ptp']:5.2f} "
              f"({LINKAGE[int(ptp.argmax())]}) | {r['toe_l']:+5.1f} / {r['toe_r']:+5.1f} | {r['width']:5.1f} | "
              f"{r['waist_yaw']:+5.1f} | {r['drift']:+5.1f} | {r['pitch']:+5.1f} | {tau[0]:+6.1f} / {tau[1]:+6.1f}, "
              f"{tau[2]:+5.1f} / {tau[3]:+5.1f} | {r['fall']}")  # fmt: skip
    ratios = []
    print("  speed: command -> mean 6-10 s (vx, vy, yaw) | reached/commanded | fall")
    for w in track:
        lab, cm = TRACK[worlds[w][2]]
        a = V[n6:, w].mean(0)
        k_ = int(np.argmax(np.abs(cm)))
        ratios.append(float(a[k_] / cm[k_]))
        rows[(c, "track", worlds[w][2])] = dict(v=[float(x) for x in a], ratio=ratios[-1], fall=int(F[:, w].any()))
        print(f"  {lab:10s} {cm[0]:+.2f} {cm[1]:+.2f} {cm[2]:+.2f} -> {a[0]:+.2f} {a[1]:+.2f} {a[2]:+.2f} | {ratios[-1]:.2f} | "
              f"{int(F[:, w].any())}")  # fmt: skip
    never = int(np.isnan(stills).sum())
    worst = float(np.nanmax(stills)) if never < len(stills) else float("nan")
    print(f"  SUMMARY {c}: falls {int(F[:, ids].any(0).sum())}/{len(ids)}; still after stop: worst {worst:.1f} s, never {never}; "
          f"mean reached/commanded {np.mean(ratios):.2f}; vy+ 0.25 {ratios[6]:.2f}")  # fmt: skip
if args.out:
    np.savez(args.out, P=P, V=V, Q=Q, QD=QD, F=F, YAW=YAW, PITCH=PITCH, TOE=TOE, WIDTH=WIDTH, TAU=TAU,
             worlds=np.array([f"{c}|{kind}|{k}" for c, kind, k in worlds]), rows=np.array([repr(rows)]), ckpt=args.ckpt)  # fmt: skip
    print(f"\nwrote {args.out}")
