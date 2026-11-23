"""Edit the retargeted BONES-SEED T1 clips (soma CSVs, see utils/soma_csv.py) into a new folder; the source stays as is.

1. Arms: each arm's Elbow_Pitch, Elbow_Yaw, Wrist_Pitch, Wrist_Yaw and Hand_Roll held at the T1's default pose
   (DEFAULT_POSE below: elbow yaw -0.5 / +0.5 rad for left / right, the rest 0). Elbow_Pitch is the upper arm's
   twist, Elbow_Yaw the bend.
2. Head: pitch 0, yaw toward the direction of travel. The trunk's horizontal velocity averaged over the next WINDOW
   (about one gait cycle, so the step-to-step sway cancels; looking ahead makes the head lead turns), in the trunk's
   heading frame (vx forward, vy left), gives yaw = atan2(vy, vx), clipped to the retarget's head-yaw range and
   scaled by a ramp in vx: 0 up to VX_MIN (backward and sideways walking and shuffling in place keep the head
   straight), 1 from VX_FULL. A centered SMOOTH average then makes the head turn at a head's pace.
3. Feet: both feet turned in by the clip's toe-out, so that they point parallel. Toe-out is the mean over the two feet
   of each foot's median yaw against the pelvis (Waist) heading while planted during straight forward walking,
   measured at 30 Hz: the foot slower than 0.15 m/s and within 3 cm of its lowest while walking; the pelvis faster
   than 0.4 m/s and turning slower than 20 deg/s (both averaged over 1 s); travel within 30 deg of the trunk's
   heading. Only the mean: where the pelvis faces off the travel direction both feet read turned the same way, which
   is not toe-out, and the feet keep that shared direction. A clip with under 1.5 s of that walking takes its actor's
   median (the actor tag in the file name); an actor with none of that (some only walk
   arcs here) is measured with turns up to 45 deg/s, where the two feet's mean cancels the turn's inner/outer-foot
   difference, and an actor without any walking (falls, stand-ups) keeps its feet. The turn is constant per clip, so
   planted feet stay planted and turning steps keep their shape. Every frame, each foot's target is its pose rotated
   about the vertical through its origin (the ankle), and its leg's six joints are re-solved for it (damped
   Gauss-Newton on MuJoCo Jacobians); the pelvis and everything above stay. Where the full turn is out of the leg's
   reach (an ankle the retarget already holds at its limit, or a straight knee where only a bent one reaches: the
   knee stays within KNEE_STAY of the source's, see straighten) the frame takes the largest of 3/4, 1/2, 1/4 and 0
   of the turn that is: foot within 2 mm and 1 deg of heading, tilt within 5 deg. The turn then eases down to that
   fraction and back up at TURN_RATE rather than jumping in a frame (a planted foot pivots on its ankle).

Frame, root, waist, shoulder and other untouched columns keep the source's text; edited joints are written as float32
degrees. <dst>/edits.json records each clip's toe-out and IK result.

usage (repo root): uv run python utils/edit_t1_motions.py [--src DIR] [--dst DIR] [--jobs N]
"""

import argparse
import collections
import functools
import json
import multiprocessing as mp
import os
import re
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils import soma_csv  # noqa: E402
from utils.rotations import mat_to_quat  # noqa: E402

MJCF_PATH = os.path.join(ROOT, "assets", "t1", "t1_description", "mjcf", "t1.xml")
# the T1's default pose as the old robot/t1.py's T1_CFG gave it, in exact Python floats (robot/t1.py's
# DEFAULT_JOINT_POS holds t1_actuators.npz's float32 values, and importing it would import mjlab); keys are regexes
DEFAULT_POSE = {
    ".*_Shoulder_Pitch": 0.2,
    "Left_Shoulder_Roll": -1.3,
    "Right_Shoulder_Roll": 1.3,
    "Left_Elbow_Yaw": -0.5,
    "Right_Elbow_Yaw": 0.5,
    ".*_Hip_Pitch": -0.2,
    ".*_Knee_Pitch": 0.4,
    ".*_Ankle_Pitch": -0.2,
}
M = mujoco.MjModel.from_xml_path(MJCF_PATH)
FPS = soma_csv.FPS
ARMS = [f"{s}_{j}" for s in ("Left", "Right") for j in ("Elbow_Pitch", "Elbow_Yaw", "Wrist_Pitch", "Wrist_Yaw", "Hand_Roll")]
LEGS = [[f"{s}_{j}" for j in ("Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll")]
        for s in ("Left", "Right")]
FEET = (M.body("left_foot_link").id, M.body("right_foot_link").id)
PELVIS = M.body("Waist").id
YAW_MAX = 1.0472  # the retarget's head-yaw range: soma's URDF limit, inside our MJCF's 1.309 (utils/make_booster_t1.py)
WINDOW, SMOOTH = 1.0, 0.4  # [s]
VX_MIN, VX_FULL = 0.1, 0.3  # [m/s]
EVERY = 4  # toe-out is measured at 30 Hz
IK_W = np.array([1, 1, 1, 0.1, 0.1, 1.0])  # IK weights: position, tilt (world x, y), yaw
GRID = (1.0, 0.75, 0.5, 0.25, 0.0)  # fractions of the feet's turn tried where the full turn is out of reach
TURN_RATE = 120.0  # [deg/s] fastest a partial turn changes, i.e. a planted foot pivots on its ankle
KNEE_STAY = np.radians(3)  # an IK solution's knee within this of the source's: the same side of straight
adr = lambda n: M.jnt_qposadr[M.joint(n).id]  # noqa: E731


def nominal(name: str) -> float:
    return next((v for k, v in DEFAULT_POSE.items() if re.fullmatch(k, name)), 0.0)


def load(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSV -> MuJoCo qpos (T, nq), root position (T, 3), root rotation (T, 3, 3); at the CSV rate."""
    names, pos, R, q = soma_csv.load(path)
    qpos = np.zeros((len(pos), M.nq))
    qpos[:, :3], qpos[:, 3:7] = pos.numpy(), mat_to_quat(R).numpy()  # free joint: xyz, wxyz
    for i, n in enumerate(names):
        qpos[:, adr(n)] = q[:, i].numpy()
    return qpos, pos.numpy(), R.numpy()


def centered(x: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average over k (odd) samples, edges held."""
    return np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, "valid")


def travel(pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Root position and rotation at the CSV rate -> (T, 3): vx, vy [m/s] (next-WINDOW average, trunk heading frame)
    and the head yaw toward them [rad]."""
    k = int(WINDOW * FPS)
    v = np.pad(np.gradient(pos[:, :2], axis=0) * FPS, ((0, k - 1), (0, 0)), mode="edge")
    v = np.stack([np.convolve(v[:, i], np.ones(k) / k, "valid") for i in range(2)], 1)  # v[t] = mean over [t, t + k)
    h = np.arctan2(R[:, 1, 0], R[:, 0, 0])  # trunk heading
    vx, vy = np.cos(h) * v[:, 0] + np.sin(h) * v[:, 1], -np.sin(h) * v[:, 0] + np.cos(h) * v[:, 1]
    gate = np.clip((vx - VX_MIN) / (VX_FULL - VX_MIN), 0, 1)
    head = centered(np.clip(np.arctan2(vy, vx), -YAW_MAX, YAW_MAX) * gate, int(SMOOTH * FPS) | 1)
    return np.c_[vx, vy, head]


def toe_out(path: str, max_rate: float = 20.0) -> dict:
    """-> {clip, actor, straight_s: seconds of straight forward walking (turning slower than max_rate [deg/s]), L, R:
    toe-out [deg] or None}."""
    qpos, pos, R = load(path)
    qpos, tr, fps = qpos[::EVERY], travel(pos, R)[::EVERY], FPS / EVERY
    d = mujoco.MjData(M)
    xp, xm = [], []
    for qp in qpos:
        d.qpos[:] = qp
        mujoco.mj_kinematics(M, d)
        xp.append(d.xpos.copy()), xm.append(d.xmat.copy())
    xp, xm = np.array(xp), np.array(xm)
    yaw = lambda b: np.arctan2(xm[:, b, 3], xm[:, b, 0])  # noqa: E731  heading of body b's x axis
    h = np.unwrap(yaw(PELVIS))
    k = int(fps) | 1
    speed = centered(np.hypot(*np.gradient(xp[:, PELVIS, :2], axis=0).T) * fps, k)
    rate = np.degrees(np.abs(centered(np.gradient(h) * fps, k)))
    straight = (speed > 0.4) & (rate < max_rate) & (np.abs(np.degrees(np.arctan2(tr[:, 1], tr[:, 0]))) < 30)
    out = {"clip": path, "actor": re.search(r"__(A\d+)", path).group(1), "straight_s": float(straight.sum() / fps)}
    for side, b, sign in zip("LR", FEET, (1, -1)):
        low = xp[straight, b, 2].min(initial=np.inf)  # while walking: a clip may also lie on the floor
        planted = (np.hypot(*np.gradient(xp[:, b, :2], axis=0).T) * fps < 0.15) & (xp[:, b, 2] < low + 0.03)
        toe = sign * np.degrees((yaw(b) - h + np.pi) % (2 * np.pi) - np.pi)  # + = toes out
        sel = planted & straight
        out[side] = float(np.median(toe[sel])) if sel.any() else None
    return out


def solve_leg(d: mujoco.MjData, b: int, joints: list[int], q0: np.ndarray, p_t, R_t) -> tuple[np.ndarray, np.ndarray]:
    """Leg joints from q0 (d.qpos holds the rest of the pose) putting foot body b at p_t, R_t -> joints, final error
    (position [m], world rotation vector [rad]: x, y tilt, z yaw). Tilt weighs 0.1, so where a joint limit makes the
    target unreachable the foot gives up a little tilt rather than its place or heading."""
    qadr, dadr = M.jnt_qposadr[joints], M.jnt_dofadr[joints]
    lo, hi = M.jnt_range[joints].T
    jp, jr = np.zeros((3, M.nv)), np.zeros((3, M.nv))
    d.qpos[qadr] = q0
    for _ in range(60):
        mujoco.mj_kinematics(M, d)
        mujoco.mj_comPos(M, d)
        E = R_t @ d.xmat[b].reshape(3, 3).T
        err = np.r_[p_t - d.xpos[b], 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])]
        if np.abs(err).max() < 1e-7:
            break
        mujoco.mj_jacBody(M, d, jp, jr, b)
        J = np.vstack([jp[:, dadr], jr[:, dadr]]) * IK_W[:, None]
        # damped, with steps capped at 0.1 rad: a near-straight knee makes J nearly singular
        step = np.linalg.solve(J.T @ J + 1e-4 * np.eye(6), J.T @ (IK_W * err))
        d.qpos[qadr] = np.clip(d.qpos[qadr] + step * min(1.0, 0.1 / np.abs(step).max()), lo, hi)
    return d.qpos[qadr].copy(), err


def straighten(qpos: np.ndarray, toe_deg: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Turn each foot in by its toe-out -> qpos, per frame worst over both legs: (position error [m], heading error
    [rad], tilt error [rad], fraction of the turn applied)."""
    d, out = mujoco.MjData(M), qpos.copy()
    info = np.zeros((len(qpos), 4))
    info[:, 3] = 1
    for b, names, sign, toe in zip(FEET, LEGS, (1, -1), toe_deg):
        joints = [M.joint(n).id for n in names]
        qadr = M.jnt_qposadr[joints]
        lo, hi = M.jnt_range[joints].T

        def turn(f, fractions):
            """Frame f's leg at the first of fractions of the turn within reach (else the last) -> q, error, fraction.
            Within reach: within tolerance and the knee within KNEE_STAY of the source's. Near a straight knee the leg
            has two solutions, knee either side of ~3.4 deg; keeping to the source's side keeps the leg as continuous
            as the source, where a swap jumps the knee ~7 deg in a frame, so a turn only the other side reaches is cut
            (and eased) instead."""
            d.qpos[:] = qpos[f]
            mujoco.mj_kinematics(M, d)
            p_t, R0 = d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()
            for s in fractions:
                a = s * sign * np.radians(toe)  # the foot's yaw away from the pelvis; rotate it back by -a
                R_t = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]]) @ R0
                guess = qpos[f, qadr].copy()
                guess[2] -= a  # hip yaw: nearly the answer
                # starts: the last frame's correction, then the guess
                for q0 in ([np.clip(qpos[f, qadr] + out[f - 1, qadr] - qpos[f - 1, qadr], lo, hi)] if f else []) + [guess]:
                    q, err = solve_leg(d, b, joints, q0, p_t, R_t)
                    if (np.linalg.norm(err[:3]) < 2e-3 and abs(err[5]) < np.radians(1)
                            and np.linalg.norm(err[3:5]) < np.radians(5) and abs(q[3] - qpos[f, qadr[3]]) < KNEE_STAY):
                        return q, err, s
            return q, err, s  # ponytail: not reached, fraction 0 from the guess is the source pose itself

        errs, done = np.zeros((len(qpos), 6)), np.zeros(len(qpos))
        for f in range(len(qpos)):
            out[f, qadr], errs[f], done[f] = turn(f, GRID)
        # a partial turn eases in and out at TURN_RATE instead of jumping in one frame; where the eased fraction is out
        # of reach too, the ease is redone around what that frame reached
        r = TURN_RATE / max(abs(toe), 1e-6) / FPS
        s = ease(done, r)
        while (s < done).any():
            for f in np.flatnonzero(s < done):
                out[f, qadr], errs[f], done[f] = turn(f, [s[f]] + [g for g in GRID if g < s[f]])
            s = ease(np.minimum(s, done), r)
        info[:, :3] = np.maximum(info[:, :3], np.c_[np.linalg.norm(errs[:, :3], axis=1), np.abs(errs[:, 5]),
                                                    np.linalg.norm(errs[:, 3:5], axis=1)])
        info[:, 3] = np.minimum(info[:, 3], done)
    return out, info


def ease(x: np.ndarray, r: float) -> np.ndarray:
    """The largest sequence <= x that changes by at most r per step."""
    x = x.copy()
    for f in range(1, len(x)):
        x[f] = min(x[f], x[f - 1] + r)
    for f in range(len(x) - 2, -1, -1):
        x[f] = min(x[f], x[f + 1] + r)
    return x


def edit(job: tuple[str, str, tuple[float, float]]) -> dict:
    """(source CSV, output CSV, toe-out (left, right) [deg]) -> writes the output; -> its IK summary."""
    src, dst, toe = job
    qpos, pos, R = load(src)
    new = qpos.copy()
    for n in ARMS + ["Head_pitch"]:
        new[:, adr(n)] = nominal(n)
    new[:, adr("AAHead_yaw")] = travel(pos, R)[:, 2]
    new, info = straighten(new, toe)

    with open(src, newline="") as f:
        text = f.read()
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    trailing = lines[-1] == ""
    rows = [line.split(",") for line in lines[1 : len(lines) - trailing]]
    col = {n: i for i, n in enumerate(lines[0].split(","))}
    assert len(rows) == len(new), (src, len(rows), len(new))
    for n in ARMS + ["AAHead_yaw", "Head_pitch"] + [n for names, t in zip(LEGS, toe) if t for n in names]:
        for row, v in zip(rows, np.degrees(new[:, adr(n)]).astype(np.float32)):
            row[col[n]] = str(v)
    with open(dst, "w", newline="") as f:
        f.write(nl.join([lines[0]] + [",".join(r) for r in rows]) + (nl if trailing else ""))
    return {"clip": os.path.basename(src), "toe_deg": list(toe), "frames": len(new),
            "max_pos_mm": float(info[:, 0].max() * 1e3), "max_heading_deg": float(np.degrees(info[:, 1].max())),
            "max_tilt_deg": float(np.degrees(info[:, 2].max())), "frac_partial": float((info[:, 3] < 1).mean()),
            "min_fraction": float(info[:, 3].min())}


def _init():
    import torch

    torch.set_num_threads(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=soma_csv.DATASET)
    p.add_argument("--dst", default=soma_csv.DATASET + "_edited")
    p.add_argument("--jobs", type=int, default=max(1, os.cpu_count() - 2))
    args = p.parse_args()
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    assert src != dst
    clips = soma_csv.find(src)
    with mp.Pool(args.jobs, _init) as pool:
        toes = pool.map(toe_out, clips, chunksize=8)
        own = lambda t: t["straight_s"] > 1.5 and t["L"] is not None and t["R"] is not None  # noqa: E731
        by_actor = collections.defaultdict(list)
        for t in filter(own, toes):
            by_actor[t["actor"]].append((t["L"] + t["R"]) / 2)
        rest = [t["clip"] for t in toes if t["actor"] not in by_actor]
        by_actor_turning = collections.defaultdict(list)
        for t in filter(own, pool.map(functools.partial(toe_out, max_rate=45.0), rest, chunksize=8)):
            by_actor_turning[t["actor"]].append((t["L"] + t["R"]) / 2)
        jobs = []
        for t in toes:
            a = t["actor"]
            t["from"] = "clip" if own(t) else "actor" if by_actor[a] else "actor_turning" if by_actor_turning[a] else "none"
            m = float(np.median(by_actor[a] or by_actor_turning[a] or [0.0]))
            t["toe"] = ((t["L"] + t["R"]) / 2,) * 2 if own(t) else (m, m)
            out = os.path.join(dst, os.path.relpath(t["clip"], src))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            jobs.append((t["clip"], out, t["toe"]))
        print(f"{len(clips)} clips, toe-out from: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(
            t["from"] for t in toes).items()), flush=True)
        res = {r["clip"]: r for r in pool.imap_unordered(edit, jobs, chunksize=4)}
    for t in toes:
        res[os.path.basename(t["clip"])].update(toe_from=t["from"], straight_s=t["straight_s"], measured=(t["L"], t["R"]))
    params = {k: globals()[k] for k in ("YAW_MAX", "WINDOW", "SMOOTH", "VX_MIN", "VX_FULL", "EVERY", "GRID", "TURN_RATE", "KNEE_STAY")}
    with open(os.path.join(dst, "edits.json"), "w") as f:
        json.dump({"src": src, "params": params, "clips": [res[k] for k in sorted(res)]}, f, indent=1)
    r = list(res.values())
    frames = np.array([x["frames"] for x in r])
    print(f"wrote {len(r)} clips to {dst}")
    print(f"IK worst over all clips: {max(x['max_pos_mm'] for x in r):.2f} mm, heading "
          f"{max(x['max_heading_deg'] for x in r):.2f} deg, tilt {max(x['max_tilt_deg'] for x in r):.2f} deg; frames "
          f"with a partial turn {np.average([x['frac_partial'] for x in r], weights=frames) * 100:.2f}% "
          f"(in {sum(x['frac_partial'] > 0 for x in r)} clips)")


if __name__ == "__main__":
    main()
