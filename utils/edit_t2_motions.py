"""Edit the retargeted BONES-SEED T2 clips (soma CSVs: Frame, root cm and xyz Euler deg, the 31 T2 joints in deg, 120
fps, read with utils/soma_csv.py) into a new folder; the source stays as is. The T1 v2 editor
(git show 70cdd1a:utils/edit_t1_motions.py) on the T2.

1. Arms: each arm's elbow_pitch, elbow_yaw, wrist_pitch, wrist_yaw and wrist_roll joints held at robot/t2.py's default
   pose (elbow yaw -0.5 / +0.5 rad for left / right, the rest 0). Shoulder roll: both arms turned in by one constant per
   actor (the actor tag in the file name), so that the actor's median roll magnitude over its straight forward walking
   is the default pose's 1.3 rad (a larger magnitude holds the arm closer to the trunk, on the T2 as on the T1). An actor
   already at or inside 1.3 keeps its arms, as does one without that walking; the result is clipped to the joint's
   range. Shoulder pitch and the swing stay the source's. Straight forward walking, measured at 30 Hz: the trunk (the
   CSV root) faster than 0.4 m/s and turning slower than 20 deg/s (both averaged over 1 s), and travel within 30 deg of
   its heading (see travel). The T1 editor timed the first two on the pelvis (Waist) because its toe-out measured the
   feet against the pelvis; for the shoulders the trunk, which they hang from, is the better gauge of turning (the feet,
   3, keep the T1's pelvis timing): the pelvis (waist_yaw_link) yaws with each stride, -10 to +19 deg about the trunk in
   a straight 2.3 m/s walk, and the 1 s average does not cancel that at every cadence (that walk reads up to 29 deg/s on
   the pelvis and 8 on the trunk away from the clip's ends, so only 0.9 of its 2.5 s counted). An actor's median is over
   its clips with more than 1.5 s of that walking, of each clip's median.
2. Head: pitch 0, yaw toward the direction of travel. The trunk's horizontal velocity averaged over the next WINDOW
   (about one gait cycle, so the step-to-step sway cancels; looking ahead makes the head lead turns), in the trunk's
   heading frame (vx forward, vy left), gives yaw = atan2(vy, vx), clipped to the reward's soft limit (0.9 of the T2
   MJCF's +/-1.0996 rad; the T1 editor's clip was inside its soft limit too) and scaled by a ramp in vx: 0 up to
   VX_MIN (backward and sideways walking and shuffling in place keep the head straight), 1 from VX_FULL. A centered
   SMOOTH average then makes the head turn at a head's pace. The gates are the T1's 0.1 / 0.3 m/s times 1.415, the
   median T2/T1 root-speed ratio of the same clips (1.36-1.43 over 40 pilot clips): at the T1's values the head turned
   up to 28 deg in sideways walks the T1 edit kept straight. WALK_V stays the T1's 0.4 m/s (0.28 at T1 speed): scaled,
   it moved the pilot's shifts by 0.004 rad or less but dropped the slowest walkers (A444, A445) from the measurement.
   The shoulder-roll target stays 1.3 rad although the T2's soft limit is 1.3666 (T1: 1.455): about 16% of shifted
   arm samples pass it, by 0.04 rad on average (2026-11-19 review; lowering the target to 1.21 would keep the T1's
   margin but splay the hands 3.5 cm per side, which this edit exists to remove).
3. Feet: both feet turned in by the clip's toe-out, so that they point parallel: the T1 editor's feet step
   (utils/edit_t1_motions.py, point 3) with its constants. The user asked for it on 2026-11-19 because the T2 clips toe
   out about 10-14 deg against the base heading, where the T1's feet-edited clips (which the deployed T1 policy trained
   on) toe out about 0-6, and a T2 policy trained without it copies the toe-out (about 15 deg walking forward). Toe-out
   is the mean over the two feet of each foot's median yaw against the pelvis (waist_yaw_link, which the legs hang from)
   heading while planted during straight forward walking, measured at 30 Hz: the foot (left/right_ankle_roll_link)
   slower than 0.15 m/s and within 3 cm of its lowest while walking; the pelvis faster than WALK_V and turning slower
   than WALK_RATE (both averaged over 1 s); travel within WALK_TRAVEL of the trunk's heading. Only the mean: where the
   pelvis faces off the travel direction both feet read turned the same way, which is not toe-out, and the feet keep
   that shared direction. A clip and its mirror (the same name with _M) are one motion and take one toe-out: the mean
   over the two of the own toe-out of each with more than WALK_S of that walking and both feet planted in it. The port
   departs from the T1 editor, which decided each file alone: in 130 of the 5,944 mirrored T2 pairs one side fell short
   of WALK_S (a median 0.95 s of that walking; 20 also never planted a foot in it) and took its actor's median, up to
   20 deg from its mirror's turn (diagonal and sideways walks whose travel sits near the
   WALK_TRAVEL gate), although its own short reading lay nearer its mirror's value (median 1.3 deg off, against 4.4
   from the actor's; nearer in 93 of the 110 pairs where it read both feet). A motion with neither takes its actor's
   median over the clips with their own; an actor with none is measured with turns up to 45 deg/s, where the two feet's
   mean cancels the turn's inner/outer-foot difference, and an actor without any walking (falls, stand-ups) keeps its
   feet. The turn is constant per clip, so planted feet stay planted and turning steps keep their shape. Every frame,
   each foot's target is its pose rotated about the vertical through its origin (the ankle), and its leg's six joints
   are re-solved for it (damped Gauss-Newton on MuJoCo Jacobians); the pelvis and everything above stay. Where the full
   turn is out of the leg's reach (an ankle the retarget already holds at its limit, ankle roll +/-18 deg on the T2
   against the T1's +/-25, or a straight knee where only a bent one reaches: the knee stays within KNEE_STAY of the
   source's, see straighten) the frame takes the largest of 3/4, 1/2, 1/4 and 0 of the turn that is: foot within 2 mm
   and 1 deg of heading, tilt within 5 deg. The turn then eases down to that fraction and back up at TURN_RATE rather
   than jumping in a frame (a planted foot pivots on its ankle). Over the 11,950 source clips 4,006 take their own
   toe-out (with their mirror's), 130 their mirror's, 7,708 their actor's, 77 their actor's turning measurement and 29
   (12 actors) none; the turn is 9.4 deg median (10-90th percentile 3.4-18.9, extremes -12.3 and +31.5). On 186 pilot
   clips with those turns, toe-out against the trunk heading (L / R medians) went from +16.5 / +14.4 to +4.6 / +2.3 deg
   standing, +12.3 / +7.5 to +3.7 / -0.9 walking forward and +15.2 / +8.5 to +3.8 / -2.6 walking backward; 0.93% of
   frames took a partial turn.
4. Trunk (--untwist, off by default; the user asked for it on 2026-11-19): the retarget turns the trunk (the CSV root,
   whose frame the velocity command and the heading are measured in) by one constant about its own axes in every clip,
   2.14 deg to the right and 0.45 deg leaning left, against the pelvis, legs and travel line. Measured on the raw T2
   retarget: with S the left-right reflection and D the trunk-to-pelvis rotation, D_mirror (S D S)^-1 is the same
   rotation in every frame of a clip and its mirror, half rotation vector (+0.44, 0.00, +2.14) deg (10-90th percentile
   +0.43..+0.46 and +2.12..+2.16 over 1,500 pairs, with the waist upright and bent alike; the same product taken on the
   pelvis' side moves with waist pitch), so it is the retarget's and not the motion's. In straight forward walking the
   trunk heads 2.12 deg right of the travel line in a clip and its mirror alike, the pelvis +0.01, and the feet toe out
   the same against that line (+10.1 / +10.0). The T1 library carries the same twist. The step turns the trunk by
   UNTWIST about its own axes and re-solves the three waist joints and the root position, so that the pelvis, legs and
   feet keep their place in the world; arms and head ride the trunk. A waist joint that would pass its range is
   clipped and the trunk keeps that much of the constant (the pelvis still stays). It runs first in a clip's edit, so
   the head yaw (2) is aimed in the corrected frame; the measurements of 1 and 3 stay on the source clips, so a clip
   takes the same shoulder shift and toe-out with and without it. A constant in pitch cannot be told from posture by
   mirroring and is left, as is the pelvis' own 0.15 deg sideways lean (the legs hang from it).

Frame, root, waist, shoulder pitch and the other untouched columns keep the source's text; edited joints are written as
float32 degrees (the 12 leg columns in clips with a nonzero toe-out; with --untwist also the three waist columns, and
the six root columns as float64 text). --no-feet leaves out 3 and writes the same clips, byte for byte, as the editor
did before 3 was added; without --untwist the clips are those written before 4 was added. <dst>/edits.json records
each actor's shoulder-roll measurement and shift and each clip's shift, head-yaw statistics, toe-out (toe_from: clip,
mirror, actor, actor_turning or none), IK result and, with --untwist, the fraction of frames with a clipped waist.

usage (repo root): CUDA_VISIBLE_DEVICES= python -m utils.edit_t2_motions --src DIR --dst DIR [--jobs N] [--no-feet]
                   [--untwist]
"""

import argparse
import collections
import functools
import json
import multiprocessing as mp
import os
import re

import mujoco
import numpy as np
import torch

from robot.t2 import DEFAULT_JOINT_POS, MJCF_PATH
from utils import soma_csv
from utils.rotations import mat_to_quat, so3_exp

M = mujoco.MjModel.from_xml_path(MJCF_PATH)
FPS = soma_csv.FPS
ARMS = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_yaw", "wrist_roll")]
ROLLS = ("left_shoulder_roll_joint", "right_shoulder_roll_joint")  # the left's is negative
HEAD_YAW, HEAD_PITCH = "aa_head_yaw_joint", "head_pitch_joint"
LEGS = [[f"{s}_{j}_joint" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")]
        for s in ("left", "right")]
FEET = (M.body("left_ankle_roll_link").id, M.body("right_ankle_roll_link").id)
PELVIS = M.body("waist_yaw_link").id  # the legs hang from it, as the T1's from Waist
ROLL_TARGET = (DEFAULT_JOINT_POS[ROLLS[1]] - DEFAULT_JOINT_POS[ROLLS[0]]) / 2  # 1.3 rad
YAW_RANGE = tuple(0.9 * M.jnt_range[M.joint(HEAD_YAW).id])  # +/-0.9896 rad, the reward's soft limit
WINDOW, SMOOTH = 1.0, 0.4  # [s]
VX_MIN, VX_FULL = 0.1 * 1.415, 0.3 * 1.415  # [m/s], the T1's gates at T2 speed
EVERY = 4  # straight walking and toe-out are measured at 30 Hz
WALK_V, WALK_RATE, WALK_TRAVEL, WALK_S = 0.4, 20.0, 30.0, 1.5  # [m/s], [deg/s], [deg], [s]
IK_W = np.array([1, 1, 1, 0.1, 0.1, 1.0])  # IK weights: position, tilt (world x, y), yaw
GRID = (1.0, 0.75, 0.5, 0.25, 0.0)  # fractions of the feet's turn tried where the full turn is out of reach
TURN_RATE = 120.0  # [deg/s] fastest a partial turn changes, i.e. a planted foot pivots on its ankle
KNEE_STAY = np.radians(3)  # an IK solution's knee within this of the source's: the same side of straight
UNTWIST = np.radians([0.445, 0.0, 2.14])  # rotation vector in the trunk's frame taking out the retarget's constant
WAIST = [f"waist_{a}_joint" for a in ("pitch", "roll", "yaw")]  # trunk -> pelvis in this order: about y, x, z
WAIST_POS = M.body_pos[[M.body(n.replace("joint", "link")).id for n in WAIST]]  # each link's origin in its parent
adr = lambda n: M.jnt_qposadr[M.joint(n).id]  # noqa: E731


def load(path: str) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """CSV -> joint angles {name: (T,) [rad]}, root position (T, 3) [m], root rotation (T, 3, 3); at the CSV rate."""
    names, pos, R, q = soma_csv.load(path)
    return dict(zip(names, q.numpy().T)), pos.numpy(), R.numpy()


def qpos_of(q: dict[str, np.ndarray], pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    """load()'s joints and root -> MuJoCo qpos (T, nq)."""
    qpos = np.zeros((len(pos), M.nq))
    qpos[:, :3], qpos[:, 3:7] = pos, mat_to_quat(torch.from_numpy(R)).numpy()  # free joint: xyz, wxyz
    for n, x in q.items():
        qpos[:, adr(n)] = x
    return qpos


def rot(axis: int, a: np.ndarray) -> np.ndarray:
    """Rotations (T, 3, 3) by the angles a (T,) [rad] about coordinate axis 0, 1 or 2."""
    R, (i, j) = np.zeros((len(a), 3, 3)), ((axis + 1) % 3, (axis + 2) % 3)
    R[:, axis, axis] = 1
    R[:, i, i] = R[:, j, j] = np.cos(a)
    R[:, j, i] = np.sin(a)
    R[:, i, j] = -R[:, j, i]
    return R


def euler_xyz(R: np.ndarray) -> np.ndarray:
    """Root rotations (T, 3, 3) -> the CSV's extrinsic xyz Euler angles (T, 3) [deg]: R = Rz Ry Rx."""
    return np.degrees(np.c_[np.arctan2(R[:, 2, 1], R[:, 2, 2]), -np.arcsin(np.clip(R[:, 2, 0], -1, 1)),
                            np.arctan2(R[:, 1, 0], R[:, 0, 0])])


def untwist(q: dict[str, np.ndarray], pos: np.ndarray, R: np.ndarray):
    """Point 4 on load()'s clip -> waist joints {name: (T,) [rad]}, root position (T, 3), root rotation (T, 3, 3),
    (T,) bool: a waist joint clipped at its range (there the trunk turns short of UNTWIST)."""
    mv = lambda A, v: np.einsum("tij,tj->ti", A, v)  # noqa: E731
    chain = lambda w: rot(1, w[0]) @ rot(0, w[1]) @ rot(2, w[2])  # noqa: E731  trunk -> pelvis rotation
    # the pelvis origin in the trunk's frame
    origin = lambda w: WAIST_POS[0] + mv(rot(1, w[0]), WAIST_POS[1] + rot(0, w[1]) @ WAIST_POS[2])  # noqa: E731
    w0 = [q[n] for n in WAIST]
    pelvis_R, pelvis_p = R @ chain(w0), pos + mv(R, origin(w0))
    N = so3_exp(torch.from_numpy(UNTWIST)).numpy().T @ chain(w0)  # = Ry(pitch) Rx(roll) Rz(yaw) of the turned trunk
    w = np.c_[np.arctan2(N[:, 0, 2], N[:, 2, 2]), -np.arcsin(np.clip(N[:, 1, 2], -1, 1)),
              np.arctan2(N[:, 1, 0], N[:, 1, 1])]
    wc = np.clip(w, *M.jnt_range[[M.joint(n).id for n in WAIST]].T)
    R = pelvis_R @ chain(wc.T).transpose(0, 2, 1)
    return dict(zip(WAIST, wc.T)), pelvis_p - mv(R, origin(wc.T)), R, (wc != w).any(1)


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
    head = centered(np.clip(np.arctan2(vy, vx), *YAW_RANGE) * gate, int(SMOOTH * FPS) | 1)
    return np.c_[vx, vy, head]


def measure(path: str) -> dict:
    """-> {clip, actor, straight_s: seconds of straight forward walking, roll: median shoulder roll magnitude [rad] over
    it, or None}."""
    q, pos, R = load(path)
    tr, fps, pos, R = travel(pos, R)[::EVERY], FPS / EVERY, pos[::EVERY], R[::EVERY]
    h = np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0]))  # trunk heading
    k = int(fps) | 1
    speed = centered(np.hypot(*np.gradient(pos[:, :2], axis=0).T) * fps, k)
    rate = np.degrees(np.abs(centered(np.gradient(h) * fps, k)))
    straight = (speed > WALK_V) & (rate < WALK_RATE) & (np.abs(np.degrees(np.arctan2(tr[:, 1], tr[:, 0]))) < WALK_TRAVEL)
    roll = (q[ROLLS[1]][::EVERY][straight] - q[ROLLS[0]][::EVERY][straight]) / 2
    return {"clip": path, "actor": re.search(r"__(A\d+)", os.path.basename(path)).group(1),
            "straight_s": float(straight.sum() / fps), "roll": float(np.median(roll)) if len(roll) else None}


def toe_out(path: str, max_rate: float = WALK_RATE) -> dict:
    """-> {clip, actor, straight_s: seconds of straight forward walking timed on the pelvis (turning slower than
    max_rate [deg/s]), L, R: toe-out [deg] or None}."""
    q, pos, R = load(path)
    qpos, tr, fps = qpos_of(q, pos, R)[::EVERY], travel(pos, R)[::EVERY], FPS / EVERY
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
    straight = (speed > WALK_V) & (rate < max_rate) & (np.abs(np.degrees(np.arctan2(tr[:, 1], tr[:, 0]))) < WALK_TRAVEL)
    out = {"clip": path, "actor": re.search(r"__(A\d+)", os.path.basename(path)).group(1),
           "straight_s": float(straight.sum() / fps)}
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
            has two solutions, knee either side of a small angle (~3.4 deg on the T1); keeping to the source's side
            keeps the leg as continuous as the source, where a swap jumped the T1's knee ~7 deg in a frame, so a turn
            only the other side reaches is cut (and eased) instead."""
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


def edit(job: tuple[str, str, float, tuple[float, float] | None, bool]) -> dict:
    """(source CSV, output CSV, shoulder roll shift inward [rad], toe-out (left, right) [deg] or None: no feet edit,
    untwist the trunk) -> writes the output; -> its statistics."""
    src, dst, shift, toe, twist = job
    q, pos, R = load(src)
    new = {n: np.full(len(pos), DEFAULT_JOINT_POS.get(n, 0.0)) for n in ARMS + [HEAD_PITCH]}
    if twist:  # first: the head and the feet below work in the corrected trunk frame
        waist, pos, R, waist_clipped = untwist(q, pos, R)
        new.update(waist)
    new[HEAD_YAW] = travel(pos, R)[:, 2]
    clipped = 0.0
    if shift > 0:
        for n, sign in zip(ROLLS, (-1, 1)):
            lo, hi = M.jnt_range[M.joint(n).id]
            x = q[n] + sign * shift
            new[n] = np.clip(x, lo, hi)
            clipped = max(clipped, float(((x < lo) | (x > hi)).mean()))
    if toe is not None:  # the feet last, on the edited pose
        qpos, info = straighten(qpos_of(q | new, pos, R), toe)
        new.update({n: qpos[:, adr(n)] for names, t in zip(LEGS, toe) if t for n in names})

    with open(src, newline="") as f:
        text = f.read()
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)
    trailing = lines[-1] == ""
    rows = [line.split(",") for line in lines[1 : len(lines) - trailing]]
    col = {n: i for i, n in enumerate(lines[0].split(","))}
    assert len(rows) == len(pos), (src, len(rows), len(pos))
    for n, x in new.items():
        for row, v in zip(rows, np.degrees(x).astype(np.float32)):
            row[col[n]] = str(v)
    if twist:  # the root columns: position [cm], extrinsic xyz Euler [deg]
        for row, x in zip(rows, np.c_[pos * 100, euler_xyz(R)]):
            row[1:7] = [str(float(v)) for v in x]
    with open(dst, "w", newline="") as f:
        f.write(nl.join([lines[0]] + [",".join(r) for r in rows]) + (nl if trailing else ""))
    yaw = np.degrees(new[HEAD_YAW])
    res = {"clip": os.path.basename(src), "frames": len(pos), "roll_shift": shift, "roll_clipped": clipped,
           "head_yaw_deg": {"min": float(yaw.min()), "max": float(yaw.max()), "mean_abs": float(np.abs(yaw).mean()),
                            "turned": float((np.abs(yaw) > 1).mean())}}
    if toe is not None:
        res.update(toe_deg=list(toe), max_pos_mm=float(info[:, 0].max() * 1e3),
                   max_heading_deg=float(np.degrees(info[:, 1].max())),
                   max_tilt_deg=float(np.degrees(info[:, 2].max())), frac_partial=float((info[:, 3] < 1).mean()),
                   min_fraction=float(info[:, 3].min()))
    if twist:
        res.update(waist_clipped=float(waist_clipped.mean()))
    return res


def _init():
    import torch

    torch.set_num_threads(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="folder of soma T2 CSVs (searched recursively)")
    p.add_argument("--dst", required=True, help="output folder: the same relative paths, + edits.json")
    p.add_argument("--jobs", type=int, default=max(1, os.cpu_count() - 2))
    p.add_argument("--feet", action=argparse.BooleanOptionalAction, default=True,
                   help="turn the feet in by the clip's toe-out (point 3); --no-feet keeps the legs")
    p.add_argument("--untwist", action="store_true", help="take the retarget's constant turn out of the trunk (point 4)")
    args = p.parse_args()
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    assert os.path.isdir(src), f"--src must be a folder: {src}"
    assert src != dst and not dst.startswith(src + os.sep) and not src.startswith(dst + os.sep), \
        "src and dst must not contain each other"
    assert not os.path.exists(dst) or not os.listdir(dst), f"--dst must be new or empty (no stale clips): {dst}"
    clips = soma_csv.find(src)
    with mp.Pool(args.jobs, _init) as pool:
        meas = pool.map(measure, clips, chunksize=8)
        rolls = collections.defaultdict(list)
        for m in meas:
            if m["straight_s"] > WALK_S:
                rolls[m["actor"]].append(m["roll"])
        actors = {}
        for a in sorted({m["actor"] for m in meas}):
            med = float(np.median(rolls[a])) if rolls[a] else None
            actors[a] = {"clips": sum(m["actor"] == a for m in meas), "walking_clips": len(rolls[a]), "roll_median": med,
                         "shift": max(0.0, ROLL_TARGET - med) if med is not None else 0.0}
        toes = [None] * len(meas)
        if args.feet:  # a motion's own toe-out (the clip's and its mirror's), else its actor's median, else its actor's
            # measured with turns, else 0
            toes = pool.map(toe_out, clips, chunksize=8)
            own = lambda t: t["straight_s"] > WALK_S and t["L"] is not None and t["R"] is not None  # noqa: E731
            motion = lambda t: re.sub(r"_M\.csv$", ".csv", t["clip"])  # noqa: E731  a clip and its mirror: one motion
            by_actor, by_motion = collections.defaultdict(list), collections.defaultdict(list)
            for t in filter(own, toes):
                by_actor[t["actor"]].append((t["L"] + t["R"]) / 2)
                by_motion[motion(t)].append((t["L"] + t["R"]) / 2)
            rest = [t["clip"] for t in toes if t["actor"] not in by_actor]
            by_actor_turning = collections.defaultdict(list)
            for t in filter(own, pool.map(functools.partial(toe_out, max_rate=45.0), rest, chunksize=8)):
                by_actor_turning[t["actor"]].append((t["L"] + t["R"]) / 2)
            for t in toes:
                a, pair = t["actor"], by_motion[motion(t)]
                t["from"] = ("clip" if own(t) else "mirror" if pair else "actor" if by_actor[a] else
                             "actor_turning" if by_actor_turning[a] else "none")
                t["toe"] = (float(np.median(pair or by_actor[a] or by_actor_turning[a] or [0.0])),) * 2
        jobs = []
        for m, t in zip(meas, toes):
            out = os.path.join(dst, os.path.relpath(m["clip"], src))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            jobs.append((m["clip"], out, actors[m["actor"]]["shift"], t and t["toe"], args.untwist))
        shifts = [a["shift"] for a in actors.values() if a["roll_median"] is not None]
        print(f"{len(clips)} clips, {len(actors)} actors, {len(shifts)} with straight walking; shoulder roll shift: "
              f"{sum(s > 0 for s in shifts)} actors shifted, median {np.median(shifts or [0]):.3f} rad, "
              f"max {max(shifts or [0]):.3f} rad", flush=True)
        if args.feet:
            counts = collections.Counter(t["from"] for t in toes)
            print("toe-out from: " + ", ".join(f"{k} {v}" for k, v in counts.items()), flush=True)
        res = {r["clip"]: r for r in pool.imap_unordered(edit, jobs, chunksize=4)}
    assert len(res) == len(clips), "clip file names must be unique"
    for m, t in zip(meas, toes):
        r = res[os.path.basename(m["clip"])]
        r.update(path=os.path.relpath(m["clip"], src), actor=m["actor"], straight_s=m["straight_s"], roll_measured=m["roll"])
        if t:
            r.update(toe_from=t["from"], toe_straight_s=t["straight_s"], measured=(t["L"], t["R"]))
    params = {k: globals()[k] for k in ("ROLL_TARGET", "YAW_RANGE", "WINDOW", "SMOOTH", "VX_MIN", "VX_FULL", "EVERY",
                                        "WALK_V", "WALK_RATE", "WALK_TRAVEL", "WALK_S", "GRID", "TURN_RATE",
                                        "KNEE_STAY")}
    params["YAW_RANGE"] = [float(x) for x in YAW_RANGE]
    params.update(IK_W=IK_W.tolist(), feet=args.feet,
                  untwist_deg=np.degrees(UNTWIST).tolist() if args.untwist else None)
    with open(os.path.join(dst, "edits.json"), "w") as f:
        json.dump({"src": src, "params": params, "actors": actors, "clips": [res[k] for k in sorted(res)]}, f, indent=1)
    r = list(res.values())
    frames = np.array([x["frames"] for x in r])
    print(f"wrote {len(r)} clips to {dst}; roll clipped at its limit in "
          f"{np.average([x['roll_clipped'] for x in r], weights=frames) * 100:.2f}% of frames; head yaw "
          f"[{min(x['head_yaw_deg']['min'] for x in r):+.1f}, {max(x['head_yaw_deg']['max'] for x in r):+.1f}] deg")
    if args.feet:
        print(f"IK worst over all clips: {max(x['max_pos_mm'] for x in r):.2f} mm, heading "
              f"{max(x['max_heading_deg'] for x in r):.2f} deg, tilt {max(x['max_tilt_deg'] for x in r):.2f} deg; "
              f"frames with a partial turn {np.average([x['frac_partial'] for x in r], weights=frames) * 100:.2f}% "
              f"(in {sum(x['frac_partial'] > 0 for x in r)} clips)")
    if args.untwist:
        print(f"untwist: a waist joint clipped at its range in "
              f"{np.average([x['waist_clipped'] for x in r], weights=frames) * 100:.3f}% of frames "
              f"(in {sum(x['waist_clipped'] > 0 for x in r)} clips)")


if __name__ == "__main__":
    main()
