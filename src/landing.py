"""The landing command: put a commanded foot (the swing sole) on a target pose in the planted sole's frame.

Poses are (x, y, yaw) triples; the sole is the bottom face of a foot link's collision box. Nothing in this module
imports the simulator or ``src.mdp``: it is plain tensor code that the trainer, the world model and the tests share.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from src.motion import CONTROL_DT
from utils.rotations import mv3

SOLE_C = (0.0101079, 0.0, -0.0432508)  # sole centre in the foot link's frame [m]
SOLE_HX, SOLE_HY = 0.112434, 0.05  # sole half sizes [m]
YAW_M = 0.1124  # metres of toe travel per radian of sole yaw
V_MOVE = 0.15  # m/s: a sole is moving above this speed (sole centre plus YAW_M x yaw rate, horizontal)
MIN_SWING = 5  # frames: shortest movement that counts as a swing
MIN_RISE = 0.02  # m: least height the swing sole's centre gains over its lower end point
MIN_TRAVEL = 0.05  # m: least distance + YAW_M x |yaw change| between lift-off and touchdown
LEAD = 25  # frames: a command from a stand starts this long before the lift-off; also the longest gap that chains
MARGIN = 3  # frames: rest after touchdown that the deadline waits for, at most
PLANTED_D, PLANTED_A = 0.03, 0.10  # m, rad: most the planted sole may move between command start and deadline
SETTLE = 25  # frames: hold frames needed before a command start, and either side of a free-row reset frame
RESET_SPAN = 50  # frames: label-row reset frames lie this far (from SETTLE up to SETTLE + RESET_SPAN) before a start
MIN_CLIP = 60  # frames: shorter clips get no labels at all
# widths: the command features enc_land reads (land_features), the command state C (B, STATE_DIM) and the landing
# features of the critic (land_critic_features)
CMD_DIM, STATE_DIM, CRITIC_DIM = 12, 12, 19
SWING, N, M = 0, 1, 2  # columns of the command state C (B, 12)
# T = C[:, 3:6] is the target; A = C[:, 6:12] the anchors (left, then right)


@dataclass
class LandingCfg:
    """Landing-command task (train.py --land_frac): examples, free-row sampling, clock and reward constants."""

    label_prob: float = 0.5  # share of landing rows that follow a clip's labelled steps (the rest are free rows)
    settle_steps: int = 25  # control steps of SETTLE after a reset before the first command may come
    wait_first: tuple[int, int] = (25, 100)  # free rows: control steps waited after SETTLE before the first command
    wait_next: tuple[int, int] = (25, 150)  # free rows: control steps waited after a deadline before the next command
    near_prob: float = 0.5  # chance that a free-row command is a near one (else anywhere in the box)
    near_xy: float = 0.15  # near commands: largest x and y offset from the swing sole's current pose [m]
    near_yaw: float = math.radians(25)  # near commands: largest yaw offset [rad]
    box_x: float = 0.33  # the landed stance's |x| bound in the planted sole's frame [m]
    box_y: tuple[float, float] = (0.14, 0.48)  # the landed stance's y range, mirrored for a right swing [m]
    box_yaw: tuple[float, float] = (math.radians(-20), math.radians(50))  # yaw range, toes out positive [rad]
    travel: tuple[float, float] = (0.05, 0.60)  # a command's travel (distance + YAW_M x |yaw change|) lies in this [m]
    sole_gap: float = 0.02  # least distance between the landed and the planted sole rectangles [m]
    candidates: int = 8  # free rows: targets drawn per command, the first that passes is sent
    clock_base: float = 44.0  # clock n0 = round(clock_base + clock_per_m x travel) [control steps]
    clock_per_m: float = 20.0  # control steps added per metre of travel
    clock_range: tuple[int, int] = (40, 62)  # n0 is kept within this [control steps]
    clock_jitter: int = 5  # free rows: a whole number from -clock_jitter to +clock_jitter is added to n0
    w_survival: float = 30.0  # constant per step [1/s]
    w_action_rate: float = -0.5  # x |a_t - a_{t-1}|
    w_torque: float = -0.0005  # x |joint torques| [N m]
    w_feet_jerk: float = -0.001  # x sum over the feet of |foot jerk| [m/s^3]
    w_drag: float = -10.0  # x sum over both soles of gate x (speed of the sole's centre + YAW_M x |yaw rate|) [m/s]
    w_settle_travel: float = -20.0  # x sum over both soles of (centre speed + YAW_M x |yaw rate|) in SETTLE
    w_still: float = 5.0  # x K_still in SETTLE and HOLD
    still_scale: float = 0.30  # K_still = k(largest leg-joint or trunk angular speed; still_scale) [rad/s]
    w_planted: float = 5.0  # x K_P in STEP
    planted_d: float = 0.03  # K_P: planted sole's distance scale from its anchor [m]
    planted_a: float = 0.15  # K_P: planted sole's yaw scale from its anchor [rad]
    w_progress: float = 5.0  # per metre of approach to the target in STEP
    bonus: float = 10.0  # lump paid once at the deadline, x G
    arrive_wide: float = 0.20  # G: position scale of the wide kernel [m]
    arrive_narrow: float = 0.04  # G: position scale of the narrow kernel [m]
    arrive_wide_share: float = 0.3  # G: weight of the wide kernel (the narrow one gets the rest)
    arrive_a: float = 0.10  # G: yaw scale [rad]
    down_h: float = 0.02  # G: clearance scale, the swing sole must be down [m]
    w_keep: float = 5.0  # x K_keep x K_flat in HOLD
    keep_d: float = 0.02  # K_keep: each sole's distance scale from its anchor [m]
    keep_a: float = 0.10  # K_keep: each sole's yaw scale from its anchor [rad]
    flat_tilt: float = 0.10  # K_flat: tilt scale [rad]
    w_ankle: float = -4.0  # x (1 - exp(-summed squared ankle-pitch torque / ankle_scale)) in HOLD
    ankle_scale: float = 400.0  # [N^2 m^2]
    gate_h: float = 0.005  # gate = sigmoid((gate_h - zc) / gate_w), zc the sole centre's height: half contact here [m]
    gate_w: float = 0.0015  # gate's softness [m]
    scale: float = 0.125  # every weight times scale, as VelocityCfg
    derived: bool = True  # the encoder receives the 5 derived inputs (the swing sole's pose now); False zeroes them

    @property
    def max_step_reward(self) -> float:
        """Largest per-step reward of a HOLD step (the scaled constant, keep and still terms times the control step)."""
        return self.scale * (self.w_survival + self.w_keep + self.w_still) * CONTROL_DT

    @property
    def bonus_reward(self) -> float:
        """The deadline lump at its largest (G = 1), after the scale."""
        return self.scale * self.bonus


def wrap(a: torch.Tensor) -> torch.Tensor:
    """Angle into [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def compose(P: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """The world pose whose pose in P's frame is c; P, c and the result are (..., 3) poses (x, y, yaw)."""
    cs, sn = torch.cos(P[..., 2]), torch.sin(P[..., 2])
    x = P[..., 0] + cs * c[..., 0] - sn * c[..., 1]
    y = P[..., 1] + sn * c[..., 0] + cs * c[..., 1]
    return torch.stack((x, y, wrap(P[..., 2] + c[..., 2])), -1)


def relative(X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Pose X in pose P's frame (the inverse of ``compose``); (..., 3) poses (x, y, yaw)."""
    cs, sn = torch.cos(P[..., 2]), torch.sin(P[..., 2])
    dx, dy = X[..., 0] - P[..., 0], X[..., 1] - P[..., 1]
    return torch.stack((cs * dx + sn * dy, -sn * dx + cs * dy, wrap(X[..., 2] - P[..., 2])), -1)


def _sole_offset(bR: torch.Tensor) -> torch.Tensor:
    """The sole centre in the world frame relative to the link, R @ SOLE_C; bR (..., 3, 3), result (..., 3)."""
    return bR[..., 0] * SOLE_C[0] + bR[..., 1] * SOLE_C[1] + bR[..., 2] * SOLE_C[2]


def soles(bp: torch.Tensor, bR: torch.Tensor, bv: torch.Tensor, bw: torch.Tensor) -> dict[str, torch.Tensor]:
    """The soles of the two foot links (bp, bv, bw: (B, 2, 3); bR: (B, 2, 3, 3)): "pose" (B, 2, 3) = (x, y, yaw) of the
    sole centre and the link's heading, "h" (B, 2) lowest-corner clearance, "zc" (B, 2) the centre's height,
    "tilt" (B, 2) [rad], "v" (B, 2, 2) the horizontal velocity of the sole centre, "wz" (B, 2) the yaw rate [rad/s]."""
    off = _sole_offset(bR)
    centre = bp + off
    pose = torch.cat((centre[..., :2], torch.atan2(bR[..., 1, 0], bR[..., 0, 0])[..., None]), -1)
    h = centre[..., 2] - bR[..., 2, 0].abs() * SOLE_HX - bR[..., 2, 1].abs() * SOLE_HY
    tilt = torch.atan2(torch.sqrt(bR[..., 2, 0] ** 2 + bR[..., 2, 1] ** 2 + 1e-12), bR[..., 2, 2])
    v = (bv + torch.cross(bw, off, dim=-1))[..., :2]
    return {"pose": pose, "h": h, "zc": centre[..., 2], "tilt": tilt, "v": v, "wz": bw[..., 2]}


def sole_rel(bp: torch.Tensor, bR: torch.Tensor, swing: torch.Tensor) -> torch.Tensor:
    """The swing sole in the planted sole's 3-D frame, (B, 5) = x, y, z, cos yaw, sin yaw. bp (B, 2, 3) and bR
    (B, 2, 3, 3) are the two foot links; swing (B,) is 0 (left) or 1 (right)."""
    centre = bp + _sole_offset(bR)
    s = swing.to(bp.dtype)
    w, p = s[:, None], 1 - s[:, None]
    cW = centre[:, 0] * p + centre[:, 1] * w
    cP = centre[:, 0] * w + centre[:, 1] * p
    RW = bR[:, 0] * p[..., None] + bR[:, 1] * w[..., None]
    RP = bR[:, 0] * w[..., None] + bR[:, 1] * p[..., None]
    t = mv3(RP.transpose(-1, -2), cW - cP)
    rot = RP.transpose(-1, -2) @ RW
    yaw = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])
    return torch.cat((t, torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]), -1)


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end (exclusive) of each run of True."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]


def _wrap(x: np.ndarray | float) -> np.ndarray | float:
    return (x + np.pi) % (2 * np.pi) - np.pi


def label_tables(
    xy: np.ndarray, yaw: np.ndarray, zc: np.ndarray, clip_start: np.ndarray, clip_end: np.ndarray
) -> dict[str, np.ndarray]:
    """The label rule on a library's soles: xy (F, 2, 2) sole centres, yaw (F, 2), zc (F, 2) centre heights (left,
    right); clip_start / clip_end (inclusive) per clip. Returns per frame kind uint8 (F,) (0 unlabelled, 1 hold, 2
    step), swing float32 (F,) and n float32 (F,) (the commanded sole 0 left / 1 right and the control steps to the
    deadline, on step frames, else 0), c float32 (F, 3) (on step frames the swing sole's pose at the deadline in the
    planted sole's frame at this frame, else 0), ahead int16 (F,) (labelled frames from this one to the clip's end,
    this one included, while they are consecutive); the int64 frame lists label_reset and free_reset; steps int64
    (S, 5): clip, swing, command start, deadline, chained (frames are library indices)."""
    F = len(xy)
    kind_all = np.zeros(F, np.uint8)
    swing_all, n_all = np.zeros(F, np.float32), np.zeros(F, np.float32)
    c_all, ahead_all = np.zeros((F, 3), np.float32), np.zeros(F, np.int16)
    steps, label_reset, free_reset = [], [], []
    for ci, (s, e) in enumerate(zip(clip_start, clip_end + 1)):
        T = e - s
        if T < MIN_CLIP:
            continue
        X, Y, Z = xy[s:e], yaw[s:e], zc[s:e]
        v = np.zeros((T, 2), np.float32)
        v[1:-1] = (np.linalg.norm(X[2:] - X[:-2], axis=-1) + YAW_M * np.abs(_wrap(Y[2:] - Y[:-2]))) / (2 * CONTROL_DT)
        v[0], v[-1] = v[1], v[-2]
        moving = v > V_MOVE
        for k in range(2):
            for i, j in zip(*_runs(~moving[:, k])):
                if j - i <= 2 and i > 0 and j < T:
                    moving[i:j, k] = True
        anym = moving.any(1)
        cand = []
        for w in range(2):
            o = 1 - w
            for i, j in zip(*_runs(moving[:, w])):
                if j - i < MIN_SWING or i == 0 or j >= T:
                    continue
                lo, td = i - 1, j  # lift-off: last frame at rest; touchdown: first frame at rest
                if moving[lo : td + 1, o].any():
                    continue
                rise = float(Z[lo : td + 1, w].max() - min(Z[lo, w], Z[td, w]))
                travel = float(np.linalg.norm(X[td, w] - X[lo, w])) + YAW_M * abs(float(_wrap(Y[td, w] - Y[lo, w])))
                if rise < MIN_RISE or travel < MIN_TRAVEL:
                    continue
                g = 0
                while g < MARGIN and td + g + 1 < T and not anym[td + g + 1]:
                    g += 1
                dl = td + g  # the deadline
                if dl >= T - 1:
                    continue
                cand.append((lo, td, dl, w))
        cand.sort()
        kind = np.where(anym, 0, 1).astype(np.uint8)
        swing, n, c = np.zeros(T, np.float32), np.zeros(T, np.float32), np.zeros((T, 3), np.float32)
        prev_dl, starts = None, []
        for lo, td, dl, w in cand:
            o = 1 - w
            if prev_dl is not None and prev_dl <= lo and lo - prev_dl <= LEAD and not anym[prev_dl : lo + 1].any():
                c0, chained = prev_dl, 1
            elif lo - LEAD >= 0 and not anym[lo - LEAD : lo + 1].any():
                c0, chained = lo - LEAD, 0
            else:
                prev_dl = None
                continue
            if np.linalg.norm(X[dl, o] - X[c0, o]) > PLANTED_D or abs(_wrap(Y[dl, o] - Y[c0, o])) > PLANTED_A:
                prev_dl = None
                continue
            kind[c0:dl] = 2
            t = np.arange(c0, dl)
            yp = Y[t, o]  # the landing pose in the planted sole's frame at each frame
            d = X[dl, w] - X[t, o]
            swing[t], n[t] = w, dl - t
            c[t, 0] = np.cos(yp) * d[:, 0] + np.sin(yp) * d[:, 1]
            c[t, 1] = -np.sin(yp) * d[:, 0] + np.cos(yp) * d[:, 1]
            c[t, 2] = _wrap(Y[dl, w] - yp)
            steps.append((ci, w, s + c0, s + dl, chained))
            starts.append(c0)
            prev_dl = dl
        for i, j in zip(*_runs(kind == 1)):  # free-row resets: hold frames with SETTLE frames of hold on both sides
            free_reset.extend(range(s + i + SETTLE, s + j - SETTLE))
        for c0 in starts:  # label-row resets: SETTLE to SETTLE + RESET_SPAN frames of hold before a command start
            for t in range(max(c0 - SETTLE - RESET_SPAN, 0), c0 - SETTLE + 1):
                if (kind[t:c0] == 1).all():
                    label_reset.append(s + t)
        lab = (kind > 0).astype(np.int16)
        run = np.zeros(T + 1, np.int16)
        for t in range(T - 1, -1, -1):
            run[t] = (run[t + 1] + 1) * lab[t]
        kind_all[s:e], swing_all[s:e], n_all[s:e], c_all[s:e], ahead_all[s:e] = kind, swing, n, c, run[:T]
    return {
        "kind": kind_all, "swing": swing_all, "n": n_all, "c": c_all, "ahead": ahead_all,
        "label_reset": np.array(label_reset, np.int64), "free_reset": np.array(free_reset, np.int64),
        "steps": np.array(steps, np.int64).reshape(-1, 5),
    }  # fmt: skip


def _k(x: torch.Tensor, s: float) -> torch.Tensor:
    """The kernel exp(-(x / s)^2), 1 at x = 0."""
    return torch.exp(-((x / s) ** 2))


def _dist(d: torch.Tensor) -> torch.Tensor:
    """Euclidean length over the last dim with a finite gradient at zero; (..., k) to (...)."""
    return torch.sqrt((d**2).sum(-1) + 1e-12)


def _pick(P: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Sole w of P (B, 2, ...) by blending with w (B,) in {0, 1}."""
    w = w.reshape(-1, *([1] * (P.dim() - 2)))
    return P[:, 0] * (1 - w) + P[:, 1] * w


def phase(C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(settle, step, hold) bool (B,) of the command state C (B, 12): settle m > 0, step n > 0 outside it, else hold."""
    settle = C[:, M] > 0.5
    step = ~settle & (C[:, N] > 0.5)
    return settle, step, ~settle & ~step


def land_reset(
    cfg: LandingCfg, C: torch.Tensor, pose: torch.Tensor, mask: torch.Tensor, swing: torch.Tensor
) -> torch.Tensor:
    """C with the rows in mask (B,) bool reset onto soles pose (B, 2, 3): m = settle_steps, n = 0, swing (B,) in {0, 1},
    both anchors on the soles, the target on the swing sole."""
    fresh = torch.cat(
        (swing[:, None], torch.zeros_like(swing)[:, None], torch.full_like(swing, cfg.settle_steps)[:, None],
         _pick(pose, swing), pose.flatten(1)), -1)  # fmt: skip
    return torch.where(mask[:, None], fresh, C)


def land_issue(
    C: torch.Tensor, pose: torch.Tensor, arrive: torch.Tensor, swing: torch.Tensor, c: torch.Tensor, n0: torch.Tensor
) -> torch.Tensor:
    """C with the command (swing (B,), c (B, 3) in the planted sole's frame, clock n0 (B,)) taken by the rows in arrive
    (B,) bool that are in HOLD: the planted anchor is latched on soles pose (B, 2, 3) and the target composed on it."""
    take = arrive & phase(C)[2]
    pose = pose.detach()  # the anchor and the target are constants of the reward
    A = C[:, 6:].view(-1, 2, 3)
    A = torch.stack((A[:, 0] * (1 - swing[:, None]) + pose[:, 0] * swing[:, None],
                     A[:, 1] * swing[:, None] + pose[:, 1] * (1 - swing[:, None])), 1)  # fmt: skip
    new = torch.cat((swing[:, None], n0[:, None], C[:, M : M + 1], compose(_pick(A, 1 - swing), c), A.flatten(1)), -1)
    return torch.where(take[:, None], new, C)


def land_step(
    cfg: LandingCfg,
    C: torch.Tensor,
    pose0: torch.Tensor,
    s1: dict[str, torch.Tensor],
    rate: torch.Tensor,
    torque: torch.Tensor,
    jerk: torch.Tensor,
    ankle_t2: torch.Tensor,
    still_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """One control step. C (B, 12) is the state in force (after the issue), pose0 (B, 2, 3) the sole poses the step
    starts from, s1 the ``soles`` of the state it ends in (it needs pose, h, zc, tilt, v, wz); rate, torque, jerk,
    ankle_t2 (summed squared ankle-pitch torque) and still_v (largest leg-joint or trunk angular speed) are (B,).
    SETTLE charges sole travel, so that a robot does not rearrange its stance before the anchors latch; its last step
    latches the anchors on the soles.
    Returns the reward (B,), the advanced C and the terms G, deadline, e, a, drag, planted, progress, keep (each
    (B,))."""
    settle, stepping, hold = phase(C)
    deadline = stepping & (C[:, N] < 1.5)
    w = C[:, SWING]
    T, A, pose1 = C[:, 3:6], C[:, 6:].view(-1, 2, 3), s1["pose"]
    gate = torch.sigmoid((cfg.gate_h - s1["zc"]) / cfg.gate_w)  # (B, 2) 1 = the sole's centre is on the floor
    W, P = _pick(pose1, w), _pick(pose1, 1 - w)
    e, a = _dist(W[:, :2] - T[:, :2]), torch.sqrt(wrap(W[:, 2] - T[:, 2]) ** 2 + 1e-12)
    AP = _pick(A, 1 - w)
    d_P, a_P = _dist(P[:, :2] - AP[:, :2]), torch.sqrt(wrap(P[:, 2] - AP[:, 2]) ** 2 + 1e-12)
    K_P = _k(d_P, cfg.planted_d) * _k(a_P, cfg.planted_a)
    h_W = _pick(s1["h"], w)
    G = (
        (cfg.arrive_wide_share * _k(e, cfg.arrive_wide) + (1 - cfg.arrive_wide_share) * _k(e, cfg.arrive_narrow))
        * _k(a, cfg.arrive_a)
        * _k(h_W, cfg.down_h)
        * K_P
    )
    W0 = _pick(pose0, w)
    travel0 = _dist(W0[:, :2] - T[:, :2]) + YAW_M * torch.sqrt(wrap(W0[:, 2] - T[:, 2]) ** 2 + 1e-12)
    approach = travel0 - (e + YAW_M * a)
    air = 1 - _pick(gate, w)
    progress = air * torch.relu(approach) - torch.relu(-approach)
    turn = torch.sqrt(s1["wz"] ** 2 + 1e-12) - 1e-6  # |yaw rate|, exactly 0 at rest
    speed = _dist(s1["v"]) - 1e-6 + YAW_M * turn  # (B, 2) the centre's speed and the turn, exactly 0 at rest
    drag = (gate * speed).sum(-1)
    travel = speed.sum(-1)  # the same without the gate: a sole that moves, on the floor or not
    d_A = _dist(pose1[..., :2] - A[..., :2])  # (B, 2) each sole from its anchor
    a_A = torch.sqrt(wrap(pose1[..., 2] - A[..., 2]) ** 2 + 1e-12)
    keep = (_k(d_A, cfg.keep_d) * _k(a_A, cfg.keep_a)).prod(-1)
    flat = _k(s1["tilt"], cfg.flat_tilt).prod(-1)
    K_still = _k(still_v, cfg.still_scale)
    per_s = (
        cfg.w_survival + cfg.w_action_rate * rate + cfg.w_torque * torque + cfg.w_feet_jerk * jerk + cfg.w_drag * drag
        + (settle | hold) * cfg.w_still * K_still
        + settle * cfg.w_settle_travel * travel
        + stepping * (cfg.w_planted * K_P + cfg.w_progress * progress / CONTROL_DT)
        + hold * (cfg.w_keep * keep * flat + cfg.w_ankle * (1 - torch.exp(-ankle_t2 / cfg.ankle_scale)))
    )  # fmt: skip
    reward = per_s * cfg.scale * CONTROL_DT + deadline * cfg.bonus_reward * G
    latch = (settle & (C[:, M] < 1.5)) | deadline
    new_A = torch.where(latch[:, None], pose1.detach().flatten(1), C[:, 6:])
    n1, m1 = (C[:, N] - stepping.to(C.dtype))[:, None], (C[:, M] - settle.to(C.dtype))[:, None]
    C1 = torch.cat((C[:, :N], n1, m1, C[:, 3:6], new_A), -1)
    terms = {"G": G, "deadline": deadline, "e": e, "a": a, "drag": drag, "planted": K_P, "progress": progress,
             "keep": keep}  # fmt: skip
    return reward, C1, terms


def _separated(c: torch.Tensor, gap: float) -> torch.Tensor:
    """(...,) bool: the sole rectangle landed at c (..., 3) = (x, y, yaw) is ``gap`` or more from the planted sole's
    rectangle (at the origin, same heading) along at least one of the four rectangle axes."""
    cs, sn = torch.cos(c[..., 2]), torch.sin(c[..., 2])
    zero, one = torch.zeros_like(cs), torch.ones_like(cs)
    ex, ey = (cs, sn), (-sn, cs)  # the landed rectangle's axes
    axes = ((one, zero), (zero, one), ex, ey)
    ok = torch.zeros_like(cs, dtype=torch.bool)
    for nx, ny in axes:
        r_planted = SOLE_HX * nx.abs() + SOLE_HY * ny.abs()
        r_landed = SOLE_HX * (nx * ex[0] + ny * ex[1]).abs() + SOLE_HY * (nx * ey[0] + ny * ey[1]).abs()
        ok = ok | ((c[..., 0] * nx + c[..., 1] * ny).abs() - r_planted - r_landed >= gap)
    return ok


def land_sample(
    cfg: LandingCfg, pose: torch.Tensor, u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw a command for a robot that follows no clip, from the sole poses now (B, 2, 3) and uniform numbers u
    (B, 3 + 3 cfg.candidates): u[:, 0] < 0.5 left swing else right, u[:, 1] < near_prob a near draw else anywhere,
    u[:, 2] the clock jitter, u[:, 3:] in threes each candidate's x, y, yaw draws. Candidates are built in mirrored
    coordinates (y and yaw times +1 for a left swing, -1 for a right one) and the first that passes (travel within
    cfg.travel, soles cfg.sole_gap apart) is sent. Returns swing (B,) float, c (B, 3) in the planted sole's frame,
    n0 (B,) float whole numbers and ok (B,) bool (a command was found; where none was, c and n0 are meaningless)."""
    swing = (u[:, 0] >= 0.5).to(pose.dtype)
    sg = (1 - 2 * swing)[:, None]
    mirror = torch.cat((torch.ones_like(sg), sg, sg), -1)
    cur = relative(_pick(pose, swing), _pick(pose, 1 - swing)) * mirror  # (B, 3) the stance now, mirrored
    d = u[:, 3:].reshape(len(u), cfg.candidates, 3)
    lo, hi = (cfg.box_x * -1, cfg.box_y[0], cfg.box_yaw[0]), (cfg.box_x, cfg.box_y[1], cfg.box_yaw[1])
    reach = (cfg.near_xy, cfg.near_xy, cfg.near_yaw)
    k = range(3)
    anywhere = torch.stack([lo[i] + d[..., i] * (hi[i] - lo[i]) for i in k], -1)
    near = torch.stack(
        [torch.clamp(cur[:, None, i] + (2 * d[..., i] - 1) * reach[i], lo[i], hi[i]) for i in k], -1
    )
    cand = torch.where((u[:, 1] < cfg.near_prob)[:, None, None], near, anywhere)  # (B, K, 3)
    yaw_d = torch.sqrt(wrap(cand[..., 2] - cur[:, None, 2]) ** 2 + 1e-12)
    travel = _dist(cand[..., :2] - cur[:, None, :2]) + YAW_M * yaw_d  # (B, K)
    passes = (travel >= cfg.travel[0]) & (travel <= cfg.travel[1]) & _separated(cand, cfg.sole_gap)
    first = passes.to(u.dtype).argmax(-1)
    c = cand.gather(1, first[:, None, None].expand(-1, 1, 3))[:, 0] * mirror
    tr = travel.gather(1, first[:, None])[:, 0]
    jitter = torch.floor(u[:, 2] * (2 * cfg.clock_jitter + 1)) - cfg.clock_jitter
    n0 = torch.clamp(torch.round(cfg.clock_base + cfg.clock_per_m * tr), *cfg.clock_range) + jitter
    return swing, c, n0, passes.any(-1)


def land_features(cfg: LandingCfg, C: torch.Tensor, rel5: torch.Tensor) -> torch.Tensor:
    """The 12 command numbers of ``enc_land`` from the command state C (B, 12) and rel5 (B, 5) = ``sole_rel`` of the
    observed joints: left flag, right flag (the commanded sole); target x / 0.5, y / 0.5, cos yaw, sin yaw in the
    planted anchor's frame (the command T while the clock runs, else the stance held); tau / 1.2 with tau = 0.02 n;
    then rel5's x / 0.5, y / 0.5, z / 0.5, cos, sin (zeros when ``cfg.derived`` is false)."""
    w = C[:, SWING]
    A = C[:, 6:].view(-1, 2, 3)
    AP = _pick(A, 1 - w)
    running = (C[:, N] > 0.5)[:, None]
    t = torch.where(running, relative(C[:, 3:6], AP), relative(_pick(A, w), AP))
    tau = (CONTROL_DT * C[:, N] / 1.2)[:, None]
    derived = torch.cat((rel5[:, :3] * 2, rel5[:, 3:]), -1)
    if not cfg.derived:
        derived = torch.zeros_like(derived)
    return torch.cat(
        (1 - w[:, None], w[:, None], t[:, :2] / 0.5, torch.cos(t[:, 2:]), torch.sin(t[:, 2:]), tau, derived), -1
    )


def land_critic_features(C: torch.Tensor, s: dict[str, torch.Tensor]) -> torch.Tensor:
    """The 19 landing features of the critic from the command state C (B, 12) and s = ``soles`` of the state: left
    flag, right flag; n / 50, m / 25, hold flag; x, y of the swing sole's offset from T in the planted anchor's frame,
    sin and cos of the wrapped yaw difference; clearance left, right; for the left then the right sole its x, y and
    sin yaw offset from its anchor; tilt left, tilt right."""
    w = C[:, SWING]
    T, A, pose = C[:, 3:6], C[:, 6:].view(-1, 2, 3), s["pose"]
    AP = _pick(A, 1 - w)
    err = relative(_pick(pose, w), AP) - relative(T, AP)
    yaw = wrap(err[:, 2])
    off = relative(pose, A)  # (B, 2, 3)
    own = torch.stack((off[..., 0], off[..., 1], torch.sin(off[..., 2])), -1).flatten(1)
    hold = phase(C)[2].to(C.dtype)
    return torch.cat(
        (1 - w[:, None], w[:, None], C[:, N : N + 1] / 50, C[:, M : M + 1] / 25, hold[:, None], err[:, :2],
         torch.sin(yaw)[:, None], torch.cos(yaw)[:, None], s["h"], own, s["tilt"]), -1)  # fmt: skip
