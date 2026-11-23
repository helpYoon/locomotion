"""Appendix E's data filter on soma T1 clips: writes the kept clips' paths, one per line, for utils/prepare_t1.py.

Dropped: names of scenes with props the sim lacks (stairs, boxes, chairs...) and of falls (faint, lying, death, fall;
not the paper's, which keeps floor motions: here they are 19 stand-up/lying clips in a walking set, and velocity envs
reset into their lying frames fall at once), then three kinematic feasibility gates
at the 50 Hz control rate, with z_min the lowest body-link origin (T1Kinematics):
  elevated support   z_min > 0.20 m for >= 0.6 s (standing on something);
  ground penetration z_min < -0.12 m, or < -0.08 m for over 80% of the clip;
  floating feet      both foot soles (the MJCF's collision boxes) > 0.02 m up for >= 0.5 s while the trunk is
                     upright, at standing height and nearly still.
The thresholds are the paper's, set for the G1; on this dataset every clip they flag is a fall the names drop first.

usage: python -m utils.filter_clips [paths ...] --out clips.txt
"""

from __future__ import annotations

import argparse
import os

import mujoco
import torch
import torch.nn.functional as F

from utils import soma_csv

NAME_TERMS = ("stair", "ladder", "box", "chair", "sit", "climb", "jump_off", "fence", "faint", "lying", "death", "fall")
CHUNK = 65536


# Private copy of the old code this filter ran on, verbatim from 37d5c4a (robot/t1.py, utils/rotations.py,
# src/motion.py, robot/kinematics.py), names unchanged. robot/ and src/ now hold BlinkMimic 2's motion library, which
# reads the packed 50 Hz library rather than the 120 fps CSVs, and its FK and rotation helpers, which differ from these
# in the last bits (and robot/t1.py imports mjlab). This copy keeps the keep/drop list exactly the old one. Nothing else
# may import it.
ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "t1", "t1_description")
MJCF_PATH = os.path.join(ASSET_DIR, "mjcf", "t1.xml")

T1_JOINT_NAMES = [
    "AAHead_yaw", "Head_pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
    "Waist",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]  # fmt: skip

TRACKED_BODIES = [
    "Trunk", "H2",
    "AL2", "AL3", "AL4", "left_hand_link",
    "AR2", "AR3", "AR4", "right_hand_link",
    "Hip_Roll_Left", "Hip_Yaw_Left", "Shank_Left", "Ankle_Cross_Left", "left_foot_link",
    "Hip_Roll_Right", "Hip_Yaw_Right", "Shank_Right", "Ankle_Cross_Right", "right_foot_link",
]  # fmt: skip


# 3x3 products as broadcast multiply-sums: torch.compile fuses them with the surrounding ops, while each batched
# matmul would stay a separate cuBLAS call (T1Kinematics.forward compiled at 131k frames on a 4090: forward
# 12.0 -> 2.3 ms, forward and backward 147 -> 8.8 ms)
def mm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A @ B for (..., 3, 3) matrices."""
    return (A[..., :, :, None] * B[..., None, :, :]).sum(-2)


def mv(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A @ x for (..., 3, 3) and (..., 3)."""
    return (A * x[..., None, :]).sum(-1)


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))  # fmt: skip


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """Robust matrix -> quaternion (wxyz, w >= 0)."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10)) / 2
    cands = torch.stack(
        (
            torch.stack((qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)), -1),
            torch.stack(((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)), -1),
            torch.stack(((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)), -1),
            torch.stack(((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1),
        ),
        dim=-2,
    )
    idx = torch.stack((qw, qx, qy, qz), -1).argmax(-1)
    q = torch.gather(cands, -2, idx[..., None, None].expand(idx.shape + (1, 4))).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return torch.where(q[..., :1] < 0, -q, q)


def axis_angle_to_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues for a fixed unit axis (..., 3) and angle (...)."""
    x, y, z = axis.unbind(-1)
    c, s = torch.cos(angle), torch.sin(angle)
    C = 1 - c
    return torch.stack(
        (
            c + x * x * C, x * y * C - z * s, x * z * C + y * s,
            y * x * C + z * s, c + y * y * C, y * z * C - x * s,
            z * x * C - y * s, z * y * C + x * s, c + z * z * C,
        ),
        dim=-1,
    ).unflatten(-1, (3, 3))  # fmt: skip


def so3_exp(v: torch.Tensor) -> torch.Tensor:
    """Rotation vector (..., 3) -> matrix, smooth at zero."""
    theta2 = (v * v).sum(-1, keepdim=True)
    small = theta2 < 1e-8
    theta = torch.sqrt(torch.where(small, 1.0, theta2))  # the small branch never reads it
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    h = torch.sin(theta / 2) / theta  # (1 - cos t) / t^2 = 2 sin^2(t/2) / t^2, no cancellation in float32
    b = torch.where(small, 0.5 - theta2 / 24, 2 * h * h)
    K = skew(v)
    eye = torch.eye(3, device=v.device, dtype=v.dtype).expand(K.shape)
    return eye + a[..., None] * K + b[..., None] * mm(K, K)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Matrix -> rotation vector via the quaternion (stable near 0 and pi)."""
    q = mat_to_quat(R)
    return quat_to_rotvec(q)


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    sin_half = torch.linalg.norm(v, dim=-1, keepdim=True)
    half = torch.atan2(sin_half, q[..., :1])
    small = sin_half < 1e-6
    scale = torch.where(small, 2.0 / torch.clamp(q[..., :1], min=1e-6), 2 * half / torch.where(small, 1.0, sin_half))
    return v * scale


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(v.shape[:-1] + (3, 3))


def yaw_mat(R: torch.Tensor) -> torch.Tensor:
    """Heading (yaw-only) rotation of R, from the projected x axis (still a rotation if the x axis is vertical)."""
    psi = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    c, s = torch.cos(psi), torch.sin(psi)
    o, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, o, s, c, o, o, o, one), -1).reshape(R.shape[:-2] + (3, 3))


CONTROL_DT = 0.02
BIN_FRAMES = 50  # 1 s bins for return-based reset sampling
REF_CMD_FRAMES = 25  # window of the reference velocity command's centered mean


class MotionLib:
    @torch.no_grad()
    def __init__(self, paths: str | list[str], device: str):
        """paths: soma CSV files and/or folders (searched recursively)."""
        files = soma_csv.find(paths)
        self.names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        # ponytail: serial parsing is ~26 s of the full dataset's ~29 s build (np.loadtxt holds the GIL, so threads are
        # slower); a fork pool of numpy-only workers measured ~4 s, add it if startup time starts to matter
        src_p, src_R, src_q = [], [], []
        for f in files:
            names, p, R, q = soma_csv.load(f)
            assert names == T1_JOINT_NAMES, f"{f}: joint columns differ from T1_JOINT_NAMES"
            src_p.append(p.float()), src_R.append(R.float()), src_q.append(q.float())
        src_len = torch.tensor([len(p) for p in src_p], device=device)
        src_start = src_len.cumsum(0) - src_len
        src_p, src_R, src_q = (torch.cat(x).to(device) for x in (src_p, src_R, src_q))

        # resample to CONTROL_DT, in float64: in float32 a last frame landing exactly on the last source frame
        # ((T - 1) % 12 == 0) floors away on CPU but not on CUDA, and + 1e-9 would be below its resolution
        step = CONTROL_DT * soma_csv.FPS  # source frames per control frame
        self.clip_len = ((src_len - 1).double() / step + 1e-9).floor().long() + 1
        assert (self.clip_len >= 2).all(), "velocities need clips of at least 2 control frames"
        self.clip_start = self.clip_len.cumsum(0) - self.clip_len
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(self.clip_len.sum())
        self.clip_id = torch.repeat_interleave(self.clip_len)
        k = torch.arange(self.num_frames, device=device)
        local = k - self.clip_start[self.clip_id]  # frame within its clip
        t = local.double() * step  # source frame time within the clip
        i0 = torch.minimum(t.long(), src_len[self.clip_id] - 2)
        a = (t - i0).clamp(max=1.0).float()[:, None]
        g = src_start[self.clip_id] + i0
        self.root_p, self.q = torch.lerp(src_p[g], src_p[g + 1], a), torch.lerp(src_q[g], src_q[g + 1], a)
        R0 = src_R[g]
        self.root_R = mm(R0, so3_exp(a * so3_log(mm(R0.mT, src_R[g + 1]))))  # geodesic interpolation
        del src_p, src_R, src_q, R0  # 2 GiB on the full dataset

        # central differences within each clip, one-sided at its ends (torch.gradient's edge_order=1)
        prv = torch.maximum(k - 1, self.clip_start[self.clip_id])
        nxt = torch.minimum(k + 1, self.clip_end[self.clip_id])
        dt = ((nxt - prv) * CONTROL_DT)[:, None]
        self.root_v = (self.root_p[nxt] - self.root_p[prv]) / dt
        self.root_w = so3_log(mm(self.root_R[nxt], self.root_R[prv].mT)) / dt  # world frame
        self.qd = (self.q[nxt] - self.q[prv]) / dt

        # per-frame goal parts: root velocities in the root frame
        self.root_v_local = mv(self.root_R.mT, self.root_v)
        self.root_w_local = mv(self.root_R.mT, self.root_w)

        # reference velocity command (Appendix G): planar root velocity in the heading frame and world yaw rate,
        # averaged over the REF_CMD_FRAMES centered on each frame, the window cut at the clip's ends
        heading_v = mv(yaw_mat(self.root_R).mT, self.root_v)[:, :2]
        csum = torch.cat((heading_v, self.root_w[:, 2:]), -1).double().cumsum(0)
        csum = torch.cat((csum.new_zeros(1, 3), csum))  # csum[k] = sum of frames before k
        lo = torch.maximum(k - REF_CMD_FRAMES // 2, self.clip_start[self.clip_id])
        hi = torch.minimum(k + REF_CMD_FRAMES // 2, self.clip_end[self.clip_id])
        self.ref_cmd = ((csum[hi + 1] - csum[lo]) / (hi - lo + 1)[:, None]).float()  # (num_frames, 3)

        # 1 s bins from each clip's start over the frames that have a next frame (an episode needs at least one step),
        # its last bin possibly shorter; clip_end joins that last bin instead of becoming a bin with no frame to start at
        # (upstream tiled all frames, so a clip of 50 k + 1 frames got a bin that could only start at clip_end)
        clip_bins = (self.clip_len - 2) // BIN_FRAMES + 1
        first_bin = clip_bins.cumsum(0) - clip_bins
        self.num_bins = int(clip_bins.sum())
        self.bin_clip = torch.repeat_interleave(clip_bins)
        bin_in_clip = torch.arange(self.num_bins, device=device) - first_bin[self.bin_clip]
        self.bin_start = self.clip_start[self.bin_clip] + bin_in_clip * BIN_FRAMES
        self.frame_bin = first_bin[self.clip_id] + torch.minimum(local, self.clip_len[self.clip_id] - 2) // BIN_FRAMES

    def ref_state(self, f):
        """Root position, rotation, world linear and angular velocity, joint positions and velocities at frames f."""
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]


class T1Kinematics:
    def __init__(self, device: str, tracked: list[str] = TRACKED_BODIES):
        m = mujoco.MjModel.from_xml_path(MJCF_PATH)
        # body 0 (Trunk) is free; body i > 0 hangs off its parent by hinge T1_JOINT_NAMES[i - 1] at its origin.
        # mujoco orders parents before children, so forward() can walk the bodies in order.
        assert (m.body_jntnum[1:] == 1).all() and [m.joint(j).name for j in range(1, m.njnt)] == T1_JOINT_NAMES
        assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and (m.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE).all()
        assert not m.jnt_pos.any() and not m.qpos0[7:].any()  # no joint anchors or ref angles
        f32 = lambda x: torch.tensor(x, dtype=torch.float32, device=device)  # noqa: E731
        self.body_names = [m.body(b).name for b in range(1, m.nbody)]
        self.num_bodies = len(self.body_names)
        self.parent = (m.body_parentid[1:] - 1).tolist()  # Trunk -> -1 (world)
        self.axes = f32(m.jnt_axis[1:])  # (29, 3)
        self.off_p = f32(m.body_pos[1:])
        self.off_R = quat_to_mat(f32(m.body_quat[1:]))
        self.tracked_idx = [self.body_names.index(n) for n in tracked]
        assert m.jnt_limited[1:].all()
        self.q_lo, self.q_hi = f32(m.jnt_range[1:].T)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions, rotations, linear (link-origin) and angular velocities.

        Args:
            p, R, v, w: root (trunk) link-origin position (B,3), rotation (B,3,3), world lin/ang velocity (B,3).
                From Isaac Lab use root_link_* (root_lin_vel_w is the COM velocity) and convert its xyzw quats with
                q.roll(1, -1); from MuJoCo the free joint's qvel[3:6] is in the body frame, so pass R @ qvel[3:6].
            q, qd: joint positions/velocities (B,29) in T1_JOINT_NAMES order.
        """
        R_loc = mm(self.off_R[1:], axis_angle_to_mat(self.axes, q))  # (B,29,3,3) body i + 1 in its parent's frame
        P, Rs, V, W = [p], [R], [v], [w]
        for i in range(1, self.num_bodies):
            par, j = self.parent[i], i - 1
            r = mv(Rs[par], self.off_p[i])  # parent origin -> joint (= this origin), world frame
            P.append(P[par] + r)
            Rs.append(mm(Rs[par], R_loc[..., j, :, :]))
            V.append(V[par] + torch.linalg.cross(W[par], r))
            W.append(W[par] + mv(Rs[i], self.axes[j]) * qd[..., j, None])  # the hinge leaves its own axis fixed
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )

# (end of the private copy)


def longest_run(mask: torch.Tensor) -> int:
    """Longest run of consecutive True."""
    edges = torch.diff(torch.cat((mask.new_zeros(1), mask, mask.new_zeros(1))).to(torch.int8))
    starts, ends = (edges == 1).nonzero()[:, 0], (edges == -1).nonzero()[:, 0]
    return int((ends - starts).max()) if len(starts) else 0


def sole_corners(device) -> dict[str, torch.Tensor]:
    """The 8 corners (8, 3) of each foot's collision box in its body frame."""
    m = mujoco.MjModel.from_xml_path(MJCF_PATH)
    out = {}
    for foot in ("left_foot_link", "right_foot_link"):
        body = m.body(foot).id
        g = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == body and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX)
        sign = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=torch.float32)
        R = quat_to_mat(torch.tensor(m.geom_quat[g], dtype=torch.float32))
        out[foot] = (torch.tensor(m.geom_pos[g], dtype=torch.float32) + mv(R, sign * torch.tensor(m.geom_size[g]))).to(device)
    return out


@torch.no_grad()
def feasibility(lib: MotionLib, kin: T1Kinematics) -> dict[str, list[int]]:
    """Clip indices failing each kinematic gate."""
    names = list(kin.body_names)
    corners = sole_corners(kin.q_lo.device)
    feet = [names.index(f) for f in corners]
    z_min, sole = torch.empty(lib.num_frames, device=kin.q_lo.device), torch.empty(lib.num_frames, 2, device=kin.q_lo.device)
    for s in range(0, lib.num_frames, CHUNK):
        P, R, *_ = kin.forward(*lib.ref_state(slice(s, s + CHUNK)), all_bodies=True)
        z_min[s : s + CHUNK] = P[..., 2].min(-1).values
        for i, (b, c) in enumerate(zip(feet, corners.values())):
            sole[s : s + CHUNK, i] = (P[:, b, None, 2] + mv(R[:, b, None], c)[..., 2]).min(-1).values
    standing_still = (
        (sole > 0.02).all(-1)
        & (lib.root_R[:, 2, 2] > 0.95)  # upright
        & (lib.root_p[:, 2] > 0.60)  # standing height (0.688 m with the knees bent 0.4 rad)
        & (lib.root_v[:, :2].norm(dim=-1) < 0.15)
    )
    sec = lambda t: round(t / CONTROL_DT)  # noqa: E731
    out = {"elevated": [], "penetration": [], "floating": []}
    for c, (s, e) in enumerate(zip(lib.clip_start.tolist(), (lib.clip_end + 1).tolist())):
        z = z_min[s:e]
        if longest_run(z > 0.20) >= sec(0.6):
            out["elevated"].append(c)
        if (z < -0.12).any() or (z < -0.08).float().mean() > 0.8:
            out["penetration"].append(c)
        if longest_run(standing_still[s:e]) >= sec(0.5):
            out["floating"].append(c)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", default=[soma_csv.DATASET])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    files = soma_csv.find(args.paths)
    by_name = [f for f in files if not any(t in f.rsplit("/", 1)[-1].lower() for t in NAME_TERMS)]
    lib = MotionLib(by_name, "cuda:0")
    dropped = feasibility(lib, T1Kinematics("cuda:0"))
    bad = set().union(*dropped.values())
    kept = [f for c, f in enumerate(by_name) if c not in bad]
    with open(args.out, "w") as f:
        f.writelines(k + "\n" for k in kept)
    print(f"{len(files)} clips: {len(files) - len(by_name)} dropped by name, {len(bad)} by the gates, {len(kept)} kept")
    for gate, clips in dropped.items():
        print(f"  {gate} ({len(clips)}): {', '.join(lib.names[c] for c in clips)}")


if __name__ == "__main__":
    main()
