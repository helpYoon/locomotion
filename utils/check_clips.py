r"""Kinematic checks of packed clip libraries (utils/prepare_t1.py / prepare_t2.py) on the CPU, and MP4s of named clips.

Per clip set, at the library's 50 Hz:
- joints within 0.01 rad of a hard limit; knee below, ankle pitch outside and |ankle roll| above the soft limits (0.9 of
  each range about its middle, src/mdp.py RewardCfg.soft_limit_factor); one-frame jumps over 0.3 rad;
- soles: the corners of each foot's collision box from FK. Penetration = the lowest corner below -1 cm. A foot is
  planted when its sole centre moves under 0.1 m/s horizontally and vertically, and floats when planted with its lowest
  corner above 2 cm. Slide = horizontal speed of the lowest corner in the frames MotionLib.set_ref_contact labels as
  contact (training's reference contact: the foot origin within 1.5 cm of the clip's own floor, moving under 0.5 m/s).
  Not the lowest corner's height: a tilted swing foot's corner skims the floor, as low as a planted sole's;
- self-contacts from MuJoCo's mj_collision on the training collision model (the robot module's get_spec(): no floor,
  so every contact is robot-robot): frames with any part deeper than 1 cm, by body-pair kind, and contacts per frame;
- root speed percentiles and the reference velocity command box (0.1-99.9 percentiles of MotionLib.ref_cmd);
- a per-clip table.

--video renders each named clip of every set to <video_dir>/<label>/<clip>.mp4 (the MJCF with its visual meshes and its
own floor, camera following the root), and with two sets also <video_dir>/side_by_side/<clip>.mp4 (first set left).
Rendering needs MuJoCo's EGL backend; on the CPU (Mesa) with:
  __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl

usage: CUDA_VISIBLE_DEVICES= python -m utils.check_clips --robot t2 --set 0.75=data/t2_pilot075/lib_all.npz \
           --set 0.80=data/t2_pilot080/lib_all.npz [--list names.txt] [--video CLIP ...] [--video_dir logs/t2_pilot]
--list defaults to the keep.txt next to each library.
"""

import argparse
import importlib
import itertools
import os
import re

import mujoco
import numpy as np

from robot.kinematics import Kinematics
from src.motion import MotionLib

HARD = 0.01  # rad from a hard limit
SOFT = 0.9  # soft limit factor
JUMP = 0.3  # rad in one 50 Hz frame
PEN, FLOAT = -0.01, 0.02  # m: sole penetration, planted foot floating
PLANT_V = 0.1  # m/s
DEEP = 0.01  # m: self-penetration depth reported


def part(body: str) -> str:
    """Side and kind of a body, e.g. "L-leg", "R-hand", "torso" (T1 and T2 names)."""
    n = body.lower()
    side = "L-" if "left" in n or re.match(r"al\d", n) else "R-" if "right" in n or re.match(r"ar\d", n) else ""
    if re.search(r"hip|knee|shank|ankle|foot", n):
        return side + "leg"
    if side:
        return side + ("hand" if re.search(r"wrist_roll|hand|thumb|index|middle|ring|pinky", n) else "arm")
    return "torso"


def pct(x) -> str:
    return f"{100 * np.mean(x):5.2f}%"


class ClipSet:
    def __init__(self, robot, label: str, lib: str, names: str):
        self.robot, self.label = robot, label
        kin = Kinematics(robot, "cpu")
        self.m = m = MotionLib.load(kin, "cpu", names, lib)
        m.set_ref_cmd()
        self.lo, self.hi = kin.q_lo.numpy(), kin.q_hi.numpy()
        self.q = m.q.numpy()
        self.cid = m.clip_id.numpy()
        self.T = m.num_frames
        self._soles()
        self._self_contacts()

    def _soles(self):
        r, m = self.robot, self.m
        model = mujoco.MjModel.from_xml_path(r.mjcf_path)
        signs = np.array(list(itertools.product((-1, 1), repeat=3)), float)
        corners, centres = [], []
        for f in r.feet:  # the foot's one collision box, in the body frame
            b = model.body(f).id
            (g,) = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                    and (model.geom_contype[g] or model.geom_conaffinity[g])]  # fmt: skip
            Rg = np.empty(9)
            mujoco.mju_quat2Mat(Rg, model.geom_quat[g])
            corners.append(model.geom_pos[g] + (signs * model.geom_size[g]) @ Rg.reshape(3, 3).T)
            centres.append(model.geom_pos[g] - Rg.reshape(3, 3)[:, 2] * model.geom_size[g][2])  # bottom face centre
        fi = [r.tracked_bodies.index(f) for f in r.feet]
        P, R = m.body_p[:, fi].numpy(), m.body_R[:, fi].numpy()  # (T,2,3), (T,2,3,3)
        V, W = m.body_v[:, fi].numpy(), m.body_w[:, fi].numpy()
        off = np.einsum("tfij,fcj->tfci", R, np.stack(corners))  # (T,2,8,3)
        z = P[:, :, None, 2] + off[..., 2]
        k = z.argmin(-1)
        self.low = z.min(-1)  # (T,2) lowest corner of each foot
        rk = np.take_along_axis(off, k[..., None, None].repeat(3, -1), 2)[:, :, 0]
        self.slide = np.linalg.norm((V + np.cross(W, rk))[..., :2], axis=-1)  # lowest corner's horizontal speed
        m.set_ref_contact(fi)
        self.contact = m.ref_contact.numpy() > 0
        rc = np.einsum("tfij,fj->tfi", R, np.stack(centres))
        vc = V + np.cross(W, rc)
        self.planted = (np.linalg.norm(vc[..., :2], axis=-1) < PLANT_V) & (np.abs(vc[..., 2]) < PLANT_V)

    def _self_contacts(self):
        r, m = self.robot, self.m
        cm = importlib.import_module(f"robot.{r.name}").get_spec().compile()  # training colliders, no floor
        cd = mujoco.MjData(cm)
        qadr = np.array([cm.jnt_qposadr[cm.joint(n).id] for n in r.joint_names])
        names = [part(cm.body(cm.geom_bodyid[g]).name) for g in range(cm.ngeom)]
        bodies = [cm.body(cm.geom_bodyid[g]).name for g in range(cm.ngeom)]
        p, Rr = m.root_p.numpy().astype(np.float64), m.root_R.numpy().reshape(-1, 9).astype(np.float64)
        self.ncon = np.zeros(self.T, int)
        self.depth = np.zeros(self.T)  # deepest self-contact
        self.pair = np.full(self.T, "", object)  # body pair of the deepest contact
        self.kinds_deep: dict[str, np.ndarray] = {}  # kind -> frames with a contact of that kind deeper than DEEP
        self.pairs_deep: dict[str, int] = {}
        quat = np.empty(4)
        for t in range(self.T):
            mujoco.mju_mat2Quat(quat, Rr[t])
            cd.qpos[:3], cd.qpos[3:7], cd.qpos[qadr] = p[t], quat, self.q[t]
            mujoco.mj_kinematics(cm, cd)
            mujoco.mj_collision(cm, cd)
            self.ncon[t] = n = cd.ncon
            if not n:
                continue
            c = cd.contact
            i = int(np.argmin(c.dist[:n]))
            g1, g2 = c.geom1[i], c.geom2[i]
            self.depth[t] = -c.dist[i]
            self.pair[t] = "/".join(sorted((bodies[g1], bodies[g2])))
            seen = set()
            for k in np.flatnonzero(c.dist[:n] < -DEEP):
                g1, g2 = c.geom1[k], c.geom2[k]
                kd, pr = "/".join(sorted((names[g1], names[g2]))), "/".join(sorted((bodies[g1], bodies[g2])))
                if kd not in seen:
                    self.kinds_deep.setdefault(kd, np.zeros(self.T, bool))[t] = True
                if pr not in seen:
                    self.pairs_deep[pr] = self.pairs_deep.get(pr, 0) + 1
                seen |= {kd, pr}

    def joint(self, pat: str) -> list[int]:
        return [i for i, n in enumerate(self.robot.joint_names) if re.search(pat, n.lower())]

    def report(self, clip_table: bool = True) -> None:
        q, lo, hi, m, names = self.q, self.lo, self.hi, self.m, self.robot.joint_names
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * SOFT
        slo, shi = mid - half, mid + half
        print(f"\n===== {self.label}: {len(m.names)} clips, {self.T} frames ({self.T / 50:.0f} s at 50 Hz)")
        at_lo, at_hi = q - lo < HARD, hi - q < HARD
        print(f"joints within {HARD} rad of a hard limit in >= 0.5% of frames (low / high):")
        for i, n in enumerate(names):
            if (at_lo[:, i] | at_hi[:, i]).mean() >= 0.005:
                print(f"  {n:28s} {pct(at_lo[:, i] | at_hi[:, i])}  ({pct(at_lo[:, i])} / {pct(at_hi[:, i])})  range [{lo[i]:+.3f}, {hi[i]:+.3f}]")
        knee, ap, ar = self.joint(r"knee"), self.joint(r"ankle_pitch"), self.joint(r"ankle_roll")
        print(f"knee below soft {slo[knee[0]]:.3f}: {pct(q[:, knee] < slo[knee])} (L {pct(q[:, knee[0]] < slo[knee[0]])}, "
              f"R {pct(q[:, knee[1]] < slo[knee[1]])}); median {np.median(q[:, knee]):.3f}")
        print(f"ankle pitch below soft {slo[ap[0]]:+.3f}: {pct(q[:, ap] < slo[ap])}, above soft {shi[ap[0]]:+.3f}: {pct(q[:, ap] > shi[ap])}; "
              f"at hard limit {pct(at_lo[:, ap] | at_hi[:, ap])} (low {pct(at_lo[:, ap])}, high {pct(at_hi[:, ap])})")
        print(f"|ankle roll| above soft {shi[ar[0]]:.3f}: {pct(np.abs(q[:, ar] - mid[ar]) > half[ar])}; at hard limit {pct(at_lo[:, ar] | at_hi[:, ar])}")
        same = self.cid[1:] == self.cid[:-1]
        jump = (np.abs(np.diff(q, axis=0)) > JUMP) & same[:, None]
        jc = np.unique(self.cid[1:][jump.any(1)])
        print(f"one-frame jumps > {JUMP} rad: {int(jump.any(1).sum())} frames in {len(jc)} clips"
              + ("; by joint: " + ", ".join(f"{names[i]} {int(jump[:, i].sum())}" for i in np.flatnonzero(jump.any(0))) if jump.any() else ""))
        low = self.low
        print(f"lowest sole corner: median {100 * np.median(low.min(1)):+.2f} cm; below {100 * PEN:.0f} cm {pct(low.min(1) < PEN)} of frames "
              f"(min {100 * low.min():+.1f} cm); both feet above {100 * FLOAT:.0f} cm {pct(low.min(1) > FLOAT)}")
        pl = self.planted
        print(f"planted feet ({pct(pl)} of foot-frames): floating above {100 * FLOAT:.0f} cm {pct(low[pl] > FLOAT)} of them "
              f"(median height {100 * np.median(low[pl]):+.2f} cm)")
        con = self.contact
        s = self.slide[con]
        print(f"slide of the lowest corner in reference contact ({pct(con)} of foot-frames): median {np.median(s):.3f}, "
              f"p95 {np.percentile(s, 95):.3f} m/s; above 0.2 m/s {pct(s > 0.2)}")
        print(f"self-contacts (no floor): frames with any {pct(self.ncon > 0)}, deeper than {100 * DEEP:.0f} cm {pct(self.depth > DEEP)}; "
              f"contacts per frame max {self.ncon.max()}, p99 {np.percentile(self.ncon, 99):.0f}")
        for kd, fr in sorted(self.kinds_deep.items(), key=lambda kv: -kv[1].sum()):
            clips = np.unique(self.cid[fr])
            print(f"  {kd:18s} deeper than {100 * DEEP:.0f} cm in {pct(fr)} of frames, {len(clips)} clips: "
                  + ", ".join(f"{m.names[c]} ({int(fr[self.cid == c].sum())})" for c in clips[np.argsort(-np.bincount(self.cid[fr])[clips])][:6]))
        top = sorted(self.pairs_deep.items(), key=lambda kv: -kv[1])[:8]
        print("  body pairs (frames): " + ", ".join(f"{p} {n}" for p, n in top))
        v = np.linalg.norm(m.root_v[:, :2].numpy(), axis=1)
        wz = np.abs(m.root_w[:, 2].numpy())
        print(f"root horizontal speed p50/p90/p99/max {np.percentile(v, 50):.2f}/{np.percentile(v, 90):.2f}/{np.percentile(v, 99):.2f}/{v.max():.2f} m/s; "
              f"|yaw rate| p99 {np.percentile(wz, 99):.2f} rad/s; root z p1/p50 {np.percentile(m.root_p[:, 2].numpy(), 1):.3f}/{np.median(m.root_p[:, 2].numpy()):.3f} m")
        lo_c, hi_c = np.percentile(m.ref_cmd.numpy(), [0.1, 99.9], axis=0)
        print(f"reference command box (0.1-99.9 pct): vx [{lo_c[0]:+.2f}, {hi_c[0]:+.2f}] vy [{lo_c[1]:+.2f}, {hi_c[1]:+.2f}] m/s, "
              f"yaw [{lo_c[2]:+.2f}, {hi_c[2]:+.2f}] rad/s")
        if not clip_table:
            return
        print(f"{'clip':46s} {'T':>4s} {'knee<s':>6s} {'ankP@':>6s} {'ankR@':>6s} {'jump':>4s} {'pen':>6s} {'float':>6s} "
              f"{'slide95':>7s} {'self>1':>6s} {'ncon':>4s} {'v50':>5s} {'vmax':>5s}  deepest self-contact")
        for c, name in enumerate(m.names):
            f = self.cid == c
            qc, lc = q[f], low[f]
            sc = self.slide[f][self.contact[f]]
            d = int(np.argmax(np.where(f, self.depth, -1)))
            print(f"{name:46s} {f.sum():4d} {pct(qc[:, knee] < slo[knee]):>6s} {pct(at_lo[f][:, ap] | at_hi[f][:, ap]):>6s} "
                  f"{pct(at_lo[f][:, ar] | at_hi[f][:, ar]):>6s} {int(jump[f[1:]].any(1).sum()):4d} {pct(lc.min(1) < PEN):>6s} "
                  f"{pct(lc[self.planted[f]] > FLOAT) if self.planted[f].any() else '   -  ':>6s} "
                  f"{np.percentile(sc, 95) if len(sc) else 0:7.3f} {pct(self.depth[f] > DEEP):>6s} {self.ncon[f].max():4d} "
                  f"{np.median(v[f]):5.2f} {v[f].max():5.2f}  {100 * self.depth[d]:.1f} cm {self.pair[d]}")


def render(sets: list[ClipSet], clips: list[str], out_dir: str, W: int = 640, H: int = 720) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    model = mujoco.MjModel.from_xml_path(sets[0].robot.mjcf_path)  # visual meshes and the MJCF's own floor
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    for fl in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION, mujoco.mjtRndFlag.mjRND_SKYBOX):
        renderer.scene.flags[fl] = 0
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in sets[0].robot.joint_names])
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    z0 = float(sets[0].robot.robot_cfg().init_state.pos[2])
    for clip in clips:
        writers = []
        for s in sets:
            os.makedirs(os.path.join(out_dir, s.label), exist_ok=True)
            writers.append(imageio.get_writer(os.path.join(out_dir, s.label, f"{clip}.mp4"), fps=50, codec="libx264", quality=8))
        if len(sets) == 2:
            os.makedirs(os.path.join(out_dir, "side_by_side"), exist_ok=True)
            writers.append(imageio.get_writer(os.path.join(out_dir, "side_by_side", f"{clip}.mp4"), fps=50, codec="libx264", quality=8))
        ranges = []
        for s in sets:
            c = s.m.names.index(clip)
            ranges.append((int(s.m.clip_start[c]), int(s.m.clip_len[c])))
        L = max(n for _, n in ranges)
        R0 = sets[0].m.root_R[ranges[0][0]].numpy()
        az = np.degrees(np.arctan2(R0[1, 0], R0[0, 0])) + 140.0  # front-left view of the clip's first heading
        quat = np.empty(4)
        for k in range(L):
            ims = []
            for s, (a, n) in zip(sets, ranges):
                t = a + min(k, n - 1)
                mujoco.mju_mat2Quat(quat, s.m.root_R[t].numpy().reshape(-1).astype(np.float64))
                data.qpos[:3], data.qpos[3:7], data.qpos[qadr] = s.m.root_p[t].numpy(), quat, s.q[t]
                mujoco.mj_kinematics(model, data)
                cam = mujoco.MjvCamera()
                cam.distance, cam.elevation, cam.azimuth = 3.0 * z0, -16.0, az
                cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.55 * z0]
                renderer.update_scene(data, camera=cam)
                im = Image.fromarray(renderer.render())
                dr = ImageDraw.Draw(im)
                dr.text((10, 8), f"{clip}\n{s.label}   t = {min(k, n - 1) / 50:5.2f} s", fill="white", font=font, spacing=2)  # on the black sky
                if s.depth[t] > DEEP:
                    dr.text((10, H - 28), f"self-penetration {100 * s.depth[t]:.1f} cm: {s.pair[t]}", fill=(255, 60, 60), font=font, stroke_width=2, stroke_fill="black")
                ims.append(np.asarray(im))
            for w, im in zip(writers, ims):
                w.append_data(im)
            if len(sets) == 2:
                writers[-1].append_data(np.concatenate(ims, 1))
        for w in writers:
            w.close()
        print(f"rendered {clip}: {L} frames", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, choices=["t1", "t2", "t2_measured"])
    ap.add_argument("--set", action="append", required=True, help="LABEL=path/to/lib_all.npz (repeatable, one robot)")
    ap.add_argument("--list", help="clip names to check, one per line (default: the keep.txt next to each library)")
    ap.add_argument("--video", nargs="*", default=[], help="clips to render")
    ap.add_argument("--video_dir", default="logs/t2_pilot")
    ap.add_argument("--no_clip_table", action="store_true")
    a = ap.parse_args()
    robot = importlib.import_module(f"robot.{a.robot}").SPEC  # robot.spec.get_robot knows only the T1
    sets = []
    for s in a.set:
        label, lib = s.split("=", 1)
        sets.append(ClipSet(robot, label, lib, a.list or os.path.join(os.path.dirname(lib), "keep.txt")))
        sets[-1].report(not a.no_clip_table)
    if a.video:
        render(sets, a.video, a.video_dir)


if __name__ == "__main__":
    main()
