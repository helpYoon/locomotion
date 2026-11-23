"""Pack bfm_wm's Booster T1 clips (120 fps CSVs listed one path per line) into <out_dir>/lib_all.npz + keep.txt, the
library format src/motion.py loads, with the resampling and packing of BlinkMimic 2's prepare_bones.py (120 fps ->
50 Hz; no name or feasibility filter: the list is already bfm_wm's filtered set).

usage: python -m utils.prepare_t1 /path/to/clips_filtered_edited_v2.txt data/t1
"""

import argparse
import csv
import io
from pathlib import Path

import numpy as np

from robot.t1 import T1_JOINT_NAMES

SRC_FPS = 120.0
DT = 0.02  # = motion.CONTROL_DT
FPS = 50


def _slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    dot = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), None, 1.0)
    th = np.arccos(dot)
    s = np.sin(th)
    t = t[:, None]
    small = s < 1e-6
    w0 = np.where(small, 1 - t, np.sin((1 - t) * th) / (s + 1e-12))
    w1 = np.where(small, t, np.sin(t * th) / (s + 1e-12))
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def parse_csv(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """120 fps CSV -> 50 Hz root position (m), quaternion wxyz, joints (rad), float64."""
    from scipy.spatial.transform import Rotation

    raw = np.loadtxt(io.BytesIO(data), delimiter=",", skiprows=1, ndmin=2)
    p = raw[:, 1:4] / 100.0
    q_xyzw = Rotation.from_euler("xyz", raw[:, 4:7], degrees=True).as_quat()
    q = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], 1)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)  # hemisphere continuity is handled by slerp; keep w >= 0
    j = np.deg2rad(raw[:, 7:36])
    T = raw.shape[0]
    n = int(np.floor((T - 1) / SRC_FPS / DT)) + 1
    t = np.arange(n) * DT * SRC_FPS
    i0 = np.minimum(np.floor(t).astype(np.int64), T - 2)
    a = np.clip(t - i0, 0.0, 1.0)
    p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
    j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
    q = _slerp(q[i0], q[i0 + 1], a)
    return p, q, j


def pack(shards: Path, out: Path) -> None:
    """Merge shard manifests (manifest.csv), write keep.txt (sorted names) and lib_all.npz (all clips, sorted)."""
    rows = []
    for f in sorted(shards.glob("shard_*.csv")):
        with open(f, newline="") as fh:
            rows += [(r["name"], int(r["frames"]), r["reason"], f.stem) for r in csv.DictReader(fh)]
    rows.sort()
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "frames", "reason"])
        w.writerows([r[:3] for r in rows])
    keep = [r for r in rows if not r[2]]
    (out / "keep.txt").write_text("".join(r[0] + "\n" for r in keep))
    n_name = sum(1 for r in rows if r[2].startswith("name:"))
    reasons: dict[str, int] = {}
    for r in rows:
        if r[2] and not r[2].startswith("name:"):
            reasons[r[2]] = reasons.get(r[2], 0) + 1
    print(f"raw {len(rows)} -> name filter {len(rows) - n_name} -> feasibility {len(keep)}")
    print(f"feasibility removed {reasons}; kept {sum(r[1] for r in keep) / FPS / 3600:.2f} h")
    # one packed library of every clip (sorted by name); a subset is a list of names
    N = sum(r[1] for r in rows)
    P, Q, J = np.empty((N, 3), np.float32), np.empty((N, 4), np.float32), np.empty((N, 29), np.float32)
    by_shard: dict[str, list] = {}
    for i, r in enumerate(rows):
        by_shard.setdefault(r[3], []).append(i)
    off = np.cumsum([0] + [r[1] for r in rows])
    for stem, idx in by_shard.items():
        with np.load(shards / f"{stem}.npz") as z:
            for i in idx:
                n, a = rows[i][0], off[i]
                P[a : a + rows[i][1]], Q[a : a + rows[i][1]], J[a : a + rows[i][1]] = z[f"{n}/p"], z[f"{n}/q"], z[f"{n}/j"]
    np.savez(out / "lib_all.npz", names=np.array([r[0] for r in rows]), starts=off[:-1], lengths=np.array([r[1] for r in rows]), p=P, q=Q, j=J)
    print(f"saved {out / 'lib_all.npz'}: {N} frames, {(P.nbytes + Q.nbytes + J.nbytes) / 1e9:.2f} GB")


def main(list_path: str, out: str, shard_size: int = 2000) -> None:
    out = Path(out)
    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    paths = [Path(s.strip()) for s in open(list_path) if s.strip()]
    names = [p.stem for p in paths]
    assert len(set(names)) == len(names), "clip names (file stems) must be unique"
    for k in range(0, len(paths), shard_size):
        rows, arrs = [], {}
        for path, name in zip(paths[k : k + shard_size], names[k : k + shard_size]):
            data = path.read_bytes()
            hdr = data[: data.index(b"\n")].decode().strip().split(",")
            assert len(hdr) == 36 and hdr[7:] == T1_JOINT_NAMES, path
            p, q, j = parse_csv(data)
            arrs[f"{name}/p"], arrs[f"{name}/q"], arrs[f"{name}/j"] = (x.astype(np.float32) for x in (p, q, j))
            rows.append((name, p.shape[0], ""))
        stem = shards / f"shard_{k // shard_size:04d}"
        np.savez(f"{stem}.npz", **arrs)
        with open(f"{stem}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "frames", "reason"])
            w.writerows(rows)
        print(f"{k + len(rows)}/{len(paths)}", flush=True)
    pack(shards, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("list", help="text file of clip CSV paths, one per line")
    ap.add_argument("out_dir", help="output directory: lib_all.npz, keep.txt, manifest.csv and the shards")
    a = ap.parse_args()
    main(a.list, a.out_dir)
