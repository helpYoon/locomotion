"""Pack Booster T2 clips (soma 120 fps CSVs: Frame, root cm and xyz Euler deg, the 31 T2 joints in deg) into
<out_dir>/lib_all.npz + keep.txt + manifest.csv, the library format src/motion.py loads. utils/prepare_t1.py with the
T2 joints: same 120 fps -> 50 Hz resampling and packing, no name or feasibility filter.

SRC is a text file of clip CSV paths (one per line), or a directory whose CSVs (searched recursively) are all packed.

usage: python -m utils.prepare_t2 dataset/bones_seed/soma_uniform/csv_booster_t2_pilot075 data/t2_pilot075
"""

import argparse
import csv
import io
from pathlib import Path

import numpy as np

from robot.t2 import T2_JOINT_NAMES
from utils.prepare_t1 import DT, FPS, SRC_FPS, _slerp

J = len(T2_JOINT_NAMES)  # 31


def parse_csv(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """120 fps CSV -> 50 Hz root position (m), quaternion wxyz, joints (rad), float64."""
    from scipy.spatial.transform import Rotation

    raw = np.loadtxt(io.BytesIO(data), delimiter=",", skiprows=1, ndmin=2)
    p = raw[:, 1:4] / 100.0
    q_xyzw = Rotation.from_euler("xyz", raw[:, 4:7], degrees=True).as_quat()
    q = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], 1)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)  # hemisphere continuity is handled by slerp; keep w >= 0
    j = np.deg2rad(raw[:, 7 : 7 + J])
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
    print(f"{len(rows)} clips, kept {sum(r[1] for r in keep) / FPS / 3600:.2f} h")
    N = sum(r[1] for r in rows)
    P, Q, Jq = np.empty((N, 3), np.float32), np.empty((N, 4), np.float32), np.empty((N, J), np.float32)
    by_shard: dict[str, list] = {}
    for i, r in enumerate(rows):
        by_shard.setdefault(r[3], []).append(i)
    off = np.cumsum([0] + [r[1] for r in rows])
    for stem, idx in by_shard.items():
        with np.load(shards / f"{stem}.npz") as z:
            for i in idx:
                n, a = rows[i][0], off[i]
                P[a : a + rows[i][1]], Q[a : a + rows[i][1]], Jq[a : a + rows[i][1]] = z[f"{n}/p"], z[f"{n}/q"], z[f"{n}/j"]
    np.savez(out / "lib_all.npz", names=np.array([r[0] for r in rows]), starts=off[:-1], lengths=np.array([r[1] for r in rows]), p=P, q=Q, j=Jq)
    print(f"saved {out / 'lib_all.npz'}: {N} frames, {(P.nbytes + Q.nbytes + Jq.nbytes) / 1e9:.3f} GB")


def main(src: str, out: str, shard_size: int = 2000) -> None:
    out = Path(out)
    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    if Path(src).is_dir():
        paths = sorted(Path(src).rglob("*.csv"))
    else:
        paths = [Path(s.strip()) for s in open(src) if s.strip()]
    names = [p.stem for p in paths]
    assert paths and len(set(names)) == len(names), "clip names (file stems) must be unique"
    for k in range(0, len(paths), shard_size):
        rows, arrs = [], {}
        for path, name in zip(paths[k : k + shard_size], names[k : k + shard_size]):
            data = path.read_bytes()
            hdr = data[: data.index(b"\n")].decode().strip().split(",")
            assert len(hdr) == 7 + J and hdr[7:] == T2_JOINT_NAMES, path
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
    ap.add_argument("src", help="text file of clip CSV paths (one per line), or a directory of clip CSVs")
    ap.add_argument("out_dir", help="output directory: lib_all.npz, keep.txt, manifest.csv and the shards")
    a = ap.parse_args()
    main(a.src, a.out_dir)
