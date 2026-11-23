"""Copy a clip library and add slowed copies of its slow walking clips: every clip with "walk" in its name (and none of
run, jog, jump, hop, skip, leap: a slowed flight phase would hang in the air) whose mean planar speed lies in
[--lo, --hi) m/s gets a copy played back at --factor of its speed, the time stretched by 1 / factor and the poses
unchanged, named <clip>_slow<factor x 100>. Mean planar speed: the root's planar displacement over 0.5 s windows.

The T2 library moves 1.414 times faster than the T1's (retarget scale 0.80 against 0.566, the same clips and timing),
so slow commands find few references, and the T2 policies step at about twice the clips' cadence there. The copies
add slow references with a human cadence.

usage: python -m utils.slow_clips data/t2_untwist data/t2_slow [--factor 0.7] [--lo 0.3] [--hi 0.6] [--share 0.5]
"""

import argparse
import re
import zlib
from pathlib import Path

import numpy as np

from utils.prepare_t1 import DT, _slerp

WIN = 25  # frames of the speed windows (0.5 s)


def stretch(p: np.ndarray, q: np.ndarray, j: np.ndarray, factor: float):
    """The clip (50 Hz root position, quaternion wxyz, joints) played back at ``factor`` of its speed, at 50 Hz."""
    T = len(p)
    t = np.arange(int(np.floor((T - 1) / factor + 1e-9)) + 1) * factor
    i0 = np.minimum(np.floor(t).astype(np.int64), T - 2)
    a = np.clip(t - i0, 0.0, 1.0)
    lin = lambda x: x[i0] * (1 - a[:, None]) + x[i0 + 1] * a[:, None]  # noqa: E731
    return lin(p), _slerp(q[i0], q[i0 + 1], a), lin(j)


def mean_speed(p: np.ndarray) -> float:
    """Mean planar root speed [m/s] over 0.5 s windows (0 for a clip shorter than one window)."""
    if len(p) <= WIN:
        return 0.0
    return float(np.linalg.norm(p[WIN:, :2] - p[:-WIN, :2], axis=1).mean() / (WIN * DT))


def slowed(name: str, speed: float, lo: float, hi: float, share: float = 1.0) -> bool:
    """Whether a clip gets a slowed copy; ``share`` keeps that fraction of the qualifying clips, a clip and its mirror
    (<name>_M) together (by a hash of the name)."""
    n = name.lower()
    pick = zlib.crc32(name.removesuffix("_M").encode()) % 1000 < share * 1000
    return pick and "walk" in n and not re.search("run|jog|jump|hop|skip|leap", n) and lo <= speed < hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--factor", type=float, default=0.7)
    ap.add_argument("--lo", type=float, default=0.3)
    ap.add_argument("--hi", type=float, default=0.6)
    ap.add_argument("--share", type=float, default=0.5,
                    help="fraction of the qualifying clips copied: 0.5 brings the T2's 0.2-0.5 m/s walking to about the "
                         "T1 library's hours there (all of them would double it)")
    a = ap.parse_args()
    z = np.load(a.src / "lib_all.npz", mmap_mode="r")
    names, starts, lengths = [str(n) for n in z["names"]], z["starts"], z["lengths"]
    P, Q, Jq = (np.asarray(z[k]) for k in ("p", "q", "j"))
    keep = [s.strip() for s in open(a.src / "keep.txt") if s.strip()]
    kept = set(keep)
    tag = f"_slow{round(a.factor * 100):03d}"
    rows = [(n, P[s : s + m], Q[s : s + m], Jq[s : s + m]) for n, s, m in zip(names, starts, lengths)]
    extra = []
    for n, p, q, j in rows:
        if n in kept and slowed(n, mean_speed(p), a.lo, a.hi, a.share):
            ps, qs, js = stretch(p.astype(np.float64), q.astype(np.float64), j.astype(np.float64), a.factor)
            extra.append((n + tag, ps.astype(np.float32), qs.astype(np.float32), js.astype(np.float32)))
    rows += extra
    a.out.mkdir(parents=True, exist_ok=False)
    L = np.array([len(r[1]) for r in rows])
    np.savez(a.out / "lib_all.npz", names=np.array([r[0] for r in rows]), starts=np.cumsum(np.r_[0, L[:-1]]), lengths=L,
             p=np.concatenate([r[1] for r in rows]), q=np.concatenate([r[2] for r in rows]),
             j=np.concatenate([r[3] for r in rows]))  # fmt: skip
    (a.out / "keep.txt").write_text("".join(n + "\n" for n in keep + [r[0] for r in extra]))
    print(f"{len(extra)} slowed copies ({sum(len(r[1]) for r in extra) * DT / 3600:.2f} h) added to {len(keep)} clips "
          f"({int(lengths.sum()) * DT / 3600:.2f} h): {a.out}")


if __name__ == "__main__":
    main()
