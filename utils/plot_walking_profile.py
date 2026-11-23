"""How much reference motion the dataset has per velocity-command cell (seconds), as 2D heatmaps of the root's
heading-frame velocity: (a) v_y vs v_x, (b) w_z vs v_x, (c) w_z vs v_y, each summed over the third component.

Velocities are centered differences over --window seconds (about one stride by default, so in-stride sway and pelvis
yaw oscillation don't read as commands); --window 0 gives per-frame velocities. The grid is the command grid of the
reference figure (v_x -1.5..2 by 0.25, v_y -1.2..1.2 by 0.2, w_z -3..3 by 0.5 at cell centers).

usage (from the repo root):
    uv run python utils/plot_walking_profile.py [CSV or folder ...] [--window 1.0] [--out docs/figures/walking_profile.png]
    (no arguments: every clip of the BONES-SEED T1 retarget)
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap, LogNorm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils import soma_csv  # noqa: E402

EDGES = (  # cell edges of v_x, v_y [m/s] and w_z [rad/s]; half-integer multiples keep them exactly symmetric about 0
    (np.arange(16) - 6.5) * 0.25,
    (np.arange(14) - 6.5) * 0.2,
    (np.arange(14) - 6.5) * 0.5,
)


def clip_histogram(path: str, window: float) -> tuple[np.ndarray, int]:
    """Seconds of the clip per (v_x, v_y, w_z) cell (frames outside the grid are left out), and its frame count."""
    torch.set_num_threads(1)  # one process per clip already
    _, p, R, _ = soma_csv.load(path)
    p, R = p.numpy(), R.numpy()
    if len(p) < 2:
        return np.zeros([len(e) - 1 for e in EDGES]), len(p)
    yaw = np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0]))  # heading of the trunk's forward (x) axis
    t, h = np.arange(len(p)), max(1, round(window * soma_csv.FPS / 2))
    i0, i1 = np.maximum(t - h, 0), np.minimum(t + h, len(p) - 1)  # the window shrinks at the clip ends
    dt = (i1 - i0) / soma_csv.FPS
    vw = (p[i1, :2] - p[i0, :2]) / dt[:, None]
    mid = 0.5 * (yaw[(i0 + i1) // 2] + yaw[(i0 + i1 + 1) // 2])  # heading at the window's center, also where it shrinks
    c, s = np.cos(mid), np.sin(mid)
    v = np.stack([c * vw[:, 0] + s * vw[:, 1], -s * vw[:, 0] + c * vw[:, 1], (yaw[i1] - yaw[i0]) / dt], -1)
    return np.histogramdd(v, bins=EDGES, weights=np.full(len(v), 1 / soma_csv.FPS))[0], len(p)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("motions", nargs="*", default=[soma_csv.DATASET], help="CSV files and/or folders")
    parser.add_argument("--window", type=float, default=1.0, help="velocity averaging window [s]; 0 = per frame")
    parser.add_argument("--out", default=os.path.join(ROOT, "docs", "figures", "walking_profile.png"))
    args = parser.parse_args()
    clips = soma_csv.find(args.motions)
    if not clips:
        sys.exit(f"no CSV files in {args.motions}")

    with ProcessPoolExecutor() as ex:
        hists, frames = zip(*ex.map(partial(clip_histogram, window=args.window), clips, chunksize=32))
    H, total = sum(hists), sum(frames) / soma_csv.FPS
    print(f"{len(clips)} clips, {total / 3600:.2f} h; {1 - H.sum() / total:.2%} of frames outside the grid; "
          f"busiest (v_x, v_y) cell {H.sum(2).max():.0f} s")

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 13})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    panels = [  # (x edges, y edges, cells[x, y], title)
        (EDGES[0], EDGES[1], H.sum(2), r"(a) $v_y$ vs. $v_x$ (m/s)"),
        (EDGES[0], EDGES[2], H.sum(1), r"(b) $\omega_z$ (rad/s) vs. $v_x$ (m/s)"),
        (EDGES[1], EDGES[2], H.sum(0), r"(c) $\omega_z$ (rad/s) vs. $v_y$ (m/s)"),
    ]
    vmax = 10 ** np.ceil(np.log10(max(cells.max() for *_, cells, _ in panels)))
    norm = LogNorm(vmin=1 / soma_csv.FPS, vmax=vmax)  # down to a single frame
    cmap = ListedColormap(plt.get_cmap("Greens")(np.linspace(0.12, 1.0, 256)))  # lightest data still reads as green
    for ax, (xe, ye, cells, title) in zip(axes, panels):
        # empty cells stay blank so "no data" reads differently from "a little data"
        mesh = ax.pcolormesh(xe, ye, np.ma.masked_equal(cells.T, 0), cmap=cmap, norm=norm,
                             edgecolors="white", linewidth=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_box_aspect(1)
        ax.axhline(0, color="0.6", lw=0.6, zorder=0)
        ax.axvline(0, color="0.6", lw=0.6, zorder=0)
        ax.tick_params(color="0.5")
        for side in ax.spines.values():
            side.set_color("0.5")
    fig.colorbar(mesh, ax=axes, shrink=0.8, pad=0.02, label="Reference-motion data per cell (s)")
    fig.text(0.01, -0.02, f"{len(clips)} clips, {total / 3600:.1f} h. Root velocity in the heading frame, "
             f"{args.window:g} s window; each panel sums over the third component.", color="0.4", fontsize=10)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
