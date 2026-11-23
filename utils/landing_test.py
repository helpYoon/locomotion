"""The landing test (spec 10, 11) of a --land_frac checkpoint, in the training simulator on a GPU (evaluation
conditions: no randomization, noise or pushes, the fixed evaluation delay), in two parts.

Default stance: every command of utils/landing_score.py's grid is sent from the default stance, one world each, and
scored: how close the foot lands, whether it was lifted, whether the planted foot stayed, whether the robot kept its
stance until the command, kept its balance and stood still; prints the summary and the spec's pass line.
Library stances: free rows start on hold frames of the library the checkpoint was trained on and are given the
simulator's own commands one after another, so that a later command starts from the stance the one before landed in;
prints whether a robot keeps the stance it was reset to until its anchors latch, the errors by the command's place in
its episode and by its travel, and the pass line (no fall, the stance kept in 95% of the episodes).
PASS is both parts' pass.

usage: python -m utils.landing_test --ckpt logs/<run>/ckpt_004000.pt --out logs/<run>/landing_test.json
       python -m utils.landing_test --ckpt ... --out ... --record 8   # also <out>.npz: the first 8 worlds' states
       python -m utils.landing_test --ckpt ... --no_library          # the default-stance part alone
"""

import argparse
import gc
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--out", help="write the per-command numbers and the summary here (JSON)")
parser.add_argument("--record", type=int, default=0, metavar="N", help="save the first N worlds' root pose, joints and "
                    "targets to the JSON's name with .npz (needs --out)")  # fmt: skip
parser.add_argument("--worlds", type=int, default=256, help="worlds simulated at once (the grid goes in chunks)")
parser.add_argument("--hold_s", type=float, default=3.5, help="seconds of hold after the longest clock")
parser.add_argument("--no_library", action="store_true", help="skip the second part (library stances)")
parser.add_argument("--library_worlds", type=int, default=2048, help="the second part's landing robots")
parser.add_argument("--library_steps", type=int, default=750, help="the second part's control steps")
args = parser.parse_args()
if args.record and not args.out:
    parser.error("--record needs --out")

import numpy as np
import torch

from utils.landing_rig import LibraryRig, Rig  # after the arguments: it starts Warp
from utils.landing_score import command_grid, jsonable, score, score_library

# the pass line (spec 11): name, summary key, direction, limit, format
LINE = [("falls", "falls", "<=", 0, "{:.0f}"), ("stance held", "stance_ok", ">=", 0.95, "{:.1%}"),
        ("arrived", "arrived", ">=", 0.95, "{:.1%}"),
        ("median error [cm]", "err_cm_median", "<=", 2.0, "{:.2f}"),
        ("median error [deg]", "err_deg_median", "<=", 3.0, "{:.2f}"),
        ("lifted", "lifted", ">=", 0.95, "{:.1%}"), ("planted sole kept", "planted_ok", ">=", 0.90, "{:.1%}"),
        ("still within 3 s", "still_ok", ">=", 0.95, "{:.1%}")]  # fmt: skip
# and the second part's (library stances)
LIB_LINE = [("falls", "falls", "<=", 0, "{:.0f}"), ("stance kept", "stance_kept", ">=", 0.95, "{:.1%}")]


def lines(rows: list, s: dict) -> None:
    """Print a pass line's rows (LINE or LIB_LINE) with the summary s's values."""
    print(f"{'':<22}{'value':>10}{'line':>10}")
    for name, key, op, lim, fmt in rows:
        ok = s[key] <= lim if op == "<=" else s[key] >= lim
        print(f"{name:<22}{fmt.format(s[key]):>10}{op + ' ' + fmt.format(lim):>10}  {'ok' if ok else 'FAIL'}")


def write(lib: dict | None) -> None:
    """Write the JSON document (--out): the first part's numbers, and the second part's summary lib (None before it
    has run, or with --no_library). Written after each part, so that a failed second part keeps the first's."""
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    doc = {"ckpt": args.ckpt, "iteration": iteration, "reset_stance": pose.tolist(), "summary": jsonable(s),
           "library": jsonable(lib), "grid": {k: v.tolist() for k, v in grid.items()},
           "per": jsonable({k: v.tolist() for k, v in per.items()})}  # fmt: skip
    Path(args.out).write_text(json.dumps(doc, indent=1, allow_nan=False))


def groups(title: str, rows: dict[str, dict]) -> None:
    """Print one table of command groups (a summary's by_travel, by_direction or by_place)."""
    print(f"\n{title:<16}{'n':>5}{'arrived':>9}{'err cm med':>12}{'p95':>8}{'err deg med':>13}{'p95':>8}")
    for name, g in rows.items():
        if g["n"]:
            print(f"{name:<16}{g['n']:>5}{g['arrived']:>9.1%}{g['err_cm_median']:>12.2f}{g['err_cm_p95']:>8.2f}"
                  f"{g['err_deg_median']:>13.2f}{g['err_deg_p95']:>8.2f}")  # fmt: skip


rig = Rig(args.ckpt, args.worlds)
pose = rig.reset_stance()  # every world is reset to the same clip frame: the commands are built from that stance, and
# each world latches its own target from its own soles when the command is issued
grid = command_grid(rig.cfg, pose)
K = len(grid["swing"])
print(f"{args.ckpt} (iteration {rig.iteration}): {K} commands, {(grid['direction'] == -1).sum().item()} pure turns")
print("reset stance (x y yaw): left " + " ".join(f"{v:+.4f}" for v in pose[0].tolist())
      + ", right " + " ".join(f"{v:+.4f}" for v in pose[1].tolist()))  # fmt: skip
log = rig.run(grid, args.hold_s, args.record)
out = score(log, grid, rig.cfg)
per, s = out["per"], out["summary"]
iteration, lib = rig.iteration, None
if args.out:
    write(lib)

print()
lines(LINE, s)
print(f"{'p95 error':<22}{s['err_cm_p95']:>7.2f} cm{s['err_deg_p95']:>7.2f} deg   (median 1 s later: "
      f"{float(per['err_cm_1s'].median()):.2f} cm)")  # fmt: skip
relift = int((per["relifts"] > 0).sum())
print(f"{'relifted after':<22}{relift:>10}   still_s median {float(per['still_s'].median()):.2f}")
print(f"{'scuffing':<22}{s['scuff_share']:>10.1%}   of swing travel with a sole corner on the floor; "
      f"apex median {s['apex_cm_median']:.1f} cm, floor travel median {s['floor_cm_median']:.1f} cm")
groups("by travel", s["by_travel"])
groups("by direction", s["by_direction"])

if not args.no_library:  # the second part: the first rig's simulator is freed before the second one is built
    del rig
    gc.collect()  # the simulator sits in a reference cycle
    torch.cuda.empty_cache()
    lib = score_library(LibraryRig(args.ckpt, args.library_worlds).run(args.library_steps))
    print(f"\nlibrary stances, commands one after another: {lib['commands']} commands in {lib['episodes']} episodes")
    lines(LIB_LINE, lib)
    print(f"{'arrived':<22}{lib['arrived']:>10.1%}   within 5 cm and 0.10 rad")
    print(f"{'median error':<22}{lib['err_cm_median']:>7.2f} cm{lib['err_deg_median']:>7.2f} deg   (p95 "
          f"{lib['err_cm_p95']:.2f} cm, {lib['err_deg_p95']:.2f} deg)")  # fmt: skip
    print(f"{'moved in SETTLE':<22}{lib['moved_cm_median']:>7.2f} cm   median over the episodes (p95 "
          f"{lib['moved_cm_p95']:.2f} cm)")  # fmt: skip
    groups("by place", lib["by_place"])
    groups("by travel", lib["by_travel"])
ok = s["pass"] and (lib is None or lib["pass"])
parts = "" if lib is None else f" (default stance: {s['pass']}, library stances: {lib['pass']})"
print(f"\nPASS: {ok}{parts}")

if args.out:
    write(lib)
    print(f"wrote {args.out}")
if args.record:
    n = args.record
    sel = {k: v[:n].numpy() for k, v in {**grid, "target": log["target"], "anchor": log["anchor"]}.items()}
    path = Path(args.out).with_suffix(".npz")
    np.savez(path, t_cmd=log["t_cmd"], root_p=log["root_p"].numpy(), root_quat=log["root_quat"].numpy(),
             q=log["q"].numpy(), pose=log["pose"][:, :n].numpy(), h=log["h"][:, :n].numpy(),
             zc=log["zc"][:, :n].numpy(), **sel)  # fmt: skip
    print(f"wrote {path}")
