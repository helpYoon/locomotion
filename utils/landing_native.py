"""The landing test's first part (utils/landing_test.py: every command of the grid, each from the default stance) on
an ONNX export of the landing interface (utils/export_landing.py) in native MuJoCo, on the CPU, scored by the code
that scores the training simulator's rollouts, so that the two are read side by side. With --second, a second step:
one second after each command's deadline the other foot is brought beside the one that moved, and those second
commands are scored (the use is one or two steps into a stance, then standing). With --walk, the hand-over from
walking: each rollout first walks and stops under a velocity export (the T1's walking policy), the landing model takes over,
and the commands are the grid's moves from the stance the robot then stands in.

The rollouts run in utils/play_mujoco.py's own environment (its landtest mode, started from here with `uv run`): the
robot stands in the default pose for a second, is given its command or commands, and holds; root poses and joints are
recorded. This module builds the commands (utils/landing_score.py's grid from the stance the robot is reset to),
rebuilds the soles from the recording with the trainer's kinematics, and scores them (utils.landing_score.score).

Besides the simulator, three things differ from the training simulator's test: the model runs in float32 (the trainer's
test under training's autocast); each rollout's actuator lags are drawn at random (--seed), where the trainer's
evaluation uses one fixed delay (--lag 5 sets it); and a command's deadline is the clock the model looked up when the
command arrived, from the stance of that moment, where the trainer's test looks it up from the stance at the reset (a
step apart for some commands).

usage: python -m utils.landing_native --onnx logs/<run>/ckpt_004000_land.onnx --sim logs/<run>/landing_test.json
       python -m utils.landing_native --onnx logs/<run>/ckpt_004000_land.onnx --second
       python -m utils.landing_native --onnx logs/<run>/ckpt_004000_land.onnx --walk logs/land_r4_d10_al2_s3/ckpt_004000_both.onnx
       (writes logs/<run>/ckpt_004000_land_native.json, ..._native_second.json or ..._native_walk.json, and the
       rollouts beside it as .npz)
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from robot.kinematics import Kinematics
from robot.t1 import SPEC
from src import landing
from src.landing import YAW_M, LandingCfg, _dist, compose, relative, wrap
from utils.landing_score import (ONE_S, REPO, STANCE_A, STANCE_D, STILL_QD, STILL_V, TOL, _group, command_grid,
                                 jsonable, score)  # fmt: skip
from utils.rotations import quat_to_mat

# the summary's rows: name, key, format
ROWS = [("falls", "falls", "{:.0f}"), ("stance held", "stance_ok", "{:.1%}"), ("arrived", "arrived", "{:.1%}"),
        ("median error [cm]", "err_cm_median", "{:.2f}"), ("95th percentile [cm]", "err_cm_p95", "{:.2f}"),
        ("median error [deg]", "err_deg_median", "{:.2f}"), ("95th percentile [deg]", "err_deg_p95", "{:.2f}"),
        ("lifted", "lifted", "{:.1%}"), ("planted sole kept", "planted_ok", "{:.1%}"),
        ("still within 3 s", "still_ok", "{:.1%}"), ("stance kept after", "kept_after", "{:.1%}"),
        ("scuffing", "scuff_share", "{:.1%}"),
        ("apex median [cm]", "apex_cm_median", "{:.1f}")]  # fmt: skip
FEET = [SPEC.tracked_bodies.index(b) for b in SPEC.feet]
# --walk: the velocity commands walked before the stop, (vx, vy) [m/s] in the heading frame and the yaw rate [rad/s]
# (standing, then the commands of utils/play_mujoco.py's video), and the seconds each is walked: 7 control steps
# apart, so that the stop command arrives at four points of the gait
WALKS = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, -0.5, 0.0), (0.4, 0.0, 1.0),
         (0.0, 0.0, -1.5), (-0.6, 0.0, 0.0))  # fmt: skip
WALK_S = (3.0, 3.14, 3.28, 3.42)


def sole_log(root_p: torch.Tensor, root_quat: torch.Tensor, q: torch.Tensor, swing: torch.Tensor, c: torch.Tensor,
             t_cmd: int) -> dict:  # fmt: skip
    """The sole entries of utils.landing_score.score's log from a recording: root positions (T, K, 3), wxyz
    quaternions (T, K, 4) and joints (T, K, J) of K rollouts whose command (swing (K,), c (K, 3)) was issued on state
    t_cmd. Returns pose (T, K, 2, 3), h and zc (T, K, 2) by the trainer's kinematics (MDP.sole_kin), and target (K, 3)
    and anchor (K, 2, 3) as utils/landing_rig.py's: the soles of state t_cmd, the target composed on the planted one."""
    T, K, J = q.shape
    kin, z = Kinematics(SPEC, "cpu"), torch.zeros(T * K, 3)
    bodies = kin.forward(root_p.reshape(-1, 3).float(), quat_to_mat(root_quat.reshape(-1, 4).float()), z, z,
                         q.reshape(-1, J).float(), torch.zeros(T * K, J))  # fmt: skip
    s = landing.soles(*(x[:, FEET] for x in bodies))
    out = {k: s[k].reshape(T, K, *s[k].shape[1:]) for k in ("pose", "h", "zc")}
    anchor = out["pose"][t_cmd].clone()
    target = compose(anchor[torch.arange(K), 1 - swing.long()], c)
    return {**out, "target": target, "anchor": anchor, "t_cmd": t_cmd}


def follow_commands(cfg: LandingCfg, pose: torch.Tensor, grid: dict) -> dict:
    """A second command for each command of the grid (swing (K,), c (K, 3)): the other sole is brought back beside the
    one that moved, to the pose it has, seen from that sole, in the stance ``pose`` (2, 3) the robot was reset to.
    Returns swing (K,), c (K, 3) in the moved sole's frame, travel (K,) [m] from where the other sole stands if the
    first command landed exactly, and ok (K,) bool: that travel is one the sampler sends (LandingCfg.travel)."""
    swing = 1 - grid["swing"]
    home = torch.stack((relative(pose[0], pose[1]), relative(pose[1], pose[0])))  # each sole in the other's frame
    c = home[swing.long()]
    cur = relative(torch.zeros_like(grid["c"]), grid["c"])  # the sole that stayed, seen from the landed one
    travel = _dist(c[:, :2] - cur[:, :2]) + YAW_M * wrap(c[:, 2] - cur[:, 2]).abs()
    return {"swing": swing, "c": c, "travel": travel,
            "ok": (travel >= cfg.travel[0] - TOL) & (travel <= cfg.travel[1] + TOL)}  # fmt: skip


def at_command(z: dict, i: int) -> dict:
    """The recorded rollouts z (utils/play_mujoco.py's landtest: arrays (states, rollouts, ...) and t_cmd (rollouts,
    commands), the state each command was sent on) as tensors re-indexed so that every rollout's command i is sent on
    state ONE_S, cut to the rollout that leaves the fewest states; fell is made cumulative first, so a fall before
    command i counts."""
    off = torch.as_tensor(z["t_cmd"])[:, i] - ONE_S
    ix = off[None, :] + torch.arange(len(z["q"]) - int(off.max()))[:, None]
    ar = torch.arange(len(off))
    out = {k: torch.as_tensor(z[k])[ix, ar] for k in ("root_p", "root_quat", "q", "qd_rms", "root_speed")}
    out["fell"] = torch.as_tensor(z["fell"]).int().cummax(0).values.bool()[ix, ar]
    return out


def walks(K: int, seed: int, stop_s: float) -> torch.Tensor:
    """(K, 5): what each of K rollouts does before the landing model takes over: a command of WALKS, the seconds of
    WALK_S it is walked, and stop_s, the seconds stood under the zero command. The WALKS x WALK_S pairs are dealt out
    evenly, in an order drawn from seed."""
    pair = torch.randperm(K, generator=torch.Generator().manual_seed(seed)) % (len(WALKS) * len(WALK_S))
    return torch.cat((torch.tensor(WALKS)[pair % len(WALKS)], torch.tensor(WALK_S)[pair // len(WALKS), None],
                      torch.full((K, 1), stop_s)), 1)  # fmt: skip


def stance_commands(cfg: LandingCfg, grid: dict, pose: torch.Tensor) -> tuple[dict, torch.Tensor]:
    """The grid's K commands (built from the default stance) for K robots that stand in the stances pose (K, 2, 3):
    command i becomes the same move (foot, direction, turn, travel) from stance i, as
    utils.landing_score.command_grid of that stance lists it. Returns the commands (the grid's entries, of the rows
    kept) and keep (K,) bool: the move is one the sampler would send from that stance (the landed stance inside the
    box, the sole rectangles apart)."""
    moves = lambda g: [tuple(round(float(x), 4) for x in r)  # noqa: E731  a move's travel is the same from any stance
                       for r in zip(*(g[k] for k in ("swing", "direction", "turn", "travel")))]  # fmt: skip
    rows, keep = [], torch.zeros(len(pose), dtype=torch.bool)
    for i, move in enumerate(moves(grid)):
        try:
            g = command_grid(cfg, pose[i])
        except AssertionError:  # no command of the grid can be sent from this stance
            continue
        have = moves(g)
        if move in have:
            rows.append({k: v[have.index(move)] for k, v in g.items()})
            keep[i] = True
    return {k: torch.stack([r[k] for r in rows]) for k in grid}, keep


def rollouts(args, rec_path: Path, swing: torch.Tensor, c: torch.Tensor, walk: torch.Tensor | None = None,
             ids: torch.Tensor | None = None) -> dict:  # fmt: skip
    """utils/play_mujoco.py's landtest, in that tool's own environment: K rollouts of the commands swing (K, n) and
    c (K, n, 3), each after its row of walk (K, 5; ``walks``) with the lag draw of rollout ids (K,) if those are
    given. Returns the recording, which is also left at rec_path."""
    cmds = rec_path.with_name(rec_path.stem + "_commands.npz")
    more = {} if walk is None else {"walk": walk.numpy(), "k": ids.numpy()}
    np.savez(cmds, swing=swing.numpy(), c=c.numpy(), **more)
    opts = [] if args.lag is None else ["--lag", str(args.lag)]
    opts += ["--walk_onnx", str(Path(args.walk).resolve())] if args.walk else []
    try:
        subprocess.run(["uv", "run", "utils/play_mujoco.py", "landtest", "--onnx", str(Path(args.onnx).resolve()),
                        "--commands", str(cmds.resolve()), "--out", str(rec_path.resolve()), "--seed", str(args.seed),
                        "--model", args.model, "--jobs", str(args.jobs), *opts], check=True, cwd=REPO)  # fmt: skip
    finally:
        cmds.unlink()
    return dict(np.load(rec_path))


def kept_after(pose: torch.Tensor, dl: torch.Tensor) -> torch.Tensor:
    """(K,) bool: the stance a command landed in is kept, neither sole leaving its pose at the deadline by more than
    STANCE_D or STANCE_A at any state of the 3 s after it (the planted sole stepping after the landing fails it, which
    score's lines do not look at). pose (T, K, 2, 3) the sole poses, dl (K,) the deadline states."""
    ar = torch.arange(pose.shape[1])
    d = pose[dl[None, :] + torch.arange(3 * ONE_S + 1)[:, None], ar] - pose[dl, ar]
    return ((_dist(d[..., :2]) <= STANCE_D) & (wrap(d[..., 2]).abs() <= STANCE_A)).all(-1).all(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", required=True, help="a landing export (utils/export_landing.py)")
    parser.add_argument("--sim", help="the training simulator's test of the same checkpoint (utils/landing_test.py's "
                        "JSON), printed beside the native numbers (not with --second or --walk)")  # fmt: skip
    parser.add_argument("--second", action="store_true", help="two commands per rollout: the grid's, then, one second "
                        "after its deadline, the other foot brought beside the one that moved; the second commands "
                        "are scored")  # fmt: skip
    parser.add_argument("--walk", help="a velocity export (the T1's walking policy): each rollout first walks one of WALKS "
                        "and stops under it, the landing model takes over, and a second later the command is sent, "
                        "the grid's move from the stance the robot then stands in")  # fmt: skip
    parser.add_argument("--stop_s", type=float, default=3.0, help="--walk: seconds stood under the velocity export's "
                        "zero command before the landing model takes over")  # fmt: skip
    parser.add_argument("--out", help="the JSON written (default: the ONNX file's name with _native.json, "
                        "_native_second.json, _native_walk.json or _native_walk_second.json)")  # fmt: skip
    parser.add_argument("--seed", type=int, default=0, help="actuator lag draws")
    parser.add_argument("--lag", type=int, help="this lag [physics steps] on every actuator group instead of a draw "
                        "per rollout (5: the training simulator's evaluation delay)")  # fmt: skip
    parser.add_argument("--model", choices=("newton", "mjcf"), default="newton", help="utils/play_mujoco.py's T1Mujoco")
    parser.add_argument("--jobs", type=int, default=12, help="worker processes")
    args = parser.parse_args()
    if args.sim and (args.second or args.walk):
        parser.error("--sim holds single commands from the default stance: not comparable with --second or --walk")
    import onnx

    meta = {p.key: json.loads(p.value) for p in onnx.load(args.onnx).metadata_props}
    name = "_native" + "_walk" * bool(args.walk) + "_second" * args.second + ".json"
    out_path = Path(args.out or args.onnx.removesuffix(".onnx") + name)
    rec_path = out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = LandingCfg()  # the grid and the score read its box, clock and gate, which no training flag changes
    stand = (torch.tensor([SPEC.robot_cfg().init_state.pos]), torch.eye(3)[None], torch.zeros(1, 3), torch.zeros(1, 3),
             torch.tensor([SPEC.default_joint_pos_list()]), torch.zeros(1, SPEC.num_joints))  # fmt: skip
    pose = landing.soles(*(x[:, FEET] for x in Kinematics(SPEC, "cpu").forward(*stand)))["pose"][0]
    grid = command_grid(cfg, pose)
    sim = json.loads(Path(args.sim).read_text()) if args.sim else None
    if sim is not None:
        same = sim["grid"]["swing"] == grid["swing"].tolist() and np.allclose(sim["grid"]["c"], grid["c"], atol=1e-5)
        assert same, f"{args.sim} was run on other commands than this grid"
    walk = ids = hand = None
    if args.walk:  # a first pass without commands: the stance each robot stands in when its command is due
        K = len(grid["swing"])
        walk, ids = walks(K, args.seed, args.stop_s), torch.arange(K)
        z0 = rollouts(args, rec_path, torch.zeros(K, 0), torch.zeros(K, 0, 3), walk, ids)
        z0 = {k: torch.from_numpy(v) for k, v in z0.items()}
        soles = sole_log(z0["root_p"], z0["root_quat"], z0["q"], torch.zeros(K), torch.zeros(K, 3), ONE_S)["pose"]
        up = ~(z0["walk_fell"] | z0["fell"][: ONE_S + 1].any(0))
        off = relative(soles[ONE_S, up, 0], soles[ONE_S, up, 1]) - relative(pose[0], pose[1])
        hand = {"onnx": args.walk, "stop_s": args.stop_s, "rollouts": K, "fell_walking": int(z0["walk_fell"].sum()),
                "fell_after": int((z0["fell"].any(0) & ~z0["walk_fell"]).sum()),
                "still": float(((z0["qd_rms"][0] < STILL_QD) & (z0["root_speed"][0] < STILL_V)).float().mean()),
                "kept_alone": float(kept_after(soles, torch.zeros(K, dtype=torch.long))[up].float().mean()),
                "stance_off_cm": [100 * float(f(_dist(off[:, :2]))) for f in (torch.median, torch.max)],
                "stance_off_deg": [float(torch.rad2deg(f(wrap(off[:, 2]).abs()))) for f in (torch.median, torch.max)]}
        sub, kept = stance_commands(cfg, {k: v[up] for k, v in grid.items()}, soles[ONE_S, up])
        keep = up.clone()
        keep[up] = kept
        grid, walk, ids, stance = sub, walk[keep], ids[keep], soles[ONE_S, keep]
    swing, c = grid["swing"][:, None], grid["c"][:, None]  # (rollouts, commands per rollout)
    if args.second:
        second = follow_commands(cfg, pose, grid)
        keep = second["ok"]
        grid, second = ({k: v[keep] for k, v in d.items()} for d in (grid, second))
        if hand:
            walk, ids, stance = walk[keep], ids[keep], stance[keep]
        swing, c = torch.stack((grid["swing"], second["swing"]), 1), torch.stack((grid["c"], second["c"]), 1)
    K = len(swing)
    z = rollouts(args, rec_path, swing, c, walk, ids)
    clocks = torch.from_numpy(z["n0"]).float()  # (rollouts, commands)

    def scored(i: int, cmd: dict) -> dict:
        """utils.landing_score.score of every rollout's command i (cmd: its swing, c, travel and, if any, direction)."""
        rec = at_command(z, i)
        log = sole_log(rec["root_p"], rec["root_quat"], rec["q"], cmd["swing"], cmd["c"], ONE_S)
        log.update({k: rec[k] for k in ("fell", "qd_rms", "root_speed")})
        res = score(log, {**cmd, "n0": clocks[:, i]}, cfg)
        kept = kept_after(log["pose"], ONE_S + clocks[:, i].long())
        res["per"]["kept_after"], res["summary"]["kept_after"] = kept, float(kept.float().mean())
        return {**res, "anchor": log["anchor"]}

    first = scored(0, grid)
    res = scored(1, second) if args.second else first
    per, s = res["per"], res["summary"]
    if hand:  # the walk and the stop are the first pass's: the robot stands where the commands were built from
        d = first["anchor"] - stance
        assert max(float(d[..., :2].abs().max()), float(wrap(d[..., 2]).abs().max())) < 1e-5, \
            "the rollouts did not reproduce the first pass's stance at the command"
        hand.update(sent=K, commands=[list(w) for w in WALKS], per=walk.tolist(), rollout=ids.tolist(),
                    stance=stance.tolist())
    lags = (f"actuator lags drawn per rollout (seed {args.seed})" if args.lag is None
            else f"every actuator lag {args.lag} physics steps")  # fmt: skip
    doc = {"onnx": args.onnx, "iteration": meta["iteration"], "model": args.model, "seed": args.seed, "lag": args.lag,
           "reset_stance": pose.tolist(), "summary": jsonable(s), "grid": {k: v.tolist() for k, v in grid.items()},
           "clock": clocks.tolist(), "per": jsonable({k: v.tolist() for k, v in per.items()})}  # fmt: skip
    if args.second:
        doc["second"] = {k: v.tolist() for k, v in second.items()}
    if hand:
        doc["walk"] = hand
    out_path.write_text(json.dumps(doc, indent=1, allow_nan=False))

    start = "the stance the robot stopped in" if hand else "the default stance"
    what = (f"{K} rollouts of two commands from {start}: a command of the grid, then, one second after its "
            "deadline, the other foot brought beside the one that moved. The second commands" if args.second
            else f"{K} commands from {start}")  # fmt: skip
    print(f"\n{args.onnx} (iteration {meta['iteration']}) in native MuJoCo ({args.model} model), {lags}")
    if hand:
        print(f"{hand['rollouts']} rollouts begin with a walk under {args.walk} (one of {len(WALKS)} commands, "
              f"{WALK_S[0]:g} to {WALK_S[-1]:g} s) and {args.stop_s:g} s on the zero command; then the landing model "
              "takes over, and one second later the command is sent: the grid's move from the stance of that moment\n"
              f"at the hand-over: {hand['fell_walking']} had fallen, {hand['still']:.1%} were still; a second later "
              f"the stance is {hand['stance_off_cm'][0]:.1f} cm and {hand['stance_off_deg'][0]:.1f} deg from the "
              f"default one (median; largest {hand['stance_off_cm'][1]:.1f} cm, {hand['stance_off_deg'][1]:.1f} deg)\n"
              f"without a command (the same rollouts, run first): the stance of the hand-over is kept for the 3 s "
              f"after it in {hand['kept_alone']:.1%}; falls after the hand-over: {hand['fell_after']}\n"
              f"{K} of {hand['rollouts']} commands are sent (the others: a move the sampler would not send from "
              f"that stance{', or a second command of a travel it would not send' * args.second}, or a fall)")
    print(what)
    if args.second:  # the first command is read up to its deadline; the second command cuts its hold short
        print(f"(the first commands of these rollouts: arrived {first['summary']['arrived']:.1%}, median error "
              f"{first['summary']['err_cm_median']:.2f} cm and {first['summary']['err_deg_median']:.2f} deg)")
    ss = sim["summary"] if sim else None
    rec = Path(args.sim).with_suffix(".npz") if sim else None  # utils/landing_test.py --record: the soles it logged
    if sim and rec.exists() and np.load(rec)["pose"].shape[1] == K:  # of every command: the same line on them
        dl = ONE_S + torch.tensor(sim["grid"]["n0"]).long()
        ss["kept_after"] = float(kept_after(torch.from_numpy(np.load(rec)["pose"]), dl).float().mean())
    med = lambda p, k: float(np.median([x if x is not None else np.inf for x in p[k]]))  # noqa: E731
    print(f"{'':<24}{'native':>10}" + (f"{'training sim':>14}" if sim else ""))
    for name, key, fmt in ROWS:
        there = fmt.format(ss[key]) if sim and key in ss else "-"  # kept_after needs the trainer's recording
        print(f"{name:<24}{fmt.format(s[key]):>10}" + (f"{there:>14}" if sim else ""))
    for name, key in (("median error 1 s later", "err_cm_1s"), ("still after, median [s]", "still_s")):
        row = f"{name:<24}{med({k: v.tolist() for k, v in per.items()}, key):>10.2f}"
        print(row + (f"{med(sim['per'], key):>14.2f}" if sim else ""))
    print(f"{'relifted after':<24}{int((per['relifts'] > 0).sum()):>10}"
          + (f"{sum(x > 0 for x in sim['per']['relifts']):>14}" if sim else ""))  # fmt: skip
    print(f"\n{'by travel':<12}{'n':>5}{'arrived':>10}{'median cm':>11}{'p95 cm':>9}"
          + (f"{'| sim arrived':>15}{'median cm':>11}{'p95 cm':>9}" if sim else ""))  # fmt: skip
    for name, g in s["by_travel"].items():
        if g["n"]:
            row = f"{name:<12}{g['n']:>5}{g['arrived']:>10.1%}{g['err_cm_median']:>11.2f}{g['err_cm_p95']:>9.2f}"
            if sim:
                b = ss["by_travel"][name]
                row += f"{b['arrived']:>15.1%}{b['err_cm_median']:>11.2f}{b['err_cm_p95']:>9.2f}"
            print(row)
    if hand:
        print(f"\n{'by walk (vx, vy, yaw rate)':<28}{'n':>5}{'stance held':>13}{'arrived':>10}{'median cm':>11}"
              f"{'still in 3 s':>14}{'kept after':>12}")
        for w in WALKS:
            m = (walk[:, :3] == torch.tensor(w)).all(1)
            g, share = _group(per, m), lambda k: float(per[k][m].float().mean())  # noqa: E731
            print(f"{str(w):<28}{g['n']:>5}{share('stance_ok'):>13.1%}{g['arrived']:>10.1%}{g['err_cm_median']:>11.2f}"
                  f"{share('still_ok'):>14.1%}{share('kept_after'):>12.1%}")
    if not args.second:
        off, most = int((clocks[:, 0] != grid["n0"]).sum()), int((clocks[:, 0] - grid["n0"]).abs().max())
        why = ("the grid's is here looked up from that stance too" if hand
               else "the grid's is looked up from the stance at the reset")  # fmt: skip
        print(f"\nclock: {off} of {K} commands were given a deadline other than the grid's (the largest difference: "
              f"{most} control step{'s' * (most != 1)}); the model looks it up from the stance when the command "
              f"arrives, {why}")
    print(f"PASS (the training test's default-stance line on these rollouts): {s['pass']}")
    print(f"wrote {out_path} and {rec_path}")


if __name__ == "__main__":
    main()
