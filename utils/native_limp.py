"""Left/right step evenness of a velocity policy's ONNX export in plain MuJoCo (utils/play_native.py's simulator), at
held forward commands: does one foot step shorter or slower than the other (a limp)?

Each trial: from standing, 1 s at zero, then 9 s at the command, with its own draw of the per-group lags (as the step
test; the same draws for every policy, so policies are compared on the same trials). A step = a swing's end (the
foot's horizontal speed above 0.15 m/s to start, below 0.08 to end, at least 3 control steps: lr.py's rule) by the
foot other than the last to land; its time = since the other foot's touchdown; its length = the landing foot's lead
over the other along the travel (the root's displacement over the measured window) and along the trunk's heading.
Steps in the command's first 1.5 s are left out. Per command, pooled over the trials: each foot's median step time
and length and the unevenness |L - R| / mean; per trial the same, for the spread.

usage (repo root; CPU only):
    CUDA_VISIBLE_DEVICES= .venv/bin/python -m utils.native_limp --onnx A.onnx [B.onnx ...] [--trials 8] [--speeds ...]
"""

import argparse
import json
import os

import numpy as np

from src.motion import CONTROL_DT as DT

T_STAND, T_WALK, SKIP = round(1.0 / DT), round(9.0 / DT), round(1.5 / DT)
SPEEDS = (0.4, 0.45, 0.5, 0.6, 0.7, 1.0)


def swings(p, v_on=0.15, v_off=0.08, min_len=3):
    """(start, end) indices of one foot's swings from its horizontal positions p (T, 2) (lr.py's gait.swings)."""
    sp = np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=1) / DT]
    out, s, on = [], 0, False
    for t, x in enumerate(sp):
        if not on and x > v_on:
            s, on = t, True
        elif on and x < v_off:
            if t - s >= min_len:
                out.append((s, t))
            on = False
    return out


def walk_steps(p, yaw, u):
    """Steps of a walk: feet positions p (T, 2 feet, 2), heading yaw (T,), travel direction u (2,). Returns
    (foot, time [s], lead along u [m], lead along the heading [m]) per step."""
    td = sorted((e, k) for k in range(2) for _, e in swings(p[:, k]))
    out = []
    for (e0, k0), (e1, k1) in zip(td, td[1:]):
        if k1 == k0:  # the same foot twice: a shuffle, not a step
            continue
        d = p[e1, k1] - p[e1, k0]
        out.append((k1, (e1 - e0) * DT, float(d @ u), float(d @ np.array([np.cos(yaw[e1]), np.sin(yaw[e1])]))))
    return out


def evenness(steps):
    """Each foot's median step time and lead (travel, heading) and the unevenness |L - R| / |mean| of each; nan where
    a foot has no step."""
    med = lambda k, i: float(np.median([s[i] for s in steps if s[0] == k])) if any(s[0] == k for s in steps) else float("nan")  # noqa: E731
    si = lambda a, b: abs(a - b) / abs((a + b) / 2) if abs(a + b) > 1e-9 else float("nan")  # noqa: E731
    t, l, h = ((med(0, i), med(1, i)) for i in (1, 2, 3))
    return {"n": (sum(s[0] == 0 for s in steps), sum(s[0] == 1 for s in steps)), "t": t, "l": l, "h": h,
            "si_t": si(*t), "si_l": si(*l), "si_h": si(*h)}  # fmt: skip


def measure(onnx_path, speeds=SPEEDS, trials=8, seed=0):
    from utils.play_native import NativeSim, Policy, draw_lags, load_metadata, robot_of

    meta = load_metadata(onnx_path)
    robot = robot_of(meta)
    sim, pi = NativeSim(robot), Policy(onnx_path, meta)
    feet = [sim.m.body(f"robot/{b}").id for b in robot.feet]
    out = []
    for k, v in enumerate(speeds):
        rng, pooled, per_trial, falls, speed = np.random.default_rng([seed, k]), [], [], 0, []
        for _ in range(trials):
            sim.set_lags(draw_lags(robot, rng))
            sim.reset(), pi.reset()
            P, Y, root, fell = [], [], [], False
            for t in range(T_STAND + T_WALK):
                sim.step(pi(sim, (0.0, 0.0, 0.0) if t < T_STAND else (v, 0.0, 0.0)))
                if sim.fallen():
                    fell = True
                    break
                if t >= T_STAND + SKIP:
                    R = sim.rotation()
                    P.append(sim.d.xpos[feet, :2].copy()), Y.append(np.arctan2(R[1, 0], R[0, 0]))
                    root.append(sim.d.qpos[sim.root_q : sim.root_q + 2].copy())
            if fell:
                falls += 1
                continue
            disp = root[-1] - root[0]
            u = disp / max(np.linalg.norm(disp), 1e-6)
            s = walk_steps(np.array(P), np.array(Y), u)
            pooled += s
            per_trial.append(evenness(s))
            speed.append(float(np.linalg.norm(disp)) / (len(root) * DT))
        e = evenness(pooled)
        spread = {k2: [x[k2] for x in per_trial if x[k2] == x[k2]] for k2 in ("si_t", "si_l", "si_h")}
        out.append({"cmd": v, "speed": float(np.mean(speed)) if speed else float("nan"), "falls": falls, "trials": trials, **e,
                    "trial_q": {k2: [float(np.percentile(x, q)) for q in (25, 50, 75)] if x else None for k2, x in spread.items()}})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", nargs="+", required=True)
    p.add_argument("--speeds", nargs="+", type=float, default=list(SPEEDS))
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="logs/native")
    a = p.parse_args()
    pct = lambda x: "  nan" if x != x else f"{100 * x:4.0f}%"  # noqa: E731
    for f in a.onnx:
        res = measure(f, a.speeds, a.trials, a.seed)
        name = f.split("/")[1] if f.startswith("logs/") else os.path.splitext(os.path.basename(f))[0]
        worst = max(max(r["si_t"], r["si_l"]) if r["si_t"] == r["si_t"] and r["si_l"] == r["si_l"] else 9.0 for r in res)
        print(f"== {name}  (worst time/length unevenness {pct(worst)})", flush=True)
        for r in res:
            q = r["trial_q"]["si_l"]
            print(f"  cmd {r['cmd']:.2f}  v {r['speed']:.2f}  steps L/R {r['n'][0]:2d}/{r['n'][1]:2d}  time L/R {r['t'][0]:.2f}/{r['t'][1]:.2f} ({pct(r['si_t'])})"
                  f"  length L/R {r['l'][0]:.2f}/{r['l'][1]:.2f} ({pct(r['si_l'])})  heading {pct(r['si_h'])}"
                  f"  per-trial length {'-' if q is None else f'{100 * q[0]:.0f}-{100 * q[2]:.0f}%'}"
                  + (f"  FELL {r['falls']}/{r['trials']}" if r["falls"] else ""), flush=True)
        os.makedirs(a.out_dir, exist_ok=True)
        with open(os.path.join(a.out_dir, f"limp_{name}.json"), "w") as fh:
            json.dump({"onnx": f, "speeds": a.speeds, "trials": a.trials, "seed": a.seed, "results": res}, fh, indent=1)


if __name__ == "__main__":
    main()
