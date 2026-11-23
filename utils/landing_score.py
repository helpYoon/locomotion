"""The landing test's commands and its scoring (spec 10, 11): no simulator, so it runs on any machine.

``command_grid`` lists the commands the test sends; ``score`` reads a recorded rollout of them (utils/landing_rig.py
records it, utils/landing_test.py prints it). Poses are (x, y, yaw) triples as in src/landing.py; the per-command
numbers are in centimetres, degrees and seconds. ``score_library`` scores the second part; ``library_file`` finds the
library it runs on.
"""

import math
from pathlib import Path

import torch

from src.landing import YAW_M, LandingCfg, _dist, _separated, relative, wrap
from src.motion import CONTROL_DT

DISTANCES = (0.05, 0.10, 0.15, 0.20, 0.30)  # m: how far the swing sole moves
TURNS = tuple(math.radians(d) for d in (-15, 0, 15, 30))  # rad, toes out positive
PURE_TURNS = tuple(math.radians(d) for d in (30, 45))  # rad: turns on the spot
TOL = 1e-6  # the sampler's tests with this slack, so that an exact 5 cm move or a landing on the box edge is kept
STILL_STEPS, STILL_QD, STILL_V = 25, 0.01, 0.02  # still: this many steps with joint-speed RMS and root speed under
STANCE_D, STANCE_A = 0.02, 0.10  # m, rad: the stance is held while both soles stay this close to the reset stance
LIFT_H = 0.005  # m: a sole is off the floor above this
SCUFF_H = 0.001  # m: a sole corner under this clearance is on the floor
ONE_S = round(1 / CONTROL_DT)  # steps in one second
TRAVEL_BINS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.61)  # m: edges of the travel groups of the summary
REPO = Path(__file__).resolve().parents[1]  # the repository root


def library_file(saved: str, robot: str, repo: Path = REPO) -> str:
    """A library file a checkpoint's arguments name (bones_list or bones_lib; the path on the machine that trained
    it): the saved path if it exists, else <repo>/data/<robot>/<its basename> if that exists (utils/landing_rig.py
    LibraryRig; here, not there, so that it is tested without the simulator)."""
    if Path(saved).exists():
        return saved
    local = Path(repo) / "data" / robot / Path(saved).name
    if not local.exists():
        raise FileNotFoundError(f"the library file {saved} (the checkpoint's) does not exist, nor does {local}; "
                                "python -m utils.landing_test --no_library skips the second part")  # fmt: skip
    return str(local)


def command_grid(cfg: LandingCfg, pose: torch.Tensor) -> dict[str, torch.Tensor]:
    """The commands of the test from the stance pose (2, 3) (left, right sole) the robot is reset to: for each foot the
    swing sole moved by 5, 10, 15, 20 or 30 cm in one of 8 directions (45 degrees apart; in the planted sole's frame,
    mirrored for a right swing so that direction 0 is forward, 2 outward and 6 inward for both feet) with a turn of
    -15, 0, 15 or 30 degrees (toes out positive), plus pure turns of 30 and 45 degrees, kept only where the sampler
    (landing.land_sample) would send them: landed stance inside the box, travel within cfg.travel, rectangles
    cfg.sole_gap apart. Clock n0 is the sampler's without jitter.
    Returns swing (K,), c (K, 3) in the planted sole's frame, n0 (K,), travel (K,) [m],
    direction (K,) long (0..7, -1 for a pure turn) and turn (K,) [degrees]."""
    pose = pose.double()
    rows = []
    for swing in (0, 1):
        sg = 1.0 - 2 * swing
        mirror = torch.tensor([1.0, sg, sg], dtype=pose.dtype)
        cur = relative(pose[swing], pose[1 - swing]) * mirror
        moves = [(d * math.cos(k * math.pi / 4), d * math.sin(k * math.pi / 4), t, k)
                 for d in DISTANCES for k in range(8) for t in TURNS]  # fmt: skip
        moves += [(0.0, 0.0, t, -1) for t in PURE_TURNS]
        for dx, dy, turn, k in moves:
            cand = cur + torch.tensor([dx, dy, turn], dtype=pose.dtype)
            travel = _dist(cand[:2] - cur[:2]) + YAW_M * wrap(cand[2] - cur[2]).abs()
            lo, hi = (-cfg.box_x, cfg.box_y[0], cfg.box_yaw[0]), (cfg.box_x, cfg.box_y[1], cfg.box_yaw[1])
            inside = all(lo[i] - TOL <= float(cand[i]) <= hi[i] + TOL for i in range(3))
            within = cfg.travel[0] - TOL <= float(travel) <= cfg.travel[1] + TOL
            if inside and within and bool(_separated(cand, cfg.sole_gap)):
                n0 = min(max(round(cfg.clock_base + cfg.clock_per_m * float(travel)), cfg.clock_range[0]),
                         cfg.clock_range[1])
                rows.append((swing, *(cand * mirror).tolist(), n0, float(travel), k, math.degrees(turn)))
    assert rows, f"no command of the grid can be sent from the stance {pose.tolist()}"
    t = torch.tensor(rows, dtype=torch.float64)
    return {"swing": t[:, 0].float(), "c": t[:, 1:4].float(), "n0": t[:, 4].float(), "travel": t[:, 5].float(),
            "direction": t[:, 6].long(), "turn": t[:, 7].float()}  # fmt: skip


def jsonable(x):
    """x (nested dicts and lists of numbers) with every nan and inf replaced by None, so that it dumps as valid JSON."""
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    return None if isinstance(x, float) and not math.isfinite(x) else x


def _spread(x: torch.Tensor) -> tuple[float, float]:
    """Median and 95th percentile of x (n,); nan when empty."""
    if len(x) == 0:
        return math.nan, math.nan
    return float(x.median()), float(torch.quantile(x.float(), 0.95))


def _group(per: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, float]:
    """Count, share arrived and the median and 95th percentile of the errors of the commands in mask (K,) bool."""
    m50, m95 = _spread(per["err_cm"][mask])
    d50, d95 = _spread(per["err_deg"][mask])
    return {"n": int(mask.sum()), "arrived": float(per["arrived"][mask].float().mean()) if mask.any() else math.nan,
            "err_cm_median": m50, "err_cm_p95": m95, "err_deg_median": d50, "err_deg_p95": d95}  # fmt: skip


def _by_travel(per: dict[str, torch.Tensor], travel: torch.Tensor) -> dict[str, dict[str, float]]:
    """_group of the commands in each TRAVEL_BINS interval of their travel (K,) [m], labelled in cm."""
    return {f"{100 * a:g}-{100 * b:g} cm": _group(per, (travel + TOL >= a) & (travel + TOL < b))
            for a, b in zip(TRAVEL_BINS[:-1], TRAVEL_BINS[1:])}  # fmt: skip


def score(log: dict[str, torch.Tensor], cmd: dict[str, torch.Tensor], cfg: LandingCfg) -> dict:
    """Score a recorded rollout of K commands. log: pose (T, K, 2, 3) the sole poses of states 0 .. T-1 (state j is
    after j control steps), h (T, K, 2) the soles' lowest-corner clearance, zc (T, K, 2) the soles' centre height,
    fell (T, K) bool, qd_rms (T, K) the joints' speed RMS, root_speed (T, K), t_cmd (int: the command was issued on
    state t_cmd, the step that follows is the first of its clock), target (K, 3) and anchor (K, 2, 3) of the command
    state in the world frame. cmd: the grid's swing and n0.
    The deadline is state t_cmd + n0. For still_s to be resolved up to 3 s the log must hold a still block that
    starts 3 s after the latest deadline: at least 150 + 25 states after it (utils/landing_rig.py's 3.5 s; asserted).

    Per command (K,): err_cm and err_deg, the swing sole's distance and yaw from the target at the deadline, and
    err_cm_1s 50 steps later; apex_cm, the swing sole's largest zc (centre height) over states t_cmd .. deadline;
    floor_cm, the swing sole's centre travel summed over the steps t_cmd + 1 .. deadline that end with the gate on zc
    above 0.5 (the centre is on the floor; a pitched sole with a corner down is not); travel_cm, the same travel over
    all those steps; scuff_cm, the same travel over the steps that end with the swing sole's lowest corner under 1 mm
    (a toe or heel dragging); planted_cm and planted_rad, the planted sole's largest distance and yaw from its anchor
    over states t_cmd .. deadline; fell, any state of the rollout; still_s, the time from the deadline to the start of
    the first 25 consecutive states with qd_rms < 0.01 and root_speed < 0.02 (inf if none); relifts, the runs of
    h > 5 mm of either sole that begin after the deadline (one already running at the deadline is not counted);
    stance_cm and stance_rad, the larger over the two soles of the distance and the absolute yaw difference between
    state t_cmd and state 0 (the reset state): whether the robot still stands in the stance it was reset to when the
    command is sent. Flags: arrived (err within 5 cm and 0.10 rad and the swing sole's h under 2 cm at the deadline),
    lifted (apex 1 cm or more, floor travel 2 cm or less), planted_ok (within 1 cm and 0.05 rad), still_ok (still_s 3
    or less), stance_ok (stance_cm within 2 and stance_rad within 0.10).
    Returns {"per": {name: (K,) tensor}, "summary": {...}}; the summary's pass is the spec 11 line, and it adds
    scuff_share (total scuff_cm over total travel_cm, 0 without travel) and the medians of apex_cm and floor_cm."""
    pose, h, zc, t_cmd = log["pose"], log["h"], log["zc"], log["t_cmd"]
    T, K = h.shape[:2]
    ar, sw = torch.arange(K), cmd["swing"].long()
    dl = t_cmd + cmd["n0"].round().long()  # (K,) deadline state
    assert int(dl.max()) + 3 * ONE_S + STILL_STEPS <= T, "the rollout ends before the 3 s still window after the " \
        "latest deadline: hold 3.5 s or more after the longest clock (utils/landing_test.py --hold_s)"
    t = torch.arange(T)[:, None]
    W, P = pose[:, ar, sw], pose[:, ar, 1 - sw]  # (T, K, 3) swing and planted sole
    h_w, zc_w = h[:, ar, sw], zc[:, ar, sw]
    target, anchor_p = log["target"], log["anchor"][ar, 1 - sw]

    def miss(i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = W[i, ar] - target
        return 100 * _dist(e[:, :2]), torch.rad2deg(wrap(e[:, 2]).abs())

    err_cm, err_deg = miss(dl)
    err_cm_1s, _ = miss(dl + ONE_S)
    window = (t >= t_cmd) & (t <= dl)  # states from the command to the deadline
    apex_cm = 100 * zc_w.masked_fill(~window, -math.inf).max(0).values
    step_cm = torch.cat((torch.zeros(1, K), 100 * _dist(W[1:, :, :2] - W[:-1, :, :2])), 0)
    on_floor = torch.sigmoid((cfg.gate_h - zc_w) / cfg.gate_w) > 0.5
    steps = window & (t > t_cmd)
    floor_cm = (step_cm * (steps & on_floor)).sum(0)
    travel_cm = (step_cm * steps).sum(0)
    scuff_cm = (step_cm * (steps & (h_w < SCUFF_H))).sum(0)
    planted_cm = (100 * _dist(P[..., :2] - anchor_p[:, :2])).masked_fill(~window, 0).max(0).values
    planted_rad = wrap(P[..., 2] - anchor_p[:, 2]).abs().masked_fill(~window, 0).max(0).values
    still = (log["qd_rms"] < STILL_QD) & (log["root_speed"] < STILL_V)
    starts = still.T.unfold(1, STILL_STEPS, 1).all(-1).T  # (T - 24, K): a still run starts on this state
    run = starts & (t[: T - STILL_STEPS + 1] >= dl)
    start = run.int().argmax(0)
    still_s = torch.where(run.any(0), (start - dl) * CONTROL_DT, torch.full((K,), math.inf))
    lift = (h > LIFT_H).any(-1)
    relifts = (lift[1:] & ~lift[:-1] & (t[1:] > dl)).sum(0)
    stance_cm = (100 * _dist(pose[t_cmd, :, :, :2] - pose[0, :, :, :2])).max(-1).values
    stance_rad = wrap(pose[t_cmd, :, :, 2] - pose[0, :, :, 2]).abs().max(-1).values
    per = {"err_cm": err_cm, "err_deg": err_deg, "err_cm_1s": err_cm_1s, "apex_cm": apex_cm, "floor_cm": floor_cm,
           "travel_cm": travel_cm, "scuff_cm": scuff_cm,
           "planted_cm": planted_cm, "planted_rad": planted_rad, "fell": log["fell"].any(0), "still_s": still_s,
           "relifts": relifts, "stance_cm": stance_cm, "stance_rad": stance_rad}  # fmt: skip
    per["arrived"] = (err_cm <= 5) & (err_deg <= math.degrees(0.10)) & (h_w[dl, ar] < 0.02)
    per["lifted"] = (apex_cm >= 1) & (floor_cm <= 2)
    per["planted_ok"] = (planted_cm <= 1) & (planted_rad <= 0.05)
    per["still_ok"] = still_s <= 3
    per["stance_ok"] = (stance_cm <= 100 * STANCE_D) & (stance_rad <= STANCE_A)
    share = lambda k: float(per[k].float().mean())  # noqa: E731
    summary = {"count": K, "falls": int(per["fell"].sum())}
    summary.update({k: share(k) for k in ("arrived", "lifted", "planted_ok", "still_ok", "stance_ok")})
    total = float(travel_cm.sum())
    summary["scuff_share"] = float(scuff_cm.sum()) / total if total > 0 else 0.0
    summary["apex_cm_median"], summary["floor_cm_median"] = float(apex_cm.median()), float(floor_cm.median())
    summary.update(dict(zip(("err_cm_median", "err_cm_p95"), _spread(err_cm))))
    summary.update(dict(zip(("err_deg_median", "err_deg_p95"), _spread(err_deg))))
    tr, di = cmd.get("travel"), cmd.get("direction")
    if tr is not None:
        summary["by_travel"] = _by_travel(per, tr)
    if di is not None:
        summary["by_direction"] = {("turn" if k < 0 else f"dir{k}"): _group(per, di == k) for k in range(-1, 8)}
    summary["pass"] = bool(
        summary["falls"] == 0 and summary["arrived"] >= 0.95 and summary["err_cm_median"] <= 2
        and summary["err_deg_median"] <= 3 and summary["lifted"] >= 0.95 and summary["planted_ok"] >= 0.90
        and summary["still_ok"] >= 0.95
        and summary["stance_ok"] >= 0.95
    )  # fmt: skip
    return {"per": per, "summary": summary}


def score_library(log: dict) -> dict:
    """Score the test's second part (utils/landing_rig.py LibraryRig.run): free rows started on the library's hold
    frames, given the sampler's commands one after another. log: err (C,) [m] and yaw (C,) [rad], the swing sole's
    distance and yaw from the target at each deadline; place (C,) long, the command's place in its episode (1 for the
    first: a later one starts from the stance the one before landed in); travel (C,) [m] when it was issued; moved (E,)
    [m] and turned (E,) [rad], per episode, the larger over the two soles of the distance and the absolute yaw change
    between where SETTLE ended and the reset; falls and episodes (int) of the landing robots.
    Returns commands, episodes, falls; stance_kept, the share of episodes that moved at most STANCE_D and turned at
    most STANCE_A, and the median and 95th percentile of moved [cm]; arrived, the share of commands within 5 cm and
    0.10 rad, and the errors' median and 95th percentile [cm, deg]; by_place and by_travel (groups as score's); pass:
    no fall and the stance kept in 95% of the episodes. No accuracy line: the design sets one on the default-stance
    grid only."""
    err, yaw, place = log["err"], log["yaw"], log["place"]
    per = {"err_cm": 100 * err, "err_deg": torch.rad2deg(yaw), "arrived": (err <= 0.05) & (yaw <= 0.10)}
    kept = (log["moved"] <= STANCE_D) & (log["turned"] <= STANCE_A)
    s = {"commands": len(err), "episodes": log["episodes"], "falls": log["falls"],
         "stance_kept": float(kept.float().mean())}  # fmt: skip
    s.update(dict(zip(("moved_cm_median", "moved_cm_p95"), _spread(100 * log["moved"]))))
    s["arrived"] = float(per["arrived"].float().mean())
    s.update(dict(zip(("err_cm_median", "err_cm_p95"), _spread(per["err_cm"]))))
    s.update(dict(zip(("err_deg_median", "err_deg_p95"), _spread(per["err_deg"]))))
    places = (("first", place == 1), ("second", place == 2), ("third", place == 3), ("fourth or later", place >= 4))
    s["by_place"] = {name: _group(per, m) for name, m in places}
    s["by_travel"] = _by_travel(per, log["travel"])
    s["pass"] = bool(s["falls"] == 0 and s["stance_kept"] >= 0.95)
    return s
