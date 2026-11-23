"""Offline sequence distillation labels from cuTAMP's frozen deployed T1 landing teacher.

Run with the cuTAMP checkout on PYTHONPATH and its Torch interpreter. No physics
simulator is used. The starts file is a list of CPU tensor dictionaries containing
'tensors' (nine batched issue tensors), 'dynamics', 'group', 'split', 'order',
and optionally 'commands' and 'history_source'. Group all descendants and command
perturbations of a source scene together before assigning its split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from time import perf_counter

import torch
from cutamp.landing_predictor import (
    LandingPredictor,
    LandingState,
    nominal_landing_dynamics,
    prediction_precision,
)

ORDERS = ("L", "R", "LR", "RL")
SPLITS = ("train", "val", "test")
SOURCE_NAMES = ("planner_home", "planner_retract", "planner_other", "synthetic_history")
KNOTS = (
    "quarter_deadline",
    "half_deadline",
    "three_quarters_deadline",
    "deadline",
    "mid_hold",
    "end_hold",
)


def atomic_save(value, path: Path):
    tmp = path.with_suffix(".pt.tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def concatenate_states(states):
    tensors = tuple(
        torch.cat(fields) for fields in zip(*(s.tensors for s in states), strict=True)
    )
    return LandingState(
        tensors, torch.cat([s.dynamics for s in states]), "mixed_issue_states"
    )


def take_state(state, indices):
    return LandingState(
        tuple(t[indices] for t in state.tensors),
        state.dynamics[indices],
        state.history_source,
    )


def knot_indices(clock, hold_ticks):
    """Post-tick, zero-based indexing; final hold length may differ per particle."""
    quarter = torch.arange(1, 5, device=clock.device)
    deadlines = (clock[:, None] * quarter[None] + 3) // 4
    holds = torch.stack((clock + (hold_ticks + 1) // 2, clock + hold_ticks), 1)
    return torch.cat((deadlines, holds), 1) - 1


def readouts(prediction):
    bodies, soles, clocks, holds = [], [], [], []
    for step in prediction.steps:
        hold = step.valid_ticks.sum(dim=1) - step.clock
        ticks = knot_indices(step.clock, hold)
        rows = torch.arange(len(ticks), device=ticks.device)[:, None]
        bodies.append(
            torch.cat(
                [state[rows, ticks].flatten(2) for state in step.trajectory], dim=-1
            )
        )
        soles.append(step.sole_trace[rows, ticks])
        clocks.append(step.clock)
        holds.append(hold)
    return (
        torch.stack(bodies, 1),
        torch.stack(soles, 1),
        torch.stack(clocks, 1),
        torch.stack(holds, 1),
    )


def pose_delta(a, b):
    xy = a[..., :2] - b[..., :2]
    yaw = a[..., 2:] - b[..., 2:]
    return torch.cat((xy, torch.atan2(torch.sin(yaw), torch.cos(yaw))), dim=-1)


def branch_fd(nominal, plus, minus, nominal_clock, plus_clock, minus_clock):
    # A clock change in the first step also invalidates a central slope of step two.
    ok_p = (plus_clock == nominal_clock).long().cumprod(dim=1).bool()
    ok_m = (minus_clock == nominal_clock).long().cumprod(dim=1).bool()
    extend = lambda x: x[:, :, None, None, None]
    central = 0.5 * pose_delta(plus, minus)
    single = torch.where(
        extend(ok_p), pose_delta(plus, nominal), pose_delta(nominal, minus)
    )
    diff = torch.where(extend(ok_p & ok_m), central, single)
    return torch.where(extend(ok_p | ok_m), diff, 0), (ok_p | ok_m)


def pad_steps(value, fill=0):
    if value.shape[1] == 2:
        return value
    return torch.cat((value, torch.full_like(value[:, :1], fill)), dim=1)


def random_tensor(shape, rng, device, *, normal=False):
    fn = torch.randn if normal else torch.rand
    return fn(shape, device=device, generator=rng)


def draw_commands(order, n, rng, device, *, relaxed=False):
    # Native support-frame target coordinates, including mirrored lateral and yaw.
    r = random_tensor((n, len(order), 3), rng, device)
    lower = r.new_tensor([-0.33, 0.14, -math.radians(20)])
    upper = r.new_tensor([0.33, 0.48, math.radians(50)])
    if relaxed:
        lower = r.new_tensor([-0.42, 0.10, -math.radians(35)])
        upper = r.new_tensor([0.42, 0.54, math.radians(65)])
    mirror = r.new_tensor(
        [[1, 1 if s == "L" else -1, 1 if s == "L" else -1] for s in order]
    )
    return mirror * (lower + (upper - lower) * r)


def synthetic_bank(teacher, rng, split, count):
    """Issue from diverse completed postures; retain only synchronized finite standing starts."""
    device = teacher.device
    xi = nominal_landing_dynamics(teacher.model.home_joints, count, 2)
    xi[:, 0] = (random_tensor((count,), rng, device) * 0.65 + 0.30 - 0.75) / 0.45
    lag = torch.randint(2, 9, (count, 1), generator=rng, device=device)
    xi[:, -5:] = (lag - 5) / 3
    home = teacher.standing_start(xi)
    pieces = []
    for s, idx in zip(
        ("L", "R"), torch.arange(count, device=device).chunk(2), strict=True
    ):
        state = take_state(home, idx)
        z = random_tensor((len(idx), 3), rng, device)
        cmd = torch.stack(
            (-0.18 + 0.40 * z[:, 0], 0.17 + 0.11 * z[:, 1], -0.15 + 0.35 * z[:, 2]),
            dim=1,
        )
        if s == "R":
            cmd[:, 1:] *= -1
        foot = teacher.rollout(state, s, cmd[:, None]).terminal
        valid = (foot.tensors[-1][:, 0] > 0.5) & (foot.tensors[-1][:, 2:4] == 0).all(1)
        valid &= (foot.tensors[0][:, 2] > 0.40) & (foot.tensors[1][:, 2, 2] > 0.6)
        valid &= torch.stack(
            [t.flatten(1).isfinite().all(1) for t in foot.tensors]
        ).all(0)
        if valid.any():
            pieces.append(take_state(foot, valid))
    if not pieces:
        raise ValueError("No usable synthetic issue states were produced")
    return concatenate_states(pieces)


def record_kind(record):
    source = str(record.get("source_kind", record.get("history_source", ""))).lower()
    if "home" in source or "standing" in source:
        return 0
    if "retract" in source or "supported_hold" in source:
        return 1
    return 2


class Bank:
    def __init__(self, records, synthetic, order, groups, split, *, device=None):
        if synthetic is not None:
            device = synthetic.dynamics.device
        only_retract = synthetic is None
        states, centers, keys, kinds = [], [], [], []
        self.balance_sources = split == "train"
        # Preserve actual command anchors. Single-step proposals can use any
        # planner prefix beginning on the same side. Synthetic starts fill all orders.
        for record in records:
            if record["split"] != split or (only_retract and record_kind(record) != 1):
                continue
            if (
                not self.balance_sources
                and not only_retract
                and not record["order"].startswith(order)
            ):
                continue
            state = LandingState(
                tuple(t.to(device) for t in record["tensors"]),
                record["dynamics"].to(device),
                record.get("history_source", "captured"),
            )
            n = len(state.dynamics)
            if len(state.tensors) != 9 or any(len(t) != n for t in state.tensors):
                raise ValueError("Expected nine matching batched issue state tensors")
            if not (
                (state.tensors[-1][:, 0] > 0.5)
                & (state.tensors[-1][:, 2:4] == 0).all(1)
            ).all():
                raise ValueError(
                    f"Captured non-HOLD issue state from {record['group']}"
                )
            anchor = record.get("commands")
            center = torch.full((n, len(order), 3), float("nan"), device=device)
            # Commands alternate, so a common first side implies all available
            # commands up to min(capture length, requested length) are reusable.
            if anchor is not None and order[0] == record["order"][0]:
                anchors = torch.as_tensor(
                    anchor, device=device, dtype=state.dynamics.dtype
                )
                anchors = anchors.reshape(-1, len(record["order"]), 3).expand(n, -1, -1)
                shared = min(len(order), len(record["order"]))
                center[:, :shared] = anchors[:, :shared]
            states.append(state)
            centers.append(center)
            keys.extend([groups.index(record["group"])] * n)
            kinds.extend([record_kind(record)] * n)
        self.captured = len(keys)
        self.synthetic = 0
        if synthetic is not None:
            self.synthetic = len(synthetic.dynamics)
            states.append(synthetic)
            centers.append(
                torch.full((self.synthetic, len(order), 3), float("nan"), device=device)
            )
            keys.extend([groups.index(f"synthetic/{split}")] * self.synthetic)
            kinds.extend([3] * self.synthetic)
        if not states:
            raise ValueError(
                "No usable captured post-retraction states in selected split"
            )
        self.state = concatenate_states(states)
        self.center = torch.cat(centers)
        self.group = torch.tensor(keys, device=device)
        self.kind = torch.tensor(kinds, device=device)
        self.pools = {}
        if self.balance_sources:
            for kind in (0, 1):
                pool = torch.where(self.kind == kind)[0]
                if not len(pool):
                    raise ValueError(
                        f"Training requires captured {SOURCE_NAMES[kind]} issue states"
                    )
                preferred = pool[self.center[pool, 0, 0].isfinite()]
                self.pools[kind] = pool, preferred

    def pick_source(self, idx, mask, kind, rng):
        pool, preferred = self.pools[kind]
        count = int(mask.sum())
        picks = pool[
            torch.randint(len(pool), (count,), device=idx.device, generator=rng)
        ]
        if len(preferred):
            matched = preferred[
                torch.randint(
                    len(preferred), (count,), device=idx.device, generator=rng
                )
            ]
            picks = torch.where(
                random_tensor((count,), rng, idx.device) < 0.6, matched, picks
            )
        idx[mask] = picks

    def draw(self, n, rng, order):
        device = self.center.device
        if self.synthetic:
            idx = (
                torch.randint(self.synthetic, (n,), device=device, generator=rng)
                + self.captured
            )
        else:
            idx = torch.randint(self.captured, (n,), device=device, generator=rng)
        if self.balance_sources:
            source = random_tensor((n,), rng, device)
            self.pick_source(idx, source < 0.47, 0, rng)
            self.pick_source(idx, (source >= 0.47) & (source < 0.80), 1, rng)
        elif self.captured and self.synthetic:
            captured = torch.randint(self.captured, (n,), device=device, generator=rng)
            idx = torch.where(random_tensor((n,), rng, device) < 0.80, captured, idx)
        center = self.center[idx]
        # Near the actual optimizer's seed/update, near a representative small
        # command, across the training box, and outside for failure examples.
        choice = random_tensor((n,), rng, device)
        broad = draw_commands(order, n, rng, device)
        ood = draw_commands(order, n, rng, device, relaxed=True)
        nominal = draw_commands(order, n, rng, device) * 0.18
        nominal += nominal.new_tensor(
            [[0.16, 0.21 if s == "L" else -0.21, 0] for s in order]
        )
        center = torch.where(center.isfinite(), center, nominal)
        noise = random_tensor(tuple(center.shape), rng, device, normal=True)
        sigma = center.new_tensor([0.037, 0.024, 0.105])
        local = center + noise * sigma
        commands = torch.where(
            (choice < 0.67)[:, None, None],
            local,
            torch.where((choice < 0.95)[:, None, None], broad, ood),
        )
        return take_state(self.state, idx), commands, self.group[idx], self.kind[idx]


@torch.no_grad()
def label_batch(teacher, state, order, commands, group, kind, rng, fd_fraction):
    device = commands.device
    n, k, _ = commands.shape
    nf = min(n, round(n * fd_fraction))
    # Batch nominal and +/- probes in the same teacher call, so all knots use
    # the same model and integer-clock convention. No gradient tape is retained.
    direction = torch.zeros_like(commands)
    probes = torch.randperm(n, device=device, generator=rng)[:nf]
    axes = torch.randint(3 * k, (nf,), device=device, generator=rng)
    sign = 2 * torch.randint(2, (nf,), device=device, generator=rng) - 1
    epsilon = commands.new_tensor([0.001, 0.001, math.radians(0.1)]).repeat(k)
    direction.reshape(n, -1)[probes, axes] = sign * epsilon[axes]
    owners = torch.cat((torch.arange(n, device=device), probes, probes))
    cand = torch.cat(
        (
            commands,
            commands[probes] + direction[probes],
            commands[probes] - direction[probes],
        )
    )
    prediction = teacher.rollout(take_state(state, owners), order, cand)
    body, soles, clocks, holds = readouts(prediction)
    fd = torch.zeros_like(soles[:n])
    valid = torch.zeros_like(clocks[:n], dtype=torch.bool)
    diff, good = branch_fd(
        soles[probes],
        soles[n : n + nf],
        soles[n + nf :],
        clocks[probes],
        clocks[n : n + nf],
        clocks[n + nf :],
    )
    fd[probes], valid[probes] = diff, good
    sides = torch.tensor(["LR".index(c) for c in order], device=device).expand(n, -1)
    # Keep failures in the labels. Nonfinite targets mean the teacher result cannot be trained on.
    if (
        not torch.isfinite(body[:n]).all()
        or not torch.isfinite(soles[:n]).all()
        or not torch.isfinite(fd).all()
    ):
        raise ValueError(f"Nonfinite teacher output for order {order}")
    out = {
        "state": tuple(t.cpu() for t in state.tensors),
        "dynamics": state.dynamics,
        "feet": teacher.model.feet(state.tensors[:6])[0],
        "commands": pad_steps(commands),
        "sides": pad_steps(sides, -1),
        "body": pad_steps(body[:n]),
        "soles": pad_steps(soles[:n]),
        "direction": pad_steps(direction),
        "fd_soles": pad_steps(fd),
        "fd_valid": pad_steps(valid),
        "clock": pad_steps(clocks[:n]),
        "hold_ticks": pad_steps(holds[:n]),
        "tracking_ok": prediction.tracking_ok[:n],
        "group_id": group,
        "source_kind": kind,
    }
    return {key: value if key == "state" else value.cpu() for key, value in out.items()}


def merge(chunks):
    merged = {k: torch.cat([c[k] for c in chunks]) for k in chunks[0] if k != "state"}
    merged["state"] = tuple(
        torch.cat(fields) for fields in zip(*(c["state"] for c in chunks), strict=True)
    )
    return merged


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--starts", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--train", type=int, default=16384)
    p.add_argument("--val", type=int, default=2048)
    p.add_argument("--test", type=int, default=2048)
    p.add_argument("--batch", type=int, default=384)
    p.add_argument("--synthetic-starts", type=int, default=128)
    p.add_argument("--fd-fraction", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=6419)
    p.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    p.add_argument(
        "--reuse-splits-from",
        type=Path,
        help="Copy unselected split files and manifest entries from a compatible existing dataset",
    )

    p.add_argument(
        "--supplemental-retract-pairs",
        action="store_true",
        help="Independent test-only set of alternating pairs from actual held-out planner retractions",
    )
    args = p.parse_args()
    if args.supplemental_retract_pairs:
        if args.reuse_splits_from or args.test % 2:
            p.error("Use a separate supplemental test with equal counts for LR and RL")
        if (args.output / "manifest.json").exists() or (
            args.output / "test.pt"
        ).exists():
            p.error("Choose a fresh directory for supplemental test labels")
        args.splits = ["test"]
    if (
        args.batch < 1
        or args.synthetic_starts < 2
        or not (0 <= args.fd_fraction <= 1)
        or min(args.train, args.val, args.test) < 4
    ):
        p.error(
            "Use positive batch and split sizes with at least four rows per split and two synthetic starts"
        )
    records = torch.load(args.starts, map_location="cpu", weights_only=True)
    if isinstance(records, dict):
        records = records["records"]
    groups = sorted({r["group"] for r in records})
    for group in groups:
        splits = {r["split"] for r in records if r["group"] == group}
        if len(splits) != 1 or not splits.issubset(SPLITS):
            raise ValueError(
                f"Group must be assigned to exactly one split: {group}, {splits}"
            )
    groups += ["synthetic/" + s for s in SPLITS]
    teacher = LandingPredictor(args.bundle)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "teacher_sha256": teacher.metadata["checkpoint_sha256"],
        "deployment_sha256": teacher.metadata["deployment_sha256"],
        "starts_file_sha256": digest(args.starts),
        "group_names": groups,
        "source_kinds": SOURCE_NAMES,
        "knots": KNOTS,
        "hold_ticks": teacher.hold_ticks,
        "seed": args.seed,
        "fd_fraction": args.fd_fraction,
        "splits": {},
    }
    if args.supplemental_retract_pairs:
        captured = [r for r in records if r["split"] == "test" and record_kind(r) == 1]
        if not captured:
            p.error("No post-retraction starts in held-out scene records")
        values = [
            torch.cat([v.flatten(1) for v in (*r["tensors"], r["dynamics"])], 1)
            for r in captured
        ]
        manifest["sampling"] = {
            "population": "Held-out scene planner post-retraction issue states only; independent command draws.",
            "capture_groups": sorted({r["group"] for r in captured}),
            "source_records": len(captured),
            "captured_rows": sum(len(v) for v in values),
            "unique_initial_states_and_dynamics": int(
                torch.cat(values).unique(dim=0).shape[0]
            ),
            "captured_orders": sorted({r["order"] for r in captured}),
            "command_mixture": "67% local (captured common prefix if available, otherwise small native support-frame command), 28% training box, 5% relaxed box.",
            "test_rng_seed": args.seed + 202,
        }
    unselected = [split for split in SPLITS if split not in args.splits]
    if unselected and args.reuse_splits_from:
        prior = json.loads((args.reuse_splits_from / "manifest.json").read_text())
        for field in (
            "teacher_sha256",
            "deployment_sha256",
            "starts_file_sha256",
            "group_names",
            "source_kinds",
            "knots",
            "hold_ticks",
        ):
            if prior[field] != json.loads(json.dumps(manifest[field])):
                raise ValueError(f"Incompatible reused dataset field {field}")
        for split in unselected:
            source = args.reuse_splits_from / f"{split}.pt"
            dest = args.output / f"{split}.pt"
            if source.resolve() != dest.resolve():
                shutil.copyfile(source, dest)
            manifest["splits"][split] = prior["splits"][split]
    elif unselected and (args.output / "manifest.json").exists():
        p.error("Use --reuse-splits-from to preserve existing unselected splits")
    for index, split in enumerate(SPLITS):
        if split not in args.splits:
            continue
        begun = perf_counter()
        rng = torch.Generator(device=teacher.device).manual_seed(
            args.seed + index * 101
        )
        with torch.no_grad(), prediction_precision():
            chunks = []
            total = getattr(args, split)
            if args.supplemental_retract_pairs:
                synthetic = None
                orders = ("LR", "RL")
                counts = [total // 2] * 2
            else:
                synthetic = synthetic_bank(teacher, rng, split, args.synthetic_starts)
                orders = ORDERS
                counts = [max(1, round(total * 0.15))] * 2 + [
                    max(1, round(total * 0.35))
                ]
                counts.append(total - sum(counts))
            for order, count in zip(orders, counts, strict=True):
                bank = Bank(
                    records, synthetic, order, groups, split, device=teacher.device
                )
                for start in range(0, count, args.batch):
                    state, c, g, kind = bank.draw(
                        min(args.batch, count - start), rng, order
                    )
                    chunks.append(
                        label_batch(
                            teacher, state, order, c, g, kind, rng, args.fd_fraction
                        )
                    )
                print(f"{split} {order}: {count} labeled examples", flush=True)
            data = merge(chunks)
        summary = {
            "rows": len(data["sides"]),
            "order_counts": dict(zip(orders, counts)),
            "source_by_order": {
                order: {
                    SOURCE_NAMES[i]: int(v)
                    for i, v in enumerate(part.bincount().tolist())
                    if v
                }
                for order, part in zip(
                    orders, data["source_kind"].split(counts), strict=True
                )
            },
            "teacher_tracking_pass_fraction": float(data["tracking_ok"].float().mean()),
            "branch_valid_fd_steps": int(data["fd_valid"].sum()),
            "seconds": perf_counter() - begun,
            "group_counts": {
                groups[i]: v
                for i, v in enumerate(data["group_id"].bincount().tolist())
                if v
            },
            "source_counts": {
                SOURCE_NAMES[i]: v
                for i, v in enumerate(data["source_kind"].bincount().tolist())
                if v
            },
        }
        atomic_save(data, args.output / f"{split}.pt")
        manifest["splits"][split] = summary
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(f"{split}: {summary}", flush=True)


if __name__ == "__main__":
    main()
