"""Compare exact-landing geometry and an optional TorchScript student on identical teacher rollouts.

All computation runs on CPU. XY errors use radial distance and yaw uses the
absolute shortest angle. Each active step endpoint uses its final labeled knot:
the deadline for an intermediate step and the end of HOLD for a final step.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

XY_M = 0.020
YAW_RAD = math.radians(1)


def sha256(path):
    with path.open("rb") as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
        return h.hexdigest()


def wrap(angle):
    return torch.where(
        angle.abs() <= math.pi, angle, torch.atan2(angle.sin(), angle.cos())
    )


def intended(feet, side, command):
    left = side == 0
    support = torch.where(left[:, None], feet[:, 1], feet[:, 0])
    x, y, angle = command.unbind(-1)
    heading = support[:, 2]
    c, s = heading.cos(), heading.sin()
    target = torch.stack(
        (
            support[:, 0] + c * x - s * y,
            support[:, 1] + s * x + c * y,
            wrap(heading + angle),
        ),
        -1,
    )
    pair = torch.where(
        left[:, None, None],
        torch.stack([target, support], 1),
        torch.stack([support, target], 1),
    )
    return torch.where((side >= 0)[:, None, None], pair, feet)


def geometry(feet, sides, commands):
    path = []
    for j in range(sides.shape[1]):
        feet = intended(feet, sides[:, j], commands[:, j])
        path.append(feet)
    return torch.stack(path, 1)


def pose_errors(predicted, truth):
    d = predicted - truth
    return d[..., :2].norm(dim=-1), wrap(d[..., 2]).abs()


def tracking_ratio(feet, sides, commands, predicted, nominal):
    """Worst normalized residual across both soles, all step handoffs and final nominal."""
    worst = torch.zeros(len(feet))
    previous = feet
    for j in range(sides.shape[1]):
        valid = sides[:, j] >= 0
        target = intended(previous, sides[:, j], commands[:, j])
        xy, yaw = pose_errors(predicted[:, j], target)
        step_ratio = torch.maximum(xy / XY_M, yaw / YAW_RAD).amax(-1)
        worst = torch.maximum(worst, torch.where(valid, step_ratio, 0))
        previous = torch.where(valid[:, None, None], predicted[:, j], previous)
    xy, yaw = pose_errors(previous, nominal[:, -1])
    return torch.maximum(worst, torch.maximum(xy / XY_M, yaw / YAW_RAD).amax(-1))


def quantiles(values):
    q = torch.quantile(values, torch.tensor([0.5, 0.95]))
    return {"median": q[0].item(), "p95": q[1].item()}


def ratio(a, b):
    return a / b if b else None


def summarize(subset, active, truth_pass, predicted_pass, xy_cm, yaw_deg):
    n = int(subset.sum())
    t = truth_pass[subset]
    p = predicted_pass[subset]
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    active = active[subset, :, None].expand(-1, -1, 2)
    xy = xy_cm[subset]
    yaw = yaw_deg[subset]
    return {
        "sequences": n,
        "active_sole_observations": int(active.sum()),
        "all_step_terminal_soles_xy_cm": quantiles(xy[active]),
        "all_step_terminal_soles_yaw_deg": quantiles(yaw[active]),
        "sequence_max_xy_cm": quantiles(xy.masked_fill(~active, 0).amax((1, 2))),
        "sequence_max_yaw_deg": quantiles(yaw.masked_fill(~active, 0).amax((1, 2))),
        "teacher_accepted": int(t.sum()),
        "predicted_accepted": int(p.sum()),
        "true_passes": int((p & t).sum()),
        "false_passes": fp,
        "false_rejections": fn,
        "false_passes_over_predicted_passes": ratio(fp, int(p.sum())),
        "false_passes_over_teacher_failures": ratio(fp, int((~t).sum())),
        "teacher_pass_recall": ratio(int((p & t).sum()), int(t.sum())),
    }


def strata(d, truth_ratio, source_names):
    sides = d["sides"]
    order_masks = {
        "L": (sides[:, 0] == 0) & (sides[:, 1] < 0),
        "R": (sides[:, 0] == 1) & (sides[:, 1] < 0),
        "LR": (sides[:, 0] == 0) & (sides[:, 1] == 1),
        "RL": (sides[:, 0] == 1) & (sides[:, 1] == 0),
    }
    masks = {
        "all": torch.ones_like(d["tracking_ok"]),
        "teacher_pass": truth_ratio < 1,
        "near_threshold_0.8_to_1.2": (truth_ratio >= 0.8) & (truth_ratio <= 1.2),
    }
    for order, mask in order_masks.items():
        masks[f"order/{order}"] = mask
    for i, name in enumerate(source_names):
        src = d["source_kind"] == i
        masks[f"source/{name}"] = src
        for order, mask in order_masks.items():
            masks[f"source/{name}/order/{order}"] = src & mask
    return {k: v for k, v in masks.items() if v.any()}


@torch.inference_mode()
def student_endpoints(model_file, d, batch_size, manifest):
    extra = {"step_sequence.json": ""}
    student = torch.jit.load(
        str(model_file), map_location="cpu", _extra_files=extra
    ).eval()
    metadata = json.loads(extra["step_sequence.json"])
    for field in ("teacher_sha256", "deployment_sha256", "knots", "hold_ticks"):
        if metadata[field] != manifest[field]:
            raise ValueError(f"Student and teacher split disagree on {field}")
    result = []
    for start in range(0, len(d["feet"]), batch_size):
        row = slice(start, start + batch_size)
        _, soles = student(
            tuple(t[row] for t in d["state"]),
            d["dynamics"][row],
            d["feet"][row],
            d["sides"][row],
            d["commands"][row],
        )
        result.append(soles[:, :, -1])
    return torch.cat(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Teacher split .pt shard beside manifest.json",
    )
    parser.add_argument(
        "--model", type=Path, help="Optional exported TorchScript student"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    manifest = json.loads((args.data.parent / "manifest.json").read_text())
    d = torch.load(args.data, weights_only=True, map_location="cpu")
    a = (d["feet"], d["sides"], d["commands"])
    active = d["sides"] >= 0
    truth = d["soles"][:, :, -1]
    nominal = geometry(*a)
    teacher_error = tracking_ratio(*a, truth, nominal)
    actual = teacher_error < 1
    if not torch.equal(actual, d["tracking_ok"]):
        n = int((actual != d["tracking_ok"]).sum())
        raise ValueError(
            f"Reconstructed per-step and final teacher gates disagree on {n} sequences"
        )
    masks = strata(d, teacher_error, manifest["source_kinds"])
    outputs = {"geometric": nominal}
    if args.model is not None:
        outputs["student"] = student_endpoints(args.model, d, args.batch, manifest)
    report = {
        "data": str(args.data),
        "data_sha256": sha256(args.data),
        "dataset_manifest_sha256": sha256(args.data.parent / "manifest.json"),
        "source_starts_sha256": manifest["starts_file_sha256"],
        "sampling": manifest.get("sampling"),
        "teacher_sha256": manifest["teacher_sha256"],
        "student": str(args.model) if args.model else None,
        "student_sha256": sha256(args.model) if args.model else None,
        "units": {"xy": "cm radial", "yaw": "degrees absolute wrapped"},
        "tracking_rule": "Strictly <2 cm AND <1 degree on both feet at every step terminal, plus final chained nominal.",
        "near_threshold": "Worst teacher residual / corresponding tolerance in [0.8, 1.2].",
        "models": {},
    }
    for name, prediction in outputs.items():
        if not prediction[active].isfinite().all():
            raise ValueError(f"Nonfinite {name} predicted terminals")
        passed = tracking_ratio(*a, prediction, nominal) < 1
        xy, yaw = pose_errors(prediction, truth)
        metrics = {
            key: summarize(mask, active, actual, passed, xy * 100, yaw.rad2deg())
            for key, mask in masks.items()
        }
        report["models"][name] = metrics
        print(name, json.dumps(metrics["all"]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
