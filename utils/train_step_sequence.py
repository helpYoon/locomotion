"""Distill the frozen T1 controller/world model into a causal differentiable step predictor."""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.step_sequence import (
    StepSequence,
    clock_ticks,
    pick_foot,
    planar_world,
    wrap_angle,
)

ROOT = Path(__file__).resolve().parents[1]


class IssueEncoder(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, history, latch, side, command):
        observation = self.policy.full(history)
        issue = torch.cat(
            (
                torch.ones_like(command[:, :1]),
                side[:, None].to(dtype=command.dtype),
                command,
            ),
            -1,
        )
        features, _ = self.policy.command(observation[:, -1], issue, latch)
        return self.policy.pi.encode_land(features, observation)


def load_encoder(args, sample):
    sys.path.insert(0, str(args.cutamp_root / "experiments"))
    from score_t1_landing_accuracy import Model

    teacher = Model(ROOT, args.teacher, args.device)
    policy = teacher.policy

    # Make vendor URDF geometry mobile together with the checkpoint. A Python list
    # used as an ATen index otherwise traces a fixed-device tensor construction.
    class PortableFK(nn.Module):
        forward = type(teacher.kin).forward

        def __init__(self, kin):
            super().__init__()
            self.rotations = kin.rotations
            for name in ("indices", "axis", "offset"):
                self.register_buffer(name, getattr(kin, name))

    policy.kin = PortableFK(teacher.kin)
    foot_indices = torch.as_tensor(policy.feet, device=args.device)
    del policy.feet
    policy.register_buffer("feet", foot_indices)
    # These three modules are unreachable in the initial-issue encoder.
    policy.pi.enc = policy.pi.dec = policy.pi.enc_vel = nn.Identity()
    exact = IssueEncoder(policy).eval().requires_grad_(False)
    inputs = (
        sample["state"][7],
        sample["state"][8],
        sample["sides"][:, 0],
        sample["commands"][:, 0],
    )
    traced = torch.jit.trace(
        exact, tuple(t[:2].to(args.device) for t in inputs), check_trace=False
    )
    # Trace at B=2, then verify on another batch size and both side values.
    index = torch.cat(
        (torch.arange(3), (sample["sides"][:, 0] == 1).nonzero()[:1, 0])
    ).unique()
    probe = tuple(t[index].to(args.device) for t in inputs)
    torch.testing.assert_close(traced(*probe), exact(*probe))

    def foot_geometry(body):
        # Match cuTAMP's body-derived lowest sole corner height and tilt in radians.
        batch_dims = body.shape[:-1]
        flat = body.reshape(-1, 76)
        phys = (
            flat[:, :3],
            flat[:, 3:12].reshape(-1, 3, 3),
            flat[:, 12:15],
            flat[:, 15:18],
            flat[:, 18:47],
            flat[:, 47:],
        )
        out = teacher.landing.soles(*teacher.kin.forward(*phys))
        return torch.cat(
            (out["pose"], out["h"][..., None], out["tilt"][..., None]), -1
        ).reshape(*batch_dims, 2, 5)

    return traced, foot_geometry


def read_split(folder, name, limit):
    data = torch.load(folder / (name + ".pt"), map_location="cpu", weights_only=True)
    if limit:
        idx = torch.linspace(
            0, len(data["commands"]) - 1, min(limit, len(data["commands"]))
        ).long()
        data = {
            k: tuple(x[idx] for x in v) if k == "state" else v[idx]
            for k, v in data.items()
        }
    return data


@torch.no_grad()
def cache(model, data, device):
    result = {k: v.to(device) for k, v in data.items() if k != "state"}
    state = tuple(t.to(device) for t in data["state"])
    side, command, direction = result["sides"], result["commands"], result["direction"]
    result["frame"], result["b0"], result["feet0"], result["base"] = model.prepare(
        state, result["dynamics"], result["feet"], side
    )
    for key, sign in (("latent", 0), ("positive", 1), ("negative", -1)):
        latents = []
        for i in range(0, len(command), 512):
            sl = slice(i, i + 512)
            latents.append(
                model.encode(
                    state[7][sl],
                    state[8][sl],
                    side[sl, 0],
                    command[sl, 0] + sign * direction[sl, 0],
                )
            )
        result[key] = torch.cat(latents)
    return result


def delta(a, b):
    d = a - b
    return torch.cat((d[..., :2], wrap_angle(d[..., 2:])), -1)


def predict(model, d, z, command):
    return model.from_features(
        d["frame"], d["b0"], d["feet0"], d["base"], d[z], d["sides"], command
    )


def mean_mask(value, mask):
    mask = mask.to(value).expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def huber(x):
    return F.smooth_l1_loss(x, torch.zeros_like(x), reduction="none", beta=0.7)


def derivative_prediction(model, d, nominal):
    cmd, direction = d["commands"], d["direction"]
    _, plus = predict(model, d, "positive", cmd + direction)
    _, minus = predict(model, d, "negative", cmd - direction)

    # Compare directions inside the nominal controller-clock branch. Changing
    # the number of ticks measures a jump in the horizon, not a local slope.
    def clocks(sole, command):
        a = clock_ticks(d["feet"], d["sides"][:, 0], command[:, 0])
        b = clock_ticks(sole[:, 0, -1], d["sides"][:, 1].clamp_min(0), command[:, 1])
        return torch.stack((a, b), 1)

    clock = clocks(nominal, cmd)
    pos = (clocks(plus, cmd + direction) == clock).long().cumprod(1).bool()
    neg = (clocks(minus, cmd - direction) == clock).long().cumprod(1).bool()
    mask = lambda valid: valid[:, :, None, None, None]
    result = torch.where(
        mask(pos & neg),
        0.5 * delta(plus, minus),
        torch.where(mask(pos), delta(plus, nominal), delta(nominal, minus)),
    )
    return result, pos | neg


def objective(model, d, fd_weight, fk, fk_weight):
    cmd = d["commands"]
    b, f = predict(model, d, "latent", cmd)
    active = (d["sides"] >= 0)[:, :, None]
    phase = cmd.new_tensor([0.5, 0.6, 0.7, 2.0, 0.7, 3.0])
    body_scale = cmd.new_tensor(
        [0.04] * 3 + [0.12] * 9 + [0.35] * 3 + [1.5] * 3 + [0.14] * 29 + [2.0] * 29
    )
    body = (huber((b - d["body"]) / body_scale) * phase[None, None, :, None]).mean(-1)
    foot_scale = cmd.new_tensor([0.012, 0.012, math.radians(1.0)])
    foot = huber(delta(f, d["soles"]) / foot_scale).mean((-1, -2))
    loss = mean_mask(foot * phase, active) + 0.7 * mean_mask(body, active)
    if fk_weight:
        fk_pred = fk(b[:, :, [1, 3, 5]])
        fk_target = d["geometry"][:, :, [1, 3, 5]]
        # Use exact chain derivatives to keep the independently decoded leg/body
        # poses faithful to support-floor contact and the supervised sole path.
        pose = delta(fk_pred[..., :3], fk_target[..., :3]) / foot_scale
        height_tilt = (fk_pred[..., 3:] - fk_target[..., 3:]) / cmd.new_tensor(
            [0.005, 0.05]
        )
        loss = loss + fk_weight * mean_mask(
            huber(torch.cat((pose, height_tilt), -1)).mean((-1, -2)), active
        )
    if fd_weight:
        idx = d["fd_valid"].any(1).nonzero().flatten()
        d = {k: v[idx] for k, v in d.items()}
        if len(d["commands"]):
            deriv, relevant = derivative_prediction(model, d, f[idx])
            err = deriv - d["fd_soles"]
            fd_scale = cmd.new_tensor([0.001, 0.001, math.radians(0.1)])
            fd = huber(err / fd_scale).mean((-1, -2)) * phase
            loss = loss + fd_weight * mean_mask(
                fd, (d["fd_valid"] & relevant)[:, :, None]
            )
    return loss


def intended(feet, side, cmd):
    moved = planar_world(cmd, pick_foot(feet, 1 - side))
    return torch.where(
        torch.arange(2, device=cmd.device)[None, :, None] == side[:, None, None],
        moved[:, None],
        feet,
    )


def within(feet, reference):
    error = delta(feet, reference)
    return (
        (error[..., :2].norm(dim=-1) < 0.02) & (error[..., 2].abs() < math.radians(1.0))
    ).all(1)


def tracking(d, predicted):
    actual, ideal = d["feet"], d["feet"]
    ok = torch.ones(len(actual), dtype=torch.bool, device=actual.device)
    for j in range(2):
        side = d["sides"][:, j]
        valid = side >= 0
        achieved = predicted[:, j, 5]
        ok &= ~valid | within(achieved, intended(actual, side, d["commands"][:, j]))
        actual = torch.where(valid[:, None, None], achieved, actual)
        ideal = torch.where(
            valid[:, None, None], intended(ideal, side, d["commands"][:, j]), ideal
        )
    return ok & within(actual, ideal)


def breakdown(d, b, f, df, fk_err, eligible, rows):
    valid = (d["sides"] >= 0) & rows[:, None]
    foot_error = delta(f, d["soles"])
    xy = foot_error[..., :2].norm(dim=-1)[:, :, -1].amax(-1)[valid] * 100
    yaw = foot_error[..., 2].abs()[:, :, -1].amax(-1)[valid].rad2deg()
    pred_ok = tracking(d, f) & rows
    truth = d["tracking_ok"] & rows
    masks = d["fd_valid"] & valid & eligible
    d_pred = df.flatten(2)[masks]
    d_true = d["fd_soles"].flatten(2)[masks]
    nonzero = d_true.norm(dim=-1) > 1.0e-5
    cos = F.cosine_similarity(d_pred[nonzero], d_true[nonzero], dim=-1)
    pos = (b[..., :3] - d["body"][..., :3]).norm(dim=-1)[valid]
    joint = b[..., 18:47] - d["body"][..., 18:47]
    probs = torch.tensor([0.5, 0.9, 0.95], device=xy.device)

    def quant(values):
        return (
            [float(x) for x in torch.quantile(values, probs)]
            if len(values)
            else [None] * 3
        )

    fp = int((pred_ok & ~truth).sum())
    return {
        "rows": int(rows.sum()),
        "steps": int(valid.sum()),
        "endpoint_worst_foot_xy_cm_p50_p90_p95": quant(xy),
        "endpoint_worst_foot_yaw_deg_p50_p90_p95": quant(yaw),
        "knot_root_xy_and_z_error_cm_rmse": float(pos.square().mean().sqrt() * 100),
        "knot_joint_error_rad_rmse": float(joint[valid].square().mean().sqrt()),
        "terminal_body_fk_sole_height_error_mm_p50_p90_p95": quant(
            fk_err[:, :, -1, :, 3].abs().amax(-1)[valid] * 1000
        ),
        "terminal_body_fk_sole_tilt_error_deg_p50_p90_p95": quant(
            fk_err[:, :, -1, :, 4].abs().amax(-1)[valid].rad2deg()
        ),
        "terminal_body_fk_sole_xy_error_cm_p50_p90_p95": quant(
            fk_err[:, :, -1, :, :2].norm(dim=-1).amax(-1)[valid] * 100
        ),
        "teacher_pass": int(truth.sum()),
        "student_pass": int(pred_ok.sum()),
        "false_accepts": fp,
        "false_rejects": int((truth & ~pred_ok).sum()),
        "false_fraction_among_student_passes": fp / max(1, int(pred_ok.sum())),
        "fd_steps_nonzero": len(cos),
        "fd_cosine_mean": float(cos.mean()) if len(cos) else None,
    }


@torch.no_grad()
def evaluate(model, d, fk, detail=False):
    batches = []
    for idx in torch.arange(len(d["commands"]), device=d["commands"].device).split(512):
        part = {k: v[idx] for k, v in d.items()}
        b, f = predict(model, part, "latent", part["commands"])
        if detail:
            df, eligible = derivative_prediction(model, part, f)
            fp, ft = fk(b), fk(part["body"])
            fk_err = torch.cat(
                (delta(fp[..., :3], ft[..., :3]), fp[..., 3:] - ft[..., 3:]), -1
            )
        else:
            df, eligible, fk_err = torch.zeros_like(f), part["fd_valid"], f
        batches.append((b, f, df, eligible, fk_err))
    b, f, df, eligible, fk_err = (torch.cat(field) for field in zip(*batches))
    valid = d["sides"] >= 0
    end = delta(f[:, :, -1], d["soles"][:, :, -1])[valid]
    # Use metric units and select for endpoint error on both axes. Huber-like
    # clipping limits the selection influence of isolated teacher falls.
    score = float(
        (end[..., :2].norm(dim=-1) / 0.02).clamp_max(8).mean()
        + (end[..., 2].abs() / math.radians(1.0)).clamp_max(8).mean()
    )
    if not detail:
        return score, {}
    always = torch.ones_like(valid[:, 0])
    masks = {"all": always}
    for side, name in enumerate(("L", "R")):
        masks[name] = (d["sides"][:, 0] == side) & ~valid[:, 1]
        masks[name + ("R" if side == 0 else "L")] = (d["sides"][:, 0] == side) & valid[
            :, 1
        ]
    for i, name in enumerate(("home", "retract", "planner_other", "synthetic")):
        masks[name] = d["source_kind"] == i
    return score, {
        name: breakdown(d, b, f, df, fk_err, eligible, rows)
        for name, rows in masks.items()
        if rows.any()
    }


def endpoint_derivatives(model, data, device):
    """Reverse-mode endpoint command derivatives vs branch-matched teacher probes.

    Compare with perfect command tracking and also remove that geometric component
    from both Jacobians to isolate the sensitivity of controller tracking error.
    """
    keys = ("teacher", "student", "geometry", "active", "probe_step")
    pieces = {key: [] for key in keys}
    rows = data["fd_valid"].any(1).nonzero().flatten()
    for idx in rows.split(128):
        if not idx.numel():
            continue
        state = tuple(t[idx].to(device) for t in data["state"])
        feet = data["feet"][idx].to(device)
        sides = data["sides"][idx].to(device)
        commands = data["commands"][idx].to(device).requires_grad_()
        direction = data["direction"][idx].to(device)
        _, soles = model(state, data["dynamics"][idx].to(device), feet, sides, commands)
        ideal = feet
        stages = []
        for j in range(2):
            ideal = intended(ideal, sides[:, j], commands[:, j])
            stages.append(ideal)
        nominal = torch.stack(stages, 1)
        values = {"geometry": nominal, "student": soles[:, :, -1]}
        for key, tensor in values.items():
            coords = []
            for coordinate in tensor.flatten(1).unbind(1):
                grad = torch.autograd.grad(
                    coordinate.sum(), commands, retain_graph=True
                )[0]
                coords.append((grad * direction).sum((1, 2)))
            pieces[key].append(torch.stack(coords, 1).reshape_as(nominal).detach())
        pieces["teacher"].append(data["fd_soles"][idx, :, -1].to(device))
        pieces["active"].append(data["fd_valid"][idx].to(device))
        pieces["probe_step"].append(direction.abs().sum(-1).argmax(1))
    if not pieces["teacher"]:
        return {}
    d = {key: torch.cat(parts) for key, parts in pieces.items()}
    masks = {
        "all": d["active"],
        "later_landing_wrt_first_command": d["active"]
        & (d["probe_step"] == 0)[:, None]
        & (torch.arange(2, device=device) == 1)[None, :],
    }
    scale = d["student"].new_tensor([0.001, 0.001, math.radians(0.1)])
    report = {}
    for key, mask in masks.items():
        a, b, c = (
            (d[n][mask] / scale).flatten(1) for n in ("teacher", "student", "geometry")
        )

        def cos(x, y):
            nonzero = (x.norm(dim=1) > 1.0e-4) & (y.norm(dim=1) > 1.0e-4)
            return (
                float(F.cosine_similarity(x[nonzero], y[nonzero]).mean())
                if nonzero.any()
                else None
            )

        if len(a):
            report[key] = {
                "probes": len(a),
                "teacher_student_cosine": cos(a, b),
                "teacher_geometry_cosine": cos(a, c),
                "tracking_error_gradient_cosine": cos(a - c, b - c),
                "student_normalized_rmse": float((a - b).square().mean().sqrt()),
                "geometry_normalized_rmse": float((a - c).square().mean().sqrt()),
            }
    return report


def atomic_save(state, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def export(model, out, manifest, epoch, score):
    meta = {
        "schema_version": 1,
        "teacher_sha256": manifest["teacher_sha256"],
        "deployment_sha256": manifest["deployment_sha256"],
        "knots": manifest["knots"],
        "hold_ticks": manifest["hold_ticks"],
        "epoch": epoch,
        "validation_score": score,
        "side_order": ["L", "R"],
        "padding": -1,
    }
    js = torch.jit.script(model.eval())
    tmp = out / "step_sequence.pt.tmp"
    torch.jit.save(js, str(tmp), _extra_files={"step_sequence.json": json.dumps(meta)})
    tmp.replace(out / "step_sequence.pt")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument(
        "--teacher", type=Path, default=ROOT / "logs/land_r4_d10_al2_s3/ckpt_004000.pt"
    )
    p.add_argument("--cutamp-root", type=Path, default=ROOT.parent / "cuTAMP")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.0015)
    p.add_argument("--fd-weight", type=float, default=0.5)
    p.add_argument(
        "--fk-weight",
        type=float,
        default=0.0,
        help="Additional vendor-URDF lower-body pose loss for fine-tuning",
    )
    p.add_argument("--seed", type=int, default=2089)
    p.add_argument("--limit", type=int, default=0, help="Small smoke test only")
    p.add_argument(
        "--resume",
        type=Path,
        help="Initialize from a best checkpoint and start a new learning-rate schedule",
    )
    p.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Open the locked test split after final checkpoint selection",
    )
    p.add_argument(
        "--export-only",
        action="store_true",
        help="Repackage --resume with the verified frozen encoder, preserving chosen epoch and score",
    )
    p.add_argument(
        "--evaluate-bundle",
        type=Path,
        help="Evaluate this exact serialized bundle without training or re-exporting",
    )
    args = p.parse_args()
    if args.export_only and (not args.resume or args.evaluate_bundle):
        p.error(
            "--export-only requires --resume and cannot be used with --evaluate-bundle"
        )
    if (
        args.batch_size <= 0
        or args.epochs < 0
        or args.width <= 0
        or args.limit < 0
        or (not args.epochs and not args.resume)
    ):
        p.error(
            "Use positive batch-size/width and epochs, or pass --resume with --epochs 0"
        )
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    manifest = json.loads((args.data / "manifest.json").read_text())
    with args.teacher.open("rb") as src:
        if hashlib.file_digest(src, "sha256").hexdigest() != manifest["teacher_sha256"]:
            raise ValueError("The checkpoint does not match the teacher dataset")
    raw = read_split(
        args.data,
        "train",
        32 if args.export_only or args.evaluate_bundle else args.limit,
    )
    encoder, fk = load_encoder(args, raw)
    model = StepSequence(encoder, width=args.width).to(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    best = args.output / "best.pt"
    if args.evaluate_bundle:
        extra = {"step_sequence.json": ""}
        model = torch.jit.load(
            str(args.evaluate_bundle), map_location=args.device, _extra_files=extra
        ).eval()
        metadata = json.loads(extra["step_sequence.json"])
        if (
            metadata["teacher_sha256"] != manifest["teacher_sha256"]
            or metadata["deployment_sha256"] != manifest["deployment_sha256"]
        ):
            raise ValueError("Bundle teacher mismatch")
        with args.evaluate_bundle.open("rb") as archive:
            report = {
                "bundle_sha256": hashlib.file_digest(archive, "sha256").hexdigest(),
                "metadata": metadata,
                "splits": {},
            }
        for split in ("val", "test") if args.evaluate_test else ("val",):
            raw_split = read_split(args.data, split, args.limit)
            data = cache(model, raw_split, args.device)
            report["splits"][split] = evaluate(model, data, fk, True)[1]
            report.setdefault("endpoint_reverse_mode", {})[split] = (
                endpoint_derivatives(model, raw_split, args.device)
            )
        (args.output / "bundle-metrics.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            json.dumps({k: v["all"] for k, v in report["splits"].items()}), flush=True
        )
        return
    epoch_offset = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=True)
        if (
            ck["manifest_sha256"] != manifest["teacher_sha256"]
            or ck["width"] != args.width
        ):
            raise ValueError("Checkpoint mismatch for teacher or student width")
        # Rebuild the frozen encoder entirely from the SHA-verified teacher. Earlier
        # exports used embedded FK constants in place of named buffers.
        current = model.state_dict()
        heads = {k: v for k, v in ck["model"].items() if not k.startswith("encoder.")}
        if heads.keys() != {k for k in current if not k.startswith("encoder.")}:
            raise ValueError("Unexpected student checkpoint architecture")
        model.load_state_dict({**current, **heads})
        epoch_offset = int(ck["epoch"])
        if args.export_only:
            export(model, args.output, manifest, epoch_offset, float(ck["score"]))
            return
    train = cache(model, raw, args.device)
    if args.fk_weight:
        with torch.no_grad():
            train["geometry"] = torch.cat(
                [fk(chunk) for chunk in train["body"].split(512)]
            )
    del raw
    val = cache(model, read_split(args.data, "val", args.limit), args.device)
    best_score = evaluate(model, val, fk)[0]
    best_epoch = epoch_offset

    def save_best():
        atomic_save(
            {
                "model": model.state_dict(),
                "epoch": best_epoch,
                "score": best_score,
                "width": args.width,
                "manifest_sha256": manifest["teacher_sha256"],
            },
            best,
        )
        export(model, args.output, manifest, best_epoch, best_score)

    save_best()
    print(
        f"Prepared {len(train['commands'])} train, {len(val['commands'])} validation rows, starting score {best_score:.4f}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(1, args.epochs), eta_min=args.lr * 0.06
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        for i in torch.randperm(len(train["commands"]), device=args.device).split(
            args.batch_size
        ):
            d = {k: v[i] for k, v in train.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, d, args.fd_weight, fk, args.fk_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            score, _ = evaluate(model, val, fk)
            if score < best_score:
                best_score, best_epoch = score, epoch_offset + epoch
                save_best()
            print(
                f"epoch {epoch:3d}  val {score:.4f}  best {best_score:.4f}  elapsed {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    model.load_state_dict(
        torch.load(best, map_location=args.device, weights_only=True)["model"]
    )
    summary = {
        "best_epoch": best_epoch,
        "selection_score": best_score,
        "teacher_sha256": manifest["teacher_sha256"],
        "device": str(args.device),
        "epochs": args.epochs,
        "hyperparameters": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "width": args.width,
            "fd_weight": args.fd_weight,
            "fk_weight": args.fk_weight,
            "seed": args.seed,
        },
        "splits": {"val": evaluate(model, val, fk, True)[1]},
    }
    if args.evaluate_test:
        test = cache(model, read_split(args.data, "test", args.limit), args.device)
        summary["splits"]["test"] = evaluate(model, test, fk, True)[1]
    summary["wall_seconds"] = time.perf_counter() - started
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({split: res["all"] for split, res in summary["splits"].items()}),
        flush=True,
    )


if __name__ == "__main__":
    main()
