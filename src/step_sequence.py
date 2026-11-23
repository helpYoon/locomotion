"""T1 planning student for a single step or an alternating pair. No simulator dependencies."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

State = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


def wrap_angle(angle: Tensor) -> Tensor:
    return torch.atan2(angle.sin(), angle.cos())


def yaw_rotation(angle: Tensor) -> Tensor:
    c, s = angle.cos(), angle.sin()
    z, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, one), -1).unflatten(-1, (3, 3))


def rotate(r: Tensor, vec: Tensor) -> Tensor:
    return (r @ vec.unsqueeze(-1)).squeeze(-1)


def planar_relative(feet: Tensor, frame: Tensor) -> Tensor:
    d = feet[..., :2] - frame[..., :2]
    c, s = frame[..., 2].cos(), frame[..., 2].sin()
    return torch.stack(
        (
            c * d[..., 0] + s * d[..., 1],
            -s * d[..., 0] + c * d[..., 1],
            wrap_angle(feet[..., 2] - frame[..., 2]),
        ),
        -1,
    )


def planar_world(local: Tensor, frame: Tensor) -> Tensor:
    c, s = frame[..., 2].cos(), frame[..., 2].sin()
    return torch.stack(
        (
            frame[..., 0] + c * local[..., 0] - s * local[..., 1],
            frame[..., 1] + s * local[..., 0] + c * local[..., 1],
            wrap_angle(local[..., 2] + frame[..., 2]),
        ),
        -1,
    )


def pick_foot(feet: Tensor, side: Tensor) -> Tensor:
    return torch.where((side == 0).unsqueeze(-1), feet[:, 0], feet[:, 1])


def clock_ticks(feet: Tensor, side: Tensor, cmd: Tensor) -> Tensor:
    support = pick_foot(feet, 1 - side)
    cur = planar_relative(pick_foot(feet, side), support)
    # The deterministic LandNet.clock used on issue; no training-time jitter.
    travel = (cmd[:, :2] - cur[:, :2]).square().sum(-1).add(1e-12).sqrt()
    travel = (
        travel + 0.1124 * wrap_angle(cmd[:, 2] - cur[:, 2]).square().add(1e-12).sqrt()
    )
    return (44.0 + 20.0 * travel).round().clamp(40.0, 62.0)


def orthonormal_rotation(raw: Tensor) -> Tensor:
    x = F.normalize(raw[..., 0], dim=-1, eps=1e-8)
    y0 = raw[..., 1]
    y = F.normalize(y0 - x * (y0 * x).sum(-1, keepdim=True), dim=-1, eps=1e-8)
    return torch.stack((x, y, torch.linalg.cross(x, y, dim=-1)), -1)


class StepSequence(nn.Module):
    """Predict six event-aligned knots per step in world coordinates.

    The first deadline and terminal knots have indices 3 and 5 respectively. On the
    first of two steps the deadline is also the terminal. Empty second slots are
    masked by the caller (side == -1). A shared transition cell never sees future
    commands when evaluating the first landing.
    """

    def __init__(self, encoder: nn.Module, width: int = 192):
        super().__init__()
        self.encoder = encoder.requires_grad_(False)
        self.initial = nn.Sequential(
            nn.Linear(311, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.cell = nn.Sequential(
            nn.Linear(width + 76 + 6 + 7, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.body_head = nn.Linear(width, 6 * 76)
        self.foot_head = nn.Linear(width, 6 * 2 * 3)
        for head in (self.body_head, self.foot_head):
            nn.init.normal_(head.weight, std=0.002)
            nn.init.zeros_(head.bias)
        self.register_buffer("phase", torch.tensor([0.07, 0.50, 0.93, 1.0, 1.0, 1.0]))
        self.register_buffer("foot_scale", torch.tensor([0.08, 0.08, 0.18]))
        self.register_buffer(
            "body_output_scale",
            torch.tensor(
                [0.20, 0.20, 0.12]
                + [0.45] * 9
                + [0.8] * 3
                + [2.5] * 3
                + [0.50] * 29
                + [6.0] * 29
            ),
        )
        self.register_buffer(
            "body_input_scale",
            torch.tensor(
                [3.0, 3.0, 1.0]
                + [1.0] * 9
                + [1.0] * 3
                + [0.4] * 3
                + [1.0] * 29
                + [0.2] * 29
            ),
        )

    @torch.jit.export
    def prepare(
        self, state: State, dynamics: Tensor, feet: Tensor, sides: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        p, r, v, w, q, qd, previous, hist, latch = state
        frame = pick_foot(feet, 1 - sides[:, 0])
        inv = yaw_rotation(-frame[:, 2])
        origin = torch.cat((frame[:, :2], torch.zeros_like(frame[:, :1])), -1)
        b0 = torch.cat(
            (
                rotate(inv, p - origin),
                (inv @ r).flatten(1),
                rotate(inv, v),
                rotate(inv, w),
                q,
                qd,
            ),
            -1,
        )
        feet0 = planar_relative(feet, frame[:, None])
        # The frozen latent consumes the full policy history. An additional difference
        # exposes motion across that window to the student even at similar latents.
        base = torch.cat(
            (
                b0 * self.body_input_scale,
                feet0.flatten(1),
                dynamics,
                previous * 0.25,
                latch,
                (hist[:, -1] - hist[:, 0]) * 0.25,
            ),
            -1,
        )
        return frame, b0, feet0, base

    @torch.jit.export
    def encode(
        self, history: Tensor, latch: Tensor, side: Tensor, command: Tensor
    ) -> Tensor:
        return self.encoder(history, latch, side, command)

    @torch.jit.export
    def from_features(
        self,
        frame: Tensor,
        b0: Tensor,
        feet0: Tensor,
        base: Tensor,
        latent: Tensor,
        sides: Tensor,
        commands: Tensor,
    ) -> tuple[Tensor, Tensor]:
        h = self.initial(torch.cat((base, latent), -1))
        b, feet = b0, feet0
        bodies = torch.jit.annotate(list[Tensor], [])
        footprints = torch.jit.annotate(list[Tensor], [])
        for j in range(2):
            side = sides[:, j].clamp_min(0)
            cmd = commands[:, j]
            final = (
                (sides[:, 1] < 0) if j == 0 else torch.ones_like(side, dtype=torch.bool)
            )
            clock = clock_ticks(feet, side, cmd)
            condition = torch.cat(
                (
                    cmd[:, :2] * 4,
                    cmd[:, 2:],
                    (side == 0)[:, None].to(b),
                    (side == 1)[:, None].to(b),
                    final[:, None].to(b),
                    ((clock - 50) / 10)[:, None],
                ),
                -1,
            )
            h = h + self.cell(
                torch.cat(
                    (h, b * self.body_input_scale, feet.flatten(1), condition), -1
                )
            )
            support = pick_foot(feet, 1 - side)
            target = planar_world(cmd, support)
            swing = pick_foot(feet, side)
            delta = target - swing
            delta = torch.cat((delta[:, :2], wrap_angle(delta[:, 2:])), -1)
            swing_path = swing[:, None] + self.phase[None, :, None] * delta[:, None]
            planted_path = support[:, None].expand_as(swing_path)
            f = torch.where(
                (side == 0)[:, None, None, None],
                torch.stack((swing_path, planted_path), 2),
                torch.stack((planted_path, swing_path), 2),
            )
            f = f + self.foot_head(h).reshape(-1, 6, 2, 3) * self.foot_scale
            bf = (
                b[:, None]
                + self.body_head(h).reshape(-1, 6, 76) * self.body_output_scale
            )
            p = torch.cat(
                (
                    bf[..., :2] + 0.5 * self.phase[None, :, None] * delta[:, None, :2],
                    bf[..., 2:3],
                ),
                -1,
            )
            r = orthonormal_rotation(bf[..., 3:12].unflatten(-1, (3, 3)))
            bf = torch.cat((p, r.flatten(-2), bf[..., 12:]), -1)
            # No inter-step HOLD. Also prevent a gradient through unobserved fake hold knots.
            f = torch.cat(
                (
                    f[:, :4],
                    torch.where(final[:, None, None, None], f[:, 4:], f[:, 3:4]),
                ),
                1,
            )
            bf = torch.cat(
                (bf[:, :4], torch.where(final[:, None, None], bf[:, 4:], bf[:, 3:4])), 1
            )
            bodies.append(bf)
            footprints.append(f)
            b, feet = bf[:, -1], f[:, -1]
        local = torch.stack(bodies, 1)
        fworld = planar_world(torch.stack(footprints, 1), frame[:, None, None, None])
        rot = yaw_rotation(frame[:, 2])[:, None, None]
        origin = torch.cat((frame[:, :2], torch.zeros_like(frame[:, :1])), -1)
        world = torch.cat(
            (
                rotate(rot, local[..., :3]) + origin[:, None, None],
                (rot @ local[..., 3:12].unflatten(-1, (3, 3))).flatten(-2),
                rotate(rot, local[..., 12:15]),
                rotate(rot, local[..., 15:18]),
                local[..., 18:],
            ),
            -1,
        )
        return world, fworld

    def forward(
        self,
        state: State,
        dynamics: Tensor,
        feet: Tensor,
        sides: Tensor,
        commands: Tensor,
    ) -> tuple[Tensor, Tensor]:
        frame, body, soles, context = self.prepare(state, dynamics, feet, sides)
        z = self.encode(state[7], state[8], sides[:, 0], commands[:, 0])
        return self.from_features(frame, body, soles, context, z, sides, commands)
