"""CPU regressions for the causal T1 student. No checkpoint or simulator required."""

import torch
from torch import nn

from src.step_sequence import StepSequence, planar_world, rotate, yaw_rotation


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(93 + 8 + 3 + 1, 32)

    def forward(self, history, latch, side, command):
        x = torch.cat((history[:, -1], latch, side[:, None].to(command), command), -1)
        return self.linear(x).tanh()


def inputs(batch):
    generator = torch.Generator().manual_seed(233)

    def noise(*shape):
        return 0.05 * torch.randn(batch, *shape, generator=generator)

    state = (
        noise(3),
        yaw_rotation(noise(1)[:, 0]),
        noise(3),
        noise(3),
        noise(29),
        noise(29),
        noise(58),
        noise(10, 93),
        noise(8),
    )
    feet = noise(2, 3) + torch.tensor([[[0.0, 0.10, 0.0], [0.0, -0.10, 0.0]]])
    side = torch.arange(batch) % 2
    sides = torch.stack((side, 1 - side), 1)
    commands = (
        torch.tensor([[0.12, 0.21, 0.04], [0.09, -0.22, -0.05]])
        .expand(batch, -1, -1)
        .clone()
    )
    commands[:, :, 1:] *= (1 - 2 * side)[:, None, None]
    return state, noise(38), feet, sides, commands


def test_script_export_matches_and_supports_variable_batches(tmp_path):
    torch.manual_seed(1932)
    model = StepSequence(ToyEncoder(), 32).eval()
    path = str(tmp_path / "student.pt")
    torch.jit.script(model).save(path)
    restored = torch.jit.load(path).eval()
    for batch in (1, 7):
        data = inputs(batch)
        # Pad the final row as a single step, mixing both sequence lengths.
        data[3][-1, 1] = -1
        actual = restored(*data)
        for pred, expected in zip(actual, model(*data), strict=True):
            torch.testing.assert_close(pred, expected)
        body, feet = actual
        assert body.shape == (batch, 2, 6, 76) and feet.shape == (batch, 2, 6, 2, 3)
        rotation = body[..., 3:12].reshape(batch, 2, 6, 3, 3)
        torch.testing.assert_close(
            rotation.mT @ rotation,
            torch.eye(3).expand_as(rotation),
            atol=2e-6,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            torch.linalg.det(rotation),
            torch.ones_like(rotation[..., 0, 0]),
            atol=2e-6,
            rtol=2e-5,
        )


def test_pair_is_causal_and_differentiable_with_no_intermediate_hold():
    torch.manual_seed(1932)
    model = StepSequence(ToyEncoder(), 32).eval()
    state, context, feet, sides, cmd = inputs(5)
    state[7].requires_grad_()
    cmd.requires_grad_()
    body, soles = model(state, context, feet, sides, cmd)
    for v in (body, soles):
        torch.testing.assert_close(v[:, 0, 4], v[:, 0, 3], atol=0, rtol=0)
        torch.testing.assert_close(v[:, 0, 5], v[:, 0, 3], atol=0, rtol=0)
    change = cmd.detach().clone()
    change[:, 1] += torch.tensor([0.05, 0.02, -0.04])
    b1, f1 = model(state, context, feet, sides, change)
    torch.testing.assert_close(b1[:, 0], body[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(f1[:, 0], soles[:, 0], atol=0, rtol=0)
    first_grad = torch.autograd.grad(soles[:, 0].sum(), cmd, retain_graph=True)[0]
    assert torch.count_nonzero(first_grad[:, 1]) == 0
    grad, history_grad = torch.autograd.grad(
        soles[:, 1, -1].square().sum(), (cmd, state[7])
    )
    assert (grad.abs().sum((0, 2)) > 0).all()
    assert torch.isfinite(history_grad).all() and history_grad.abs().sum() > 0
    for j in (0, 1):
        perturb = torch.zeros_like(cmd)
        perturb[:, j, 0] = 0.0002
        plus = (
            model(state, context, feet, sides, cmd.detach() + perturb)[1][:, 1, -1]
            .square()
            .sum()
        )
        minus = (
            model(state, context, feet, sides, cmd.detach() - perturb)[1][:, 1, -1]
            .square()
            .sum()
        )
        torch.testing.assert_close(
            grad[:, j, 0].sum(), (plus - minus) / 0.0004, atol=0.002, rtol=0.005
        )


def test_rigid_planar_scene_transform_preserves_prediction():
    model = StepSequence(ToyEncoder(), 32).eval()
    state, context, feet, sides, cmd = inputs(3)
    body, soles = model(state, context, feet, sides, cmd)
    frame = torch.tensor([0.7, -0.2, 1.2]).expand(3, -1)
    rot = yaw_rotation(frame[:, 2])
    shift = torch.cat((frame[:, :2], torch.zeros_like(frame[:, :1])), 1)
    p, r, v, w, *other = state
    moved = (rotate(rot, p) + shift, rot @ r, rotate(rot, v), rotate(rot, w), *other)
    new_body, new_soles = model(
        moved, context, planar_world(feet, frame[:, None]), sides, cmd
    )
    torch.testing.assert_close(
        new_soles, planar_world(soles, frame[:, None, None, None])
    )
    rot = rot[:, None, None]
    torch.testing.assert_close(
        new_body[..., :3], rotate(rot, body[..., :3]) + shift[:, None, None]
    )
    torch.testing.assert_close(
        new_body[..., 3:12], (rot @ body[..., 3:12].unflatten(-1, (3, 3))).flatten(-2)
    )
    torch.testing.assert_close(new_body[..., 12:15], rotate(rot, body[..., 12:15]))
    torch.testing.assert_close(new_body[..., 15:18], rotate(rot, body[..., 15:18]))
    torch.testing.assert_close(new_body[..., 18:], body[..., 18:])
