"""Left-right (sagittal-plane) mirror of states, actions and sim context, for world-model data augmentation, and of
the velocity interface's inputs (observations, commands), for its mirror loss (train.py --vel_mirror_weight).

The mirror is the reflection M = diag(1, -1, 1) of the world y axis: positions and linear velocities
map to M x, rotations to M R M, angular velocities (pseudo-vectors) to -M w, and joints to their
left/right partner (RobotSpec.mirror_perm) with the signs of RobotSpec.mirror_sign.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from robot.spec import RobotSpec

if TYPE_CHECKING:
    from src.networks import Policy


class Mirror:
    def __init__(self, robot: RobotSpec, xi_slices: dict, device: str):
        perm, sign = list(robot.mirror_perm), [float(x) for x in robot.mirror_sign]
        self.perm = torch.tensor(perm, device=device)
        self.sign = torch.tensor(sign, device=device)
        self.M = torch.diag(torch.tensor([1.0, -1.0, 1.0], device=device))
        self.xi_slices = xi_slices
        # delay groups span both sides; so does the trunk's added mass
        unknown = set(xi_slices) - {"friction", "joint_offset", "com", "delay", "joint_friction", "kp", "kd", "trunk_mass"}
        assert not unknown, f"mirror of sim-context entries {unknown} not defined"
        # mdp.obs's layout: q - default - offset, qd, body angular velocity, gravity in the body frame, the previous
        # action (J each, but 3 for the two vectors), then the upright-relative orientation's 6D (X's first two
        # columns, row-major; M X M flips the entries with one index on y)
        J = len(perm)
        idx, sgn = list(range(3 * J + 12)), [1.0] * (3 * J + 12)
        for b in (0, J, 2 * J + 6):
            idx[b : b + J], sgn[b : b + J] = [b + i for i in perm], sign
        sgn[2 * J : 2 * J + 6] = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        sgn[3 * J + 6 :] = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0]
        self.obs_idx, self.obs_sign = torch.tensor(idx, device=device), torch.tensor(sgn, device=device)
        self.cmd_sign = torch.tensor([1.0, -1.0, -1.0], device=device)

    def joints(self, x: torch.Tensor) -> torch.Tensor:
        """Mirror of joint vectors; a trailing dim of k * J (a_prev with k past actions) mirrors each block."""
        return (x.unflatten(-1, (-1, len(self.perm)))[..., self.perm] * self.sign).flatten(-2)

    def obs(self, o: torch.Tensor) -> torch.Tensor:
        """Mirror of mdp.obs observations (..., obs_dim) of upright (velocity) rows, raw (before the policy's
        normalisation); any leading shape, e.g. a history (..., hist, obs_dim)."""
        return o[..., self.obs_idx] * self.obs_sign

    def cmd_n(self, c: torch.Tensor) -> torch.Tensor:
        """Mirror of commands (..., 3): (vx, vy, yaw rate) -> (vx, -vy, -yaw rate). Normalised commands too, where
        the command box is symmetric in vy and yaw (assert_mirrorable_box)."""
        return c * self.cmd_sign

    def state(self, S):
        p, R, v, w, q, qd = S
        M = self.M
        return (p @ M, M @ R @ M, v @ M, -(w @ M), self.joints(q), self.joints(qd))

    def xi(self, xi: torch.Tensor) -> torch.Tensor:
        out = xi.clone()
        sl = self.xi_slices["joint_offset"]
        out[..., sl] = self.joints(xi[..., sl])
        c = self.xi_slices["com"]
        out[..., c.start + 1] = -xi[..., c.start + 1]  # lateral COM offset
        for g in ("joint_friction", "kp", "kd"):  # per-joint factors: the partner joint's, no sign (they are positive)
            if g in self.xi_slices:
                sl = self.xi_slices[g]
                out[..., sl] = xi[..., sl][..., self.perm]
        return out


def assert_mirrorable_box(cmd_lo: torch.Tensor, cmd_hi: torch.Tensor):
    """Mirror.cmd_n maps normalised commands onto normalised commands only for a box symmetric in vy and yaw."""
    ok = torch.allclose(cmd_lo[1:], -cmd_hi[1:], atol=1e-6)
    assert ok, f"command box {cmd_lo.tolist()} .. {cmd_hi.tolist()} is not symmetric in vy and yaw"


def policy_mirror_error(pi: Policy, mirror: Mirror, cmd_n: torch.Tensor, o_hist: torch.Tensor) -> torch.Tensor:
    """Per row (B,): the mean squared difference between the velocity path's action in the mirrored situation
    (command and observation history mirrored) and the mirror of its action. Through the frozen decoder, so only the
    velocity encoder gets a gradient; pass the history detached."""
    mo = mirror.obs(o_hist)
    a = pi.decode_frozen(pi.encode_vel(cmd_n, o_hist), o_hist)
    a_m = pi.decode_frozen(pi.encode_vel(mirror.cmd_n(cmd_n), mo), mo)
    return ((a_m - mirror.joints(a)) ** 2).mean(-1)
