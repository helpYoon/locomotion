"""Policy observation as the robot's sensors produce it.

Observation layout (same as :meth:`src.mdp.MDP.obs`): q - q0 - offset, qd, angular velocity,
projected gravity, previous action, reference-relative root orientation (6D). :class:`SensingCfg`
is the paper's: true state plus uniform noise (Table A.1), laid out per dimension by :func:`obs_noise`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.dr import XiLayout
    from src.mdp import MDP
    from robot.spec import RobotSpec


@dataclass(frozen=True)
class SensingCfg:
    noise_joint_pos: float = 0.01
    noise_joint_vel: float = 0.5
    noise_ang_vel: float = 0.2
    noise_gravity: float = 0.05
    noise_ref_ori: float = 0.05
    """Uniform observation-noise half-widths (defaults: paper Table A.1)."""
    # ---- a measured robot's sensor errors (robot/t2_measured.py; off by default). p is the ramp (MDP.ramp)
    gyro_bias: float = 0.0
    """Per episode, a bias on each axis of the observed angular velocity, U(-gyro_bias p, gyro_bias p) [rad/s]."""
    imu_tilt: float = 0.0
    """Per episode, the IMU's mounting error: the orientation the observation uses is the root's turned about its
    roll and pitch axes by U(-imu_tilt p, imu_tilt p) [rad] each."""
    noise_scale_to: float = 1.0
    """The noise half-widths above are multiplied by 1 + (noise_scale_to - 1) p. 1 = no growth."""


def obs_noise(cfg: SensingCfg, J: int) -> torch.Tensor:
    return torch.cat(
        (
            torch.full((J,), cfg.noise_joint_pos),
            torch.full((J,), cfg.noise_joint_vel),
            torch.full((3,), cfg.noise_ang_vel),
            torch.full((3,), cfg.noise_gravity),
            torch.zeros(J),
            torch.full((6,), cfg.noise_ref_ori),
        )
    )


class ObsPipeline:
    """Device tables of the observation's sensor groups: delay group and delayed mask per dimension, joint-velocity
    filter time constants. Nothing reads them (the observation is :meth:`src.mdp.MDP.obs`); they stay because
    building them allocates and copies to the device.
    """

    def __init__(self, robot: RobotSpec, mdp: MDP, layout: XiLayout, device: str, physics_dt: float):
        J = robot.num_joints
        # delay group of every observation dim: q, qd, angular velocity, attitude (gravity + ref ori); -1 = exact
        group = [0] * J + [1] * J + [2] * 3 + [3] * 3 + [-1] * J + [3] * 6
        self.group = torch.tensor([max(g, 0) for g in group], device=device)
        self.delayed = torch.tensor([g >= 0 for g in group], device=device)
        tau = [0.0] * J
        self.qd_tau = torch.tensor(tau, device=device)
