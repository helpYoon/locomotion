"""Stand-in for booster_deploy.controllers.base_controller: robot data as torch tensors in the real robot's joint
order, the velocity command, the Policy base class and a controller that builds them as the vendor's does."""

import inspect
import os

import torch

from .controller_cfg import ControllerCfg, PolicyCfg, RobotCfg, VelocityCommandCfg


class RobotData:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_quat_w: torch.Tensor
    root_ang_vel_b: torch.Tensor

    def __init__(self, cfg: RobotCfg) -> None:
        n = len(cfg.joint_names)
        self.joint_pos, self.joint_vel = torch.zeros(n), torch.zeros(n)
        self.root_ang_vel_b, self.root_quat_w = torch.zeros(3), torch.zeros(4)


class BoosterRobot:
    cfg: RobotCfg
    data: RobotData
    joint_stiffness: torch.Tensor
    joint_damping: torch.Tensor

    def __init__(self, cfg: RobotCfg) -> None:
        self.cfg, self.data = cfg, RobotData(cfg)
        self.joint_stiffness = torch.tensor(cfg.joint_stiffness, dtype=torch.float32)
        self.joint_damping = torch.tensor(cfg.joint_damping, dtype=torch.float32)


class VelocityCommand:
    lin_vel_x: float
    lin_vel_y: float
    ang_vel_yaw: float

    def __init__(self, cfg: VelocityCommandCfg) -> None:
        self.vx_forward_max = cfg.vx_forward_max if cfg.vx_forward_max is not None else cfg.vx_max
        self.vx_backward_max = cfg.vx_backward_max if cfg.vx_backward_max is not None else cfg.vx_max
        self.vy_max, self.vyaw_max = cfg.vy_max, cfg.vyaw_max
        self.lin_vel_x = self.lin_vel_y = self.ang_vel_yaw = 0.0


class Policy:
    def __init__(self, cfg: PolicyCfg, controller: "BaseController"):
        self.cfg, self.controller = cfg, controller
        self.task_path = os.path.dirname(inspect.getmodule(self.__class__).__file__)

    def reset(self) -> None:
        raise NotImplementedError

    def inference(self) -> torch.Tensor:
        raise NotImplementedError


class BaseController:
    cfg: ControllerCfg
    robot: BoosterRobot
    vel_command: VelocityCommand
    policy: Policy

    def __init__(self, cfg: ControllerCfg) -> None:
        self.cfg, self.is_running = cfg, False
        self.robot = BoosterRobot(cfg.robot)
        self.vel_command = VelocityCommand(cfg.vel_command) if cfg.vel_command is not None else None
        self.policy = self.cfg.policy.constructor(self.cfg.policy, self)

    def start(self):
        self.is_running = True
        self.policy.reset()

    def stop(self) -> None:
        self.is_running = False
