"""Stand-in for booster_deploy.controllers.controller_cfg: the fields deploy/booster_t2 sets or reads."""

from dataclasses import MISSING, field
from typing import Callable, List, Optional

from ..utils.isaaclab.configclass import configclass


@configclass
class PrepareStateCfg:
    stiffness: List[float] = MISSING
    damping: List[float] = MISSING
    joint_pos: List[float] = MISSING


@configclass
class MujocoControllerCfg:
    init_pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.6])
    decimation: int = 10


@configclass
class BoosterRobotControllerCfg:
    exit_mode: str = "walking"


@configclass
class RobotCfg:
    name: str = MISSING
    prepare_mode: str = "walking"
    joint_names: List[str] = MISSING
    joint_stiffness: List[float] = MISSING
    joint_damping: List[float] = MISSING
    default_joint_pos: List[float] = MISSING
    effort_limit: List[float] = MISSING
    mjcf_path: str = MISSING
    prepare_state: PrepareStateCfg = MISSING


@configclass
class VelocityCommandCfg:
    vx_max: float = 1.0
    vx_forward_max: Optional[float] = None
    vx_backward_max: Optional[float] = None
    vy_max: float = 1.0
    vyaw_max: float = 1.0


@configclass
class PolicyCfg:
    constructor: Callable = MISSING
    checkpoint_path: str = MISSING
    enable_safety_fallback: bool = True
    device: str = "cpu"


@configclass
class ControllerCfg:
    policy_dt: float = 0.02
    robot: RobotCfg = MISSING
    vel_command: Optional[VelocityCommandCfg] = None
    policy: PolicyCfg = MISSING
    mujoco: MujocoControllerCfg = field(default_factory=MujocoControllerCfg)
    booster: BoosterRobotControllerCfg = field(default_factory=BoosterRobotControllerCfg)
