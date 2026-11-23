"""Robot description shared by the motion library, FK, MDP and simulator.

``robot/t1.py`` and ``robot/t2.py`` define each robot's ``SPEC``: the MJCF, joint/body names, actuators (an mjlab
``EntityCfg``) and its contact, mirror, action-delay and velocity-task settings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field
from typing import Callable

from mjlab.entity import EntityCfg

from src.dr import DomainRandCfg
from src.sensing import SensingCfg


@dataclass(frozen=True)
class RobotSpec:
    name: str
    mjcf_path: str
    """Robot MJCF; the simulated entity and the torch FK are both built from it."""
    joint_names: tuple[str, ...]
    """Actuated joints in motion-file / action order (must equal the MJCF joint order)."""
    tracked_bodies: tuple[str, ...]
    """Bodies in the tracking reward; the first one is the root (anchor) body."""
    default_joint_pos: dict[str, float]
    """Regex -> nominal joint position; actions are offsets from it."""
    action_scale: dict[str, float]
    """Joint name -> action scale (rad per unit action)."""
    com_body: str
    """Body whose center of mass is randomized (part of the sim context xi)."""
    com_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    robot_cfg: Callable[[], EntityCfg]
    dr: DomainRandCfg = field(default_factory=DomainRandCfg)
    """Simulator randomization (default: the paper's setup, which the T1 keeps)."""
    sensing: SensingCfg = field(default_factory=SensingCfg)
    """How the policy observation is sensed (default: true state plus Table A.1 noise)."""
    njmax: int = 500
    nconmax: int = 128
    """MuJoCo-Warp constraint / contact capacity per world."""
    solver: tuple[int, int, float, int] = (100, 50, 1e-6, 35)
    """Newton solver iterations, line-search iterations, tolerance and CCD iterations."""
    feet: tuple[str, ...] = ()
    """Foot bodies (in tracked_bodies) of the contact reward."""
    contact_z: float = 0.045
    """Foot-body origin height below which a foot counts as in contact."""
    mirror_perm: tuple[int, ...] | None = None
    mirror_sign: tuple[float, ...] | None = None
    """Left-right joint mirror (src/mirror.py): each joint's partner index and sign."""
    delay_groups: tuple[tuple[int, ...], ...] = ()
    """Joint indices per delay group: TrackingSim delays each group's position target by one lag (physics steps)
    drawn per episode from delay_range (eval: eval_delay), written to xi["delay"]."""
    delay_range: tuple[int, int] = (0, 0)
    eval_delay: int = 0
    velocity: dict = field(default_factory=dict)
    """src.mdp.VelocityCfg field -> this robot's value, applied by MDP.enable_velocity over the cfg it is given (the
    command box, fall height, ankle torque scale). A plain dict: importing src.mdp here would be an import cycle."""
    sent_gains: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    """(kp, kd) per joint in joint order as deployment sends them, where they differ from the simulated ones of
    pd_gains() (robot/t2_measured.py: the robot applies the waist and ankle kd through its linkages). None: not given."""

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def default_joint_pos_list(self) -> list[float]:
        out = []
        for name in self.joint_names:
            val = 0.0
            for pat, v in self.default_joint_pos.items():
                if re.fullmatch(pat, name):
                    val = v
            out.append(val)
        return out

    def action_scale_list(self) -> list[float]:
        return [self.action_scale[n] for n in self.joint_names]

    def pd_gains(self) -> tuple[list[float], list[float], list[float]]:
        """Stiffness, damping and effort limit of each joint's position actuator, in joint order."""
        kp, kd, lim = ([None] * self.num_joints for _ in range(3))
        for act in self.robot_cfg().articulation.actuators:
            for i, name in enumerate(self.joint_names):
                if any(re.fullmatch(pat, name) for pat in act.target_names_expr):
                    kp[i], kd[i], lim[i] = float(act.stiffness), float(act.damping), float(act.effort_limit)
        assert None not in kp, "a joint without a position actuator"
        return kp, kd, lim

    @property
    def n_prev_actions(self) -> int:
        """Past actions the world model sees: 2, as the delay_groups' lags of up to 8 physics steps reach a_{t-2}.
        a_prev holds max(1, n) actions, newest first."""
        return 2


ROBOTS = ("t1", "t2", "t2_measured")


def get_robot(name: str) -> RobotSpec:
    """Load ``robot.<name>.SPEC`` (robot/t1.py, robot/t2.py, robot/t2_measured.py)."""
    if name not in ROBOTS:
        raise ValueError(f"unknown robot {name!r}: one of {ROBOTS} (the G1 packages were removed)")
    import importlib  # here, not at the top: robot/t1.py and robot/t2.py import this module

    return importlib.import_module(f"robot.{name}").SPEC


def robot_numbers(spec: RobotSpec) -> dict:
    """The numbers a policy is trained against, for a run's saved arguments: simulated gains, effort limits, action
    scale, each joint's Coulomb friction in the model and the gains deployment sends (None where the spec gives none).
    Plain lists in joint order."""
    kp, kd, effort = spec.pd_gains()
    mjs = spec.robot_cfg().spec_fn()  # held in a name: the joints are views into it
    loss = {j.name: float(j.frictionloss) for j in mjs.joints}
    sent = spec.sent_gains
    return {"name": spec.name, "kp": kp, "kd": kd, "effort": effort, "action_scale": spec.action_scale_list(),
            "friction": [loss[n] for n in spec.joint_names],
            "sent_kp": None if sent is None else list(sent[0]), "sent_kd": None if sent is None else list(sent[1])}  # fmt: skip


def check_robot_numbers(spec: RobotSpec, saved: dict | None) -> None:
    """Refuse a checkpoint whose saved robot_numbers differ from ``spec``'s (another robot, or constants edited since
    it was trained): its policy would run on physics it never saw. ``saved`` None (an older checkpoint): no check."""
    if saved is None:
        return
    now = robot_numbers(spec)
    for key, old in saved.items():
        if key == "name":  # a label: the numbers decide
            continue
        new = now.get(key)
        if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
            bad = [i for i, (a, b) in enumerate(zip(old, new)) if abs(a - b) > 1e-9 * max(abs(a), abs(b), 1e-12)]
            if bad:
                i = bad[0]
                raise ValueError(f"robot numbers differ from the checkpoint's: {key} of {spec.joint_names[i]} is "
                                 f"{new[i]!r} for robot {spec.name!r}, the checkpoint was trained with {old[i]!r} "
                                 f"(robot {saved.get('name')!r})")  # fmt: skip
        elif old != new:
            raise ValueError(f"robot numbers differ from the checkpoint's: {key} is {new!r} for robot {spec.name!r}, "
                             f"the checkpoint was trained with {old!r}")  # fmt: skip
