"""Simulator domain randomization and the layout of the sim context xi.

Every randomized simulator parameter is part of xi, which the world model and critic see (the
policy does not). The terms are the paper's setup: friction once per env, joint (encoder) offsets
per episode, and a CoM offset of ``RobotSpec.com_body`` once per env; and the per-group actuator
delays of ``RobotSpec.delay_groups`` per episode. A measured robot (robot/t2_measured.py) adds, through
``DomainRandCfg``'s optional fields: a factor per joint on the joint friction and on kp and kd per episode, and
link masses and mass on the trunk once per env (the link masses alone are not in xi).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from robot.spec import RobotSpec


def ramp_at(iteration: int, ramp_iters: int) -> float:
    """The ramp p of train.py --dr_ramp_iters at an iteration: 0 at the start, 1 from ramp_iters on (0 = no ramp:
    always 1). It scales the gain randomization, the sensor errors and the observation noise's growth (MDP.ramp)."""
    return 1.0 if ramp_iters <= 0 else min(1.0, iteration / ramp_iters)


@dataclass(frozen=True)
class DomainRandCfg:
    friction_range: tuple[float, float] = (0.3, 1.2)
    """Contact friction, sampled once per env."""
    joint_offset: dict[str, float] = field(default_factory=lambda: {".*": 0.01})
    """Joint regex -> half-range (rad) of the encoder offset, per episode: the policy reads q + bias and
    the PD target moves by -bias, as on a robot whose encoder zero is off."""
    reset_pos: tuple[float, float, float] = (0.05, 0.05, 0.01)
    reset_rot: tuple[float, float, float] = (0.1, 0.1, 0.2)
    reset_lin_vel: tuple[float, float, float] = (0.5, 0.5, 0.2)
    reset_ang_vel: tuple[float, float, float] = (0.52, 0.52, 0.78)
    reset_joint_pos: float = 0.1
    """Half-ranges of the offsets from the reference state at episode start."""
    # ---- robot-side randomization of a measured robot (robot/t2_measured.py; all off by default: the T1 and the T2
    # keep the paper's setup above). Each one that is on adds its group to xi, after the paper's four
    joint_friction_scale: tuple[float, float] | None = None
    """Per episode, a factor per joint on the model's Coulomb joint friction (xi "joint_friction")."""
    gain_scale: float = 0.0
    """Per episode, kp and kd of every joint x U(1 - gain_scale p, 1 + gain_scale p), a factor per joint each (xi "kp",
    "kd"); p is the ramp (MDP.ramp). 0 = off."""
    link_mass_scale: tuple[float, float] | None = None
    """Once per env, the mass of every link but RobotSpec.com_body x a factor per link, inertia unchanged (not in xi)."""
    trunk_mass_add: tuple[float, float] | None = None
    """Once per env, mass [kg] added to RobotSpec.com_body at its centre of mass, inertia unchanged (xi "trunk_mass")."""


def _match(patterns: dict[str, float], names, default: float = 0.0) -> list[float]:
    out = []
    for n in names:
        val = default
        for pat, v in patterns.items():
            if re.fullmatch(pat, n):
                val = v
        out.append(val)
    return out


class XiLayout:
    """Slices of xi and its normalization (center, half-range) for the networks."""

    def __init__(self, robot: RobotSpec, device: str):
        c = robot.dr
        J = robot.num_joints
        self.slices: dict[str, slice] = {}
        centers, scales = [], []

        def add(name: str, center: list[float], scale: list[float]):
            n = sum(sl.stop - sl.start for sl in self.slices.values())
            self.slices[name] = slice(n, n + len(center))
            centers.extend(center), scales.extend(scale)

        lo, hi = c.friction_range
        add("friction", [(lo + hi) / 2], [(hi - lo) / 2])
        add("joint_offset", [0.0] * J, [max(h, 1e-6) for h in _match(c.joint_offset, robot.joint_names)])
        add("com", [0.0] * 3, [0.05] * 3)  # fixed scale (the paper setup's normalization)
        lo, hi = robot.delay_range
        add("delay", [(lo + hi) / 2] * len(robot.delay_groups), [max((hi - lo) / 2, 1e-6)] * len(robot.delay_groups))
        # a measured robot's groups, after the paper's four: xi[:, 1 : 1 + J] is read by position (src/env.py, src/mdp.py)
        if c.joint_friction_scale is not None:
            lo, hi = c.joint_friction_scale
            add("joint_friction", [(lo + hi) / 2] * J, [(hi - lo) / 2] * J)
        if c.gain_scale > 0:
            add("kp", [1.0] * J, [c.gain_scale] * J)
            add("kd", [1.0] * J, [c.gain_scale] * J)
        if c.trunk_mass_add is not None:
            lo, hi = c.trunk_mass_add
            add("trunk_mass", [(lo + hi) / 2], [(hi - lo) / 2])
        self.dim = len(centers)
        self.center = torch.tensor(centers, device=device)
        self.scale = torch.tensor(scales, device=device)

    def normalize(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.center) / self.scale


class DomainRandomizer:
    """Samples and applies the per-env simulator parameters of ``robot.dr`` and writes them into xi.

    ``env`` is the :class:`src.env.TrackingSim` (it carries ``sim``, ``robot``, ``num_envs`` and ``device``).
    """

    def __init__(self, robot: RobotSpec, layout: XiLayout, env, eval_mode: bool):
        self.cfg, self.spec, self.layout, self.env = robot.dr, robot, layout, env
        self.eval_mode = eval_mode
        c, sim, ent = self.cfg, env.sim, env.robot
        dev = env.device
        self.J = robot.num_joints
        self.com_body_id = ent.indexing.body_ids[ent.find_bodies(robot.com_body)[0][0]]
        self.joint_offset_half = torch.tensor(_match(c.joint_offset, robot.joint_names), device=dev)
        self.friction_geoms = ent.indexing.geom_ids
        self.ground_geoms = torch.nonzero(torch.as_tensor(sim.mj_model.geom_bodyid == 0, device=dev)).flatten()
        # src/evaluate.py --eval_delay random: eval episodes draw their delays U(delay_range) from this generator
        self.eval_delay_gen = None

        fields = {"geom_friction"}
        if not eval_mode:
            fields.add("body_ipos")
        # ---- a measured robot's items (DomainRandCfg's optional fields); evaluation keeps the model's own values
        self.jf, self.gain = c.joint_friction_scale, c.gain_scale
        self.masses = (c.link_mass_scale is not None or c.trunk_mass_add is not None) and not eval_mode
        if self.jf is not None:
            self.dof_adr = ent.indexing.joint_v_adr.reshape(-1)
            self.friction0 = sim.get_default_field("dof_frictionloss")[self.dof_adr].to(dev).float()
            assert (self.friction0 > 0).all(), "joint_friction_scale needs every joint's friction in the compiled model"
            if not eval_mode:
                fields.add("dof_frictionloss")
        if self.gain > 0:
            # the actuators' gains are slice views into their fused group's tensors (mjlab fused_group): one in-place
            # write per group and episode. (group, its nominal kp, its nominal kd), columns in group.target_ids' order
            groups = ent._fused_actuator_group._groups
            assert sorted(int(i) for g in groups for i in g.target_ids) == list(range(self.J)), \
                "gain_scale needs every joint on a fused PD actuator"
            self.gain_groups = [(g, g.params["stiffness"].clone(), g.params["damping"].clone()) for g in groups]
        # the worlds' sensor errors of the episode they are in (not in xi: the policy sees them, the dynamics do not)
        sc = robot.sensing
        self.sens = None if eval_mode or not (sc.gyro_bias or sc.imu_tilt) else (
            torch.zeros(env.num_envs, 3, device=dev), torch.zeros(env.num_envs, 2, device=dev))  # fmt: skip
        self.per_episode = self.jf is not None or self.gain > 0 or self.sens is not None
        if self.masses:
            from mjlab.managers.event_manager import _DERIVED_FIELDS, RecomputeLevel

            self.link_ids = torch.tensor([int(b) for b in ent.indexing.body_ids if int(b) != int(self.com_body_id)], device=dev)
            fields |= {"body_mass", *_DERIVED_FIELDS[RecomputeLevel.set_const]}
        sim.expand_model_fields(tuple(sorted(fields)))

    # ------------------------------------------------------------------ once per env
    def startup(self, xi: torch.Tensor):
        c, sim, N, dev = self.cfg, self.env.sim, self.env.num_envs, self.env.device
        xi[:, self.layout.slices["friction"]] = 1.0 if self.eval_mode else torch.empty(N, 1, device=dev).uniform_(*c.friction_range)
        # ground (world body): lowest friction of the range, since MuJoCo takes the max of a pair
        sim.model.geom_friction[:, self.ground_geoms, 0] = c.friction_range[0] if not self.eval_mode else 1.0
        sim.model.geom_friction[:, self.friction_geoms, 0] = xi[:, self.layout.slices["friction"]]
        if not self.eval_mode:
            com = torch.stack([torch.empty(N, device=dev).uniform_(lo, hi) for lo, hi in self.spec.com_range], -1)
            xi[:, self.layout.slices["com"]] = com
            default_com = sim.get_default_field("body_ipos")[self.com_body_id]
            sim.model.body_ipos[:, self.com_body_id] = default_com + com
        xi[:, self.layout.slices["delay"]] = float(self.spec.eval_delay)
        L = self.layout.slices
        for g in ("joint_friction", "kp", "kd"):  # factors: 1 is the model's own value (evaluation stays there)
            if g in L:
                xi[:, L[g]] = 1.0
        if self.eval_mode:
            return
        if self.masses:  # mass only, the inertia stays (the third party's dr.body_mass); then the derived constants
            from mjlab.managers.event_manager import RecomputeLevel

            mass = sim.get_default_field("body_mass").to(dev).float().repeat(N, 1)
            if c.link_mass_scale is not None:
                mass[:, self.link_ids] *= torch.empty(N, len(self.link_ids), device=dev).uniform_(*c.link_mass_scale)
            if c.trunk_mass_add is not None:
                add = torch.empty(N, 1, device=dev).uniform_(*c.trunk_mass_add)
                mass[:, self.com_body_id] += add[:, 0]
                xi[:, L["trunk_mass"]] = add
            sim.model.body_mass[:] = mass
            sim.recompute_constants(RecomputeLevel.set_const)  # covers the centre-of-mass offset above too
        self._episode(xi, torch.ones(N, dtype=torch.bool, device=dev))  # every world's first factors and sensor errors

    def sensor(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Each world's gyro bias (N, 3) and IMU tilt (N, 2) for MDP.obs(sensor=...); None in evaluation and for a
        robot without sensor errors."""
        return self.sens

    def sim_error(self, xi: torch.Tensor | None) -> tuple[float, float]:
        """How far the simulator is from what xi says, for a log row (host-side reads: never inside a captured graph):
        the largest |kp or kd / nominal - xi's factor| over worlds and joints, and the largest |joint friction /
        table - xi's factor|. (0, 0) for a robot without those items. train.py logs both and tacc/t2m_probe.sh gates
        them: the writes of _episode happen inside the captured collection step, which no CPU test runs."""
        L, gain_err, fric_err = self.layout.slices, 0.0, 0.0
        if self.gain > 0:
            for g, kp0, kd0 in self.gain_groups:
                gain_err = max(gain_err, float((g.params["stiffness"] / kp0 - xi[:, L["kp"]][:, g.target_ids]).abs().max()),
                               float((g.params["damping"] / kd0 - xi[:, L["kd"]][:, g.target_ids]).abs().max()))  # fmt: skip
        if self.jf is not None:
            fl = self.env.sim.model.dof_frictionloss
            fl = fl if fl.dim() == 2 and fl.shape[0] == xi.shape[0] else fl.reshape(-1, fl.shape[-1])[:1]  # evaluation: shared
            fric_err = float((fl[:, self.dof_adr] / self.friction0 - xi[:, L["joint_friction"]]).abs().max())
        return gain_err, fric_err

    def _episode(self, xi: torch.Tensor, mask: torch.Tensor):
        """A measured robot's per-episode draws for the worlds where mask is true (fixed size and in place:
        capturable): a factor per joint on the joint friction and on kp and kd, written to xi and to the simulator,
        and the sensor errors."""
        L, N, dev = self.layout.slices, xi.shape[0], xi.device
        m = mask.unsqueeze(-1)
        if self.sens is not None:
            b, t = self.env.mdp.draw_sensor((N,))
            self.sens[0].copy_(torch.where(m, b, self.sens[0]))
            self.sens[1].copy_(torch.where(m, t, self.sens[1]))
        if self.jf is not None:
            lo, hi = self.jf
            f = torch.where(m, lo + (hi - lo) * torch.rand(N, self.J, device=dev), xi[:, L["joint_friction"]])
            xi[:, L["joint_friction"]] = f
            self.env.sim.model.dof_frictionloss[:, self.dof_adr] = self.friction0 * f
        if self.gain > 0:
            w = self.gain * self.env.mdp.ramp  # the ramp widens the range from nothing to +-gain_scale
            kpf = torch.where(m, 1 + w * (2 * torch.rand(N, self.J, device=dev) - 1), xi[:, L["kp"]])
            kdf = torch.where(m, 1 + w * (2 * torch.rand(N, self.J, device=dev) - 1), xi[:, L["kd"]])
            xi[:, L["kp"]], xi[:, L["kd"]] = kpf, kdf
            for g, kp0, kd0 in self.gain_groups:
                g.params["stiffness"].copy_(kp0 * kpf[:, g.target_ids])
                g.params["damping"].copy_(kd0 * kdf[:, g.target_ids])

    def _draw_delay(self, n: int, dev) -> torch.Tensor:
        lo, hi = self.spec.delay_range
        return torch.randint(lo, hi + 1, (n, len(self.spec.delay_groups)), device=dev).float()

    # ------------------------------------------------------------------ per episode
    def reset_masked(self, xi: torch.Tensor, mask: torch.Tensor):
        """reset() of the worlds where mask is true (fixed size: every world draws, the masked ones take the draw)."""
        if self.eval_mode:
            return
        sl = self.layout.slices["joint_offset"]
        new = (2 * torch.rand(xi.shape[0], self.J, device=xi.device) - 1) * self.joint_offset_half
        xi[:, sl] = torch.where(mask.unsqueeze(-1), new, xi[:, sl])
        sl = self.layout.slices["delay"]
        xi[:, sl] = torch.where(mask.unsqueeze(-1), self._draw_delay(xi.shape[0], xi.device), xi[:, sl])
        if self.per_episode:
            self._episode(xi, mask)

    def reset(self, xi: torch.Tensor, env_ids: torch.Tensor):
        if self.eval_mode:
            if self.eval_delay_gen is not None and "delay" in self.layout.slices:
                lo, hi = self.spec.delay_range
                xi[env_ids, self.layout.slices["delay"]] = torch.randint(
                    lo, hi + 1, (len(env_ids), len(self.spec.delay_groups)), device=xi.device, generator=self.eval_delay_gen).float()
            return
        env, L = self.env, self.layout
        n, dev = len(env_ids), env.device
        xi[env_ids, L.slices["joint_offset"]] = (2 * torch.rand(n, self.J, device=dev) - 1) * self.joint_offset_half
        xi[env_ids, L.slices["delay"]] = self._draw_delay(n, dev)
        if self.per_episode:
            self._episode(xi, torch.zeros(xi.shape[0], dtype=torch.bool, device=dev).index_fill_(0, env_ids, True))
