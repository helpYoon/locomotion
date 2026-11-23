"""Motion-tracking simulator on mjlab (MuJoCo-Warp).

The environment is a thin batched wrapper: it applies the joint setpoint, steps MuJoCo-Warp, labels
the transition with the analytic reward/termination of :mod:`src.mdp`, and resets episodes to
reference-motion states drawn by return-based bin sampling (paper Appendix D.1). Domain randomization
(:mod:`src.dr`) and the policy's sensing (:mod:`src.sensing`) follow ``RobotSpec``.
"""

from __future__ import annotations

import torch
import mujoco_warp as mjwarp
import warp as wp
from mjlab.entity.data import EntityData
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from src import landing
from src.dr import DomainRandomizer, XiLayout
from src.mdp import GAIT_DIM, MAX_STEP_REWARD, MDP, flatten_state, gait_reset
from src.motion import CONTROL_DT, MotionLib
from robot.spec import RobotSpec
from utils.rotations import mat_to_quat, quat_to_mat, so3_exp
from src.sensing import ObsPipeline


# A python scalar written through a tensor index (x[ids] = 0.0) is copied host-to-device from pageable memory, which
# synchronizes the stream: 13 of the 14 host syncs per training step came from the reset path.
# The same writes with a device-side 0-dim value do not synchronize and give identical values.
def _dev_scalar(t: torch.Tensor, v):
    return torch.full((), v, dtype=t.dtype, device=t.device)


def _clear_state(self, env_ids=None) -> None:  # mjlab EntityData.clear_state without host syncs
    if self.is_actuated:
        env_ids = self._resolve_env_ids(env_ids)
        for t in (self.joint_pos_target, self.joint_vel_target, self.joint_effort_target, self.tendon_len_target,
                  self.tendon_vel_target, self.tendon_effort_target, self.site_effort_target):  # fmt: skip
            t[env_ids] = _dev_scalar(t, 0.0)


def _sim_reset(self, env_ids=None) -> None:  # mjlab Simulation.reset without host syncs
    with wp.ScopedDevice(self.wp_device):
        if env_ids is None:
            self._reset_mask.fill_(True)
        else:
            self._reset_mask.fill_(False)
            self._reset_mask[env_ids] = _dev_scalar(self._reset_mask._tensor, True)
        if self.use_cuda_graph and self.reset_graph is not None:
            wp.capture_launch(self.reset_graph)
        else:
            mjwarp.reset_data(self.wp_model, self.wp_data, reset=self._reset_mask_wp)


EntityData.clear_state, Simulation.reset = _clear_state, _sim_reset

DECIMATION = 4  # 4 physics substeps of 5 ms per 50 Hz control step
PHYSICS_DT = CONTROL_DT / DECIMATION


def make_scene_cfg(robot: RobotSpec, num_envs: int) -> SceneCfg:
    return SceneCfg(
        num_envs=num_envs,
        env_spacing=0.0,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": robot.robot_cfg()},
    )


def make_sim(robot: RobotSpec) -> SimulationCfg:
    return SimulationCfg(
        njmax=robot.njmax,
        nconmax=robot.nconmax,
        mujoco=MujocoCfg(
            timestep=PHYSICS_DT,
            integrator="implicitfast",
            cone="pyramidal",
            solver="newton",
            iterations=robot.solver[0],
            ls_iterations=robot.solver[1],
            tolerance=robot.solver[2],
            ccd_iterations=robot.solver[3],
        ),
    )


def _bin_ema_w(stat: torch.Tensor, bins: torch.Tensor, x: torch.Tensor, wt: torch.Tensor, old: float) -> torch.Tensor:
    """EMA update of per-bin statistics with possibly repeated bin indices, over the samples with weight wt = 1 (the
    others weight 0): the samples of a bin are averaged, then one EMA step is taken for every bin that received at
    least one sample (fixed size, no host-side selection, no host sync)."""
    sums = torch.zeros_like(stat).index_add_(0, bins, x * wt)
    cnt = torch.zeros_like(stat).index_add_(0, bins, wt)
    return torch.where(cnt > 0, old * stat + (1 - old) * sums / cnt.clamp(min=1), stat)


class TrackingSim:
    """Batched motion-tracking simulator for the robot of ``robot``."""

    def __init__(self, sim_cfg: SimulationCfg, num_envs: int, robot: RobotSpec, motion: MotionLib, mdp: MDP,
                 device: str, gamma: float = 0.99, eval_mode: bool = False, max_episode_steps: int = 500,
                 reset_noise: bool = True, randomize: bool | None = None, fail_reset: float = 0.0, bin_fix: bool = False,
                 n_vel: int = 0, n_land: int = 0):  # fmt: skip
        """``randomize`` (default: not eval_mode) switches domain randomization, observation noise and
        delay; ``eval_mode`` switches evaluation bookkeeping (no auto-reset, episode starts
        exactly on the reference). ``n_vel`` (train.py --vel_frac): worlds 0 .. n_vel-1 follow a velocity command
        (MDP.enable_velocity) instead of their clip: velocity reward, fall-only termination, starts uniform over the
        library; their clip only supplies the reset state, the episode length and, half of the time, the command.
        ``n_land`` (train.py --land_frac): worlds n_vel .. n_vel+n_land-1 are landing robots (MDP.enable_landing,
        MotionLib.set_land_labels): a foot-landing command (land_command), then a stand; landing reward, fall-only
        termination. At each of its resets each landing robot becomes either a label row (LandingCfg.label_prob),
        which starts on a hold frame before one of the library's labelled steps, follows its clip and is given the
        labels' commands, its episode ending at its first unlabelled frame, the clip's end, the episode cap or a fall;
        or a free row, which starts on a hold frame, stays on it (its frame does not advance) and is given sampled
        commands, its episode ending at a fall or the episode cap."""
        self.device = device
        self.scene = Scene(make_scene_cfg(robot, num_envs), device=device)
        self.sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=self.scene.compile(), device=device)
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
        self.robot = self.scene["robot"]
        assert list(self.robot.joint_names) == list(robot.joint_names), self.robot.joint_names
        self.N = num_envs
        self.J = robot.num_joints
        self.spec = robot
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.randomize = (not eval_mode) if randomize is None else randomize
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        dev = self.device
        self.default_q = self.robot.data.default_joint_pos[0].clone()
        self.alpha = torch.tensor(robot.action_scale_list(), device=dev)

        # ---- sim context xi (layout in src.dr; paper setup: friction, joint offsets, com offset) ----
        self.layout = XiLayout(robot, dev)
        assert self.layout.dim == mdp.xi_dim, (self.layout.dim, mdp.xi_dim)
        self.xi = torch.zeros(self.N, self.layout.dim, device=dev)
        self.randomizer = DomainRandomizer(robot, self.layout, self, eval_mode=not self.randomize)
        self.randomizer.startup(self.xi)

        # ---- sensing (device tables only; the observation is MDP.obs) ----
        self.pipe = ObsPipeline(robot, mdp, self.layout, dev, PHYSICS_DT)
        self.obs_fresh = torch.ones(self.N, dtype=torch.bool, device=dev)
        self.push_timer = torch.zeros(self.N, device=dev)

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        # a_prev: the last max(1, n_prev_actions) actions, newest first (RobotSpec.n_prev_actions)
        self.AW = self.J * max(1, robot.n_prev_actions)
        self.a_prev = torch.zeros(self.N, self.AW, device=dev)
        # RobotSpec.delay_groups: each group's position target delayed by its lag (xi["delay"], physics steps), as bfm_wm's
        # DelayedPDBatch: write the target, read the one written min(lag, pushes since reset) physics steps ago
        gj = torch.zeros(self.J, dtype=torch.long)
        for g, js in enumerate(robot.delay_groups):
            gj[list(js)] = g
        self.delay_gj = gj.to(dev)
        self.ring = torch.zeros(robot.delay_range[1] + 1, self.N, self.J, device=dev)
        self.ring_w = torch.zeros(1, dtype=torch.long, device=dev)
        self.pushes = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=dev)
        # share of resets sent one bin before recent failure points (per-bin termination rate)
        self.fail_reset = fail_reset
        # index_put_ with duplicate bin indices keeps one arbitrary episode per bin and step (undefined
        # behaviour); bin_fix averages the episodes of a bin first (as supermimic_vibed's update_bin_returns)
        self.bin_fix = bin_fix
        # externally set reset distribution (train.py: the frontier regions of failing clips), mixed in with share
        # focus_share_t
        self.focus_q = None
        self.focus_static, self.focus_share_t = False, None  # use_static_focus()
        self.track_fail = fail_reset > 0
        # --reset_aprev_ref (train.py): a reset in the middle of a clip records the previous action as the one that holds the
        # reset pose, (q - q0) / alpha, which is the actuator target the reset writes; starts on a clip's first frame keep
        # a_prev = 0, as the evaluation. Default off (a_prev = 0 for every reset)
        self.aprev_ref = False
        self.aprev_gate = None  # train.py --phase: 0-dim bool tensor that switches --reset_aprev_ref on/off in place
        self.exam = None  # --exam_frac (set_exam): mask of the exam worlds
        self.bin_fail = torch.zeros(motion.num_bins, device=dev)
        prev = torch.arange(motion.num_bins, device=dev) - 1
        self.bin_prev = torch.where((prev >= 0) & (motion.bin_clip[prev.clamp(min=0)] == motion.bin_clip), prev, prev + 1)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        # velocity worlds: command source (read from the clip, else cmd_held) and the feet's velocities one and two
        # control steps ago (the jerk term); all updated in place, so captured graphs follow them
        self.n_vel = n_vel
        self.is_vel = torch.arange(self.N, device=dev) < n_vel
        self.cmd_ref = torch.ones(self.N, dtype=torch.bool, device=dev)
        self.cmd_held = torch.zeros(self.N, 3, device=dev)
        self.foot_v = torch.zeros(n_vel, 2, len(mdp.feet), 3, device=dev)
        self.foot_z0 = torch.zeros(n_vel, len(mdp.feet), device=dev)  # the feet's heights one control step ago (the step term)
        # --sym_weight: the velocity worlds' gait memory (src/mdp.py GAIT_DIM), as of the current state; updated in place
        self.sym = bool(n_vel) and mdp.vel.w_sym != 0
        if self.sym:
            self.gait = torch.zeros(n_vel, GAIT_DIM, device=dev)
        # landing robots: the command state, label or free row, a free row's wait to its next command, the sole poses
        # the next step starts from and the feet's velocities one and two control steps ago; all updated in place, so
        # captured graphs follow them. is_cmd: the worlds whose observation carries no reference
        self.n_land, self.n_cmd = n_land, n_vel + n_land
        self.is_cmd = torch.arange(self.N, device=dev) < self.n_cmd
        if n_land:
            assert hasattr(mdp, "land"), "call mdp.enable_landing first"
            assert hasattr(motion, "land_kind"), "call motion.set_land_labels first"
            assert len(motion.land_free_reset) > 0, "the library has no free-row reset frames"
        # a library without label-reset frames (a one-clip standing library) has free rows only; decided here, once
        self.land_has_label = n_land > 0 and len(motion.land_label_reset) > 0
        self.land_auto = True  # the sampler issues the free rows' commands (a test rig issues its own)
        self.land_C = torch.zeros(n_land, landing.STATE_DIM, device=dev)
        self.land_lab = torch.zeros(n_land, dtype=torch.bool, device=dev)
        self.land_wait = torch.zeros(n_land, dtype=torch.long, device=dev)
        self.land_pose = torch.zeros(n_land, 2, 3, device=dev)
        self.land_fv = torch.zeros(n_land, 2, 2, 3, device=dev)
        self.reset(torch.arange(self.N, device=dev))

    @property
    def num_envs(self) -> int:
        return self.N

    def set_exam(self, n_ex: int, clips: torch.Tensor):
        """--exam_frac (train.py): worlds N - n_ex .. N - 1 run exam episodes, as the evaluation runs a clip (from its
        first frame to its end, no episode cap) but in training conditions (randomization, reset and observation
        noise); train.py collects them with the Polyak policy plus noise. Each episode takes a clip drawn uniformly from
        ``clips``. Their episode ends do not enter the reset statistics (bin_G, bin_fail): only their data are used."""
        assert not self.eval_mode
        dev = self.device
        self.exam = torch.arange(self.N, device=dev) >= self.N - n_ex
        self.exam_clips = clips
        # train.py --exam_mix: (mask of the n_ex exam worlds being reset) -> a clip id for every exam world
        self.exam_sampler_masked, self.n_ex = None, n_ex
        self.max_ep_base = self.max_episode_steps
        self.max_episode_steps = torch.where(self.exam, 10**9, self.max_episode_steps)
        ids = self.exam.nonzero(as_tuple=False).squeeze(-1)
        self.reset(ids, self.exam_frames(len(ids)))

    def set_exam_active(self, n_act: int) -> torch.Tensor:
        """train.py --phase: of the n_ex exam slots (the last n_ex worlds) the last n_act run exam episodes; the others
        run practice episodes. self.exam and the episode cap change in place (captured graphs follow them). Returns the
        mask of the worlds whose role changed (the caller resets them)."""
        assert 0 <= n_act <= self.n_ex
        new = torch.arange(self.N, device=self.device) >= self.N - n_act
        changed = new != self.exam
        self.exam.copy_(new)
        self.max_episode_steps.copy_(torch.where(self.exam, 10**9, self.max_ep_base))
        return changed

    def exam_frames(self, n: int) -> torch.Tensor:
        c = self.exam_clips[torch.randint(0, len(self.exam_clips), (n,), device=self.device)]
        return self.motion.clip_start[c]

    # ------------------------------------------------------------------ state
    def state(self):
        # root state from the free joint (qpos/qvel are current after a step; derived xpos lag a substep)
        idx = self.robot.indexing
        root_q = self.sim.data.qpos[:, idx.free_joint_q_adr]
        root_v = self.sim.data.qvel[:, idx.free_joint_v_adr]
        p = root_q[:, :3] - self.scene.env_origins
        R = quat_to_mat(root_q[:, 3:7])  # MuJoCo uses wxyz
        # free joint velocity: world-frame linear velocity of the root origin, angular velocity in the root frame
        w = (R @ root_v[:, 3:6].unsqueeze(-1)).squeeze(-1)
        d = self.robot.data
        return (p, R, root_v[:, :3].clone(), w, d.joint_pos.clone(), d.joint_vel.clone())

    # ------------------------------------------------------------------ velocity worlds
    def command(self) -> torch.Tensor:
        """Velocity command (N, 3) of every world (only the velocity worlds use it)."""
        return torch.where(self.cmd_ref.unsqueeze(-1), self.motion.ref_cmd[self.f], self.cmd_held)

    def _resample_cmd(self, mask: torch.Tensor):
        """A new command source for the velocity worlds in ``mask``: the clip's command with ref_cmd_prob, else a
        sampled command that is held."""
        mask = mask & self.is_vel
        from_ref = torch.rand(self.N, device=self.device) < self.mdp.vel.ref_cmd_prob
        self.cmd_ref.copy_(torch.where(mask, from_ref, self.cmd_ref))
        self.cmd_held.copy_(torch.where(mask.unsqueeze(-1), self.mdp.sample_commands(self.N), self.cmd_held))
        if self.sym:  # a new command: the next step is not compared with one taken under the old
            self.gait.copy_(gait_reset(self.gait, mask[: self.n_vel], full=False))

    def _uniform_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """n episode starts uniform over the library, and their bins (the velocity worlds' resets: the return and
        frontier sampling follow the motion interface's failures)."""
        m = self.motion
        b = torch.multinomial(m.bin_w, n, replacement=True)
        return m.bin_start[b] + (torch.rand(n, device=self.device) * (m.bin_end[b] - m.bin_start[b]).float()).long(), b

    # ------------------------------------------------------------------ landing robots
    def land_command(self, o: torch.Tensor) -> torch.Tensor:
        """Issue the commands that arrive on the current state, then the 12 command features of ``enc_land`` (n_land,
        12) from the observation o (N, obs). A label row in HOLD on a step frame of its clip takes that frame's label
        (so its clock is the table's); a free row in HOLD whose wait is over takes the sampler's draw (land_auto) and
        begins its next wait, also when no candidate passed. landing.land_issue leaves rows outside HOLD as they are."""
        nv, nc, m, cfg, dev = self.n_vel, self.n_cmd, self.motion, self.mdp.land, self.device
        f, lab = self.f[nv:nc], self.land_lab
        hold = landing.phase(self.land_C)[2]
        arrive = lab & hold & (m.land_kind[f] == 2)
        swing, c, n0 = m.land_swing[f], m.land_c[f], m.land_n[f]
        if self.land_auto:
            due = ~lab & hold & (self.land_wait == 0)
            u = torch.rand(self.n_land, 3 + 3 * cfg.candidates, device=dev)
            s_swing, s_c, s_n0, ok = landing.land_sample(cfg, self.land_pose, u)
            arrive = arrive | (due & ok)
            swing, n0 = torch.where(due, s_swing, swing), torch.where(due, s_n0, n0)
            c = torch.where(due[:, None], s_c, c)
            lo, hi = cfg.wait_next
            wait = lo + torch.floor(torch.rand(self.n_land, device=dev) * (hi - lo + 1)).long()
            self.land_wait.copy_(torch.where(due, wait, self.land_wait))
        self.land_C.copy_(landing.land_issue(self.land_C, self.land_pose, arrive, swing, c, n0))
        return self.mdp.land_features(self.land_C, o[nv:nc])

    def _land_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """n landing-robot starts, and which are label rows (LandingCfg.label_prob): a label row's frame uniform over
        motion.land_label_reset, a free row's uniform over motion.land_free_reset. Free rows only when the library has
        no label-reset frames."""
        m, dev = self.motion, self.device
        lab = torch.rand(n, device=dev) < self.mdp.land.label_prob
        pick = lambda fr: fr[(torch.rand(n, device=dev) * len(fr)).long().clamp(max=len(fr) - 1)]  # noqa: E731
        frames = pick(m.land_free_reset)
        if not self.land_has_label:
            return frames, torch.zeros_like(lab)
        return torch.where(lab, pick(m.land_label_reset), frames), lab

    def _land_reset(self, mask: torch.Tensor, S, lab: torch.Tensor):
        """Reset the landing rows in ``mask`` (n_land,) onto their reset state S (the landing rows of (p, R, v, w, q,
        qd)): SETTLE with both anchors on the soles of S and a random swing sole, label or free row ``lab``, a free
        row's first wait (LandingCfg.wait_first), and the sole poses the first step starts from."""
        cfg, dev, n = self.mdp.land, self.device, self.n_land
        pose = self.mdp.sole_kin(S)[0]["pose"]
        swing = (torch.rand(n, device=dev) < 0.5).float()
        lo, hi = cfg.wait_first
        wait = lo + torch.floor(torch.rand(n, device=dev) * (hi - lo + 1)).long()
        self.land_C.copy_(landing.land_reset(cfg, self.land_C, pose, mask, swing))
        self.land_lab.copy_(torch.where(mask, lab, self.land_lab))
        self.land_wait.copy_(torch.where(mask, wait, self.land_wait))
        self.land_pose.copy_(torch.where(mask[:, None, None], pose, self.land_pose))

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int, gate: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``gate`` (reset_masked): a 0-dim bool; the running average bin_w_avg only advances where it is true (the
        eager reset samples only when an episode ended)."""
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        if gate is None:
            self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        else:
            self.bin_w_avg.copy_(torch.where(gate, self.bin_w_avg * 0.999 + 0.001 * soft, self.bin_w_avg))
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        if self.fail_reset > 0:
            fail = torch.zeros_like(self.bin_fail).index_add_(0, self.bin_prev, self.bin_fail)
            fail = fail / fail.sum().clamp(min=1e-6)
            probs = (1 - self.fail_reset) * probs + self.fail_reset * torch.where(fail.sum() > 0, fail, probs)
        if self.focus_static:  # reset_masked: device-side focus (share 0 leaves probs bitwise unchanged)
            probs = (1 - self.focus_share_t) * probs + self.focus_share_t * self.focus_q
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = m.bin_end[bins]  # = min(start + 50, clip end) unless tail-merged; keeps >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
            if self.n_vel:
                vf, vb = self._uniform_frames(n)
                frames, bins = torch.where(self.is_vel[env_ids], vf, frames), torch.where(self.is_vel[env_ids], vb, bins)
            if self.n_land:  # landing robots: a label row's or a free row's start
                lf, lab = self._land_frames(n)
                is_land = (env_ids >= self.n_vel) & (env_ids < self.n_cmd)
                frames, bins = torch.where(is_land, lf, frames), torch.where(is_land, self.motion.frame_bin[lf], bins)
        else:
            bins = self.motion.frame_bin[frames]
            if self.n_land:  # given frames: the landing robots are free rows
                lab = torch.zeros(n, dtype=torch.bool, device=dev)
        m, c = self.motion, self.spec.dr
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            u = lambda half: torch.stack([torch.empty(n, device=dev).uniform_(-h, h) for h in half], -1)  # noqa: E731
            p = p + u(c.reset_pos)
            R = so3_exp(u(c.reset_rot)) @ R
            v = v + u(c.reset_lin_vel)
            w = w + u(c.reset_ang_vel)
            q = q + torch.empty_like(q).uniform_(-c.reset_joint_pos, c.reset_joint_pos)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        # mjlab's reset clears solver warm-start data and actuator targets, so it runs before the writes
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)
        pose = torch.cat((p + self.scene.env_origins[env_ids], mat_to_quat(R)), -1)
        self.robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_position_to_sim(q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim(qd, env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        self.f[env_ids] = frames
        self.ep_step.index_fill_(0, env_ids, 0)  # index_fill_: no host sync (see _dev_scalar)
        self.a_prev.index_fill_(0, env_ids, 0.0)
        self.pushes.index_fill_(0, env_ids, 0)
        if self.aprev_ref and not self.eval_mode:
            first = frames == self.motion.clip_start[self.motion.clip_id[frames]]
            if self.aprev_gate is not None:
                first = first | ~self.aprev_gate
            hold = ((q - self.default_q) / self.alpha).repeat(1, self.AW // self.J)
            self.a_prev[env_ids] = torch.where(first.unsqueeze(-1), 0.0, hold)
        self.start_bin[env_ids] = bins
        self.ep_disc_ret.index_fill_(0, env_ids, 0.0)
        self.ep_ret.index_fill_(0, env_ids, 0.0)
        self.randomizer.reset(self.xi, env_ids)  # after scene.reset, which would restore actuator defaults
        self.obs_fresh.index_fill_(0, env_ids, True)
        if self.n_vel:
            mask = torch.zeros(self.N, dtype=torch.bool, device=dev).index_fill_(0, env_ids, True)
            if self.sym:
                self.gait.copy_(gait_reset(self.gait, mask[: self.n_vel], full=True))
            self._resample_cmd(mask)
        if self.n_land:  # on all worlds' rows, the reset ones copied in (the others are masked out)
            nv, nc = self.n_vel, self.n_cmd
            mask = torch.zeros(self.N, dtype=torch.bool, device=dev).index_fill_(0, env_ids, True)
            full = lambda x: x.new_zeros((self.N, *x.shape[1:])).index_copy_(0, env_ids, x)  # noqa: E731
            self._land_reset(mask[nv:nc], tuple(full(x)[nv:nc] for x in (p, R, v, w, q, qd)), full(lab)[nv:nc])

    def use_static_focus(self):
        """Keep the focus distribution and share in fixed device tensors (updated in place by set_focus), so a captured
        reset_masked follows them."""
        self.focus_static = True
        self.focus_q = torch.full((self.motion.num_bins,), 1.0 / self.motion.num_bins, device=self.device)
        self.focus_share_t = torch.zeros((), device=self.device)

    def set_focus(self, q: torch.Tensor, share: float):
        self.focus_q.copy_(q), self.focus_share_t.fill_(share)

    def reset_masked(self, mask: torch.Tensor, frames: torch.Tensor | None = None):
        """reset() of the worlds where ``mask`` is true, as fixed-size masked updates of all worlds (no host-side ids, so
        it can be captured in a CUDA graph). Worlds outside the mask are left bitwise unchanged; the reset worlds get the
        same values as reset() would give them for the same frames and random draws (random draws are made for all
        worlds, so the random stream differs from reset()).
        Training mode only."""
        assert not self.eval_mode
        N, dev, m, c = self.N, self.device, self.motion, self.spec.dr
        nv, nc = self.n_vel, self.n_cmd  # the landing robots are rows nv .. nc-1
        d = mask
        d1 = d.unsqueeze(-1)
        if frames is None:
            frames, bins = self.sample_frames(N, gate=d.any())
            if self.exam is not None:  # exam episodes start on a clip's first frame (the last n_ex worlds)
                n = self.n_ex
                ex = self.exam[N - n :]  # active exam slots (all of them unless train.py --phase lowers the share)
                if self.exam_sampler_masked is not None:
                    ec = self.exam_sampler_masked(d[N - n :] & ex)
                else:
                    ec = self.exam_clips[(torch.rand(n, device=dev) * len(self.exam_clips)).long().clamp(max=len(self.exam_clips) - 1)]
                ef = m.clip_start[ec]
                ef, eb = torch.where(ex, ef, frames[N - n :]), torch.where(ex, m.frame_bin[ef], bins[N - n :])
                frames, bins = torch.cat((frames[: N - n], ef)), torch.cat((bins[: N - n], eb))
            if self.n_vel:
                vf, vb = self._uniform_frames(N)
                frames, bins = torch.where(self.is_vel, vf, frames), torch.where(self.is_vel, vb, bins)
            if self.n_land:  # landing robots: a label row's or a free row's start
                lf, lab = self._land_frames(self.n_land)
                frames = torch.cat((frames[:nv], lf, frames[nc:]))
                bins = torch.cat((bins[:nv], m.frame_bin[lf], bins[nc:]))
        else:
            bins = m.frame_bin[frames]
            if self.n_land:  # given frames: the landing robots are free rows
                lab = torch.zeros(self.n_land, dtype=torch.bool, device=dev)
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            u = lambda half: torch.stack([torch.empty(N, device=dev).uniform_(-h, h) for h in half], -1)  # noqa: E731
            p = p + u(c.reset_pos)
            R = so3_exp(u(c.reset_rot)) @ R
            v = v + u(c.reset_lin_vel)
            w = w + u(c.reset_ang_vel)
            q = q + torch.empty_like(q).uniform_(-c.reset_joint_pos, c.reset_joint_pos)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        # Simulation.reset (solver warm start, controls, ...) of the masked worlds, before the writes
        sim = self.sim
        sim._reset_mask.copy_(d)
        with wp.ScopedDevice(sim.wp_device):
            if sim.use_cuda_graph and sim.reset_graph is not None:
                wp.capture_launch(sim.reset_graph)
            else:
                mjwarp.reset_data(sim.wp_model, sim.wp_data, reset=sim._reset_mask_wp)
        # Scene.reset: the robot's actuator targets (no actuator delays, no sensors; the terrain has no state)
        ed = self.robot.data
        for t in (ed.joint_pos_target, ed.joint_vel_target, ed.joint_effort_target, ed.tendon_len_target,
                  ed.tendon_vel_target, ed.tendon_effort_target, ed.site_effort_target):  # fmt: skip
            t.masked_fill_(d.view(-1, *([1] * (t.dim() - 1))), 0.0)
        # the writes of reset(): root pose, root velocity (angular in the new root frame), joints, position targets
        idx = self.robot.indexing
        qa, va = idx.free_joint_q_adr, idx.free_joint_v_adr
        qpos, qvel = sim.data.qpos, sim.data.qvel
        pose = torch.cat((p + self.scene.env_origins, mat_to_quat(R)), -1)
        qpos[:, qa] = torch.where(d1, pose, qpos[:, qa])
        qvel[:, va] = torch.where(d1, torch.cat((v, quat_apply_inverse(pose[:, 3:7], w)), -1), qvel[:, va])
        qpos[:, idx.joint_q_adr] = torch.where(d1, q, qpos[:, idx.joint_q_adr])
        qvel[:, idx.joint_v_adr] = torch.where(d1, qd, qvel[:, idx.joint_v_adr])
        ed.joint_pos_target.copy_(torch.where(d1, q, ed.joint_pos_target))
        self.f.copy_(torch.where(d, frames, self.f))
        self.ep_step.masked_fill_(d, 0)
        self.a_prev.masked_fill_(d1, 0.0)
        self.pushes.masked_fill_(d, 0)
        if self.aprev_ref and not self.eval_mode:
            mid = d & (frames != self.motion.clip_start[self.motion.clip_id[frames]])
            if self.aprev_gate is not None:
                mid = mid & self.aprev_gate
            hold = ((q - self.default_q) / self.alpha).repeat(1, self.AW // self.J)
            self.a_prev.copy_(torch.where(mid.unsqueeze(-1), hold, self.a_prev))
        self.start_bin.copy_(torch.where(d, bins, self.start_bin))
        self.ep_disc_ret.masked_fill_(d, 0.0)
        self.ep_ret.masked_fill_(d, 0.0)
        self.randomizer.reset_masked(self.xi, d)
        self.obs_fresh |= d
        if self.n_vel:
            if self.sym:
                self.gait.copy_(gait_reset(self.gait, d[: self.n_vel], full=True))
            self._resample_cmd(d)
        if self.n_land:
            self._land_reset(d[nv:nc], tuple(x[nv:nc] for x in (p, R, v, w, q, qd)), lab)

    def step_tail_masked(self, out: dict) -> dict:
        """Episode ends of a training step without host-side ids (capturable): bin statistics of the ended episodes and
        reset_masked. Instead of per-episode tensors, out gets n_done and the sums ep_len_sum / ep_ret_norm_sum over the
        ended episodes."""
        assert self.bin_fix, "reset_masked needs bin_fix (the index_put_ update keeps an arbitrary episode per bin)"
        done, term, f = out["done"], out["term"], out["f"]
        dw = done.float()
        T = self.ep_step.float()
        boot = torch.where(term, 0.0, (self.gamma**T) * (self.ep_ret / T) / (1 - self.gamma))
        G = self.ep_disc_ret + boot
        fb = self.motion.frame_bin[f.clamp(max=self.motion.num_frames - 1)]
        if self.exam is not None:  # exam episodes: outside the statistics
            dw = (done & ~self.exam).float()
        if self.n_cmd:  # velocity and landing episodes too: the bins rank the motion interface's returns
            dw = dw * (~self.is_cmd).float()
        self.bin_G.copy_(_bin_ema_w(self.bin_G, self.start_bin, G, dw, 0.8))
        if self.track_fail:
            bf = torch.where(dw.sum() > 0, self.bin_fail * 0.999, self.bin_fail)
            self.bin_fail.copy_(_bin_ema_w(bf, fb, term.float(), dw, 0.9))
        out["n_done"] = dw.sum()
        out["ep_len_sum"] = (T * dw).sum()
        out["ep_ret_norm_sum"] = (self.ep_ret / (T * MAX_STEP_REWARD) * dw).sum()
        self.reset_masked(done)
        return out

    # ------------------------------------------------------------------ observation
    def observe(self) -> torch.Tensor:
        """Policy observation of the current state (training: noisy; eval: noise- and delay-free)."""
        S = self.state()
        o = self.mdp.obs(S, self.a_prev, self.xi, self.f, self.is_cmd if self.n_cmd else None,
                         sensor=self.randomizer.sensor())  # a measured robot's sensor errors (training worlds)
        self._obs = self.mdp.add_noise(o) if self.randomize else o
        return self._obs

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor) -> dict:
        """Evaluation step (eval_mode: training steps through step_core and step_tail_masked)."""
        out = self.step_core(a)
        self.failed_eval |= out["term"] & ~self.done_eval
        self.done_eval |= out["done"]
        return out

    def step_core(self, a: torch.Tensor) -> dict:
        """The step up to the episode ends: no host synchronization, so it can be captured in a CUDA graph (train.py's
        captured collection). The simulator wrapper rebinds f, ep_step and a_prev."""
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        nv, nc = self.n_vel, self.n_cmd  # rows 0 .. nv-1 velocity worlds, nv .. nc-1 landing robots
        cmd = self.command() if self.n_vel else None
        if self.n_land:  # the landing command state in force during this step (land_command issued on this state)
            Cl, lab = self.land_C.clone(), self.land_lab.clone()
        target = self.default_q + xi[:, 1 : 1 + self.J] + self.alpha * a
        lag = xi[:, self.layout.slices["delay"]][:, self.delay_gj].long()
        for _ in range(DECIMATION):
            self.robot.set_joint_position_target(self._delayed(target, lag))
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(PHYSICS_DT)
        S1 = self.state()
        f1 = f + 1
        if self.n_land:  # a free landing row follows no clip: its frame stays
            f1 = torch.cat((f1[:nv], f[nv:nc] + lab.long(), f1[nc:]))
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        if self.n_vel:  # velocity worlds: their own reward and a fall-only termination
            Sv = tuple(x[:nv] for x in S1)
            foot_z, foot_v, foot_xy = self.mdp.foot_kin(Sv)
            r_v, vel_terms = self.mdp.velocity_reward(Sv, cmd[:nv], a[:nv], a_prev[:nv], xi[:nv], foot_z, foot_v,
                                                      self.foot_v[:, 0], self.foot_v[:, 1], ep_step[:nv] >= 2, terms=True,
                                                      foot_xy=foot_xy, foot_z0=self.foot_z0)  # fmt: skip
            self.foot_v.copy_(torch.stack((foot_v, self.foot_v[:, 0]), 1))
            self.foot_z0.copy_(foot_z)
            if self.sym:  # the memory as of the start state goes to the buffer; then this step's touchdown, if any
                gait0 = self.gait.clone()
                pen, scored, gait1 = self.mdp.gait_step(gait0, foot_v, foot_xy, Sv[1],
                                                        self.mdp.gait_gate(cmd[:nv], self.cmd_ref[:nv]),
                                                        ep_step[:nv] >= self.mdp.vel.sym_settle)  # fmt: skip
                self.gait.copy_(gait1)
                r_v = r_v + self.mdp.vel.w_sym * pen * (self.mdp.vel.scale * CONTROL_DT)
                vel_terms.update(sym=pen, sym_n=scored.float())
            r = torch.cat((r_v, r[nv:]))
            term = torch.cat((self.mdp.fall_termination(Sv), term[nv:]))
        if self.n_land:  # landing robots: their own reward and command state, a fall-only termination
            Sl = tuple(x[nv:nc] for x in S1)
            r_l, C1, pose1, fv, land_terms = self.mdp.land_step(
                Sl, Cl, self.land_pose, a[nv:nc], a_prev[nv:nc], xi[nv:nc], self.land_fv[:, 0], self.land_fv[:, 1],
                ep_step[nv:nc] >= 2)  # fmt: skip
            r = torch.cat((r[:nv], r_l, r[nc:]))
            term = torch.cat((term[:nv], self.mdp.fall_termination(Sl), term[nc:]))
            self.land_fv.copy_(torch.stack((fv, self.land_fv[:, 0]), 1))
            self.land_C.copy_(C1)
            self.land_pose.copy_(pose1)
            wait = (self.land_wait - 1).clamp(min=0)  # a free row's wait for its next command runs down while it holds
            self.land_wait.copy_(torch.where(landing.phase(C1)[2] & ~lab, wait, self.land_wait))
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        if self.n_land:  # a label row also ends on its first unlabelled frame (a free row: a fall or the episode cap)
            gone = lab & (self.motion.land_kind[f1[nv:nc]] == 0)
            timeout = torch.cat((timeout[:nv], timeout[nv:nc] | gone, timeout[nc:]))
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms, "cut": done,
        }  # fmt: skip
        if self.n_vel:
            out.update(cmd=cmd, cmd_ref=self.cmd_ref.clone(), vel_terms=vel_terms)
        if self.sym:  # padded to all worlds with zeros, as the buffer stores them
            out["gait"] = gait0.new_zeros((self.N, GAIT_DIM)).slice_scatter(gait0, 0, 0, nv)
        if self.n_land:  # padded to all worlds with zeros, as the buffer stores them
            pad = lambda x: x.new_zeros((self.N, *x.shape[1:])).slice_scatter(x, 0, nv, nc)  # noqa: E731
            out.update(land=pad(Cl), lab=pad(lab), land_terms=land_terms)
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = torch.cat((a, a_prev[:, : self.AW - self.J]), -1)
        if self.n_vel:
            self._resample_cmd(self.ep_step % self.mdp.vel.resample_steps == 0)
        return out

    def _delayed(self, target: torch.Tensor, lag: torch.Tensor) -> torch.Tensor:
        """One physics step of the delay ring (capturable: in-place, device-side write head)."""
        S = self.ring.shape[0]
        self.ring.index_copy_(0, self.ring_w, target.unsqueeze(0))
        lag = torch.minimum(lag, self.pushes.unsqueeze(-1))
        out = torch.gather(self.ring, 0, ((self.ring_w - lag) % S).unsqueeze(0))[0]
        self.pushes.add_(1)
        self.ring_w.add_(1).remainder_(S)
        return out
