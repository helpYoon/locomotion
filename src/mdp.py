"""Analytic MDP heads shared by the simulator and the world model (paper Appendix A).

State tuple ``S = (p, R, v, w, q, qd)``: root position, rotation matrix, world linear/angular
velocity of the root link frame, joint positions and velocities. ``J`` is the number of joints.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch

from robot.kinematics import Kinematics
from src import landing
from src.landing import LandingCfg
from src.motion import CONTROL_DT, MotionLib
from utils.rotations import mat_to_6d, mm3, mv3, quat_to_mat, mat_to_quat, rot_angle_sq, so3_exp, yaw_mat
from src.dr import XiLayout
from src.sensing import SensingCfg, obs_noise


def state_dim(J: int) -> int:
    return 3 + 4 + 3 + 3 + J + J


def obs_dim(J: int) -> int:
    return J + J + 3 + 3 + J + 6


# the step-symmetry term's gait memory (VelocityCfg.w_sym), per velocity row: time since the last counted touchdown
# [s], the previous step's time [s] and lead [m], the last counted foot (+1 left, -1 right, 0 none), whether the
# previous step is valid (0/1), each foot's swing state (0/1) and its length so far [control steps] (left, right)
GAIT_DIM = 9
GAIT_FEAT = 5  # MDP.gait_features: the velocity critic's view of it


def swing_step(on: torch.Tensor, n: torch.Tensor, sp: torch.Tensor, v_on: float, v_off: float, n_min: int):
    """One control step of a swing detector (gait.swings in streaming form) on horizontal speeds sp: in swing (bool),
    its length so far, and whether a swing of at least n_min steps ended here (a touchdown)."""
    n1 = torch.where(on, n + 1, torch.zeros_like(n))
    end = on & (sp < v_off)
    return (on & ~end) | (~on & (sp > v_on)), n1, end & (n1 >= n_min)


def gait_reset(mem: torch.Tensor, mask: torch.Tensor, full: bool) -> torch.Tensor:
    """The gait memory with rows ``mask`` reset: all of it at an episode reset (full), at a command resample only the
    step comparison (timer, last foot, valid), keeping the swings in progress."""
    z = torch.zeros_like(mem[..., :1])  # built on the device: the simulator's step runs inside a captured CUDA graph
    reset = torch.zeros_like(mem) if full else torch.cat((z, mem[..., 1:3], z, z, mem[..., 5:]), -1)
    return torch.where(mask.unsqueeze(-1), reset, mem)


CONTACT_Z = 0.045  # ankle-roll link origin height below which a foot counts as in contact
MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT


@dataclass
class RewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8


@dataclass
class VelocityCfg:
    """Velocity-command task (train.py --vel_frac), for the T1.

    Commands and termination as bfm_wm (SuperMimic Appendix G on the T1 library): command = (vx, vy) [m/s] in the
    heading frame and the world yaw rate [rad/s]. Reward = HuMBLE's goal-conditioned reward (arXiv 2610.10489,
    Section S4 and Table S6, G1 column), every weight times ``scale``."""

    cmd_low: tuple[float, float, float] = (-1.05, -0.8, -2.8)  # the T1 library's 0.1-99.9 percentiles (bfm_wm)
    cmd_high: tuple[float, float, float] = (1.4, 0.8, 2.8)
    zero_cmd_prob: float = 0.1  # a sampled command is zeroed (standing)
    ref_cmd_prob: float = 0.5  # the command is read from the reference clip every step, else a sampled one is held
    resample_steps: int = 250  # the command source is resampled at each reset and this often
    w_lin_vel: float = 5.0  # x exp(-|planar velocity error|^2 / lin_vel_scale)
    lin_vel_scale: float = 0.25
    w_yaw_rate: float = 5.0  # x exp(-(yaw rate error)^2 / yaw_rate_scale)
    yaw_rate_scale: float = 0.25
    # not HuMBLE's (train.py --track_sharp; 0 = off): this share of each tracking term is paid by a kernel four times
    # narrower, the ceiling unchanged. HuMBLE's kernel is flat at its top: a policy trained on it settles about
    # 0.04 m/s short, where the kernel's pull (40 x error) meets what speed costs in the smoothness penalties (1.1-1.6
    # per m/s), and stands through commands under 0.3 m/s. At 0.75 the pull near the target is 3.25 times stronger
    track_sharp: float = 0.0
    w_survival: float = 30.0  # constant per step
    w_action_rate: float = -0.5  # x |a_t - a_{t-1}|
    w_torque: float = -0.0005  # x |joint torques| [N m]
    w_feet_slide: float = -0.25  # x sum over the feet in contact of |foot velocity| [m/s]
    w_feet_jerk: float = -0.001  # x sum over the feet of |foot jerk| [m/s^3]
    # not HuMBLE's (train.py --stance_weight; 0 = off): at a standing command, x (1 - exp(-d2 / stance_scale)), d2 the
    # summed squared distance [rad^2] of every joint from the default pose. The clips stand with the hips ahead of the
    # ankles, which loads the ankle motors; their standing frames sit at a median 0.29 rad^2 from the default pose.
    # Every joint, not the legs alone: the joints left out drift (a run on the six leg pitch joints stood with one arm
    # raised and the feet split). Bounded: the steps that stop a walk cost at most |w_stance|. With it on, train.py
    # also drops the latent alignment where the clip's own command is a standing one
    w_stance: float = 0.0
    stance_scale: float = 0.2
    # not HuMBLE's either (train.py --ankle_weight; 0 = off): at a standing command, x (1 - exp(-t2 / ankle_scale)),
    # t2 the summed squared torque [N^2 m^2] of the two ankle pitch motors. It asks for the weight over the ankles and
    # leaves the pose to the clips (the alignment stays): a policy trained without it already stands so, with 5.5 N m
    # in all, when commanded 0.1 m/s backward, against 18 N m at a zero command. Scale = (20 N m)^2
    w_ankle: float = 0.0
    ankle_scale: float = 400.0
    # the ankle term's zero point (train.py --ankle_target; 0 = as before): the torque [N m] asked of each ankle pitch
    # motor in place of none. No torque puts the weight over the ankle axis, and that axis is 10 mm behind the middle of
    # the T1's sole (45% of it from the heel) but 42 mm behind the T2's (30%): every T2 policy trained with the term
    # rests 10-41 mm behind mid-foot. The T2's weight over mid-foot is 8.5 N m per ankle (40.9 kg above the axes)
    ankle_target: float = 0.0
    # and the ankle roll motors' share of t2 (train.py --ankle_roll_weight; 0 = off): their squared torque times this,
    # no torque meaning the pressure on the sole's centre line. On the T2's ankle linkage a roll torque costs the two
    # motors (2.30 / 1.37)^2 = 2.8 times a pitch torque's squared effort. 3 of the 8 T2 seeds rest a foot 20-36 mm
    # toward its outer edge (of 41)
    ankle_roll: float = 0.0
    # not HuMBLE's (train.py --rest_weight; 0 = off): at a standing command, x (1 - exp(-d2 / rest_scale)), d2 the
    # summed squared distance [rad^2] of rest_pose's joints from the angles given there (joint name, rad): the joints a
    # stance leaves free (hip roll and yaw, the waist) at the clips' own standing pose, mirrored. Nothing else says how
    # to stand: the alignment pulls a zero command toward every standing clip at once, and each seed settles on its own
    # stance (the feet 2-12 deg off the chest, the pelvis tipped back 2-14 deg). Not the leg pitch joints, which the
    # ankle term sets, nor the arms. The clips' own standing frames sit at a median 0.026 rad^2 from the T2's pose
    w_rest: float = 0.0
    rest_scale: float = 0.05
    rest_pose: tuple[tuple[str, float], ...] = ()  # a robot's own (RobotSpec.velocity)
    # SuperMimic's own velocity regularizers (its Table A.6), not HuMBLE's (train.py --vel_limit_weight, --lean_weight,
    # --tilt_rate_weight; 0 = off), at every command: the summed overshoot [rad] of the joints past the motion reward's
    # soft limits; sin^2 of the trunk's sideways lean in the heading frame (leaning forward or back is free); the
    # trunk's squared roll and pitch rates [rad^2/s^2] in the heading frame. The paper weighs them -2, -4 and -0.05
    # against tracking terms of 2; times 2.5 (-5, -10, -0.125) they keep that ratio to these tracking terms of 5
    w_limit: float = 0.0
    w_lean: float = 0.0
    w_tilt_rate: float = 0.0
    # not HuMBLE's (train.py --step_weight; 0 = off): a human step length. Each touchdown pays this x ((d - l) / l)^2,
    # d the landing foot's lead over the other foot along the commanded planar direction, l = v T(v) / 2 the step of
    # the cycle law T(v) = A max(v, step_v_min)^-gamma [s] at the commanded planar speed v: Alexander's dynamic-
    # similarity power law in speed form, as ~/humanoid_nmpc's adaptive gait timing, step_cycle = (A, gamma) a robot's
    # own fit to its clips' touchdowns as this term reads them (RobotSpec.velocity), the lead in the root's heading
    # frame. A touchdown: the soft contact indicator rising, now x (1 - one step earlier), about 1 per landing. A
    # penalty, not a bonus: a bonus per touchdown would pay for more steps.
    # Forward commands only (times the forward share vx^2 / |v|^2: side and back steps have no such law), between
    # stand_lin and step_v_max (the fit's fastest walking)
    w_step: float = 0.0
    step_cycle: tuple[float, float] = (0.0, 0.0)
    step_v_min: float = 0.2  # m/s: slower, the cycle time holds at T(step_v_min) and the step shortens
    step_v_max: float = 1.3  # m/s
    # not HuMBLE's (train.py --sym_weight; 0 = off): step symmetry. Each scored touchdown pays this x
    # ((t - t_p) / (t + t_p))^2 + ((d - d_p) / max(|d + d_p|, sym_floor))^2: t the time since the other foot's
    # touchdown, d the landing foot's lead over the other in the root's heading frame, (t_p, d_p) the other foot's
    # previous step (MDP.gait_step; the memory is carried from the simulator through the buffer into imagination).
    # Scored on held forward commands only: within sym_angle of ahead, planar speed >= sym_v_min, turning radius
    # >= sym_radius, steps <= sym_t_max. The T2 clips are even there (step pairs left/right vs the same foot's next
    # step: time 3.6 vs 3.5%, lead 3.7 vs 2.6% within 10 deg; 1-2 m turns 4.1/3.4, 4.8/2.8%) and uneven on diagonals,
    # side and back steps (x2-3), with no steady forward turn tighter than 1 m. A touchdown is a swing's end by the
    # foot's horizontal speed (gait.swings' rule: above swing_on starts one, below swing_off ends it, at least
    # swing_min control steps), not its height: a flat T2 foot rests 1 cm under contact_z
    w_sym: float = 0.0
    sym_v_min: float = 0.2  # m/s
    sym_angle: float = 20.0  # deg
    sym_radius: float = 1.0  # m: planar speed / |yaw rate|
    sym_t_max: float = 2.0  # s
    sym_settle: int = 50  # control steps into an episode before touchdowns count: a reset's recovery shuffle is no step
    sym_floor: float = 0.2  # m
    swing_on: float = 0.15  # m/s
    swing_off: float = 0.08  # m/s
    swing_min: int = 3  # control steps
    # a standing command: planar speed and yaw rate both below these (zero included). A clip that stands still reads
    # as a small command, not zero: 16% of the T1 library's frames are below both
    stand_lin: float = 0.1  # m/s
    stand_yaw: float = 0.2  # rad/s
    # the ceiling (w_lin_vel + w_yaw_rate + w_survival) x scale equals the tracking reward's 5, so both critics share
    # one value bound and neither interface dominates the policy's joint gradient clip
    scale: float = 0.125
    fall_height: float = 0.32  # root height [m] below which the robot has fallen (0.688 m standing)
    fall_tilt: float = 0.7  # of 1 - R_zz: 0 upright, 0.7 at about 73 deg

    @property
    def max_step_reward(self) -> float:
        return self.scale * (self.w_lin_vel + self.w_yaw_rate + self.w_survival) * CONTROL_DT


def _joint_key(name: str) -> str:
    """A joint name in either robot's style: T1 'Left_Ankle_Pitch' and T2 'left_ankle_pitch_joint' -> 'left_ankle_pitch'."""
    return name.lower().removesuffix("_joint")


def ankle_pitch_joints(joint_names) -> list[int]:
    """Indices of the ankle pitch joints (the velocity task's ankle torque term)."""
    return [i for i, n in enumerate(joint_names) if _joint_key(n).endswith("ankle_pitch")]


def leg_joints(joint_names) -> list[int]:
    """Indices of the hip, knee and ankle joints (the landing command's 12 leg joints)."""
    return [i for i, n in enumerate(joint_names) if any(k in _joint_key(n) for k in ("hip", "knee", "ankle"))]


def _norm(x: torch.Tensor) -> torch.Tensor:
    """|x| over the last dim, with a finite gradient at 0."""
    return torch.sqrt((x**2).sum(-1) + 1e-12)


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor):
    J = (x.shape[-1] - 13) // 2
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13 : 13 + J], x[..., 13 + J :]


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(mm3(H.transpose(-1, -2), R)),
            mv3(Rt, v),
            mv3(Rt, w),
            q,
            qd,
        ),
        -1,
    )


def inv_dim(J: int) -> int:
    return 1 + 6 + 3 + 3 + J + J


def upright_rel(g_b: torch.Tensor) -> torch.Tensor:
    """The six reference-orientation entries of a velocity row's observation (MDP.obs with ``upright``) from the
    gravity direction g_b in the root frame alone: mat_to_6d(R^T yaw_mat(R)), which R's heading cancels out of. A
    robot has no reference, so the deploy export (utils/export_onnx.py) rebuilds the entries this way."""
    gx, gy, gz = g_b[..., 0], g_b[..., 1], g_b[..., 2]
    s = torch.sqrt((gx * gx + gy * gy + gz * gz).clamp(min=1e-12))  # written out: a norm exports as an opset-18 op
    gx, gy, gz = gx / s, gy / s, gz / s
    n = torch.sqrt((gy * gy + gz * gz).clamp(min=1e-9))  # cos of the pitch; 0 only lying on the front or the back
    return torch.stack((n, torch.zeros_like(n), -gx * gy / n, -gz / n, -gx * gz / n, gy / n), -1)


class MDP:
    def __init__(self, kin: Kinematics, motion: MotionLib, default_q: torch.Tensor, xi_dim: int,
                 sensing: SensingCfg = SensingCfg(), cfg: RewardCfg = RewardCfg()):  # fmt: skip
        self.kin, self.motion, self.cfg = kin, motion, cfg
        self.default_q = default_q
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * cfg.soft_limit_factor, mid + half * cfg.soft_limit_factor
        dev = default_q.device
        J = default_q.shape[0]
        self.num_j = J
        self.obs_dim, self.xi_dim = obs_dim(J), xi_dim
        self.critic_extra = J + len(kin.tracked_idx) * 3 + 3 + 6 + self.xi_dim
        self.obs_noise = obs_noise(sensing, J).to(dev)
        self.sensing = sensing
        self.noise_grow = sensing.noise_scale_to - 1.0  # add_noise: 0 on every robot but a measured one
        # train.py --dr_ramp_iters: 0 at the start of training, 1 from the ramp's end on (and wherever nothing sets it).
        # A 0-dim device tensor filled in place, so captured graphs follow it. It scales a measured robot's gain
        # randomization (src/dr.py), sensor errors and noise growth; robots without those never read it
        self.ramp = torch.ones((), device=dev)
        self.gain_sl = None  # enable_velocity: xi's (kp, kd) factor slices of a robot whose gains are randomized
        # contact reward (training only): tracked-body indices of the feet, set by the trainer
        self.feet: list[int] = []
        self.contact_z = CONTACT_Z

    # ---------------- observation ----------------
    def obs(self, S, a_prev, xi, f, upright=None, sensor=None) -> torch.Tensor:
        """Noise-free proprioceptive observation (+ reference-relative root orientation). ``upright`` (bool, shaped as
        f or broadcastable to it; train.py --vel_frac): rows that follow no reference. Theirs stands upright at the
        robot's own heading, so their orientation entries show the robot's tilt alone.
        ``sensor`` (draw_sensor's pair, broadcastable to the rows): a gyro bias (..., 3) added to the angular velocity
        and an IMU mounting tilt (..., 2): the orientation every orientation entry uses (gravity, the upright heading,
        the reference-relative one) is the root's turned about its roll and pitch axes by it, as one IMU gives them."""
        p, R, v, w, q, qd = S
        w_b = mv3(R.transpose(-1, -2), w)
        if sensor is not None:
            bias, tilt = sensor
            w_b = w_b + bias
            R = mm3(R, so3_exp(torch.cat((tilt, torch.zeros_like(tilt[..., :1])), -1)))
        Rt = R.transpose(-1, -2)
        dq0 = xi[..., 1 : 1 + q.shape[-1]]
        g_b = -R[..., 2, :]  # R^T (0,0,-1)
        rR = self.motion.root_R[f]
        if upright is not None:
            rR = torch.where(upright[..., None, None], yaw_mat(R), rR)
        rel = mat_to_6d(mm3(Rt, rR))
        return torch.cat((q - self.default_q - dq0, qd, w_b, g_b, a_prev[..., : self.num_j], rel), -1)

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        if self.noise_grow:  # a measured robot: the half-widths grow with the ramp, x1 to x noise_scale_to
            return o + (2 * torch.rand_like(o) - 1) * (self.obs_noise * (1 + self.noise_grow * self.ramp))
        return o + (2 * torch.rand_like(o) - 1) * self.obs_noise

    def draw_sensor(self, shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor] | None:
        """A gyro bias (*shape, 3) and an IMU tilt (*shape, 2) for obs(sensor=...), uniform within the robot's
        SensingCfg half-widths x the ramp; None, with no random draw, for a robot without sensor errors."""
        c = self.sensing
        if not (c.gyro_bias or c.imu_tilt):
            return None
        dev = self.default_q.device
        return ((2 * torch.rand(*shape, 3, device=dev) - 1) * (c.gyro_bias * self.ramp),
                (2 * torch.rand(*shape, 2, device=dev) - 1) * (c.imu_tilt * self.ramp))  # fmt: skip

    # ---------------- reward / termination ----------------
    def reward(self, S, f, a, a_prev, terms: bool = False, w_contact: float = 0.0):
        """Per-step reward (already multiplied by dt) comparing S with reference frame f.

        ``w_contact`` > 0 adds a contact term: a smooth foot-contact indicator (ankle-roll
        origin below 4.5 cm) matched to the reference's contact state, plus low horizontal foot speed while
        the reference foot is down. Evaluation keeps ``w_contact = 0``, i.e. the paper's reward."""
        c, m = self.cfg, self.motion
        p, R, v, w, q, qd = S
        bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd)
        rbp, rbR, rbv, rbw = m.ref_bodies(f)
        rp, rR = m.root_p[f], m.root_R[f]
        # reference bodies re-anchored to the robot's root (xy + yaw), as in BeyondMimic
        dR = yaw_mat(mm3(R, rR.transpose(-1, -2)))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + mv3(dR.unsqueeze(-3), rbp - rp.unsqueeze(-2))
        ref_bR = mm3(dR.unsqueeze(-3), rbR)
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((rbv - bv) ** 2).sum(-1).mean(-1)
        e_av = ((rbw - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        r_track = (
            torch.exp(-e_pos / c.body_pos_std**2)
            + torch.exp(-e_ori / c.body_ori_std**2)
            + torch.exp(-e_lv / c.body_linvel_std**2)
            + torch.exp(-e_av / c.body_angvel_std**2)
            + 0.5 * torch.exp(-e_rp / c.root_pos_std**2)
            + 0.5 * torch.exp(-e_ro / c.root_ori_std**2)
        )
        r_rate = ((a - a_prev[..., : a.shape[-1]]) ** 2).sum(-1)
        r_lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        r = (r_track + c.w_action_rate * r_rate + c.w_joint_limit * r_lim) * CONTROL_DT
        if w_contact > 0:
            fz = bp[..., self.feet, 2]
            c_sim = torch.sigmoid((self.contact_z - fz) / 0.005)
            rc = getattr(m, "ref_contact", None)  # MotionLib.set_ref_contact (train.py sets it first)
            c_ref = rc[f]
            slip = (bv[..., self.feet, :2] ** 2).sum(-1)
            r_contact = (1 - (c_sim - c_ref) ** 2) + c_ref * torch.exp(-slip / 0.3**2)
            r = r + w_contact * r_contact.mean(-1) * CONTROL_DT
        if terms:
            return r, {
                "track": r_track * CONTROL_DT,
                "e_body_pos": e_pos,
                "e_body_ori": e_ori,
                "e_root_pos": e_rp,
                "action_rate": r_rate,
                "joint_limit": r_lim,
            }
        return r

    def termination(self, S, f) -> torch.Tensor:
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

    def danger(self, S, f) -> torch.Tensor:
        """Danger ratio max(e_z / term_height, e_tilt / term_tilt) of the errors that ``termination`` tests (>= 1:
        terminated)."""
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        return torch.maximum(e_z / self.cfg.term_height, e_tilt / self.cfg.term_tilt)

    def soft_termination(self, S, f, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts."""
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        alive = (1 - torch.sigmoid((e_z - self.cfg.term_height) / tz)) * (1 - torch.sigmoid((e_tilt - self.cfg.term_tilt) / tt))
        return 1 - alive

    def critic_features(self, S, f, xi) -> torch.Tensor:
        """Critic input: invariant state, goal, sim context, and robot-vs-reference errors."""
        m = self.motion
        p, R, v, w, q, qd = S
        H = yaw_mat(R)
        Ht = H.transpose(-1, -2)
        rp, rR = m.root_p[f], m.root_R[f]
        bp = self.kin.forward(p, R, v, w, q, qd)[0]
        dR = yaw_mat(mm3(R, rR.transpose(-1, -2)))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + mv3(dR.unsqueeze(-3), m.ref_bodies(f)[0] - rp.unsqueeze(-2))
        e_body = mv3(Ht.unsqueeze(-3), ref_bp - bp).flatten(-2)
        e_root = mv3(Ht, rp - p)
        rel = mat_to_6d(mm3(R.transpose(-1, -2), rR))
        return torch.cat((invariant_features(S), m.q[f] - q, e_body, e_root, rel, xi), -1)

    # ---------------- velocity-command interface (train.py --vel_frac) ----------------
    def enable_velocity(self, robot, cfg: VelocityCfg = VelocityCfg()) -> None:
        """Bind the velocity task: cfg with the robot's own values (RobotSpec.velocity) over it."""
        assert self.feet, "set mdp.feet first"
        dev = self.default_q.device
        self.vel = cfg = dataclasses.replace(cfg, **robot.velocity)
        self.feet_idx = torch.tensor(self.feet, device=dev)  # a device index: a list would be copied from the host inside captured graphs
        self.cmd_lo, self.cmd_hi = (torch.tensor(x, device=dev) for x in (cfg.cmd_low, cfg.cmd_high))
        self.alpha = torch.tensor(robot.action_scale_list(), device=dev)
        self.kp, self.kd, self.tau_lim = (torch.tensor(x, device=dev) for x in robot.pd_gains())
        sl = XiLayout(robot, dev).slices
        self.gain_sl = (sl["kp"], sl["kd"]) if "kp" in sl else None
        self.ankle_idx = torch.tensor(ankle_pitch_joints(robot.joint_names), device=dev)
        names = list(robot.joint_names)
        self.ankle_roll_idx = torch.tensor([i for i, n in enumerate(names) if _joint_key(n).endswith("ankle_roll")], device=dev)
        assert len(cfg.rest_pose) or not cfg.w_rest, f"w_rest needs a rest_pose, and {robot.name!r} has none"
        assert cfg.step_cycle[0] > 0 or not cfg.w_step, f"w_step needs a step_cycle, and {robot.name!r} has none"
        self.rest_idx = torch.tensor([names.index(n) for n, _ in cfg.rest_pose], dtype=torch.long, device=dev)
        self.rest_q = torch.tensor([v for _, v in cfg.rest_pose], device=dev)

    def enable_landing(self, robot, cfg: LandingCfg = LandingCfg()) -> None:
        """Bind the landing command (after enable_velocity, whose gains and indices it reuses) and the 12 leg joints."""
        assert hasattr(self, "vel"), "call enable_velocity first"
        # ponytail: src/landing.py's sole box (SOLE_C, SOLE_HX/HY) and LandingCfg.ankle_scale are the T1's; move them
        # into RobotSpec, as the velocity values, before the landing command trains on another robot
        assert robot.name == "t1", f"the landing command is T1-only so far (src/landing.py's sole box), not {robot.name!r}"
        self.land = cfg
        self.leg_idx = torch.tensor(leg_joints(robot.joint_names), device=self.default_q.device)  # a device index, as ankle_idx

    def sample_commands(self, n: int) -> torch.Tensor:
        """n commands uniform over the command box, each zeroed with zero_cmd_prob."""
        u = torch.rand(n, 4, device=self.cmd_lo.device)
        return (self.cmd_lo + (self.cmd_hi - self.cmd_lo) * u[:, :3]) * (u[:, 3:] >= self.vel.zero_cmd_prob)

    def normalize_cmd(self, cmd: torch.Tensor) -> torch.Tensor:
        """Commands mapped from the command box to [-1, 1]."""
        return (2 * cmd - self.cmd_hi - self.cmd_lo) / (self.cmd_hi - self.cmd_lo)

    def standing(self, cmd: torch.Tensor) -> torch.Tensor:
        """Commands (..., 3) that ask the robot to stand: below VelocityCfg's stand_lin and stand_yaw."""
        return ((cmd[..., :2] ** 2).sum(-1) < self.vel.stand_lin**2) & (cmd[..., 2].abs() < self.vel.stand_yaw)

    def foot_kin(self, S) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Heights (B, feet), world linear velocities (B, feet, 3) and world planar positions (B, feet, 2) of the feet's
        body origins."""
        bp, _, bv, _ = self.kin.forward(*S)
        return bp[:, self.feet_idx, 2], bv[:, self.feet_idx], bp[:, self.feet_idx, :2]

    def pd_torque(self, q, qd, a, xi) -> torch.Tensor:
        """The joint torques of action a at joint state (q, qd), as the velocity reward estimates them: the PD law with
        each row's own gains (the robot's, times xi's kp and kd factors where its gains are randomized)."""
        # ponytail: the PD law on the newest setpoint; ignores the actuator delay and the torque-speed knee
        kp, kd = self.kp, self.kd
        if self.gain_sl is not None:
            kp, kd = kp * xi[..., self.gain_sl[0]], kd * xi[..., self.gain_sl[1]]
        return torch.clamp(kp * (self.default_q + xi[..., 1 : 1 + q.shape[-1]] + self.alpha * a - q) - kd * qd,
                           -self.tau_lim, self.tau_lim)  # fmt: skip

    def velocity_reward(self, S, cmd, a, a_prev, xi, foot_z, foot_v, foot_v0, foot_v1, jerk_on, terms: bool = False,
                        foot_xy=None, foot_z0=None):
        """VelocityCfg's reward (already multiplied by dt) of the state S that action a led to, under command cmd.

        foot_z, foot_v, foot_xy: foot_kin(S); foot_v0, foot_v1: the feet's velocities one and two control steps
        earlier, foot_z0 their heights one step earlier (both of the latter only with w_step); jerk_on (bool): these
        belong to this episode (false on an episode's first two steps: no jerk or step term there)."""
        c = self.vel
        p, R, v, w, q, qd = S
        Rh = yaw_mat(R).transpose(-1, -2)  # world to heading frame
        e_lin = ((mv3(Rh, v)[..., :2] - cmd[..., :2]) ** 2).sum(-1)
        e_yaw = (w[..., 2] - cmd[..., 2]) ** 2
        rate = _norm(a - a_prev[..., : a.shape[-1]])
        tau = self.pd_torque(q, qd, a, xi)
        torque = _norm(tau)
        slide = (torch.sigmoid((self.contact_z - foot_z) / 0.005) * _norm(foot_v)).sum(-1)
        jerk = (_norm(foot_v - 2 * foot_v0 + foot_v1) / CONTROL_DT**2).sum(-1) * jerk_on
        stand = self.standing(cmd)
        stance = stand * (1 - torch.exp(-((q - self.default_q) ** 2).sum(-1) / c.stance_scale))
        ankle_t2 = ((tau[..., self.ankle_idx] - c.ankle_target) ** 2).sum(-1)
        if c.ankle_roll:
            ankle_t2 = ankle_t2 + c.ankle_roll * (tau[..., self.ankle_roll_idx] ** 2).sum(-1)
        ankle = stand * (1 - torch.exp(-ankle_t2 / c.ankle_scale))
        rest = stand * (1 - torch.exp(-((q[..., self.rest_idx] - self.rest_q) ** 2).sum(-1) / c.rest_scale))
        limit = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        lean = mv3(Rh, R[..., :, 2])[..., 1] ** 2  # the trunk z axis' heading-frame y, squared
        tilt_rate = (mv3(Rh, w)[..., :2] ** 2).sum(-1)
        step = torch.zeros_like(e_lin)
        assert foot_xy is not None or not c.w_step, "w_step needs foot_xy and foot_z0"
        if c.step_cycle[0] > 0 and foot_xy is not None:  # measured wherever the robot has a law (a gait log), paid at w_step
            A, gamma = c.step_cycle
            vn = cmd[..., :2].norm(dim=-1)
            ell = vn * A * vn.clamp(min=c.step_v_min) ** -gamma / 2
            u = cmd[..., :2] / vn.clamp(min=1e-6).unsqueeze(-1)  # the commanded direction, heading frame
            d = foot_xy[..., 0, :] - foot_xy[..., 1, :]
            lead = (mv3(Rh, torch.cat((d, torch.zeros_like(d[..., :1])), -1))[..., :2] * u).sum(-1)
            lead = torch.stack((lead, -lead), -1)  # each foot's lead over the other
            down = lambda z: torch.sigmoid((self.contact_z - z) / 0.005)  # noqa: E731  the slide term's contact
            err = ((lead - ell.unsqueeze(-1)) / ell.clamp(min=1e-3).unsqueeze(-1)) ** 2
            share = torch.relu(cmd[..., 0]) ** 2 / (vn**2).clamp(min=1e-12)
            gate = (vn >= c.stand_lin) & (vn <= c.step_v_max) & jerk_on
            step = (down(foot_z) * (1 - down(foot_z0)) * err).sum(-1) * share * gate
        k_lin, k_yaw = torch.exp(-e_lin / c.lin_vel_scale), torch.exp(-e_yaw / c.yaw_rate_scale)
        if c.track_sharp:  # exp(-4 e^2 / scale) = k^4
            k_lin, k_yaw = (k + c.track_sharp * (k**4 - k) for k in (k_lin, k_yaw))
        r = (
            c.w_lin_vel * k_lin
            + c.w_yaw_rate * k_yaw
            + c.w_survival
            + c.w_action_rate * rate
            + c.w_torque * torque
            + c.w_feet_slide * slide
            + c.w_feet_jerk * jerk
            + c.w_stance * stance
            + c.w_ankle * ankle
            + c.w_rest * rest
            + c.w_limit * limit
            + c.w_lean * lean
            + c.w_tilt_rate * tilt_rate
            + c.w_step * step
        ) * (c.scale * CONTROL_DT)
        if terms:
            return r, {"e_lin": e_lin, "e_yaw": e_yaw, "rate": rate, "torque": torque, "slide": slide, "jerk": jerk,
                       "stance": stance, "ankle": ankle, "rest": rest, "limit": limit, "lean": lean,
                       "tilt_rate": tilt_rate, "step": step}  # fmt: skip
        return r

    def _fall_margins(self, S):
        """Root height above fall_height [m] and 1 - R_zz below fall_tilt; either negative means fallen."""
        return S[0][..., 2] - self.vel.fall_height, self.vel.fall_tilt - (1 - S[1][..., 2, 2])

    def fall_termination(self, S) -> torch.Tensor:
        h, t = self._fall_margins(S)
        return (h < 0) | (t < 0)

    def soft_fall_termination(self, S, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
        """fall_termination as a differentiable probability, smoothed as soft_termination."""
        h, t = self._fall_margins(S)
        return 1 - torch.sigmoid(h / tz) * torch.sigmoid(t / tt)

    def velocity_critic_features(self, S, cmd, xi_n, a_prev, gait_f=None) -> torch.Tensor:
        """Velocity critic input: invariant state, normalized sim context and command, the pending actions, and with
        --sym_weight the gait memory's features (gait_f, gait_features)."""
        x = (invariant_features(S), xi_n, self.normalize_cmd(cmd), a_prev)
        return torch.cat(x if gait_f is None else (*x, gait_f), -1)

    # ---------------- step symmetry (VelocityCfg.w_sym, train.py --sym_weight) ----------------
    def gait_gate(self, cmd: torch.Tensor, cmd_ref: torch.Tensor) -> torch.Tensor:
        """Commands (..., 3) whose steps the symmetry term scores: held (cmd_ref false), forward within sym_angle of
        ahead, planar speed >= sym_v_min, turning radius >= sym_radius."""
        c = self.vel
        vx, vy, w = cmd.unbind(-1)
        sp = torch.hypot(vx, vy)
        return (~cmd_ref & (vx > 0) & (torch.atan2(vy.abs(), vx) <= math.radians(c.sym_angle)) & (sp >= c.sym_v_min)
                & (sp >= c.sym_radius * w.abs()))  # fmt: skip

    def gait_step(self, mem, foot_v, foot_xy, R, gate, settled):
        """One control step of the gait memory (GAIT_DIM) on the state just reached: foot_v (B, 2, 3), foot_xy (B, 2, 2)
        (foot_kin's, left then right), R (B, 3, 3) its root rotation, gate (B,) gait_gate, settled (B,) the step's
        episode step >= sym_settle. A touchdown counts where settled, if it is
        the other foot's than the last counted one (one per step, left first); it is scored where gate, the previous
        step valid and both steps no longer than sym_t_max. Returns the unweighted penalty (B,), zero where not scored
        and differentiable through foot_xy alone, scored (B,) and the updated memory (detached)."""
        c = self.vel
        t = mem[..., 0] + CONTROL_DT
        t_p, d_p, last, valid = mem[..., 1], mem[..., 2], mem[..., 3], mem[..., 4] > 0.5
        on, n, td = swing_step(mem[..., 5:7] > 0.5, mem[..., 7:9], foot_v[..., :2].norm(dim=-1), c.swing_on, c.swing_off,
                               c.swing_min)  # fmt: skip
        left = td[..., 0] & (last != 1)
        counted = (left | (td[..., 1] & (last != -1))) & settled
        diff = foot_xy[..., 0, :] - foot_xy[..., 1, :]
        lead = mv3(yaw_mat(R.detach()).transpose(-1, -2), torch.cat((diff, torch.zeros_like(diff[..., :1])), -1))[..., 0]
        d = torch.where(left, lead, -lead)
        scored = counted & gate & valid & (t <= c.sym_t_max)
        # t + t_p >= CONTROL_DT and the floor keep both denominators positive: torch.where's unselected branch stays
        # finite, so its zero gradient stays zero
        err = ((t - t_p) / (t + t_p)) ** 2 + ((d - d_p) / (d + d_p).abs().clamp(min=c.sym_floor)) ** 2
        pen = torch.where(scored, err, torch.zeros_like(err))
        mem1 = torch.cat((torch.stack((
            torch.where(counted, torch.zeros_like(t), t),
            torch.where(counted, t, t_p),
            torch.where(counted, d.detach(), d_p),
            torch.where(counted, torch.where(left, 1.0, -1.0), last),
            torch.where(counted, (last != 0) & (t <= c.sym_t_max), valid).float(),
        ), -1), on.float(), n), -1)  # fmt: skip
        return pen, scored, mem1.detach()

    def gait_rollout(self, mem0, foot_v, foot_xy, R, gate, settled):
        """gait_step over H steps (foot_v (H, B, 2, 3), foot_xy (H, B, 2, 2), R (H, B, 3, 3), settled (H, B)) from mem0: penalties and
        scored (H, B) and the memories (H + 1, B, GAIT_DIM), mems[0] = mem0."""
        pens, scs, mems = [], [], [mem0]
        for h in range(foot_v.shape[0]):
            p, s, m = self.gait_step(mems[-1], foot_v[h], foot_xy[h], R[h], gate, settled[h])
            pens.append(p), scs.append(s), mems.append(m)
        return torch.stack(pens), torch.stack(scs), torch.stack(mems)

    def gait_features(self, mem: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """The velocity critic's view of the gait memory (..., GAIT_FEAT): time since the last counted touchdown (at
        most 2 s), the previous step's time and lead, the last foot, and whether the next touchdown can be scored."""
        return torch.stack((mem[..., 0].clamp(max=2.0), mem[..., 1], mem[..., 2], mem[..., 3], mem[..., 4] * gate.float()), -1)

    # ---------------- landing command ----------------
    def sole_kin(self, S) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """``landing.soles`` of the two foot links of S, and the velocities of the link origins (B, 2, 3)."""
        bp, bR, bv, bw = (x[:, self.feet_idx] for x in self.kin.forward(*S))
        return landing.soles(bp, bR, bv, bw), bv

    def land_step(self, S1, C, pose0, a, a_prev, xi, foot_v0, foot_v1, jerk_on):
        """LandingCfg's reward (already multiplied by dt) of the state S1 that action a led to, under the command state
        C (B, 12), and C advanced one step; pose0 (B, 2, 3): the sole poses the step started from; foot_v0, foot_v1: the
        feet's link velocities one and two control steps earlier; jerk_on (bool): both belong to this episode.
        Returns the reward (B,), the advanced C, the sole poses of S1 (B, 2, 3), the feet's link velocities on S1
        (B, 2, 3) and landing.land_step's terms."""
        p, R, v, w, q, qd = S1
        sole, foot_v = self.sole_kin(S1)
        rate = _norm(a - a_prev[..., : a.shape[-1]])
        tau = torch.clamp(self.kp * (self.default_q + xi[..., 1 : 1 + q.shape[-1]] + self.alpha * a - q) - self.kd * qd,
                          -self.tau_lim, self.tau_lim)  # the PD law of velocity_reward
        torque = _norm(tau)
        jerk = (_norm(foot_v - 2 * foot_v0 + foot_v1) / CONTROL_DT**2).sum(-1) * jerk_on
        ankle_t2 = (tau[..., self.ankle_idx] ** 2).sum(-1)
        still_v = torch.maximum(qd[..., self.leg_idx].abs().amax(-1), mv3(R.transpose(-1, -2), w).abs().amax(-1))
        r, C1, terms = landing.land_step(self.land, C, pose0, sole, rate, torque, jerk, ankle_t2, still_v)
        return r, C1, sole["pose"], foot_v, terms

    def land_features(self, C, o) -> torch.Tensor:
        """The 12 command numbers of ``enc_land`` (B, 12) from the command state C (B, 12) and one unnormalized
        observation o (B, obs_dim): the swing sole in the planted sole's frame, with the root at the identity and the
        joints as o reads them. Zeros in the derived inputs when LandingCfg.derived is false."""
        B, kw = len(C), {"device": C.device, "dtype": C.dtype}
        if self.land.derived:
            q = o[..., : self.num_j] + self.default_q
            z3 = torch.zeros(B, 3, **kw)
            bp, bR, _, _ = self.kin.forward(z3, torch.eye(3, **kw).expand(B, 3, 3), z3, z3, q, torch.zeros_like(q))
            rel5 = landing.sole_rel(bp[:, self.feet_idx], bR[:, self.feet_idx], C[:, landing.SWING])
        else:
            rel5 = torch.zeros(B, 5, **kw)
        return landing.land_features(self.land, C, rel5)

    def land_critic_features(self, S, C, xi_n, a_prev) -> torch.Tensor:
        """Landing critic input: invariant state, normalized sim context, the pending actions, the landing features."""
        lf = landing.land_critic_features(C, self.sole_kin(S)[0])
        return torch.cat((invariant_features(S), xi_n, a_prev, lf), -1)
