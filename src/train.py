"""BlinkMimic 2 training: SuperMimic's learner (collect -> fit world model -> BPTT policy + critic update) with exams,
frontier practice, a deployment stream and a step size planned for the budget. The defaults are the recipe of the
T1's velocity controller (run logs/t1_vel_ankle_sharp_x2_s1; in deploy/ until the T2's policy replaced it).
--land_frac adds the landing command (src/landing.py): a foot lands on a target pose by a deadline, then the robot
stands. The add-on run (utils/t1_land_run.sh) trains its encoder alone on a finished policy that it loads and freezes.

usage: bash utils/t1_run.sh <run name> [train flags]   or   python -m src.train --log_dir logs/<run> [flags]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The defaults are the deployed controller's recipe: BlinkMimic 2's schedule at twice its length (every iteration number
# below, --iterations and --ckpt_every doubled: utils/t1_run.sh's STRETCH=2) with the velocity interface, the ankle term
# and --track_sharp. BlinkMimic 2's own motion-only recipe:
#   STRETCH=1 bash utils/t1_run.sh <run> --pi_ref_skip --vel_frac 0 --ankle_weight 0 --track_sharp 0
# The paper's baseline recipe (utils/t1_run.sh's comment) turns the late-phase rules off.
FINAL_PHASE = ("0:exam_frac=0,front_share=0,fail_penalty=0,reset_aprev_ref=0,lr_decay_iters=500000,lr_end=0.03,wstream_frac=0;"
               "4000:exam_frac=0.5,front_share=1,fail_penalty=3,reset_aprev_ref=1,lr_decay_iters=11514,lr_end=0.3;"
               "6000:exam_frac=0.5,lr_decay_iters=12400,wstream_frac=0.25,critic_epochs=1;"
               "9000:exam_frac=0.9")
D = os.path.join(REPO, "data", "t1")
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
# ---- run ----
parser.add_argument("--iterations", type=int, default=13000)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "run"))
parser.add_argument("--ckpt_every", type=int, default=1000, help="deployed (averaged) policy checkpoints")
parser.add_argument("--log_every", type=int, default=100)
parser.add_argument("--bones_lib", type=str, default=os.path.join(D, "lib_all.npz"), help="packed library (utils/prepare_t1.py)")
parser.add_argument("--bones_list", type=str, default=os.path.join(D, "keep.txt"), help="clip names to train on, one per line")
parser.add_argument("--motion_store", type=str, default="fp32", choices=["full", "fp32", "fp16"],
                    help="full: every per-frame quantity precomputed; fp32 / fp16: root pose and joints only, the rest computed when indexed")
parser.add_argument("--robot", type=str, default="t1", choices=["t1", "t2", "t2_measured"], help="robot spec: robot/<name>.py")
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
# ---- learner (SuperMimic with the BlinkMimic changes) ----
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--buffer_steps", type=int, default=1000, help="replay buffer length (simulator steps)")
parser.add_argument("--warmup_steps", type=int, default=64, help="simulator steps before the first update")
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32, help="imagined steps H")
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8, help="world-model training window (steps)")
parser.add_argument("--wm_batches", type=int, default=5, help="world-model updates per iteration")
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--wm_hidden", type=str, default="1024,1024,1024", help="world-model MLP widths")
parser.add_argument("--wm_mirror", type=float, default=0.5, help="share of each world-model batch replaced by its left-right mirror")
parser.add_argument("--enc_hidden", type=str, default="2048,1024,512,256", help="motion-encoder MLP widths")
parser.add_argument("--dec_hidden", type=str, default="2048,1024,512,256", help="shared-decoder MLP widths")
parser.add_argument("--pi_batch", type=int, default=2048, help="imagined rollouts per policy update")
parser.add_argument("--pi_ref_skip", action=argparse.BooleanOptionalAction, default=False,
                    help="actions on top of the reference pose, a = (q_ref - q0) / alpha + network output")
parser.add_argument("--lr", type=float, default=4e-4)
parser.add_argument("--pi_lr_mult", type=float, default=1.6, help="policy learning rate = --lr x this")
parser.add_argument("--lr_decay_iters", type=int, default=250_000, help="the cosine reaches its floor here (also a --phase key)")
parser.add_argument("--lr_end", type=float, default=0.03, help="floor of the cosine, x --lr (also a --phase key)")
parser.add_argument("--pi_beta1", type=float, default=0.9)
parser.add_argument("--pi_clip_value", type=float, default=0.1, help="element-wise policy gradient clip after the norm clip")
parser.add_argument("--pi_ema", type=float, default=0.995, help="EMA of the policy weights; checkpoints store the average")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5, help="also a --phase key")
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--contact_reward", type=float, default=0.25, help="contact-matching term in imagination (evaluation reward unchanged)")
parser.add_argument("--fail_reset", type=float, default=0.3, help="share of practice resets one bin before recent failure points")
# ---- velocity-command interface (the T1 only; on in the default recipe, --vel_frac 0 turns it off) ----
parser.add_argument("--vel_frac", type=float, default=0.5,
                    help="share of the simulated robots and of each imagined batch that follows a velocity command through "
                         "its own encoder and critic, the decoder's weights frozen for them (SuperMimic Appendix G with "
                         "HuMBLE's goal reward, mdp.VelocityCfg); needs --no-pi_ref_skip")
parser.add_argument("--align_weight", type=float, default=0.5,
                    help="pull of the velocity latent toward the motion latent on rollouts whose command is read from the clip")
parser.add_argument("--vel_mirror_weight", type=float, default=0.0,
                    help="weight of the velocity path's left-right mirror loss (src/mirror.py policy_mirror_error: the "
                         "action in the mirrored situation against the mirror of the action; trains the velocity encoder "
                         "only); 0 = off")
parser.add_argument("--stance_weight", type=float, default=0.0,
                    help="velocity reward, not HuMBLE's: at a standing command (zero, or below mdp.VelocityCfg's stand_lin "
                         "and stand_yaw), this x a bounded distance of the joints from the default pose; "
                         "negative penalizes, 0 = off. When on, --align_weight skips the rollouts whose clip command is "
                         "a standing one")
parser.add_argument("--track_sharp", type=float, default=0.75,
                    help="velocity reward, not HuMBLE's: this share of each command-tracking term is paid by a kernel four "
                         "times narrower, for tighter tracking near the target (mdp.VelocityCfg.track_sharp; 0 = off)")
parser.add_argument("--ankle_weight", type=float, default=-4.0,
                    help="velocity reward, not HuMBLE's: at a standing command, this x a bounded measure of the two ankle "
                         "pitch motors' torque (mdp.VelocityCfg.w_ankle); negative penalizes, 0 = off. The alignment is "
                         "untouched")
parser.add_argument("--ankle_target", type=float, default=0.0,
                    help="the torque [N m] --ankle_weight's term asks of each ankle pitch motor in place of none "
                         "(mdp.VelocityCfg.ankle_target): 8.5 puts the T2's weight over mid-foot, 0 over the ankle axis")
parser.add_argument("--ankle_roll_weight", type=float, default=0.0,
                    help="share of the ankle roll motors' squared torque in --ankle_weight's term "
                         "(mdp.VelocityCfg.ankle_roll; 0 = off; 2.8 = the T2 linkage's motor effort)")
parser.add_argument("--rest_weight", type=float, default=0.0,
                    help="velocity reward, not HuMBLE's: at a standing command, this x a bounded distance of the robot's "
                         "rest_pose joints (hip roll and yaw, the waist) from the clips' standing pose "
                         "(mdp.VelocityCfg.w_rest); negative penalizes, 0 = off. The alignment is untouched")
parser.add_argument("--vel_limit_weight", type=float, default=0.0,
                    help="velocity reward, SuperMimic's Table A.6 (not HuMBLE's): this x the joints' summed overshoot "
                         "[rad] past the soft joint limits (mdp.VelocityCfg.w_limit; 0 = off; -5 = the paper's -2 at "
                         "its ratio to these tracking terms)")
parser.add_argument("--lean_weight", type=float, default=0.0,
                    help="velocity reward, SuperMimic's Table A.6: this x sin^2 of the trunk's sideways lean, forward "
                         "and back free (mdp.VelocityCfg.w_lean; 0 = off; -10 = the paper's -4 at that ratio)")
parser.add_argument("--tilt_rate_weight", type=float, default=0.0,
                    help="velocity reward, SuperMimic's Table A.6: this x the trunk's squared roll and pitch rates "
                         "(mdp.VelocityCfg.w_tilt_rate; 0 = off; -0.125 = the paper's -0.05 at that ratio)")
parser.add_argument("--step_weight", type=float, default=0.0,
                    help="velocity reward, not HuMBLE's: this x ((d - l) / l)^2 at each touchdown, d the landing foot's "
                         "lead over the other along the command, l the step of the robot's cycle law at the commanded "
                         "speed (mdp.VelocityCfg.w_step, step_cycle; forward commands 0.1-1.3 m/s; negative penalizes, "
                         "0 = off; the term is logged as vel_imag_step either way)")
parser.add_argument("--sym_weight", type=float, default=0.0,
                    help="velocity reward, not HuMBLE's: this x the step-symmetry penalty at each scored touchdown "
                         "(mdp.VelocityCfg.w_sym; held forward commands within 20 deg, radius >= 1 m; negative "
                         "penalizes, 0 = off)")
# ---- landing-command interface (the T1 only; off in the default recipe: utils/t1_land_run.sh is its add-on run) ----
parser.add_argument("--land_frac", type=float, default=0.0,
                    help="share of the simulated robots and of each imagined batch that puts a commanded foot on a "
                         "target pose by a deadline and then stands, through its own encoder and critic, the decoder's "
                         "weights frozen for them (landing.LandingCfg); needs --vel_frac > 0; 0 = off")
parser.add_argument("--land_align_weight", type=float, default=2.0,
                    help="pull of the landing latent toward the motion latent on the landing rows that follow a clip's "
                         "labelled steps")
parser.add_argument("--land_label_prob", type=float, default=0.5,
                    help="share of the landing robots' episodes that follow a clip and are given its labelled steps as "
                         "commands; the others stay on a standing frame and are given sampled commands "
                         "(landing.LandingCfg.label_prob)")
parser.add_argument("--land_bonus", type=float, default=10.0,
                    help="landing reward: the lump paid once at the deadline, x the landing score G "
                         "(landing.LandingCfg.bonus)")
parser.add_argument("--land_drag", type=float, default=-10.0,
                    help="landing reward: this x the sum over both soles of (the sole centre's horizontal speed [m/s] "
                         "+ 0.1124 m x its |yaw rate| [rad/s]), counted while the sole's centre is within about 5 mm "
                         "of the floor (landing.LandingCfg.w_drag); negative penalizes a foot slid or turned on the "
                         "floor to its target, 0 = off")
parser.add_argument("--land_arrive", type=float, default=0.04,
                    help="landing reward: position scale [m] of the narrow kernel of the landing score G "
                         "(landing.LandingCfg.arrive_narrow)")
parser.add_argument("--land_progress", type=float, default=5.0,
                    help="landing reward: paid per metre the swing sole approaches its target in the air and charged "
                         "per metre it retreats (landing.LandingCfg.w_progress); 0 = off")
parser.add_argument("--land_derived", action=argparse.BooleanOptionalAction, default=True,
                    help="the landing encoder also receives where the swing sole is now in the planted sole's frame, "
                         "computed from the observed joint angles (landing.LandingCfg.derived); off zeroes those five "
                         "inputs")
parser.add_argument("--init_from", type=str, default="",
                    help="checkpoint to start from (a run's final checkpoint or last.pt: the earlier ones hold no "
                         "world model): its policy, where only the landing encoder may be missing, its world model "
                         "and, if the file has them, its critics; the optimizers and the iteration count start fresh")
parser.add_argument("--freeze_base", action="store_true",
                    help="with --init_from and --land_frac: the loaded motion encoder, velocity encoder and decoder "
                         "get no gradient, the observation and goal normalizers are not updated and the policy "
                         "optimizer holds the landing encoder alone; at every checkpoint the loaded tensors are "
                         "compared with the policy about to be saved and the run stops if one differs")
# ---- schedule ----
parser.add_argument("--phase", type=str, default=FINAL_PHASE,
                    help="'IT:key=v,...;IT:key=v' sets, from iteration IT on, the keys exam_frac, front_share, fail_penalty, "
                         "start_prac_frac, reset_aprev_ref (0/1), lr_decay_iters, lr_end, wstream_frac, critic_epochs, wm_batches "
                         "(each entry keeps the earlier ones' values). The flags give the capacity: --exam_frac is the largest "
                         "exam share, --fail_penalty / --front_share > 0 and --reset_aprev_ref switch those paths on")
parser.add_argument("--dr_ramp_iters", type=int, default=0,
                    help="a measured robot (robot/t2_measured.py): the iterations over which its gain randomization, "
                         "sensor errors and observation-noise growth go from zero to full (src.dr.ramp_at); 0 = full "
                         "from the start. No effect on a robot without those")
parser.add_argument("--phase_refill", action=argparse.BooleanOptionalAction, default=True,
                    help="when the exam worlds change, collect a full buffer before the updates resume")
parser.add_argument("--share_ramp", type=str, default="0.5,0.9,50,0.1",
                    help="'lo,hi,n0,step': after each pre-exam, exam share = hi - (hi - lo) min(1, n1 / n0) on the step grid, "
                         "n1 = clips the pre-exam predicts to fail; '' = off")
# ---- exams and frontier practice ----
parser.add_argument("--exam_frac", type=float, default=0.9,
                    help="largest share of the simulated robots that play whole clips from their first frame with the EMA "
                         "policy (exams); 0 = off")
parser.add_argument("--exam_mix", type=str, default="0.6,0.2",
                    help="'p,q': shares of the exam clips drawn by margin priority (p), by training failure rate (q) and "
                         "uniformly (rest); '' = uniform")
parser.add_argument("--exam_s0", type=float, default=0.15, help="--exam_mix: noise of one exam's danger ratio")
parser.add_argument("--exam_v", type=float, default=5e-5, help="--exam_mix: margin drift variance per iteration")
parser.add_argument("--exam_every", type=int, default=250, help="exam clip table refresh interval (iterations)")
parser.add_argument("--exam_pre_every", type=int, default=500,
                    help="pre-exams: every this many iterations every exam clip is played in the world model from its first "
                         "frame with the deployed policy; its max danger ratio is a second estimate of the clip's margin; 0 = off")
parser.add_argument("--exam_pre_steps", type=int, default=300, help="imagined steps of a pre-exam")
parser.add_argument("--exam_pre_s", type=float, default=0.2, help="noise of a pre-exam score as a margin estimate")
parser.add_argument("--exam_pre_near", type=float, default=0.3,
                    help="re-score only clips whose margin estimate is >= this, plus a rotating 1/--exam_pre_rot of the rest")
parser.add_argument("--exam_pre_rot", type=int, default=8)
parser.add_argument("--front_share", type=float, default=1.0,
                    help="share of practice resets sent to the frontier of clips that failed (their last --front_win frames "
                         "before the failure to the clip end); 0 = off")
parser.add_argument("--front_win", type=int, default=50)
parser.add_argument("--front_every", type=int, default=50, help="frontier refresh interval (iterations)")
parser.add_argument("--start_prac_frac", type=float, default=0.1,
                    help="share of imagination starts from practice transitions (the rest from exams); < 0 = uniform")
parser.add_argument("--fail_penalty", type=float, default=3.0, help="penalty x soft termination probability per imagined step")
parser.add_argument("--pen_horizon_only", action=argparse.BooleanOptionalAction, default=True,
                    help="the penalty acts inside the imagined horizon only (the critic learns the tracking reward)")
parser.add_argument("--reset_aprev_ref", action=argparse.BooleanOptionalAction, default=True,
                    help="a reset in the middle of a clip sets the previous action to the one that holds the reset pose")
# ---- deployment stream ----
parser.add_argument("--wstream_frac", type=float, default=0.25,
                    help="share of imagination starts taken from robots that run the EMA policy in the world model as the "
                         "evaluation does (first frame, no noise); also a --phase key")
parser.add_argument("--wstream_worlds", type=int, default=8192)
parser.add_argument("--wstream_steps", type=int, default=2, help="stream steps per iteration")
parser.add_argument("--wstream_burn", type=int, default=300, help="stream advances when it starts")
args = parser.parse_args()

import torch
import torch.nn as nn


from src.buffer import ReplayBuffer
from src.dr import XiLayout, ramp_at
from src.env import PHYSICS_DT, TrackingSim, make_sim
from robot.kinematics import Kinematics
from src import landing
from src.landing import LandingCfg
from src.mdp import GAIT_FEAT, MAX_STEP_REWARD, MDP, VelocityCfg, inv_dim, unflatten_state
from src.motion import CONTROL_DT, MotionLib
import src.networks as networks
import utils.rotations as rotations
from src.networks import Critic, Policy, WorldModel, policy_kwargs
from src.mirror import Mirror, assert_mirrorable_box, policy_mirror_error
from src.objective import actor_objective
from src.mdp import flatten_state
from robot.spec import get_robot, robot_numbers
from src.sensing import ObsPipeline

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
dev = "cuda:0"
os.makedirs(args.log_dir, exist_ok=True)
robot = get_robot(args.robot)
# the gains, friction and action scale this run trains against, saved with its arguments (args.json and every
# checkpoint): a checkpoint is refused on a robot whose numbers differ (robot.spec.check_robot_numbers)
args.robot_numbers = robot_numbers(robot)
json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

J = robot.num_joints
kin = Kinematics(robot, dev)
motion = MotionLib.load(kin, dev, args.bones_list, args.bones_lib, args.motion_store)
print(f"motion library: {motion.merge_tail_bins()} clip-end bins merged, {motion.num_bins} bins", flush=True)
layout = XiLayout(robot, dev)
normalize_xi = layout.normalize
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), layout.dim, robot.sensing)
mdp.ramp.fill_(ramp_at(0, args.dr_ramp_iters))  # --dr_ramp_iters: the simulator's first draws are made at the ramp's start
# a measured robot's log columns: the ramp, the spread of the simulator's kp factors and of the imagined gyro bias
MEASURED = "kp" in layout.slices or bool(robot.sensing.gyro_bias or robot.sensing.imu_tilt)
SENS_STD = torch.zeros((), device=dev)  # written in place by the captured policy update
pipe = ObsPipeline(robot, mdp, layout, dev, PHYSICS_DT)
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
GOAL_DIM = motion.goal_dim(K)
CRITIC_IN = inv_dim(J) + mdp.critic_extra + GOAL_DIM

widths = lambda s: tuple(int(x) for x in s.split(","))  # noqa: E731
networks.RunningNorm.no_sync = True
rotations.ELEMENTWISE = True

AW = J * max(1, robot.n_prev_actions)  # a_prev: the last max(1, n) actions, newest first
# --vel_frac: envs 0 .. N_V - 1 and the last B_V rows of each imagined batch run the velocity interface; the motion
# interface (practice and exam worlds) has the other N_M envs and the first B_M rows.
# --land_frac: envs N_V .. N_C - 1 are the landing robots. Their B_L imagined rows are a batch of their own
# (land_rollout), so the motion and velocity batch holds B_M + B_V rows, sliced [:B_M] / [B_M:] as without --land_frac
N_V, B_V = int(round(args.vel_frac * args.num_envs)), int(round(args.vel_frac * args.pi_batch))
N_L, B_L = int(round(args.land_frac * args.num_envs)), int(round(args.land_frac * args.pi_batch))
N_C = N_V + N_L  # the envs that follow a command come first, the motion envs after them
N_M, B_M = args.num_envs - N_C, args.pi_batch - B_V - B_L
assert (N_V > 0) == (B_V > 0) and N_M > 0 and B_M > 0, \
    "--vel_frac rounds to 0 or 1 on one side, or --vel_frac plus --land_frac leave no motion envs or rows"
assert (N_L > 0) == (B_L > 0), "--land_frac rounds to 0 on one side"
assert not N_L or N_V, "--land_frac needs --vel_frac > 0"
assert not args.freeze_base or (args.init_from and N_L), "--freeze_base needs --init_from and --land_frac > 0"
assert not N_V or not args.pi_ref_skip, "--vel_frac needs --no-pi_ref_skip"
assert args.vel_mirror_weight == 0 or N_V, "--vel_mirror_weight needs --vel_frac > 0"
MIRROR_ON = args.vel_mirror_weight != 0  # off: _imag_step_v and policy_update compute exactly what they did without it
assert args.sym_weight == 0 or N_V, "--sym_weight needs --vel_frac > 0"
SYM_ON = args.sym_weight != 0  # off: no gait memory in the simulator or buffer, the critic's input unchanged
# the velocity rows of an imagined batch, and the velocity envs through env.is_vel, follow no reference: their
# observation shows an upright one at the robot's own heading (mdp.obs). Nor does any row of the landing batch
VEL_ROW = torch.arange(B_M + B_V, device=dev) >= B_M if N_V else None
UP_L = torch.ones(B_L, dtype=torch.bool, device=dev) if N_L else None


def push_a(a, a_prev):
    """a_prev after action a: a, then the older actions of a_prev (a alone when a_prev holds one action)."""
    return torch.cat((a, a_prev[..., : a_prev.shape[-1] - a.shape[-1]]), -1)


wm = WorldModel(J, layout.dim, use_prev_action=robot.n_prev_actions, hidden=widths(args.wm_hidden),
                dt=CONTROL_DT).to(dev)  # fmt: skip
pi = Policy(mdp.obs_dim, GOAL_DIM, J, hist=Hh, enc_hidden=widths(args.enc_hidden), dec_hidden=widths(args.dec_hidden),
            **policy_kwargs(vars(args), robot)).to(dev)
BASE = None  # --freeze_base: the loaded tensors that must not change
if args.init_from:  # before the optimizers and the EMA copy, which starts from the loaded policy
    _ck = torch.load(args.init_from, map_location="cpu", weights_only=False)  # dropped after its last use, below
    assert "wm" in _ck, (f"--init_from {args.init_from} holds no world model (a periodic checkpoint): pass one that "
                         "carries \"wm\", the averaged one of utils/t1_run.sh (logs/<run>/policy/ckpt_*.pt) or the "
                         "run's last (its final ckpt_*.pt or last.pt)")  # fmt: skip
    _miss = pi.load_state_dict(_ck["pi"], strict=False)
    assert not _miss.unexpected_keys and all(k.startswith("enc_land.") for k in _miss.missing_keys), _miss
    wm.load_state_dict(_ck["wm"])
    print(f"--init_from {args.init_from}: policy and world model of iteration {_ck.get('iteration')}", flush=True)
    if args.freeze_base:
        for _net in (pi.enc, pi.dec, pi.enc_vel):
            for _p in _net.parameters():
                _p.requires_grad_(False)
        BASE = {k: v.detach().clone() for k, v in pi.state_dict().items() if not k.startswith("enc_land.")}
        # the checkpoint guard compares bit patterns; the EMA copy's lerp turns -0.0 into +0.0, so a base with a
        # negative zero (or a NaN) would stop the run at its first checkpoint instead of here
        assert not any((((v == 0) & v.signbit()) | v.isnan()).any() for v in BASE.values() if v.is_floating_point()), \
            f"--freeze_base: {args.init_from} holds a NaN or a negative zero"
        _bits = lambda t: t.contiguous().view(torch.int32) if t.is_floating_point() else t  # noqa: E731  bit pattern
N_EX = int(round(args.exam_frac * N_M))  # --exam_frac (a share of the motion envs): envs N - N_EX .. N - 1 run exam episodes
PEN = args.fail_penalty
PH_TYPES = {"exam_frac": float, "front_share": float, "fail_penalty": float, "start_prac_frac": float,
            "reset_aprev_ref": lambda v: bool(int(v)), "lr_decay_iters": int, "lr_end": float, "wstream_frac": float, "critic_epochs": int, "wm_batches": int}
PHASES = sorted(((int(e.split(":")[0]), {k: PH_TYPES[k](v) for k, v in (kv.split("=") for kv in e.split(":")[1].split(",") if kv)})
                for e in args.phase.split(";") if e.strip()), key=lambda p: p[0])  # fmt: skip
PHASE = bool(PHASES)  # --phase; off: code path and random stream unchanged
PH_ITS = {p[0] for p in PHASES}
if PHASE:
    PEN = torch.full((), args.fail_penalty, device=dev)


PH_BASE = {k: getattr(args, k) for k in PH_TYPES}  # flag values (phase_apply changes some args fields later)
RAMP = tuple(float(x) for x in args.share_ramp.split(",")) if args.share_ramp else None  # --share_ramp (lo, hi, n0, step)
RAMP_OV = {}  # --share_ramp: the exam_frac it set last (applied over the --phase entries while exams run)


def phase_at(it: int) -> dict:
    """--phase: the settings in force at iteration it (flag values, then every entry with IT <= it in order)."""
    cur = dict(PH_BASE)
    for p_it, ov in PHASES:
        if p_it <= it:
            cur.update(ov)
    if RAMP_OV and cur["exam_frac"] > 0:
        cur.update(RAMP_OV)
    return cur


if PHASE:
    _all = [phase_at(p[0]) for p in PHASES]
    assert N_EX > 0, "--phase: needs --exam_frac > 0"
    assert all(round(c["exam_frac"] * N_M) <= N_EX for c in _all), "--phase: exam_frac above the --exam_frac capacity"
    assert args.start_prac_frac >= 0
    assert args.fail_penalty > 0 or all(c["fail_penalty"] == 0 for c in _all), "--phase fail_penalty > 0 needs --fail_penalty > 0"
    assert args.front_share > 0 or all(c["front_share"] == 0 for c in _all), "--phase front_share > 0 needs --front_share > 0"
    assert args.reset_aprev_ref or not any(c["reset_aprev_ref"] for c in _all), "--phase reset_aprev_ref needs --reset_aprev_ref"
    assert all(c["wstream_frac"] in (0.0, args.wstream_frac) for c in _all), "--phase wstream_frac: 0 or the --wstream_frac value"
assert N_EX == 0 or args.pi_ema > 0, "--exam_frac: needs --pi_ema"
assert not args.front_share or (N_EX > 0), \
    "--front_share needs --exam_frac"
critic = Critic(CRITIC_IN).to(dev)
if args.init_from and "critic" in _ck:  # a full checkpoint (last.pt) carries the critics too
    critic.load_state_dict(_ck["critic"])
VMAX = (MAX_STEP_REWARD + max(args.contact_reward, 0.0) * 2 * CONTROL_DT) / (1 - args.gamma)  # bound on the critic value (largest per-step reward over 1 - gamma)
VCL = torch.zeros(4, device=dev)  # summed clamped fractions (GAE V, targets, actor bootstrap), count
print(f"critic value bound V_max = {VMAX:.4g}", flush=True)
GRAPH = True
# CUDA graphs: the learning rate is a device tensor (filled every iteration) and Adam keeps its step on the GPU, so the
# captured optimizer steps follow the schedule
adam_kw = lambda: ({"fused": True, "capturable": True, "lr": torch.tensor(args.lr, device=dev)})  # noqa: E731
from src.optim import MuonAdam
opt_wm = MuonAdam(wm, [wm.net], lr=args.lr, betas=(0.9, 0.999), fused=True, capturable=GRAPH)
# --freeze_base: the policy optimizer holds enc_land alone (MuonAdam's Adam group leaves out the parameters without
# requires_grad, and the frozen MLPs are not named, so their hidden matrices are not in its Muon group either)
PI_MLPS = ([pi.enc_land] if args.freeze_base
           else [pi.enc, pi.dec] + ([pi.enc_vel] if N_V else []) + ([pi.enc_land] if N_L else []))  # fmt: skip
opt_pi = MuonAdam(pi, PI_MLPS, lr=args.lr, betas=(args.pi_beta1, 0.95), fused=True, capturable=GRAPH)
NONFINITE = torch.zeros((), dtype=torch.long, device=dev)  # skipped non-finite policy updates
pi_avg = torch.optim.swa_utils.AveragedModel(pi, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.pi_ema),
                                             use_buffers=True) if args.pi_ema > 0 else None  # fmt: skip
opt_v = torch.optim.Adam(critic.parameters(), **adam_kw())
OPTS = [(opt_wm, 1.0), (opt_pi, args.pi_lr_mult), (opt_v, 1.0)]  # optimizers and their learning-rate multipliers

start_it, wall_offset, sim_steps = 0, 0.0, 0


# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((J,), 3e-3), torch.full((J,), 5e-2))).to(dev)  # fmt: skip


def _wm_rollout(s, a, a_prev, xi):
    S = unflatten_state(s[:, 0])
    preds = []
    for j in range(args.wm_window):
        S, c = wm.step_c(S, a[:, j], a_prev[:, j], normalize_xi(xi[:, j]))
        preds.append(S)
    S_hat = tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)
    return S_hat


_feet = robot.feet
FEET = [robot.tracked_bodies.index(b) for b in _feet]
mdp.feet, mdp.contact_z = FEET, robot.contact_z
motion.set_ref_contact(FEET)
if N_V:
    mdp.enable_velocity(robot, VelocityCfg(w_stance=args.stance_weight, w_ankle=args.ankle_weight, track_sharp=args.track_sharp,
                                           ankle_target=args.ankle_target, ankle_roll=args.ankle_roll_weight,
                                           w_rest=args.rest_weight, w_limit=args.vel_limit_weight,
                                           w_lean=args.lean_weight, w_tilt_rate=args.tilt_rate_weight,
                                           w_step=args.step_weight, w_sym=args.sym_weight))
    if MIRROR_ON:
        assert_mirrorable_box(mdp.cmd_lo, mdp.cmd_hi)
    motion.set_ref_cmd()
    critic_v = Critic(inv_dim(J) + layout.dim + 3 + AW + (GAIT_FEAT if SYM_ON else 0)).to(dev)
    if args.init_from and "critic_v" in _ck:
        critic_v.load_state_dict(_ck["critic_v"])
    opt_vv = torch.optim.Adam(critic_v.parameters(), **adam_kw())
    OPTS.append((opt_vv, 1.0))
    VMAX_V = mdp.vel.max_step_reward / (1 - args.gamma)
if N_L:
    mdp.enable_landing(robot, LandingCfg(label_prob=args.land_label_prob, bonus=args.land_bonus, w_drag=args.land_drag,
                                         arrive_narrow=args.land_arrive, w_progress=args.land_progress,
                                         derived=args.land_derived))  # fmt: skip
    motion.set_land_labels(FEET)
    print(f"landing labels: {len(motion.land_steps)} steps, {len(motion.land_label_reset)} / "
          f"{len(motion.land_free_reset)} reset frames", flush=True)  # fmt: skip
    critic_l = Critic(inv_dim(J) + layout.dim + AW + landing.CRITIC_DIM).to(dev)
    if args.init_from and "critic_land" in _ck:
        critic_l.load_state_dict(_ck["critic_land"])
    opt_vl = torch.optim.Adam(critic_l.parameters(), **adam_kw())
    OPTS.append((opt_vl, 1.0))
    # the largest HOLD return plus one deadline lump
    VMAX_L = mdp.land.max_step_reward / (1 - args.gamma) + mdp.land.bonus_reward
if args.init_from:
    del _ck  # its last use was the critics above: the loaded checkpoint does not stay in memory for the run


def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.inc_norm.std(), ERR_FLOOR)
    S1 = unflatten_state(s1)
    e = (wm.state_error(S_hat, S1) / std) ** 2  # (B, k, D)
    e = e.mean(-1)
    loss = e
    return (loss * mask).sum() / s1.shape[0], e[:, -1]


def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g, sens=None):
    """``sens``: the rollout's sensor errors (mdp.draw_sensor), held over its steps; None on a robot without them."""
    a = pi(o_hist, g)
    S1 = wm.step(S, a, a_prev, xi_n)
    f1 = f + 1
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1, sensor=sens))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, push_a(a, a_prev), o_hist, f1, motion.goal(f1, K)


def _imag_step_v(S, o_hist, a_prev, f, xi, xi_n, g, cmd_held, cmd_ref, sens=None):
    """``_imag_step`` of a batch whose last B_V rows run the velocity interface (--vel_frac): their action comes from
    the velocity encoder through the decoder, with a stop-gradient on the decoder's weights. Also returns their command
    and, where it is read from the clip, the squared distance of the velocity latent to the motion latent of the same
    step (the alignment loss)."""
    ov = o_hist[B_M:]
    cmd = torch.where(cmd_ref.unsqueeze(-1), motion.ref_cmd[f[B_M:]], cmd_held)
    cmd_n = mdp.normalize_cmd(cmd)
    a = torch.cat((pi(o_hist[:B_M], g[:B_M]), pi.decode_frozen(pi.encode_vel(cmd_n, ov), ov)))
    # on the detached history: the pull moves the velocity encoder, not the actions that led to the history
    z_vel = pi.encode_vel(cmd_n, ov.detach())
    with torch.no_grad():
        z_ref = pi.encode(g[B_M:])
    pull = cmd_ref
    if mdp.vel.w_stance:  # --stance_weight: where the clip itself stands, the stance term sets the pose, not the clip
        pull = cmd_ref & ~mdp.standing(cmd)
    align = ((z_vel - z_ref) ** 2).mean(-1) * pull
    S1 = wm.step(S, a, a_prev, xi_n)
    f1 = f + 1
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1, VEL_ROW, sensor=sens))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    if MIRROR_ON:  # --vel_mirror_weight: the mirror error of each velocity row, on the history the alignment uses
        sym = policy_mirror_error(pi, mirror, cmd_n, ov.detach())  # (`mirror` is read at call time, defined below)
        return S1, push_a(a, a_prev), o_hist, f1, motion.goal(f1, K), cmd, align, sym
    return S1, push_a(a, a_prev), o_hist, f1, motion.goal(f1, K), cmd, align


def _vel_reward_term(S, cmd, a, a_prev, xi, foot_z, foot_v, foot_v0, foot_v1, jerk_on, foot_xy, foot_z0):
    r, terms = mdp.velocity_reward(S, cmd, a, a_prev, xi, foot_z, foot_v, foot_v0, foot_v1, jerk_on, terms=True,
                                   foot_xy=foot_xy, foot_z0=foot_z0)
    return r, mdp.soft_fall_termination(S), terms


def vel_rewards(S_prev, S0, Ss, As, Aps, Cs, xi, ep0, gait0=None, gate=None):
    """Velocity reward and soft fall probability (H, B_V) of the velocity rows' imagined transitions, the batch
    means of the reward's terms, and with --sym_weight the gait memories of states 0 .. H (else None). S_prev: the
    velocity rows' state one step before the start (the jerk term needs the feet's velocities two steps back); ep0:
    their starts' episode steps; gait0: their starts' gait memory (the buffer's), gate: mdp.gait_gate of their
    commands."""
    v = lambda x: x[B_M:]  # noqa: E731
    seq = [S_prev, tuple(v(x) for x in S0)] + [tuple(v(x) for x in St) for St in Ss]  # states -1, 0, 1 .. H
    St = tuple(torch.stack([q[i] for q in seq]) for i in range(6))  # (H + 2, B_V, ...)
    foot_z, foot_v, foot_xy = mdp.foot_kin(tuple(x.flatten(0, 1) for x in St))
    foot_z, foot_v, foot_xy = foot_z.view(HZ + 2, B_V, -1), foot_v.view(HZ + 2, B_V, -1, 3), foot_xy.view(HZ + 2, B_V, -1, 2)
    jerk_on = ep0.unsqueeze(0) + torch.arange(HZ, device=dev).unsqueeze(-1) >= 2  # as the simulator: not on an episode's first two steps
    fl = lambda x: x.flatten(0, 1)  # noqa: E731
    r, d, terms = vel_reward_term(
        tuple(fl(x[2:]) for x in St), fl(torch.stack(Cs)), fl(torch.stack([v(a) for a in As])),
        fl(torch.stack([v(a) for a in Aps])), fl(v(xi).unsqueeze(0).expand(HZ, -1, -1)), fl(foot_z[2:]), fl(foot_v[2:]),
        fl(foot_v[1:-1]), fl(foot_v[:-2]), fl(jerk_on), fl(foot_xy[2:]), fl(foot_z[1:-1]))  # fmt: skip
    r, terms = r.view(HZ, B_V), {k: x.detach().mean() for k, x in terms.items()}
    if not SYM_ON:
        return r, d.view(HZ, B_V), terms, None
    # the step-symmetry penalty: the memory carried over the rollout from the start's (the simulator's own steps)
    settled = ep0.unsqueeze(0) + torch.arange(HZ, device=dev).unsqueeze(-1) >= mdp.vel.sym_settle  # as jerk_on
    pen, scored, mems = mdp.gait_rollout(gait0, foot_v[2:], foot_xy[2:], St[1][2:], gate, settled)
    r = r + mdp.vel.w_sym * pen * (mdp.vel.scale * CONTROL_DT)
    terms.update(sym=pen.detach().sum() / scored.sum().clamp(min=1), sym_n=scored.sum() / B_V)
    return r, d.view(HZ, B_V), terms, mems


def _land_imag_step(S, o_hist, a_prev, f, xi, xi_n, C, lab, pose, fv0, fv1, jerk_on):
    """One imagined step of the landing rows: the clip's command where a label row's frame starts one, the action
    through enc_land and the decoder (weights frozen), the world model, the reward and the advanced command state."""
    C = landing.land_issue(C, pose, lab & (motion.land_kind[f] == 2), motion.land_swing[f], motion.land_c[f],
                           motion.land_n[f])  # fmt: skip
    cf = mdp.land_features(C, o_hist[:, -1])
    a = pi.decode_frozen(pi.encode_land(cf, o_hist), o_hist)
    z = pi.encode_land(cf.detach(), o_hist.detach())  # the pull moves the encoder, not the actions behind the history
    with torch.no_grad():
        z_ref = pi.encode(motion.goal(f, K))
    align = ((z - z_ref) ** 2).mean(-1) * lab
    S1 = wm.step(S, a, a_prev, xi_n)
    r, C1, pose1, fv, terms = mdp.land_step(S1, C, pose, a, a_prev, xi, fv0, fv1, jerk_on)
    f1 = f + lab.long()  # a free row follows no clip
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1, UP_L))
    return (S1, push_a(a, a_prev), torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1), f1, C1, pose1, fv, r,
            mdp.soft_fall_termination(S1), align, terms)  # fmt: skip


def land_rollout():
    """The landing rows' imagined rollout from B_L starts of the landing robots: their objective J (B_L,), survival,
    the pull, the critic's inputs (H + 1, B_L, .), rewards and fall probabilities (H, B_L), and for the log the mean
    imagined reward over its ceiling and the sums (3,) of the landing score, the landing error and the deadlines."""
    s, h = sample_valid_starts(B_L, land=True)
    C, lab, ep0 = s["land"], s["lab"], s["ep_step"].long()
    S, xi, f, a_prev = unflatten_state(s["s"]), s["xi"], s["f"], s["a_prev"]
    xi_n = normalize_xi(xi)
    with torch.no_grad():
        o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"], UP_L.unsqueeze(-1))
        sole, fv0 = mdp.sole_kin(S)
        fv1 = mdp.sole_kin(unflatten_state(h["s"][:, -2]))[1]  # the feet one step before the start, for the jerk term
    o_hist, pose = mdp.add_noise(o_hist), sole["pose"]
    Ss, Cs, Aps = [S], [C], [a_prev]  # states 0 .. H with their command state and pending actions, for the critic
    R, D, A, G = [], [], [], []
    for t in range(HZ):
        S, a_prev, o_hist, f, C, pose, fv, r, d, al, tm = land_imag_step(
            S, o_hist, a_prev, f, xi, xi_n, C, lab, pose, fv0, fv1, ep0 + t >= 2)  # fmt: skip
        fv0, fv1 = fv, fv0
        Ss.append(S), Cs.append(C), Aps.append(a_prev)
        with torch.no_grad():
            dl = tm["deadline"].float()
            G.append(torch.stack((tm["G"] * dl, tm["e"] * dl, dl)))
        R.append(r), D.append(d), A.append(al)
    r, d = torch.stack(R), torch.stack(D)
    v_boot = critic_l(mdp.land_critic_features(S, C, xi_n, a_prev)).clamp(max=VMAX_L)
    J_l, c_t, c = actor_objective(r, d, v_boot, args.gamma)
    with torch.no_grad():  # the critic's inputs in one call on every state: one pass of the kinematics, not H + 1
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  ((H + 1) * B_L, ...)
        X = mdp.land_critic_features(tuple(flat([St[i] for St in Ss]) for i in range(6)), flat(Cs),
                                     xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1), flat(Aps))  # fmt: skip
    g = torch.stack(G).sum((0, 2))  # summed landing score, landing error and number of deadlines
    align = torch.stack(A).sum() / (HZ * lab.sum().clamp(min=1))
    imag_r = ((c_t * r).detach().sum(0) / HZ / mdp.land.max_step_reward).mean()
    return J_l, c, align, X.view(HZ + 1, B_L, -1), r.detach(), d.detach(), imag_r, g


def _reward_term(S, f, a, a_prev):
    return mdp.reward(S, f, a, a_prev, w_contact=args.contact_reward), mdp.soft_termination(S, f)


def _gae_targets(V, rews, dones):
    """GAE(lambda) returns over the imagined transitions; V is (H+1, B), bootstrapped from V[-1]."""
    R = V[-1]
    targets = []
    for t in range(HZ - 1, -1, -1):
        R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
        targets.append(R)
    return torch.stack(targets[::-1])


def _critic_loss(X, Y):
    return ((critic(X) - Y) ** 2).mean()


def _wm_norm_data(S0, S1, a, a_prev, xi):
    return wm.inputs(S0, a, a_prev, normalize_xi(xi)), wm.targets(S0, S1), WorldModel.increments(S0, S1)


def _critic_input(S, f, xi_n, g):
    return torch.cat((mdp.critic_features(S, f, xi_n), g), -1)


wm_rollout = torch.compile(_wm_rollout)
wm_err = torch.compile(_wm_err)
imag_step = torch.compile(_imag_step)
reward_term = torch.compile(_reward_term)
imag_step_v = torch.compile(_imag_step_v)
vel_reward_term = torch.compile(_vel_reward_term)
land_imag_step = torch.compile(_land_imag_step)
critic_input_c = torch.compile(_critic_input)
wm_norm_data_c = torch.compile(_wm_norm_data)
gae_targets_c = torch.compile(_gae_targets)
critic_loss_c = torch.compile(_critic_loss)


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return args.lr * args.lr_end
    t0 = 0
    span = 0.97 if args.lr_end == 0.03 else 1.0 - args.lr_end  # literal kept: default bitwise unchanged
    return args.lr * (args.lr_end + span * 0.5 * (1 + math.cos(math.pi * (it - t0) / (args.lr_decay_iters - t0))))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


# ------------------------------------------------------------------------------------------------
mirror = Mirror(robot, layout.slices, dev)


def _mirror_windows(w: dict, n: int) -> dict:
    """Replace the first n windows by their mirror images (states, actions, sim context)."""
    out = dict(w)
    ms = lambda x: flatten_state(mirror.state(unflatten_state(x)))  # noqa: E731
    for k, fn in (("s", ms), ("s1", ms), ("a", mirror.joints), ("a_prev", mirror.joints), ("xi", mirror.xi)):
        out[k] = torch.cat((fn(w[k][:n]), w[k][n:]))
    return out


def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    w = _mirror_windows(w, int(args.wm_batch * args.wm_mirror))
    S_hat = wm_rollout(w["s"], w["a"], w["a_prev"], w["xi"])
    loss, e_last = wm_err(S_hat, w["s1"], w["mask"])
    opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    opt_wm.step()
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1)


def sample_valid_starts(B: int, mix: bool = True, vel: bool = False, land: bool = False):
    """B starts with HZ frames left in their clip: one draw of 2B candidates, the valid ones first (a stable sort), the
    first B kept (so invalid ones only if fewer than B are valid). ``vel``: from the velocity envs (with their command,
    its source and their episode step), else from the motion envs. ``land``: from the landing robots (with their
    command state, whether they follow a clip's labels and their episode step); a label row's start also needs HZ
    labelled frames ahead (imagination must not run off the labels, where its episode ends), and a free row's start,
    which follows no clip, is always valid. The deployment stream is mixed into the motion starts alone."""
    if land:
        s, h = buffer.sample_land_starts(2 * B, Hh, N_V, N_L)
    else:
        s, h = buffer.sample_vel_starts(2 * B, Hh, N_V) if vel else buffer.sample_starts(2 * B, Hh)
    f = s["f"]
    ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
    if land:
        ok = torch.where(s["lab"], ok & (motion.land_ahead[f] >= HZ), torch.ones_like(ok))
    idx = torch.sort((~ok).to(torch.uint8), stable=True).indices[:B]
    s, h = {k: v[idx] for k, v in s.items()}, {k: v[idx] for k, v in h.items()}
    return dstart_mix(s, h) if mix and not vel and not land else (s, h)


WS = args.wstream_frac > 0  # --wstream_frac: deployment stream
assert not WS or (args.exam_pre_every > 0), "--wstream_frac: needs --exam_pre_every (nominal context)"
WS_ON = torch.zeros((), dtype=torch.bool, device=dev)  # device-side switch of the mixing
WS_ACTIVE, WST = [False], {}  # host-side switch of the advance; the stream's fixed tensors


def ws_mix(s, h):
    """--wstream_frac: the last int(B * frac) starts are replaced by uniform draws of stream worlds (worlds with fewer than H
    frames left in their clip keep the buffer draw)."""
    k = int(len(s["f"]) * args.wstream_frac)
    j = (torch.rand(k, device=dev) * args.wstream_worlds).long().clamp(max=args.wstream_worlds - 1)
    f = WST["s"]["f"][j]
    ok = WS_ON & (f + HZ <= motion.clip_end[motion.clip_id[f]])
    for d, b in ((s, WST["s"]), (h, WST["h"])):
        for key in d:
            v = b[key][j]
            d[key][-k:] = torch.where(ok.view(-1, *([1] * (v.dim() - 1))), v, d[key][-k:])
    return s, h


def dstart_mix(s, h):
    """Imagination starts: the deployment stream replaces a random subset of the buffer draws."""
    if WS:
        s, h = ws_mix(s, h)
    return s, h


@torch.no_grad()
def ema_update():
    """The EMA step of ``AveragedModel.update_parameters`` (same lerp) without its host-side first-call test; the
    host-side counter n_averaged is advanced by the training loop."""
    avg = list(pi_avg.module.parameters()) + list(pi_avg.module.buffers())
    live = [t.detach() for t in list(pi.parameters()) + list(pi.buffers())]
    assert all(t.is_floating_point() for t in avg)
    torch._foreach_lerp_(avg, live, 1 - args.pi_ema)


class Graphed:
    """Run ``fn`` eagerly until ``ready()``; then eagerly on a side stream for ``warmup`` calls, capture it as a CUDA
    graph on the next call and replay it from then on. Each call is one update; outputs are cloned out of the graph pool."""

    def __init__(self, fn, ready, warmup: int = 3):
        self.fn, self.ready, self.warmup, self.calls, self.graph, self.out = fn, ready, warmup, 0, None, None

    def __call__(self):
        if self.graph is None and not self.ready():
            return self.fn()
        self.calls += 1
        if self.calls <= self.warmup:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                out = self.fn()
            torch.cuda.current_stream().wait_stream(side)
            return out
        if self.graph is None:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # return the eager phase's cached blocks before the graph takes its own pool
            t0 = time.time()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self.fn()
            torch.cuda.synchronize()
            print(f"cuda_graph: captured {self.fn.__name__} in {time.time() - t0:.1f} s, "
                  f"{torch.cuda.memory_reserved() / 2**30:.1f} GB reserved", flush=True)  # fmt: skip
        self.graph.replay()
        return tuple(x.clone() for x in self.out)


def policy_update():
    s, h = sample_valid_starts(B_M)
    if N_V:  # the velocity rows, appended: their own envs' starts
        sv, hv = sample_valid_starts(B_V, vel=True)
        cmd_held, cmd_ref, ep0 = sv.pop("cmd"), sv.pop("cmd_ref"), sv.pop("ep_step").long()
        gait0 = sv.pop("gait") if SYM_ON else None  # --sym_weight: the simulator's gait memory at each start
        S_prev = unflatten_state(hv["s"][:, -2])
        s, h = {k: torch.cat((x, sv[k])) for k, x in s.items()}, {k: torch.cat((x, hv[k])) for k, x in h.items()}
    S = unflatten_state(s["s"])
    xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
    xi_n = normalize_xi(xi)
    # a measured robot's sensor errors: the policy learns only from these rollouts, so they carry them too. One draw per
    # rollout at the current ramp, held over its rebuilt history and its horizon (None on a robot without them)
    sens = mdp.draw_sensor((S[0].shape[0],))
    assert not N_L or sens is None, "the landing rollout passes no sensor errors"
    if sens is not None:
        SENS_STD.copy_(sens[0].std())
    with torch.no_grad():
        o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"], VEL_ROW.unsqueeze(-1) if N_V else None,
                         sensor=None if sens is None else tuple(x.unsqueeze(1) for x in sens))  # fmt: skip
    o_hist = mdp.add_noise(o_hist)
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    if N_V:
        set_requires_grad(critic_v, False)
    if N_L:
        set_requires_grad(critic_l, False)
    g0 = motion.goal(f, K)
    g = g0
    S0 = S
    Ss, As, Aps, Fs, Gs = [], [], [], [], []
    Cs, Als, Ms = [], [], []  # velocity rows: the command, the alignment error and the mirror error of each step
    for t in range(HZ):
        Aps.append(a_prev)
        if N_V:
            S, a_prev, o_hist, f, g, cmd, al, *sym = imag_step_v(S, o_hist, a_prev, f, xi, xi_n, g, cmd_held, cmd_ref, sens)
            Cs.append(cmd), Als.append(al), Ms.extend(sym)  # sym: [] unless --vel_mirror_weight
        else:
            S, a_prev, o_hist, f, g = imag_step(S, o_hist, a_prev, f, xi, xi_n, g, sens)
        Ss.append(S), As.append(a_prev[..., : robot.num_joints]), Fs.append(f), Gs.append(g)  # the action just taken
    flat = lambda xs: torch.stack(xs)[:, :B_M].flatten(0, 1)  # noqa: E731  (H*B_M, ...): the motion rows
    S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
    f_all = flat(Fs)
    r, d = reward_term(S_all, f_all, flat(As), flat(Aps))
    r, d = r.view(HZ, B_M), d.view(HZ, B_M)
    r_log = r  # the tracking reward alone (log column imag_r)
    if args.fail_penalty > 0:
        r = r - PEN * d
    if N_V:  # no fail penalty on the velocity rows: their reward has the survival term
        gate = mdp.gait_gate(Cs[0], cmd_ref) if SYM_ON else None  # a row's command is held over its rollout
        r_v, d_v, vel_terms, mems = vel_rewards(S_prev, S0, Ss, As, Aps, Cs, xi, ep0, gait0, gate)
        r, r_log, d = torch.cat((r, r_v), 1), torch.cat((r_log, r_v), 1), torch.cat((d, d_v), 1)
    rews, dones = (r_log if args.pen_horizon_only else r).detach(), d.detach()

    def critic_fit(crit, opt, loss_fn, X, rews, dones, vmax, count):
        """Fit ``crit`` to the GAE(lambda) targets over its rows' imagined transitions: X (H + 1, rows, features),
        rews and dones (H, rows). ``count``: add the clamped shares to VCL (the motion critic's log)."""
        v_loss = torch.zeros(())
        crit.norm.update(X)
        with torch.no_grad():
            Vx = crit(X)
            if count:
                VCL[0].add_((Vx > vmax).float().mean())
            Vx = Vx.clamp(max=vmax)
            targets = (gae_targets_c)(Vx, rews, dones)  # (H, rows)
            if count:
                VCL[1].add_((targets > vmax).float().mean())
                VCL[3].add_(1.0)
            targets = targets.clamp(max=vmax)
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(args.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = loss_fn(Xf[idx], Yf[idx])
                opt.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(crit.parameters(), 1.0)
                opt.step()
        return v_loss

    v_boot = critic(critic_input_c(tuple(x[:B_M] for x in S), f[:B_M], xi_n[:B_M], g[:B_M]))
    VCL[2].add_((v_boot.detach() > VMAX).float().mean())
    v_boot = v_boot.clamp(max=VMAX)
    if N_V:  # the velocity rows bootstrap from their own critic
        cmd_H = torch.where(cmd_ref.unsqueeze(-1), motion.ref_cmd[f[B_M:]], cmd_held)
        x_v = mdp.velocity_critic_features(tuple(x[B_M:] for x in S), cmd_H, xi_n[B_M:], a_prev[B_M:],
                                           gait_f=mdp.gait_features(mems[-1], gate) if SYM_ON else None)  # fmt: skip
        v_boot = torch.cat((v_boot, critic_v(x_v).clamp(max=VMAX_V)))
    # actor_objective takes the survival products as a loop, for the CUDA graph (cumprod's backward reads the host)
    J, c_t, c = actor_objective(r, d, v_boot, args.gamma)
    r_sum = (c_t * r_log).detach().sum(0)
    if N_L:  # the landing rows: an imagined rollout of their own, then one objective over every row
        J_l, c_l, align_l, X_l, r_l, d_l, imag_r_l, g_l = land_rollout()
        loss = -torch.cat((J, J_l)).mean() + args.land_align_weight * align_l
    else:
        loss = -(J).mean()
    if N_V:
        align = torch.stack(Als).sum() / (HZ * cmd_ref.sum().clamp(min=1))
        loss = loss + args.align_weight * align
        if MIRROR_ON:
            sym = torch.stack(Ms).mean()
            loss = loss + args.vel_mirror_weight * sym
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    nn.utils.clip_grad_value_(pi.parameters(), args.pi_clip_value)
    finite = torch.isfinite(gn)
    NONFINITE.add_((~finite).long())
    for p in pi.parameters():
        if p.grad is not None:
            p.grad.copy_(torch.where(finite, p.grad, 0.0))
    opt_pi.step(ok=finite)
    if pi_avg is not None:
        if int(pi_avg.n_averaged) > 0:
            ema_update()
        else:
            pi_avg.update_parameters(pi)
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    # ---------------- critics: each interface's own, on its own rows ----------------
    with torch.no_grad():
        mo = lambda xs: torch.stack(xs)[:, :B_M]  # noqa: E731
        S_c = tuple(torch.cat((S0[i][:B_M].unsqueeze(0), mo([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
        f_c = torch.cat((s["f"][:B_M].unsqueeze(0), mo(Fs)), 0).flatten()
        g_c = torch.cat((g0[:B_M].unsqueeze(0), mo(Gs)), 0).flatten(0, 1)
        xi_c = xi_n[:B_M].unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
        X = critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B_M, -1)
    v_loss = critic_fit(critic, opt_v, critic_loss_c, X, rews[:, :B_M], dones[:, :B_M], VMAX, True)
    out = (loss.detach(), v_loss.detach(), (r_sum[:B_M] / HZ / 0.1).mean(), gn.detach(), c[:B_M].mean().detach())
    if N_V:
        set_requires_grad(critic_v, True)
        with torch.no_grad():
            ve = lambda xs: torch.stack(xs)[:, B_M:]  # noqa: E731
            S_c = tuple(torch.cat((S0[i][B_M:].unsqueeze(0), ve([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            cmd_c = torch.cat((torch.stack(Cs), cmd_H.unsqueeze(0)), 0).flatten(0, 1)  # the command at states 0 .. H
            ap_c = torch.cat((ve(Aps), a_prev[B_M:].unsqueeze(0)), 0).flatten(0, 1)
            xi_c = xi_n[B_M:].unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            gf = mdp.gait_features(mems, gate).flatten(0, 1) if SYM_ON else None  # the memory at states 0 .. H
            X = mdp.velocity_critic_features(S_c, cmd_c, xi_c, ap_c, gait_f=gf).view(HZ + 1, B_V, -1)
        vv_loss = critic_fit(critic_v, opt_vv, lambda x_, y_: ((critic_v(x_) - y_) ** 2).mean(), X, rews[:, B_M:], dones[:, B_M:],
                             VMAX_V, False)  # fmt: skip
        out += (vv_loss.detach(), (r_sum[B_M:] / HZ / mdp.vel.max_step_reward).mean(), c[B_M:].mean().detach(), align.detach(),
                *(vel_terms[k] for k in VEL_TERMS))  # fmt: skip
        if MIRROR_ON:
            out += (sym.detach(),)
    if N_L:
        set_requires_grad(critic_l, True)
        vl_loss = critic_fit(critic_l, opt_vl, lambda x_, y_: ((critic_l(x_) - y_) ** 2).mean(), X_l, r_l, d_l, VMAX_L,
                             False)  # fmt: skip
        nan = torch.full((), float("nan"), device=dev)  # the landing score and error of an update without a deadline
        out += (vl_loss.detach(), imag_r_l, c_l.mean().detach(), align_l.detach(),
                torch.where(g_l[2] > 0, g_l[0] / g_l[2].clamp(min=1), nan),
                torch.where(g_l[2] > 0, g_l[1] / g_l[2].clamp(min=1), nan))  # fmt: skip
    return out


VEL_TERMS = ("e_lin", "e_yaw", "rate", "torque", "slide", "jerk", "stance", "ankle", "rest", "limit", "lean",
             "tilt_rate", "step") + (("sym", "sym_n") if SYM_ON else ())  # mdp.velocity_reward's logged terms (+ vel_rewards')
# log columns of the velocity interface: policy_update's extra outputs, then the simulator's (collect)
VEL_COLS = (["vel_v_loss", "vel_imag_rew", "vel_imag_survival", "align"] + [f"vel_imag_{k}" for k in VEL_TERMS]
            + (["vel_mirror"] if MIRROR_ON else []) + ["vel_sim_rew", "vel_sim_e_lin", "vel_sim_slide", "vel_sim_jerk", "vel_term_frac"]
            + (["vel_sim_sym"] if SYM_ON else []))  # fmt: skip
# and of the landing interface: policy_update's six extra outputs, then the simulator's. G and err are the landing
# score and the swing sole's distance from its target [m] on the deadline steps
LAND_COLS = ["land_v_loss", "land_imag_rew", "land_imag_survival", "land_align", "land_imag_G", "land_imag_err",
             "land_sim_rew", "land_sim_G", "land_sim_err", "land_term_frac"]  # fmt: skip


# ------------------------------------------------------------------------------------------------
sim_cfg = make_sim(robot)
mdp.reward, mdp.termination, mdp.obs = torch.compile(mdp.reward), torch.compile(mdp.termination), torch.compile(mdp.obs)
motion.goal = torch.compile(motion.goal)
env = TrackingSim(sim_cfg, args.num_envs, robot, motion, mdp, dev, gamma=args.gamma, fail_reset=args.fail_reset, bin_fix=True,
                  n_vel=N_V, n_land=N_L)  # fmt: skip
env.aprev_ref = args.reset_aprev_ref
if PHASE:
    env.aprev_gate = torch.ones((), dtype=torch.bool, device=dev)
if N_EX:
    EX_CLIPS = torch.arange(len(motion.names), device=dev)
    env.set_exam(N_EX, EX_CLIPS)
    EX_MASK = torch.zeros(len(motion.names), dtype=torch.bool, device=dev)
    EX_MASK[EX_CLIPS] = True
    EX_D = torch.full((len(motion.names),), float("nan"), device=dev)  # last exam's max danger ratio per clip
    EX_T = torch.zeros(len(motion.names), device=dev)  # iteration of the last exam
    EX_K = torch.zeros(len(motion.names), device=dev)  # current failure streak
    EX_RUN = torch.zeros(N_EX, device=dev)  # running max danger of each exam world's episode
    EX_P = torch.zeros(len(motion.names), device=dev)  # exams of each clip in flight (diminishing returns per clip)
    EX_SRC = torch.zeros(3, device=dev)  # exam episodes started by source (priority, training failure, uniform)
    IT_T = torch.full((), float(start_it), device=dev)  # the current iteration (filled at the top of each)
    EX_PD = torch.full((len(motion.names),), float("nan"), device=dev)  # --exam_pre_every: last WM pre-exam score
    EX_PT = torch.zeros(len(motion.names), device=dev)  # iteration of the last pre-exam
    EX_FF = torch.full((len(motion.names),), float("nan"), device=dev)  # last exam: failure frame, -1 = passed
    EX_PF = torch.full((len(motion.names),), float("nan"), device=dev)  # last pre-exam: imagined failure frame, -1 = none
    EX_KF = torch.full((len(motion.names),), float("nan"), device=dev)  # last exam: max-danger (or failure) frame
    EX_PK = torch.full((len(motion.names),), float("nan"), device=dev)  # last pre-exam: max-danger frame
    EX_RUNF = torch.zeros(N_EX, dtype=torch.long, device=dev)  # frame of the running max danger of each exam world
    EX_WPI = torch.full((N_EX,), 1.0 / len(motion.names), device=dev)  # inclusion probability of each exam world's clip
    EX_IW = torch.full((len(motion.names),), float("nan"), device=dev)  # last exam: 1 / its inclusion probability


    EX_PIC = torch.full((len(motion.names) + 1,), 1.0 / len(motion.names), device=dev)  # per clip, the inclusion probability of its last draw (+ a dump slot)
    if args.exam_mix:
        _p, _q = (float(x) for x in args.exam_mix.split(","))
        EX_MIXW = torch.tensor([_p, _q, max(0.0, 1.0 - _p - _q)], device=dev)

        def exam_fused():
            """The margin estimate per clip: D, its sd, and whether the clip has any measurement."""
            now = IT_T
            sd = torch.sqrt(args.exam_s0**2 + args.exam_v * (now - EX_T).clamp(min=0))
            if args.exam_pre_every:  # two measurements of the margin, each drifting with its age
                sp = torch.sqrt(args.exam_pre_s**2 + args.exam_v * (now - EX_PT).clamp(min=0))
                pr = torch.where(torch.isfinite(EX_D), sd**-2, 0.0)
                pp = torch.where(torch.isfinite(EX_PD), sp**-2, 0.0)
                prec = (pr + pp).clamp(min=1e-12)
                D = (pr * torch.nan_to_num(EX_D) + pp * torch.nan_to_num(EX_PD)) / prec
                sd = prec.rsqrt()
                seen = ((pr + pp) > 0) & EX_MASK
            else:
                D = torch.nan_to_num(EX_D)
                seen = torch.isfinite(EX_D) & EX_MASK
            return D, sd, seen

        def exam_weights():
            """--exam_mix: priority weights wp and training-failure weights prior over all clips (0: not drawable)."""
            D, sd, seen = exam_fused()
            z = (1 - D) / sd
            # one exam per clip in flight (its result, which updates the table, is not back yet); clips more than 3 sd
            # from the boundary are left to the other sources
            wp = torch.where(seen & (z <= 3) & (EX_P <= 0), torch.exp(-0.5 * z**2) / sd, 0.0)
            prior = torch.zeros(len(motion.names), device=dev).scatter_reduce_(0, motion.bin_clip, env.bin_fail, "amax")
            prior = prior * EX_MASK
            return wp, prior

        def _gumbel_top(w, k):
            """k draws without replacement with probability proportional to w (Gumbel top-k, the same distribution as
            torch.multinomial without replacement, no host sync): clip ids, and whether each draw exists (w > 0)."""
            u = torch.rand_like(w).clamp(min=1e-20)
            key = torch.where(w > 0, torch.log(w.clamp(min=1e-30)) - torch.log(-torch.log(u)), float("-inf"))
            v, i = torch.topk(key, k)
            return i, torch.isfinite(v)


        def exam_sample_masked(dn):
            """--exam_mix: exam clips for env.reset_masked (capturable) from the priority / training-failure / uniform
            mixture: a clip for every exam world; only the worlds in dn (being reset) count. The priority draws (without
            replacement) go to the reset worlds of that source in order; draws that do not exist fall to the
            training-failure source, then to uniform."""
            n = N_EX
            src = torch.multinomial(EX_MIXW, n, replacement=True)
            c = EX_CLIPS[(torch.rand(n, device=dev) * len(EX_CLIPS)).long().clamp(max=len(EX_CLIPS) - 1)]
            wp, prior = exam_weights()
            wp_s, pr_s = wp, prior
            i0, ok0 = _gumbel_top(wp, n)
            m0 = dn & (src == 0)
            r0 = (torch.cumsum(m0.long(), 0) - 1).clamp(min=0)
            use0 = m0 & ok0[r0]
            c = torch.where(use0, i0[r0], c)
            src = torch.where(m0 & ~use0, 1, src)
            prior = prior * (EX_P <= 0)
            prior = prior.scatter(0, c, torch.where(use0, 0.0, prior[c]))  # a clip drawn by priority is not drawn again
            i1, ok1 = _gumbel_top(prior, n)
            m1 = dn & (src == 1)
            r1 = (torch.cumsum(m1.long(), 0) - 1).clamp(min=0)
            use1 = m1 & ok1[r1]
            c = torch.where(use1, i1[r1], c)
            src = torch.where(m1 & ~use1, 2, src)
            EX_SRC.add_((torch.where(dn, src, 3).unsqueeze(1) == torch.arange(3, device=dev)).sum(0).float())  # no bincount (host sync)
            EX_P.index_add_(0, c, dn.to(EX_P.dtype))
            # probability that one exam draw picks clip c (the source mixture; draws without replacement and the
            # fallbacks ignored)
            pi = (EX_MIXW[0] * wp_s[c] / wp_s.sum().clamp(min=1e-30) + EX_MIXW[1] * pr_s[c] / pr_s.sum().clamp(min=1e-30)
                  + EX_MIXW[2] / len(EX_CLIPS))  # fmt: skip
            EX_WPI.copy_(torch.where(dn, pi.clamp(min=1e-12), EX_WPI))
            EX_PIC.index_put_((torch.where(dn, c, len(motion.names)),), pi.clamp(min=1e-12))
            return c

        env.exam_sampler_masked = exam_sample_masked
    EXAM_NOISE = args.action_noise
    EX_FLIP = torch.tensor([n.startswith("flip_") for n in motion.names], device=dev)
    EX_ENDS, EX_FAILS = torch.zeros((), device=dev), torch.zeros(len(motion.names), device=dev)  # since the last log
    if args.exam_pre_every:
        assert args.exam_mix, "--exam_pre_every feeds the --exam_mix priority"
        _e1 = TrackingSim(make_sim(robot), 1, robot, motion, mdp, dev, eval_mode=True)  # only for the nominal context
        EX_XI1, EX_XIN1 = _e1.xi[:1].clone(), layout.normalize(_e1.xi)[:1].clone()
        del _e1
        _g = torch.Generator().manual_seed(args.seed + 7919)  # own stream: the training random stream is unchanged
        EX_ROTK, EX_PRE_N = torch.randperm(len(motion.names), generator=_g).to(dev), [0]

        @torch.no_grad()
        def exam_preexam(it):
            """--exam_pre_every: every exam clip deployed in the world model as the evaluation runs it;
            EX_PD = the imagined episode's max danger ratio."""
            torch.cuda.synchronize()
            t0 = time.time()
            pol = pi_avg.module
            todo = EX_CLIPS
            if args.exam_pre_near > 0 and bool(torch.isfinite(EX_PD[EX_CLIPS]).all()):
                # the fused estimate (IT_T == it at every call); clips below the band only in their rotation slice
                D = exam_fused()[0]
                near = D[EX_CLIPS] >= args.exam_pre_near
                rot = (EX_ROTK[EX_CLIPS] % args.exam_pre_rot) == (EX_PRE_N[0] % args.exam_pre_rot)
                todo = EX_CLIPS[near | rot]
            EX_PRE_N[0] += 1
            for b0 in range(0, len(todo), 16384):
                c = todo[b0 : b0 + 16384]
                n = len(c)
                f, end = motion.clip_start[c].clone(), motion.clip_end[c]
                S = tuple(x.clone() for x in motion.ref_state(f))
                a_prev = torch.zeros(n, AW, device=dev)
                xi, xin = EX_XI1.expand(n, -1), EX_XIN1.expand(n, -1)
                hist = mdp.obs(S, a_prev, xi, f).unsqueeze(1).repeat(1, Hh, 1)
                dmax = torch.zeros(n, device=dev)
                alive = torch.ones(n, dtype=torch.bool, device=dev)
                ff = torch.full((n,), -1, dtype=torch.long, device=dev)
                pk = f.clone()
                for t in range(args.exam_pre_steps):
                    a = pol(hist, motion.goal(f, K))
                    S = wm.step(S, a, a_prev, xin)
                    f1 = torch.minimum(f + 1, end)
                    d = mdp.danger(S, f1)
                    pk = torch.where(alive & (d > dmax), f1, pk)
                    dmax = torch.where(alive, torch.maximum(dmax, d), dmax)
                    ff = torch.where(alive & (d >= 1), f1, ff)
                    alive &= (f1 < end) & (d < 1)
                    if t % 50 == 49 and not bool(alive.any()):
                        break
                    f = torch.minimum(f1, end - 1)
                    hist = torch.cat((hist[:, 1:], mdp.obs(S, a, xi, f).unsqueeze(1)), 1)
                    a_prev = push_a(a, a_prev)
                EX_PD[c] = dmax
                EX_PF[c] = ff.float()
                EX_PK[c] = pk.float()
                EX_PT[c] = float(it)
            torch.cuda.synchronize()
            p = EX_PD[EX_CLIPS]
            print(f"--exam_pre it {it}: {len(todo)}/{len(EX_CLIPS)} clips, {args.exam_pre_steps} steps, {time.time() - t0:.1f} s | "
                  f"score >= 1 {int((p >= 1).sum())}, 0.8-1 {int(((p >= 0.8) & (p < 1)).sum())}, "
                  f"0.5-0.8 {int(((p >= 0.5) & (p < 0.8)).sum())}", flush=True)  # fmt: skip

        if not PHASE or phase_at(start_it)["exam_frac"] > 0:
            exam_preexam(start_it)
    print(f"--exam: {N_EX} exam worlds over {len(EX_CLIPS)} clips, noise {EXAM_NOISE}", flush=True)
buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, J, layout.dim,
                      a_dim=AW, vel=N_V > 0, land=N_L > 0, gait=SYM_ON)  # fmt: skip
buffer.env_lo = N_C
obs_hist = torch.zeros(args.num_envs, Hh, mdp.obs_dim, device=dev)
if args.start_prac_frac >= 0:
    assert 0 < N_EX < args.num_envs, "--start_prac_frac needs --exam_frac in (0, 1)"
    buffer.start_split = (args.num_envs - N_EX, args.start_prac_frac)
    if PHASE:  # device scalars, set in place by phase_apply (the captured policy update follows them)
        SPLIT_N0, SPLIT_P = torch.full((), args.num_envs - N_EX, device=dev), torch.full((), args.start_prac_frac, device=dev)
        buffer.start_split = (SPLIT_N0, SPLIT_P)


@torch.no_grad()
def exam_record(out):
    """Count the exam episodes' ends and failures per clip and update the clip table."""
    global EX_ENDS
    n = N_EX
    term, f, done_ex = out["term"][-n:], out["f"][-n:], out["done"][-n:]
    if PHASE:  # only the exam slots inside the current share
        term, done_ex = term & env.exam[-n:], done_ex & env.exam[-n:]
    EX_ENDS += done_ex.float().sum()
    EX_FAILS.index_add_(0, motion.clip_id[f], term.float())
    if args.exam_mix:  # the clip table: last exam's max danger ratio (the exam measures the margin)
        rho = mdp.danger(unflatten_state(out["s"][-n:]), f)  # pre-step state (s1 at a clip end is scored against the next clip); a failure counts via term
        new = (out["ep_step"][-n:] == 0) | (rho > EX_RUN)
        EX_RUNF.copy_(torch.where(new, f, EX_RUNF))
        EX_RUN.copy_(torch.where(out["ep_step"][-n:] == 0, rho, torch.maximum(EX_RUN, rho)))
        dn = done_ex
        cl = motion.clip_id[f]
        k = torch.where(term, EX_K[cl] + 1, torch.zeros_like(EX_K[cl]))
        d = torch.where(term, 1.0 + 0.1 * (k - 1), EX_RUN.clamp(max=0.999))
        now = IT_T
        EX_P.index_add_(0, cl, -dn.float())
        idx = torch.where(dn, cl, len(motion.names))  # dropped: worlds whose episode goes on
        for tab, val in ((EX_D, d), (EX_K, k), (EX_T, now.expand_as(d)), (EX_FF, torch.where(term, f.float(), -1.0)),
                         (EX_KF, torch.where(term, f, EX_RUNF).float()), (EX_IW, 1.0 / EX_WPI)):
            tab.copy_(torch.cat((tab, tab[:1])).index_put_((idx,), val)[:-1])
    return


@torch.no_grad()
def exam_refresh(it):
    """Every --exam_every iterations: log the exam episodes."""
    global EX_ENDS
    if (it - start_it) % args.exam_every:
        return
    if WS and WS_ACTIVE[0]:
        print(f"--wstream it {it}: {int(WST['ends'])} episode ends, {int(WST['fails'])} imagined failures since the last log", flush=True)
        WST["ends"].zero_(), WST["fails"].zero_()
    fc = EX_FAILS > 0
    if args.exam_mix:
        seen = torch.isfinite(EX_D)
        s = EX_SRC / EX_SRC.sum().clamp(min=1)
        extra_mix = (f" | table: {int(seen.sum())} clips examined, {int((EX_D >= 1).sum())} failed last time, "
                     f"{int(((EX_D >= 0.8) & (EX_D < 1)).sum())} at 0.8-1 | sources {s[0]:.2f} / {s[1]:.2f} / {s[2]:.2f}")
        EX_SRC.zero_()
    else:
        extra_mix = ""
    print(f"--exam it {it}: {float(EX_ENDS):.0f} episodes, {float(EX_FAILS.sum()):.0f} failures on {int(fc.sum())} clips "
          f"({int((fc & ~EX_FLIP).sum())} non-flip, {int((fc & EX_FLIP).sum())} flip)"
          + extra_mix, flush=True)  # fmt: skip
    EX_ENDS.zero_(), EX_FAILS.zero_()


@torch.no_grad()
def front_refresh(it):
    """--front_share: reset focus on each failing clip's frontier region (equal mass per clip, uniform over its frames)."""
    F = torch.where(torch.isfinite(EX_FF), EX_FF, torch.nan_to_num(EX_PF, nan=-1.0))  # real exam first
    Fb = F[motion.bin_clip]
    lo = torch.maximum(motion.bin_start.float(), Fb - args.front_win)
    w = torch.where(Fb >= 0, (motion.bin_end.float() - lo).clamp(min=0), 0.0)
    per = torch.zeros(len(motion.names), device=dev).index_add_(0, motion.bin_clip, w)
    q = w / per[motion.bin_clip].clamp(min=1e-9)
    n_real, n_pre = int((torch.isfinite(EX_FF) & (EX_FF >= 0)).sum()), int((~torch.isfinite(EX_FF) & (EX_PF >= 0)).sum())
    on = float(q.sum()) > 0
    env.set_focus(q / q.sum() if on else torch.full_like(q, 1.0 / len(q)), args.front_share if on else 0.0)
    if (it - start_it) % args.exam_every == 0:
        print(f"--front it {it}: {n_real + n_pre} frontier clips ({n_real} from exams, {n_pre} from pre-exams), "
              f"{int((q > 0).sum())} bins, share {args.front_share if on else 0.0}", flush=True)


CG = {"calls": 0}  # captured collection step: static tensors and the graph


@torch.no_grad()
def _collect_core():
    """One collection step up to the episode ends, on the static tensors of CG (the captured body)."""
    o = env.observe()
    fresh = env.ep_step == 0
    oh = CG["obs_hist"]
    oh_new = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(oh), torch.cat((oh[:, 1:], o.unsqueeze(1)), 1))
    g = motion.goal(env.f, K)
    if not args.freeze_base:  # --freeze_base: the loaded normalizers stay as they are
        pi.obs_norm.update(o[N_C:])  # the motion envs' statistics, as without --vel_frac: the decoder is trained on the
        pi.goal_norm.update(g[N_C:])  # motion rows alone
    a = pi(oh_new[N_C:], g[N_C:])
    if N_L:  # landing robots: the commands that arrive on this state, then enc_land
        ol = oh_new[N_V:N_C]
        a_l = pi.decode_frozen(pi.encode_land(env.land_command(o), ol), ol).to(a.dtype)
        a = torch.cat((a_l, a))
    if N_V:  # velocity envs: the velocity encoder
        ov = oh_new[:N_V]
        a = torch.cat((pi.decode_frozen(pi.encode_vel(mdp.normalize_cmd(env.command()[:N_V]), ov), ov).to(a.dtype), a))
    a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
    if N_EX:  # exam worlds: the deployed (Polyak) policy plus noise
        ae = pi_avg.module(oh_new[-N_EX:], g[-N_EX:]).to(a.dtype)
        ae = torch.clamp(ae + EXAM_NOISE * torch.randn_like(ae), -20, 20)
        if PHASE:  # exam slots outside the current share run practice episodes
            ae = torch.where(env.exam[-N_EX:, None], ae, a[-N_EX:])
        a = torch.cat((a[:-N_EX], ae))
    out = env.step_core(a)
    if N_EX:  # before the resets below, which draw the next exam clips from the updated table
        exam_record(out)
    buffer.add_device(out)
    S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
    wm_in, wm_out, wm_inc = (wm_norm_data_c)(S0, S1, out["a"], out["a_prev"], out["xi"])
    wm.in_norm.update(wm_in)
    wm.out_norm.update(wm_out)
    wm.inc_norm.update(wm_inc)
    oh.copy_(oh_new)
    # the simulator wrapper rebinds these per step: copy them back into the static tensors the graph reads
    CG["f"].copy_(env.f), CG["ep_step"].copy_(env.ep_step), CG["a_prev"].copy_(env.a_prev)
    env.f, env.ep_step, env.a_prev = CG["f"], CG["ep_step"], CG["a_prev"]
    core = {"done": out["done"], "term": out["term"], "f": out["f"], "r_mean": out["r"][N_C:].mean() / 0.1}
    if N_V:
        vt = out["vel_terms"]
        core.update(vel_r=out["r"][:N_V].mean() / mdp.vel.max_step_reward, vel_e_lin=vt["e_lin"].mean(),
                    vel_slide=vt["slide"].mean(), vel_jerk=vt["jerk"].mean(), vel_term=out["term"][:N_V].float().sum(),
                    vel_done=out["done"][:N_V].float().sum())  # fmt: skip
        if SYM_ON:  # the simulator's scored steps: their summed penalty and their number
            core.update(vel_sym=vt["sym"].sum(), vel_sym_n=vt["sym_n"].sum())
    if N_L:  # the landing score and error count on the deadline steps alone: their sums and the number of deadlines
        lt = out["land_terms"]
        dl = lt["deadline"].float()
        core.update(land_r=out["r"][N_V:N_C].mean() / mdp.land.max_step_reward, land_G=(lt["G"] * dl).sum(),
                    land_e=(lt["e"] * dl).sum(), land_dl=dl.sum(), land_term=out["term"][N_V:N_C].float().sum(),
                    land_done=out["done"][N_V:N_C].float().sum())  # fmt: skip
    out = env.step_tail_masked(out)
    n = out["n_done"]
    nan = torch.full((), float("nan"), device=dev)
    core.update(ep_len=torch.where(n > 0, out["ep_len_sum"] / n.clamp(min=1), nan),
                ep_ret=torch.where(n > 0, out["ep_ret_norm_sum"] / n.clamp(min=1), nan),
                term_sum=out["term"][N_C:].float().sum(), done_sum=n)  # fmt: skip
    return core


@torch.no_grad()
def collect_graphed_step():
    import warp as wp
    from mjlab.sim.sim_data import TorchArray

    global obs_hist
    if CG["calls"] == 0:
        CG.update(obs_hist=obs_hist.clone(), f=env.f.clone(), ep_step=env.ep_step.clone(), a_prev=env.a_prev.clone())
        env.f, env.ep_step, env.a_prev = CG["f"], CG["ep_step"], CG["a_prev"]
    CG["calls"] += 1
    if CG["calls"] <= 3:  # warmup on the default stream (Warp's own stream orders with it implicitly)
        out = _collect_core()
    elif "graph" not in CG:
        torch.cuda.synchronize()
        t0 = time.time()
        CG["graph"] = torch.cuda.CUDAGraph()
        setitem, env.sim.use_cuda_graph = TorchArray.__setitem__, False
        TorchArray.__setitem__ = lambda self, idx, value: self._tensor.__setitem__(idx, value)
        try:
            with torch.cuda.graph(CG["graph"], capture_error_mode="thread_local"):
                ws = wp.stream_from_torch(torch.cuda.current_stream())
                with wp.ScopedStream(ws, sync_enter=False, sync_exit=False):
                    wp.capture_begin(stream=ws, external=True)
                    try:
                        CG["out"] = _collect_core()
                    finally:
                        CG["wp_graph"] = wp.capture_end(stream=ws)  # keep: it holds the captured kernels' modules
        finally:
            TorchArray.__setitem__, env.sim.use_cuda_graph = setitem, True
        torch.cuda.synchronize()
        print(f"captured one collection step in {time.time() - t0:.1f} s", flush=True)
        CG["graph"].replay()  # the capture itself does not run the step
        out = CG["out"]
    else:
        CG["graph"].replay()
        out = CG["out"]
    buffer.advance_host()
    obs_hist = CG["obs_hist"]
    return out


VEL_STATS = (("vel_r", "vel_e_lin", "vel_slide", "vel_jerk", "vel_term", "vel_done") + (("vel_sym", "vel_sym_n") if SYM_ON else ())
             if N_V else ())  # fmt: skip
LAND_STATS = ("land_r", "land_G", "land_e", "land_dl", "land_term", "land_done") if N_L else ()


def new_stats() -> dict:
    return {k: [] for k in ("r", "ep_len", "ep_ret", "term", "done") + VEL_STATS + LAND_STATS}


@torch.no_grad()
def collect(n_steps: int, stats: dict):
    global obs_hist, sim_steps
    for _ in range(n_steps):
        core = collect_graphed_step()
        sim_steps += args.num_envs
        stats["r"].append(core["r_mean"].clone())
        for k_s, k_c in (("ep_len", "ep_len"), ("ep_ret", "ep_ret"), ("term", "term_sum"), ("done", "done_sum")):
            stats[k_s].append(core[k_c].clone())
        for k_s in VEL_STATS + LAND_STATS:
            stats[k_s].append(core[k_s].clone())
        continue
    return

env.use_static_focus()
buffer.use_device_ptr()
# captured sampling reads the buffer size on the device, so the graphs are captured at once
full = (lambda: True)
wm_update_g = Graphed(wm_update, full)
policy_update_g = Graphed(policy_update, lambda: int(pi_avg.n_averaged if pi_avg is not None else 1) > 0)


if WS:
    _V = args.wstream_worlds
    _f0 = motion.clip_start[torch.zeros(_V, dtype=torch.long, device=dev)]
    _fl = flatten_state(tuple(x.clone() for x in motion.ref_state(_f0)))
    _xi = EX_XI1.expand(_V, -1).clone()
    WST["s"] = {"s": _fl.clone(), "a_prev": torch.zeros(_V, AW, device=dev), "xi": _xi, "f": _f0.clone()}
    WST["h"] = {"s": _fl.unsqueeze(1).repeat(1, Hh, 1), "a_prev": torch.zeros(_V, Hh, AW, device=dev),
                "xi": _xi.unsqueeze(1).repeat(1, Hh, 1), "f": _f0.unsqueeze(1).repeat(1, Hh)}  # fmt: skip
    WST["o"] = torch.zeros(_V, Hh, mdp.obs_dim, device=dev)
    WST["xin"] = EX_XIN1.expand(_V, -1).clone()
    _len = (motion.clip_end - motion.clip_start + 1).to(torch.float64)
    WST["cdf"] = torch.cumsum(_len ** (-1.0) / (_len ** (-1.0)).sum(), 0)  # equal time per clip
    WST["ends"], WST["fails"] = torch.zeros((), device=dev), torch.zeros((), device=dev)


    @torch.no_grad()
    def ws_draw():
        """A first frame for every stream world (clip probability WST['cdf']), its reference state and observation."""
        u = torch.rand(args.wstream_worlds, device=dev, dtype=torch.float64) * WST["cdf"][-1]
        c = torch.searchsorted(WST["cdf"], u, right=True).clamp(max=len(WST["cdf"]) - 1)
        f = motion.clip_start[c]
        S = tuple(x.clone() for x in motion.ref_state(f))
        return f, S, mdp.obs(S, torch.zeros(args.wstream_worlds, J, device=dev), WST["xi_s"], f)


    WST["xi_s"] = WST["s"]["xi"]

    @torch.no_grad()
    def ws_advance():
        """--wstream_steps steps of the stream in the world model (fixed size, capturable); as exam_preexam."""
        V, st, hi = args.wstream_worlds, WST["s"], WST["h"]
        for _ in range(args.wstream_steps):
            S, f, end = unflatten_state(st["s"]), st["f"], motion.clip_end[motion.clip_id[st["f"]]]
            a = pi_avg.module(WST["o"], motion.goal(f, K)).float()
            S1 = wm.step(S, a, st["a_prev"], WST["xin"])
            f1 = torch.minimum(f + 1, end)
            fail = mdp.danger(S1, f1) >= 1
            done = fail | (f1 >= end)
            f_new, S_new, o_new = ws_draw()
            S2 = tuple(torch.where(done.view(-1, *([1] * (x.dim() - 1))), xn, x) for xn, x in zip(S_new, S1))
            f2 = torch.where(done, f_new, f1)
            a2 = torch.where(done.unsqueeze(-1), 0.0, push_a(a, st["a_prev"]))
            fl = flatten_state(S2)
            o2 = torch.where(done.unsqueeze(-1), o_new, mdp.obs(S2, a2, st["xi"], f2))
            d2 = done.view(V, 1, 1)
            WST["o"].copy_(torch.where(d2, o2.unsqueeze(1).expand_as(WST["o"]), torch.cat((WST["o"][:, 1:], o2.unsqueeze(1)), 1)))
            hi["s"].copy_(torch.where(d2, fl.unsqueeze(1).expand_as(hi["s"]), torch.cat((hi["s"][:, 1:], fl.unsqueeze(1)), 1)))
            hi["a_prev"].copy_(torch.where(d2, 0.0, torch.cat((hi["a_prev"][:, 1:], a2.unsqueeze(1)), 1)))
            d1 = done.view(V, 1)
            hi["f"].copy_(torch.where(d1, f2.unsqueeze(1).expand_as(hi["f"]), torch.cat((hi["f"][:, 1:], f2.unsqueeze(1)), 1)))
            st["s"].copy_(fl), st["a_prev"].copy_(a2), st["f"].copy_(f2)
            WST["ends"] += done.float().sum()
            WST["fails"] += fail.float().sum()
        return (WST["ends"],)


    ws_advance_g = Graphed(ws_advance, lambda: True)


    @torch.no_grad()
    def ws_set(on: bool, it: int):
        """Switch the stream's mixing; when it starts, fresh episodes and a burn-in so that the worlds spread along their clips."""
        if on and not WS_ACTIVE[0]:
            t0 = time.time()
            f, S, o = ws_draw()
            fl = flatten_state(S)
            WST["s"]["s"].copy_(fl), WST["s"]["a_prev"].zero_(), WST["s"]["f"].copy_(f)
            WST["o"].copy_(o.unsqueeze(1).expand_as(WST["o"]))
            WST["h"]["s"].copy_(fl.unsqueeze(1).expand_as(WST["h"]["s"]))
            WST["h"]["a_prev"].zero_(), WST["h"]["f"].copy_(f.unsqueeze(1).expand_as(WST["h"]["f"]))
            for _ in range(args.wstream_burn):
                ws_advance_g()
            torch.cuda.synchronize()
            print(f"--wstream it {it}: on, {args.wstream_worlds} worlds, share {args.wstream_frac}, burn-in {time.time() - t0:.1f} s", flush=True)
        WS_ACTIVE[0] = on
        WS_ON.fill_(on)


assert not RAMP or (PHASE and args.exam_pre_every and 0 < RAMP[0] <= RAMP[1] <= args.exam_frac and RAMP[2] > 0 and RAMP[3] > 0), \
    "--share_ramp: needs --phase, --exam_pre_every and lo <= hi <= --exam_frac"

PH_STATE = {"cur": None, "pre_at": None}  # --phase: settings in force; iteration of a pre-exam run by phase_apply


def ph_exams() -> bool:
    """--phase: whether any exam world is active (always true without --phase)."""
    return not PHASE or PH_STATE["cur"]["exam_frac"] > 0


@torch.no_grad()
def ramp_eval(it: int) -> bool:
    """--share_ramp: set RAMP_OV from the pre-exam just run; True when the share changes."""
    lo, hi, n0, st = RAMP
    n1 = int((EX_PD[EX_CLIPS] >= 1).sum())
    s = hi - (hi - lo) * min(1.0, n1 / n0)
    s = min(hi, lo + st * round((s - lo) / st))
    old = PH_STATE["cur"]["exam_frac"]
    RAMP_OV["exam_frac"] = s
    print(f"--share_ramp it {it}: n1 {n1} -> exam_frac {s:.3f}" + ("" if abs(old - s) < 1e-9 else f" (was {old:.3f})"), flush=True)
    return abs(old - s) >= 1e-9


def phase_apply(it: int, initial: bool = False, ramp: bool = False):
    """--phase: apply the settings in force at iteration it (initial: at the start of this process; ramp: a --share_ramp
    change, without a refill)."""
    global policy_update_g
    new, old = phase_at(it), PH_STATE["cur"]
    PH_STATE["cur"] = new
    args.wm_batches = new["wm_batches"]  # host-side loop over the captured world-model update
    if new["critic_epochs"] != args.critic_epochs:  # the epoch loop is unrolled in the captured update: capture it anew
        args.critic_epochs = new["critic_epochs"]
        ready = policy_update_g.ready
        policy_update_g = None  # frees the old graph's pool before the new capture
        policy_update_g = Graphed(policy_update, ready)
        print(f"--phase it {it}: critic epochs {args.critic_epochs}", flush=True)
    n_act = int(round(new["exam_frac"] * N_M))
    n_old = N_EX if old is None else int(round(old["exam_frac"] * N_M))
    args.lr_decay_iters, args.lr_end, args.front_share = new["lr_decay_iters"], new["lr_end"], new["front_share"]
    PEN.fill_(new["fail_penalty"])
    env.aprev_gate.fill_(new["reset_aprev_ref"])
    if WS:
        ws_set(new["wstream_frac"] > 0, it)
    if n_act > 0 and n_old == 0 and not initial and args.exam_pre_every:  # exams start: as a process started here
        exam_preexam(it)
        PH_STATE["pre_at"] = it
        if RAMP:
            ramp_eval(it)
            new = phase_at(it)
            PH_STATE["cur"] = new
            n_act = int(round(new["exam_frac"] * N_M))
    if n_act != n_old or initial:
        ex_old = env.exam.clone()
        changed = env.set_exam_active(n_act)
        if not initial:  # exam episodes in flight of the worlds that stop examining (the setup draws are not counted)
            gone = (changed & ex_old)[-N_EX:]
            EX_P.index_add_(0, motion.clip_id[env.f[-N_EX:]], -gone.float())
        if bool(changed.any()):
            if buffer.size > 0:  # the reset ends these episodes: mark their last transition (world-model windows stop there)
                last = (buffer.ptr_t - 1) % buffer.T
                for k in ("cut", "done"):
                    buffer.data[k][last.view(1)] = buffer.data[k][last.view(1)] | changed
            env.reset_masked(changed)
            if args.phase_refill and not initial and not ramp:  # as a process restarted here: the whole buffer comes from the new roles
                t_fill = time.time()
                if args.front_share > 0 and old["front_share"] == 0:  # the frontier first, as a restarted process has it
                    front_refresh(it)
                SPLIT_N0.fill_(args.num_envs - n_act)
                collect(args.buffer_steps, new_stats())
                print(f"--phase it {it}: buffer refilled with {args.buffer_steps} steps in {time.time() - t_fill:.0f} s", flush=True)
    else:
        changed = torch.zeros(args.num_envs, dtype=torch.bool, device=dev)
    p = new["start_prac_frac"]
    SPLIT_N0.fill_(args.num_envs - n_act)
    SPLIT_P.fill_(1.0 if n_act == 0 else (p if p >= 0 else (N_M - n_act) / N_M))
    if args.front_share == 0:
        env.set_focus(torch.full_like(env.focus_q, 1.0 / len(env.focus_q)), 0.0)
    elif n_act > 0 and (initial or old["front_share"] == 0) and (it - start_it) % args.front_every:
        front_refresh(it)
    print(f"--phase it {it}: " + ", ".join(f"{k}={v}" for k, v in new.items()) + f" | {n_act} exam worlds, "
          f"{int(changed.sum())} worlds reset, start split ({int(SPLIT_N0)}, {float(SPLIT_P):.3f})", flush=True)


stats = new_stats()
if PHASE:
    phase_apply(start_it, initial=True)
elif WS:
    ws_set(True, start_it)
collect(args.warmup_steps, stats)  # the replay buffer's first steps, before any update
log_path = os.path.join(args.log_dir, "train_log.csv")
log_f = open(log_path, "a")
log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr"
            + ",pi_lr,pi_mult,nonfinite" + "".join(f",{k}" for k in VEL_COLS if N_V)
            + "".join(f",{k}" for k in LAND_COLS if N_L) + (",dr_ramp,kp_std,gyro_std,gain_err,fric_err" if MEASURED else "") + "\n")  # fmt: skip
log_f.flush()
LOG_EXTRA = "pi_lr" in open(log_path).readline()  # lr = base LR; pi_lr = policy LR (x pi_mult)
# checkpoints: one background writer; a failed write is re-raised in the training loop (or at the end), at most 2 queued
CKPT_POOL, CKPT_PENDING = (ThreadPoolExecutor(max_workers=1)), []


def ckpt_wait(drain=False):
    while CKPT_PENDING and (drain or CKPT_PENDING[0].done() or len(CKPT_PENDING) > 2):
        CKPT_PENDING.pop(0).result()


def _save_atomic(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


t_start = time.time() - wall_offset
t_log = time.time()
step_acc = 0.0
agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": [], "vel": [], "land": []}


for it in range(start_it, args.iterations):
    mdp.ramp.fill_(ramp_at(it, args.dr_ramp_iters))  # in place: the captured collection step and policy update read it
    if N_EX:
        IT_T.fill_(it)
    if PHASE and it in PH_ITS and it != start_it:
        phase_apply(it)
    lr = lr_at(it)
    pi_mult = args.pi_lr_mult
    for o, mult in OPTS:
        for gparam in o.param_groups:
            gparam["lr"].fill_(lr * mult)
    step_acc += 1.0 / args.utd
    n = int(step_acc)
    step_acc -= n
    # exams: refresh the exam clip table, pre-exams in the world model, frontier practice regions
    if N_EX and it > start_it and (it - start_it) % args.exam_every == 0 and ph_exams():
        exam_refresh(it)
    if (N_EX and args.exam_pre_every and it > start_it and (it - start_it) % args.exam_pre_every == 0 and ph_exams()
            and PH_STATE["pre_at"] != it):
        exam_preexam(it)
        if RAMP and ramp_eval(it):
            phase_apply(it, ramp=True)
    if args.front_share and (it - start_it) % args.front_every == 0:
        front_refresh(it)
    collect(n, stats)
    for b_ in range(args.wm_batches):
        l, e8 = wm_update_g()
    agg["wm"].append(l), agg["wm8"].append(e8)
    n_avg0 = int(pi_avg.n_averaged) if pi_avg is not None else 0
    if WS and WS_ACTIVE[0]:
        ws_advance_g()
    pl, vl, ir, gn, surv, *vel_out = policy_update_g()
    if N_L:  # the velocity interface's outputs come first, the landing interface's six after them
        n_vo = 4 + len(VEL_TERMS) + MIRROR_ON  # + vel_mirror
        vel_out, land_out = vel_out[:n_vo], vel_out[n_vo:]
        agg["land"].append(land_out)
    agg["vel"].append(vel_out)
    if (pi_avg is not None and int(pi_avg.n_averaged) == n_avg0):
        pi_avg.n_averaged += 1  # ema_update (and a graph replay) does not advance the host-side counter
    agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
    if (it + 1) % args.log_every == 0:
        mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
        mean_nn = lambda x: mean([v for v in (float(v) for v in x) if v == v])  # noqa: E731  (NaN = no episode ended)
        ips = args.log_every / (time.time() - t_log)
        t_log = time.time()
        wall_h = (time.time() - t_start) / 3600
        term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
        row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
               mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean_nn(stats["ep_len"]), mean_nn(stats["ep_ret"]), term_frac, ips, lr]  # fmt: skip
        extra = [lr * pi_mult, float(pi_mult), int(NONFINITE)] if LOG_EXTRA else []
        if N_V:  # VEL_COLS: policy_update's extra outputs, then the simulator's
            vel_row = [mean(col) for col in zip(*agg["vel"])] + [mean(stats[k]) for k in VEL_STATS[:4]]
            vel_row.append(float(sum(float(v) for v in stats["vel_term"]) / max(sum(float(v) for v in stats["vel_done"]), 1)))
            if SYM_ON:  # vel_sim_sym: the penalty per scored step over the interval
                vel_row.append(float(sum(float(v) for v in stats["vel_sym"]) / max(sum(float(v) for v in stats["vel_sym_n"]), 1)))
            extra += vel_row
        if N_L:  # LAND_COLS: policy_update's extra outputs, then the simulator's
            # G and err: an update without a deadline gives NaN and is left out of the mean; the simulator's are sums
            # over the interval divided by its number of deadlines
            cols, total = list(zip(*agg["land"])), lambda k: sum(float(v) for v in stats[k])  # noqa: E731
            land_row = [mean(col) for col in cols[:4]] + [mean_nn(col) for col in cols[4:]] + [mean(stats["land_r"])]
            land_row += [float(total(k) / max(total("land_dl"), 1)) for k in ("land_G", "land_e")]
            land_row.append(float(total("land_term") / max(total("land_done"), 1)))
            extra += land_row
        if MEASURED:  # read back from the device: what the captured collection step and policy update actually used
            # gain_err, fric_err: the simulator's gains and joint friction against xi (DomainRandomizer.sim_error), ~0
            extra += [float(mdp.ramp), float(env.xi[:, layout.slices["kp"]].std()) if "kp" in layout.slices else 0.0,
                      float(SENS_STD), *env.randomizer.sim_error(env.xi)]  # fmt: skip
            print(f"   dr | ramp {extra[-5]:.3f} kp_std {extra[-4]:.4f} gyro_std {extra[-3]:.5f} | simulator vs xi: gains "
                  f"{extra[-2]:.2e} friction {extra[-1]:.2e}", flush=True)  # fmt: skip
        log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row + extra) + "\n")
        log_f.flush()
        vc = VCL.tolist()
        VCL.zero_()
        print(f"   vclamp | V {vc[0] / max(vc[3], 1):.4f} targets {vc[1] / max(vc[3], 1):.4f} boot {vc[2] / max(vc[3], 1):.4f} above V_max", flush=True)
        print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
              f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | {ips:.2f} it/s", flush=True)  # fmt: skip
        if N_V:
            vr = dict(zip(VEL_COLS, vel_row))
            print(f"   vel | v {vr['vel_v_loss']:.4f} imag_r {vr['vel_imag_rew']:.3f} surv {vr['vel_imag_survival']:.3f} align {vr['align']:.4f} stance {vr['vel_imag_stance']:.4f} ankle {vr['vel_imag_ankle']:.4f} "
                  f"| sim_r {vr['vel_sim_rew']:.3f} e_lin {vr['vel_sim_e_lin']:.4f} slide {vr['vel_sim_slide']:.3f} "
                  f"jerk {vr['vel_sim_jerk']:.0f} term {vr['vel_term_frac']:.2f}"
                  + (f" sym {vr['vel_sim_sym']:.4f} n {vr['vel_imag_sym_n']:.3f}" if SYM_ON else ""), flush=True)  # fmt: skip
        if N_L:
            ld = dict(zip(LAND_COLS, land_row))
            print(f"   land | v {ld['land_v_loss']:.4f} imag_r {ld['land_imag_rew']:.3f} "
                  f"surv {ld['land_imag_survival']:.3f} align {ld['land_align']:.4f} G {ld['land_imag_G']:.3f} "
                  f"err {ld['land_imag_err']:.4f} | sim_r {ld['land_sim_rew']:.3f} G {ld['land_sim_G']:.3f} "
                  f"err {ld['land_sim_err']:.4f} term {ld['land_term_frac']:.2f}", flush=True)  # fmt: skip
        stats = {k: [] for k in stats}
        agg = {k: [] for k in agg}
    final_ck = it + 1 == args.iterations
    if BASE is not None and ((it + 1) % args.ckpt_every == 0 or final_ck):
        # --freeze_base: the policy about to be saved (the averaged one, and the live one beside it in the last
        # checkpoint) must hold the loaded tensors bit for bit; the run stops before a file is written if it does not
        for _name, _m in (("pi", pi_avg.module if pi_avg is not None else pi), ("pi_live", pi)):
            _sd = _m.state_dict()
            _bad = next((k for k, v in BASE.items() if not torch.equal(_bits(_sd[k]), _bits(v))), None)
            if _bad is not None:
                raise RuntimeError(f"--freeze_base: {_name}'s {_bad} differs from the tensor loaded from "
                                   f"{args.init_from} (iteration {it + 1})")  # fmt: skip
    if (not final_ck and (it + 1) % args.ckpt_every == 0):
        sd = pi_avg.module.state_dict() if pi_avg is not None else pi.state_dict()
        small = {"pi": {k: v.detach().to("cpu", copy=True) for k, v in sd.items()}, "iteration": it + 1,
                 "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
                 "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
        ckpt_wait()
        CKPT_PENDING.append(CKPT_POOL.submit(_save_atomic, small, path))
    elif ((it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations):
        ck = {"wm": wm.state_dict(), "pi": pi_avg.module.state_dict() if pi_avg is not None else pi.state_dict(), "pi_live": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
              "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
              "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
              "pi_avg": pi_avg.state_dict() if pi_avg is not None else None, "wm_avg": None,
              "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        if N_V:
            ck.update(critic_v=critic_v.state_dict(), opt_vv=opt_vv.state_dict())
        if N_L:
            ck.update(critic_land=critic_l.state_dict(), opt_vl=opt_vl.state_dict())
        last = os.path.join(args.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        small = {k: ck[k] for k in ("pi", "pi_live", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at")}
        path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
        torch.save(small, path + ".tmp")
        os.replace(path + ".tmp", path)
# every background checkpoint written (a failed write raises here: non-zero exit)
ckpt_wait(drain=True)
CKPT_POOL.shutdown()
