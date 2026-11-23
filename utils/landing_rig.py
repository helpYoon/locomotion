"""A --land_frac checkpoint's landing interface in the training simulator, for utils/landing_test.py, in evaluation
conditions (no randomization, noise or pushes, the fixed evaluation delay). Rig (the test's first part): every world a
landing robot that starts standing in the default pose and is given the commands by the rig (the simulator's own
sampler is off). LibraryRig (the second part): every landing robot a free row that starts on a hold frame of the
library the checkpoint was trained on and is given the simulator's own commands, one after another.

usage: from utils.landing_rig import Rig;  log = Rig(ckpt, worlds).run(cmd)   (GPU only; utils/landing_score.py scores)
       from utils.landing_rig import LibraryRig;  log = LibraryRig(ckpt).run()
"""

import torch
import warp as wp

wp.config.log_level = wp.LOG_WARNING  # no start-up banner or per-kernel load lines

import src.networks as networks
from src import landing
from src.dr import XiLayout
from src.env import TrackingSim, make_sim
from src.landing import LandingCfg
from src.mdp import MDP, unflatten_state
from src.motion import CONTROL_DT, MotionLib
from src.networks import Policy, policy_kwargs
from robot.kinematics import Kinematics
from robot.spec import get_robot
from utils.landing_score import library_file
from utils.rotations import mat_to_quat
from utils.velocity_rig import ACTION_CLIP

SETTLE_S = 1.0  # s: the first second after a reset (the landing SETTLE and some hold), before the commands
# the checkpoint's saved training arguments that make up its LandingCfg (train.py's block `if N_L:`)
CFG_ARGS = {"land_label_prob": "label_prob", "land_bonus": "bonus", "land_drag": "w_drag",
            "land_arrive": "arrive_narrow", "land_progress": "w_progress", "land_derived": "derived"}
N_MOTION = 16  # LibraryRig: motion robots after the landing robots, stepped as the reference diagnostic did, not scored


def _load(ckpt: str) -> dict:
    """The checkpoint of a --land_frac run, on the CPU."""
    torch.backends.cuda.matmul.allow_tf32 = True
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert ck["args"].get("land_frac", 0) > 0, "not a --land_frac checkpoint"
    return ck


def _policy(ck: dict, robot, mdp: MDP, motion: MotionLib, dev: str) -> Policy:
    """The checkpoint's (deployed) policy in evaluation mode, with its training's autocast."""
    a_ = ck["args"]
    widths = lambda key: tuple(int(x) for x in a_.get(key, "2048,1024,512,256").split(","))  # noqa: E731
    pi = Policy(mdp.obs_dim, motion.goal_dim(a_["goal_frames"]), robot.num_joints, hist=a_["hist"],
                enc_hidden=widths("enc_hidden"), dec_hidden=widths("dec_hidden"),
                **policy_kwargs(a_, robot)).to(dev)  # fmt: skip
    pi.load_state_dict(ck["pi"])
    pi.eval()
    networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
    return pi


class Rig:
    def __init__(self, ckpt: str, worlds: int, dev: str = "cuda:0"):
        torch.manual_seed(0)  # each world's commanded sole before the first command is drawn at its reset
        ck = _load(ckpt)
        a_ = self.args = ck["args"]
        self.iteration, self.dev, self.N, self.hist_len = ck["iteration"], dev, worlds, a_["hist"]
        robot = self.robot = get_robot(a_["robot"])
        kin = Kinematics(robot, dev)
        # the library is one clip of the robot standing in its default pose: the start state. Nothing else of it is used
        T = 100
        quat = torch.tensor([1.0, 0, 0, 0]).repeat(T, 1)
        stand = (torch.tensor(robot.robot_cfg().init_state.pos).repeat(T, 1), quat,
                 torch.tensor(robot.default_joint_pos_list()).repeat(T, 1))
        motion = MotionLib.from_clips(["stand"], *([x] for x in stand), kin, dev)
        q0 = torch.tensor(robot.default_joint_pos_list(), device=dev)
        mdp = self.mdp = MDP(kin, motion, q0, XiLayout(robot, dev).dim, robot.sensing)
        mdp.feet, mdp.contact_z = [robot.tracked_bodies.index(b) for b in robot.feet], robot.contact_z
        mdp.enable_velocity(robot)
        self.cfg = LandingCfg(**{f: a_[k] for k, f in CFG_ARGS.items() if k in a_})
        mdp.enable_landing(robot, self.cfg)
        motion.set_land_labels(mdp.feet)
        self.pi = _policy(ck, robot, mdp, motion, dev)
        self.env = env = TrackingSim(make_sim(robot), worlds, robot, motion, mdp, dev, eval_mode=True, n_land=worlds)
        env.land_auto = False
        self.ids, self.f0 = torch.arange(worlds, device=dev), torch.zeros(worlds, dtype=torch.long, device=dev)
        self.hist = None

    def reset(self):
        """Every world standing on the stand clip's first frame, in SETTLE with a fresh observation history."""
        self.env.reset(self.ids, self.f0.clone())
        self.hist = None

    @torch.no_grad()
    def control(self):
        """One control step of every world: observe, push the history, the landing features of the command state in
        force, the policy's action (clamped as training's) and the simulator step."""
        env, pi = self.env, self.pi
        o = env.observe()
        x = o.unsqueeze(1)
        self.hist = x.repeat(1, self.hist_len, 1) if self.hist is None else torch.cat((self.hist[:, 1:], x), 1)
        cf = env.land_command(o)
        a = pi.decode_frozen(pi.encode_land(cf, self.hist), self.hist)
        env.step(torch.clamp(a.float(), -ACTION_CLIP, ACTION_CLIP))

    @torch.no_grad()
    def snapshot(self) -> dict[str, torch.Tensor]:
        """The current state of every world: sole poses (N, 2, 3), clearances (N, 2) and centre heights (N, 2), the
        fall test, the joints' speed RMS [rad/s], the root's speed [m/s], and for the recording its position, wxyz
        quaternion and joints."""
        S = self.env.state()
        sole, _ = self.mdp.sole_kin(S)
        return {"pose": sole["pose"], "h": sole["h"], "zc": sole["zc"], "fell": self.mdp.fall_termination(S),
                "qd_rms": torch.sqrt((S[5] ** 2).mean(-1)), "root_speed": torch.linalg.norm(S[2], dim=-1),
                "root_p": S[0], "root_quat": mat_to_quat(S[1]), "q": S[4]}  # fmt: skip

    @torch.no_grad()
    def reset_stance(self) -> torch.Tensor:
        """The sole poses (2, 3) (left, right) every world is reset to: the stance the commands are built from."""
        self.reset()
        return self.env.land_pose[0].cpu().clone()

    @torch.no_grad()
    def run(self, cmd: dict[str, torch.Tensor], hold_s: float = 3.5, record: int = 0) -> dict:
        """Send every command of cmd (swing, c, n0 as utils.landing_score.command_grid; K of them, any K: the worlds
        take them in chunks) and log the rollouts: the first second of SETTLE and hold, the commands, then hold_s
        seconds of hold after the longest clock. Returns pose (T, K, 2, 3), h (T, K, 2) the lowest corner's clearance,
        zc (T, K, 2) the centre's height, fell (T, K) (once down, down), qd_rms (T, K), root_speed (T, K), t_cmd (the
        state the commands were issued on), target (K, 3) and anchor (K, 2, 3) of the commands (world frame); state j
        is after j control steps. With record > 0 also root_p (T, record, 3), root_quat (T, record, 4) wxyz and q
        (T, record, J) of the first ``record`` commands."""
        K, W, dev = len(cmd["swing"]), self.N, self.dev
        t_cmd = round(SETTLE_S / CONTROL_DT)
        steps = t_cmd + int(cmd["n0"].max()) + round(hold_s / CONTROL_DT)
        keys = ("pose", "h", "zc", "fell", "qd_rms", "root_speed") + (("root_p", "root_quat", "q") if record else ())
        chunks = []
        for a in range(0, K, W):
            ix = torch.arange(a, a + W).clamp(max=K - 1)  # the last chunk repeats its final command
            swing, c, n0 = (cmd[k][ix].to(dev) for k in ("swing", "c", "n0"))
            self.reset()
            rec = {k: [] for k in keys}
            for j in range(steps + 1):
                if j == t_cmd:
                    env = self.env
                    hold = landing.phase(env.land_C)[2].all()
                    assert hold, "the first second did not bring every world out of SETTLE"
                    arrive = torch.ones(W, dtype=torch.bool, device=dev)
                    env.land_C.copy_(landing.land_issue(env.land_C, env.land_pose, arrive, swing, c, n0))
                    target, anchor = env.land_C[:, 3:6].clone(), env.land_C[:, 6:].reshape(W, 2, 3).clone()
                s = self.snapshot()
                for k in keys:
                    rec[k].append(s[k])
                if j < steps:
                    self.control()
            n = min(W, K - a)
            out = {k: torch.stack(rec[k])[:, :n].cpu() for k in keys}
            out["fell"] = out["fell"].int().cummax(0).values.bool()
            out["target"], out["anchor"] = target[:n].cpu(), anchor[:n].cpu()
            chunks.append(out)
        log = {k: torch.cat([c[k] for c in chunks], 1 if k in keys else 0) for k in chunks[0]}
        if record:
            for k in ("root_p", "root_quat", "q"):
                log[k] = log[k][:, :record]
        log["t_cmd"] = t_cmd
        return log


class LibraryRig:
    """The test's second part (a port of the ledger's diagnostic D2): ``worlds`` landing robots, all free rows, that
    start on hold frames of the library the checkpoint was trained on (any stance of it) and are given the simulator's
    own commands one after another (near or anywhere in the box, clock with jitter), in evaluation conditions with
    auto-resets; then the diagnostic's N_MOTION motion robots, stepped and not scored."""

    def __init__(self, ckpt: str, worlds: int = 2048, dev: str = "cuda:0"):
        torch.manual_seed(0)  # each world's commanded sole before its first command is drawn at its reset
        ck = _load(ckpt)
        a_ = self.args = ck["args"]
        self.iteration, self.dev, self.NL = ck["iteration"], dev, worlds
        self.hist_len, self.K = a_["hist"], a_["goal_frames"]
        robot = self.robot = get_robot(a_["robot"])
        kin = Kinematics(robot, dev)
        # the library src/train.py trained on, prepared as it prepares it; found in data/<robot> when the checkpoint
        # was copied from the machine that trained it
        lst, lib = (library_file(a_[k], a_["robot"]) for k in ("bones_list", "bones_lib"))
        if (lst, lib) != (a_["bones_list"], a_["bones_lib"]):
            print(f"library stances: the saved library paths do not exist here, using {lst} and {lib}", flush=True)
        motion = self.motion = MotionLib.load(kin, dev, lst, lib, a_["motion_store"])
        motion.merge_tail_bins()
        q0 = torch.tensor(robot.default_joint_pos_list(), device=dev)
        mdp = self.mdp = MDP(kin, motion, q0, XiLayout(robot, dev).dim, robot.sensing)
        mdp.feet, mdp.contact_z = [robot.tracked_bodies.index(b) for b in robot.feet], robot.contact_z
        motion.set_ref_contact(mdp.feet)
        mdp.enable_velocity(robot)
        motion.set_ref_cmd()
        self.cfg = LandingCfg(**{**{f: a_[k] for k, f in CFG_ARGS.items() if k in a_}, "label_prob": 0.0})
        mdp.enable_landing(robot, self.cfg)
        motion.set_land_labels(mdp.feet)
        self.pi = _policy(ck, robot, mdp, motion, dev)
        self.env = TrackingSim(make_sim(robot), worlds + N_MOTION, robot, motion, mdp, dev, bin_fix=True,
                               reset_noise=False, randomize=False, n_land=worlds)  # fmt: skip

    @torch.no_grad()
    def run(self, steps: int = 750) -> dict:
        """Run every world for ``steps`` control steps. Returns CPU tensors, per deadline (C of them): err (C,) [m] and
        yaw (C,) [rad], the swing sole's distance and yaw from its target; place (C,) long, the command's place in its
        episode (1 for the first); travel (C,) [m], the command's travel when it was issued (distance + YAW_M x |yaw|
        from the swing sole to the target); per episode whose SETTLE ended (E of them): moved (E,) [m] and turned (E,)
        [rad], the larger over the two soles of the distance and the absolute yaw change between the soles where
        SETTLE ends and the soles at the reset; and the ints falls (terminations) and episodes (finished episodes) of
        the landing robots."""
        env, mdp, motion, pi, NL, Hh = self.env, self.mdp, self.motion, self.pi, self.NL, self.hist_len
        L = slice(0, NL)
        count = torch.zeros(NL, device=self.dev)  # deadlines of this episode so far
        trav0 = torch.zeros(NL, device=self.dev)  # the running command's travel when it was sent
        rows, settle_rows = [torch.zeros(0, 4)], [torch.zeros(0, 2)]
        falls = episodes = 0
        hist = pose_reset = None
        for _ in range(steps):
            o = env.observe()
            fresh = (env.ep_step == 0)[:, None, None]
            o_true = mdp.obs(env.state(), env.a_prev, env.xi, env.f, None)
            hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.where(
                fresh, o.unsqueeze(1).expand(-1, Hh, -1), torch.cat((hist[:, 1:], o.unsqueeze(1)), 1))  # fmt: skip
            g = motion.goal(env.f, self.K)
            fr = (env.ep_step == 0)[L]
            pose_reset = env.land_pose.clone() if pose_reset is None else torch.where(
                fr[:, None, None], env.land_pose, pose_reset)  # fmt: skip
            m_before, n_before = env.land_C[:, landing.M].clone(), env.land_C[:, landing.N].clone()
            cf = env.land_command(o)
            C = env.land_C
            issued = (n_before == 0) & (C[:, landing.N] > 0)
            sw = C[:, landing.SWING]
            W = env.land_pose[:, 0] * (1 - sw[:, None]) + env.land_pose[:, 1] * sw[:, None]
            d = C[:, 3:6] - W
            trav0 = torch.where(issued, d[:, :2].norm(dim=-1) + landing.YAW_M * landing.wrap(d[:, 2]).abs(), trav0)
            a_l = pi.decode_frozen(pi.encode_land(cf, hist[L]), hist[L])
            hist_true = o_true.unsqueeze(1).repeat(1, Hh, 1)  # the motion robots are not scored
            a = torch.clamp(torch.cat((a_l.float(), pi(hist_true[NL:], g[NL:]).float())), -ACTION_CLIP, ACTION_CLIP)
            out = env.step_core(a)
            lt = out["land_terms"]
            ended = (m_before > 0.5) & (env.land_C[:, landing.M] < 0.5) & ~out["done"][L]  # SETTLE ends on this step
            if bool(ended.any()):
                S1 = tuple(x[L] for x in unflatten_state(out["s1"]))
                dp = mdp.sole_kin(S1)[0]["pose"] - pose_reset  # the soles where SETTLE ends against the reset stance
                moved = dp[..., :2].norm(dim=-1).max(-1).values
                turned = landing.wrap(dp[..., 2]).abs().max(-1).values
                settle_rows.append(torch.stack((moved[ended], turned[ended]), 1).cpu())
            dl = lt["deadline"]
            if bool(dl.any()):
                rows.append(torch.stack((lt["e"][dl], lt["a"][dl], count[dl] + 1, trav0[dl]), 1).cpu())
            count = count + dl.float()
            done = out["done"][L]
            count = torch.where(done, torch.zeros_like(count), count)
            falls += int(out["term"][L].sum())
            episodes += int(done.sum())
            env.step_tail_masked(out)
        R, ST = torch.cat(rows), torch.cat(settle_rows)
        return {"err": R[:, 0], "yaw": R[:, 1], "place": R[:, 2].long(), "travel": R[:, 3], "moved": ST[:, 0],
                "turned": ST[:, 1], "falls": falls, "episodes": episodes}  # fmt: skip
