"""A --vel_frac checkpoint's velocity interface in the training simulator, for utils/play_velocity.py and
utils/export_onnx.py: evaluation conditions (no randomization, noise or pushes, the fixed evaluation delay), every
world starting from the robot standing in its default pose."""

import torch
import warp as wp

wp.config.log_level = wp.LOG_WARNING  # no start-up banner or per-kernel load lines

import src.networks as networks
from src.dr import XiLayout
from src.env import TrackingSim, make_sim
from robot.kinematics import Kinematics
from src.mdp import MDP
from src.motion import MotionLib
from src.networks import Policy, policy_kwargs
from robot.spec import check_robot_numbers, get_robot
from utils.rotations import mv3, yaw_mat

# (duration [s], command (vx, vy) [m/s] in the heading frame and yaw rate [rad/s])
SCRIPT = [(2, (0, 0, 0)), (4, (0.5, 0, 0)), (4, (1.0, 0, 0)), (4, (-0.5, 0, 0)), (4, (0, 0.4, 0)), (4, (0, 0, 1.0)),
          (4, (0.6, 0, -1.0)), (3, (0, 0, 0))]  # fmt: skip
ACTION_CLIP = 20.0  # the clamp of train.py's collection step (_collect_core): training never applied a larger action


class Rig:
    def __init__(self, ckpt: str, worlds: int = 1, dev: str = "cuda:0"):
        torch.backends.cuda.matmul.allow_tf32 = True
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        a_ = self.args = ck["args"]
        assert a_.get("vel_frac", 0) > 0, "not a --vel_frac checkpoint"
        self.iteration, self.dev, self.N, self.hist_len = ck["iteration"], dev, worlds, a_["hist"]
        robot = self.robot = get_robot(a_["robot"])
        check_robot_numbers(robot, a_.get("robot_numbers"))  # a checkpoint runs on the numbers it was trained with
        J = robot.num_joints
        kin = Kinematics(robot, dev)
        # the library is one clip of the robot standing in its default pose: the start state. Nothing else of it is used
        T = 100
        quat = torch.tensor([1.0, 0, 0, 0]).repeat(T, 1)
        stand = (torch.tensor(robot.robot_cfg().init_state.pos).repeat(T, 1), quat, torch.tensor(robot.default_joint_pos_list()).repeat(T, 1))
        motion = MotionLib.from_clips(["stand"], *([x] for x in stand), kin, dev)
        mdp = self.mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
        mdp.feet, mdp.contact_z = [robot.tracked_bodies.index(b) for b in robot.feet], robot.contact_z
        mdp.enable_velocity(robot)
        widths = lambda key: tuple(int(x) for x in a_.get(key, "2048,1024,512,256").split(","))  # noqa: E731
        pi = self.pi = Policy(mdp.obs_dim, motion.goal_dim(a_["goal_frames"]), J, hist=self.hist_len, enc_hidden=widths("enc_hidden"),
                              dec_hidden=widths("dec_hidden"), **policy_kwargs(a_, robot)).to(dev)  # fmt: skip
        pi.load_state_dict(ck["pi"])
        pi.eval()
        networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
        self.env = TrackingSim(make_sim(robot), worlds, robot, motion, mdp, dev, eval_mode=True)
        self.ids, self.up = torch.arange(worlds, device=dev), torch.ones(worlds, dtype=torch.bool, device=dev)
        self.f0 = torch.zeros(worlds, dtype=torch.long, device=dev)  # every world stays on the stand clip's first frame
        self.hist = None

    def reset(self):
        self.env.reset(self.ids, self.f0.clone())
        self.hist = None

    @torch.no_grad()
    def observe(self) -> torch.Tensor:
        """A velocity row's observation of the current state (no clip reference), pushed onto the history (N, hist, obs)."""
        env = self.env
        o = self.mdp.obs(env.state(), env.a_prev, env.xi, env.f, self.up)
        self.hist = o.unsqueeze(1).repeat(1, self.hist_len, 1) if self.hist is None else torch.cat((self.hist[:, 1:], o.unsqueeze(1)), 1)
        return self.hist

    @torch.no_grad()
    def act(self, cmd) -> torch.Tensor:
        """The policy's action (N, J) on the current history under command cmd, clamped as training's."""
        c = torch.tensor(cmd, dtype=torch.float32, device=self.dev).expand(self.N, 3)
        a = self.pi.decode_frozen(self.pi.encode_vel(self.mdp.normalize_cmd(c), self.hist), self.hist)
        return torch.clamp(a.float(), -ACTION_CLIP, ACTION_CLIP)

    @torch.no_grad()
    def apply(self, a: torch.Tensor) -> tuple[list[float], bool]:
        """One control step with action a; world 0's velocity (vx, vy in its heading frame, yaw rate) and whether it
        has fallen."""
        self.env.step(a)
        self.env.f = self.f0.clone()
        S = self.env.state()
        v = mv3(yaw_mat(S[1]).transpose(-1, -2), S[2])
        return [float(v[0, 0]), float(v[0, 1]), float(S[3][0, 2])], bool(self.mdp.fall_termination(S)[0])

    def step(self, cmd) -> tuple[list[float], bool]:
        """One control step under command cmd."""
        self.observe()
        return self.apply(self.act(cmd))
