"""GPU check of the landing robots in the training simulator (run on the cluster node).

    pytest tests/test_landing_env.py
"""

import os

import pytest
import torch

LIB = "data/t1/lib_all.npz"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists(LIB), reason="needs CUDA and the packed library"
)


def test_landing_robots_in_the_simulator(tmp_path):
    from robot.kinematics import Kinematics
    from robot.spec import get_robot
    from src import landing
    from src.dr import XiLayout
    from src.env import TrackingSim, make_sim
    from src.mdp import MDP, unflatten_state
    from src.motion import MotionLib

    dev, nv, nl, N = "cuda:0", 8, 40, 64
    names = [s.strip() for s in open("data/t1/keep.txt")
             if s.split("_")[0] in ("idle", "step")][:300]  # idle_turn, step_rotate  # fmt: skip
    (tmp_path / "list.txt").write_text("\n".join(names))
    robot = get_robot("t1")
    kin = Kinematics(robot, dev)
    motion = MotionLib.load(kin, dev, str(tmp_path / "list.txt"), LIB, "fp32")
    motion.merge_tail_bins()
    mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim,
              robot.sensing)  # fmt: skip
    mdp.feet, mdp.contact_z = [robot.tracked_bodies.index(b) for b in robot.feet], robot.contact_z
    motion.set_ref_contact(mdp.feet)
    mdp.enable_velocity(robot)
    motion.set_ref_cmd()
    mdp.enable_landing(robot)
    motion.set_land_labels(mdp.feet)
    env = TrackingSim(make_sim(robot), N, robot, motion, mdp, dev, bin_fix=True, n_vel=nv, n_land=nl)
    L = slice(nv, nv + nl)
    outside = torch.ones(N, dtype=torch.bool, device=dev)
    outside[L] = False
    # a label row's time-out needs a label start whose labelled run ends on an unlabelled frame of its clip within the
    # episode cap; without one the strict count below would fail for lack of opportunity
    kind, cid, lr = motion.land_kind, motion.clip_id, motion.land_label_reset
    edges = int(((kind[:-1] > 0) & (kind[1:] == 0) & (cid[1:] == cid[:-1])).sum())
    ahead = motion.land_ahead[lr].long()
    reach = int(((lr + ahead <= motion.clip_end[cid[lr]]) & (ahead < env.max_episode_steps)).sum())
    assert edges > 0 and reach > 0, f"{edges} labelled frames before an unlabelled one, {reach} label starts reach one"
    seen = {"label_cmd": 0, "free_cmd": 0, "deadline": 0, "label_timeout": 0, "reset": 0}
    with torch.no_grad():
        # 16 s at least, 80 s at most: every event must occur; label time-outs need a robot to get through a step
        for t in range(4000):
            o = env.observe()
            n_before, pose0, fv = env.land_C[:, landing.N].clone(), env.land_pose.clone(), env.land_fv.clone()
            feat = env.land_command(o)
            assert feat.shape == (nl, landing.CMD_DIM) and torch.isfinite(feat).all()
            issued = (env.land_C[:, landing.N] > 0) & (n_before == 0)
            seen["label_cmd"] += int((issued & env.land_lab).sum())
            seen["free_cmd"] += int((issued & ~env.land_lab).sum())
            # every world's PD target is its clip's pose at its frame (a free row: the frame it was reset on), so that
            # episodes last
            a = torch.clamp((motion.q[env.f] - env.default_q) / env.alpha, -20, 20)
            out = env.step_core(a)
            assert not out["land"][outside].any() and not out["lab"][outside].any()
            C, lab, f = out["land"][L], out["lab"][L], out["f"][L]
            S1 = tuple(x[L] for x in unflatten_state(out["s1"]))
            r, C1, pose1, _, tm = mdp.land_step(S1, C, pose0, a[L], out["a_prev"][L], out["xi"][L], fv[:, 0], fv[:, 1],
                                                out["ep_step"][L] >= 2)  # fmt: skip
            assert (torch.allclose(r, out["r"][L], atol=1e-5)
                    and torch.allclose(C1, env.land_C, atol=1e-5))  # the pure function, replayed  # fmt: skip
            assert torch.equal(env.f[L], f + lab.long())  # a free row's frame does not advance
            running = lab & (C[:, landing.N] > 0)
            assert torch.equal(C[running, landing.N], motion.land_n[f[running]])  # a label row's clock is the table's
            gone = lab & (motion.land_kind[env.f[L]] == 0)
            assert out["done"][L][gone].all()  # leaving the labelled frames ends the episode
            seen["deadline"] += int(tm["deadline"].sum())
            seen["label_timeout"] += int(gone.sum())
            done = out["done"][L].clone()
            env.step_tail_masked(out)
            free = ~env.land_lab
            assert ((env.land_wait[free] >= 0) & (env.land_wait[free] <= mdp.land.wait_next[1])).all()
            if bool(done.any()):
                seen["reset"] += int(done.sum())
                assert ((env.land_C[done, landing.M] == mdp.land.settle_steps).all()
                        and (env.land_C[done, landing.N] == 0).all())  # fmt: skip
                fr, lb = env.f[L][done], env.land_lab[done]
                assert (torch.isin(fr[lb], motion.land_label_reset).all()
                        and torch.isin(fr[~lb], motion.land_free_reset).all())  # fmt: skip
                # both anchors on the soles of the state the robot was reset on
                assert torch.equal(env.land_C[done, 6:], env.land_pose[done].flatten(1))
                sole = mdp.sole_kin(tuple(x[L] for x in env.state()))[0]["pose"]
                assert torch.allclose(env.land_pose[done], sole[done], atol=1e-4)
            if t + 1 >= 800 and all(v > 0 for v in seen.values()):
                break
    assert all(v > 0 for v in seen.values()), seen
