"""CPU check of the simulator's gait memory (src/env.py, train.py --sym_weight): an existing T2 velocity policy walks at
0.4 m/s in the training simulator; the memory carries from step to step, steps are scored, the simulator and
imagination's code (MDP.gait_rollout) give the same memories and penalties, and a command resample restarts the step
comparison. Skipped where the checkpoint is missing (logs/ is git-ignored). About a minute.

    CUDA_VISIBLE_DEVICES= pytest tests/test_gait_sym_env.py
"""

import dataclasses
import os

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO, "logs/t2al1_vel_s1/policy/ckpt_013000.pt")
STEPS, RESAMPLE_AT = 450, 300
CMD = (0.4, 0.0, 0.0)
LAST, VALID = 3, 4

pytestmark = pytest.mark.skipif(not os.path.exists(CKPT), reason="no t2al1_vel_s1 checkpoint")


@pytest.fixture(scope="module")
def run():
    torch.set_num_threads(4)
    from src.env import TrackingSim, make_sim
    from utils.velocity_rig import Rig

    rig = Rig(CKPT, 1, dev="cpu")
    rig.env.motion.set_ref_cmd()  # the velocity worlds read it (unused here: every command is held)
    mdp, pi = rig.mdp, rig.pi
    up = torch.ones(1, dtype=torch.bool)
    c = torch.tensor([CMD])

    def make():
        env = TrackingSim(make_sim(rig.robot), 1, rig.robot, rig.env.motion, mdp, "cpu", eval_mode=True, n_vel=1)
        env.reset(torch.arange(1), torch.zeros(1, dtype=torch.long))
        env.cmd_held.copy_(c), env.cmd_ref.fill_(False)
        return env

    def step(env, hist):
        o = mdp.obs(env.state(), env.a_prev, env.xi, env.f, up)
        hist = o.unsqueeze(1).repeat(1, rig.hist_len, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        with torch.no_grad():
            a = pi.decode_frozen(pi.encode_vel(mdp.normalize_cmd(c), hist), hist).float().clamp(-20, 20)
            out = env.step(a)
        env.f = torch.zeros(1, dtype=torch.long)  # stay on the stand clip's first frame, as Rig.apply
        return out, hist

    mdp.vel = dataclasses.replace(mdp.vel, ref_cmd_prob=0.0, resample_steps=10**6, w_sym=0.0)
    off = make()
    out_off, hist = step(off, None)
    mdp.vel = dataclasses.replace(mdp.vel, w_sym=-20.0)
    env = make()
    rec, hist = [], None
    for t in range(STEPS):
        out, hist = step(env, hist)
        rec.append({"gait": out["gait"].clone(), "s1": out["s1"].clone(), "sym": out["vel_terms"]["sym"].clone(),
                    "sym_n": out["vel_terms"]["sym_n"].clone(), "after": env.gait.clone(), "ep": out["ep_step"][:1].clone()})
        if t == RESAMPLE_AT:
            env._resample_cmd(torch.ones(1, dtype=torch.bool))
            env.cmd_held.copy_(c), env.cmd_ref.fill_(False)
    return {"off": off, "out_off": out_off, "env": env, "rec": rec, "mdp": mdp}


def test_memory_chain_and_scoring(run):
    rec = run["rec"]
    for t in range(RESAMPLE_AT):
        assert torch.equal(rec[t + 1]["gait"], rec[t]["after"])
    # ~1.3 s to the first touchdown, then ~1.1 s steps (lr.py at 0.4 m/s), the first two counted steps unscored
    assert sum(float(r["sym_n"].sum()) for r in rec[:RESAMPLE_AT]) >= 3


def test_sim_and_imagination_agree(run):
    from src.mdp import unflatten_state

    mdp, rec = run["mdp"], run["rec"][: RESAMPLE_AT + 1]
    S = unflatten_state(torch.cat([r["s1"] for r in rec]))
    foot_v, foot_xy = mdp.foot_kin(S)[1:]
    gate = mdp.gait_gate(torch.tensor([CMD]), torch.zeros(1, dtype=torch.bool))
    settled = torch.stack([r["ep"] for r in rec]) >= mdp.vel.sym_settle
    assert not settled[0].any() and settled[-1].all()  # the run crosses the settle step
    pen, scored, mems = mdp.gait_rollout(rec[0]["gait"], foot_v.unsqueeze(1), foot_xy.unsqueeze(1), S[1].unsqueeze(1), gate,
                                         settled)  # fmt: skip
    for h, r in enumerate(rec):
        assert torch.allclose(mems[h], r["gait"], atol=1e-6), h
        assert torch.allclose(pen[h], r["sym"], atol=1e-6) and torch.equal(scored[h].float(), r["sym_n"]), h
    assert torch.allclose(mems[-1], rec[-1]["after"], atol=1e-6)


def test_resample_mid_walk(run):
    rec = run["rec"]
    g = rec[RESAMPLE_AT + 1]["gait"]
    assert float(g[0, LAST]) == 0 and float(g[0, VALID]) == 0
    counted = [t for t in range(RESAMPLE_AT + 1, len(rec)) if float(rec[t]["after"][0, 0]) == 0.0]
    assert len(counted) >= 3
    assert [float(rec[t]["sym_n"][0]) for t in counted[:3]] == [0.0, 0.0, 1.0]


def test_off_has_no_memory(run):
    assert not hasattr(run["off"], "gait")
    assert "gait" not in run["out_off"] and "sym" not in run["out_off"]["vel_terms"]
