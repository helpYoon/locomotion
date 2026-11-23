"""utils/slow_clips.py: a slowed copy keeps the poses and the path and scales the speed by the factor."""

import numpy as np

from utils.slow_clips import mean_speed, slowed, stretch


def test_stretch_and_selection():
    T = 101
    t = np.arange(T) * 0.02
    p = np.stack((0.5 * t, 0.05 * np.sin(6 * t), np.full(T, 0.95)), 1)  # 0.5 m/s with a sway
    yaw = 0.3 * t
    q = np.stack((np.cos(yaw / 2), 0 * t, 0 * t, np.sin(yaw / 2)), 1)
    j = np.stack([np.sin(3 * t + k) for k in range(31)], 1)
    ps, qs, js = stretch(p, q, j, 0.7)
    assert len(ps) == 143  # 100 / 0.7 = 142.9 intervals: frame 142 samples source time 99.4
    assert np.allclose(ps[0], p[0]) and np.allclose(js[0], j[0]) and np.allclose(qs[0], q[0])
    assert np.allclose(ps[10], p[7]) and np.allclose(js[10], j[7]) and np.allclose(qs[10], q[7])  # 10 x 0.7 = frame 7
    mid = 0.5 * (j[3] + j[4])  # frame 5 samples source time 3.5
    assert np.allclose(js[5], mid) and np.allclose(np.linalg.norm(qs, axis=1), 1)
    assert abs(mean_speed(ps) - 0.7 * mean_speed(p)) < 0.01 and abs(mean_speed(p) - 0.5) < 0.02
    assert slowed("walk_slow_forward_001__A005_M", 0.45, 0.3, 0.6)
    assert not slowed("walk_slow_forward_001__A005", 0.65, 0.3, 0.6)  # too fast
    assert not slowed("run_to_walk_001__A005", 0.45, 0.3, 0.6)  # a flight phase
    assert not slowed("idle_turn_360_001__A005", 0.45, 0.3, 0.6)  # not walking
    names = [f"walk_slow_forward_001__A{k:03d}" for k in range(400)]
    half = [slowed(n, 0.45, 0.3, 0.6, 0.5) for n in names]
    assert 160 < sum(half) < 240 and half == [slowed(n + "_M", 0.45, 0.3, 0.6, 0.5) for n in names]  # mirrors follow
