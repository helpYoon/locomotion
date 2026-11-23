"""CPU self-check of utils/native_limp.py's step pairing on synthetic two-foot walks (no simulator).

    CUDA_VISIBLE_DEVICES= pytest tests/test_native_limp.py
"""

import numpy as np

from utils.native_limp import DT, evenness, walk_steps


def synthetic(times, leads, swing=10, n0=20, tail=5):
    """Feet positions (T, 2, 2) heading +x: step i by foot i % 2, a swing of ``swing`` control steps at 0.5 m/s ending
    at its touchdown, times[i] after the previous touchdown, landing leads[i] ahead of the other foot."""
    taus = [n0 + swing]
    for t in times[1:]:
        taus.append(taus[-1] + round(t / DT))
    T = taus[-1] + tail
    p = np.zeros((T, 2, 2))
    p[:, 0, 1], p[:, 1, 1] = 0.1, -0.1
    for i, (tau, d) in enumerate(zip(taus, leads)):
        k = i % 2
        start, end = p[tau - swing, k, 0], p[tau - swing, 1 - k, 0] + d
        p[tau - swing : tau, k, 0] = np.linspace(start, end, swing, endpoint=False)
        p[tau:, k, 0] = end
    return p


def test_even_walk_is_even():
    p = synthetic([0.6] * 9, [0.3] * 9)
    s = walk_steps(p, np.zeros(len(p)), np.array([1.0, 0.0]))
    assert len(s) == 8 and {f for f, *_ in s} == {0, 1}
    e = evenness(s)
    assert e["si_t"] < 1e-9 and e["si_l"] < 1e-9 and e["si_h"] < 1e-9


def test_length_limp():
    p = synthetic([0.6] * 9, [0.3, 0.2] * 4 + [0.3])
    e = evenness(walk_steps(p, np.zeros(len(p)), np.array([1.0, 0.0])))
    assert abs(e["si_l"] - 0.1 / 0.25) < 1e-6 and abs(e["si_h"] - 0.1 / 0.25) < 1e-6 and e["si_t"] < 1e-9


def test_time_limp():
    p = synthetic([0.6] + [0.4, 0.8] * 4, [0.3] * 9)  # the right foot lands 0.4 s after the left, the left 0.8 s after
    e = evenness(walk_steps(p, np.zeros(len(p)), np.array([1.0, 0.0])))
    assert e["t"] == (0.8, 0.4) and abs(e["si_t"] - 0.4 / 0.6) < 1e-6
