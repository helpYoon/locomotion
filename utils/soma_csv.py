"""soma-retargeter's Booster T1 CSVs in bfm_wm's conventions.

CSV columns: Frame, root_translateX/Y/Z [cm], root_rotateX/Y/Z [deg, extrinsic xyz Euler], then one column per joint
[deg], at 120 fps. soma's T1 URDF has the same frames as ours (Booster's T1_29dof.urdf).
"""

import glob
import os

import numpy as np
import torch

from utils.rotations import axis_angle_to_mat

DATASET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "bones_seed",
                       "soma_uniform", "csv_booster_t1")
FPS = 120.0


def find(paths: str | list[str]) -> list[str]:
    """CSV files and folders (searched recursively) -> CSV paths; raises on a path that doesn't exist."""
    if isinstance(paths, str):  # else its characters are the paths, and "/" globs the whole filesystem
        paths = [paths]
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "**", "*.csv"), recursive=True))
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise FileNotFoundError(f"no such file or folder: {p}")
    return out


def load(path: str) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> joint names, root (Trunk origin) position (T, 3) [m], root rotation (T, 3, 3), joint angles (T, J) [rad].

    Float64, in the CSV's world frame.
    """
    with open(path) as f:
        names = f.readline().strip().split(",")[7:]
    d = torch.from_numpy(np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2))
    rx, ry, rz = torch.deg2rad(d[:, 4:7]).unbind(-1)
    e = torch.eye(3, dtype=d.dtype)
    R = axis_angle_to_mat(e[2], rz) @ axis_angle_to_mat(e[1], ry) @ axis_angle_to_mat(e[0], rx)  # extrinsic x, y, z
    return names, d[:, 1:4] / 100, R, torch.deg2rad(d[:, 7:])
