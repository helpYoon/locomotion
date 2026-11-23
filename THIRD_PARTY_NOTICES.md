# Third-party notices

The training path comes from BlinkMimic 2 (vendored as `blinkmimic2/`, then moved into this layout). These files are
derived from it and are covered by BlinkMimic 2's MIT License and its third-party notices, both reproduced verbatim
below:

- everything under `src/`;
- `robot/spec.py`, `robot/kinematics.py`, `robot/t1.py`, `robot/t2.py`;
- `utils/t2_run.sh` and `utils/prepare_t2.py` (from `utils/t1_run.sh` and `utils/prepare_t1.py`);
- `utils/rotations.py`, `utils/avg_ckpt.py`, `utils/t1_run.sh`, `utils/prepare_t1.py`, `utils/record_clips.py`,
  `utils/render_gif.py`, `utils/export_onnx.py`, `utils/velocity_rig.py`, `utils/play_velocity.py`,
  `utils/landing_score.py`, `utils/landing_rig.py`, `utils/landing_test.py`, `utils/t1_land_run.sh`;
- `tests/test_velocity.py`, `tests/test_optim.py`, `tests/test_landing.py`, `tests/test_landing_env.py`.

In BlinkMimic 2's notices, `src/blinkmimic2/` and `tools/` are where these files were before the move, and the parts
of `tools/prepare_bones.py` still used are in `utils/prepare_t1.py`.

The robot models come from Booster Robotics and holosoma, under their own licences:

- **booster_assets (BSD-3-Clause, Copyright (c) 2025, BoosterRobotics):** the T1's URDF and meshes in
  `assets/t1/t1_description/` (licence: `assets/t1/LICENSE`, source: `assets/t1/SOURCE.txt`) and the T2's MJCF and
  meshes in `assets/t2/t2_description/` (licence: `assets/t2/LICENSE`, source: `assets/t2/SOURCE.txt`). The MJCF files
  are built from Booster's files by `utils/build_t1_mjcf.py` and `utils/build_t2_mjcf.py`.
- **holosoma (Apache-2.0, Copyright Amazon.com, Inc. or its affiliates):** `assets/t1/config/sysid.txt`, holosoma's
  23-DoF T1 robot configuration (joint friction, damping, armature), copied from an earlier version of
  [amazon-far/holosoma](https://github.com/amazon-far/holosoma) and not changed in this repository (licence:
  `assets/t1/config/LICENSE-holosoma`, notice: `assets/t1/config/NOTICE-holosoma`).

The trained policy `deploy/t2al1_vel_s1_013000.onnx` is derived from the BONES-SEED motion data (Motion Data by Bones
Studio, https://bones.studio/): it is subject to the BONES-SEED License, with credit to Bones Studio, and is not
covered by this repository's MIT licence. The data are not included here.

Everything else in this repository is under its MIT licence (`LICENSE`).

## BlinkMimic 2: LICENSE

```
MIT License

Copyright (c) 2026 Jiayu Li
Portions Copyright (c) 2026 Fuyi Chen (BlinkMimic), Copyright (c) 2026 Huy-Quang-Dao and Copyright (c) 2026 ETH Zurich, Chong Zhang
(see THIRD_PARTY_NOTICES.md)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## BlinkMimic 2: THIRD_PARTY_NOTICES.md

# Third-party notices

Portions of `src/blinkmimic2/` and `tools/` come from [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)
(Chong Zhang) and [InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack) (Huy-Quang-Dao), by way of
[BlinkMimic](https://github.com/fuyichen0725/blinkmimic) (Fuyi Chen), used under their MIT Licenses, reproduced below. The
BONES-SEED keyword filter in `tools/prepare_bones.py` follows
[SuperMimic-Rep](https://github.com/LaoYang1994/SuperMimic-Rep) (MIT).

## BlinkMimic

```
MIT License

Copyright (c) 2026 Fuyi Chen
Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## InstantTrack

```
MIT License

Copyright (c) 2026 Huy-Quang-Dao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## supermimic_vibed

```
MIT License

Copyright (c) 2026 ETH Zurich, Chong Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Dependencies and data

The Python dependencies (mjlab, MuJoCo, MuJoCo Warp, NVIDIA Warp, PyTorch, NumPy, Pillow) are installed from their own
distributions under their own licenses (Apache-2.0, BSD-3-Clause and others) and are not redistributed here.

Training data includes Motion Data by Bones Studio ([https://bones.studio/](https://bones.studio/)). Use of the
underlying dataset is subject to the BONES Motion Capture Dataset License Agreement. The data are not included here;
`tools/prepare_bones.py` converts a copy that you download yourself after accepting that license. Files derived from
the dataset (rendered animations, trained policies) are not covered by this repository's MIT License.
