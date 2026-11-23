# locomotion

Developed under the name `bfm_wm`, which the package name and some code comments still use.

[BlinkMimic 2](https://github.com/jyl681/blinkmimic2) on the **Booster T1** humanoid (29 DoF, 7-DoF arms). The
learner is [SuperMimic](https://supermimic.github.io)'s: a world model is fit to data from 4096 simulated robots, and
the policy learns by backpropagating through short rollouts imagined in it. BlinkMimic 2 adds a schedule for a short
run (exams that play whole clips, practice where they fail, a deployment stream inside the world model, a learning
rate planned for the run's length), so its motion-tracking recipe trains in about 15 minutes on one RTX 4090. This
port trains one policy with two input modes that share one decoder:

- **motion tracking**: follow a full-body reference clip;
- **velocity commands**: walk at a commanded forward, sideways and turning speed (vx, vy, yaw rate).

The velocity mode is the one that runs on the robot. Training simulates with
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo-Warp), and a standalone script runs an exported policy in native
MuJoCo. The same code also trains the [Booster T2](#booster-t2), the robot it runs on: the policy in
[deploy/](deploy) is the T2's, with the task that runs it through Booster's
[booster_deploy](https://github.com/BoosterRobotics/booster_deploy) ([Deploy on the robot](#deploy-on-the-robot)). A
T2 velocity policy trained with this code has run on the robot through that task; the file now in deploy/ has not yet.

An earlier SuperMimic implementation (Isaac Lab 3 with the Newton backend: its training, evaluation, equivalence
checks, clip viewer and the MuJoCo export mode) came before this code; it is not part of this repository.

## Requirements

- Linux x86_64 with an NVIDIA GPU and a driver for CUDA 12.8 (the torch wheels are `cu128`). `uv.lock` also holds the
  aarch64 wheels, and runs have trained on TACC Horizon's GB200 nodes.
- GPU memory: the T1 recipe's run logged a peak of 23,466 MiB on a 24 GB RTX 4090, about 2.4 GiB of it already
  used by the desktop's display before the run started (the peak is the whole driver run's, evaluations included).
  [utils/t1_run.sh](utils/t1_run.sh) sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, without which a
  full-size run runs out of memory on 24 GB.
- [uv](https://docs.astral.sh/uv/). It installs Python 3.13 and every dependency (mjlab 1.6.0, MuJoCo 3.11.0, Warp
  1.16.0, torch 2.9.0).
- Disk: about 8 GB for the Python environment, 4 GB for the retargeted clips (plus 23 GB if you keep the BONES-SEED
  source files), 1.2 GB for the packed library, and per run about 38 MB for each periodic checkpoint, 85 MB for the
  final and the averaged ones and 150 MB for `last.pt` (0.8 GB for the T1 recipe's run).

## Setup

```bash
git clone https://github.com/helpYoon/locomotion.git
cd locomotion
uv sync
```

Run every command in this README from the repository root. The scripts read `data/` and `dataset/` and write `logs/`
relative to it, and a run that utils/t1_run.sh starts stores its data paths in `args.json` relative to it
(`data/t1/lib_all.npz`; `python -m src.train` alone stores full paths). `uv sync` also
installs pytest. The commands below use `uv run`; `.venv/bin/python` is the same environment, and
[utils/t1_run.sh](utils/t1_run.sh) calls it directly.

### Motion data

Training reads the [BONES-SEED](https://bones.studio/datasets/seed) motion-capture dataset retargeted to the T1 and
packed into one library, `data/t1/lib_all.npz`, with its clip list `data/t1/keep.txt`. The retargeted clips are one
CSV per clip under `dataset/bones_seed/soma_uniform/csv_booster_t1/` (11,950 clips, 4 GB; the format is in
[utils/soma_csv.py](utils/soma_csv.py)). The data is not in this repository and is under its owner's terms.

1. Put BONES-SEED's `soma_uniform` BVH files, in their date folders, under `dataset/bones_seed/soma_uniform/bvh/`.
2. Clone [NVIDIA soma-retargeter](https://github.com/NVIDIA/soma-retargeter), check out commit `cbdb2b4` and set it
   up as its README says. Then retarget:

   ```bash
   bash utils/retarget_bones_seed_t1.sh ~/soma-retargeter
   ```

   It uses soma-retargeter's Newton retargeter and its bundled `booster_t1` robot, with each joint range narrowed to
   this repository's MJCF so the clips stay inside the training model's limits. That robot is built into
   soma-retargeter's `local-robots/` by [utils/make_booster_t1.py](utils/make_booster_t1.py), which the script runs
   first. The script converts one date folder at a time and skips folders that are already done, so you can rerun it
   after an interruption.
3. Filter out scenes with props, falls and kinematically infeasible clips ([utils/filter_clips.py](utils/filter_clips.py),
   on the GPU) into a list of clips:

   ```bash
   uv run python -m utils.filter_clips --out dataset/clips_filtered.txt
   ```

   It keeps 11,900 of the 11,950 clips. The list holds absolute paths, so build it on each machine.
4. Optionally, edit the clips into a new folder ([utils/edit_t1_motions.py](utils/edit_t1_motions.py)): the
   elbows, wrists and hand roll held at the default pose (the shoulders keep the retargeted swing), the head turned
   toward the direction of travel, and the feet turned in so they point parallel when walking. Then filter that folder
   into its own list:

   ```bash
   uv run python utils/edit_t1_motions.py
   uv run python -m utils.filter_clips $PWD/dataset/bones_seed/soma_uniform/csv_booster_t1_edited \
       --out dataset/clips_filtered_edited.txt
   ```

   The edit takes about 4 minutes on 30 cores and writes `csv_booster_t1_edited/` next to the source, with each clip's
   toe-out and IK result in its `edits.json`. The T1 policy trained on `dataset/clips_filtered_edited_v2.txt`:
   these edited clips with each actor's shoulder roll also turned in, a step that is not in `utils/`.
5. Pack a list into the library that training and evaluation read ([utils/prepare_t1.py](utils/prepare_t1.py)). The
   T1 run's library:

   ```bash
   uv run python -m utils.prepare_t1 dataset/clips_filtered_edited_v2.txt data/t1
   ```

   Steps 1 to 4 make `dataset/clips_filtered.txt` and `dataset/clips_filtered_edited.txt`; pack either the same way.

   It resamples the 120 fps clips to 50 Hz and writes `lib_all.npz`, `keep.txt`, `manifest.csv` and the shards it
   packs from into `data/t1/`.

[utils/plot_walking_profile.py](utils/plot_walking_profile.py) plots how much reference motion the clips hold per
velocity-command cell (`uv run python utils/plot_walking_profile.py [CSV or folder ...] --out profile.png`). The MJCF,
`assets/t1/t1_description/mjcf/t1.xml`, is built from Booster's URDF and `assets/t1/config/sysid.txt` by
[utils/build_t1_mjcf.py](utils/build_t1_mjcf.py) (`uv run python utils/build_t1_mjcf.py`); it is tracked, so rerun it
only after changing either source. The tracked file is MuJoCo 3.12's output: under this environment's MuJoCo 3.11 a
rebuild rewrites its text (attribute order and explicitly written defaults) while the compiled model stays the same,
so build it under MuJoCo 3.12 or do not commit that text-only diff.

### Check the install

```bash
uv run pytest
```

The tests run on the CPU and need neither the motion data nor `data/`. The one CUDA test (the capturable optimizer in
a CUDA graph, [tests/test_optim.py](tests/test_optim.py)) skips itself without a GPU.

## Train

```bash
bash utils/t1_run.sh my_run
```

This trains the T1 recipe (`logs/t1_vel_ankle_sharp_x2_s1`, seed 1), which is
[src/train.py](src/train.py)'s defaults: BlinkMimic 2's schedule at twice its length, 13,000 iterations, 32 minutes of
training on an RTX 4090, then about 2.5 minutes per evaluation pass. Train on an otherwise idle GPU: the training time
depends on it. [utils/t1_run.sh](utils/t1_run.sh):

- **deletes `logs/<run>`**, then trains src/train.py's defaults plus the given flags on `data/t1/`;
- `STRETCH` (default 2) sets the length: BlinkMimic 2's 6,500-iteration schedule times `STRETCH`, with every schedule
  point (the phases, the learning-rate decay, the checkpoints averaged and evaluated) moved with it; `STRETCH=1` is
  BlinkMimic 2's own length. `SEED=2` trains another seed (the default is 1);
- averages the last five checkpoints into `logs/<run>/policy/ckpt_XXXXXX.pt` ([utils/avg_ckpt.py](utils/avg_ckpt.py)),
  the policy to evaluate and export;
- evaluates it on every clip at the evaluation delay (lag 5) and at random delays, and checkpoints 1000 to 6000 (times
  `STRETCH`) at lag 5, and writes the training time and peak GPU memory to `logs/<run>/time.txt`.

Besides the length, four defaults make it the T1 recipe:

- `--vel_frac 0.5` puts half of the simulated robots, and half of each imagined batch, on velocity commands, through
  their own encoder and critic with the decoder frozen for them. It needs `--no-pi_ref_skip`, which stops building the
  actions on top of the reference pose (a velocity command has none).
- `--ankle_weight -4` penalizes the torque of the two ankle pitch motors at a standing command, so the policy does not
  stand leaning forward with its weight on the ankles.
- `--track_sharp 0.75` pays 75% of each command-tracking term through a kernel four times narrower.

To train motion tracking alone, BlinkMimic 2's own recipe (about 15 minutes), turn them off at its length:

```bash
STRETCH=1 bash utils/t1_run.sh my_run --pi_ref_skip --vel_frac 0 --ankle_weight 0 --track_sharp 0
```

`uv run python -m src.train --help` lists every option, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -m src.train --log_dir logs/<run> [flags]` trains the
same defaults without the driver's averaging and evaluations.

## Evaluate

```bash
uv run python -m src.evaluate --run_dir logs/my_run/policy --bones_list data/t1/keep.txt --tag full
```

Plays every clip once from its first frame in the evaluation-mode simulator (no randomization, no reset noise) with
each checkpoint's deployed policy, and writes `eval_<tag>.csv` and `ckpt_*_eval_<tag>.json` (per-clip returns, failed
clips and the frame each fails at) into the run directory. `norm_return` is the tracking return normalized by the clip
length; a failure is a termination before the clip ends. `--eval_delay random` draws the actuator delays per episode,
and `--ckpts` and `--group` pick checkpoints and evaluate them side by side. The T1 policy scores 0.9699 at lag 5
on the 11,900 clips, with no failed clip.

The velocity mode runs in the training simulator under evaluation conditions, driven from the keyboard in the MuJoCo
viewer (needs a display; the keys are those of the native-MuJoCo viewer below) or by a command script that prints
commanded against achieved velocity:

```bash
uv run python -m utils.play_velocity --ckpt logs/my_run/policy/ckpt_013000.pt
uv run python -m utils.play_velocity --ckpt logs/my_run/policy/ckpt_013000.pt --script
```

To record evaluation rollouts of a few clips and render them as a grid GIF or MP4 (`.gitignore` covers `.npz`
recordings at the repository root):

```bash
uv run python -m utils.record_clips --ckpt logs/my_run/policy/ckpt_013000.pt --clips walk_180_R_001__A326 --out traj.npz
MUJOCO_GL=egl uv run python -m utils.render_gif traj.npz logs/walk.gif --clip walk_180_R_001__A326
```

## Export to ONNX

```bash
uv run python -m utils.export_onnx --ckpt logs/my_run/policy/ckpt_013000.pt
```

Writes `ckpt_013000.onnx` next to the checkpoint: the velocity interface only (velocity encoder and decoder, about
20 MB), with the observation history (1, 10, 93) and the command (1, 3) in and the action (1, 29) out, all float32. The
model's metadata lists the observation terms, joint order, default pose, action scale and gains a robot-side runner
needs. The file is written only once it reproduces the checkpoint: the export drives the training simulator through a
command script with the ONNX model under onnxruntime and compares each action with PyTorch's, so it needs a GPU, or
`--device cpu` (with `CUDA_VISIBLE_DEVICES=`) to run without one. A T2 export has (1, 10, 99) and (1, 31).

## Run a policy in native MuJoCo

[utils/play_mujoco.py](utils/play_mujoco.py) runs an ONNX export in native MuJoCo, on the training model of the old
SuperMimic simulator (`assets/t1/mujoco/t1_newton.mjb`) with its actuators reimplemented from
`assets/t1/mujoco/t1_actuators.npz` (gains, torque limits, delays, action scale; [robot/t1.py](robot/t1.py) reads the
same table). Both files were written by the earlier SuperMimic code's export mode. The script is standalone: it imports
nothing from this repository, and `uv run` builds its own environment on first use from its header and
`utils/play_mujoco.py.lock` (CPU torch, onnxruntime, and MuJoCo 3.12.0, the only version that loads the `.mjb`).

| Mode | What it does |
|---|---|
| `viewer` | Drive the robot from the keyboard in MuJoCo's viewer (needs a display): up/down vx ±0.1 m/s, a/d vy ±0.1 m/s, left/right yaw rate ±0.2 rad/s, space stops, backspace resets. It also resets itself after a fall. |
| `video` | A scripted command sequence from standing, rendered offscreen to `logs/mujoco/walk_<iteration>_*.mp4` with the velocities printed on each frame, plus a `.png` plot of commanded against actual velocity and a `.npy` log. Without a display, set `MUJOCO_GL=egl`. |
| `steptest` | The step test: from standing, each of 672 commands on a grid over the command range is held for 5 s, scored by the relative steady-state error over the last 2 s; on 12 worker processes (`--jobs`). Writes `logs/mujoco/steptest_<iteration>_*.json`, named by iteration, so a second policy of the same iteration overwrites it. |
| `landtest` | Rollouts of a landing export with one or two foot-landing commands each, from the default stance or from where a walk and a stop under a velocity export (`--walk_onnx`) leave the robot. Started by [utils/landing_native.py](utils/landing_native.py) (see "Landing command"), not by hand. |
| `landvideo` | One rollout of the hand-over rendered to `logs/mujoco/land_walk_<iteration>_*.mp4`: 3 s of walking at 0.5 m/s and a 3 s stop under a velocity export (`--walk_onnx`), then the landing export (`--onnx`) takes over, moves the left foot 20 cm forward and brings the right foot beside it. Without a display, set `MUJOCO_GL=egl`. |

```bash
uv run utils/play_mujoco.py viewer --onnx logs/land_r4_d10_al2_s3/ckpt_004000_both.onnx
uv run utils/play_mujoco.py steptest --onnx logs/land_r4_d10_al2_s3/ckpt_004000_both.onnx
MUJOCO_GL=egl uv run utils/play_mujoco.py video --onnx logs/land_r4_d10_al2_s3/ckpt_004000_both.onnx
```

`logs/land_r4_d10_al2_s3/ckpt_004000_both.onnx` is the T1's policy (a local file, not published: `logs/` is not in the repository, and
[deploy/](deploy) holds the T2's): the averaged policy of the recipe under
"Train" with the landing command added to it (see "Landing command"), exported as one file with
`utils/export_landing.py --both`. Its inputs are `obs_hist`, `mode`, `vel_cmd`, `land_cmd` and `state`; the three
modes above run its velocity interface (`mode` 0), which is the earlier velocity-only file's
(`t1_vel_ankle_sharp_x2_s1_013000.onnx`, not published): the walk, stop and step test gives the same trajectories
with either. In this step test: median error 0.049 and mean 0.079,
no falls. At a zero command it is still within 1.5 s of a reset or a stop and holds 2.4 N m on the two ankle pitch
motors. Earlier T1 policies, not published: `t1_vel_ankle_x2_s1_013000.onnx` (the same recipe without
`--track_sharp`; median error 0.053, 8.9 N m at rest) and
`t1_vel_rtx_edited_v2_align05_run1_100000.onnx` (trained with the SuperMimic code; median error 0.0245, still within
3 s in 3 of 12 stand trials, 13.2 N m).

### The T2's policy in plain MuJoCo

`deploy/t2al1_vel_s1_013000.onnx` is the policy for the robot: the averaged policy of `--robot t2_measured` under the
recipe of [The measured T2](#the-measured-t2) plus `--vel_limit_weight -5 --lean_weight -10 --tilt_rate_weight -0.125
--align_weight 1`, on `data/t2_slow` (the T2 library plus slowed copies of its slow walking clips,
`utils/slow_clips.py`), seed 1 of four, exported with `utils/export_onnx.py`. Of the eight runs of this recipe with and
without the step-symmetry term (`--sym_weight`), it steps the most evenly left to right. It is velocity
only: the landing command was never trained on the T2. [utils/play_native.py](utils/play_native.py) runs it in plain
MuJoCo on the model the training simulator compiles, with its own observation and motor code and no window unless
asked:

```bash
CUDA_VISIBLE_DEVICES= uv run python -m utils.play_native --onnx deploy/t2al1_vel_s1_013000.onnx              # keyboard viewer
CUDA_VISIBLE_DEVICES= uv run python -m utils.play_native --onnx deploy/t2al1_vel_s1_013000.onnx --script     # a scripted drive
CUDA_VISIBLE_DEVICES= uv run python -m utils.play_native --onnx deploy/t2al1_vel_s1_013000.onnx --steptest   # about 6 minutes
CUDA_VISIBLE_DEVICES= uv run python -m utils.play_native --onnx deploy/t2al1_vel_s1_013000.onnx --stand --start handover
CUDA_VISIBLE_DEVICES= uv run python -m utils.native_limp --onnx deploy/t2al1_vel_s1_013000.onnx              # left/right step evenness
```

| Option | What it does |
|---|---|
| `--steptest` | The step test above on the T2's command box: 672 commands (15 values per axis, the one nearest zero set to zero), each held for 5 s from standing and scored over the last 2 s, with 4 draws (`--trials`) of the five actuator groups' lags (0-40 ms) per command. One process. Writes `logs/native/steptest_<robot>_<run>_<variant>.json`. |
| `--stand` | The ten stop cases of `utils/check_velocity.py` (3 s at a command, then zero), each with its own lag draw: a fall, seconds until still for good after the start and after the stop, shaking (direction reversals of a waist or ankle joint over the last 2 s), the ankle torques at rest, setpoints beyond the joint ranges. `--start handover` starts as the robot will: 2 s held at the default pose under Booster's prepare gains with no policy, then the policy, whose first previous action is the one that holds that pose (`--prev_action zeros` for the other choice). It ends with the exit hand-over: after 5 s of standing the last setpoint is frozen for 2 s, as the motors hold it until Booster's walking controller takes over. Writes `logs/native/stand_*.json`. |
| `--variant` | One difference from training, for `--script`, `--steptest` and `--stand`: `gains_x0.8` / `gains_x1.2` (every kp and kd; training randomized ±10%), `latency` (the sensors one control step old), `push` (a 0.5 m/s kick every 2 s while standing), `no_friction` (no joint friction), and the ways the robot's firmware might apply the gains on the two-motor waist and ankle linkages, which is not documented: `H1f1` (kd per motor at Booster's rule, not the measured 0.69), `H2` (kp per motor too: those joints 1.8-3.4 times stiffer, pitch and roll coupled), `H3` (kd at the joint), `H4` (kp per motor, kd at the joint); `_swapped` is the other motor order. |
| `--delay N` | One lag of N physics steps (5 ms) for every joint in the viewer and `--script` (default 4, the evaluation delay). |

On this file (2026-11-23), as trained, with random lags: no fall in the step test's 2,688 rollouts, median error 0.070
(forward 0.03, backward 0.05, sideways 0.14, turning 0.03); no fall in the scripted drive; the stand check passes from
the standing reset and from the hand-over: still within 2.0 s of every stop and 1.5 s of the start, no shaking, about
11.3 N m held at each ankle pitch at rest, setpoints past a joint's range in at most 12% of steps and by at most
0.14 rad. On the frozen setpoint the trunk tilts 1.0-1.2 degrees in 2 s. Walking straight ahead from 0.4 m/s up, its
left and right steps differ by at most 6% in time and length (`utils/native_limp.py`, 8 trials of 10 s per speed); at
0.4 m/s it takes steps of about 1 s and 30 cm and walks about 0.27 m/s. The variants (step test with 4 lag draws per
command, stand check from the hand-over):

| Variant | Step test: falls of 2,688, median error | Stand check: falls of 10 | Slowest settle after a stop | Shaking | Most torque asked of a linkage motor (limit 60 N m) |
|---|---|---|---|---|---|
| as trained | 0, 0.070 | 0 | 2.0 s | none | |
| `gains_x0.8` | 0, 0.092 | 0 | 2.0 s | none | |
| `gains_x1.2` | 0, 0.070 | 0 | 1.5 s | none | |
| `latency` | 0, 0.080 | 0 | 2.5 s | none | |
| `push` | not run | 0 | never still (kicks every 2 s) | | |
| `no_friction` | 0, 0.056 | 0 | 2.0 s | none | |
| `H1f1` | 0, 0.078 | 0 | 2.0 s | none | 33 |
| `H2` | 0, 0.086 | 0 | 2.5 s | none | 74 |
| `H2_swapped` | 0, 0.131 | 0 | 3.0 s | none | 77 |
| `H3` | 0, 0.071 | 0 | 2.0 s | none | 34 |
| `H4` | 7, 0.105 | 0 | one stop (after walking and turning) never settles | that stop: the waist and ankles shake | 258 |
| `H4_swapped` | 8, 0.155 | 0 | 6.0 s after 0.4 m/s forward; one stop (after walking and turning) never settles | that stop: the waist and ankles shake | 271 |

So in simulation nothing falls except under the two `H4` cases, where kp acts per motor and kd at the joint (7 and 8
of 2,688 rollouts, no command in all of its draws), and the stand check passes under all but pushes (by construction)
and the two `H4` cases, where one stop keeps shaking. Under per-motor kp (`H2`, `H4`) the linkage motors are also
asked for more than their 60 N m. The stand check with a zero previous action passes too, from the reset and from the
hand-over.

The previous file, `t2m_vel_s2_013000.onnx` (seed 2 of [The measured T2](#the-measured-t2)'s recipe; not published),
ran on the T2 on 2026-11-22: it took quick short steps at slow commands (about 0.4 s and 15 cm at 0.4 m/s,
walking 0.35 m/s), stepped within 14% left to right, and passed the same checks (median error 0.056; with 4 lag
draws it fell 0 and 1 times of 2,688 under `H4` and `H4_swapped`, against this file's 7 and 8; its other variant step
tests used one lag draw and had no fall).

These are simulation results. They say the file and the tool are right and what the policy tolerates in simulation;
how the robot behaves is not known until it runs there.

## Landing command (experimental)

A third interface, added to a finished policy: given one foot and a landing pose for it (x, y and turn, in the other
foot's frame), the robot lifts that foot, puts it down there by a deadline that follows from the size of the step,
and stands still until the next command. It is an add-on: [src/train.py](src/train.py) loads the finished policy
(`--init_from`), freezes its motion encoder, velocity encoder, decoder and normalizers (`--freeze_base`, checked bit
for bit at every checkpoint), and trains a landing encoder and its critic (`--land_frac`) on the library's own steps
and on sampled commands ([src/landing.py](src/landing.py)). Motion tracking and the velocity mode are unchanged. It
has an ONNX export and a test in native MuJoCo (below). It has not been on the robot. The T1's file
(`logs/land_r4_d10_al2_s3/ckpt_004000_both.onnx`) holds it together with the velocity interface (`--both`, below).

```bash
bash utils/t1_land_run.sh my_land_run
```

[utils/t1_land_run.sh](utils/t1_land_run.sh) **deletes `logs/<run>`**, trains 4,000 iterations on top of
`logs/t1_vel_ankle_sharp_x2_s1/policy/ckpt_013000.pt` (`BASE=...` names another finished policy, `SEED=2` another
seed), then runs the landing test and the velocity script on the last checkpoint. Training took 15 minutes on one
GB200 of TACC's Horizon; it has not been timed on the RTX 4090.

```bash
uv run python -m utils.landing_test --ckpt logs/my_land_run/ckpt_004000.pt --out logs/my_land_run/landing_test.json
```

The landing test has two parts, both under evaluation conditions (no randomization, noise or pushes). The first
sends 226 single commands from the default stance (moves of 5 to 30 cm in 8 directions with turns of -15 to +30
degrees, and turns on the spot) and scores each at its deadline. The second starts 2,048 robots on standing frames
of the motion library, gives each of them sampled commands one after another for 15 s, and reports the falls,
whether the feet stayed where they were before the first command, and the landing error by the command's place in
the sequence (a later command starts from the previous landing). `--no_library` skips the second part, and
`--record N` saves the first N commands' states of the first part for rendering.

Measured on 2026-11-17 on three seeds of the default recipe:

| | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| **From the default stance** (226 commands): falls | 0 | 0 | 0 |
| arrived: within 5 cm and 0.10 rad at the deadline, foot down | 99.1% | 96.9% | 99.6% |
| median error | 0.55 cm, 0.27° | 1.06 cm, 0.29° | 0.61 cm, 0.26° |
| 95th-percentile error | 1.49 cm | 3.11 cm | 1.33 cm |
| foot lifted; other foot kept in place | 100%; 100% | 100%; 100% | 100%; 100% |
| standing still within 3 s of the deadline | 96.0% | 97.8% | 99.6% |
| **From library stances** (about 9,400 commands): falls | 0 | 0 | 0 |
| feet kept in place before the first command | 99.6% | 12.4% | 99.3% |
| within 5 cm and 0.10 rad at the deadline | 96.7% | 91.8% | 97.7% |
| median error, first command | 0.62 cm | 1.47 cm | 0.71 cm |
| median error, second to fourth and later commands | 0.78 to 0.94 cm | 1.25 to 1.38 cm | 0.83 to 0.96 cm |
| steps of 45 to 60 cm: within 5 cm and 0.10 rad | 87.6% | 71.8% | 90.0% |

Seeds 1 and 3 pass both parts. Seed 2 passes the first part and fails the second: started from an unusual stance it
moves a foot right after the reset (8.8 cm, median), before any command. The second part is there to catch that, so
test a new checkpoint with both parts and train another seed if it fails. The longest steps (over 45 cm) are the
least accurate on every seed.

### Export and native MuJoCo

```bash
uv run python -m utils.export_landing --ckpt logs/my_land_run/ckpt_004000.pt   # writes .../ckpt_004000_land.onnx
uv run python -m utils.landing_native --onnx logs/my_land_run/ckpt_004000_land.onnx --sim logs/my_land_run/landing_test.json
uv run python -m utils.landing_native --onnx logs/my_land_run/ckpt_004000_land.onnx --walk logs/land_r4_d10_al2_s3/ckpt_004000_both.onnx
```

[utils/export_landing.py](utils/export_landing.py) writes the landing interface with its command state inside the
model, next to the checkpoint. It is not in the velocity export's format:

| | Name and shape | Contents |
|---|---|---|
| input | `obs_hist` (1, 10, 93) | The velocity export's observation history. |
| input | `cmd` (1, 5) | go, foot, x, y, yaw. go = 1 on the tick a command arrives; foot 0 = left, 1 = right; the landing pose of that sole in the other sole's frame [m, rad]. Zeros otherwise. |
| input | `state` (1, 8) | Zeros at a reset, then the previous tick's `next_state`. |
| output | `action` (1, 29) | As the velocity export's. |
| output | `next_state` (1, 8) | started, foot, steps left, SETTLE steps left, latch, x, y, yaw. |

A caller needs no foot kinematics, clock or anchors, and no estimate of where the robot is: the model rebuilds the
command state from the joint angles and the gravity direction, looks the clock up from the step's size, and takes a
command only while the robot holds (otherwise the command is ignored, not queued). The state tells the caller
whether its command was taken (steps left above 0 in the `next_state` of the tick it was sent on), when the step is
over (steps left back at 0) and then the stance the robot landed in (x, y, yaw of the commanded sole in the other
sole's frame, once latch is 0). The model's metadata describes every input and the trained range. The file is
written only once the ONNX model reproduces the checkpoint on five commands in the training simulator (actions
against PyTorch's, clocks against the trainer's). That check runs one world on the CPU with CUDA hidden, in about
half a minute, so the export needs no GPU.

With `--both` the export is one file (`..._both.onnx`) that also holds the checkpoint's velocity interface, in front
of the same decoder. Its inputs are `obs_hist`, `state`, the landing command as `land_cmd` (1, 5), `vel_cmd` (1, 3)
(vx, vy, yaw rate, as the velocity export's `cmd`) and `mode` (1, 1): above 0.5 the landing interface runs, else
the velocity interface, which answers a `next_state` of zeros, so the first landing tick after a walk is a reset's.
A runner stands on the zero velocity command for a second or more, then switches `mode`. The self-check also
compares velocity mode with PyTorch. The native tools take the file as both `--onnx` and `--walk` (`--walk_onnx`).

[utils/landing_native.py](utils/landing_native.py) runs the first part of the landing test on the export in native
MuJoCo: the same 226 commands, rollouts by `utils/play_mujoco.py landtest` on the CPU (about half a minute), scored
by the same code; `--sim` prints the training simulator's test beside it. Each rollout's actuator lags are drawn at
random (`--seed`); `--lag 5` fixes them at the training simulator's evaluation delay. Measured on 2026-11-17 on
`logs/land_r4_d10_al2_s3/ckpt_004000.pt` (seed 3 above):

| 226 commands from the default stance | training simulator | native, every lag 5 | native, random lags (3 draws) |
|---|---|---|---|
| falls | 0 | 0 | 0 |
| arrived: within 5 cm and 0.10 rad at the deadline, foot down | 99.6% | 99.6% | 99.6% |
| median error | 0.61 cm, 0.26° | 0.61 cm, 0.26° | 0.66 to 0.69 cm, 0.26 to 0.33° |
| 95th-percentile error | 1.33 cm, 0.88° | 1.27 cm, 0.88° | 1.31 to 1.42 cm, 1.04 to 1.11° |
| foot lifted; other foot kept in place until the deadline | 100%; 100% | 100%; 100% | 99.6 to 100%; 100% |
| standing still within 3 s of the deadline (median) | 99.6% (1.19 s) | 100% (1.19 s) | 100% (1.18 to 1.21 s) |
| stance kept for the 3 s after the deadline | 96.9% | 96.9% | 96.5 to 96.9% |

The one command that does not arrive is the same in every column (left foot, 30 cm forward and outward with a 30°
turn: 9.0 to 9.7 cm off). The model gives 40 of the 226 commands a deadline one control step shorter than the
training test did, because it looks the clock up from the stance when the command arrives and not from the stance at
the reset.

The last row is not in the training test's pass line, which watches the other foot only until the deadline. After
7 of the 226 commands, the same ones in both simulators and all of them 30 cm steps, that foot moves once the step
has landed (more than 2 cm or 0.10 rad): by 2.5 to 5 cm after four of them, and by a step of 32 to 50 cm after three
(the right foot sent 30 cm backward, straight or diagonally outward). The robot then stands still, but not in the
commanded stance.

`--second` tests a second step, since the use is one or two steps into a stance and then standing: one second
after each command's deadline the other foot is brought beside the one that moved (the default stance seen from
that foot), and those second commands are scored. Four pairs whose second step would be under 5 cm are left out:

| 222 second commands | native, every lag 5 | native, random lags (3 draws) |
|---|---|---|
| falls | 0 | 0 |
| arrived | 100% | 100% |
| median error | 0.73 cm, 0.22° | 0.70 to 0.74 cm, 0.24 to 0.29° |
| 95th-percentile error | 1.51 cm, 0.98° | 1.50 to 1.58 cm, 1.06 to 1.14° |
| foot lifted; other foot kept in place until the deadline | 100%; 100% | 99.5 to 100%; 100% |
| standing still within 3 s of the deadline (median) | 100% (1.18 s) | 100% (1.18 s) |
| stance kept for the 3 s after the deadline | 100% | 100% |

The training simulator has no like-for-like number for this: its second commands come from library stances (0.83
cm median on this seed). Its test's second part has no native counterpart.

`--walk <velocity export>` tests the hand-over from walking, done the way a runner would do it with the two files.
Each rollout walks under the velocity export, stands `--stop_s` seconds (default 3) on the zero command, and then
the landing model takes over the same observation history with its state at zeros. One second later the command is
sent: the grid's move from the stance the robot then stands in, which a first pass of the same rollouts without a
command finds. The walks are standing, 0.5 and 1.0 m/s forward, 0.6 m/s backward, 0.5 m/s to either side, a turn on
the spot at 1.5 rad/s and 0.4 m/s forward while turning at 1.0 rad/s, each for 3.0 to 3.4 s so that the stop arrives
at four points of the gait. Moves the sampler would not send from the stopped stance are left out. With the T1
policy and a 3 s stop (about a minute and a half per run):

| after a walk and a 3 s stop | native, every lag 5 | native, random lags (3 draws) |
|---|---|---|
| commands sent, of 226 | 213 | 206 to 214 |
| falls, walking or afterwards | 0 | 0 |
| standing still at the hand-over | 100% | 99.6 to 100% |
| feet kept in place from the hand-over to the command | 100% | 100% |
| arrived | 99.5% | 99.0 to 99.5% |
| median error | 0.57 cm, 0.32° | 0.60 to 0.63 cm, 0.34 to 0.36° |
| 95th-percentile error | 1.48 cm, 1.04° | 1.49 to 1.64 cm, 1.05 to 1.22° |
| foot lifted; other foot kept in place until the deadline | 99.5%; 100% | 100%; 100% |
| standing still within 3 s of the deadline (median) | 100% (1.22 s) | 99.5% (1.14 to 1.21 s) |
| stance kept for the 3 s after the deadline | 99.1% | 98.1 to 99.1% |

The stance a stop leaves is about 4 cm and 3° from the default one (median; up to 18 cm and 16°), and the landing
model takes it over without moving a foot: with no command sent, the stance of the hand-over is kept for the 3 s
after it in every rollout. The commands that fail are long steps, most of them the ones that fail from the default
stance: the same command does not arrive in any run (one draw has a second miss, at 5.0 cm), and the other foot
moves after the landing for two of the seven commands named above in every run and, on two of the three draws, for
two more steps of 20 to 30 cm. A second step after the walk (`--walk --second`, 203 to 207 pairs, the same four lag
settings) lands like one from the default stance: no fall, 100% arrived, 0.75 to 0.76 cm median, the stance kept
afterwards every time.

How long the robot stands before the hand-over matters little once its feet are down (one lag draw each):

| seconds on the zero command before the hand-over | 3 | 2 | 1 | 0 |
|---|---|---|---|---|
| falls | 0 | 0 | 0 | 0 |
| standing still at the hand-over | 100% | 95.6% | 12.8% | 12.8% |
| feet kept in place from the hand-over to the command | 100% | 100% | 100% | 25.1% |
| commands sent, of 226 | 214 | 214 | 212 | 167 |
| arrived | 99.5% | 99.5% | 99.5% | 99.4% |
| median error | 0.62 cm | 0.63 cm | 0.63 cm | 0.76 cm |
| 95th-percentile error | 1.57 cm | 1.61 cm | 1.82 cm | 2.54 cm |

One second after the stop command the feet are where they stay, while the body still settles (only the rollouts
that never walked count as still), and the landing is as accurate as after 3 s. Handed over while still walking
(0 s), the landing model brings the robot to a stand itself without a fall; its feet come to rest 0.4 s later
(median; 1.0 s at most), so a command sent before that would be built on a stance that is still changing. Starts
from library stances, and from any stance a walk does not leave, have no native counterpart.

## Booster T2

The same code trains the **Booster T2** (31 DoF, Booster's Revo2 hands locked open) with `--robot t2`
([robot/t2.py](robot/t2.py)). Both input modes train and are evaluated in simulation, a velocity policy exports to
ONNX and runs in plain MuJoCo ([the measured T2](#the-measured-t2) below), and [deploy/](deploy) holds one with the
task that runs it on the robot ([Deploy on the robot](#deploy-on-the-robot)). An earlier velocity policy has run on
the robot through that task; the file now in deploy/ has not yet.

- **Model.** `assets/t2/t2_description/mjcf/t2.xml` is Booster's `T2_31dof_Revo2Hand.xml` from
  [booster_assets](https://github.com/BoosterRobotics/booster_assets) (the commit is in `assets/t2/SOURCE.txt`) with
  the fingers welded open, one collision box per hand and the base T2's ankle pitch range, built by
  [utils/build_t2_mjcf.py](utils/build_t2_mjcf.py) (`uv run python -m utils.build_t2_mjcf`, from a clone in
  `~/booster_assets`). It is tracked. The actuators follow Booster's T2 recipe in
  [booster_train](https://github.com/BoosterRobotics/booster_train) and
  [booster_deploy](https://github.com/BoosterRobotics/booster_deploy).
- **Clips.** [utils/make_booster_t2.py](utils/make_booster_t2.py) registers the T2 with soma-retargeter,
  [utils/soma_t2_configs.py](utils/soma_t2_configs.py) writes its retargeting configs (human scale 0.75 or 0.80) and
  [utils/retarget_bones_seed_t2.sh](utils/retarget_bones_seed_t2.sh) retargets BONES-SEED into
  `dataset/bones_seed/soma_uniform/csv_booster_t2/`; each file's header has its usage.
  [utils/edit_t2_motions.py](utils/edit_t2_motions.py) edits the clips into a new folder (the fixes are listed
  below), and [utils/prepare_t2.py](utils/prepare_t2.py) packs a folder or a list of clip paths into a library:

  ```bash
  uv run python -m utils.edit_t2_motions --src dataset/bones_seed/soma_uniform/csv_booster_t2 \
      --dst dataset/bones_seed/soma_uniform/csv_booster_t2_edited --untwist
  uv run python -m utils.prepare_t2 dataset/bones_seed/soma_uniform/csv_booster_t2_edited data/t2
  ```

  [utils/check_clips.py](utils/check_clips.py) checks a packed library of either robot (joint limits, soles,
  self-contacts) and renders named clips.

  What was set or fixed in the T2 clips. Two things are in the retarget:

  - **Scale 0.80.** The human motion is retargeted at 80% of its size (`TARGET=booster_t2_080`).
  - **Leg calibration.** The retarget's zero pose for the legs is slightly bent (hip pitch -0.075, knee 0.15, ankle
    pitch -0.075 rad), which is where the T2's leg is longest. With zero legs the knee sat under its soft limit in 18
    to 21% of the pilot's frames.

  Four are the editor's (its header has the measurements behind each):

  1. **Arms.** Elbows and wrists are held at the default pose. Shoulder roll is turned in by one constant per actor,
     so that the arms hang close to the trunk (a median of 1.3 rad over the actor's straight forward walking).
  2. **Head.** Pitch is 0 and yaw follows the direction of travel; the head stays straight when walking backward,
     sideways or in place.
  3. **Feet** (on by default, `--no-feet` leaves it out). Both feet are turned in by the clip's toe-out, so that they
     point parallel instead of outward; the turn is 9.4° at the median, and a clip and its mirror share one value.
     Standing toe-out against the base went from +14° / +10° (left / right) to +5.5° / +1.2°.
  4. **Trunk** (`--untwist`). The retarget leaves the trunk, whose frame the velocity command is measured in, turned
     2.1° to the right of the pelvis, legs and travel line in every clip, with 0.45° of sideways lean. The step takes
     both out and keeps the pelvis and feet where they were. Left and right toe-out against the base then agree
     within 0.1°: +3.3° standing, +1.4° walking forward, +2.0° walking backward.

  Known and left as they are: toe-out is measured in forward walking, so against the pelvis slow backward walking
  keeps +2.3° and fast backward walking ends at -1.4°; a constant trunk pitch, which mirroring cannot tell from
  posture; and 0.15° of sideways lean of the pelvis.
- **Train.** [utils/t2_run.sh](utils/t2_run.sh) is `utils/t1_run.sh` for the T2: the same recipe on `data/t2` (or
  `DATA=<folder>`), then one evaluation at the T2's fixed action delay.

  ```bash
  bash utils/t2_run.sh my_t2_run --ankle_target 8.5 --ankle_roll_weight 2.8 --rest_weight -2
  ```

  The three flags are standing-only reward terms, off by default (the T1's recipe does not use them): the ankle
  torque term asks for the torque that puts the T2's weight over mid-foot instead of none and counts the ankle roll
  motors too, and a rest-pose term holds the hip roll, hip yaw and waist joints at the clips' standing pose.

### The measured T2

`--robot t2_measured` ([robot/t2_measured.py](robot/t2_measured.py)) is the same model simulated as a real T2 was
measured, so that training matches what the robot does with the gains it is sent:

- **Gains.** Booster's T2 walking table (booster_deploy's `tasks/locomotion/robots/t2`: natural frequency 10 Hz,
  damping ratio 1.0), with the default table's head and wrist values.
- **Joint friction and linkage damping.** Coulomb friction on every joint (0.3-3.3 N·m) and the waist and ankle
  damping at 0.69 times Booster's rule, both measured on a real T2 by a third party
  ([T2_supermimic](https://github.com/BarryLi159159/T2_supermimic), `src/supermimic/robots/t2.py`). The robot applies
  the kd sent to a waist or ankle joint through its linkage, so the simulated kd of those six joints differs from the
  kd to send: the spec carries both, and the ONNX export records the table to send (`kp_sent`, `kd_sent`).
- **Randomization.** Their set around those numbers: a factor per joint on the friction (0.5-1.5) and on kp and kd
  (±10%), link masses (±5%), mass on the trunk (±5 kg), ground friction 0.3-1.6, a gyro bias, an IMU mounting tilt and
  observation noise growing to twice the usual. `--dr_ramp_iters N` phases the gain randomization, the sensor errors
  and the noise growth in over the first N iterations. The world model and the critics see the randomized values;
  the sensor errors are applied in the simulator and in the imagined rollouts.
- **The numbers are in one place.** The gains, the friction table and the damping factor are constants at the top of
  the file. A run saves them with its arguments, and the tools refuse a checkpoint on a robot whose numbers differ.

Train (the recipe above on this robot), check, export and drive, the last three without a GPU:

```bash
ROBOT=t2_measured bash utils/t2_run.sh my_run --ankle_target 8.5 --ankle_roll_weight 2.8 --rest_weight -2 --dr_ramp_iters 2000
CUDA_VISIBLE_DEVICES= uv run python -m utils.check_velocity --ckpt logs/my_run/policy/ckpt_013000.pt
CUDA_VISIBLE_DEVICES= uv run python -m utils.export_onnx --ckpt logs/my_run/policy/ckpt_013000.pt --device cpu
CUDA_VISIBLE_DEVICES= uv run python -m utils.play_native --onnx logs/my_run/policy/ckpt_013000.onnx
```

- [utils/check_velocity.py](utils/check_velocity.py) runs stop, walking-speed and stance checks in the training
  simulator: at the robot as measured, at the edges of the randomization (friction, gains, trunk mass, delay) and
  outside it (no joint friction, Booster's damping rule).
- [utils/play_native.py](utils/play_native.py) drives the ONNX file with the keyboard in plain MuJoCo (`--script`: a
  scripted drive without a window; `--steptest`, `--stand` and `--variant`:
  [The T2's policy in plain MuJoCo](#the-t2s-policy-in-plain-mujoco)), on the model the training simulator compiles and the spec's gains, torque-speed
  curve and delay; its model, motor law and observation are tested against the training simulator's.
  [utils/play_velocity.py](utils/play_velocity.py) does the same for a checkpoint in the training simulator
  (`--device cpu` runs it without a GPU).

### Deploy on the robot

This task ran the previous file (`t2m_vel_s2_013000.onnx`) on the T2 on 2026-11-22; the current file has not run on
the robot yet. [deploy/booster_t2/](deploy/booster_t2) is a task for Booster's
[booster_deploy](https://github.com/BoosterRobotics/booster_deploy) (read at its commit 7bb1462; on this workstation run only in its MuJoCo mode) that
runs `deploy/t2al1_vel_s1_013000.onnx` on the T2. It runs where booster_deploy runs: on the robot's own computer (a
Jetson Thor), over SSH, in booster_deploy's `.venv` with the robot's ROS 2 sourced. The model runs on the CPU, and the
policy code ([policy.py](deploy/booster_t2/policy.py)) imports only numpy and onnxruntime, which booster_deploy
already requires. The workstation this was written on is x86-64; on the Thor, the task has run the previous file.

What the task does:

- **Gains and pose.** It sends the gains the policy was trained for (the ONNX file's `kp_sent` / `kd_sent`; the file's
  `kd` is the joint-side damping in training and is never sent) and uses the policy's default pose as Booster's
  prepare pose, with `prepare_mode="standing"`: X puts the robot in Custom mode and moves it to that pose in 1 s, held
  stiffly with no balance control, and A starts the policy from there.
- **Checks before the robot is touched.** When booster_deploy starts, before Custom mode, the task compares joint
  order, default and prepare pose, control rate, observation layout and the gains it will send with what the ONNX file
  records, and refuses to load on a mismatch. It refuses to start without the file (when another task is started it
  only warns). It also refuses a booster_deploy that reads the exit mode another way than the task's exit relies on,
  or whose controller lacks something the task reads. A refusal stops booster_deploy for every task until the folder
  is fixed or removed.
- **Stops.** A fall (tilt beyond about 60 degrees), a robot state that stops updating (three identical readings of
  everything, or five of the IMU alone), or a reading that is not a number. The stiffness is zeroed first, so the
  command the motors keep is damping only. A command that is not a number is taken as zero, and one outside the
  trained box is held at its edge.
- **Exit.** Booster switches the robot to one mode at the end of every run. Here: Booster's walking mode after a normal
  stop (Ctrl-C, a start cancelled before A), damping after one of the stops above. `--exit-mode walking` or
  `--exit-mode damping` on Booster's command line overrides both for that run.
- **Timing.** After a stall Booster's loop runs the missed steps back to back; calls less than 10 ms apart repeat the
  last setpoint and do not advance the observation history. Compute time and loop period are printed every 5 s.
- **Remote.** First-session caps: forward 0.5, backward 0.3, sideways 0.3 m/s, turning 1.0 rad/s (the trained box is
  +1.90 / -1.39, ±1.14, ±2.78); a stick within a tenth of its range of centre gives an exact zero command. The Revo2
  hands are not commanded.

Install, on the robot's computer (`<bd>` is the booster_deploy folder there):

```bash
python -c "import onnxruntime, numpy"    # in booster_deploy's .venv; if this fails, set booster_deploy up first (its README)
git -C <bd> rev-parse --short HEAD       # the task was written against 7bb1462; another commit is untested
cp -r deploy/booster_t2 <bd>/tasks/bfm_t2
mkdir -p <bd>/tasks/bfm_t2/models && cp deploy/t2al1_vel_s1_013000.onnx <bd>/tasks/bfm_t2/models/
python scripts/deploy.py --list          # from <bd>: must print "[bfm_t2] checked against ..." and list bfm_t2_vel
```

Then the passive dry run, from `<bd>`: it reads `/low_state`, runs the task's observation and policy, publishes
nothing and changes no mode.

```bash
python -m tasks.bfm_t2.dry_run
```

Before any powered run, check in its output:

- **Gravity.** (0, 0, -1) with the trunk upright; x grows when the robot is tilted forward; y goes negative when its
  right side goes down. With both soles flat on level ground, the IMU's gravity and the one the leg joints imply agree
  within about 2 degrees (training covered a constant IMU tilt up to that).
- **Gyro.** Near zero at rest (training covered a bias up to 0.02 rad/s); z positive when the robot is turned
  counter-clockwise.
- **Joints.** Moving one joint by hand changes that joint and only it, with the right sign.
- **Command.** (0, 0, 0) with the sticks released. A pad whose yaw axis is a trigger reads as full yaw.
- **Compute time.** The summary at Ctrl-C must end in PASS: 99% of steps under 5 ms of the 20 ms step. It also gives the
  `/low_state` rate, the loop period's spread and the largest gap between a joint's reported speed and its
  finite-differenced position.
- **Would-be stops.** Let it run for at least a minute with the robot standing still; the summary fails a shorter
  run, and it fails if a powered run would have stopped on a frozen or invalid reading. Those stops fire on three
  bit-identical readings of the whole state in a row, or five of the IMU alone, on the assumption that a live sensor
  never repeats itself exactly. Each would-be stop is printed when it happens, and the summary gives the longest runs
  of identical readings it saw. If this robot's readings do repeat at rest, the stop would fire by mistake in a
  powered run and drop a standing robot (`STALE_STEPS` and `STALE_IMU_STEPS` in `policy.py` are the settings).
- **Actions.** Not zero at rest. Standing in simulation in its own default pose, the policy's actions stay within about
  ±1 (largest: shoulder pitch -1.0 and -0.5, hip pitch 0.4 to 0.6) and hold about 11.5 N m at each ankle pitch. In
  Booster's straight-legged stance the robot is 0.2-0.4 rad from that pose at the hips and knees, so the values
  differ; they should be of that order and steady.

The first powered runs:

1. The robot on a gantry with slack, the hands opened with Booster's own tools, the hardware e-stop within reach.
   Start booster_deploy inside `tmux` (or `screen`) on the robot: if the SSH connection drops, the program is killed
   with no mode switch and the motors keep its last command. Booster's keyboard controls are live in that terminal
   (`x`, `r`, `w`/`s`/`a`/`d`/`q`/`e` change the command, Space zeroes it).
2. The robot standing still in Booster's own mode, both feet flat, before X (the hold starts from the pose it is in).
3. `python scripts/deploy.py --task bfm_t2_vel`, then X, then A within a few seconds of reaching the prepare pose.
4. Short stretches first, under 10 s (training's episodes and every check here end at 10 s).
5. To stop: release the sticks, wait until it stands still, then Ctrl-C. The robot goes back to Booster's walking
   mode. After a fall or a dead sensor it goes to damping instead.
6. If the gantry catches the robot, it hangs at less than 60 degrees of tilt, so the task has not stopped and is still
   driving the motors: use the e-stop. A plain Ctrl-C would hand a hanging robot to Booster's walking controller.
   Starting with `--exit-mode damping` makes every ending, Ctrl-C included, end in damping; that suits sessions where
   the gantry carries the robot anyway.

`python scripts/deploy.py --task bfm_t2_vel --mujoco` runs the task in Booster's own MuJoCo mode (2 ms physics, no
delays, torque-speed curve or joint friction, kd applied at the joint). A command typed there has its vx held to the
remote's cap; vy and yaw are not.

That mode was checked here on 2026-11-21 with Booster's controller (booster_deploy 7bb1462) stepped without its
viewer: the scripted drive's six commands for 10 s each and the ten stop cases, on Booster's model and on the model
training uses, each with the waist and ankle kd as sent and at training's joint-side values. No fall in the 64
rollouts; still within 2.5 s of every stop, no shaking; 0.47 m/s at a 0.5 m/s command and 1.05 at 1.0 on Booster's
model (0.44 and 1.01 on training's). A forced fall (600 N on the trunk) stops the task at 60 degrees with the
stiffness zeroed, and the exit setting then reads damping.

Not tested, and what to watch for:

- **This file on the robot.** The task ran the previous file on the T2 on 2026-11-22, so the Thor, its aarch64
  onnxruntime and the task have run there; this file has not, and its response to real latency and IMU noise, the
  hands, the hand-over at A and the exit hand-over to Booster's walking mode are untested with it.
- **How the firmware applies the gains on the waist and ankle linkages.** Not measured on this robot. Training assumes
  kp at the joint and kd per motor at 0.69 of Booster's rule (another group's measurement on another T2). Signs of
  another case: the waist and ankles visibly stiffer than the other joints, or motors running hot or current-limited
  (kp per motor); the ankles ringing after a step (less damping than assumed); slow, sticky ankles and poor slow
  sideways steps (more friction or damping than assumed). The task has no setting for another case: that is a new
  decision.
- **Longer runs.** In simulation one ankle roll joint creeps slowly at rest (0.2-0.3 degrees per second, never
  reversing); the 10 s rollouts cannot show whether it stops. Watch for it, for yaw creep and for posture drift.
- **Training against the robot.** No pushes during training episodes, a box-shaped foot collider, flat ground only,
  observation latency not randomized.

## Repository layout

| Path | Contents |
|---|---|
| [robot/t1.py](robot/t1.py) | The T1's spec: MJCF, joint and body names, explicit Booster PD actuators with the torque-speed clip, action delays, from `t1_actuators.npz`. |
| [robot/t2.py](robot/t2.py) | The T2's spec (31 DoF, Revo2 hands locked open): the MJCF of `assets/t2/`, Booster's T2 actuator recipe, action delays, and the T2's values for the velocity task (command box, fall height, ankle torque scale, rest pose). |
| [robot/t2_measured.py](robot/t2_measured.py) | The T2 as a real one was measured: Booster's walking gain table, joint friction, the waist and ankle damping the linkages give, and the randomization around them. |
| [robot/spec.py](robot/spec.py), [robot/kinematics.py](robot/kinematics.py) | `RobotSpec`, `get_robot(name)` (`t1`, `t2`, `t2_measured`) and the robot numbers a checkpoint is checked against; differentiable forward kinematics. |
| [src/train.py](src/train.py) | Training loop: collect in the simulator, fit the world model, train the policy and critics through imagined rollouts, with exams, frontier practice, the deployment stream and the step-size schedule. |
| [src/env.py](src/env.py), [src/dr.py](src/dr.py), [src/sensing.py](src/sensing.py) | Batched simulator on mjlab (MuJoCo-Warp) and its resets; domain randomization and the sim context ξ; observation noise. |
| [src/mdp.py](src/mdp.py) | Observations, goals, rewards (tracking, velocity and landing) and terminations, shared by the simulator and the world model. |
| [src/landing.py](src/landing.py) | The landing command: sole geometry, the library's steps read as commands, the command's state, its reward, the command sampler and the encoder's inputs. |
| [src/networks.py](src/networks.py), [src/objective.py](src/objective.py), [src/optim.py](src/optim.py) | World model, policy (motion, velocity and landing encoders, one decoder) and critics; the actor objective; Muon + Adam. |
| [src/motion.py](src/motion.py), [src/buffer.py](src/buffer.py), [src/mirror.py](src/mirror.py) | The packed motion library at 50 Hz; the GPU replay buffer; the left-right mirror for world-model data. |
| [src/evaluate.py](src/evaluate.py) | Evaluation on every clip. |
| [utils/t1_run.sh](utils/t1_run.sh), [utils/avg_ckpt.py](utils/avg_ckpt.py) | Training driver; checkpoint averaging. |
| [utils/play_velocity.py](utils/play_velocity.py), [utils/velocity_rig.py](utils/velocity_rig.py), [utils/export_onnx.py](utils/export_onnx.py) | The velocity mode in the training simulator (keyboard or script) and its ONNX export. |
| [utils/t1_land_run.sh](utils/t1_land_run.sh), [utils/landing_test.py](utils/landing_test.py), [utils/landing_rig.py](utils/landing_rig.py), [utils/landing_score.py](utils/landing_score.py) | The landing add-on's training driver; its test in the training simulator (the rigs and the scoring). |
| [utils/export_landing.py](utils/export_landing.py), [utils/landing_native.py](utils/landing_native.py) | The landing interface's ONNX export (command state inside the model) and its test in native MuJoCo. |
| [utils/record_clips.py](utils/record_clips.py), [utils/render_gif.py](utils/render_gif.py) | Recording and rendering a few clips. |
| [utils/soma_csv.py](utils/soma_csv.py), [utils/filter_clips.py](utils/filter_clips.py), [utils/prepare_t1.py](utils/prepare_t1.py) | Clip pipeline: CSV loading, the dataset filter, the packer. |
| [utils/play_mujoco.py](utils/play_mujoco.py) | Native-MuJoCo viewer, video, step test and landing rollouts, with its own environment (`play_mujoco.py.lock`). |
| [utils/rotations.py](utils/rotations.py) | Rotation math. |
| [utils/edit_t1_motions.py](utils/edit_t1_motions.py), [utils/plot_walking_profile.py](utils/plot_walking_profile.py) | Clip editor; velocity-command coverage of the clips. |
| [utils/build_t1_mjcf.py](utils/build_t1_mjcf.py), [utils/make_booster_t1.py](utils/make_booster_t1.py), [utils/retarget_bones_seed_t1.sh](utils/retarget_bones_seed_t1.sh) | MJCF builder; soma-retargeter's T1 with this MJCF's joint ranges; the retargeting driver. |
| [utils/t2_run.sh](utils/t2_run.sh), [utils/build_t2_mjcf.py](utils/build_t2_mjcf.py), [utils/make_booster_t2.py](utils/make_booster_t2.py), [utils/soma_t2_configs.py](utils/soma_t2_configs.py), [utils/retarget_bones_seed_t2.sh](utils/retarget_bones_seed_t2.sh), [utils/edit_t2_motions.py](utils/edit_t2_motions.py), [utils/prepare_t2.py](utils/prepare_t2.py) | The T2's counterparts: training driver, MJCF builder, soma-retargeter target and its configs, retargeting driver, clip editor, packer (see "Booster T2"). |
| [utils/check_velocity.py](utils/check_velocity.py), [utils/play_native.py](utils/play_native.py) | For the T2s: stop, walking-speed and stance checks of a velocity policy in the training simulator on the CPU; keyboard control of an ONNX export in plain MuJoCo. |
| [utils/check_clips.py](utils/check_clips.py) | Kinematic checks of a packed library of either robot (joint limits, soles, self-contacts) and videos of named clips. |
| [tests/](tests) | Unit tests (pytest): rotations, forward kinematics against MuJoCo, the T1 model against the URDF, the T2 model and its clip editor, the robot spec, the optimizer, the velocity mode's parts, the landing command's parts, the plain-MuJoCo player, the T2 deploy task (against stand-ins for booster_deploy in `tests/standins`). |
| [assets/t1/](assets/t1) | Booster's T1 URDF and meshes (booster_assets, BSD-3: `LICENSE`, `SOURCE.txt`); the MJCF built from them and `config/sysid.txt` (holosoma's T1 config: joint friction, damping, armature; Apache-2.0: `config/LICENSE-holosoma`, `config/NOTICE-holosoma`); `mujoco/`, the native-MuJoCo model and the actuator table. |
| [assets/t2/](assets/t2) | The T2 MJCF built by `utils/build_t2_mjcf.py` and the meshes it references, from Booster's booster_assets (BSD-3: its `LICENSE`, and `SOURCE.txt`). |
| [deploy/](deploy) | The policy for the robot (the measured T2's velocity policy, run `t2al1_vel_s1`), exported to ONNX, and [deploy/booster_t2/](deploy/booster_t2), the booster_deploy task that runs it (see "Deploy on the robot"). |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | Which files come from BlinkMimic 2 (MIT), booster_assets (BSD-3) and holosoma (Apache-2.0), with BlinkMimic 2's licence and notices. |
| [LICENSE](LICENSE) | MIT, for everything not listed in THIRD_PARTY_NOTICES.md. |

## Not in this repository

Kept out by [.gitignore](.gitignore): `dataset/` (the motion data and the clip lists), `data/` (the packed library),
`logs/` (training runs, checkpoints, evaluation and native-MuJoCo output), `.npz` recordings at the repository root,
and `docs/` and `tacc/` (the author's notes and cluster job scripts, not published).

## Licence

This repository's own code is under the MIT licence ([LICENSE](LICENSE)). The files derived from BlinkMimic 2 and the
robot models from Booster Robotics and holosoma are under their own licences, listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The trained policy in [deploy/](deploy) is derived from the
BONES-SEED motion data and is subject to the BONES-SEED License, not MIT.

## References

- Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, K. Sreenath. *SuperMimic: Scaling Humanoid Whole-Body
  Control with One Differentiable World Model.* arXiv:2610.28480, 2026.
- [jyl681/blinkmimic2](https://github.com/jyl681/blinkmimic2) (Jiayu Li, MIT): the learner and its schedule, which
  build on [BlinkMimic](https://github.com/fuyichen0725/blinkmimic) (Fuyi Chen),
  [InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack) (Huy-Quang-Dao) and
  [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) (Chong Zhang, ETH Zurich).
- HuMBLE (arXiv:2610.10489): the goal-conditioned reward of the velocity mode (Table S6).
- [mjlab](https://github.com/mujocolab/mjlab), [MuJoCo](https://mujoco.org) and
  [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp): the simulators.
- Booster Robotics: the T1 URDF and meshes (booster_assets `12a9751`) and actuator models (booster_train `8bb5b53`).
- holosoma: the T1 config in `assets/t1/config/sysid.txt` (Apache-2.0).
- [NVIDIA soma-retargeter](https://github.com/NVIDIA/soma-retargeter): motion retargeting.
- [BONES-SEED](https://bones.studio/datasets/seed): the motion-capture data.
