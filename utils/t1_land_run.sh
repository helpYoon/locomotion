#!/bin/bash
# One T1 landing add-on run (src/landing.py): the landing encoder alone is trained on the finished velocity policy
# BASE, whose other tensors are loaded and frozen (--init_from, --freeze_base), then the landing test on the last
# checkpoint and the velocity script (utils/play_velocity.py --script). The script is printed for comparison with
# BASE's; the simulator is not repeatable to the last digit (two runs of one checkpoint differ by 0.01 in a few
# entries), so train.py's own check of the loaded tensors at every checkpoint is what proves the base untouched.
# A plain `bash utils/t1_land_run.sh <run name>` trains the recipe that passes the landing test (train.py's defaults
# for the landing flags).
# Usage: bash utils/t1_land_run.sh <run name> [train flags]   (it cds to the repo root itself)
# BASE (default logs/t1_vel_ankle_sharp_x2_s1/policy/ckpt_013000.pt): the policy to extend.
# IT (default 4000): iterations.
# SEED=2 ...: another seed (default 1). The train flags go last and win: e.g. --land_label_prob 0 (free rows only),
# --land_label_prob 1 (label rows only), --no-land_derived (the encoder without the 5 derived inputs).
# The run's checkpoints, landing_test.json and landing_test.log, walk_script.log and stdout.log are in logs/<run name>.
set -uo pipefail
cd "$(dirname "$0")/.."
R=logs/${1:?run name}
shift
BASE=${BASE:-logs/t1_vel_ankle_sharp_x2_s1/policy/ckpt_013000.pt}
IT=${IT:-4000}
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf "$R"
mkdir -p "$R"
date +%s > "$R/process_start.txt"
$PY -u -m src.train --log_dir "$R" --seed "${SEED:-1}" --init_from "$BASE" --freeze_base \
  --vel_frac 0.125 --land_frac 0.75 \
  --iterations "$IT" --ckpt_every 1000 --lr_decay_iters "$IT" --lr_end 0.3 \
  --phase= --exam_frac 0 --exam_mix= --exam_pre_every 0 --exam_pre_near 0 --front_share 0 --start_prac_frac -1 \
  --fail_penalty 0 --no-pen_horizon_only --no-reset_aprev_ref --no-phase_refill --share_ramp= --wstream_frac 0 \
  "$@" 2>&1 | tee "$R/stdout.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then echo "DRIVER_DONE $R training failed, exit $rc"; exit 1; fi
CK=$(ls "$R"/ckpt_[0-9]*.pt | sort | tail -1)  # the newest: train flags may carry their own --iterations
[ -n "$CK" ] || { echo "DRIVER_DONE $R no numbered checkpoint"; exit 1; }
s=$(( $(stat -c %Y "$CK") - $(cat "$R/process_start.txt") ))
echo "training time: $((s / 60)) min $((s % 60)) s from process start" | tee "$R/time.txt"
quiet() { grep --line-buffered -v "^Module \|^Warp "; }
$PY -u -m utils.landing_test --ckpt "$CK" --out "$R/landing_test.json" 2>&1 | quiet | tee "$R/landing_test.log"
rc_test=${PIPESTATUS[0]}
$PY -u -m utils.play_velocity --ckpt "$CK" --script 2>&1 | quiet > "$R/walk_script.log"
rc_walk=${PIPESTATUS[0]}
if [ "$rc_test" -ne 0 ]; then echo "DRIVER_DONE $R landing test failed, exit $rc_test"; exit 1; fi
if [ "$rc_walk" -ne 0 ]; then echo "DRIVER_DONE $R velocity script failed, exit $rc_walk"; exit 1; fi
echo "DRIVER_DONE $R ok"
