#!/bin/bash
# One Booster T2 run (seed 1, src/train.py's defaults = the deployed T1 controller's recipe, plus the given flags) on the
# T2 library (data/t2), then one evaluation: the averaged policy at the T2's fixed delay (RobotSpec.eval_delay 4).
# utils/t1_run.sh with --robot t2; its other two evaluation passes (random lags, the checkpoints 1000 .. 6000) are left
# out: on Horizon they took 37 of the T1 driver's 71 min (port plan decision 10). Run it on a GPU node, never locally.
# Usage: bash utils/t2_run.sh <run name> [train flags]   (it cds to the repo root itself)
# STRETCH (default 2) and SEED as utils/t1_run.sh; DATA (default data/t2) the library folder; ROBOT (default t2) the
# robot spec (t2_measured: robot/t2_measured.py). BlinkMimic 2's
# motion-only recipe (a retarget check, ~15-20 min):
#   STRETCH=1 bash utils/t2_run.sh <run> --pi_ref_skip --vel_frac 0 --ankle_weight 0 --track_sharp 0
set -uo pipefail
cd "$(dirname "$0")/.."
R=logs/${1:?run name}
shift
X=${STRETCH:-2}
x() { local out=() i; for i in "$@"; do out+=("$((i * X))"); done; (IFS=,; echo "${out[*]}"); }  # iterations x STRETCH, comma-joined
SCHED=(--iterations "$(x 6500)" --ckpt_every "$(x 500)" --phase "0:exam_frac=0,front_share=0,fail_penalty=0,reset_aprev_ref=0,lr_decay_iters=$(x 250000),lr_end=0.03,wstream_frac=0;$(x 2000):exam_frac=0.5,front_share=1,fail_penalty=3,reset_aprev_ref=1,lr_decay_iters=$(x 5757),lr_end=0.3;$(x 3000):exam_frac=0.5,lr_decay_iters=$(x 6200),wstream_frac=0.25,critic_epochs=1;$(x 4500):exam_frac=0.9")
PY=.venv/bin/python
D=${DATA:-data/t2}  # the T2 library folder (lib_all.npz, keep.txt)
LIST=$D/keep.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf "$R"
mkdir -p "$R"
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$R/gpumem.log" &
trap "kill $! 2>/dev/null" EXIT
date +%s > "$R/process_start.txt"
$PY -u -m src.train --log_dir "$R" --seed "${SEED:-1}" --robot "${ROBOT:-t2}" --bones_lib "$D/lib_all.npz" --bones_list "$LIST" \
  "${SCHED[@]}" "$@" 2>&1 | tee "$R/stdout.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then echo "DRIVER_DONE $R training failed, exit $rc"; exit 1; fi
$PY utils/avg_ckpt.py --run_dir "$R" --ckpts "$(x 4500 5000 5500 6000 6500)" --out_dir "$R/policy"
s=$(( $(stat -c %Y "$R/$(printf 'ckpt_%06d.pt' "$(x 6500)")") - $(cat "$R/process_start.txt") ))
echo "training time: $((s / 60)) min $((s % 60)) s from process start" | tee "$R/time.txt"
quiet() { grep --line-buffered -v "^Module \|^Warp "; }
$PY -u -m src.evaluate --run_dir "$R/policy" --bones_list $LIST --tag lag4 2>&1 | quiet | tee "$R/policy/eval_lag4.log"
rc=${PIPESTATUS[0]}
echo "peak GPU memory: $(sort -n "$R/gpumem.log" | tail -1) MiB" | tee -a "$R/time.txt"
if [ "$rc" -ne 0 ]; then echo "DRIVER_DONE $R evaluation failed, exit $rc"; exit 1; fi
echo "DRIVER_DONE $R ok"
