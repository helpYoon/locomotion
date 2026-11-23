#!/bin/bash
# One T1 run (seed 1, src/train.py's defaults, the deployed controller's recipe, plus the given flags), then the
# evaluations: the averaged policy at lag 5 and at random lags, and the checkpoints 1000 .. 6000 (x STRETCH) at lag 5.
# Usage: bash utils/t1_run.sh <run name> [train flags]   (it cds to the repo root itself)
# STRETCH (default 2): BlinkMimic 2's 6,500-iteration schedule at this many times its length, every schedule point (the
# phases, the learning-rate decay, the checkpoints averaged and evaluated) at this many times its iteration; 2 is
# train.py's default schedule, 1 is BlinkMimic 2's own. The refresh intervals (exam table, pre-exams, frontier) and the
# policy EMA keep their iteration counts. SEED=2 ...: another seed (default 1).
# BlinkMimic 2's motion-only recipe (train.py's default before 2026-11-16, about 15 minutes):
#   STRETCH=1 bash utils/t1_run.sh <run> --pi_ref_skip --vel_frac 0 --ankle_weight 0 --track_sharp 0
# Motion-only runs from then, each with that recipe plus:
#   t1_noskip_s1        --no-pi_ref_skip              the reference skip removed, as --vel_frac needs
#   t1_reffree_s1, t1_reffree_skip_s1: the reference orientation moved from the decoder to the motion encoder
#   (--ref_free, with and without the skip). That flag is gone; logs/ref_free_variant.patch has the code that ran them.
# The paper's baseline recipe, as BlinkMimic 2's scripts/presets.sh gave it (pass the flags as train flags, after the
# motion-only flags above):
#   # Same-time baseline of the paper: the same learner and code with the first-phase settings for the whole run (no exams,
#   # frontier practice, failure penalty, deployment stream or step schedule; the step falls by about 0.5% over the run).
#   PRESET[baseline]="--phase= --exam_frac 0 --exam_mix= --exam_pre_every 0 --exam_pre_near 0 --exam_pre_steps 500 --front_share 0 --start_prac_frac -1 --fail_penalty 0 --no-pen_horizon_only --no-reset_aprev_ref --no-phase_refill --share_ramp= --wstream_frac 0 --lr_decay_iters 250000 --lr_end 0.03 --iterations 7000 --ckpt_every 100"
set -uo pipefail
cd "$(dirname "$0")/.."
R=logs/${1:?run name}
shift
X=${STRETCH:-2}
x() { local out=() i; for i in "$@"; do out+=("$((i * X))"); done; (IFS=,; echo "${out[*]}"); }  # iterations x STRETCH, comma-joined
# BlinkMimic 2's schedule with its iteration numbers x STRETCH; at 2 it equals train.py's defaults (FINAL_PHASE)
SCHED=(--iterations "$(x 6500)" --ckpt_every "$(x 500)" --phase "0:exam_frac=0,front_share=0,fail_penalty=0,reset_aprev_ref=0,lr_decay_iters=$(x 250000),lr_end=0.03,wstream_frac=0;$(x 2000):exam_frac=0.5,front_share=1,fail_penalty=3,reset_aprev_ref=1,lr_decay_iters=$(x 5757),lr_end=0.3;$(x 3000):exam_frac=0.5,lr_decay_iters=$(x 6200),wstream_frac=0.25,critic_epochs=1;$(x 4500):exam_frac=0.9")
PY=.venv/bin/python
LIST=data/t1/keep.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # without it the full-size run runs out of memory on 24 GB
rm -rf "$R"
mkdir -p "$R"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$R/gpumem.log" &
trap "kill $! 2>/dev/null" EXIT
date +%s > "$R/process_start.txt"
$PY -u -m src.train --log_dir "$R" --seed "${SEED:-1}" --robot t1 --bones_lib data/t1/lib_all.npz --bones_list $LIST \
  "${SCHED[@]}" "$@" 2>&1 | tee "$R/stdout.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then echo "DRIVER_DONE $R training failed, exit $rc"; exit 1; fi
$PY utils/avg_ckpt.py --run_dir "$R" --ckpts "$(x 4500 5000 5500 6000 6500)" --out_dir "$R/policy"
s=$(( $(stat -c %Y "$R/$(printf 'ckpt_%06d.pt' "$(x 6500)")") - $(cat "$R/process_start.txt") ))
echo "training time: $((s / 60)) min $((s % 60)) s from process start" | tee "$R/time.txt"
quiet() { grep --line-buffered -v "^Module \|^Warp "; }
$PY -u -m src.evaluate --run_dir "$R/policy" --bones_list $LIST --tag lag5 2>&1 | quiet | tee "$R/policy/eval_lag5.log"
$PY -u -m src.evaluate --run_dir "$R/policy" --bones_list $LIST --tag lagrand --eval_delay random 2>&1 | quiet | tee "$R/policy/eval_lagrand.log"
$PY -u -m src.evaluate --run_dir "$R" --bones_list $LIST --tag lag5 --ckpts "$(x 1000 2000 3000 4000 5000 6000)" --group 3 2>&1 | quiet | tee "$R/eval_lag5.log"
echo "peak GPU memory: $(sort -n "$R/gpumem.log" | tail -1) MiB" | tee -a "$R/time.txt"
echo "DRIVER_DONE $R ok"
