#!/usr/bin/env bash
# Retarget BONES-SEED (SOMA BVH) to the Booster T2, one date folder per run so a failure only loses that folder. Folders
# whose CSV count already matches their BVH count are skipped. The robot and its configs must exist first (both scales
# share them, so this script does not rewrite them under a running twin):
#   python3 utils/make_booster_t2.py <soma>
#   cd <soma>; CUDA_VISIBLE_DEVICES= .venv/bin/python <bfm>/utils/soma_t2_configs.py .
# TARGET picks the scale (booster_t2_075 or booster_t2_080); SRC and DST (absolute paths) override the input and output
# folders; LIST (a file of <date>/<clip> lines, e.g. utils/t2_pilot_clips.txt) retargets only those clips; DEVICE is
# passed to --device.
# usage: TARGET=booster_t2_075 bash utils/retarget_bones_seed_t2.sh <soma-retargeter root>
#        on the CPU: CUDA_VISIBLE_DEVICES= DEVICE=cpu TARGET=... bash utils/retarget_bones_seed_t2.sh <soma root>
set -euo pipefail
BFM=$(cd "$(dirname "$0")/.." && pwd)
TARGET=${TARGET:?set TARGET=booster_t2_075 or booster_t2_080}
SRC=${SRC:-$BFM/dataset/bones_seed/soma_uniform/bvh}
DST=${DST:-$BFM/dataset/bones_seed/soma_uniform/csv_booster_t2}
LIST=${LIST:+$(realpath "$LIST")}
CFG=$(mktemp --suffix .json)
TMP=$(mktemp -d)
trap 'rm -rf "$CFG" "$TMP"' EXIT
unset PYTHONPATH  # keep ROS Humble's Python 3.10 packages out of this env
cd "$1"
test -f "local-robots/booster_t2/configs/soma_to_${TARGET}_retargeter_config.json"
if [ -n "$LIST" ]; then  # a folder tree of symlinks to the listed BVHs
    grep -v '^#' "$LIST" | while read -r clip _; do
        [ -z "$clip" ] && continue
        mkdir -p "$TMP/src/$(dirname "$clip")"
        ln -s "$SRC/$clip.bvh" "$TMP/src/$clip.bvh"
    done
    SRC=$TMP/src
fi

for dir in "$SRC"/*/; do
    name=$(basename "$dir")
    n_bvh=$(find "$dir" -name '*.bvh' | wc -l)
    n_csv=$(find "$DST/$name" -name '*.csv' 2>/dev/null | wc -l || true)  # find fails before the folder exists
    if [ "$n_csv" -eq "$n_bvh" ]; then echo "[skip] $name ($n_csv done)"; continue; fi
    cat > "$CFG" <<EOF
{"import_folder": "$dir", "export_folder": "$DST/$name", "batch_size": 100, "retargeter": "Newton",
 "retarget_source": "soma", "retarget_target": "$TARGET", "retarget_source_facing_direction": "Mujoco",
 "extra_robot_paths": ["$PWD/local-robots"]}
EOF
    echo "[run] $name ($n_bvh motions) $(date +%T)"
    # ponytail: .venv/bin/python instead of the T1 script's `uv run python`, so nothing re-syncs soma's venv
    .venv/bin/python app/bvh_to_csv_converter.py --config "$CFG" ${DEVICE:+--device "$DEVICE"} 2>&1 \
        | grep -E "Retargeted|ERROR|Error|Traceback" || true
done
echo "done: $(find "$DST" -name '*.csv' | wc -l) / $(find -L "$SRC" -name '*.bvh' | wc -l) motions"
