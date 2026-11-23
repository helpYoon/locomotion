#!/usr/bin/env bash
# Retarget BONES-SEED (SOMA BVH) to the Booster T1 with bfm_wm's joint limits (utils/make_booster_t1.py builds that
# robot in <soma-retargeter>/local-robots/), one date folder per run so a failure only loses that folder. Folders whose
# CSV count already matches their BVH count are skipped. SRC and DST (absolute paths) override the input and output
# folders.
# usage: bash utils/retarget_bones_seed_t1.sh <soma-retargeter root>
set -euo pipefail
BFM=$(cd "$(dirname "$0")/.." && pwd)
SRC=${SRC:-$BFM/dataset/bones_seed/soma_uniform/bvh}
DST=${DST:-$BFM/dataset/bones_seed/soma_uniform/csv_booster_t1}
CFG=$(mktemp --suffix .json)
trap 'rm -f "$CFG"' EXIT
unset PYTHONPATH  # keep ROS Humble's Python 3.10 packages out of this env
cd "$1"
python3 "$BFM/utils/make_booster_t1.py" .

for dir in "$SRC"/*/; do
    name=$(basename "$dir")
    n_bvh=$(find "$dir" -name '*.bvh' | wc -l)
    n_csv=$(find "$DST/$name" -name '*.csv' 2>/dev/null | wc -l || true)  # find fails before the folder exists
    if [ "$n_csv" -eq "$n_bvh" ]; then echo "[skip] $name ($n_csv done)"; continue; fi
    cat > "$CFG" <<EOF
{"import_folder": "$dir", "export_folder": "$DST/$name", "batch_size": 100, "retargeter": "Newton",
 "retarget_source": "soma", "retarget_target": "booster_t1", "retarget_source_facing_direction": "Mujoco",
 "extra_robot_paths": ["$PWD/local-robots"]}
EOF
    echo "[run] $name ($n_bvh motions) $(date +%T)"
    uv run python app/bvh_to_csv_converter.py --config "$CFG" 2>&1 | grep -E "Retargeted|ERROR|Error|Traceback" || true
done
echo "done: $(find "$DST" -name '*.csv' | wc -l) / $(find "$SRC" -name '*.bvh' | wc -l) motions"
