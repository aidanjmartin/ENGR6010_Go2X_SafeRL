#!/usr/bin/env bash
# Job 041: find a renderer launch setting that does not crash (jobs 040, 045, 065 ended with a segmentation
# fault in librtx.scenedb while creating the first stage). Each variant renders one camera frame.
set -uo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
NV_ICD=/usr/share/vulkan/icd.d/nvidia_icd.json
run() {  # name, Kit arguments, then env options and assignments
    local name=$1 kit=$2; shift 2
    stamp "variant $name"
    env "$@" timeout 600 python scripts/isaac/render_probe.py --out "$RESULTS_DIR/$name" ${kit:+--kit_args "$kit"} \
        > "$DATA_DIR/$name.log" 2>&1
    local code=$?
    grep -a "render_probe" "$DATA_DIR/$name.log"
    echo "variant $name exit $code" | tee -a "$RESULTS_DIR/summary.txt"
}
SINGLE="--/renderer/multiGpu/enabled=false"
run a_default ""
run b_nvidia_icd "" VK_ICD_FILENAMES=$NV_ICD
run c_no_display "" -u DISPLAY
run d_icd_no_display "" -u DISPLAY VK_ICD_FILENAMES=$NV_ICD
run e_icd_no_display_single_gpu "$SINGLE" -u DISPLAY VK_ICD_FILENAMES=$NV_ICD
cat "$RESULTS_DIR/summary.txt"
stamp "done"
