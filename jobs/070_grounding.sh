#!/usr/bin/env bash
# Job 070: Task 5 grounding of text prompts on the rendered scenes from job 045 with OWLv2.
set -euo pipefail
source "$(dirname "$0")/common.sh"
activate_venv
export HF_HOME="$JOB_CACHE/hf"
python -m pip install -q "transformers==4.57.1"
python -c "import transformers, huggingface_hub; print('transformers', transformers.__version__, 'hub', huggingface_hub.__version__)"
python scripts/ground_prompts.py --scenes "$DATA_ROOT/045-render-scenes/scenes" --out "$RESULTS_DIR"
stamp "done"
