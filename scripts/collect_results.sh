#!/usr/bin/env bash
# Copy a finished job's outputs from the dell-results branch into results/phase2/<dest>/.
#   bash scripts/collect_results.sh 040-walk-eval walk
set -euo pipefail
job=$1; dest=results/phase2/$2
git fetch -q origin dell-results
mkdir -p "$dest"
tmp=$(mktemp -d)
git archive origin/dell-results "$job/output" "$job/status.json" | tar -x -C "$tmp"
cp -r "$tmp/$job/output/." "$dest/"
python3 - "$tmp/$job/status.json" "$dest/job_status_$job.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
keep = {k: s.get(k) for k in ("job", "attempt", "commit", "status", "started", "finished", "exit_code", "kept_on_dell")}
json.dump(keep, open(sys.argv[2], "w"), indent=1)
PY
rm -rf "$tmp"
echo "copied $job -> $dest (results commit $(git rev-parse --short origin/dell-results))"; ls "$dest"
