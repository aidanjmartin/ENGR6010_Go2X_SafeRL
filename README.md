# Dell job results

The runner on the Dell T2 tower writes one folder per job here:
`status.json`, `log.txt` (the end of the job log), and `output/` (files the job
wrote to `$RESULTS_DIR`). `_runner/heartbeat.json` shows the runner's health.
Only the runner writes to this branch.
