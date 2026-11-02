"""Exit after a job's outputs are written.

In job 010 (attempt 5) `SimulationApp.close()` hung for more than 8 minutes after all work was done, so the
job never ended. Scripts call `finish()` instead: flush, then end the process without running Kit's shutdown.
"""
import os
import sys


def finish(code=0):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
