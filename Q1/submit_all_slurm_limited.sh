#!/bin/bash
# This script submits all run_*.slurm files, but waits if you hit the SLURM job submission limit.
# Set your max allowed jobs here (adjust if needed):
MAX_JOBS=50
SLEEP_TIME=60  # seconds to wait before checking again

for f in run_*.slurm; do
    # Skip the literal file 'run_*.slurm' if it exists (from failed wildcard expansion)
    if [ "$f" = "run_*.slurm" ]; then
        continue
    fi
    if [ -f "$f" ]; then
        # Wait until the number of your jobs is below the limit
        while true; do
            JOBS=$(squeue -u "$USER" | grep -c '^[0-9]')
            if [ "$JOBS" -lt "$MAX_JOBS" ]; then
                break
            fi
            echo "Job limit reached ($JOBS jobs). Waiting $SLEEP_TIME seconds..."
            sleep $SLEEP_TIME
        done
        echo "Submitting $f"
        sbatch "$f"
    fi
done
