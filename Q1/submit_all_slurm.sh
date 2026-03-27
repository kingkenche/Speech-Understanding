#!/bin/bash
# This script submits all run_*.slurm files in the current directory to SLURM

for f in run_*.slurm; do
    if [ -f "$f" ]; then
        echo "Submitting $f"
        sbatch "$f"
    fi
done
