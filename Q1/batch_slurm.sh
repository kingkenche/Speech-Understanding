
# 2. Create and submit a SLURM job for each .wav file
for wav in $DATA_DIR/*.wav; do
    base=$(basename "$wav" .wav)
    slurm_script="run_${base}.slurm"
    cat > "$slurm_script" <<EOF
#!/bin/bash
#SBATCH --job-name=libri_${base}
#SBATCH --output=${RESULTS_DIR}/${base}_slurm.out
#SBATCH --error=${RESULTS_DIR}/${base}_slurm.err
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

cd $(pwd)
# source venv/bin/activate  # (removed to avoid error if venv does not exist)
python run_all.py --audio $wav --out_dir ${RESULTS_DIR}/${base}
EOF
    sbatch "$slurm_script"
    echo "Submitted SLURM job for $wav"
done
