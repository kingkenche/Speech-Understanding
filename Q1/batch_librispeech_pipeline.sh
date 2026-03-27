#!/bin/bash
# batch_librispeech_pipeline.sh
# 1. Convert all .flac files to 16kHz mono .wav in data/
# 2. Update data/manifest.txt
# 3. Submit a SLURM job for each .wav file

set -e

LIBRISPEECH_DIR="LibriSpeech/test-clean"
DATA_DIR="data"
RESULTS_DIR="results"
MANIFEST="$DATA_DIR/manifest.txt"
SLURM_TEMPLATE="run_pipeline_template.slurm"

# 1. Convert all .flac files to .wav
find "$LIBRISPEECH_DIR" -name "*.flac" | while read flac; do
    wav="$DATA_DIR/$(basename "${flac%.flac}.wav")"
    if [ ! -f "$wav" ]; then
        ffmpeg -y -i "$flac" -ar 16000 -ac 1 "$wav"
    fi
done

echo "# Auto-generated manifest for LibriSpeech test-clean" > "$MANIFEST"
echo "# filename | source | duration_s | sr_hz | description" >> "$MANIFEST"

# 2. Update manifest with all .wav files
total=$(ls $DATA_DIR/*.wav | wc -l)
count=0
for wav in $DATA_DIR/*.wav; do
    # Get duration in seconds
    dur=$(soxi -D "$wav" 2>/dev/null || ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$wav")
    echo "$wav | LibriSpeech test-clean | $dur | 16000 | Converted from .flac" >> "$MANIFEST"
    count=$((count+1))
    echo "[$count/$total] Added $wav to manifest."
done

# 3. Create and submit a SLURM job for each .wav file
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
source venv/bin/activate
python run_all.py --audio $wav --out_dir ${RESULTS_DIR}/${base}
EOF
    sbatch "$slurm_script"
    echo "Submitted SLURM job for $wav"
done
