#!/usr/bin/env bash
# Slurm array job for the GGUF (llama.cpp) runs: one row of gguf_experiments.tsv per array task.
#   sbatch --array=1-8 slurm_gguf.sh         # all GGUF runs
#   sbatch --array=1 slurm_gguf.sh --video-ids 1 --max-scenes-per-video 3   # quick test of row 1
# Needs llama.cpp built with CUDA (see README) and the GGUF files in the HF cache.
#SBATCH --job-name=vlm-gguf
#SBATCH --partition=k2-gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G                     # the 27B BF16 file (54 GB) is read through host memory
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%a_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load libs/nvidia-cuda/12.8.0/bin compilers/gcc/13.2.0   # CUDA + C++ runtime llama-server was built with
source venv/bin/activate
export HF_HOME=/mnt/scratch2/users/$USER/hf_cache
export HF_HUB_OFFLINE=1
LLAMA_SERVER=${LLAMA_SERVER:-/mnt/scratch2/users/$USER/llama.cpp/build/bin/llama-server}

row=$(awk -F'\t' -v n="$SLURM_ARRAY_TASK_ID" 'NR == n + 1' gguf_experiments.tsv)
[[ -z "$row" ]] && { echo "No row $SLURM_ARRAY_TASK_ID in gguf_experiments.tsv"; exit 1; }
IFS=$'\t' read -r name repo file quant <<< "$row"
echo "=== $name: $repo $file ($quant) on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python generate_gguf.py --repo "$repo" --file "$file" --quant "$quant" --server-bin "$LLAMA_SERVER" \
    --port $((20000 + SLURM_JOB_ID % 20000)) --output "scene_metadata_${name}.json" "$@"
