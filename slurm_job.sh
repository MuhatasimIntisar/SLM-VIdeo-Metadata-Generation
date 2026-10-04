#!/usr/bin/env bash
# Slurm array job: one experiment (row of experiments.tsv) per array task.
#   sbatch --array=1-9 slurm_job.sh          # all experiments
#   sbatch --array=7-9 slurm_job.sh          # only the Qwen3.5-9B quantization set
# Adjust partition / GPU / module lines to the cluster before submitting.
#SBATCH --job-name=vlm-meta
#SBATCH --partition=k2-gpu-a100        # Kelvin2 A100 80GB nodes (check: sinfo -p k2-gpu-a100)
#SBATCH --gres=gpu:a100:1              # one GPU per experiment (type name: sinfo -p k2-gpu-a100 -o "%G")
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                    # 27B models are loaded in bf16 before HQQ quantization
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%a_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load python3/3.10.5/gcc-9.3.0   # Kelvin2; CUDA comes with the pip torch wheel
source venv/bin/activate

# Compute nodes are often offline: download models on the login node first (see README),
# then run from the local Hugging Face cache.
export HF_HOME=/mnt/scratch2/users/$USER/hf_cache   # models live on scratch (home quota is 50 GB)
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

bash run_experiment.sh "$SLURM_ARRAY_TASK_ID" --workers "$SLURM_CPUS_PER_TASK" "$@"   # extra sbatch args, e.g. --video-dir
