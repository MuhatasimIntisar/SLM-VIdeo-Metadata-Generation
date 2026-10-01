#!/usr/bin/env bash
# Slurm array job: one experiment (row of experiments.tsv) per array task.
#   sbatch --array=1-9 slurm_job.sh          # all experiments
#   sbatch --array=7-9 slurm_job.sh          # only the Qwen3.5-9B quantization set
# Adjust partition / GPU / module lines to the cluster before submitting.
#SBATCH --job-name=vlm-meta
#SBATCH --partition=gpu                # <- cluster's GPU partition
#SBATCH --gres=gpu:1                   # one GPU per experiment
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%a_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# module load python/3.11 cuda/12.4    # <- cluster-specific
source venv/bin/activate

# Compute nodes are often offline: download models on the login node first (see README),
# then run from the local Hugging Face cache.
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

bash run_experiment.sh "$SLURM_ARRAY_TASK_ID" --workers "$SLURM_CPUS_PER_TASK"
