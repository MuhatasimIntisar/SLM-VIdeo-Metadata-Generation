#!/usr/bin/env bash
# Run one experiment from experiments.tsv (generation + evaluation).
#   bash run_experiment.sh qwen3_5_4b              # by name
#   bash run_experiment.sh 5                       # by row number (1 = first experiment)
# Extra arguments are passed to generate.py, e.g. --batch-size 4 or --video-dir /path/to/videos
set -euo pipefail
cd "$(dirname "$0")"

key="$1"; shift || true
if [[ "$key" =~ ^[0-9]+$ ]]; then
  row=$(awk -F'\t' -v n="$key" 'NR == n + 1' experiments.tsv)
else
  row=$(awk -F'\t' -v k="$key" '$1 == k' experiments.tsv)
fi
[[ -z "$row" ]] && { echo "No experiment '$key' in experiments.tsv"; exit 1; }
IFS=$'\t' read -r name model quant <<< "$row"

echo "=== $name: $model (quant=$quant) on $(hostname) at $(date)"
python generate.py --model "$model" --quant "$quant" --output "scene_metadata_${name}.json" "$@"
python evaluate.py --prediction "scene_metadata_${name}.json"
python compare.py
