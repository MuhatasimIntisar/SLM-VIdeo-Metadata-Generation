#!/usr/bin/env bash
# Download the GGUF files listed in gguf_experiments.tsv (+ vision encoder) into $HF_HOME.
# Run in the background so it survives logging out:
#   nohup bash download_gguf.sh > download.log 2>&1 &
#   tail -5 download.log            # finished when it prints "ALL DONE"
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
: "${HF_HOME:?HF_HOME is not set - run: source ~/.bashrc}"
# The xet downloader is needed for large files but its default parallelism uses enough memory to get
# killed on the login nodes, so download one file at a time with fewer parallel chunks.
export HF_XET_HIGH_PERFORMANCE=0
export HF_XET_NUM_CONCURRENT_RANGE_GETS=4
echo "HF_HOME=$HF_HOME"
for repo in unsloth/Qwen3.5-9B-GGUF unsloth/Qwen3.6-27B-GGUF; do
  echo "=== $repo mmproj-BF16.gguf"
  hf download "$repo" mmproj-BF16.gguf --max-workers 1
  awk -F'\t' -v r="$repo" 'NR > 1 && $2 == r {print $3}' gguf_experiments.tsv | while read -r pattern; do
    echo "=== $repo $pattern  ($(date))"
    hf download "$repo" --include "$pattern" --max-workers 1
  done
done
echo "ALL DONE $(date)"
