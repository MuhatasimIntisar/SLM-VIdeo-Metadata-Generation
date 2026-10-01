# Scene metadata generation with small VLMs: benchmark

Each model watches every PySceneDetect scene (`_all_scenes.csv`) of the videos in `VIDEO FILES/` and
writes metadata in the same JSON schema as `scene_metadata_gemini3.8_flash.json`. The output is then
scored against the Gemini output (`evaluate.py`).

**Set 1, model families × size:** InternVL3.5 (1B, 2B, 4B) and Qwen3.5 (0.8B, 2B, 4B), bf16.
**Set 2, quantization:** Qwen3.5-9B at bf16, 8-bit and 4-bit (bitsandbytes LLM.int8 / NF4; the vision
encoder stays in bf16, only the language model is quantized).

All runs are listed in `experiments.tsv`. A majority-class baseline (`make_baseline.py`) is the floor.

## Files
| File | Purpose |
|---|---|
| `generate.py` | Run one model over all scenes → `scene_metadata_<name>.json` + `scene_metadata_<name>.run.json` (GPU, memory, speed, versions) |
| `evaluate.py` | Score one output against Gemini → `eval_scene_metadata_<name>/summary.json`, `per_scene.csv` |
| `make_baseline.py` | Build the no-video majority baseline (`scene_metadata_baseline_majority.json`) |
| `compare.py` | Collect all evaluated runs into `results_table.csv` / `results_table.md` |
| `experiments.tsv` | The experiment list (name, model id, quantization) |
| `run_experiment.sh` | generate metadata for one experiment (by name or row number) |
| `slurm_job.sh` | Slurm array wrapper around `run_experiment.sh` |

## Cluster: generation only
Needed in this folder: `VIDEO FILES/` (or pass `--video-dir`) and `_all_scenes.csv`.

```bash
# once, on the login node
python -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # match the cluster CUDA
pip install -r requirements.txt
# optional, faster: pip install flash-attn flash-linear-attention causal-conv1d --no-build-isolation
for m in $(cut -f2 experiments.tsv | tail -n +2 | sort -u); do hf download "$m"; done   # compute nodes are offline

# sanity check on a GPU node (send these outputs back before the full run)
python generate.py --model OpenGVLab/InternVL3_5-1B-HF --video-ids 1 2 --max-scenes-per-video 3 --output sanity_internvl.json
python generate.py --model Qwen/Qwen3.5-0.8B --video-ids 1 2 --max-scenes-per-video 3 --output sanity_qwen.json
python generate.py --model Qwen/Qwen3.5-9B --quant 4bit --video-ids 1 --max-scenes-per-video 2 --output sanity_q4.json

# full run (edit partition/modules in slurm_job.sh first)
sbatch --array=1-9 slurm_job.sh
```
Send back every `scene_metadata_*.json`, `scene_metadata_*.run.json` and the `logs/` folder.

## Evaluation (done locally, not on the cluster)
```bash
pip install rouge-score pycocoevalcap nltk
python make_baseline.py && python evaluate.py --prediction scene_metadata_baseline_majority.json
for f in scene_metadata_internvl*.json scene_metadata_qwen*.json; do
  case "$f" in *.run.json) ;; *) python evaluate.py --prediction "$f";; esac
done
python compare.py
```

## Notes
- **Resuming:** outputs are saved after every batch; re-running the same command continues where it
  stopped (e.g. after a Slurm time limit). The script refuses to resume from an output file produced by a
  different model or quantization; use a new `--output` or `--no-resume`.
- **Identical for every model:** prompt, schema and output normalisation; frames sampled at 1 fps from
  scene start, capped at 16 (`--max-frames`); about 448×448 pixels per frame; greedy decoding;
  Qwen3.5 thinking mode switched off.
- **Out of memory:** a batch that runs out of GPU memory is split in half automatically. Lower
  `--batch-size` (default 8) if it happens constantly.
- **Repairs:** records whose model output needed repairing (out-of-vocabulary tags, invalid enum values,
  missing keys) keep `warnings` and `raw_output`; the `repaired` column counts them.
- **Untested here:** the pipeline was tested end-to-end with tiny random InternVL/Qwen3.5 models on CPU;
  the real checkpoints and bitsandbytes quantization run for the first time on the cluster, so run the
  sanity command for one InternVL and one Qwen model first.
