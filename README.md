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
| `run_experiment.sh` | generate → evaluate → compare for one experiment (by name or row number) |
| `slurm_job.sh` | Slurm array wrapper around `run_experiment.sh` |

## Setup (once, on the login node)
```bash
python -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # match the cluster CUDA
pip install -r requirements.txt
# optional, faster: pip install flash-attn flash-linear-attention causal-conv1d --no-build-isolation

# compute nodes are usually offline: fetch models + NLTK data now
for m in $(cut -f2 experiments.tsv | tail -n +2 | sort -u); do hf download "$m"; done
python -c "import nltk; nltk.download('wordnet')"
```
Data needed in this folder: `VIDEO FILES/` (or pass `--video-dir`), `_all_scenes.csv`,
`scene_metadata_gemini3.8_flash.json`.

## Run
```bash
python make_baseline.py && python evaluate.py --prediction scene_metadata_baseline_majority.json

# quick check that a model works (6 scenes) before submitting everything
python generate.py --model Qwen/Qwen3.5-0.8B --video-ids 1 2 --max-scenes-per-video 3 --output sanity.json

sbatch --array=1-9 slurm_job.sh        # all experiments, one GPU each (edit partition/modules first)
# or interactively:  bash run_experiment.sh qwen3_5_4b
```
After each experiment `results_table.md` is regenerated with every finished run, baseline first.

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
