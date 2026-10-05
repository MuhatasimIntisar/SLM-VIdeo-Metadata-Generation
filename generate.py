"""
Generate per-scene video metadata with a local vision-language model.

Supported model families (native transformers >= 5.17, no trust_remote_code):
  - InternVL3.5  : OpenGVLab/InternVL3_5-{1B,2B,4B,...}-HF
  - Qwen3.5      : Qwen/Qwen3.5-{0.8B,2B,4B,9B,...}   (thinking mode is switched off)
Optional weight quantization with bitsandbytes: --quant 8bit | 4bit (vision encoder kept in bf16).

Output matches scene_metadata_gemini3.8_flash.json: a JSON list of records
    {video, scene, start_sec, end_sec, sample_fps, sampled_frame_count, metadata}
or, on failure,
    {video, scene, start_sec, end_sec, sample_fps, error, raw_output}
"warnings" (and the raw output) are kept on records whose model output needed repairing.
Run statistics (model, precision, GPU, peak memory, seconds/scene, versions) are written to
<output>.run.json next to the output.

Fairness controls, identical for every model:
  - same prompt, schema and output normalisation
  - frames sampled at 1 fps from scene start (as in the Gemini run), capped at --max-frames
  - the same per-frame pixel budget (--frame-pixels, default 448x448) for both families
  - greedy decoding

Examples:
    python generate.py --model OpenGVLab/InternVL3_5-1B-HF --output scene_metadata_internvl3_5_1b.json
    python generate.py --model Qwen/Qwen3.5-9B --quant 4bit --output scene_metadata_qwen3_5_9b_4bit.json
    python generate.py --model Qwen/Qwen3.5-0.8B --video-ids 1 2 --max-scenes-per-video 3 --output sanity.json

The output is saved after every batch; re-running the same command resumes where it stopped.
"""

import argparse
import csv
import json
import math
import os
import platform
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import torch
from PIL import Image

# --------------------------------------------------------------------------- #
# Schema (taken from the Gemini reference file)
# --------------------------------------------------------------------------- #
VISUAL_TAGS = [
    "audience", "building exterior", "city street", "crowd of people", "fire engine",
    "graphic or title card", "indoor scene", "interview", "landscape", "logo",
    "news studio", "office", "outdoor scene", "panel discussion", "person speaking",
    "podium", "politician", "presentation slide", "protest", "public meeting", "road",
    "school", "sign or banner", "stage", "text on screen", "uniform", "vehicle",
]
CONTENT_TYPES = ["interview", "b-roll package", "vox pop", "studio segment", "graphics", "unknown"]
FRAMINGS = ["close-up", "medium", "wide", "unknown"]
SETTINGS = ["studio", "field", "unknown"]

PROMPT = """You are annotating one scene from a local news / community video.
The frames above are sampled from the scene in time order (about one per second).

Return ONLY a JSON object, no prose and no markdown, with exactly the keys in this example.
The values in the example are only illustrations of the format:
{{
  "on_screen_text": ["Jane Smith", "City Council"],
  "visual_tags": ["interview", "person speaking", "indoor scene", "text on screen"],
  "people_count_numeric": 1,
  "description": "A woman speaks to camera in an office, with a caption giving her name and organisation.",
  "geographical_location": {{"area": "", "city": "", "country": ""}},
  "activity": "woman speaking",
  "shot_type": {{"framing": "close-up", "setting": "field"}},
  "content_type": "interview",
  "uncertainty_notes": []
}}

Rules:
- on_screen_text: distinct text legibly visible in the frames, copied verbatim, each item once, at most 10 items; [] if none.
- visual_tags: choose ONLY from: {tags}.
  "text on screen" = any visible caption, name label, headline or other overlaid/visible text.
  "graphic or title card" = a designed graphic, logo animation or title screen rather than camera footage.
- people_count_numeric: number of clearly visible people; 0 if none; -1 for a crowd or more than about 7 people.
- description: one or two sentences describing what happens in the scene.
- geographical_location: fill a part only if visible evidence (signs, captions) supports it, otherwise "".
- activity: short phrase for the main activity; "none" if nothing happens.
- shot_type is an object with BOTH keys: "framing" is one of {framings}; "setting" is one of {settings}
  (studio = filmed in a TV studio; field = filmed on location).
- content_type: one of {content_types}.
- uncertainty_notes: short notes on anything you were unsure about; [] if none."""


def build_prompt():
    q = lambda xs: ", ".join(f'"{x}"' for x in xs)
    return PROMPT.format(tags=q(VISUAL_TAGS), framings=q(FRAMINGS),
                         settings=q(SETTINGS), content_types=q(CONTENT_TYPES))


# --------------------------------------------------------------------------- #
# Frame sampling
# --------------------------------------------------------------------------- #
def sample_times(start, end, fps, max_frames):
    """1/fps-spaced timestamps from scene start (as in the Gemini run), thinned to max_frames."""
    step = 1.0 / fps
    n = int(math.floor((end - start) * fps + 1e-9)) + 1  # matches Gemini's sampled_frame_count
    times = [start + i * step for i in range(n)]
    if max_frames and len(times) > max_frames:
        idx = [round(i * (len(times) - 1) / (max_frames - 1)) for i in range(max_frames)] \
            if max_frames > 1 else [len(times) // 2]
        times = [times[i] for i in idx]
    return times


def prepare_scene(scene, path, fps, max_frames):
    """CPU side (runs in background threads): decode the scene's frames.
    Returns (frames, times_relative_to_scene_start, native_video_fps)."""
    cap = cv2.VideoCapture(str(path))
    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames, rel_times = [], []
        for t in sample_times(scene["start"], scene["end"], fps, max_frames):
            # a timestamp at the very end of the file often fails to decode, so step back a little
            for tt in (min(t, scene["end"] - 0.05), t - 0.5, t - 1.0):
                cap.set(cv2.CAP_PROP_POS_MSEC, max(tt, 0) * 1000.0)
                ok, bgr = cap.read()
                if ok:
                    frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                    rel_times.append(max(tt, 0) - scene["start"])
                    break
    finally:
        cap.release()
    if not frames:
        raise RuntimeError("could not decode any frames")
    return frames, rel_times, native_fps


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
# Vision-side modules kept un-quantized (names differ per family; unknown names are ignored).
# transformers matches these as prefixes of the full module name, so they must start at "model.".
SKIP_QUANT_MODULES = ["model.visual", "model.vision_tower", "model.multi_modal_projector", "lm_head"]


def pick_attention():
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            pass
    return "sdpa"


def quantize_hqq(model, nbits, group_size, dtype):
    """Replace every nn.Linear of the language model with an HQQ-quantized layer, after loading.

    transformers 5.18 cannot apply HqqConfig at load time, so the official bf16 weights are loaded and
    quantized in place. The vision encoder, projector and lm_head stay in bf16 (same as the bnb runs).
    """
    from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
    cfg = BaseQuantizeConfig(nbits=nbits, group_size=group_size)
    n, kept = 0, []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, torch.nn.Linear) and full.startswith("model.language_model."):
                try:  # low-bit packing needs the output size to be a multiple of 8 (1-bit) / 4 (2-bit)
                    q = HQQLinear(child, cfg, compute_dtype=dtype, device=str(child.weight.device), del_orig=False)
                    with torch.no_grad():  # some shapes only fail when the packed weight is first used
                        q(torch.zeros(1, child.in_features, dtype=dtype, device=child.weight.device))
                except Exception as e:
                    kept.append(f"{full} {tuple(child.weight.shape)}: {type(e).__name__}")
                    continue
                # free the original bf16 weights (del_orig=False above keeps them so a layer that fails can
                # stay in bf16; once quantization succeeded they must go, or both copies stay on the GPU)
                for p_name, _ in list(child.named_parameters()):
                    setattr(child, p_name, None)
                del q.linear_layer
                setattr(module, child_name, q)
                del child
                n += 1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if n == 0:
        sys.exit("HQQ: no language-model linear layers could be quantized")
    if kept:
        print(f"[warn] HQQ: {len(kept)} layers left in {dtype} (could not be quantized), e.g. {kept[:3]}")
    return n, len(kept)


def load_model(model_id, quant, attn):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    if not torch.cuda.is_available():
        print("[warn] no GPU found - running on CPU will be very slow")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else \
        (torch.float16 if torch.cuda.is_available() else torch.float32)

    kwargs = dict(dtype=dtype, attn_implementation=attn, device_map="auto")
    if quant == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_skip_modules=SKIP_QUANT_MODULES)
    elif quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=False, llm_int8_skip_modules=SKIP_QUANT_MODULES)

    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs).eval()
    if quant.startswith("hqq"):
        n, n_kept = quantize_hqq(model, int(quant[3:]), 64, dtype)
        print(f"[info] HQQ {quant[3:]}-bit (group size 64): quantized {n} language-model linear layers"
              + (f", {n_kept} left unquantized" if n_kept else ""))
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    family = model.config.model_type  # "internvl" or "qwen3_5"
    if family == "internvl":
        # The -HF checkpoints ship no video config, so the video processor would fall back to 384x384,
        # which breaks InternVL's pixel shuffle. Use the vision encoder's native size (448x448).
        h, w = model.config.vision_config.image_size
        processor.internvl_frame_size = {"height": h, "width": w}
    if family not in ("internvl", "qwen3_5", "qwen3_5_moe", "qwen3_vl"):
        print(f"[warn] model type '{family}' has not been tested with this script")
    return model, processor, family, dtype


# --------------------------------------------------------------------------- #
# Batched generation
# --------------------------------------------------------------------------- #
def build_inputs(processor, family, batch, prompt, frame_pixels):
    """batch: list of (frames, rel_times, native_fps). Returns processor outputs (CPU tensors)."""
    from transformers.video_utils import VideoMetadata

    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    # enable_thinking is read by the Qwen3.5 template and ignored by others
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
    texts = [text] * len(batch)

    if family.startswith("qwen3"):
        videos, metas = [], []
        for frames, rel_times, native_fps in batch:
            if len(frames) == 1:  # Qwen merges frames in pairs and needs at least 2
                frames, rel_times = frames * 2, rel_times * 2
            videos.append(frames)
            metas.append(VideoMetadata(total_num_frames=len(frames), fps=native_fps,
                                       frames_indices=[round(t * native_fps) for t in rel_times]))
        n = max(len(v) for v in videos)
        return processor(text=texts, videos=videos, video_metadata=metas, do_sample_frames=False,
                         size={"shortest_edge": 128 * 32 * 32, "longest_edge": n * frame_pixels},
                         cap_pixels_per_frame=False, padding=True, return_tensors="pt")

    # InternVL: each frame is resized to the model's native 448x448 tile ("Frame1: <image> ...")
    return processor(text=texts, videos=[b[0] for b in batch], do_sample_frames=False,
                     size=processor.internvl_frame_size, padding=True, return_tensors="pt")


def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def generate_batch(model, processor, family, dtype, batch, prompt, gen_cfg, frame_pixels):
    """One generate() call for several scenes; halves the batch on out-of-memory."""
    try:
        inputs = build_inputs(processor, family, batch, prompt, frame_pixels).to(model.device)
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(dtype)
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_cfg, pad_token_id=processor.tokenizer.pad_token_id)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return [strip_thinking(t) for t in processor.batch_decode(new_tokens, skip_special_tokens=True)]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(batch) == 1:
            raise
        mid = len(batch) // 2
        print(f"[warn] OOM on batch of {len(batch)}, splitting", flush=True)
        return (generate_batch(model, processor, family, dtype, batch[:mid], prompt, gen_cfg, frame_pixels)
                + generate_batch(model, processor, family, dtype, batch[mid:], prompt, gen_cfg, frame_pixels))


# --------------------------------------------------------------------------- #
# Parsing + normalisation into the reference schema
# --------------------------------------------------------------------------- #
def extract_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e <= s:
        raise ValueError("no JSON object in output")
    chunk = text[s:e + 1]
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        chunk = re.sub(r",\s*([}\]])", r"\1", chunk)  # trailing commas
        return json.loads(chunk)


def _str(x):
    return x.strip() if isinstance(x, str) else ("" if x is None else str(x).strip())


def _str_list(x):
    if isinstance(x, str):
        x = [x] if x.strip() else []
    return [_str(v) for v in (x or []) if _str(v)] if isinstance(x, list) else []


def _blank_unknown(x):
    """Reference uses "" for unknown location parts; models often write "unknown"/"N/A"."""
    v = _str(x)
    return "" if v.lower() in {"unknown", "n/a", "na", "none", "null", "not visible", "-"} else v


def _setting(value, warnings):
    """Setting is binary (studio vs on location). 1B models often answer with a location type
    ("indoor", "office", "city street"); anything that is not a studio means filmed on location."""
    v = _str(value).lower()
    if v in SETTINGS:
        return v
    if not v or v in {"none", "null", "n/a"}:
        warnings.append(f"setting: '{value}' -> 'unknown'")
        return "unknown"
    mapped = "studio" if "studio" in v else "field"
    warnings.append(f"setting: '{value}' -> '{mapped}'")
    return mapped


COUNTRY_ALIASES = {"uk": "United Kingdom", "u.k.": "United Kingdom", "great britain": "United Kingdom",
                   "britain": "United Kingdom", "republic of ireland": "Ireland"}


def _enum(value, allowed, field, warnings):
    v = _str(value).lower()
    if v in allowed:
        return v
    warnings.append(f"{field}: '{value}' -> 'unknown'")
    return "unknown"


def normalise(raw):
    """Coerce model output into the exact reference schema. Returns (metadata, warnings)."""
    w = []
    expected = {"on_screen_text", "visual_tags", "people_count_numeric", "description",
                "geographical_location", "activity", "shot_type", "content_type", "uncertainty_notes"}
    missing = expected - set(raw)
    if missing:
        w.append(f"missing keys: {sorted(missing)}")

    tags, dropped = [], []
    for t in _str_list(raw.get("visual_tags")):
        t = t.lower()
        (tags if t in VISUAL_TAGS else dropped).append(t)
    if dropped:
        w.append(f"out-of-vocab tags dropped: {dropped}")

    try:
        people = int(raw.get("people_count_numeric", -1))
    except (TypeError, ValueError):
        w.append(f"people_count_numeric: '{raw.get('people_count_numeric')}' -> -1")
        people = -1

    geo = raw.get("geographical_location") or {}
    geo = geo if isinstance(geo, dict) else {}
    shot = raw.get("shot_type") or {}
    if isinstance(shot, str):  # e.g. "shot_type": "close-up"
        shot = {"framing": shot} if shot.strip().lower() in FRAMINGS else {"setting": shot}
    shot = dict(shot) if isinstance(shot, dict) else {}
    for k in ("framing", "setting"):  # keys sometimes emitted at top level
        if not shot.get(k) and raw.get(k):
            shot[k] = raw[k]

    meta = {
        "on_screen_text": _str_list(raw.get("on_screen_text")),
        "visual_tags": list(dict.fromkeys(tags)),
        "people_count_numeric": people,
        "description": _str(raw.get("description")),
        "geographical_location": {
            "area": _blank_unknown(geo.get("area")),
            "city": _blank_unknown(geo.get("city")),
            "country": COUNTRY_ALIASES.get(_blank_unknown(geo.get("country")).lower(),
                                           _blank_unknown(geo.get("country"))),
        },
        "activity": _str(raw.get("activity")).lower() or "none",
        "shot_type": {
            "framing": _enum(shot.get("framing"), FRAMINGS, "framing", w),
            "setting": _setting(shot.get("setting"), w),
        },
        "content_type": _enum(raw.get("content_type"), CONTENT_TYPES, "content_type", w),
        "uncertainty_notes": _str_list(raw.get("uncertainty_notes")),
    }
    if not meta["description"]:
        w.append("empty description")
    return meta, w


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def load_scenes(csv_path):
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        return [dict(video=r["video"], scene=int(r["scene"]),
                     start=float(r["start_sec"]), end=float(r["end_sec"]))
                for r in csv.DictReader(f)]


def video_number(name):
    m = re.match(r"\s*(\d+)\.", name)
    return int(m.group(1)) if m else None


def save_json(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def make_batches(todo, args):
    """Group scenes with the same number of frames (needed for InternVL, less padding for all)."""
    by_len = defaultdict(list)
    for s in todo:
        by_len[len(sample_times(s["start"], s["end"], args.fps, args.max_frames))].append(s)
    return [group[i:i + args.batch_size] for _, group in sorted(by_len.items())
            for i in range(0, len(group), args.batch_size)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face model id or local path")
    ap.add_argument("--quant", choices=["none", "8bit", "4bit", "hqq8", "hqq4", "hqq2", "hqq1"], default="none",
                    help="8bit/4bit = bitsandbytes at load; hqqN = HQQ N-bit applied after loading bf16 weights")
    ap.add_argument("--output", required=True)
    ap.add_argument("--video-dir", default="VIDEO FILES")
    ap.add_argument("--scenes-csv", default="_all_scenes.csv")
    ap.add_argument("--fps", type=float, default=1.0, help="sampling rate before capping (Gemini used 1.0)")
    ap.add_argument("--max-frames", type=int, default=16, help="max frames per scene")
    ap.add_argument("--frame-pixels", type=int, default=448 * 448, help="per-frame pixel budget (Qwen)")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=8, help="scenes per generate() call (halves on OOM)")
    ap.add_argument("--workers", type=int, default=8, help="CPU threads for frame decoding")
    ap.add_argument("--retries", type=int, default=1, help="extra attempts (sampled) if JSON parsing fails")
    ap.add_argument("--attn", default=None, help="attention implementation (default: flash_attention_2 if "
                                                 "available on the GPU, else sdpa)")
    ap.add_argument("--video-ids", type=int, nargs="*", help="only these video numbers, e.g. 1 2 15")
    ap.add_argument("--max-scenes-per-video", type=int, help="first N scenes of each video")
    ap.add_argument("--limit", type=int, help="stop after N scenes in total")
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing output file")
    args = ap.parse_args()

    import transformers
    if int(transformers.__version__.split(".")[0]) < 5:
        sys.exit(f"transformers {transformers.__version__} is too old; run: pip install -r requirements.txt")

    video_dir, out_path = Path(args.video_dir), Path(args.output)
    run_path = out_path.with_suffix(".run.json")
    all_scenes = load_scenes(args.scenes_csv)
    order = {(s["video"] + ".mp4", s["scene"]): i for i, s in enumerate(all_scenes)}

    on_disk = {p.stem: p for p in video_dir.glob("*.mp4")}
    missing = sorted({s["video"] for s in all_scenes} - set(on_disk))
    if missing:
        print(f"[info] {len(missing)} CSV videos not in {video_dir}, skipped: {missing}")
    scenes = [s for s in all_scenes if s["video"] in on_disk]
    if args.video_ids:
        scenes = [s for s in scenes if video_number(s["video"]) in set(args.video_ids)]
    if args.max_scenes_per_video:
        scenes = [s for s in scenes if s["scene"] <= args.max_scenes_per_video]
    if args.limit:
        scenes = scenes[:args.limit]

    records = {}
    if out_path.exists() and not args.no_resume:
        prev = json.load(open(run_path, encoding="utf-8")) if run_path.exists() else {}
        if prev.get("model") != args.model or prev.get("quant") != args.quant:
            sys.exit(f"{out_path} exists but was not produced by --model {args.model} --quant {args.quant} "
                     f"(run file says: {prev.get('model')}, {prev.get('quant')}). "
                     f"Use a different --output, or --no-resume to overwrite it.")
        for r in json.load(open(out_path, encoding="utf-8")):
            records[(r["video"], r["scene"])] = r
    todo = [s for s in scenes if "metadata" not in records.get((s["video"] + ".mp4", s["scene"]), {})]
    print(f"[info] {len(scenes)} scenes selected, {len(scenes) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        return

    attn = args.attn or pick_attention()
    t_load = time.time()
    model, processor, family, dtype = load_model(args.model, args.quant, attn)
    load_s = time.time() - t_load
    loaded_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
    print(f"[info] {args.model} ({family}) quant={args.quant} dtype={dtype} attn={attn} "
          f"loaded in {load_s:.0f}s, weights {model.get_memory_footprint() / 1e9:.2f} GB")

    run = json.load(open(run_path, encoding="utf-8")) if run_path.exists() and not args.no_resume else {}
    run.update({
        "model": args.model, "family": family, "quant": args.quant, "dtype": str(dtype), "attention": attn,
        "weights_memory_gb": round(model.get_memory_footprint() / 1e9, 2),
        "loaded_gpu_memory_gb": round(loaded_mem / 1e9, 2) if loaded_mem is not None else None,
        "settings": {k: getattr(args, k) for k in ("fps", "max_frames", "frame_pixels", "max_new_tokens",
                                                    "batch_size", "retries")},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
    })
    run.setdefault("sessions", [])
    save_json(run, run_path)

    prompt = build_prompt()
    greedy = dict(max_new_tokens=args.max_new_tokens, do_sample=False)
    sampled = dict(max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9,
                   repetition_penalty=1.1)  # retry only: breaks repetition loops
    batches = make_batches(todo, args)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def prepare(batch):
        futs = [pool.submit(prepare_scene, s, on_disk[s["video"]], args.fps, args.max_frames) for s in batch]
        out = []
        for f in futs:
            try:
                out.append(f.result())
            except Exception as e:
                out.append(e)
        return out

    n_ok = n_err = done = 0
    gen_seconds = 0.0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, ThreadPoolExecutor(max_workers=1) as prefetch:
        next_inputs = prefetch.submit(prepare, batches[0])
        for bi, batch in enumerate(batches):
            inputs = next_inputs.result()
            if bi + 1 < len(batches):  # decode the next batch while the GPU works on this one
                next_inputs = prefetch.submit(prepare, batches[bi + 1])

            recs = [dict(video=s["video"] + ".mp4", scene=s["scene"], start_sec=s["start"], end_sec=s["end"],
                         sample_fps=args.fps) for s in batch]
            pending, raws, last_err = [], {}, {}
            for j, x in enumerate(inputs):
                if isinstance(x, Exception):
                    recs[j].update(error=f"{type(x).__name__}: {x}", raw_output="")
                else:
                    recs[j]["sampled_frame_count"] = len(x[0])
                    pending.append(j)

            for attempt in range(args.retries + 1):
                if not pending:
                    break
                # sub-batches with identical frame counts (a decode failure can change a scene's count)
                groups = defaultdict(list)
                for j in pending:
                    groups[len(inputs[j][0])].append(j)
                still = []
                for js in groups.values():
                    t_gen = time.time()
                    try:
                        outs = generate_batch(model, processor, family, dtype, [inputs[j] for j in js], prompt,
                                              greedy if attempt == 0 else sampled, args.frame_pixels)
                    except Exception as e:  # e.g. OOM on a single scene
                        msg = "CUDA out of memory (try a lower --max-frames)" \
                            if isinstance(e, torch.cuda.OutOfMemoryError) else f"{type(e).__name__}: {e}"
                        for j in js:
                            recs[j]["error"] = msg
                        continue
                    finally:
                        gen_seconds += time.time() - t_gen
                    for j, raw in zip(js, outs):
                        raws[j] = raw
                        try:
                            meta, warnings = normalise(extract_json(raw))
                            recs[j]["metadata"] = meta
                            recs[j].pop("error", None)
                            if warnings:
                                recs[j]["warnings"] = warnings
                                recs[j]["raw_output"] = raw
                        except (ValueError, json.JSONDecodeError) as e:
                            last_err[j] = str(e)
                            still.append(j)
                pending = still
            for j in pending:
                recs[j]["error"] = f"unparseable JSON after {args.retries + 1} attempts: {last_err.get(j, '')}"

            for j, r in enumerate(recs):
                if "metadata" in r:
                    n_ok += 1
                else:
                    r.pop("sampled_frame_count", None)
                    r.setdefault("error", "unknown error")
                    r["raw_output"] = raws.get(j, "")
                    n_err += 1
                records[(r["video"], r["scene"])] = r
            save_json(sorted(records.values(), key=lambda r: order.get((r["video"], r["scene"]), 1e9)), out_path)

            done += len(batch)
            rate = (time.time() - t0) / done
            print(f"[{done}/{len(todo)}] batch of {len(batch)} ({len(inputs[0][0]) if not isinstance(inputs[0], Exception) else '?'} frames): "
                  f"{n_ok} ok, {n_err} errors so far ({rate:.1f}s/scene, ~{rate * (len(todo) - done) / 60:.0f} min left)",
                  flush=True)

    # run statistics (appended per session so resumed runs keep their history)
    session = {
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenes_run": done, "ok": n_ok, "errors": n_err,
        "wall_seconds": round(time.time() - t0, 1), "generate_seconds": round(gen_seconds, 1),
        "seconds_per_scene": round((time.time() - t0) / max(done, 1), 3),
        "load_seconds": round(load_s, 1),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
    }
    run["sessions"].append(session)
    save_json(run, run_path)
    print(f"[done] {n_ok} ok, {n_err} errors -> {out_path} (run stats: {run_path})")


if __name__ == "__main__":
    sys.exit(main())
