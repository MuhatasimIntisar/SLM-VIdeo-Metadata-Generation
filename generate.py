"""
Generate per-scene video metadata with a local VLM (default: InternVL3.5-1B).

Output matches scene_metadata_gemini3.8_flash.json: a JSON list of records
    {video, scene, start_sec, end_sec, sample_fps, sampled_frame_count, metadata}
or, on failure,
    {video, scene, start_sec, end_sec, sample_fps, error, raw_output}
A record may also carry "warnings" when the model output needed repairing
(invalid enum values, out-of-vocabulary tags, missing keys). evaluate.py ignores it.

Frame sampling mirrors the Gemini run (one frame per second from scene start),
but is capped at --max-frames (uniformly thinned) because a 1B model cannot take
Gemini's 200+ frames. sampled_frame_count records what the model actually saw.

Sanity run (local):
    python generate.py --video-ids 1 2 --max-scenes-per-video 3 --output sanity_internvl.json

Speed: --batch-size scenes go through the GPU in one generate() call (halved automatically
on out-of-memory), while the next batch is decoded on CPU threads in the background.
Scenes are processed shortest-first to keep padding low; the output is still saved in CSV order.

The output is written after every scene, so an interrupted run can be resumed
by re-running the same command (scenes that already have metadata are skipped).
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

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

Return ONLY a JSON object, no prose and no markdown, with exactly these keys:
{{
  "on_screen_text": [list of strings: text legibly visible in the frames, copied verbatim; [] if none],
  "visual_tags": [list of tags chosen ONLY from: {tags}],
  "people_count_numeric": integer: number of clearly visible people; 0 if none; -1 if a crowd or not countable,
  "description": "one or two sentences describing what happens in the scene",
  "geographical_location": {{"area": "", "city": "", "country": ""}}  (fill only if visible evidence such as signs or captions supports it, otherwise leave ""),
  "activity": "short phrase for the main activity, e.g. person speaking; 'none' if nothing happens",
  "shot_type": {{"framing": one of {framings}, "setting": one of {settings}}},
  "content_type": one of {content_types},
  "uncertainty_notes": [list of short strings noting anything you were unsure about; [] if none]
}}"""


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


def read_frames(cap, times, end):
    frames = []
    for t in times:
        # a timestamp at the very end of the file often fails to decode, so step back a little
        for tt in (min(t, end - 0.05), t - 0.5, t - 1.0):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(tt, 0) * 1000.0)
            ok, bgr = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                break
    return frames


# --------------------------------------------------------------------------- #
# InternVL preprocessing (one 448px tile per frame, as in the official video example)
# --------------------------------------------------------------------------- #
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(size):
    return T.Compose([
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model(model_id, device):
    import transformers
    if int(transformers.__version__.split(".")[0]) >= 5:
        sys.exit(f"transformers {transformers.__version__} is installed, but InternVL's remote code needs 4.x.\n"
                 f'Run: pip install "transformers>=4.52.1,<5"')
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True,  # falls back automatically if flash-attn is missing
    ).eval().to(device)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    return model, tok, dtype


def run_batch(model, tok, device, dtype, batch, prompt, gen_cfg):
    """One generate() call for several scenes. batch = list of pixel_values tensors (frames x 3 x H x W).

    Same prompt construction as InternVL's model.chat() (Frame1: <image> ...), but left-padded
    across scenes. InternVL's own batch_chat() only allows a single <image> per sample.
    On CUDA OOM the batch is split in half and retried.
    """
    try:
        return _generate(model, tok, device, dtype, batch, prompt, gen_cfg)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(batch) == 1:
            raise
        mid = len(batch) // 2
        print(f"[warn] OOM on batch of {len(batch)}, splitting", flush=True)
        return (run_batch(model, tok, device, dtype, batch[:mid], prompt, gen_cfg)
                + run_batch(model, tok, device, dtype, batch[mid:], prompt, gen_cfg))


def _generate(model, tok, device, dtype, batch, prompt, gen_cfg):
    get_conv_template = sys.modules[type(model).__module__].get_conv_template
    image_tokens = "<img>" + "<IMG_CONTEXT>" * model.num_image_token + "</img>"
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")

    queries = []
    for pv in batch:
        question = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(pv))) + prompt
        template = get_conv_template(model.template)
        template.system_message = model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        queries.append(template.get_prompt().replace("<image>", image_tokens))
    eos = tok.convert_tokens_to_ids(template.sep.strip())

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(queries, return_tensors="pt", padding=True)
    pixel_values = torch.cat(batch).to(device=device, dtype=dtype)
    with torch.inference_mode():
        out = model.generate(pixel_values=pixel_values, input_ids=enc["input_ids"].to(device),
                             attention_mask=enc["attention_mask"].to(device), eos_token_id=eos, **gen_cfg)
    return [r.split(template.sep.strip())[0].strip() for r in tok.batch_decode(out, skip_special_tokens=True)]


def prepare_scene(scene, path, fps, max_frames, transform):
    """CPU side: decode + preprocess one scene. Runs in background threads."""
    cap = cv2.VideoCapture(str(path))
    try:
        frames = read_frames(cap, sample_times(scene["start"], scene["end"], fps, max_frames), scene["end"])
    finally:
        cap.release()
    if not frames:
        raise RuntimeError("could not decode any frames")
    return torch.stack([transform(f) for f in frames])


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
        "geographical_location": {k: _blank_unknown(geo.get(k)) for k in ("area", "city", "country")},
        "activity": _str(raw.get("activity")).lower() or "none",
        "shot_type": {
            "framing": _enum(shot.get("framing"), FRAMINGS, "framing", w),
            "setting": _enum(shot.get("setting"), SETTINGS, "setting", w),
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


def save(records, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-dir", default="VIDEO FILES")
    ap.add_argument("--scenes-csv", default="_all_scenes.csv")
    ap.add_argument("--output", default="scene_metadata_internvl3_5_1b.json")
    ap.add_argument("--model", default="OpenGVLab/InternVL3_5-1B")
    ap.add_argument("--fps", type=float, default=1.0, help="sampling rate before capping (Gemini used 1.0)")
    ap.add_argument("--max-frames", type=int, default=16, help="max frames per scene fed to the model")
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=4, help="scenes per GPU call (auto-halves on OOM)")
    ap.add_argument("--workers", type=int, default=4, help="CPU threads for frame decoding")
    ap.add_argument("--retries", type=int, default=1, help="extra attempts (sampled) if JSON parsing fails")
    ap.add_argument("--video-ids", type=int, nargs="*", help="only these video numbers, e.g. 1 2 15")
    ap.add_argument("--max-scenes-per-video", type=int, help="first N scenes of each video")
    ap.add_argument("--limit", type=int, help="stop after N scenes in total")
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing output file")
    args = ap.parse_args()

    video_dir, out_path = Path(args.video_dir), Path(args.output)
    scenes = load_scenes(args.scenes_csv)

    on_disk = {p.stem: p for p in video_dir.glob("*.mp4")}
    missing = sorted({s["video"] for s in scenes} - set(on_disk))
    if missing:
        print(f"[info] {len(missing)} CSV videos not in {video_dir}, skipped: {missing}")
    scenes = [s for s in scenes if s["video"] in on_disk]
    if args.video_ids:
        scenes = [s for s in scenes if video_number(s["video"]) in set(args.video_ids)]
    if args.max_scenes_per_video:
        scenes = [s for s in scenes if s["scene"] <= args.max_scenes_per_video]
    if args.limit:
        scenes = scenes[:args.limit]

    records = {}
    if out_path.exists() and not args.no_resume:
        for r in json.load(open(out_path, encoding="utf-8")):
            records[(r["video"], r["scene"])] = r
    todo = [s for s in scenes if "metadata" not in records.get((s["video"] + ".mp4", s["scene"]), {})]
    print(f"[info] {len(scenes)} scenes selected, {len(scenes) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] no CUDA GPU found - running on CPU will be very slow")
    model, tok, dtype = load_model(args.model, device)
    transform = build_transform(args.image_size)
    prompt = build_prompt()
    greedy = dict(max_new_tokens=args.max_new_tokens, do_sample=False)
    sampled = dict(max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)

    order = {(s["video"] + ".mp4", s["scene"]): i for i, s in enumerate(load_scenes(args.scenes_csv))}
    # group scenes with similar frame counts so a batch wastes little padding
    todo.sort(key=lambda s: len(sample_times(s["start"], s["end"], args.fps, args.max_frames)))
    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]

    def prepare(batch):
        futs = [pool.submit(prepare_scene, s, on_disk[s["video"]], args.fps, args.max_frames, transform)
                for s in batch]
        out = []
        for f in futs:
            try:
                out.append(f.result())
            except Exception as e:
                out.append(e)
        return out

    n_ok = n_err = done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, ThreadPoolExecutor(max_workers=1) as prefetch:
        next_inputs = prefetch.submit(prepare, batches[0])
        for bi, batch in enumerate(batches):
            inputs = next_inputs.result()
            if bi + 1 < len(batches):  # decode the next batch while the GPU works on this one
                next_inputs = prefetch.submit(prepare, batches[bi + 1])

            recs = [dict(video=s["video"] + ".mp4", scene=s["scene"], start_sec=s["start"], end_sec=s["end"],
                         sample_fps=args.fps) for s in batch]
            pending, raws = [], {}
            for j, x in enumerate(inputs):
                if isinstance(x, Exception):
                    recs[j].update(error=f"{type(x).__name__}: {x}", raw_output="")
                else:
                    recs[j]["sampled_frame_count"] = len(x)
                    pending.append(j)

            for attempt in range(args.retries + 1):
                if not pending:
                    break
                try:
                    outs = run_batch(model, tok, device, dtype, [inputs[j] for j in pending], prompt,
                                     greedy if attempt == 0 else sampled)
                except Exception as e:  # e.g. OOM on a single scene
                    msg = "CUDA out of memory (try a lower --max-frames)" \
                        if isinstance(e, torch.cuda.OutOfMemoryError) else f"{type(e).__name__}: {e}"
                    for j in pending:
                        recs[j].update(error=msg, raw_output=raws.get(j, ""))
                    pending = []
                    break
                still = []
                for j, raw in zip(pending, outs):
                    raws[j] = raw
                    try:
                        meta, warnings = normalise(extract_json(raw))
                        recs[j]["metadata"] = meta
                        if warnings:
                            recs[j]["warnings"] = warnings
                            recs[j]["raw_output"] = raw  # keep for debugging repairs
                    except (ValueError, json.JSONDecodeError) as e:
                        recs[j]["_last_err"] = str(e)
                        still.append(j)
                pending = still
            for j in pending:
                recs[j].pop("sampled_frame_count", None)
                recs[j].update(error=f"unparseable JSON after {args.retries + 1} attempts: "
                                     f"{recs[j].pop('_last_err', '')}", raw_output=raws.get(j, ""))

            for r in recs:
                r.pop("_last_err", None)
                if "metadata" in r:
                    r.pop("error", None)
                    if "warnings" not in r:
                        r.pop("raw_output", None)
                    n_ok += 1
                else:
                    r.pop("sampled_frame_count", None)
                    n_err += 1
                records[(r["video"], r["scene"])] = r
            save(sorted(records.values(), key=lambda r: order.get((r["video"], r["scene"]), 1e9)), out_path)

            done += len(batch)
            rate = (time.time() - t0) / done
            print(f"[{done}/{len(todo)}] batch of {len(batch)}: {n_ok} ok, {n_err} errors so far "
                  f"({rate:.1f}s/scene, ~{rate * (len(todo) - done) / 60:.0f} min left)", flush=True)

    print(f"[done] {n_ok} ok, {n_err} errors -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
