"""
Generate per-scene video metadata with a GGUF model served by llama.cpp (llama-server).

Same prompt, frame sampling, JSON parsing/normalisation and output format as generate.py, so
evaluate.py and compare.py work unchanged. Only the runtime differs: the model is a community GGUF
checkpoint (e.g. unsloth/Qwen3.5-9B-GGUF) with its vision encoder in a separate mmproj file, and the
scene's frames are sent to llama-server as a sequence of images.

    python generate_gguf.py --repo unsloth/Qwen3.5-9B-GGUF --file "Qwen3.5-9B-Q4_K_M.gguf" \
        --quant q4_k_m --server-bin ~/llama.cpp/build/bin/llama-server \
        --output scene_metadata_qwen3_5_9b_gguf_q4_k_m.json

--file is a glob inside the repo; for split files (e.g. "BF16/*.gguf") the first shard is loaded
and llama.cpp picks up the rest. Models are taken from the local Hugging Face cache
(download them first with `hf download`). The output is saved after every scene and re-running the
same command resumes where it stopped.
"""

import argparse
import base64
import io
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# shared with the transformers pipeline so both runtimes see identical prompts, frames and parsing
from generate import (build_prompt, extract_json, load_scenes, normalise, prepare_scene, save_json,
                      strip_thinking, video_number)


# --------------------------------------------------------------------------- #
# Model files
# --------------------------------------------------------------------------- #
def resolve_files(repo, pattern, mmproj):
    """Return (model_path, [all model shards], mmproj_path) from the local HF cache."""
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(repo, allow_patterns=[pattern, mmproj]))
    shards = sorted(root.glob(pattern))
    if not shards:
        sys.exit(f"no file matching '{pattern}' in {repo} (cache: {root}); download it first with "
                 f"hf download {repo} --include \"{pattern}\" \"{mmproj}\"")
    mm = root / mmproj
    if not mm.exists():
        sys.exit(f"{mmproj} not found in {repo} (cache: {root})")
    return shards[0], shards, mm


# --------------------------------------------------------------------------- #
# llama-server
# --------------------------------------------------------------------------- #
class Server:
    def __init__(self, args, model_path, mmproj_path, log_path):
        self.url = f"http://127.0.0.1:{args.port}"
        cmd = [args.server_bin, "-m", str(model_path), "--mmproj", str(mmproj_path),
               "--host", "127.0.0.1", "--port", str(args.port),
               "-ngl", "999", "--parallel", str(args.parallel), "-c", str(args.ctx_per_slot * args.parallel),
               "--jinja", "--seed", "0"]
        cmd += args.server_args
        print("[info] starting:", " ".join(cmd), flush=True)
        self.log = open(log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)

    def wait_ready(self, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}; see {self.log.name}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return time.time() - t0
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(2)
        raise TimeoutError(f"llama-server not ready after {timeout}s; see {self.log.name}")

    def chat(self, content, params, timeout):
        body = {"messages": [{"role": "user", "content": content}],
                "chat_template_kwargs": {"enable_thinking": False}, **params}
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
        return strip_thinking(out["choices"][0]["message"].get("content") or "")

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


def server_version(server_bin):
    try:
        out = subprocess.run([server_bin, "--version"], capture_output=True, text=True, timeout=60)
        return (out.stdout + out.stderr).strip().splitlines()[0] if (out.stdout + out.stderr).strip() else None
    except Exception:
        return None


def gpu_info():
    """(gpu name, MiB used on GPU 0) via nvidia-smi; works for any process on the GPU."""
    try:
        # under Slurm only the allocated GPU is visible, so the first line is ours
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
        name, used = [x.strip() for x in out.split(",")]
        return name, float(used)
    except Exception:
        return None, None


class PeakMemory(threading.Thread):
    def __init__(self, every=2.0):
        super().__init__(daemon=True)
        self.every, self.peak, self._stop = every, 0.0, threading.Event()

    def run(self):
        while not self._stop.is_set():
            used = gpu_info()[1]
            if used is not None:
                self.peak = max(self.peak, used)
            self._stop.wait(self.every)

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
def frame_to_data_url(img, frame_pixels):
    """Resize to the same per-frame pixel budget as generate.py (aspect ratio kept), JPEG-encode."""
    w, h = img.size
    scale = min(1.0, (frame_pixels / (w * h)) ** 0.5)
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def scene_content(frames, prompt, frame_pixels):
    return ([{"type": "image_url", "image_url": {"url": frame_to_data_url(f, frame_pixels)}} for f in frames]
            + [{"type": "text", "text": prompt}])


def run_scene(server, s, path, args, prompt, greedy, sampled):
    """Decode frames, query the model (with retries), parse. Returns (record, generate_seconds)."""
    rec = dict(video=s["video"] + ".mp4", scene=s["scene"], start_sec=s["start"], end_sec=s["end"],
               sample_fps=args.fps)
    try:
        frames, _, _ = prepare_scene(s, path, args.fps, args.max_frames)
    except Exception as e:
        rec.update(error=f"{type(e).__name__}: {e}", raw_output="")
        return rec, 0.0
    content = scene_content(frames, prompt, args.frame_pixels)
    raw, err, gen_s = "", "", 0.0
    for attempt in range(args.retries + 1):
        t = time.time()
        try:
            raw = server.chat(content, greedy if attempt == 0 else sampled, args.request_timeout)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            continue
        finally:
            gen_s += time.time() - t
        try:
            meta, warnings = normalise(extract_json(raw))
        except (ValueError, json.JSONDecodeError) as e:
            err = f"unparseable JSON after {attempt + 1} attempts: {e}"
            continue
        rec.update(sampled_frame_count=len(frames), metadata=meta)
        if warnings:
            rec.update(warnings=warnings, raw_output=raw)
        return rec, gen_s
    rec.update(error=err or "unknown error", raw_output=raw)
    return rec, gen_s


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="Hugging Face repo, e.g. unsloth/Qwen3.5-9B-GGUF")
    ap.add_argument("--file", required=True, help="GGUF file (glob) inside the repo, e.g. Qwen3.5-9B-Q4_K_M.gguf")
    ap.add_argument("--mmproj", default="mmproj-BF16.gguf", help="vision-encoder file inside the repo")
    ap.add_argument("--quant", required=True, help="label stored in the run file, e.g. bf16, q8_0, q4_k_m")
    ap.add_argument("--output", required=True)
    ap.add_argument("--server-bin", default="llama-server", help="path to llama.cpp's llama-server")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--parallel", type=int, default=4, help="concurrent requests (llama-server slots)")
    ap.add_argument("--ctx-per-slot", type=int, default=32768,
                    help="context tokens per slot (16 frames + prompt + answer must fit)")
    ap.add_argument("--server-args", nargs=argparse.REMAINDER, default=[],
                    help="extra llama-server arguments (must come last)")
    ap.add_argument("--startup-timeout", type=int, default=1800)
    ap.add_argument("--request-timeout", type=int, default=900)
    ap.add_argument("--video-dir", default="VIDEO FILES")
    ap.add_argument("--scenes-csv", default="_all_scenes.csv")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--frame-pixels", type=int, default=448 * 448)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--workers", type=int, default=None, help="ignored (accepted for slurm_job.sh compatibility)")
    ap.add_argument("--video-ids", type=int, nargs="*")
    ap.add_argument("--max-scenes-per-video", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    model_label = f"{args.repo}:{args.file}"
    out_path = Path(args.output)
    run_path = out_path.with_suffix(".run.json")
    all_scenes = load_scenes(args.scenes_csv)
    order = {(s["video"] + ".mp4", s["scene"]): i for i, s in enumerate(all_scenes)}

    video_dir = Path(args.video_dir)
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
        if prev.get("model") != model_label or prev.get("quant") != args.quant:
            sys.exit(f"{out_path} exists but was not produced by {model_label} ({args.quant}); "
                     f"run file says {prev.get('model')} ({prev.get('quant')}). Use another --output or --no-resume.")
        for r in json.load(open(out_path, encoding="utf-8")):
            records[(r["video"], r["scene"])] = r
    todo = [s for s in scenes if "metadata" not in records.get((s["video"] + ".mp4", s["scene"]), {})]
    print(f"[info] {len(scenes)} scenes selected, {len(scenes) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        return

    model_path, shards, mmproj_path = resolve_files(args.repo, args.file, args.mmproj)
    weights_gb = sum(p.stat().st_size for p in shards) / 1e9
    mmproj_gb = mmproj_path.stat().st_size / 1e9
    gpu_name, idle_mib = gpu_info()

    server = Server(args, model_path, mmproj_path, out_path.with_suffix(".server.log"))
    try:
        load_s = server.wait_ready(args.startup_timeout)
        loaded_mib = gpu_info()[1]
        print(f"[info] {model_label} ready in {load_s:.0f}s; model file {weights_gb:.2f} GB + mmproj "
              f"{mmproj_gb:.2f} GB; GPU memory after load {loaded_mib / 1024 if loaded_mib else float('nan'):.2f} GiB",
              flush=True)

        run = json.load(open(run_path, encoding="utf-8")) if run_path.exists() and not args.no_resume else {}
        run.update({
            "model": model_label, "family": "qwen3_5", "quant": args.quant, "backend": "llama.cpp",
            "model_files": [p.name for p in shards], "mmproj": mmproj_path.name,
            "weights_memory_gb": round(weights_gb + mmproj_gb, 2),
            "language_model_file_gb": round(weights_gb, 2), "mmproj_file_gb": round(mmproj_gb, 2),
            # includes llama.cpp's KV cache and compute buffers, so it is reported separately from the
            # weights (compare.py reports weights_memory_gb = GGUF file + mmproj file)
            "gpu_memory_after_load_gb": round((loaded_mib - (idle_mib or 0)) * 1.048576 / 1e3, 2) if loaded_mib else None,
            "settings": {k: getattr(args, k) for k in ("fps", "max_frames", "frame_pixels", "max_new_tokens",
                                                        "retries", "parallel", "ctx_per_slot")},
            "versions": {"python": sys.version.split()[0], "llama_server": server_version(args.server_bin),
                         "gpu": gpu_name},
        })
        run.setdefault("sessions", [])
        save_json(run, run_path)

        prompt = build_prompt()
        greedy = dict(max_tokens=args.max_new_tokens, temperature=0.0, top_k=1)
        sampled = dict(max_tokens=args.max_new_tokens, temperature=0.7, top_p=0.9, repeat_penalty=1.1)

        peak = PeakMemory()
        peak.start()
        n_ok = n_err = done = 0
        gen_seconds = 0.0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futs = [pool.submit(run_scene, server, s, on_disk[s["video"]], args, prompt, greedy, sampled)
                    for s in todo]
            for f in as_completed(futs):
                rec, g = f.result()
                gen_seconds += g
                if "metadata" in rec:
                    n_ok += 1
                else:
                    n_err += 1
                records[(rec["video"], rec["scene"])] = rec
                done += 1
                save_json(sorted(records.values(), key=lambda r: order.get((r["video"], r["scene"]), 1e9)), out_path)
                if done % 8 == 0 or done == len(todo):
                    rate = (time.time() - t0) / done
                    print(f"[{done}/{len(todo)}] {n_ok} ok, {n_err} errors so far ({rate:.1f}s/scene, "
                          f"~{rate * (len(todo) - done) / 60:.0f} min left)", flush=True)
        peak.stop()

        session = {
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scenes_run": done, "ok": n_ok, "errors": n_err,
            "wall_seconds": round(time.time() - t0, 1), "generate_seconds": round(gen_seconds, 1),
            "seconds_per_scene": round((time.time() - t0) / max(done, 1), 3),
            "load_seconds": round(load_s, 1),
            "peak_gpu_memory_gb": round((peak.peak - (idle_mib or 0)) * 1.048576 / 1e3, 2) if peak.peak else None,
        }
        run["sessions"].append(session)
        save_json(run, run_path)
        print(f"[done] {n_ok} ok, {n_err} errors -> {out_path} (run stats: {run_path})")
    finally:
        server.stop()


if __name__ == "__main__":
    sys.exit(main())
