"""
Build a "no-video" baseline in the same JSON format as generate.py, so it can be scored
with evaluate.py exactly like a model run.

Every scene gets the same metadata:
  - categorical fields : the most common value in the reference file (majority class)
  - visual_tags        : every tag present in more than half of the reference scenes
  - on_screen_text     : [] (nothing read)
  - description        : one fixed generic sentence (DESCRIPTION below), chosen a priori,
                         not tuned against the references

Because the majority values come from the same reference file the models are scored
against, this is a slightly optimistic floor. A model that cannot beat it on a field
is not extracting that field from the video.

    python make_baseline.py
    python evaluate.py --prediction scene_metadata_baseline_majority.json
"""

import argparse
import csv
import json
from collections import Counter

DESCRIPTION = "A person speaks to the camera during an interview in an indoor setting."


def mode(values):
    return Counter(values).most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default="scene_metadata_gemini3.8_flash.json")
    ap.add_argument("--scenes-csv", default="_all_scenes.csv")
    ap.add_argument("--output", default="scene_metadata_baseline_majority.json")
    args = ap.parse_args()

    ref = [r["metadata"] for r in json.load(open(args.reference, encoding="utf-8")) if "metadata" in r]
    n = len(ref)
    tag_counts = Counter(t for m in ref for t in set(m.get("visual_tags", [])))

    meta = {
        "on_screen_text": [],
        "visual_tags": [t for t, c in tag_counts.most_common() if c / n > 0.5],
        "people_count_numeric": mode(m["people_count_numeric"] for m in ref),
        "description": DESCRIPTION,
        "geographical_location": {k: mode(m["geographical_location"][k] for m in ref)
                                  for k in ("area", "city", "country")},
        "activity": mode(m["activity"] for m in ref),
        "shot_type": {k: mode(m["shot_type"][k] for m in ref) for k in ("framing", "setting")},
        "content_type": mode(m["content_type"] for m in ref),
        "uncertainty_notes": [],
    }

    with open(args.scenes_csv, encoding="utf-8-sig", newline="") as f:
        records = [dict(video=r["video"] + ".mp4", scene=int(r["scene"]),
                        start_sec=float(r["start_sec"]), end_sec=float(r["end_sec"]),
                        sample_fps=0.0, sampled_frame_count=0, metadata=meta)
                   for r in csv.DictReader(f)]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} scenes to {args.output} using:")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
