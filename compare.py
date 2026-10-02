"""
Collect every evaluated experiment into one results table (baseline first).

Reads eval_scene_metadata_<name>/summary.json and scene_metadata_<name>.run.json for each row of
experiments.tsv (plus the majority baseline), and writes results_table.csv and results_table.md.
Experiments that have not been run yet are skipped.

    python compare.py
"""

import csv
import json
from pathlib import Path

ROWS = [("baseline_majority", "majority-class baseline (no video)", "-")]
with open("experiments.tsv", encoding="utf-8") as f:
    ROWS += [(r["name"], r["model"], r["quant"]) for r in csv.DictReader(f, delimiter="\t")]

COLUMNS = [  # (header, getter on (summary, run))
    ("ROUGE-L", lambda s, r: s["description"]["rougeL_f"]),
    ("METEOR", lambda s, r: s["description"]["meteor"]),
    ("CIDEr-D", lambda s, r: s["description"]["cider_d"]),
    ("SODA_c", lambda s, r: s["description"]["soda_c_f1"]),
    ("content_type", lambda s, r: s["categorical_accuracy"]["content_type"]),
    ("framing", lambda s, r: s["categorical_accuracy"]["framing"]),
    ("setting", lambda s, r: s["categorical_accuracy"]["setting"]),
    ("activity", lambda s, r: s["categorical_accuracy"]["activity"]),
    ("city", lambda s, r: s["categorical_accuracy"]["geo_city"]),
    ("area", lambda s, r: s["categorical_accuracy"]["geo_area"]),
    ("people", lambda s, r: s["categorical_accuracy"]["people_count"]),
    ("tags_F1", lambda s, r: s["visual_tags"]["f1"]),
    ("ocr_ROUGE-1", lambda s, r: s["on_screen_text"]["rouge1_f"]),
    ("ocr_ROUGE-L", lambda s, r: s["on_screen_text"]["rougeL_f"]),
    ("scored", lambda s, r: s["coverage"]["scored_scenes"]),
    ("parse_errors", lambda s, r: s["coverage"]["prediction_errors"]),
    ("repaired", lambda s, r: s["coverage"]["prediction_warnings"]),
    ("s/scene", lambda s, r: r["sessions"][-1]["seconds_per_scene"] if r else None),
    ("peak_GB", lambda s, r: max((x["peak_gpu_memory_gb"] or 0) for x in r["sessions"]) if r else None),
    ("weights_GB", lambda s, r: r.get("weights_memory_gb") if r else None),
]


def fmt(v):
    return "" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


table = []
for name, model, quant in ROWS:
    summary = Path(f"eval_scene_metadata_{name}/summary.json")
    if not summary.exists():
        continue
    s = json.load(open(summary, encoding="utf-8"))
    run_file = Path(f"scene_metadata_{name}.run.json")
    r = json.load(open(run_file, encoding="utf-8")) if run_file.exists() else None
    table.append([name, model, quant] + [fmt(get(s, r)) for _, get in COLUMNS])

header = ["name", "model", "quant"] + [h for h, _ in COLUMNS]
with open("results_table.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([header] + table)
with open("results_table.md", "w", encoding="utf-8") as f:
    f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
    f.writelines("| " + " | ".join(row) + " |\n" for row in table)
print(f"{len(table)} experiments -> results_table.csv, results_table.md")
