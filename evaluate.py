"""
Evaluate generated scene metadata against the Gemini reference.

Scenes are matched on (video, scene). Only scenes where BOTH files have "metadata"
are scored; failed scenes are reported as coverage, not scored as zero.

  Description (free text)  : ROUGE-1/2/L (F1), CIDEr-D, SODA_c (METEOR-based)
  Categorical fields       : exact-match accuracy (case/whitespace-insensitive)
      content_type, shot_type.framing, shot_type.setting, activity,
      geographical_location.{area, city, country}, people_count_numeric
  visual_tags (set)        : per-scene precision / recall / F1 / Jaccard (averaged) + micro-F1
  on_screen_text (set)     : per-scene F1 on normalised strings (averaged)

Note on SODA_c: it aligns predicted and reference event captions per video using
temporal IoU and METEOR (Fujita et al., ECCV 2020). Here both sides share the same
PySceneDetect boundaries, so the alignment is the diagonal and SODA_c becomes a
per-video METEOR F-score in which failed scenes lower recall.
METEOR is NLTK's implementation (no Java needed). CIDEr-D is pycocoevalcap's, with
a single reference per scene and IDF computed over the evaluated scenes, so CIDEr
from a small sanity subset is not comparable to a full-run value.

Usage:
    python evaluate.py --prediction sanity_internvl.json
    python evaluate.py --reference scene_metadata_gemini3.8_flash.json \
                       --prediction scene_metadata_internvl3_5_1b.json --out-dir eval_internvl3_5_1b
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import nltk
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from rouge_score import rouge_scorer

CATEGORICAL = {
    "content_type": lambda m: m.get("content_type"),
    "framing": lambda m: (m.get("shot_type") or {}).get("framing"),
    "setting": lambda m: (m.get("shot_type") or {}).get("setting"),
    "activity": lambda m: m.get("activity"),
    "geo_area": lambda m: (m.get("geographical_location") or {}).get("area"),
    "geo_city": lambda m: (m.get("geographical_location") or {}).get("city"),
    "geo_country": lambda m: (m.get("geographical_location") or {}).get("country"),
    "people_count": lambda m: m.get("people_count_numeric"),
}
SODA_TIOUS = (0.3, 0.5, 0.7, 0.9)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def norm(x):
    return re.sub(r"\s+", " ", str(x if x is not None else "")).strip().lower()


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", str(text).lower())


def load(path):
    """Returns ({key: record with metadata}, set of every key attempted, incl. failures)."""
    out, attempted = {}, set()
    for r in json.load(open(path, encoding="utf-8")):
        key = (re.sub(r"\.mp4$", "", r["video"]), int(r["scene"]))
        attempted.add(key)
        if "metadata" in r:
            out[key] = r
    return out, attempted


def set_prf(pred, ref):
    p, r = set(pred), set(ref)
    if not p and not r:
        return 1.0, 1.0, 1.0, 1.0
    tp = len(p & r)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(r) if r else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp / len(p | r)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def ensure_wordnet():
    """METEOR needs WordNet; download it once if it is not installed."""
    from nltk.corpus import wordnet
    try:
        wordnet.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet=True)


# --------------------------------------------------------------------------- #
# SODA_c
# --------------------------------------------------------------------------- #
def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def chased_dp(score):
    """Order-preserving max-sum alignment (as in the SODA reference implementation)."""
    n, m = len(score), len(score[0]) if score else 0
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + score[i - 1][j - 1])
    return dp[n][m]


def soda_c(pred_events, ref_events, meteor_cache):
    """pred_events / ref_events: lists of (start, end, key) in time order. Returns (P, R, F)."""
    if not pred_events or not ref_events:
        return 0.0, 0.0, 0.0
    fs = []
    for thr in SODA_TIOUS:
        S = []
        for ps, pe, pk in pred_events:
            row = []
            for rs, re_, rk in ref_events:
                iou = tiou((ps, pe), (rs, re_))
                row.append(meteor_cache(pk, rk) if iou >= thr else 0.0)
            S.append(row)
        total = chased_dp(S)
        p, r = total / len(pred_events), total / len(ref_events)
        fs.append((p, r, 2 * p * r / (p + r) if p + r else 0.0))
    return tuple(statistics.mean(x[k] for x in fs) for k in range(3))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default="scene_metadata_gemini3.8_flash.json")
    ap.add_argument("--prediction", required=True)
    ap.add_argument("--out-dir", default=None, help="default: eval_<prediction file stem>")
    args = ap.parse_args()

    ref, ref_attempted = load(args.reference)
    pred, pred_attempted = load(args.prediction)
    ref_err, pred_err = len(ref_attempted) - len(ref), len(pred_attempted) - len(pred)
    keys = sorted(set(ref) & set(pred))
    if not keys:
        raise SystemExit("No scenes with metadata in both files - nothing to evaluate.")

    ensure_wordnet()
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    meteor_memo = {}

    def meteor(pk, rk):
        if (pk, rk) not in meteor_memo:
            meteor_memo[(pk, rk)] = meteor_score(
                [tokenize(ref[rk]["metadata"]["description"])],
                tokenize(pred[pk]["metadata"]["description"]))
        return meteor_memo[(pk, rk)]

    rows = []
    for k in keys:
        rm, pm = ref[k]["metadata"], pred[k]["metadata"]
        row = {"video": k[0], "scene": k[1]}
        rs = rouge.score(rm.get("description", ""), pm.get("description", ""))
        row.update(rouge1=rs["rouge1"].fmeasure, rouge2=rs["rouge2"].fmeasure, rougeL=rs["rougeL"].fmeasure)
        row["meteor"] = meteor(k, k)
        for name, get in CATEGORICAL.items():
            row[f"{name}_ref"], row[f"{name}_pred"] = get(rm), get(pm)
            row[f"{name}_match"] = int(norm(get(rm)) == norm(get(pm)))
        tp, tr, tf, tj = set_prf([norm(t) for t in pm.get("visual_tags", [])],
                                 [norm(t) for t in rm.get("visual_tags", [])])
        row.update(tags_precision=tp, tags_recall=tr, tags_f1=tf, tags_jaccard=tj)
        row["ost_f1"] = set_prf([norm(t) for t in pm.get("on_screen_text", [])],
                                [norm(t) for t in rm.get("on_screen_text", [])])[2]
        rows.append(row)

    # CIDEr-D (corpus-level)
    gts = {i: [" ".join(tokenize(ref[k]["metadata"]["description"]))] for i, k in enumerate(keys)}
    res = {i: [" ".join(tokenize(pred[k]["metadata"]["description"]))] for i, k in enumerate(keys)}
    cider, cider_per = Cider().compute_score(gts, res)
    for row, c in zip(rows, cider_per):
        row["cider"] = float(c)

    # SODA_c per video, over the reference scenes the prediction attempted
    # (so a sanity subset is not penalised for scenes it never ran; failed scenes lower recall)
    by_vid_ref, by_vid_pred = defaultdict(list), defaultdict(list)
    for k, r in ref.items():
        if k in pred_attempted:
            by_vid_ref[k[0]].append((r["start_sec"], r["end_sec"], k))
    for k, r in pred.items():
        by_vid_pred[k[0]].append((r["start_sec"], r["end_sec"], k))
    soda = {}
    for v in sorted(by_vid_ref):
        soda[v] = soda_c(sorted(by_vid_pred[v]), sorted(by_vid_ref[v]), meteor)

    # people count MAE where both give an actual count
    pc = [(r["people_count_ref"], r["people_count_pred"]) for r in rows
          if isinstance(r["people_count_ref"], int) and isinstance(r["people_count_pred"], int)
          and r["people_count_ref"] >= 0 and r["people_count_pred"] >= 0]

    all_tags = sorted({t for k in keys for t in ref[k]["metadata"].get("visual_tags", [])}
                      | {t for k in keys for t in pred[k]["metadata"].get("visual_tags", [])})
    per_tag, micro = {}, [0, 0, 0]
    for t in all_tags:
        tp = sum(t in pred[k]["metadata"]["visual_tags"] and t in ref[k]["metadata"]["visual_tags"] for k in keys)
        fp = sum(t in pred[k]["metadata"]["visual_tags"] and t not in ref[k]["metadata"]["visual_tags"] for k in keys)
        fn = sum(t not in pred[k]["metadata"]["visual_tags"] and t in ref[k]["metadata"]["visual_tags"] for k in keys)
        micro = [micro[0] + tp, micro[1] + fp, micro[2] + fn]
        per_tag[t] = {"support": tp + fn, "f1": round(2 * tp / (2 * tp + fp + fn), 4) if tp + fp + fn else None}
    micro_f1 = 2 * micro[0] / (2 * micro[0] + micro[1] + micro[2]) if sum(micro) else None

    summary = {
        "reference": args.reference,
        "prediction": args.prediction,
        "coverage": {
            "reference_scenes_with_metadata": len(ref),
            "reference_errors": ref_err,
            "prediction_scenes_with_metadata": len(pred),
            "prediction_errors": pred_err,
            "attempted_scenes_with_reference": len(set(ref) & pred_attempted),
            "scored_scenes": len(keys),
            "scored_videos": len({k[0] for k in keys}),
            "prediction_warnings": sum(1 for k in keys if pred[k].get("warnings")),
        },
        "description": {
            "rouge1_f": mean(r["rouge1"] for r in rows),
            "rouge2_f": mean(r["rouge2"] for r in rows),
            "rougeL_f": mean(r["rougeL"] for r in rows),
            "meteor": mean(r["meteor"] for r in rows),
            "cider_d": round(float(cider), 4),
            "soda_c_precision": mean(v[0] for v in soda.values()),
            "soda_c_recall": mean(v[1] for v in soda.values()),
            "soda_c_f1": mean(v[2] for v in soda.values()),
        },
        "categorical_accuracy": {n: mean(r[f"{n}_match"] for r in rows) for n in CATEGORICAL},
        "people_count_mae_when_both_countable": round(statistics.mean(abs(a - b) for a, b in pc), 3) if pc else None,
        "visual_tags": {
            "precision": mean(r["tags_precision"] for r in rows),
            "recall": mean(r["tags_recall"] for r in rows),
            "f1": mean(r["tags_f1"] for r in rows),
            "jaccard": mean(r["tags_jaccard"] for r in rows),
            "micro_f1": round(micro_f1, 4) if micro_f1 is not None else None,
            "per_tag": per_tag,
        },
        "on_screen_text_f1": mean(r["ost_f1"] for r in rows),
        "soda_c_per_video": {v: {"p": round(s[0], 4), "r": round(s[1], 4), "f1": round(s[2], 4)}
                             for v, s in soda.items()},
    }

    out_dir = Path(args.out_dir or f"eval_{Path(args.prediction).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "per_scene.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    c = summary["coverage"]
    print(f"Scored {c['scored_scenes']} scenes / {c['scored_videos']} videos "
          f"(prediction errors: {c['prediction_errors']}, scenes with repair warnings: {c['prediction_warnings']})")
    print("\nDescription")
    for k, v in summary["description"].items():
        print(f"  {k:<18} {v}")
    print("\nCategorical accuracy")
    for k, v in summary["categorical_accuracy"].items():
        print(f"  {k:<18} {v}")
    print(f"  {'people_count_mae':<18} {summary['people_count_mae_when_both_countable']}")
    print("\nVisual tags")
    for k in ("precision", "recall", "f1", "jaccard", "micro_f1"):
        print(f"  {k:<18} {summary['visual_tags'][k]}")
    print(f"\nOn-screen text F1    {summary['on_screen_text_f1']}")
    print(f"\nWrote {out_dir / 'summary.json'} and {out_dir / 'per_scene.csv'}")


if __name__ == "__main__":
    main()
