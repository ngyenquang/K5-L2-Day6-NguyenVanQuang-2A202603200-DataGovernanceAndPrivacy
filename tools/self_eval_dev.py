"""Self-evaluation on the dev split using OUR OWN labels (reports/self_labels_dev.csv).

This is NOT the official metric (that needs the instructor's gt/). Dev numbers are optimistic for the
PoC because its thresholds were chosen on these same images; the baseline has no tuned parameters.

    python -m tools.self_eval_dev --baseline out/baseline/ranked_review.csv --poc out/poc/ranked_review.csv \
        --out reports/self_eval_dev.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _metrics(pred: pd.DataFrame, lab: pd.DataFrame, images: pd.DataFrame, b: int) -> dict:
    dev = lab.merge(images[["image_id", "light", "camera_id"]], on="image_id")
    dev["is_defect"] = (dev.defect_type != "none").astype(int)
    p = pred[pred.image_id.isin(dev.image_id)].sort_values("score", ascending=False, kind="mergesort")
    top = list(p.image_id.head(b))
    pos = set(dev.loc[dev.is_defect == 1, "image_id"])
    hit = len(pos & set(top))
    out = {"n": len(dev), "n_defect": len(pos), "recall": hit / len(pos), "precision": hit / b, "hit": hit}
    neg = dev[dev.is_defect == 0]
    for col in ("light", "camera_id"):
        for k, sub in neg.groupby(col):
            out[f"fa_{col}={k}"] = (f"{sub.image_id.isin(top).sum()}/{len(sub)}")
    if "pred_defect_type" in pred.columns:
        m = dev[dev.is_defect == 1].merge(p[["image_id", "pred_defect_type"]], on="image_id", how="left")
        out["type_acc"] = float((m.defect_type == m.pred_defect_type).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--poc", required=True)
    ap.add_argument("--labels", default="reports/self_labels_dev.csv")
    ap.add_argument("--images", default="fixtures/t02_raw_qc/images.csv")
    ap.add_argument("--out", default="reports/self_eval_dev.md")
    a = ap.parse_args()
    lab = pd.read_csv(a.labels)
    images = pd.read_csv(a.images)
    runs = {"baseline (Laplacian)": pd.read_csv(a.baseline), "PoC": pd.read_csv(a.poc)}
    n_def = int((lab.defect_type != "none").sum())
    lines = ["# Self-evaluation on dev (own labels, NOT the official metric)", "",
             f"dev N={len(lab)}, defects={n_def}. Seed 42. Pool = dev images only.", "",
             "| Run | B | Recall@B | Precision@B | hits | type acc |", "|---|---|---|---|---|---|"]
    detail = []
    for b in (20, n_def):
        for name, pred in runs.items():
            m = _metrics(pred, lab, images, b)
            ta = f"{m['type_acc']:.2f}" if "type_acc" in m else "n/a"
            lines.append(f"| {name} | {b} | {m['recall']:.3f} | {m['precision']:.3f} | {m['hit']}/{m['n_defect']} | {ta} |")
            if b == 20:
                detail.append(f"- {name}: false alarms in top-20 (clean images in top-20 / clean images) -> "
                              + ", ".join(f"{k[3:]}: {v}" for k, v in m.items() if k.startswith("fa_")))
    lines += ["", "Note: B=20 caps recall below 1.0 on dev because dev has more than 20 defects; B=#defects is the ceiling case.",
              "", "## False alarms by slice (B=20)", *detail, "",
              "Caveat: PoC thresholds were chosen on these same dev images (optimistic); baseline has no tuned parameters."]
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
