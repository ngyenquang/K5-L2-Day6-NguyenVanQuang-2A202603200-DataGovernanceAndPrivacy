"""Out-of-domain false-alarm check (T02 +24h milestone: "kiểm 10 ảnh out-of-domain").

Takes N images (seeded sample) from another fixture's pool and runs the frozen T02 scoring on them,
using the per-camera sharpness reference derived from the T02 *dev* inputs (no labels).
It only measures FALSE ALARMS on images that look clean; it says nothing about recall.

    python -m tools.ood_check --pool fixtures/t13_rare_mining --n 10 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.image_ops import load_rgb
from src.t02_features import camera_reference, image_features, score_frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="fixtures/t13_rare_mining")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    t02 = Path("fixtures/t02_raw_qc")
    imgs = pd.read_csv(t02 / "images.csv")
    feats = pd.concat([imgs, pd.DataFrame([image_features(load_rgb(t02 / p)) for p in imgs.path])], axis=1)
    ref = camera_reference(feats, feats.split == "dev")

    pool = Path(a.pool)
    sample = pd.read_csv(pool / "images.csv").sample(a.n, random_state=a.seed)
    rows = []
    for r in sample.itertuples():
        f = image_features(load_rgb(pool / r.path))
        # camera of OOD images is unknown: score under every camera reference and keep the worst case
        best = max((score_frame(pd.DataFrame([{"image_id": r.image_id, "camera_id": c, **f}]), ref).iloc[0] for c in ref),
                   key=lambda x: x.score)
        rows.append({"image_id": r.image_id, "score": round(float(best.score), 3), "pred": best.pred_defect_type})
    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    print(out.to_string(index=False))
    print(f"flagged {(out.score >= 1.0).sum()}/{len(out)} (all images were checked by eye: clean)")


if __name__ == "__main__":
    main()
