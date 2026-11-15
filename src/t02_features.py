"""T02 Raw QC features + rule-based scoring.

Each defect type gets a sub-score where >= 1.0 means "defect". The image score is the max sub-score.
Thresholds were chosen on the `dev` split only (self-labelled, see reports/self_labels_dev.csv)
and frozen before any eval-split result was seen. Override via configs/lab.yaml `params:` if needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common.image_ops import laplacian_variance, to_gray

# ---- frozen thresholds (dev only) -------------------------------------------------
DEFAULTS = {
    "blur_log_lap_floor": 2.5,    # log(1+lapvar) at/below which blur is certain (dev blur max = 1.94)
    "dark_p95": 75.0,             # 95th-percentile gray below this => dark (dev dark max 62, night-normal min 118)
    "glare_cell_sat": 0.06,       # fraction of saturated px in the worst 4x4 grid cell (dev none max .02, glare min .10)
    "compression_block": 1.8,     # 8px block-boundary ratio (dev compression min 1.88, others max 1.70)
}
DEFECT_TYPES = ["blur", "dark", "glare", "compression"]


def _block_ratio(g: np.ndarray, p: int = 8) -> float:
    """Mean |diff| on p-px block boundaries / inside blocks, both axes (JPEG blockiness proxy)."""
    out = []
    for ax in (0, 1):
        d = np.abs(np.diff(g, axis=ax))
        idx = np.arange(d.shape[ax])
        edge = (idx % p) == (p - 1)
        out.append(d.take(np.where(edge)[0], axis=ax).mean() / (d.take(np.where(~edge)[0], axis=ax).mean() + 1e-3))
    return float(np.mean(out))


def image_features(img: np.ndarray) -> dict:
    g = to_gray(img)
    h, w = g.shape
    cells = [(g[y:y + h // 4, x:x + w // 4] > 240).mean() for y in range(0, h - h // 4 + 1, h // 4) for x in range(0, w - w // 4 + 1, w // 4)]
    return {
        "log_lap": float(np.log(laplacian_variance(img) + 1.0)),
        "p95": float(np.percentile(g, 95)),
        "mean": float(g.mean()),
        "cell_sat": float(max(cells)),
        "block": _block_ratio(g),
    }


def camera_reference(feats: pd.DataFrame, dev_mask: pd.Series) -> dict:
    """Per-camera reference sharpness = median log-Laplacian of the dev inputs (no labels used)."""
    ref = feats[dev_mask].groupby("camera_id").log_lap.median().to_dict()
    fallback = float(feats[dev_mask].log_lap.median())
    return {c: ref.get(c, fallback) for c in feats.camera_id.unique()}


def score_frame(feats: pd.DataFrame, ref: dict, params: dict | None = None) -> pd.DataFrame:
    """Add sub-scores, `score`, `pred_defect_type`, `pred_action`, `reason`."""
    p = {**DEFAULTS, **(params or {})}
    f = feats.copy()
    cam_ref = f.camera_id.map(ref)
    # blur: distance below the camera's own typical sharpness (cam_B optics are softer than cam_A)
    f["s_blur"] = ((cam_ref - f.log_lap) / (cam_ref - p["blur_log_lap_floor"])).clip(lower=0)
    f["s_dark"] = (p["dark_p95"] / f.p95.clip(lower=1.0))
    f["s_glare"] = f.cell_sat / p["glare_cell_sat"]
    f["s_compression"] = f.block / p["compression_block"]
    S = f[[f"s_{t}" for t in DEFECT_TYPES]]
    f["score"] = S.max(axis=1)
    top = S.idxmax(axis=1).str.replace("s_", "", regex=False)
    flagged = f.score >= 1.0
    f["pred_defect_type"] = np.where(flagged, top, "none")
    f["pred_action"] = np.where(flagged, "RECAPTURE", "KEEP")
    f["reason"] = np.where(flagged, top, "ok")
    return f
