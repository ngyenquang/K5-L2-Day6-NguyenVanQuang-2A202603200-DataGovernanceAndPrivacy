"""T02 baseline: one global Laplacian-variance score, no per-light / per-camera normalisation.

    python -m src.main --track t02 --pipeline src.baseline_laplacian --output out/baseline --run-name baseline --offline --seed 42
"""
from __future__ import annotations

import pandas as pd

from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.image_ops import laplacian_variance, load_rgb
from .common.review import make_review_queue, write_review_queue
from .metrics.tracks import get_track


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    items = pd.read_csv(ctx.input_dir / "images.csv")
    items["score"] = [1.0 / (laplacian_variance(load_rgb(ctx.input_dir / p)) + 1e-6) for p in items["path"]]
    items["reason"] = "low_sharpness"
    ctx.stats["n_items"] = len(items)

    budget = int(ctx.param("budget", 20))
    q = make_review_queue(items, "image_id", "score", budget, reason_col="reason", slice_col="light", evidence_col="path")
    write_review_queue(q, ctx.out("review_queue.csv"))
    render_contact_sheet(q, ctx.out("contact_sheet.html"), image_root=ctx.input_dir, title="t02 Laplacian baseline queue")

    items.sort_values("score", ascending=False)[["image_id", "score"]].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(
        Decision(status="HOLD", reason_code="BASELINE_ONLY", reason="Global Laplacian baseline: comparison only, not for release",
                 owner=ctx.config.get("owner", "baseline"), scope=ctx.track, evidence=[str(ctx.out(spec.pred_file))]),
        ctx.out("decision.json"), ctx.out("decision_log.csv"))
