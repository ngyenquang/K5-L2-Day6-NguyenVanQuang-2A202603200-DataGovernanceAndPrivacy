"""YOUR PoC PIPELINE — this is the file your group owns.

Lab contract = 5 automated steps, run by `python -m src.main ...`:
    01 load      input + provenance check (manifest.csv)
    02 process   your algorithm / model (the research part)
    03 rank      prioritise for a fixed review budget B
    04 review    human-in-the-loop: queue -> reviewer decisions -> overrides + audit log
    05 decide    GO / HOLD / REWORK (or track equivalent) with reason_code + owner, write outputs

Required output: out/<pred_file> with the columns listed for your track in
docs/tracks/TXX.md (also in src/metrics/tracks.py::TRACKS). Then evaluate with
    python -m src.evaluate --track tXX --pred out/<pred_file>

Rules: never read fixtures/*/gt/ here (the GT guard will stop you); seed everything via ctx.rng;
no network calls; fail loudly on missing models instead of downloading.
See src/examples/random_baseline.py for a minimal end-to-end wiring example.

Group G27 / T02 Raw QC Triage: rule-based multi-defect scoring (blur / dark / glare / compression),
see src/t02_features.py. Thresholds frozen on the dev split (reports/self_labels_dev.csv).
"""
from __future__ import annotations

import pandas as pd

from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.image_ops import load_rgb
from .common.manifest import verify_manifest
from .common.review import load_review_decisions, make_review_queue, write_review_queue
from .metrics.tracks import get_track
from .t02_features import DEFAULTS, DEFECT_TYPES, camera_reference, image_features, score_frame

MAX_DEFECT_RATE = 0.10   # frozen policy: batch is RECAPTURE if > 10% of images are flagged after review


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> dict:
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    data = {"manifest": pd.read_csv(manifest) if manifest.exists() else None}
    if ctx.track == "t02":
        data["images"] = pd.read_csv(ctx.input_dir / "images.csv")
    return data


# ---------------------------------------------------------------- 02 process
def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    """Per-image features -> per-defect sub-scores -> score = max (>= 1.0 means defect).

    Reference sharpness per camera comes from the dev split inputs only (no labels).
    """
    images = data["images"].copy()
    feats = pd.DataFrame([image_features(load_rgb(ctx.input_dir / r.path)) for r in images.itertuples(index=False)])
    feats = pd.concat([images.reset_index(drop=True), feats], axis=1)
    dev_mask = feats.split == ctx.param("dev_split", "dev")
    ref = camera_reference(feats, dev_mask)
    overrides = {k: ctx.param(k) for k in DEFAULTS if ctx.param(k) is not None}
    out = score_frame(feats, ref, overrides)
    out["slice"] = out.light + "/" + out.camera_id
    ctx.stats["camera_reference_log_lap"] = {k: round(v, 3) for k, v in ref.items()}
    ctx.stats["n_flagged_by_model"] = int((out.score >= 1.0).sum())
    return out


# ---------------------------------------------------------------- 03 rank
def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    """Top-B by `score`; writes review_queue.csv and an offline contact sheet for the reviewer."""
    budget = int(ctx.param("budget", 20))
    if "score" not in scored.columns:
        raise ValueError("rank(): process() output needs a numeric `score` column (higher = review first)")
    q = make_review_queue(scored, id_col="image_id", score_col="score", budget=budget, reason_col="reason",
                          action_col="pred_action", slice_col="slice", evidence_col="path")
    write_review_queue(q, ctx.out("review_queue.csv"))
    render_contact_sheet(q, ctx.out("contact_sheet.html"), image_root=ctx.input_dir,
                         title=f"{ctx.track} PoC review queue (top-{budget})")
    return q


# ---------------------------------------------------------------- 04 review
def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """Apply out/<run>/review_queue_reviewed.csv if present; every change goes to the audit log.

    approve  -> reviewer confirms the model's flag (no change, logged)
    reject   -> false alarm: pred = none / KEEP, score -> 0 (raw score kept in `score_raw`)
    edit     -> override_value is the correct defect type (none|blur|glare|dark|compression)
    escalate -> marked `needs_escalation` (decision becomes HOLD)
    skip     -> ignored
    """
    final = scored.copy()
    final["score_raw"] = final.score
    final["reviewer_decision"] = ""
    final["needs_escalation"] = False
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    if not reviewed_path.exists():
        ctx.stats["n_reviewed"] = 0
        return final
    reviewed = load_review_decisions(reviewed_path)
    ctx.stats["n_reviewed"] = len(reviewed)
    valid_types = {"none", *DEFECT_TYPES}
    for r in reviewed.itertuples(index=False):
        m = final.image_id == r.item_id
        if not m.any():
            raise ValueError(f"reviewed item {r.item_id!r} is not in the scored images")
        before = {"type": final.loc[m, "pred_defect_type"].iloc[0], "action": final.loc[m, "pred_action"].iloc[0],
                  "score": float(final.loc[m, "score"].iloc[0])}
        final.loc[m, "reviewer_decision"] = r.reviewer_decision
        if r.reviewer_decision == "reject":
            final.loc[m, ["pred_defect_type", "pred_action", "score"]] = ["none", "KEEP", 0.0]
        elif r.reviewer_decision == "edit":
            new_type = str(r.override_value).strip().lower()
            if new_type not in valid_types:
                raise ValueError(f"edit for {r.item_id}: override_value {r.override_value!r} not in {sorted(valid_types)}")
            if new_type == "none":
                final.loc[m, ["pred_defect_type", "pred_action", "score"]] = ["none", "KEEP", 0.0]
            else:
                final.loc[m, ["pred_defect_type", "pred_action"]] = [new_type, "RECAPTURE"]
                final.loc[m, "score"] = max(before["score"], 1.0)
        elif r.reviewer_decision == "escalate":
            final.loc[m, "needs_escalation"] = True
        after = {"type": final.loc[m, "pred_defect_type"].iloc[0], "action": final.loc[m, "pred_action"].iloc[0],
                 "score": float(final.loc[m, "score"].iloc[0])}
        ctx.audit.log(actor=r.reviewer, action=f"review:{r.reviewer_decision}", item_id=r.item_id,
                      before=before, after=after, reason=r.review_note)
    return final


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    """Batch policy (frozen before eval): human gate first, then defect-rate limit.

    HOLD      if any top-B item is still unreviewed, or a reviewer escalated
    RECAPTURE if flagged-after-review rate > MAX_DEFECT_RATE
    KEEP      otherwise
    """
    owner = ctx.config.get("owner", "TODO-owner")
    n = len(final)
    flagged = final[final.pred_action == "RECAPTURE"]
    rate = len(flagged) / max(n, 1)
    budget = int(ctx.param("budget", 20))
    top_ids = set(final.sort_values("score_raw", ascending=False, kind="mergesort").head(budget).image_id)
    unreviewed = sorted(top_ids - set(final.loc[final.reviewer_decision != "", "image_id"]))
    metrics = {"n_images": n, "n_flagged": len(flagged), "defect_rate": round(rate, 4), "max_defect_rate": MAX_DEFECT_RATE,
               "budget": budget, "n_reviewed": ctx.stats.get("n_reviewed", 0), "n_unreviewed_in_top_b": len(unreviewed)}
    evidence = [str(ctx.out("ranked_review.csv")), str(ctx.out("review_queue.csv")), str(ctx.out("audit_log.jsonl"))]
    if unreviewed:
        return Decision("HOLD", "REVIEW_PENDING", f"{len(unreviewed)}/{budget} top-B images not reviewed by a human yet",
                        owner, ctx.track, evidence, metrics)
    if final.needs_escalation.any():
        ids = ",".join(final.loc[final.needs_escalation, "image_id"])
        return Decision("HOLD", "REVIEWER_ESCALATED", f"Reviewer escalated: {ids}", owner, ctx.track, evidence, metrics)
    if rate > MAX_DEFECT_RATE:
        return Decision("RECAPTURE", "DEFECT_RATE_ABOVE_LIMIT",
                        f"{len(flagged)}/{n} images flagged ({rate:.0%}) > {MAX_DEFECT_RATE:.0%}; recapture flagged images",
                        owner, ctx.track, evidence, metrics)
    return Decision("KEEP", "DEFECT_RATE_WITHIN_LIMIT", f"{len(flagged)}/{n} flagged ({rate:.0%}) <= {MAX_DEFECT_RATE:.0%}",
                    owner, ctx.track, evidence, metrics)


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored)
    final = review(ctx, scored, queue)
    final[spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decide(ctx, final), ctx.out("decision.json"), ctx.out("decision_log.csv"))
