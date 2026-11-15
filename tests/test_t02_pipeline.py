"""Group G27 tests for the T02 Raw QC pipeline (success, edge and failure paths). No gt/ needed."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

from src import main as runner
from src.t02_features import camera_reference, image_features, score_frame


def _run(out, *extra):
    return runner.main(["--config", "configs/lab.yaml", "--track", "t02", "--output", str(out), "--offline",
                        "--seed", "42", "--run-name", "poc", *extra])


def _scene(seed=0):
    """Synthetic sharp 'street' image: textured, mid brightness."""
    r = np.random.default_rng(seed)
    img = np.clip(r.normal(120, 40, (180, 320, 3)), 0, 255)
    img = gaussian_filter(img, (0.8, 0.8, 0))
    return img.astype(np.uint8)


def _frame(img, camera="cam_A", light="day"):
    f = pd.DataFrame([{"image_id": "x", "camera_id": camera, "light": light, **image_features(img)}])
    return score_frame(f, {camera: 6.5}).iloc[0]


# ---------------------------------------------------------------- success path
def test_poc_end_to_end_outputs_and_human_gate(tmp_path):
    assert _run(tmp_path) == 0
    pred = pd.read_csv(tmp_path / "ranked_review.csv")
    assert len(pred) == 96 and pred.image_id.is_unique
    assert {"image_id", "score", "pred_defect_type", "pred_action"} <= set(pred.columns)
    assert np.isfinite(pred.score).all()
    assert len(pd.read_csv(tmp_path / "review_queue.csv")) == 20
    assert (tmp_path / "contact_sheet.html").exists()
    d = json.loads((tmp_path / "decision.json").read_text())
    # no human review yet -> must not auto-release
    assert d["status"] == "HOLD" and d["reason_code"] == "REVIEW_PENDING" and d["owner"].strip()


def test_poc_is_reproducible(tmp_path):
    _run(tmp_path / "a")
    _run(tmp_path / "b")
    assert (tmp_path / "a" / "ranked_review.csv").read_text() == (tmp_path / "b" / "ranked_review.csv").read_text()


# ---------------------------------------------------------------- edge cases (the 3 failure modes)
def test_clean_sharp_image_is_not_flagged():
    r = _frame(_scene())
    assert r.score < 1.0 and r.pred_action == "KEEP"


def test_blur_is_flagged():
    r = _frame(gaussian_filter(_scene(), (6, 6, 0)).astype(np.uint8))
    assert r.pred_defect_type == "blur" and r.score >= 1.0


def test_local_glare_on_sharp_image_is_flagged():
    img = _scene()
    img[40:80, 100:180] = 255          # blown-out patch near "plate"; rest stays sharp
    r = _frame(img)
    assert r.pred_defect_type == "glare"


def test_normal_night_scene_is_not_called_dark():
    """Night images are dark overall but have bright windows/lights: they must stay KEEP."""
    img = np.full((180, 320, 3), 20, np.uint8)
    img[::3, ::3] = 200                # ~11% bright pixels (windows / lamps) -> p95 stays high
    assert _frame(img, light="night").pred_defect_type != "dark"


def test_truly_dark_image_is_flagged_dark():
    img = (_scene() * 0.1).astype(np.uint8)
    assert _frame(img).pred_defect_type == "dark"


def test_softer_camera_gets_its_own_sharpness_reference():
    f = pd.DataFrame({"camera_id": ["cam_A"] * 3 + ["cam_B"] * 3, "log_lap": [7, 7, 7, 5.5, 5.5, 5.5]})
    ref = camera_reference(f, pd.Series([True] * 6))
    assert ref["cam_B"] < ref["cam_A"]


# ---------------------------------------------------------------- failure / review paths
def _reviewed(tmp_path, rows):
    q = pd.read_csv(tmp_path / "review_queue.csv", dtype=str).fillna("")
    for i, (dec, reviewer, ov) in rows.items():
        q.loc[i, ["reviewer_decision", "reviewer", "override_value"]] = [dec, reviewer, ov]
    return q


def test_review_without_reviewer_name_fails_loudly(tmp_path):
    _run(tmp_path)
    q = _reviewed(tmp_path, {0: ("approve", "", "")})
    q.to_csv(tmp_path / "review_queue_reviewed.csv", index=False)
    assert _run(tmp_path) == 2           # owner is mandatory (ValueError -> exit 2)


def test_edit_with_invalid_defect_type_fails_loudly(tmp_path):
    _run(tmp_path)
    _reviewed(tmp_path, {0: ("edit", "Quang", "banana")}).to_csv(tmp_path / "review_queue_reviewed.csv", index=False)
    assert _run(tmp_path) == 2


def test_full_review_writes_audit_log_and_policy_decision(tmp_path):
    _run(tmp_path)
    rows = {i: ("approve", "Quang", "") for i in range(20)}
    rows[0] = ("reject", "Quang", "")      # reviewer says top-1 is a false alarm
    top1 = pd.read_csv(tmp_path / "review_queue.csv").item_id[0]
    _reviewed(tmp_path, rows).to_csv(tmp_path / "review_queue_reviewed.csv", index=False)
    assert _run(tmp_path) == 0
    pred = pd.read_csv(tmp_path / "ranked_review.csv").set_index("image_id")
    assert pred.loc[top1, "score"] == 0.0 and pred.loc[top1, "pred_action"] == "KEEP"
    log = [json.loads(line) for line in (tmp_path / "audit_log.jsonl").read_text().splitlines()]
    assert any(e["action"] == "review:reject" and e["item_id"] == top1 and e["actor"] == "Quang" for e in log)
    d = json.loads((tmp_path / "decision.json").read_text())
    assert d["status"] == "RECAPTURE" and d["reason_code"] == "DEFECT_RATE_ABOVE_LIMIT"


def test_escalation_forces_hold(tmp_path):
    _run(tmp_path)
    rows = {i: ("approve", "Quang", "") for i in range(20)}
    rows[3] = ("escalate", "Quang", "")
    _reviewed(tmp_path, rows).to_csv(tmp_path / "review_queue_reviewed.csv", index=False)
    assert _run(tmp_path) == 0
    assert json.loads((tmp_path / "decision.json").read_text())["reason_code"] == "REVIEWER_ESCALATED"


def test_provenance_mismatch_stops_pipeline(tmp_path, monkeypatch):
    from src import pipeline
    monkeypatch.setattr(pipeline, "verify_manifest", lambda p: [{"problem": "hash_mismatch"}])
    assert _run(tmp_path) == 2
