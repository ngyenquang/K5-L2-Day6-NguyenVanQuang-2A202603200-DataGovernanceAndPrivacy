# Group Report — Day 06 Data Lab (Field-to-Release PoC)

> Mục có chữ **TODO** là phần thành viên phải tự điền/xác nhận (không thể suy ra từ repo).
> Số liệu official (`metrics_*.json`, `table1`) cần gt/ của giảng viên nên **chưa có**; bảng dưới là *self-eval trên nhãn dev tự gắn*.

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | G27 |
| Challenge | T02 · Raw QC Triage (blur / glare / dark / compression) |
| Thành viên + vai trò | **Làm cá nhân** (theo labcoach): Nguyen Van Quang — Lead/Integrator, Core algorithm, QA, Review + report |
| Git URL | https://github.com/ngyenquang/K5-Day6-L2-NguyenVanQuang-2A202603200-DataGovernanceAndPrivacy |
| Tag `demo-180m` (commit) | `7315207` |
| Tag `final-24h` (commit) | commit cuối cùng của repo (không thể tự ghi SHA vào chính commit đó) — xem `git rev-list -n 1 final-24h` |
| Deadline (giờ VN) | 12:00 trưa 2026-11-15 |

## 01 · Problem statement

- Ai bị ảnh hưởng / ai chịu hậu quả khi hệ thống sai? Đội huấn luyện/gán nhãn nhận lô ảnh thô: ảnh lỗi lọt qua làm hỏng dữ liệu train; ảnh tốt bị loại làm mất dữ liệu hiếm (đặc biệt ảnh đêm) và tốn công chụp lại.
- Điểm nghẽn cụ thể: người review chỉ xem được B=20 ảnh; cần xếp hạng để 20 ảnh đó chứa nhiều ảnh lỗi nhất.
- Đầu vào → đầu ra: `images.csv` + 96 ảnh 320×180 → `ranked_review.csv` (`image_id, score, pred_defect_type, pred_action`, …) → queue top-20 + contact sheet → quyết định lô KEEP / RECAPTURE / HOLD.
- Tiêu chí GO / HOLD / REWORK (chốt trước khi chạy eval): HOLD nếu còn ảnh top-B chưa review hoặc có escalate; RECAPTURE (REWORK) nếu tỉ lệ ảnh bị gắn cờ sau review > 10%; ngược lại KEEP (GO).
- Giả định của nhóm (cần domain owner xác nhận): ngưỡng 10% và định nghĩa "lỗi" (4 loại) do nhóm tự đặt; dữ liệu là synthetic nên chưa phản ánh ảnh thật.

## 02 · Data & provenance

| Split | N | Nguồn / link | License | Synthetic? |
|---|---|---|---|---|
| dev | 56 | `fixtures/t02_raw_qc` (giảng viên cấp) | theo `manifest.csv` | có |
| eval / heldout | 40 | `fixtures/t02_raw_qc` (giảng viên cấp) | theo `manifest.csv` | có |

- Schema & class mapping: `image_id, path, camera_id, light, split`; lớp lỗi ∈ {blur, dark, glare, compression}, `none` = sạch.
- Seed: 42 · Cách chia split: theo cảnh gốc (24 cảnh × 4 ảnh), do fixture định sẵn.
- `manifest.csv` verified (sha256): có — `python -m tools.verify_manifests` (preflight báo "hashes match").
- Dữ liệu thật dùng (nếu có): không. GT cho dev: nhóm **tự gắn nhãn bằng mắt** 56 ảnh dev (`reports/self_labels_dev.csv`, 28 lỗi / 28 sạch); không đọc `gt/`.

## 03 · Research & design

- References và vì sao chọn kỹ thuật: xem mục 11. Laplacian variance làm baseline (đo độ nét nhưng cần ngưỡng thủ công); ACDC làm cơ sở cho các điều kiện bất lợi (đêm/sương/mưa) và ảnh đêm là ca khó; CVAT QA cho ý tưởng review/validation set. ISO/IEC TR 5259-6 có trong đề nhưng nhóm chưa đọc nên không trích.
- Sơ đồ data flow: `images.csv` → (load + kiểm hash) → 5 feature/ảnh (`log_lap`, `p95`, `mean`, `cell_sat`, `block`) → 4 điểm lỗi, `score = max` → top-20 + contact sheet → reviewer (approve/reject/edit/escalate) → override + `audit_log.jsonl` → `decide()` → `decision.json`.
- Thành phần offline: chỉ numpy/scipy/Pillow, không model, không mạng.
- Trade-offs: luật + ngưỡng thủ công (giải thích được, nhanh ~1 ms/ảnh, nhưng ngưỡng chỉ tin cậy trên miền dev); không dùng học máy vì chỉ có 56 ảnh dev tự gắn nhãn.
- Thiết kế đối phó 3 failure mode bắt buộc: (1) dark dựa **p95** (ảnh đêm có đèn/cửa sổ nên p95 cao, không bị coi là tối); (2) glare dựa **ô lưới 4×4 cháy nhiều nhất**, bắt được glare cục bộ trên ảnh nét; (3) blur chuẩn hoá theo **tham chiếu độ nét của từng camera** (cam_B mềm hơn).

## 04 · Baseline vs PoC

- Baseline: điểm = `1/(laplacian_variance + 1e-6)` toàn ảnh, một thang duy nhất (`src/baseline_laplacian.py`). Không có tham số tune. Commit: `c0f1612`.
- PoC: `src/pipeline.py` + `src/t02_features.py` (rule-based đa lỗi, chuẩn hoá camera). Commit: `e2d37ad`.
- Cùng budget (B=20) / cùng pool / cùng seed 42: có (cả hai chạy bằng `src.main --seed 42`, cùng `images.csv`).

**Bảng 1 · Baseline vs PoC (self-eval trên DEV, nhãn tự gắn — không phải số official)** — sinh bởi `python -m tools.self_eval_dev` → `reports/self_eval_dev.md`:

| Metric (unit) | Baseline | PoC | Δ & chú thích |
|---|---|---|---|
| Recall@20 (dev, N=56, 28 lỗi) | 0.571 (16/28) | 0.714 (20/28) | +0.143; trần của B=20 là 20/28 = 0.714 nên PoC đã chạm trần |
| Precision@20 | 0.800 | 1.000 | +0.200 |
| Recall@28 (B = #lỗi) | 0.821 (23/28) | 1.000 (28/28) | +0.179 |
| False alarm top-20, cam_B (ảnh sạch) | 4/10 | 0/10 | baseline báo nhầm ảnh cam_B mềm |
| Defect-type accuracy | n/a | 1.00 | baseline không dự đoán loại lỗi |
| Runtime / 100 items | 0.21 s | 0.39 s | do `src.compare` đo; dao động 0.3–0.9 s giữa các lần chạy; CPU, Python 3.13, Windows 11 |

**Cảnh báo quan trọng:** ngưỡng PoC được chọn *trên chính các ảnh dev này* và nhãn do chính nhóm gắn, nên con số PoC trên dev là **lạc quan**, chưa phải bằng chứng tổng quát hoá. Số official trên eval (`reports/metrics_*.json`, `table1_baseline_vs_poc.md`) sẽ do giảng viên chạy `src.evaluate` với gt/ — **TODO: dán vào khi có**.
Quan sát không dùng nhãn trên eval: PoC gắn cờ 17/40 ảnh eval (43%) so với 28/56 (50%) trên dev; vài ảnh sát ngưỡng (score 0.88–0.90: `img_06_0`, `img_06_1`, `img_19_2`) **nằm ngoài top-20** (hạng 20 có score 1.66) nên người review không thấy; nếu chúng là lỗi thật thì sẽ là miss — rủi ro cần kiểm khi có số official.

## 05 · Evaluation protocol

- Công thức: Recall@B = lỗi thật trong top-B / tổng lỗi thật (eval, N=40); Precision@B = lỗi thật trong top-B / B (xem `reports/metric_contract.md`).
- Threshold / budget B: ngưỡng chọn trên dev, B=20; chốt ngày 2026-11-14 trước khi có bất kỳ kết quả eval nào.
- Độ bất định: N nhỏ → báo Wilson 95% CI cho Recall@B khi có số official (**TODO**).
- Lệnh evaluate: `python -m src.evaluate --track t02 --pred out/poc/ranked_review.csv --run-name poc --out reports/metrics_poc.json` (cần gt/ giảng viên; repo học viên trả exit 2).

## 06 · Result slices

| Slice (dev, B=20, ảnh sạch bị đưa vào top-20) | N sạch | Baseline | PoC | Ghi chú |
|---|---|---|---|---|
| cam_A | 18 | 0/18 | 0/18 | |
| cam_B | 10 | 4/10 | 0/10 | cam_B quang học mềm → baseline báo nhầm |
| light=night | 2 | 0/2 | 0/2 | N rất nhỏ |
| light=day / dusk | 18 / 8 | 3/18, 1/8 | 0/18, 0/8 | |

- **Failure case 1 — glare cục bộ bị bỏ sót (baseline):** `img_02_2`, `img_20_3`, `img_18_2`, `img_22_3` là ảnh nét nhưng đèn pha/sương cháy sáng; Laplacian toàn ảnh xếp chúng hạng 60, 79, 32, 53 trên 96 ảnh. Root cause: sharpness không đo độ cháy. PoC dùng `cell_sat` (ô cháy nhất) → cả 4 ảnh có score 1.66–2.06.
- **Failure case 2 — cam_B bị báo nhầm (baseline):** `img_09_0` (cam_B day) và `img_12_0` (cam_B dusk) nhìn bình thường nhưng vào top-20 baseline (hạng 19 và 17). Root cause: cam_B có median log-Laplacian thấp hơn cam_A (≈6.3 vs ≈7.0 trên ảnh sạch dev; `ctx.stats["camera_reference_log_lap"]`). PoC chuẩn hoá theo camera: `img_09_0` score 0.67.
- **Failure case 3 — ảnh tối bị xếp cao vì sai lý do (baseline):** ảnh dark có Laplacian thấp nên baseline gộp với blur; PoC tách riêng `dark` qua p95 và không phạt ảnh đêm có đèn.
- **Giới hạn đã biết của PoC:** các ảnh compression nhẹ nằm sát ngưỡng blockiness (dev: lỗi min 1.88 vs sạch max 1.70, ngưỡng 1.8) N dev nhỏ.
- **Giới hạn: điểm của các loại lỗi không cùng thang.** `score` là max của 4 điểm con có thang khác nhau (glare tới 16, dark tới 6, blur ≈ 1.1–1.2), nên khi số ảnh bị gắn cờ > B thì blur bị xếp thấp nhất. Dấu hiệu trên dev (B=20 < 28 lỗi): Recall@20 theo loại lỗi là blur 3/8, compression 9/10, glare 4/4, dark 4/6 (số tính bằng evaluator của repo với nhãn dev tự gắn). Trên eval PoC chỉ gắn cờ 17/40 (≤ B) nên ảnh blur không bị cắt, nhưng đây là rủi ro khi lô lỗi nhiều hơn B. Cách sửa (chưa làm, cần chọn trên dev): chuẩn hoá từng điểm con về cùng thang (ví dụ rank/percentile theo loại lỗi).
- **Kiểm ngoài miền (mốc +24h):** `python -m tools.ood_check` lấy 10 ảnh (seed 42) từ pool `fixtures/t13_rare_mining` (cùng bộ sinh synthetic nên chỉ khác nội dung cảnh, **không** phải ảnh thật), chấm bằng ngưỡng đã chốt: **0/10 bị gắn cờ**, cao nhất `pool_158` = 0.90 (<1). Cả 10 ảnh đã xem bằng mắt: đều sạch. Kiểm này chỉ đo báo nhầm, không đo recall; chưa có ảnh thật (`data/real/` không có).
- Latency / compute: 0.3–0.9 s cho 96 ảnh (≈ 0.4 s/100 ảnh theo `src.compare`), CPU, Windows 11, Python 3.13.7.

## 07 · Human & decision

- Review queue: `out/poc/review_queue.csv` (top-B=20 theo `score`) + `contact_sheet.html` (ảnh, score, loại lỗi, slice).
- Override + audit log: reviewer chọn approve/reject/edit/escalate/skip → `out/poc/review_queue_reviewed.csv` → chạy lại → mỗi thay đổi ghi vào `out/poc/audit_log.jsonl` (actor, before, after, lý do). `reject` = báo nhầm (score→0, KEEP); `edit` = sửa loại lỗi qua `override_value`; `escalate` → HOLD.
- Decision policy + `reason_code` + owner: xem `src/pipeline.py::decide` — `REVIEW_PENDING`, `REVIEWER_ESCALATED`, `DEFECT_RATE_ABOVE_LIMIT`, `DEFECT_RATE_WITHIN_LIMIT`; owner = `Nguyen Van Quang - reviewer`.
- Khi nào HOLD/REWORK thay vì tự động release: chưa có người duyệt đủ top-B → HOLD (không bao giờ tự KEEP); tỉ lệ lỗi sau review > 10% → RECAPTURE.
- **Vòng review (2026-11-14 → 2026-11-15):** bản nháp quyết định do trợ lý AI chuẩn bị (log ghi actor `AI`, 20 dòng; nhãn actor ở bản trong `reports/` được rút gọn thành `AI` sau khi chạy, bản gốc `out/poc/audit_log.jsonl` và lịch sử git `f277734` giữ nguyên chuỗi dài), sau đó **Nguyen Van Quang (người thật, owner) đã mở contact sheet, kiểm tra từng ảnh và xác nhận** (log ghi actor `Nguyen Van Quang`, thêm 20 dòng; log chỉ ghi nối thêm nên giữ cả hai lượt, tổng 40 dòng). Kết quả: 20/20 ảnh top-20 `approve` (0 reject, 0 edit, 0 escalate): 6 glare, 6 dark, 8 compression; chi tiết trong `reports/review_queue_reviewed.csv`. Decision sau review: **RECAPTURE** (`DEFECT_RATE_ABOVE_LIMIT`, 45/96 = 47% > 10%). Bằng chứng: `reports/review_queue_reviewed.csv`, `reports/audit_log.jsonl` (40 dòng), `reports/decision.json`, `reports/decision_log.csv`. Lưu ý trung thực: reviewer chỉ có một người và bản nháp do AI đề xuất nên có rủi ro anchoring; người review không phải là người thứ hai độc lập.
- Hạn chế của vòng review: 0 override nên chưa chứng minh được đường `reject`/`edit` trên dữ liệu thật (chỉ có test tự động). Ba ảnh sát ngưỡng ngoài top-20 (`img_06_0`, `img_06_1`, `img_19_2`) không được xem vì vượt budget B=20.

## 08 · Reproduce & tests

```bash
python -m venv .venv && .\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.main --config configs/lab.yaml --track t02 --output out/poc --run-name poc --offline --seed 42
.\.venv\Scripts\python.exe -m src.main --track t02 --pipeline src.baseline_laplacian --output out/baseline --run-name baseline --offline --seed 42
.\.venv\Scripts\python.exe -m tools.self_eval_dev --baseline out/baseline/ranked_review.csv --poc out/poc/ranked_review.csv
.\.venv\Scripts\python.exe -m pytest -q
```

- Output của `pytest` (`addopts = -q` đã có sẵn trong pyproject): `60 passed, 39 skipped` — 0 failed. 13 test của nhóm trong `tests/test_t02_pipeline.py`; test yêu cầu gt/ bị skip theo thiết kế.
- Máy / OS / Python: Windows 11 · Python 3.13.7 · git sha: `7315207` (HEAD khi chạy; có thể thêm commit sau đó).

## 09 · Limitations & ethics

- Rủi ro pháp lý / PII / license: fixtures synthetic, không có người/biển số thật. Nếu dùng ảnh thật (có người, biển số) thì không đưa lên repo công khai.
- Điều chưa biết: hiệu năng thật trên eval (chờ gt/); tổng quát hoá sang ảnh thật; false alarm khi có điều kiện ánh sáng mới. Nhãn dev do 1 người gắn bằng mắt (không có người thứ hai kiểm) — có thể sai ở ca "nhẹ". Khi scale 10×: luật cứng cần hiệu chỉnh lại theo từng camera/lô.
- Không tuyên bố "đạt GDPR / luật Việt Nam" từ checklist tự động; mọi giả định cần DPO/legal/domain owner xác nhận.
- Kế hoạch cải tiến: gắn nhãn dev bởi ≥2 người, thêm ảnh thật làm hard negative, ngưỡng theo camera × light, thêm ROI vùng biển số.

## 10 · Team contribution

**Bảng 2 · Individual contribution evidence** (đối chiếu `git shortlog -sne --all` và `reports/team_contributions.csv`)

| Học viên | Git author/email | Module / vai trò | Commit SHA / diff chứng minh | Trình bày? |
|---|---|---|---|---|
| Nguyen Van Quang (làm cá nhân) | nguyenquang / nguyenvanquang30112k4@gmail.com | pipeline, features, baseline, tests, report | `a82b40b`, `c0f1612`, `e2d37ad`, `4cc1095`, `f277734`, `6862674`, `7315207` | Có (trình bày một mình) |

## 11 · References

| URL | Ngày xem | 1–2 dòng ý nghĩa (không chép nguyên văn) |
|---|---|---|
| https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/ | 2026-11-15 | Variance của Laplacian làm thước đo độ nét, cần ngưỡng chọn tay → cơ sở baseline và lý do PoC chuẩn hoá theo camera |
| https://arxiv.org/abs/2104.13395 | 2026-11-15 | ACDC: ảnh lái xe điều kiện bất lợi (sương, đêm, mưa, tuyết) có ảnh cặp bình thường → ảnh đêm là ca khó, không phải lỗi mặc định |
| https://docs.cvat.ai/docs/qa-analytics/auto-qa/ | 2026-11-15 | CVAT Automated QA: so với tập validation/ground truth nhỏ (5–15%) và có chế độ review → tương tự ý tưởng top-B review và nhãn dev tự gắn |

> Ba nguồn trên do trợ lý AI tra cứu ngày 2026-11-15 từ kết quả tìm kiếm, **chưa phải do tác giả đọc kỹ**: hãy đọc và sửa "ý nghĩa" bằng lời của mình trước khi nộp.

---

**Bảng 3 · Checklist chốt khi nộp**

- [ ] Link repo và tag cuối cùng trong README; input mẫu không chứa PII thật.
- [x] Đúng 1 lệnh run, 1 lệnh evaluate, 1 lệnh tests; không cần download khi chạy.
- [ ] Có baseline/PoC metrics official (chờ gt/), ≥1 failure case (đã có 3), reviewer evidence (có, người review: Nguyen Van Quang), decision log.
- [x] Có license/source provenance (manifest), data splits, seed, offline environment, N.
- [x] Làm cá nhân: tác giả có nhiều commit thực chất (xem bảng trên).
