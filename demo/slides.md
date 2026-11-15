# G27 · T02 Raw QC Triage — 5 slide demo (Nguyen Van Quang)

## 1. Bài toán
- 96 ảnh synthetic 320×180 (dev 56 / eval 40). Người review chỉ xem **B = 20 ảnh**.
- Mục tiêu: 20 ảnh đó chứa nhiều ảnh lỗi nhất (Defect Recall@20) → quyết định lô KEEP / RECAPTURE / HOLD.

## 2. Baseline vs PoC
- Baseline: 1 điểm Laplacian toàn ảnh (`src/baseline_laplacian.py`).
- PoC: 4 điểm lỗi (blur / dark / glare / compression), `score = max`, độ nét chuẩn hoá theo camera, glare đo theo ô lưới 4×4, dark đo bằng p95.
- Dev (nhãn tự gắn, lạc quan): Recall@20 0.571 → 0.714 (trần của B=20), Precision@20 0.80 → 1.00, báo nhầm cam_B 4/10 → 0/10.

## 3. 3 failure mode của baseline
- Glare cục bộ bị bỏ sót: `img_02_2`, `img_20_3`, `img_18_2`, `img_22_3` (hạng 60 / 79 / 32 / 53 trên 96).
- cam_B mềm bị báo nhầm: `img_09_0`, `img_12_0` (hạng 19 / 17).
- Ảnh tối bị gộp với ảnh mờ; ảnh đêm bình thường không phải lỗi.

## 4. Human-in-the-loop + quyết định
- `contact_sheet.html` → reviewer approve/reject/edit/escalate → `audit_log.jsonl` (40 dòng: nháp AI + xác nhận người).
- Policy: chưa review đủ top-B → HOLD; tỉ lệ lỗi sau review > 10% → RECAPTURE; còn lại KEEP.
- Kết quả: 20/20 approve, decision **RECAPTURE** (45/96 = 47%), owner Nguyen Van Quang.

## 5. Giới hạn trung thực
- Chưa có số official (không có `gt/`); số PoC trên dev lạc quan vì ngưỡng chọn trên dev.
- Điểm các loại lỗi không cùng thang → blur bị xếp thấp khi số cờ > B.
- Kiểm ngoài miền: 0/10 báo nhầm trên ảnh pool khác (vẫn synthetic). Chỉ 1 reviewer, nháp do AI.
