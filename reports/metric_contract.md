# Metric contract (khai báo bắt buộc — chốt ở phút 30, trước khi chạy eval)

> Các metric chuẩn đã được cài trong `src/metrics/tracks.py`. File này là nơi nhóm **cam kết** cách dùng chúng.
> Không đổi testset, threshold hay trọng số sau khi đã thấy kết quả eval.

| Mục | Khai báo của nhóm |
|---|---|
| Track | T02 · Raw QC Triage (nhóm G27) |
| Primary metric (tên đúng như evaluator in ra) | **Defect Recall@B** (kèm Precision@B) |
| Tăng tốt hay giảm tốt? | higher |
| Đơn vị đánh giá (image / region / pair / release / batch / case …) | image |
| Công thức + numerator / denominator | Recall@B = #ảnh lỗi thật (GT) nằm trong top-B theo `score` / #ảnh lỗi thật (GT) trong split eval. Precision@B = cùng numerator / B |
| Cách ghép prediction ↔ GT (IoU? id? tolerance?) | khớp theo `image_id`, không tolerance; tie-break `mergesort` (ổn định) theo thứ tự file |
| Candidate pool cố định | 40 ảnh split `eval` của `fixtures/t02_raw_qc/images.csv` (pipeline chấm cả 96 ảnh, evaluator chỉ giữ eval) |
| Review budget B / K | B = 20 (`configs/tracks/t02.yaml`) |
| Seed | 42 |
| Threshold policy (chọn trên dev như thế nào) | 4 ngưỡng quy tắc (blur / dark / glare / compression) chọn bằng cách nhìn phân bố feature trên **56 ảnh dev đã tự gắn nhãn** (`reports/self_labels_dev.csv`), đặt ở giữa khoảng trống giữa nhóm lỗi và nhóm sạch. Giá trị đóng băng nằm trong `src/t02_features.py::DEFAULTS`. Tham chiếu độ nét theo camera = median log-Laplacian của ảnh dev (không dùng nhãn) |
| Eval split (không dùng để tune) | `eval` (40 ảnh). Không đọc `gt/`, không tune trên eval. Lưu ý trung thực: người làm đã xem vài ảnh eval bằng mắt khi khảo sát dữ liệu (ảnh glare, ảnh đêm) trước khi chốt ngưỡng; ngưỡng cuối chỉ dựa trên số liệu dev |
| Chỉ số lỗi xấu (miss / false alarm / false reject) | miss = ảnh lỗi thật không vào top-B; false alarm = ảnh sạch vào top-B (đếm theo slice `light`) |
| Slices sẽ báo cáo (≥2) | `light` (day/dusk/night), `camera_id` (cam_A/cam_B), `defect_type` |
| Cách báo độ bất định khi N nhỏ (Wilson CI / bootstrap) | N_eval = 40 (khoảng 20 ảnh lỗi) → báo Wilson 95% CI cho Recall@B; không diễn giải chênh lệch dưới ~15 điểm % là chắc chắn |
| Ngày giờ chốt + người chốt | 2026-11-14 · Nguyen Van Quang |

Nếu baseline = 0 thì ghi rõ "baseline = 0" thay vì phần trăm cải thiện.

## Chính sách quyết định (chốt cùng lúc)

- Mỗi lô: **HOLD** (`REVIEW_PENDING`) nếu còn ảnh trong top-B chưa được người review; **HOLD** (`REVIEWER_ESCALATED`) nếu reviewer escalate;
  **RECAPTURE** (`DEFECT_RATE_ABOVE_LIMIT`) nếu tỉ lệ ảnh bị gắn cờ sau review > 10% (`MAX_DEFECT_RATE` trong `src/pipeline.py`); ngược lại **KEEP**.
- `owner` lấy từ `configs/lab.yaml`.
