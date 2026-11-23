# Họp tuần 15/11: FP-Growth và AR-Rec (phần C)

Tài liệu chuẩn bị cho nội dung họp của C trong kế hoạch tuần 2 (`docs/project_plan.md`, mục 4.2): trình bày FP-Growth và AR-Rec, và vẽ lại luồng dữ liệu của hai phần còn lại. Số liệu lấy từ `reports/` và mã trong `src/`; mọi kết quả AR-Rec là trên bộ chia validation, chưa chạy test.

## 1. Luồng dữ liệu phần C

```text
data/processed/interactions.csv + data/splits/leave_last_out.csv
        │  src/patterns.py: load_train_transactions (chỉ tương tác thuộc train)
        ▼
giao dịch = tập item duy nhất của chuỗi train mỗi user (22.363 giao dịch, 12.101 item)
        │  build_transaction_matrix (ma trận thưa) → run_fpgrowth (mlxtend, min_support, max_len)
        ▼
tập phổ biến ── generate_rules (min_confidence, |X ∪ Y| ≤ max_len) ──► luật X ⇒ Y: support, confidence, lift
        │  src/ar_rec.py: ARRecommender
        ▼
với mỗi user, H = seen_item_ids của validation.jsonl
  1. khớp luật có X ⊆ H (chỉ mục đảo item → luật)
  2. điểm của item y = max(confidence × lift) trên các luật khớp có y ∈ Y
  3. xếp giảm dần theo điểm; bổ sung Popularity (tần suất train) cho đủ độ dài
        │  xuất max_k + len(seen_item_ids) item mỗi user
        ▼
src/evaluate.py: Recall@K, NDCG@K, exclude_seen = false (quyết định 19/11)
```

Ràng buộc chống rò rỉ: luật và Popularity chỉ lấy từ train; khi chấm test, `seen_item_ids` gồm cả item validation nhưng luật vẫn chỉ từ train.

## 2. FP-Growth

- Bước 1: đếm item, bỏ item dưới ngưỡng, dựng FP-tree (hai lần duyệt dữ liệu). Bước 2: với mỗi item, lấy cơ sở mẫu điều kiện, dựng cây điều kiện, khai phá đệ quy. Không sinh tập ứng viên như Apriori.
- Tập phổ biến thành luật bằng support(X ∪ Y) / support(X) ≥ min_confidence; lift = confidence / support(Y).
- Trên Beauty (`reports/fpgrowth_sweep.json`, `max_len` 4, `min_confidence` 0,1):

| min_support | Đếm tối thiểu | Item phổ biến | Tập phổ biến | Luật | Luật 1 ⇒ 1 | Thời gian FP-Growth (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 0,002 | 45 | 601 | 716 | 290 | 204 | 0,9 |
| 0,001 | 23 | 1.452 | 4.698 | 20.024 | 2.837 | 2,0 |
| 0,0005 | 12 | 3.218 | 69.033 | 656.681 | 13.127 | 2,8 |

- Số luật tăng rất nhanh khi giảm min_support; dưới 1% cặp item đồng xuất hiện đạt ngưỡng ở các mức này (`reports/ar_rec_sparsity.md`, mục 2).
- Lift ≤ 1 / max(sup(X), sup(Y)), nên ở support thấp các luật lift cao nhất dựa trên khoảng 12 giao dịch (`reports/fpgrowth_top_rules.md`); đối chiếu thêm Kulczynski.

## 3. AR-Rec (validation)

Cấu hình được chọn từ lưới 105 cấu hình theo NDCG@10 không lọc item đã xem (`reports/ar_rec_sweep.md`): `max_len` 2, `min_support` 0,0001, `min_confidence` 0,05; 84.883 luật; 91,9% user có luật khớp.

| Mô hình | Recall@10 | NDCG@10 |
|---|---:|---:|
| Popularity | 0,01628 | 0,00785 |
| AR-Rec, min_support 0,002 | 0,02392 | 0,01323 |
| AR-Rec, min_support 0,001 | 0,03345 | 0,01867 |
| AR-Rec, min_support 0,0005 | 0,04185 | 0,02306 |
| AR-Rec, cấu hình chọn (0,0001) | 0,06985 | 0,03728 |

Nhận xét:

- NDCG@10 tăng khi giảm min_support đến đếm tối thiểu 3, rồi giảm ở đếm tối thiểu 2. Hiệu so với Popularity là +0,02943 (sai số chuẩn 0,00105).
- `max_len` 3 và 4 không vượt `max_len` 2 tại các mức support dùng chung, trừ `max_len` 3 ở min_support 0,001 (NDCG@10 cao hơn 0,00018); lưới của chúng chỉ xuống đến 0,0005 vì số luật.
- Cấu hình chọn yêu cầu ít nhất 3 giao dịch mỗi luật, không phải đúng 3: 22.011 / 84.883 luật (25,9%) có đúng 3 giao dịch, trung vị 5, lớn nhất 88. Phần lớn luật dựa trên ít giao dịch, gần với thống kê đồng xuất hiện item–item hơn là luật mua kèm bền vững.
- Nửa số user (11.383 / 22.363) có không quá 4 item train; ở support 0,002 chỉ 7,0% nhóm này có luật khớp, phần còn lại do Popularity.
- Giới hạn: cấu hình chọn bằng argmax trên chính validation nên mức cải thiện chưa hiệu chỉnh cho việc chọn; ước lượng không chệch cần chạy test một lần.

## 4. Luồng dữ liệu của A và B (theo tài liệu và log trong repo)

Chỉ ghi những gì đã có trong repo; kiến trúc RPG (loss MTP, giải mã đồ thị) do A trình bày từ code gốc, không suy ra ở đây.

- B: `scripts/download_data.py` → `src/preprocessing.py` (5-core, sắp theo timestamp, leave-last-out) → `data/processed/*` và `data/splits/*` → `src/evaluate.py` (Recall@K, NDCG@K dùng chung); baseline Popularity, ItemKNN, SASRec (`configs/baselines_recbole.yaml`). Nguồn: `docs/data_contract.md`.
- A: văn bản item ghép theo `_extract_meta_sentences` của code gốc → embedding `sentence-t5-base` → OPQ (m = 32, codebook 256) → semantic ID 32 token → huấn luyện RPG bằng code gốc. Hàng i của embedding và semantic ID tương ứng `item_id` i + 1 (`reports/decisions.md`, 15/11). Kết quả 3 seed (seed 42, 43, 44; `experiments/log.csv`): NDCG@10 test 0,0420 ± 0,0018.
- Điểm giao với C: C dùng chung `interactions.csv`, bộ chia và `src/evaluate.py` của B; embedding và semantic ID của A dùng cho các bước đối chiếu luật ↔ token ở 18/11.
