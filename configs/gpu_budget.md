# Ngân sách GPU

Người theo dõi: thành viên A. Cập nhật sau mỗi lần train.

- Thực tế: 1,2–1,6 giờ/lần train RPG trên Colab T4 (trung bình 1,37 giờ qua 3 seed main, 14/11); ước tính ban đầu 3–4 giờ.
- SASRec-BCE theo S³-Rec hoàn tất 3 seed ngày 20/11 với khoảng 0,54 giờ. Tổng thời gian gồm các lần chạy thử và chạy chính là khoảng 5,98/45 giờ, còn khoảng 39,02 giờ.
- Kiểm tra adapter ngày 23/11 (chế độ a 0,11 giờ; Cổng 2 1,71 giờ; chấm lại 0,05 giờ) nâng tổng lên khoảng 7,85/45 giờ, còn khoảng 37,15 giờ.
- RQ1 ngày 24/11 trên Kaggle (PQ và ngẫu nhiên, mỗi biến thể 3 seed, 8,66 giờ) nâng tổng lên khoảng 16,51/45 giờ, còn khoảng 28,49 giờ.
- Nền tảng: Google Colab (chính) + Kaggle (dự phòng, chạy song song từ Tuần 3). Mọi output ghi vào Drive `MyDrive/rpg/`.
- Ngân sách trần: 45 giờ GPU. Dự kiến dùng khoảng 20–25 giờ (17 lần train nếu các biến thể RQ1 chạy 3 seed).
- Ngưỡng cảnh báo: 80% ngân sách (36 giờ) → áp dụng danh sách cắt giảm trong `docs/project_plan.md`, mục 1.3.

## Kế hoạch các lần train

| # | Run | Seed | Số lần | Ghi chú |
|---|---|---|---|---|
| 1 | `rpg_main` | 42, 43, 44 | 3 | Cấu hình chính, không cắt giảm. **Xong 14/11** |
| 2 | `rpg_pq`, `rpg_rq`, `rpg_random` | 42 (đề xuất 42, 43, 44) | 3 (đề xuất 9) | RQ1. Đề xuất 3 seed vì chênh lệch giữa các seed của main khoảng 8%; cắt theo thứ tự: RQ trước |
| 3 | `rpg_behavior` | 42, 43, 44 | 3 | Nâng cấp chính; giảm còn 2 seed nếu thiếu ngân sách |
| 4 | Dự phòng | – | 2 | Chạy lại lỗi, α thứ hai hoặc encoder mới |
| 5 | SASRec-BCE theo S³-Rec | 42, 43, 44 | 3 | **Xong 20/11**, best epoch 54/78/54 |
| 6 | Kiểm tra adapter (Cổng 2) | 42 | 1 + 1 thử chế độ (a) | **Xong 23/11**, kết quả giống hệt `main_s42` |

## Nhật ký sử dụng

| Ngày | Run | Seed | Nền tảng / tài khoản | Giờ GPU | Tích lũy (giờ) | Còn lại (giờ) |
|---|---|---|---|---|---|---|
| 14/11 | `smoke` (2 epoch) | 42 | Colab / TK A | 0,18 | 0,18 | 44,82 |
| 14/11 | `main_s42` | 42 | Colab / TK A | 1,60 | 1,78 | 43,22 |
| 14/11 | `main_s42_salvage` (chỉ chấm test) | 42 | Colab / TK A | 0,05 | 1,83 | 43,17 |
| 14/11 | `main_s43` | 43 | Colab / TK A | 1,27 | 3,10 | 41,90 |
| 14/11 | `main_s44` | 44 | Colab / TK A | 1,23 | 4,33 | 40,67 |
| 14/11 | `main_s43_salvage` (chỉ chấm test) | 43 | Colab / TK A | 0,04 | 4,37 | 40,63 |
| 19/11 | `sasrec_ce_s42` | 42 | Kaggle / B | ≈ 0,35 | ≈ 4,72 | ≈ 40,28 |
| 19/11 | `sasrec_ce_s43` | 43 | Kaggle / B | ≈ 0,39 | ≈ 5,11 | ≈ 39,89 |
| 19/11 | `sasrec_ce_s44` | 44 | Kaggle / B | ≈ 0,33 | ≈ 5,44 | ≈ 39,56 |
| 20/11 | `sasrec_paper_s42` | 42 | Kaggle / B | ≈ 0,16 | ≈ 5,60 | ≈ 39,40 |
| 20/11 | `sasrec_paper_s43` | 43 | Kaggle / B | ≈ 0,22 | ≈ 5,82 | ≈ 39,18 |
| 20/11 | `sasrec_paper_s44` | 44 | Kaggle / B | ≈ 0,16 | ≈ 5,98 | ≈ 39,02 |
| 23/11 | `adapter_emb_smoke` (chế độ a, 1 epoch) | 42 | Colab / TK A | 0,11 | ≈ 6,09 | ≈ 38,91 |
| 23/11 | `adapter_opq_s42` (Cổng 2) | 42 | Colab / TK A | 1,71 | ≈ 7,80 | ≈ 37,20 |
| 23/11 | `adapter_opq_s42_salvage` (chỉ chấm test) | 42 | Colab / TK A | 0,05 | ≈ 7,85 | ≈ 37,15 |
| 24/11 | `pq_s42` (RQ1) | 42 | Kaggle / TK A | 1,63 | ≈ 9,48 | ≈ 35,52 |
| 24/11 | `random_s42` (RQ1) | 42 | Kaggle / TK A | 1,41 | ≈ 10,89 | ≈ 34,11 |
| 24/11 | `pq_s43` (RQ1) | 43 | Kaggle / TK A | 1,41 | ≈ 12,30 | ≈ 32,70 |
| 24/11 | `random_s43` (RQ1) | 43 | Kaggle / TK A | 1,22 | ≈ 13,52 | ≈ 31,48 |
| 24/11 | `pq_s44` (RQ1) | 44 | Kaggle / TK A | 1,49 | ≈ 15,01 | ≈ 29,99 |
| 24/11 | `random_s44` (RQ1) | 44 | Kaggle / TK A | 1,50 | ≈ 16,51 | ≈ 28,49 |

Các lượt SASRec dùng `elapsed_seconds / 3600` trong `run.json` hoặc
`run_seconds / 3600` trong `result.json` làm ước tính thời gian chạy
CUDA một tiến trình. Ước tính này tính từ lúc bắt đầu chạy mô hình đến lúc
chấm xong test. Lượt S³-Rec ghi nhận Tesla T4 trong `result.json`.
