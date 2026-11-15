# Ngân sách GPU

Người theo dõi: thành viên A. Cập nhật sau mỗi lần train.

- Thực tế: 1,2–1,6 giờ/lần train RPG trên Colab T4 (trung bình 1,37 giờ qua 3 seed main, 14/11); ước tính ban đầu 3–4 giờ.
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

## Nhật ký sử dụng

| Ngày | Run | Seed | Nền tảng / tài khoản | Giờ GPU | Tích lũy (giờ) | Còn lại (giờ) |
|---|---|---|---|---|---|---|
| 14/11 | `smoke` (2 epoch) | 42 | Colab / TK A | 0,18 | 0,18 | 44,82 |
| 14/11 | `main_s42` | 42 | Colab / TK A | 1,60 | 1,78 | 43,22 |
| 14/11 | `main_s42_salvage` (chỉ chấm test) | 42 | Colab / TK A | 0,05 | 1,83 | 43,17 |
| 14/11 | `main_s43` | 43 | Colab / TK A | 1,27 | 3,10 | 41,90 |
| 14/11 | `main_s44` | 44 | Colab / TK A | 1,23 | 4,33 | 40,67 |
| 14/11 | `main_s43_salvage` (chỉ chấm test) | 43 | Colab / TK A | 0,04 | 4,37 | 40,63 |