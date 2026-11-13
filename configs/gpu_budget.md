# Ngân sách GPU

Người theo dõi: thành viên A. Cập nhật sau mỗi lần train.

- Ước tính: 3–4 giờ/lần train RPG trên GPU T4/P100 (Kaggle).
- Tổng kế hoạch: khoảng 11 lần train, tương đương 35–45 giờ GPU, trải trên 3 tuần và 2–3 tài khoản Kaggle/Colab.
- Ngưỡng cảnh báo: 80% ngân sách (khoảng 28–36 giờ) → áp dụng danh sách cắt giảm trong `docs/project_plan.md`, mục 1.3.

## Kế hoạch các lần train

| # | Run | Seed | Số lần | Ghi chú |
|---|---|---|---|---|
| 1 | `rpg_main` | 42, 43, 44 | 3 | Cấu hình chính, không cắt giảm |
| 2 | `rpg_pq`, `rpg_rq`, `rpg_random` | 42 | 3 | RQ1; cắt theo thứ tự: RQ trước |
| 3 | `rpg_behavior` | 42, 43, 44 | 3 | Nâng cấp chính; giảm còn 2 seed nếu thiếu ngân sách |
| 4 | Dự phòng | – | 2 | Chạy lại lỗi, α thứ hai hoặc encoder mới |

## Nhật ký sử dụng

| Ngày | Run | Seed | Nền tảng / tài khoản | Giờ GPU | Tích lũy (giờ) | Còn lại (giờ) |
|---|---|---|---|---|---|---|
| | | | | | 0 | |
