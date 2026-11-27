# Embedding đồng xuất hiện từ phần train

22,363 giao dịch train, 12,101 item, 128 chiều. Sinh bởi `python -m src.cooccurrence`.

- Giao dịch là tập item duy nhất trong phần train của mỗi user; validation và test không dùng để huấn luyện.
- SHA-256 của file SVD chỉ tái lập khi cùng số luồng BLAS: khác số luồng làm các phần tử lệch cỡ 1e-11 (đã đo trên một mẫu 8.000 giao dịch). File item2vec tái lập theo seed với `workers = 1`.
- Hàng j ứng với `item_id` j + 1; mỗi hàng chuẩn hóa L2; float32. Hàng toàn 0 nghĩa là item không có vector (không phải giá trị suy đoán).
- item2vec: skip-gram, negative = 5, 100 epoch, cửa sổ phủ cả giao dịch (window = 202), min_count = 1, seed 42.
- SVD: PPMI của XᵀX (bỏ đường chéo), SVD cắt cụt 128 chiều, embedding U·√Σ; phương sai giải thích của ma trận PPMI 0.301.

| Embedding | File | Hàng toàn 0 | SHA-256 |
|---|---|---:|---|
| item2vec | `item_emb_cooc_item2vec.npy` | 33 | `9f4a76d457205f8a25e8bec4de3bd70173e16cbfa7013a8a140f079f40348aa0` |
| svd | `item_emb_cooc_svd.npy` | 33 | `f2043e9fe7dbc5abe354fc1ed5d4c587a3228172d8681d96eebf42a0489f2919` |

Kiểm tra tín hiệu trên 22,363 user validation: xếp hạng theo cosine giữa vector item và trung bình vector lịch sử, loại item đã xem, bộ chấm `src/evaluate.py`. Popularity cũng loại item đã xem. Đây chỉ là phép thử có tín hiệu hay không; số liệu không so sánh trực tiếp với bảng kết quả chính (bảng chính giữ item đã xem).

| Phương pháp | Recall@10 | NDCG@10 |
|---|---:|---:|
| Popularity (train) | 0.0164 | 0.0080 |
| item2vec | 0.0348 | 0.0190 |
| SVD (PPMI) | 0.0351 | 0.0191 |
