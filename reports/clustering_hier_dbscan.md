# So sánh phân cụm: Ward, k-means và DBSCAN trên embedding

Embedding 12101 item × 768 chiều, chuẩn hóa L2, PCA 50 chiều (giữ 0.6947 phương sai), seed 42. Silhouette dùng khoảng cách cosine; Davies–Bouldin dùng khoảng cách Euclid. Sinh bởi `python -m src.clustering_compare`.

Nguồn embedding: `item_embeddings.npy`, SHA-256 `a2b542bdf8f34b1f6c81695b5cff7f3c3b98d75d06bab51721bd10b54628a836`. File không nằm trong repo (`*.npy` bị `.gitignore`). Theo `README.txt` của gói chia sẻ do nhóm A gửi (`share-20261125T050341Z-1-001.zip`), file mặc định `data/processed/item_embeddings.npy` là `item_emb_sentence_t5.npy` của gói: code gốc `RPG_KDD2025` commit `7dcf95c`, `sentence-t5-base`, 12101 × 768 float32 trước PCA, hàng i ứng với `item_id` i + 1 (đã đối chiếu với `items.jsonl`). Repo chưa có script tạo lại file (`scripts/build_semantic_ids.py` mới là khung rỗng) và chưa có liên kết tải cố định; người tái lập cần nhận gói từ nhóm A, chép file vào đường dẫn mặc định (hoặc truyền `--embeddings`) và đối chiếu SHA-256.

## Ward và k-means trên cùng mẫu 5000 item

| Thuật toán | k | Silhouette | Davies–Bouldin | Kích thước cụm nhỏ nhất | Cụm đơn lẻ | Giây |
|---|---:|---:|---:|---:|---:|---:|
| ward | 64 | 0.1123 | 2.6190 | 26 | 0 | 0.97 |
| kmeans | 64 | 0.1583 | 2.5082 | 24 | 0 | 0.80 |
| ward | 256 | 0.1142 | 2.3358 | 4 | 0 | 0.73 |
| kmeans | 256 | 0.1183 | 2.4554 | 3 | 0 | 2.51 |
| ward | 1024 | 0.1699 | 1.6063 | 1 | 14 | 1.74 |
| kmeans | 1024 | 0.1108 | 1.6721 | 1 | 92 | 9.13 |

## DBSCAN (cosine) trên toàn bộ item

eps `knee` lấy tại điểm gãy của đường k-distance; `qX` là phân vị X của đường đó. Silhouette và Davies–Bouldin chỉ tính trên điểm không nhiễu (— khi dưới 2 cụm). Silhouette tính trên mẫu tối đa 5000 điểm không nhiễu nên các cụm rất nhỏ có thể không vào mẫu; giá trị giữa các dòng chỉ so sánh được một cách tương đối.

| PCA | MinPts | eps (quy tắc) | eps | Số cụm | Tỉ lệ nhiễu | Cụm lớn nhất / mọi item | Silhouette | Davies–Bouldin |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 50 | 5 | knee | 0.3147 | 4 | 0.0429 | 0.9557 | -0.1379 | 1.7628 |
| 50 | 5 | q0.50 | 0.2068 | 56 | 0.3468 | 0.4604 | -0.2788 | 1.6066 |
| 50 | 5 | q0.75 | 0.2632 | 18 | 0.1354 | 0.8519 | -0.2893 | 1.7988 |
| 50 | 5 | q0.90 | 0.3184 | 4 | 0.0393 | 0.9592 | -0.0727 | 1.7595 |
| 50 | 10 | knee | 0.3596 | 2 | 0.0179 | 0.9813 | -0.0085 | 1.9522 |
| 50 | 10 | q0.50 | 0.2431 | 11 | 0.2716 | 0.5505 | -0.1042 | 1.7808 |
| 50 | 10 | q0.75 | 0.3007 | 2 | 0.0849 | 0.9140 | -0.0051 | 1.6832 |
| 50 | 10 | q0.90 | 0.3558 | 1 | 0.0207 | 0.9793 | — | — |
| 50 | 20 | knee | 0.4070 | 1 | 0.0064 | 0.9936 | — | — |
| 50 | 20 | q0.50 | 0.2791 | 3 | 0.2113 | 0.7403 | -0.0340 | 1.8615 |
| 50 | 20 | q0.75 | 0.3397 | 2 | 0.0516 | 0.9474 | 0.0025 | 1.8041 |
| 50 | 20 | q0.90 | 0.3949 | 1 | 0.0091 | 0.9909 | — | — |
| 20 | 5 | knee | 0.1821 | 2 | 0.0362 | 0.9636 | -0.0154 | 1.3452 |
| 20 | 5 | q0.50 | 0.1007 | 56 | 0.3472 | 0.3783 | -0.2942 | 1.4544 |
| 20 | 5 | q0.75 | 0.1380 | 15 | 0.1312 | 0.8122 | -0.3475 | 1.6576 |
| 20 | 5 | q0.90 | 0.1783 | 3 | 0.0413 | 0.9563 | -0.1490 | 1.8672 |
| 20 | 10 | knee | 0.2121 | 1 | 0.0186 | 0.9814 | — | — |
| 20 | 10 | q0.50 | 0.1225 | 6 | 0.2879 | 0.5406 | -0.0242 | 1.5226 |
| 20 | 10 | q0.75 | 0.1645 | 5 | 0.0883 | 0.9084 | -0.2278 | 1.8089 |
| 20 | 10 | q0.90 | 0.2085 | 1 | 0.0212 | 0.9788 | — | — |
| 20 | 20 | knee | 0.2427 | 1 | 0.0120 | 0.9880 | — | — |
| 20 | 20 | q0.50 | 0.1464 | 6 | 0.2276 | 0.5733 | 0.0312 | 1.5359 |
| 20 | 20 | q0.75 | 0.1936 | 1 | 0.0571 | 0.9429 | — | — |
| 20 | 20 | q0.90 | 0.2406 | 1 | 0.0135 | 0.9865 | — | — |

## Tóm tắt DBSCAN

- Tại eps ở điểm gãy (6 cấu hình): số cụm 1–4, cụm lớn nhất chiếm 95.6%–99.4% số item, nhiễu 0.6%–4.3%.
- Số cấu hình có ít nhất 2 cụm và cụm lớn nhất dưới 50% số item: 2 / 24 cấu hình đã chạy.
