# Bảng mapping kiến thức học phần

Bảng thống nhất nằm tại mục 3.5g của [docs/project_plan.md](../docs/project_plan.md#g-đối-chiếu-kiến-thức-học-phần). Bản dùng cho báo cáo nằm tại file này. Cột "Mục báo cáo" theo đề cương của [final_report.md](final_report.md); file này là Phụ lục A của báo cáo.

| Phần trong project | Kỹ thuật / công thức | Nội dung học phần | Mục báo cáo |
|---|---|---|---|
| Lọc 5-core, sắp theo thời gian, leave-last-out, cắt chuỗi | Làm sạch, lọc, chia dữ liệu | Chuẩn bị dữ liệu – Tiền xử lý | 2 |
| EDA: độ thưa, long-tail, độ dài chuỗi | Thống kê mô tả, phân phối | Hiểu dữ liệu | 2 |
| PCA giảm chiều embedding | Thu gọn dữ liệu, phương sai giữ lại | Chuẩn bị dữ liệu – Thu gọn dữ liệu | 2, 4.2 |
| Encode mô tả, kNN graph, ItemKNN | Cosine, inner product, k láng giềng gần nhất | Độ đo khoảng cách và tương đồng; kNN | 3, 6.1 |
| OPQ / PQ → semantic ID | k-means trên từng không gian con; codebook = tâm cụm | Phân cụm phẳng (k-means) | 4.1 |
| So sánh k-means / phân cấp / DBSCAN | Silhouette, Davies–Bouldin, ARI | Phân cụm phân cấp, dựa trên mật độ; đánh giá thuật toán phân cụm | 4.2 |
| FP-Growth, AR-Rec | Support, confidence, lift; FP-tree | Phát hiện luật kết hợp – FP-Growth | 5.1 |
| Semantic ID như phân cụm (OPQ, PQ, RQ, ngẫu nhiên) | Purity, NMI, ARI, entropy codebook, tỉ lệ va chạm | Đánh giá phân cụm bằng độ đo ngoài | 4.1, 4.3 |
| PrefixSpan, đối chiếu luật ↔ semantic ID | Mẫu tuần tự, support; so khớp mẫu | Khai phá mẫu tuần tự – mở rộng của luật kết hợp | 5.1, 5.2 |
| Phân tích đồ thị kNN, hubness, Louvain | Bậc vào, độ lệch, modularity | Khai phá đồ thị – phát hiện cộng đồng | 6.1, 6.2, 6.3 |
| Embedding đồng xuất hiện cho ID có tín hiệu hành vi | Item2vec hoặc SVD ma trận đồng xuất hiện | Thu gọn dữ liệu (SVD); biểu diễn đặc trưng | 5.3 |
| Phân khúc người dùng | k-means trên đặc trưng hành vi | Phân cụm ứng dụng – phân khúc khách hàng | 7 |
| Giải mã lan truyền trên đồ thị item–item | Đồ thị kNN, lan truyền theo láng giềng | Recommender systems – mô hình đồ thị cho CF | 3, 6.1, 6.3 |
| Popularity, ItemKNN, SASRec, cold-start, độ đa dạng | Implicit feedback, long-tail, coverage | Recommender systems | 3, 7 |
| Kiến trúc RPG, loss MTP, softmax với τ (đọc hiểu, chạy bằng code gốc) | Công thức 1 của paper | Phân lớp (softmax đa lớp) – nền tảng | 3 |
| Recall@K, NDCG@K, paired t-test | Độ đo xếp hạng; kiểm định thống kê | Đánh giá mô hình; độ đo khách quan (CLO3) | 3, 4.3, 7, Phụ lục B |
