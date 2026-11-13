# KẾ HOẠCH DỰ ÁN

**Khai phá cấu trúc ẩn sau Semantic ID: Phân cụm, mẫu mua kèm và hiện tượng hub trong hệ gợi ý sinh RPG (KDD 2025)**

| Mục | Nội dung |
|---|---|
| Học phần | Khai phá & Phân tích dữ liệu |
| Nhóm | 3 thành viên (A, B, C) |
| Thời gian | Thứ Sáu 6/11 → Thứ Năm 10/12/2026 (5 tuần, khoảng 12–15 giờ/người/tuần) |
| Phiên bản | Bản cập nhật 13/11: trọng tâm là khai phá và phân tích dữ liệu; huấn luyện RPG bằng code gốc |

## Mục lục

1. [Mục tiêu và sản phẩm đầu ra](#1-mục-tiêu-và-sản-phẩm-đầu-ra)
2. [Phân công và nguyên tắc phân vai](#2-phân-công-và-nguyên-tắc-phân-vai)
3. [Lựa chọn kỹ thuật và công cụ](#3-lựa-chọn-kỹ-thuật-và-công-cụ-stack)
4. [Kế hoạch triển khai theo tuần](#4-kế-hoạch-triển-khai-theo-từng-tuần)
5. [Mẫu bảng kết quả](#5-mẫu-bảng-kết-quả)
6. [Cấu trúc repository](#6-cấu-trúc-repository)
7. [Quản lý rủi ro](#7-quản-lý-rủi-ro-và-phương-án-xử-lý)
8. [Quy trình làm việc và quy tắc phát triển](#8-quy-trình-làm-việc-và-quy-tắc-phát-triển)

---

## 1. Mục tiêu và sản phẩm đầu ra

### 1.1. Mục tiêu

Dự án dùng mô hình RPG (Hou và cộng sự, KDD 2025) làm công cụ để khai phá và phân tích cấu trúc của dữ liệu sản phẩm Amazon Beauty.

**Câu hỏi trung tâm:** semantic ID khai phá được cấu trúc gì trong dữ liệu sản phẩm, và cấu trúc đó giúp hay hại việc gợi ý?

RPG được huấn luyện bằng code gốc của tác giả (repo `RPG_KDD2025`) với encoder mở `sentence-t5-base`. Mô hình không được cài đặt lại. Phần tự viết gồm: dữ liệu, tạo semantic ID, đánh giá, khai phá mẫu, phân cụm và phân tích đồ thị.

Câu hỏi trung tâm được chia thành bốn câu hỏi con, mỗi câu gắn với một nhóm kỹ thuật của học phần:

| Mã | Chủ đề | Nội dung |
|---|---|---|
| **RQ1** | Phân cụm | Semantic ID có phải là một phép phân cụm tốt không? So sánh OPQ, PQ, RQ và token ngẫu nhiên bằng các độ đo phân cụm (purity, NMI theo danh mục và thương hiệu, tỉ lệ va chạm ID, mức dùng codebook); đặt cạnh các thuật toán phân cụm kinh điển (k-means, phân cấp, DBSCAN); kiểm tra chất lượng cụm có đi cùng NDCG hay không. |
| **RQ2** | Khai phá mẫu | Mẫu mua kèm (luật kết hợp bằng FP-Growth, mẫu tuần tự bằng PrefixSpan) có được phản ánh trong semantic ID không? Đưa tín hiệu hành vi mua kèm vào semantic ID có cải thiện gợi ý không (nâng cấp chính). |
| **RQ3** | Đồ thị | Đồ thị kNN mà RPG dùng để giải mã có hiện tượng hub không? Sản phẩm hub có phải toàn sản phẩm phổ biến và có bị gợi ý quá mức không? Cộng đồng trên đồ thị (Louvain) có trùng với danh mục không? |
| **RQ4** | Đánh giá và phân khúc | Ngoài Recall/NDCG, các mô hình phủ được bao nhiêu sản phẩm, thiên lệch độ phổ biến ra sao, tốt và kém với nhóm sản phẩm (cold-start) và nhóm người dùng nào? |

### 1.2. Sản phẩm cuối (Deliverables)

| Sản phẩm | Nội dung |
|---|---|
| **Source code** | Repository GitHub tái chạy được từ đầu. Code gốc `RPG_KDD2025` nằm nguyên trong `third_party/` (git submodule); chỉ thêm adapter để nạp semantic ID hoặc embedding do nhóm tạo. Phần tự viết: pipeline dữ liệu, tokenizer (OPQ, PQ, RQ, ngẫu nhiên, ID có tín hiệu hành vi), đánh giá, khai phá mẫu, phân cụm, phân tích đồ thị. README ghi rõ phần của tác giả và phần tự viết. |
| **Bảng kết quả nền** | Recall@5/10, NDCG@5/10 của RPG (code gốc) và các baseline trên cùng cách chia dữ liệu, mean ± std qua 3 seed; cột đối chiếu với paper (Table 4), cột chênh lệch; paired t-test. Bảng là nền so sánh, không phải trọng tâm. |
| **Bảng biến thể semantic ID (RQ1)** | RPG chạy với ID từ OPQ, PQ, RQ và token ngẫu nhiên; mỗi dòng kèm độ đo phân cụm (purity, NMI, tỉ lệ va chạm, entropy codebook) để vẽ tương quan chất lượng cụm ↔ NDCG@10; thêm dòng không dùng đồ thị giải mã. |
| **Module khai phá mẫu (RQ2)** | Tập mục phổ biến và luật kết hợp (support, confidence, lift) bằng FP-Growth; mẫu tuần tự bằng PrefixSpan; recommender dựa trên luật (AR-Rec) đánh giá cùng protocol với RPG; tỉ lệ cặp sản phẩm trong luật mạnh có semantic ID gần nhau so với cặp ngẫu nhiên. |
| **Module phân cụm (RQ1)** | So sánh k-means, phân cụm phân cấp và DBSCAN trên embedding sản phẩm (sau PCA), đánh giá bằng silhouette và Davies–Bouldin; đối chiếu cấu trúc cụm với semantic ID bằng ARI, purity và NMI. |
| **Phân tích đồ thị và đánh giá (RQ3, RQ4)** | Notebook EDA và báo cáo chất lượng dữ liệu; phân bố bậc vào, mức độ hub, hub ↔ gợi ý quá mức, cộng đồng Louvain; coverage, tỉ lệ long-tail, cold-start theo nhóm tần suất, NDCG theo phân khúc người dùng. |
| **Báo cáo tổng kết** | Tổ chức theo quy trình khai phá dữ liệu: câu hỏi → dữ liệu và tiền xử lý → nền tảng RPG → RQ1 → RQ2 (kèm nâng cấp chính) → RQ3 → RQ4 → hạn chế. Mỗi mục kỹ thuật ghi rõ công thức và bài giảng tương ứng trong học phần. |
| **Slide thuyết trình** | Dữ liệu và câu hỏi → RPG trong 3 phút → semantic ID nhìn như phân cụm → mẫu mua kèm và nâng cấp → hub trên đồ thị → ai được lợi, ai bị bỏ rơi → bài học. |

### 1.3. Project Charter – KPI và Definition of Done

Project Charter được chốt lần đầu tại Tuần 1 và cập nhật ngày 13/11: do được phép dùng code gốc, trọng tâm chuyển từ tái tạo mô hình sang khai phá và phân tích dữ liệu.

Bản đầy đủ (phạm vi, sản phẩm bàn giao, KPI, Definition of Done, giả định cần xác minh): [project_charter.md](project_charter.md).

**KPI mô hình nền (bắt buộc).** RPG chạy bằng code gốc với `sentence-t5-base`, m = 32 trên Beauty đạt NDCG@10 trong khoảng 0,038–0,045. Mốc tham chiếu là 0,0429 (Table 4 của paper, dòng RPG + sentence-t5-base + ID dài). Không so với 0,0464 ở Table 2 vì bảng đó dùng `text-embedding-3-large` (API trả phí, không được sử dụng).

**KPI so sánh baseline.** RPG vượt SASRec về NDCG@10 với paired t-test p < 0,05 qua 3 seed (paper dùng cùng mức ý nghĩa). Nếu không vượt, báo cáo trung thực kèm phân tích nguyên nhân.

**KPI khai phá dữ liệu (trọng tâm).** Mỗi câu hỏi RQ1–RQ4 có ít nhất một bảng hoặc hình và một câu kết luận kèm con số. Cụ thể:

1. FP-Growth và PrefixSpan chạy được, AR-Rec có trong bảng kết quả chung, có số đo độ khớp luật ↔ semantic ID.
2. Ít nhất 3 thuật toán phân cụm và ít nhất 3 cách tạo semantic ID được so bằng cùng độ đo, có hình tương quan chất lượng cụm ↔ NDCG.
3. Có số đo hubness và kiểm định hub ↔ gợi ý quá mức.
4. Có coverage, long-tail và cold-start cho mọi mô hình.

**Quyết định dùng code gốc (chốt 13/11).** Mọi lần huấn luyện RPG chạy bằng code gốc; không tự viết mô hình, loss hay vòng train, nhưng phải hiểu và trình bày được kiến trúc. Việc kỹ thuật duy nhất phía mô hình là viết adapter nạp semantic ID hoặc embedding tự tạo vào code gốc (hạn 18/11). Nếu không nạp được ID tự tạo: thay file embedding đầu vào để code gốc tự chạy OPQ (đủ cho nâng cấp chính); các biến thể PQ, RQ, ngẫu nhiên chỉ được đánh giá bằng độ đo cụm, không train.

**Ngân sách tính toán.** Mỗi lần train RPG trên Beauty ước 3–4 giờ trên GPU T4/P100 của Kaggle (paper: < 2 giờ trên RTX 3090). Kế hoạch khoảng 11 lần train (cấu hình chính 3 seed, 3 biến thể RQ1 × 1 seed, nâng cấp chính 3 seed, dự phòng 2), tương đương 35–45 giờ GPU, trải trên 3 tuần và 2–3 tài khoản Kaggle/Colab. Thành viên A theo dõi `configs/gpu_budget.md`; khi vượt 80% ngân sách, áp dụng danh sách cắt giảm.

**Danh sách cắt giảm (theo thứ tự):**

1. Encoder mới – chỉ còn là phương án dự phòng.
2. Biến thể RQ – chỉ giữ OPQ, PQ và ngẫu nhiên.
3. Phát hiện cộng đồng Louvain.
4. Giảm seed của nâng cấp chính từ 3 xuống 2 (cấu hình chính luôn giữ 3 seed).

Không bao giờ cắt: EDA, RQ1 đánh giá cụm, RQ2 khai phá mẫu, báo cáo.

**Scope tối thiểu bắt buộc (MVP).** Dữ liệu khớp Table 1 + RPG chạy bằng code gốc + baseline Popularity/ItemKNN/SASRec/AR-Rec + RQ1 (≥ 3 thuật toán phân cụm, ≥ 3 cách tạo ID, độ đo cụm) + RQ2 (FP-Growth, PrefixSpan, đối chiếu luật ↔ token) + RQ3 (phân bố bậc, hubness) + RQ4 (coverage, long-tail, cold-start) + báo cáo + slide.

**Mở rộng.** Tương quan cụm ↔ NDCG đủ 4 biến thể, nâng cấp chính, Louvain, phân khúc người dùng, đồ thị giảm hub.

**Nguyên tắc trung thực.** Kết quả âm (ID có tín hiệu hành vi không cải thiện, chất lượng cụm không đi cùng NDCG, FP-Growth yếu, RPG lệch mốc paper) được viết thành phát hiện kèm lý giải, không bị bỏ khỏi báo cáo.

---

## 2. Phân công và nguyên tắc phân vai

### 2.1. Phân công trách nhiệm

| TV | Vai trò chính | Trách nhiệm chi tiết |
|---|---|---|
| **A** | Semantic ID & Mô hình (RQ1, nâng cấp chính) | Chạy RPG bằng code gốc (cấu hình chính, 3 seed) và viết adapter nạp semantic ID/embedding tự tạo; viết tokenizer OPQ, PQ, RQ và token ngẫu nhiên; đánh giá semantic ID như một phép phân cụm (purity, NMI theo danh mục và thương hiệu, tỉ lệ va chạm, entropy codebook); train các biến thể và vẽ tương quan chất lượng cụm ↔ NDCG; thực hiện nâng cấp chính (ID có tín hiệu hành vi) cùng C; theo dõi ngân sách GPU; viết phần Nền tảng, RQ1 và Nâng cấp chính. |
| **B** | Dữ liệu, Đánh giá & Đồ thị (RQ3, RQ4) | Pipeline dữ liệu (5-core, sắp theo thời gian, leave-last-out) khớp Table 1; EDA và báo cáo chất lượng dữ liệu; module đánh giá Recall/NDCG, coverage, long-tail, độ đa dạng; baseline Popularity/ItemKNN/SASRec bằng RecBole; dựng đồ thị kNN như paper và phân tích: phân bố bậc vào, hubness, hub ↔ gợi ý quá mức, cộng đồng Louvain; cold-start và phân khúc người dùng; paired t-test; nâng cấp phụ (đồ thị giảm hub); viết phần Dữ liệu, RQ3 và RQ4. |
| **C** | Khai phá mẫu, Phân cụm & Tổng hợp (RQ1, RQ2) | Module khai phá mẫu: FP-Growth, luật A→B với support/confidence/lift, AR-Rec, PrefixSpan cho mẫu tuần tự, đối chiếu luật ↔ semantic ID; dựng embedding đồng xuất hiện (item2vec hoặc SVD) cho nâng cấp chính; module phân cụm: PCA, k-means, phân cụm phân cấp, DBSCAN, silhouette/Davies–Bouldin, ARI với semantic ID; bảng mapping kiến thức học phần ↔ mục báo cáo; hợp nhất báo cáo và slide, thống nhất văn phong; Integration Owner mặc định; chuẩn bị câu hỏi phản biện. |

### 2.2. Quy định phối hợp chung

- Cả 3 thành viên review code chéo qua Pull Request: A review B, B review C, C review A (luân phiên theo tuần nếu lệch tải).
- Mỗi thành viên tự soạn nội dung báo cáo và slide cho phạm vi phụ trách; thành viên C hợp nhất, thống nhất văn phong và ký hiệu toán.
- Mỗi thành viên phải vẽ lại được luồng dữ liệu của hai phần còn lại sau buổi họp tuần (kiểm tra ở Tuần 2 và Tuần 5). Đây là yêu cầu của giảng viên: hiểu nội dung công việc quan trọng hơn con số.
- Module của C, phần đồ thị và đánh giá của B được thiết kế để chạy độc lập với tiến độ huấn luyện: chỉ cần dữ liệu đã chia, embedding và semantic ID (có từ Tuần 1). Phân tích đồ thị dùng checkpoint RPG khi đã có; trước đó dùng đồ thị dựng từ embedding của semantic ID.

---

## 3. Lựa chọn kỹ thuật và công cụ (Stack)

### 3.1. Dữ liệu

**Amazon Reviews 2014 – Beauty**: sau lọc 5-core và tiền xử lý theo paper gồm 22.363 user, 12.101 item, 176.139 tương tác, độ dài chuỗi trung bình 8,87 (Table 1). Nhãn là item kế tiếp trong chuỗi mua theo thời gian; dữ liệu phản hồi ngầm (implicit feedback), không dùng rating.

**Nguồn:** bản review và metadata Amazon 2014 (`reviews_Beauty_5.json.gz`, `meta_Beauty.json.gz`) hoặc dữ liệu đã xử lý trong repo `RPG_KDD2025`. Tải qua `scripts/download_data.py` kèm kiểm tra checksum; dữ liệu thô không được commit lên GitHub.

**Đặc điểm cần xử lý:**

- Trường văn bản làm mô tả sản phẩm: ghép title, brand, category, description theo đúng thứ tự repo gốc; sản phẩm thiếu description dùng title + category, ghi lại số lượng.
- Lọc 5-core lặp đến hội tụ (user và item đều ≥ 5 tương tác); sắp xếp theo timestamp; trùng timestamp thì giữ thứ tự xuất hiện trong file.
- Độ dài chuỗi tối đa 50 (cắt bớt từ đầu); chuỗi đầu vào cho FP-Growth lấy phần train (bỏ 2 item cuối của mỗi user).
- Metadata có link ảnh nhưng không được sử dụng (đã loại hướng CLIP).

### 3.2. Công nghệ sử dụng

**Ngôn ngữ và thư viện:** Python 3.10+, PyTorch, code gốc `RPG_KDD2025` (submodule), RecBole (baseline), FAISS (OPQ, PQ, k-means, kNN graph), sentence-transformers (`sentence-t5-base`; BGE-M3 / E5 nếu dùng phương án dự phòng), scikit-learn (PCA, KMeans, AgglomerativeClustering, DBSCAN, silhouette, Davies–Bouldin, NMI, ARI), mlxtend (`fpgrowth`, `association_rules`), prefixspan, gensim (item2vec), networkx + python-louvain (đồ thị, cộng đồng), scipy (paired t-test, Spearman), pandas, matplotlib.

**Tính toán:** Kaggle (GPU T4/P100, 30 giờ/tuần/tài khoản) và Colab dự phòng; CPU cục bộ cho FP-Growth, phân cụm, đánh giá.

**Quản lý:** GitHub, GitHub Actions (pytest cho `src/`), file `configs/*.yaml` cho từng thí nghiệm, bảng theo dõi thí nghiệm `experiments/log.csv` (tên run, config, seed, giờ GPU, kết quả).

**Môi trường ảo (bắt buộc).** Mỗi thành viên làm việc trong venv/conda riêng, cài đúng phiên bản pin trong `requirements.txt`; trên Kaggle chạy `pip install -r requirements.txt` ở cell đầu notebook.

```bash
python -m venv venv
# Windows: venv/Scripts/activate      # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Thư mục môi trường ảo không được commit (đã có trong `.gitignore`). Khi thêm hoặc nâng cấp thư viện, cập nhật phiên bản cụ thể trong `requirements.txt` và thông báo cho cả nhóm.

### 3.3. Danh sách mô hình so sánh

| Mô hình | Mô tả | Kiến thức học phần |
|---|---|---|
| **Popularity** | Gợi ý item phổ biến nhất trong tập train (baseline tối thiểu). | – |
| **ItemKNN** | Item-based CF, độ tương đồng item–item cosine trên ma trận user–item nhị phân; chạy bằng RecBole. | Neighborhood-based CF |
| **AR-Rec** | FP-Growth trên chuỗi train → luật A→B; với lịch sử user, chọn các luật có tiền đề ⊆ lịch sử, xếp hạng hậu quả theo confidence × lift, bổ sung Popularity khi thiếu. | Luật kết hợp, FP-Growth |
| **kNN nội dung** (tùy chọn) | Gợi ý k item gần nhất (cosine trên embedding `sentence-t5-base`) với item cuối của user. | kNN, độ đo tương đồng |
| **SASRec** | Transformer tuần tự dùng item ID, chạy bằng RecBole với cùng dữ liệu và cách chia (baseline mạnh nhất không dùng semantic ID). | – |
| **RPG (code gốc)** | Semantic ID OPQ m = 32, codebook 256, `sentence-t5-base`; cấu hình chính của paper, chạy bằng `main.py` của repo `RPG_KDD2025`. | – |
| **RPG + semantic ID có tín hiệu hành vi** (nâng cấp chính, RQ2) | Ghép embedding văn bản với embedding đồng xuất hiện (item2vec hoặc SVD trên ma trận đồng xuất hiện của phần train) theo trọng số α ∈ {0,25; 0,5} rồi mới chạy OPQ; mô hình giữ nguyên, nạp qua adapter. | – |
| **RPG + đồ thị giảm hub** (nâng cấp phụ, RQ3) | Dựng đồ thị giải mã bằng mutual kNN hoặc giới hạn bậc vào; không cần train lại. | – |
| **RPG + encoder mới** (dự phòng) | Thay `sentence-t5-base` bằng BGE-M3 hoặc E5-base; chỉ chạy khi còn ngân sách GPU. | – |

### 3.4. Chỉ số đánh giá (Evaluation Metrics)

| Nhóm | Chỉ số |
|---|---|
| **Gợi ý** | Recall@5, Recall@10, NDCG@5, NDCG@10; xếp hạng toàn bộ item (full ranking), loại item đã xem khỏi danh sách ứng viên. Báo cáo mean ± std qua 3 seed cho cấu hình chính. |
| **So sánh cặp mô hình** | Paired t-test trên NDCG@10 theo user (hoặc theo seed), mức ý nghĩa p < 0,05. |
| **Độ phổ biến và đa dạng** | Coverage (tỉ lệ item từng xuất hiện trong top-10); tỉ lệ item long-tail (ngoài 20% phổ biến nhất) trong top-10; NDCG@10 theo nhóm tần suất item [0,5], [6,10], [11,15], [16,20] (cold-start). |
| **Luật kết hợp** | Support, confidence, lift của luật; số tập phổ biến theo min-support; Recall/NDCG@10 của AR-Rec. |
| **Phân cụm** | Silhouette (mẫu 5.000 item, khoảng cách cosine), Davies–Bouldin; với DBSCAN thêm số cụm và tỉ lệ nhiễu; độ nhất quán cụm ↔ semantic ID: số token trùng trung bình của cặp item cùng cụm so với cặp khác cụm, và Adjusted Rand Index giữa phân hoạch k-means và phân hoạch theo token đầu tiên của OPQ. |
| **Semantic ID như phân cụm** | Purity và NMI giữa phân hoạch theo token (từng vị trí và toàn ID) với danh mục lá và thương hiệu; tỉ lệ va chạm (item trùng ID hoàn toàn); entropy sử dụng từng codebook; hệ số Spearman giữa độ đo cụm và NDCG@10 qua các biến thể. |
| **Mẫu tuần tự** | Số mẫu PrefixSpan theo min-support, độ dài mẫu, tỉ lệ cặp có hướng (A→B mạnh hơn B→A); so với luật không thứ tự. |
| **Đồ thị** | Phân bố bậc vào; độ lệch (skewness) của bậc vào làm số đo hubness; tương quan Spearman giữa bậc vào và số lần xuất hiện trong top-10; modularity và NMI giữa cộng đồng Louvain và danh mục. |
| **Hiệu năng** (tùy chọn) | Thời gian inference và bộ nhớ GPU khi tăng số item (thêm item giả), so RPG với SASRec như Figure 3 của paper; chỉ thực hiện nếu còn thời gian. |

### 3.5. Quy ước kỹ thuật chính

#### a) Chia dữ liệu và đánh giá

- Leave-last-out: item cuối của mỗi user làm test, item kế cuối làm validation, phần còn lại là train; lưu file chỉ số dùng chung cho mọi mô hình và module của C.
- Chọn checkpoint và siêu tham số trên validation; test chỉ chạy cho checkpoint đã chọn. Early stopping theo NDCG@10 trên validation, kiên nhẫn 20 epoch.
- Mọi mô hình (RecBole, RPG, AR-Rec) dùng chung hàm đánh giá trong `src/evaluate.py`, có unit test với ví dụ tính tay.

#### b) Semantic ID

- Encoder `sentence-t5-base` (768 chiều) trên mô tả sản phẩm; OPQ của FAISS với m = 32 đoạn con, mỗi đoạn codebook 256 → mỗi item là 32 token, tổng từ vựng 32 × 256 = 8.192. OPQ chỉ fit trên embedding item (không dùng nhãn) nên không có rò rỉ.
- Lưu ánh xạ item → 32 token vào `data/processed/semantic_ids.npy`; ghi số item trùng ID hoàn toàn và mức sử dụng từng codebook.
- Biến thể semantic ID (RQ1): (i) token ngẫu nhiên; (ii) PQ thường (k-means độc lập trên từng đoạn con bằng sklearn, không xoay); (iii) RQ (k-means trên phần dư qua nhiều tầng, cùng m = 32 và codebook 256). Mục đích là tách vai trò của ngữ nghĩa, của phép xoay OPQ và của cấu trúc phân tầng. Mỗi biến thể được đo độ đo cụm trước khi train.

#### c) Huấn luyện và giải mã RPG (code gốc)

- Huấn luyện bằng `main.py` của `RPG_KDD2025` với siêu tham số theo paper (d = 448, FFN 1024, 4 head, τ = 0,03, batch 256, lr 0,01, early stopping 20 epoch), seed {42, 43, 44}; không sửa kiến trúc hay loss. Kiến trúc vẫn phải được hiểu và trình bày (embedding theo codebook, mean pooling, Transformer 2 lớp, đầu MLP từng vị trí, loss MTP).
- Adapter (`adapters/`): nạp file `semantic_ids.npy` hoặc embedding tự tạo vào code gốc thay cho bước tạo ID mặc định. Kiểm tra bằng cách nạp lại chính ID OPQ của code gốc và xác nhận kết quả không đổi.
- Giải mã: kết quả chính dùng giải mã đồ thị của code gốc (k, q theo paper). B dựng lại đồ thị kNN (FAISS, k = 100) từ embedding token của checkpoint để phân tích RQ3 và thử đồ thị giảm hub; đồ thị tự dựng cộng giải mã tự viết phải cho NDCG gần code gốc thì mới được dùng để phân tích.

#### d) Khai phá mẫu mua kèm (FP-Growth, PrefixSpan)

- Giao dịch = tập item (không thứ tự) trong phần train của mỗi user → 22.363 giao dịch, trung bình khoảng 7 item. Chuỗi ngắn nên min-support phải thấp: thử `min_support` ∈ {0,0005; 0,001; 0,002} (≈ 11, 22, 45 giao dịch), `min_confidence` 0,1; ghi số tập phổ biến và số luật ở mỗi mức, kèm thời gian chạy.
- Dùng `mlxtend.frequent_patterns.fpgrowth` và `association_rules`; dữ liệu đưa vào dạng ma trận thưa one-hot (sparse DataFrame) để không tràn bộ nhớ với 12.101 cột.
- Phân tích: 20 luật lift cao nhất kèm tên sản phẩm (cùng thương hiệu / cùng danh mục hay không), phân bố support; tỉ lệ cặp (A, B) trong 200 luật lift cao nhất có nhiều token chung so với cặp ngẫu nhiên. Đây là câu trả lời cho RQ2 và là căn cứ chọn nâng cấp chính.
- PrefixSpan trên chuỗi train có thứ tự (min-support tương đương FP-Growth, độ dài mẫu ≤ 3): mẫu tuần tự A → B; so với luật không thứ tự để xác định thứ tự mua có thêm thông tin hay không. Gộp item lên cấp danh mục khi luật cấp item quá thưa.
- AR-Rec đánh giá đúng protocol leave-last-out như mọi mô hình khác; kỳ vọng thấp hơn SASRec, cần giải thích nguyên nhân (độ thưa, không dùng thứ tự).

#### e) Phân cụm và semantic ID như phân cụm (RQ1)

- Đầu vào: embedding `sentence-t5-base` 768 chiều của 12.101 item, chuẩn hóa L2. Giảm chiều bằng PCA xuống 50 chiều (ghi tỉ lệ phương sai giữ lại) để DBSCAN và phân cụm phân cấp hoạt động được; k-means chạy cả trên 768 và 50 chiều để đối chiếu.
- k-means: k ∈ {64, 256, 1024}, `n_init` = 5. Phân cụm phân cấp: liên kết Ward trên mẫu 5.000 item, cắt cây ở cùng số cụm. DBSCAN: khoảng cách cosine, eps chọn bằng đồ thị k-distance, `min_samples` ∈ {5, 10}.
- Đánh giá: silhouette và Davies–Bouldin trên cùng mẫu; đối chiếu với semantic ID theo mục 3.4; vẽ 2D (PCA 2 thành phần) tô màu theo cụm và theo token đầu của OPQ.
- Đánh giá semantic ID như phân cụm: với từng biến thể (OPQ, PQ, RQ, ngẫu nhiên), tính purity và NMI theo danh mục lá và thương hiệu, tỉ lệ va chạm, entropy codebook; ghép với NDCG@10 của biến thể đó để vẽ hình tương quan và tính hệ số Spearman.
- Mục tiêu trả lời (RQ1): semantic ID của OPQ có phải là một phân cụm "tốt" theo nghĩa của học phần không; thuật toán phân cụm nào phù hợp với embedding văn bản; chất lượng cụm có dự báo được chất lượng gợi ý không.

#### f) Phân tích đồ thị kNN (RQ3)

- Đồ thị: đỉnh là item, mỗi đỉnh nối tới k = 100 láng giềng theo độ tương đồng embedding token (như paper); phân tích bậc vào trên đồ thị có hướng, phát hiện cộng đồng trên bản vô hướng.
- Hubness: phân bố bậc vào, độ lệch (skewness), danh sách 20 hub; hub có phải sản phẩm phổ biến không (tương quan với số tương tác train); hub có bị gợi ý quá mức không (tương quan với tần suất xuất hiện trong top-10 của RPG).
- Cộng đồng: Louvain trên đồ thị vô hướng; modularity; NMI giữa cộng đồng và danh mục; so với cụm k-means của C.
- Nâng cấp phụ: đồ thị mutual kNN hoặc giới hạn bậc vào; đo lại NDCG, coverage, tỉ lệ long-tail và hubness; không cần train lại.

#### g) Đối chiếu kiến thức học phần

Mỗi mục kỹ thuật trong báo cáo mở đầu bằng một câu nêu kỹ thuật và bài giảng tương ứng. Bảng mapping thống nhất:

| Phần trong project | Kỹ thuật / công thức | Nội dung học phần |
|---|---|---|
| Lọc 5-core, sắp theo thời gian, leave-last-out, cắt chuỗi | Làm sạch, lọc, chia dữ liệu | Chuẩn bị dữ liệu – Tiền xử lý |
| EDA: độ thưa, long-tail, độ dài chuỗi | Thống kê mô tả, phân phối | Hiểu dữ liệu |
| PCA giảm chiều embedding | Thu gọn dữ liệu, phương sai giữ lại | Chuẩn bị dữ liệu – Thu gọn dữ liệu |
| Encode mô tả, kNN graph, ItemKNN | Cosine, inner product, k láng giềng gần nhất | Độ đo khoảng cách và tương đồng; kNN |
| OPQ / PQ → semantic ID | k-means trên từng không gian con; codebook = tâm cụm | Phân cụm phẳng (k-means) |
| So sánh k-means / phân cấp / DBSCAN | Silhouette, Davies–Bouldin, ARI | Phân cụm phân cấp, dựa trên mật độ; đánh giá thuật toán phân cụm |
| FP-Growth, AR-Rec | Support, confidence, lift; FP-tree | Phát hiện luật kết hợp – FP-Growth |
| Semantic ID như phân cụm (OPQ, PQ, RQ, ngẫu nhiên) | Purity, NMI, ARI, entropy codebook, tỉ lệ va chạm | Đánh giá phân cụm bằng độ đo ngoài |
| PrefixSpan, đối chiếu luật ↔ semantic ID | Mẫu tuần tự, support; so khớp mẫu | Khai phá mẫu tuần tự – mở rộng của luật kết hợp |
| Phân tích đồ thị kNN, hubness, Louvain | Bậc vào, độ lệch, modularity | Khai phá đồ thị – phát hiện cộng đồng |
| Embedding đồng xuất hiện cho ID có tín hiệu hành vi | Item2vec hoặc SVD ma trận đồng xuất hiện | Thu gọn dữ liệu (SVD); biểu diễn đặc trưng |
| Phân khúc người dùng | k-means trên đặc trưng hành vi | Phân cụm ứng dụng – phân khúc khách hàng |
| Giải mã lan truyền trên đồ thị item–item | Đồ thị kNN, lan truyền theo láng giềng | Recommender systems – mô hình đồ thị cho CF |
| Popularity, ItemKNN, SASRec, cold-start, độ đa dạng | Implicit feedback, long-tail, coverage | Recommender systems |
| Kiến trúc RPG, loss MTP, softmax với τ (đọc hiểu, chạy bằng code gốc) | Công thức 1 của paper | Phân lớp (softmax đa lớp) – nền tảng |
| Recall@K, NDCG@K, paired t-test | Độ đo xếp hạng; kiểm định thống kê | Đánh giá mô hình; độ đo khách quan (CLO3) |

---

## 4. Kế hoạch triển khai theo từng tuần

Lịch tính từ thứ Sáu 6/11. Tuần 1 giữ nguyên như đã thực hiện; bản cập nhật áp dụng từ Tuần 2 (13/11). Mỗi tuần có họp 30 phút vào Chủ nhật và một "Cổng" vào thứ Năm. Khối lượng ước tính: A ~13 giờ, B ~14 giờ, C ~13 giờ mỗi tuần. Hạn chót 10/12 giữ nguyên.

| Tuần | Thời gian | Chủ đề |
|---|---|---|
| 1 | T6 6/11 → T5 12/11 | Nền tảng và dữ liệu |
| 2 | T6 13/11 → T5 19/11 | RPG bằng code gốc; khai phá mẫu; semantic ID như phân cụm |
| 3 | T6 20/11 → T5 26/11 | Nối cấu trúc dữ liệu với chất lượng gợi ý |
| 4 | T6 27/11 → T5 3/12 | Nâng cấp xuất phát từ dữ liệu |
| 5 | T6 4/12 → T5 10/12 | Đóng gói và trình bày |

### 4.1. Tuần 1 (T6 6/11 → T5 12/11): Nền tảng và dữ liệu

**Mục tiêu:** hiểu bài báo, chạy được code gốc, có dữ liệu sạch khớp Table 1, semantic ID, baseline và dữ liệu đầu vào cho hai module khai phá.

| Ngày | Thành viên A (Semantic ID & Mô hình) | Thành viên B (Dữ liệu, Đánh giá & Đồ thị) | Thành viên C (Khai phá mẫu & Phân cụm) |
|---|---|---|---|
| T6 6/11 | Cùng nhóm hỏi giảng viên: được dùng code gốc không; chấm theo kỹ thuật trong đề cương hay chất lượng chung. Đọc RPG mục 1–2, hình 1; tóm tắt nửa trang. | Cùng hỏi giảng viên. Đọc SASRec mục 3 và phần xử lý dữ liệu của repo `RPG_KDD2025` (phiên bản Amazon, 5-core, trường văn bản). | Cùng hỏi giảng viên. Đọc lại bài giảng luật kết hợp (FP-Growth) và phân cụm (k-means, phân cấp, DBSCAN, đánh giá cụm); viết Project Charter nháp (mục 1.3). |
| T7 7/11 | Clone repo `RPG_KDD2025`, cài môi trường trên Kaggle, chạy thử Beauty vài epoch. Mục tiêu: pipeline chạy thông. | Viết `scripts/download_data.py` kèm checksum; dựng cấu trúc repo, `requirements.txt`, CI tối thiểu. | Cài mlxtend, scikit-learn; viết `src/evaluate.py` bản đầu (Recall@K, NDCG@K) kèm unit test tính tay – dùng chung cho cả nhóm. |
| CN 8/11 | Họp tuần (30′): đọc chung TIGER và Product Quantization; thống nhất cấu trúc repo và file chỉ số chia dữ liệu. | Họp tuần. | Họp tuần; chốt Project Charter v1. |
| T2 9/11 | Thử OPQ của FAISS trên vector ngẫu nhiên: hiểu codebook, số đoạn con, cách lấy mã. | Pipeline dữ liệu: 5-core, sắp theo thời gian, leave-last-out; khớp Table 1 (22.363 / 12.101 / 176.139). Lưu file chỉ số chia. | Từ file chỉ số chia, xây tập giao dịch (tập item train của mỗi user) dạng ma trận thưa; chạy FP-Growth thử với min_support 0,002, ghi số tập phổ biến và thời gian. |
| T3 10/11 | Encode mô tả bằng `sentence-t5-base`; chạy OPQ (m = 32, codebook 256); lưu ánh xạ item → 32 token. | Notebook EDA: độ dài chuỗi, long-tail, độ thưa, ví dụ mô tả sản phẩm. | Nhận embedding từ A; PCA 768 → 50 (ghi phương sai giữ lại); k-means k = 256 chạy thử; silhouette trên mẫu 5.000. |
| T4 11/11 | Baseline Popularity, ItemKNN và SASRec bằng RecBole, cùng dữ liệu, cùng cách chia, cùng metric (chạy đêm). | Phân tích chất lượng semantic ID: số item trùng ID, mức dùng từng codebook, soi 10 cặp item có ID gần nhau. | Quét min_support ∈ {0,0005; 0,001; 0,002}; sinh luật với min_confidence 0,1; bảng số luật và 20 luật lift cao nhất kèm tên sản phẩm. |
| T5 12/11 | **Cổng 1:** số liệu khớp Table 1, ít item trùng ID, SASRec có NDCG@10 quanh 0,03; baseline đã log. | **Cổng 1** (cùng A). | **Cổng 1:** FP-Growth chạy được ở cả 3 mức support; k-means có silhouette; mapping kiến thức (mục 3.5g) điền xong. |

**Deliverables Tuần 1:** Project Charter, script tải dữ liệu, file chỉ số chia dữ liệu, notebook EDA, `semantic_ids.npy` và báo cáo chất lượng ID, kết quả baseline, tập luật FP-Growth sơ bộ, kết quả k-means sơ bộ, `src/evaluate.py` có unit test.

**Dự phòng Tuần 1:**

- Dữ liệu chưa khớp Table 1 → dùng dữ liệu đã xử lý của repo gốc, ghi chú trong báo cáo.
- Code gốc không chạy trên Kaggle → thử Colab; vẫn lỗi thì ghi lại và A bắt đầu tự viết từ 11/11.
- Giảng viên chưa trả lời → theo phương án A, quyết định lại ngày 20/11.
- FP-Growth tràn bộ nhớ → tăng min_support hoặc chỉ giữ 5.000 item phổ biến nhất trong giao dịch (ghi rõ).

### 4.2. Tuần 2 (T6 13/11 → T5 19/11): RPG bằng code gốc; khai phá mẫu; semantic ID như phân cụm

**Mục tiêu:** RPG code gốc có kết quả cấu hình chính và adapter nạp semantic ID hoạt động; RQ1 có bảng độ đo cụm cho 4 cách tạo ID; RQ2 có luật, mẫu tuần tự và AR-Rec; RQ3 có đồ thị và phân bố bậc.

**Mốc quan trọng:** Cổng 2 ngày 17/11 – RPG code gốc đạt NDCG@10 ≈ 0,038–0,045 và adapter nạp lại ID OPQ cho cùng kết quả. Hạn chót adapter: 18/11.

| Ngày | Thành viên A (Semantic ID & Mô hình) | Thành viên B (Dữ liệu, Đánh giá & Đồ thị) | Thành viên C (Khai phá mẫu & Phân cụm) |
|---|---|---|---|
| T6 13/11 | Chạy RPG code gốc, cấu hình chính (`sentence-t5-base`, m = 32), seed 42 (chạy đêm). Đọc mã nguồn `genrec/models/RPG`: tìm chỗ tạo semantic ID và chỗ đọc embedding. | Hoàn thiện module đánh giá: chấm Recall/NDCG trên top-K xuất từ mọi mô hình; thêm coverage, tỉ lệ long-tail, độ đa dạng danh mục trong top-10; unit test. | Viết AR-Rec: khớp tiền đề ⊆ lịch sử user, xếp hạng hậu quả theo confidence × lift, bổ sung Popularity; đánh giá leave-last-out. |
| T7 14/11 | Viết adapter nạp `semantic_ids.npy` hoặc embedding tự tạo; kiểm tra bằng cách nạp lại chính ID của code gốc. | Dựng đồ thị kNN (FAISS, k = 100) từ embedding token của ID OPQ; thử trước với embedding ngẫu nhiên. | Quét min_support/min_confidence cho AR-Rec, chọn trên validation; phân tích nguyên nhân luật thưa (độ dài chuỗi 8,87). |
| CN 15/11 | Đặt chạy seed 43, 44 (chạy đêm). Họp tuần: A trình bày kiến trúc RPG đọc từ code gốc, vẽ lại sơ đồ. | Họp tuần: B trình bày đánh giá và đồ thị. | Họp tuần: C trình bày FP-Growth và AR-Rec. Mỗi thành viên vẽ lại được luồng dữ liệu của hai phần kia. |
| T2 16/11 | Viết tokenizer PQ, RQ (cùng m = 32, codebook 256) và token ngẫu nhiên; lưu 4 file semantic ID. | Phân bố bậc vào, độ lệch (hubness), 20 hub hàng đầu; hub có phải sản phẩm phổ biến không. | PrefixSpan trên chuỗi train có thứ tự; số mẫu theo min-support; so mẫu tuần tự với luật không thứ tự. |
| T3 17/11 | **Cổng 2:** so NDCG@10 với 0,0429; adapter cho cùng kết quả. Đo độ đo cụm cho 4 cách tạo ID: purity, NMI theo danh mục lá và thương hiệu, tỉ lệ va chạm, entropy codebook. | **Cổng 2** (cùng A): kiểm tra metric của code gốc khớp `src/evaluate.py` trên cùng top-K. | Phân cụm phân cấp (Ward, mẫu 5.000) và DBSCAN (cosine, eps từ k-distance) trên PCA-50; bảng silhouette / Davies–Bouldin / số cụm / tỉ lệ nhiễu. |
| T4 18/11 | Đặt train biến thể PQ qua adapter (đêm, seed 42). Bảng RQ1 v1: độ đo cụm của 4 biến thể. | Báo cáo chất lượng dữ liệu: tỉ lệ thiếu description/brand/category, độ dài văn bản, dữ liệu thay đổi thế nào qua 5-core; nháp phần Dữ liệu. | Đối chiếu luật ↔ semantic ID: tỉ lệ cặp (A, B) trong 200 luật lift cao nhất có nhiều token chung so với cặp ngẫu nhiên; ARI giữa k-means và token đầu OPQ. |
| T5 19/11 | Viết nháp phần Nền tảng (RPG trong 1 trang, sơ đồ kiến trúc vẽ lại) và phương pháp RQ1. | Gom bảng kết quả v1: Popularity, ItemKNN, AR-Rec, SASRec, RPG (3 seed) cạnh số paper; paired t-test RPG vs SASRec. | Viết nháp hai mục "Khai phá mẫu mua kèm" (RQ2) và "So sánh thuật toán phân cụm" (RQ1); cập nhật mapping kiến thức. |

**Deliverables Tuần 2:** kết quả RPG code gốc 3 seed; adapter nạp semantic ID (`adapters/`); 4 file semantic ID (OPQ, PQ, RQ, ngẫu nhiên) kèm bảng độ đo cụm; đồ thị kNN và thống kê hubness; AR-Rec và mẫu PrefixSpan; bảng phân cụm v1; bảng kết quả v1; nháp phần Nền tảng, Dữ liệu và hai mục khai phá.

**Dự phòng Tuần 2:**

- RPG code gốc lệch mốc → kiểm tra encoder, siêu tham số, cách loại item đã xem; vẫn lệch thì ghi nhận và đi tiếp, vì các phân tích không phụ thuộc con số tuyệt đối.
- Adapter chưa chạy đến hết 18/11 → thay file embedding đầu vào để code gốc tự chạy OPQ (dùng được cho nâng cấp chính); các biến thể PQ, RQ, ngẫu nhiên chỉ đánh giá độ đo cụm, không train.
- DBSCAN ra một cụm + nhiễu → thử eps khác và PCA-20; vẫn thế thì viết thành nhận xét về giới hạn của phân cụm mật độ trên embedding văn bản.

### 4.3. Tuần 3 (T6 20/11 → T5 26/11): Nối cấu trúc dữ liệu với chất lượng gợi ý

**Mục tiêu:** mỗi câu hỏi RQ1–RQ4 có kết luận sơ bộ kèm con số; có hình tương quan chất lượng cụm ↔ NDCG; chốt nâng cấp chính dựa trên phát hiện của RQ2 và RQ3.

| Ngày | Thành viên A (Semantic ID & Mô hình) | Thành viên B (Dữ liệu, Đánh giá & Đồ thị) | Thành viên C (Khai phá mẫu & Phân cụm) |
|---|---|---|---|
| T6 20/11 | Đặt train biến thể RQ (đêm). Phân tích ID va chạm: sản phẩm nào bị gộp chung, nguyên nhân (mô tả rỗng, cùng dòng sản phẩm). | Ablation không cần train: bỏ đồ thị giải mã (lấy ngẫu nhiên cùng số item). | Luật ở cấp thương hiệu/danh mục (gộp item lên cấp danh mục) để vẫn có kết luận khi luật cấp item thưa. |
| T7 21/11 | Đặt train biến thể token ngẫu nhiên (đêm). Trực quan hóa PCA/UMAP 2D tô màu theo token đầu và theo danh mục. | Hub có bị gợi ý quá mức không: tương quan bậc vào ↔ tần suất trong top-10 của RPG; so với SASRec. | Dựng embedding đồng xuất hiện từ phần train: item2vec (gensim) và SVD ma trận đồng xuất hiện; bàn giao cho A. |
| CN 22/11 | Họp tuần: xem kết quả RQ1, cập nhật danh sách cắt giảm. | Họp tuần. | Họp tuần; nếu đã chậm 2 ngày, đề xuất cắt mục (2) và (3) của danh sách cắt giảm. |
| T2 23/11 | Bảng RQ1 hoàn chỉnh: mỗi biến thể có độ đo cụm và NDCG@10; hình phân tán và hệ số Spearman. | Cold-start: NDCG@10 theo nhóm tần suất item [0,5], [6,10], [11,15], [16,20]; so RPG, SASRec, AR-Rec. | Kiểm tra embedding đồng xuất hiện: láng giềng gần có trùng luật mạnh không; so AR-Rec với ItemKNN. |
| T3 24/11 | Thử nhanh nâng cấp chính (chưa train): ghép embedding đồng xuất hiện với embedding văn bản theo α, chạy OPQ, đo lại độ đo cụm và độ khớp với luật. | Louvain trên đồ thị vô hướng: modularity, NMI cộng đồng ↔ danh mục; coverage và tỉ lệ long-tail cho mọi mô hình. | Phân tích PrefixSpan sâu: thứ tự mua có thêm thông tin không (tỉ lệ mẫu A→B mạnh hơn B→A). |
| T4 25/11 | Viết nháp kết quả RQ1. | Phân khúc người dùng: đặc trưng độ dài lịch sử, độ đa dạng danh mục, mức ưa hàng phổ biến → k-means; NDCG theo phân khúc; paired t-test. | Hoàn thiện hai mục khai phá (bảng, hình, kết luận một câu mỗi mục). |
| T5 26/11 | **Cổng 3:** chốt nâng cấp chính dựa trên RQ2 (token có khớp luật mua kèm không) và kết quả thử nhanh. | **Cổng 3** (cùng nhóm): bảng RQ3, RQ4 v1. Thử nhanh đồ thị giảm hub (mutual kNN) trên checkpoint có sẵn. | **Cổng 3:** RQ2 có kết luận; mục phân cụm ở trạng thái "có thể nộp". |

**Deliverables Tuần 3:** bảng RQ1 đủ biến thể và hình tương quan cụm ↔ NDCG; bảng kết quả chính v2 (có p-value); hình hubness, hub ↔ gợi ý và Louvain; cold-start, coverage, phân khúc người dùng; embedding đồng xuất hiện; hai mục khai phá hoàn chỉnh; quyết định nâng cấp chính.

**Dự phòng Tuần 3:**

- Thiếu GPU → bỏ biến thể RQ, chỉ giữ PQ và ngẫu nhiên.
- Token đã khớp tốt với luật mua kèm (RQ2 cho kết quả dương) → nâng cấp chính chuyển sang đồ thị giảm hub hoặc làm giàu văn bản mô tả.
- Tương quan cụm ↔ NDCG yếu → viết thành phát hiện: độ đo cụm "theo sách" chưa đủ để chọn tokenizer.

### 4.4. Tuần 4 (T6 27/11 → T5 3/12): Nâng cấp xuất phát từ dữ liệu

**Mục tiêu:** nâng cấp chính (semantic ID có tín hiệu hành vi) có kết quả 3 seed; nâng cấp phụ (đồ thị giảm hub) có kết quả không cần train; mọi nâng cấp được đo lại bằng độ đo của RQ1–RQ4 để giải thích nguyên nhân tăng hay không. C bắt đầu hợp nhất báo cáo.

| Ngày | Thành viên A (Semantic ID & Mô hình) | Thành viên B (Dữ liệu, Đánh giá & Đồ thị) | Thành viên C (Khai phá mẫu & Phân cụm) |
|---|---|---|---|
| T6 27/11 | Nâng cấp chính: tạo ID có tín hiệu hành vi với α tốt nhất (từ thử nhanh), train qua adapter, seed 42 (chạy đêm). | Nâng cấp phụ: đồ thị mutual kNN và giới hạn bậc vào; đo NDCG, coverage, long-tail trên checkpoint có sẵn. | Đo lại RQ2 trên ID mới: tỉ lệ cặp trong luật mạnh có nhiều token chung tăng bao nhiêu. Bắt đầu khung báo cáo tổng. |
| T7 28/11 | Đọc kết quả; hỏng thì sửa tối đa một lần; đặt seed 43 (chạy đêm). | Đo lại hubness và tần suất hub trong top-10 sau nâng cấp phụ. | Chạy FP-Growth trên tập "gợi ý đúng của RPG": gợi ý đúng có nằm trong luật kết hợp không (tỉ lệ phủ), trước và sau nâng cấp. |
| CN 29/11 | Đặt seed 44 (chạy đêm). Họp tuần: quyết định có chạy α thứ hai hoặc encoder mới không (chỉ khi còn ngân sách GPU). | Họp tuần. | Họp tuần; rà soát mục nào của báo cáo còn thiếu hình/bảng. |
| T2 30/11 | Đo lại độ đo cụm của ID mới; đặt vào hình tương quan RQ1. | Vẽ biểu đồ RQ3, RQ4 bản cuối: phân bố bậc, hub ↔ gợi ý, cold-start, phân khúc. | Viết phần "Thảo luận": semantic ID nhìn từ góc độ phân cụm; luật kết hợp nhìn từ góc độ gợi ý tuần tự. |
| T3 1/12 | Phân tích nâng cấp chính theo nhóm cold-start và phân khúc người dùng (phối hợp B). | Paired t-test giữa các cấu hình nâng cấp và cấu hình chính. | Hợp nhất báo cáo v1 từ các phần của A và B; thống nhất ký hiệu toán và tên bảng. |
| T4 2/12 | Viết phần Nâng cấp chính (kể cả khi là kết quả âm). | Viết phần RQ3, RQ4 và Nâng cấp phụ. | Đọc chéo các phần nâng cấp; chuẩn bị danh sách câu hỏi phản biện. |
| T5 3/12 | **Cổng 4:** nâng cấp chính có bảng 3 seed, p-value và một câu kết luận. | **Cổng 4** (cùng nhóm). | **Cổng 4:** báo cáo v1 đủ mục. |

**Deliverables Tuần 4:** kết quả nâng cấp chính (3 seed) kèm p-value và phân tích qua độ đo RQ1–RQ4; kết quả đồ thị giảm hub; bộ hình RQ3–RQ4 bản cuối; báo cáo v1 đầy đủ các mục.

**Dự phòng Tuần 4:**

- Nâng cấp chính không cải thiện → viết thành kết quả âm, lý giải bằng độ đo cụm và độ khớp luật.
- Hết GPU → chỉ giữ 2 seed cho nâng cấp chính, giữ nguyên nâng cấp phụ (không cần GPU); A chuyển sang hỗ trợ phân tích và viết.

### 4.5. Tuần 5 (T6 4/12 → T5 10/12): Đóng gói và trình bày

**Mục tiêu:** repo sạch, báo cáo hoàn chỉnh, slide, tổng duyệt và nộp bài.

| Ngày | Thành viên A (Semantic ID & Mô hình) | Thành viên B (Dữ liệu, Đánh giá & Đồ thị) | Thành viên C (Khai phá mẫu & Phân cụm) |
|---|---|---|---|
| T6 4/12 | Chạy nốt thí nghiệm còn thiếu – chỉ những gì báo cáo thật sự cần. | Vẽ lại toàn bộ biểu đồ cho thống nhất: cùng màu, cùng font, có nhãn trục và đơn vị. | Rà soát báo cáo: mỗi mục kỹ thuật có câu dẫn về bài giảng và công thức; bảng mapping ở phụ lục. |
| T7 5/12 | **Cổng:** đóng băng thí nghiệm. Từ đây chỉ viết, không chạy thêm. | Đóng băng thí nghiệm. | Đóng băng thí nghiệm. |
| CN 6/12 | Dọn repo: README, config từng thí nghiệm, script chạy lại bằng một lệnh; ghi rõ phần code gốc (submodule, adapter) và phần tự viết. Họp: đọc chéo báo cáo. | Hoàn thiện phần Dữ liệu, RQ3, RQ4; kiểm tra số trong bảng khớp file log. Họp: đọc chéo. | Hoàn thiện báo cáo v2: câu hỏi, dữ liệu, RQ1–RQ4, nâng cấp, hạn chế, hướng tiếp theo. Họp: đọc chéo. |
| T2 7/12 | Kiểm tra chạy lại dự án từ một bản clone sạch theo README. | Soát lại toàn bộ số liệu và hình trong báo cáo lần cuối. | Hoàn thiện báo cáo cuối; xuất PDF. |
| T3 8/12 | Slide phần Nền tảng, RQ1 và Nâng cấp chính. | Slide phần Dữ liệu, RQ3, RQ4. | Slide phần Mở đầu, RQ2, Phân cụm, Kết luận; ghép bộ slide. |
| T4 9/12 | Tổng duyệt; chuẩn bị câu hỏi phản biện. Mỗi thành viên giải thích được toàn bộ project, kể cả phần của hai thành viên còn lại. | Tổng duyệt. | Tổng duyệt; chủ trì phần hỏi đáp thử. |
| T5 10/12 | **Nộp bài:** báo cáo, link repo, slide. | Nộp bài. | Nộp bài. |

**Deliverables Tuần 5:** báo cáo cuối (PDF), repository hoàn chỉnh chạy lại được từ clone sạch, bộ slide.

**Dự phòng Tuần 5:** ngày 9–10/12 là đệm. Nếu trễ, cắt α thứ hai và encoder mới khỏi báo cáo trước; giữ nguyên EDA, RQ1, RQ2 và phần kết luận.

---

## 5. Mẫu bảng kết quả

Sử dụng thống nhất cho báo cáo. Mọi số đo trên tập test với protocol leave-last-out, full ranking; cột "NDCG@10 paper" lấy từ Table 2/4 của bài báo (ghi rõ bảng nguồn).

| Mô hình | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 (mean ± std) | NDCG@10 paper | Chênh lệch | p-value vs RPG | Kiến thức học phần |
|---|---|---|---|---|---|---|---|---|
| Popularity | _điền_ | _điền_ | _điền_ | _điền_ | – | – | _điền_ | Baseline |
| ItemKNN | _điền_ | _điền_ | _điền_ | _điền_ | – | – | _điền_ | Item-based CF, cosine |
| AR-Rec (FP-Growth) | _điền_ | _điền_ | _điền_ | _điền_ | – | – | _điền_ | RQ2: luật kết hợp |
| kNN nội dung (tùy chọn) | _điền_ | _điền_ | _điền_ | _điền_ | – | – | _điền_ | kNN, tương đồng |
| SASRec | _điền_ | _điền_ | _điền_ | _điền_ | 0,0381 (T2) | _điền_ | _điền_ | Baseline tuần tự |
| RPG (code gốc, ID OPQ) | _điền_ | _điền_ | _điền_ | _điền_ | 0,0429 (T4) | _điền_ | – | Phân cụm (OPQ), đồ thị |
| RPG – ID PQ | _điền_ | _điền_ | _điền_ | _điền_ | – | – | _điền_ | RQ1: k-means từng đoạn con |
| RPG – ID RQ | _điền_ | _điền_ | _điền_ | _điền_ | – | – | _điền_ | RQ1: k-means phân tầng |
| RPG – ID ngẫu nhiên | _điền_ | _điền_ | _điền_ | _điền_ | _điền_ (T3) | _điền_ | _điền_ | RQ1: đối chứng |
| RPG – không đồ thị | _điền_ | _điền_ | _điền_ | _điền_ | _điền_ (T3) | _điền_ | _điền_ | Ablation, đồ thị kNN |
| RPG + đồ thị giảm hub | _điền_ | _điền_ | _điền_ | _điền_ | – | – | _điền_ | RQ3: hubness |
| RPG + ID có tín hiệu hành vi | _điền_ | _điền_ | _điền_ | _điền_ | – | – | _điền_ | RQ2: đồng xuất hiện, SVD |
| RPG + encoder mới (dự phòng) | _điền_ | _điền_ | _điền_ | _điền_ | – | – | _điền_ | Biểu diễn văn bản |

**Bảng phụ 1 – Semantic ID như phân cụm (RQ1).** Biến thể (OPQ, PQ, RQ, ngẫu nhiên, ID có tín hiệu hành vi), purity/NMI theo danh mục và thương hiệu, tỉ lệ va chạm, entropy codebook, NDCG@10; kèm hình tương quan và hệ số Spearman.

**Bảng phụ 2 – So sánh thuật toán phân cụm (RQ1).** Thuật toán, số chiều đầu vào, số cụm, silhouette, Davies–Bouldin, tỉ lệ nhiễu (DBSCAN), ARI với token đầu OPQ, số token trùng trung bình trong/ngoài cụm, thời gian chạy.

**Bảng phụ 3 – Khai phá mẫu (RQ2).** min_support, số tập phổ biến, số luật, số mẫu tuần tự, confidence/lift trung bình, thời gian chạy; 20 luật lift cao nhất kèm tên sản phẩm; tỉ lệ cặp trong luật mạnh có nhiều token chung, trước và sau nâng cấp.

**Bảng phụ 4 – Đồ thị (RQ3).** Bậc vào trung bình, độ lệch, 20 hub, tương quan bậc vào ↔ độ phổ biến và ↔ tần suất trong top-10, modularity, NMI cộng đồng ↔ danh mục; trước và sau đồ thị giảm hub.

**Bảng phụ 5 – Cold-start, đa dạng và phân khúc (RQ4).** NDCG@10 theo nhóm tần suất item và theo phân khúc người dùng; coverage và tỉ lệ long-tail trong top-10 của từng mô hình.

---

## 6. Cấu trúc repository

```text
rpg-semantic-ids/
├── .github/workflows/ci.yml
├── configs/
│   ├── rpg_main.yaml            # code gốc: m=32, sentence-t5-base
│   ├── variants_*.yaml          # pq, rq, random, behavior
│   ├── baselines_recbole.yaml
│   └── gpu_budget.md            # giờ GPU đã dùng / còn lại
├── data/
│   ├── raw/                     # gitignore
│   ├── processed/               # gitignore (sequences, semantic_ids.npy, embeddings)
│   └── splits/                  # commit: chỉ số leave-last-out dùng chung
├── third_party/RPG_KDD2025/     # submodule: code gốc, không sửa
├── adapters/                    # nạp semantic ID / embedding vào code gốc
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_semantic_id_as_clustering.ipynb
│   ├── 03_pattern_mining.ipynb
│   ├── 04_clustering_comparison.ipynb
│   ├── 05_coldstart_segments.ipynb
│   └── 06_graph_hubness.ipynb
├── scripts/
│   ├── download_data.py
│   ├── build_semantic_ids.py    # encode → OPQ / PQ / RQ / random / behavior
│   ├── run_baselines.py         # RecBole: Pop, ItemKNN, SASRec
│   ├── run_rpg.py               # gọi code gốc qua adapter → top-K → đánh giá
│   └── run_all.sh               # chạy lại toàn bộ bằng một lệnh
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # 5-core, sắp theo thời gian, leave-last-out
│   ├── tokenizer.py             # OPQ, PQ, RQ, random, ghép embedding hành vi
│   ├── cluster_metrics.py       # purity, NMI, va chạm, entropy codebook
│   ├── patterns.py              # PrefixSpan, luật ↔ token, item2vec/SVD
│   ├── graph.py                 # kNN graph, giải mã, hubness, Louvain, giảm hub
│   ├── evaluate.py              # Recall@K, NDCG@K, t-test, coverage
│   ├── ar_rec.py                # recommender dựa trên luật kết hợp
│   └── clustering.py            # PCA, k-means, phân cấp, DBSCAN, độ đo
├── tests/
├── experiments/log.csv          # run, config, seed, giờ GPU, kết quả
├── reports/
│   ├── figures/
│   ├── knowledge_mapping.md     # bảng mapping kiến thức học phần
│   ├── decisions.md             # câu trả lời của giảng viên, quyết định của nhóm
│   └── final_report.md
├── docs/project_plan.md         # tài liệu này
├── .gitignore
├── requirements.txt
└── README.md
```

CI (GitHub Actions) chạy pytest cho `src/` và `adapters/` ở mỗi Pull Request. Code gốc nằm nguyên trong `third_party/` (git submodule), không sửa trực tiếp; mọi thay đổi đi qua `adapters/` và được ghi trong README.

---

## 7. Quản lý rủi ro và phương án xử lý

| Rủi ro | Mô tả | Phương án xử lý |
|---|---|---|
| Không nạp được semantic ID tự tạo vào code gốc | Code gốc tự sinh ID từ văn bản; điểm nạp ID có thể khó tìm hoặc gắn chặt với cache. | Đọc `genrec/models/RPG` từ 13/11, hạn 18/11; nếu không được thì thay file embedding đầu vào để code gốc tự chạy OPQ (đủ cho nâng cấp chính), các biến thể PQ, RQ, ngẫu nhiên chỉ đánh giá độ đo cụm. |
| Hết ngân sách GPU | Kaggle giới hạn 30 giờ/tuần/tài khoản; nhiều seed và ablation cộng dồn. | Theo dõi `configs/gpu_budget.md`; dùng 2–3 tài khoản và Colab; áp dụng danh sách cắt giảm theo thứ tự; cấu hình chính luôn giữ 3 seed. |
| So sai mốc paper | So với Table 2 (`text-embedding-3-large`) thay vì Table 4 (`sentence-t5-base`). | Ghi rõ bảng nguồn ở mỗi số paper trong bảng kết quả; C kiểm tra khi hợp nhất. |
| Rò rỉ dữ liệu | Item test/validation lọt vào train, hoặc OPQ/FP-Growth dùng dữ liệu test. | Một file chỉ số chia dùng chung; FP-Growth và ItemKNN chỉ nhìn phần train; unit test kiểm tra item cuối không xuất hiện trong chuỗi train; B chịu trách nhiệm chính. |
| Đánh giá không nhất quán | Mỗi mô hình dùng một cách tính metric (RecBole, code gốc, tự viết). | Mọi mô hình xuất top-K rồi chấm bằng cùng `src/evaluate.py`; kiểm tra chéo với metric của RecBole trên một mô hình. |
| FP-Growth cho luật quá thưa | Chuỗi trung bình 8,87 item, 12.101 item → support rất thấp, ít luật hữu ích. | Quét min_support 3 mức; chấp nhận AR-Rec yếu và phân tích nguyên nhân; bổ sung phân tích luật theo thương hiệu/danh mục để vẫn có kết luận có giá trị. |
| Phân cụm mật độ thất bại | DBSCAN trên embedding cao chiều cho một cụm lớn + nhiễu. | PCA trước, khoảng cách cosine, chọn eps bằng k-distance; nếu vẫn thất bại, ghi thành nhận xét về giới hạn của thuật toán trên dữ liệu này (đúng mục "đánh giá thuật toán phân cụm"). |
| Kết quả trái với paper | Ablation hoặc nâng cấp cho kết luận ngược. | Chạy lại seed khác; nếu vẫn trái, viết thành phát hiện có lý giải, không giấu. |
| Trễ tiến độ | Tuần 2–3 nặng nhất; nếu RPG hoặc adapter chậm, các biến thể RQ1 và nâng cấp bị kẹt. | Module của C và phần đồ thị/đánh giá của B không phụ thuộc mô hình đã train; họp Chủ nhật quyết định cắt giảm ngay khi chậm 2 ngày. |
| Thiếu nhãn để đánh giá cụm | Nhiều sản phẩm thiếu brand hoặc category nên purity/NMI thiếu căn cứ. | Dùng danh mục lá trong đường dẫn category làm nhãn chính; báo purity/NMI trên phần có nhãn và ghi rõ tỉ lệ thiếu. |
| Hiểu không đều giữa thành viên | Mỗi thành viên chỉ nắm phần mình làm, không trả lời được phản biện. | Mỗi họp tuần một thành viên trình bày phần của mình; Tuần 2 và Tuần 5 kiểm tra mỗi thành viên vẽ lại được luồng dữ liệu của hai phần còn lại. |

---

## 8. Quy trình làm việc và quy tắc phát triển

### 8.1. Quy định sử dụng Git

- **Cấu trúc branch:** `<role>/<task>` (ví dụ: `adapter/load-ids`, `data/leave-last-out`, `dm/fpgrowth-rules`, `dm/prefixspan`, `dm/clustering-eval`, `graph/hubness`). Không thao tác trực tiếp trên nhánh `main`.
- **Push/Pull:** mỗi cá nhân chỉ push trên branch của mình; pull/rebase từ `main` vào đầu mỗi ngày làm việc.
- **Merge:** mọi thay đổi vào `main` phải qua Pull Request, được ít nhất 1 thành viên khác approve và CI chạy thành công.
- **Quy mô PR:** PR nhỏ, mỗi PR một tính năng hoặc một thí nghiệm.
- **Quản lý tập tin:** không commit dữ liệu thô, embedding, checkpoint (đưa vào `.gitignore`). Xóa output notebook trước khi commit hoặc chuyển code chuẩn sang `src/`.
- **Quy tắc đặt tên file:** tiếng Anh, không dấu, viết thường, snake_case (`preprocessing.py`, `03_pattern_mining.ipynb`). Nội dung báo cáo, comment, docstring viết tiếng Việt.
- **Merge conflict:** người tạo ra xung đột tự xử lý trước khi yêu cầu review lại.

### 8.2. Quy định code và quản lý thí nghiệm

- **Tính tái lập:** seed cố định {42, 43, 44}; một file chỉ số chia dữ liệu dùng chung; mọi thí nghiệm có file config riêng trong `configs/`; chạy trong môi trường từ `requirements.txt`.
- **Tập test:** chỉ chấm checkpoint đã chọn trên validation; không chọn siêu tham số theo test.
- **Theo dõi thí nghiệm:** mỗi lần train ghi một dòng vào `experiments/log.csv` (tên run, config, seed, giờ GPU, Recall/NDCG trên validation và test, link notebook Kaggle). Số trong báo cáo phải truy được về một dòng log.
- **Ghi nguồn code:** code gốc `RPG_KDD2025` nằm nguyên trong `third_party/` (submodule); adapter và mọi file tự viết ghi rõ ở đầu file; README liệt kê phần của tác giả và phần tự viết.
- **Commit message:** Conventional Commits (`feat: add prefixspan mining`, `fix: adapter id order`, `exp: rpg pq seed42`).
- **Tài liệu:** README phản ánh đúng luồng chạy thực tế; `run_all.sh` chạy lại được từ clone sạch (kiểm tra ngày 7/12).

### 8.3. Cơ chế giao tiếp và quản lý nhóm

- **Họp tuần:** Chủ nhật, 30 phút: đối chiếu với kế hoạch, một thành viên trình bày phần của mình, quyết định cắt giảm nếu chậm.
- **Cổng kiểm tra:** thứ Năm hàng tuần; không qua cổng thì áp dụng dự phòng của tuần đó, không dồn sang tuần sau mà không điều chỉnh.
- **Xử lý blocker:** thông báo ngay trên kênh chat chung, không giữ đến họp định kỳ. Thí nghiệm chạy đêm bị lỗi phải báo trước 9h sáng hôm sau để xếp lại lịch GPU.
- **Integration Owner:** luân phiên theo tuần (Tuần 1: B, Tuần 2: C, Tuần 3: A, Tuần 4: B, Tuần 5: C). Người đảm nhiệm rà soát PR cuối tuần, merge và đảm bảo `main` luôn chạy được.
- **Nguyên tắc với giảng viên:** mọi câu hỏi gom lại hỏi một lần; câu trả lời ghi vào `reports/decisions.md` để cả nhóm cùng theo dõi.
