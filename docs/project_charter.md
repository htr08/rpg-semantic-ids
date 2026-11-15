# PROJECT CHARTER

**Khai phá cấu trúc ẩn sau Semantic ID: Phân cụm, mẫu mua kèm và hiện tượng hub trong hệ gợi ý sinh RPG (KDD 2025)**

| Mục | Nội dung |
|---|---|
| Học phần | Khai phá & Phân tích dữ liệu |
| Nhóm | 3 thành viên (A, B, C) |
| Thời gian | Thứ Sáu 6/11 → Thứ Năm 10/12/2026 (5 tuần, 12–15 giờ/người/tuần) |
| Phiên bản | v0.2 – bản nháp (Tuần 1), soạn bởi C; cập nhật theo kết quả pipeline dữ liệu (PR #1) |
| Trạng thái | Chờ chốt tại họp tuần; sau khi chốt, nội dung đồng bộ về mục 1.3 của [project_plan.md](project_plan.md) |
| Tài liệu liên quan | [project_plan.md](project_plan.md), [data_contract.md](data_contract.md), [reports/data_quality.md](../reports/data_quality.md), [reports/decisions.md](../reports/decisions.md), [configs/gpu_budget.md](../configs/gpu_budget.md) |

---

## 1. Bối cảnh

RPG (Hou và cộng sự, KDD 2025) là mô hình gợi ý sinh: mỗi sản phẩm được biểu diễn bằng một semantic ID gồm 32 token, tạo bằng Optimized Product Quantization (OPQ) trên embedding văn bản của mô tả sản phẩm. Mô hình dự đoán song song toàn bộ token của sản phẩm kế tiếp và giải mã bằng lan truyền trên đồ thị kNN giữa các sản phẩm.

Bản chất của OPQ là k-means trên từng không gian con, nên semantic ID có thể được xem như một phép phân cụm nhiều tầng của tập sản phẩm. Đồ thị giải mã là một đồ thị kNN, đối tượng quen thuộc của khai phá đồ thị. Dữ liệu hành vi (chuỗi mua hàng) chứa mẫu mua kèm có thể khai phá bằng luật kết hợp. Dự án tận dụng ba điểm này để đặt RPG vào khung kỹ thuật của học phần: mô hình là công cụ, đối tượng phân tích là cấu trúc của dữ liệu sản phẩm.

Dữ liệu sau tiền xử lý có các đặc điểm định hình phạm vi phân tích (nguồn: [reports/data_quality.md](../reports/data_quality.md)):

| Đặc điểm | Giá trị | Hệ quả cho dự án |
|---|---|---|
| Độ thưa ma trận user–item | 99,93% | Bằng chứng đồng xuất hiện ít; luật kết hợp cấp item dự kiến thưa (RQ2). |
| Độ dài chuỗi train | Trung bình 6,88; trung vị 4; p95 19 | Giao dịch ngắn, khó đạt ngưỡng support với tập mục độ dài ≥ 3. |
| Tần suất item trong train | Trung vị 7; p95 44; nhỏ nhất 0 | Phần lớn item không đủ support ở mọi mức min_support dự kiến; có item không xuất hiện trong train (cold-start, RQ4). |
| Tập trung độ phổ biến | 20% item (2.421) chiếm 60,39% tương tác train | Nguy cơ thiên lệch phổ biến; cơ sở cho phân tích long-tail và hub (RQ3, RQ4). |
| Tương tác dư khi gộp theo (user, timestamp) | 77.402 (số tương tác trừ số cặp user–timestamp khác nhau) | Thứ tự trong các nhóm cùng timestamp dựa vào số dòng file gốc, không phản ánh thứ tự mua thực; ảnh hưởng tới PrefixSpan (RQ2). |
| Metadata thiếu | brand 17,34%; description 7,78%; categories 0% | Nhãn danh mục đủ cho purity/NMI; nhãn thương hiệu chỉ phủ 82,66% item (RQ1). |

## 2. Mục tiêu và câu hỏi nghiên cứu

**Câu hỏi trung tâm:** semantic ID khai phá được cấu trúc gì trong dữ liệu sản phẩm, và cấu trúc đó giúp hay hại việc gợi ý?

| Mã | Câu hỏi | Kỹ thuật học phần |
|---|---|---|
| RQ1 | Semantic ID có phải là một phép phân cụm tốt không? Chất lượng cụm có đi cùng NDCG không? | k-means, phân cụm phân cấp, DBSCAN; silhouette, Davies–Bouldin, purity, NMI, ARI |
| RQ2 | Mẫu mua kèm có được phản ánh trong semantic ID không? Đưa tín hiệu hành vi vào semantic ID có cải thiện gợi ý không? | FP-Growth, luật kết hợp (support, confidence, lift), PrefixSpan |
| RQ3 | Đồ thị kNN dùng để giải mã có hiện tượng hub không? Hub có bị gợi ý quá mức không? | Khai phá đồ thị: bậc vào, độ lệch, phát hiện cộng đồng (Louvain) |
| RQ4 | Ngoài Recall/NDCG, các mô hình phủ được bao nhiêu sản phẩm, thiên lệch độ phổ biến ra sao, tốt và kém với nhóm sản phẩm/người dùng nào? | Đánh giá mô hình, phân khúc (k-means trên đặc trưng hành vi) |

## 3. Phạm vi

### 3.1. Trong phạm vi

- Dữ liệu Amazon Reviews 2014 – Beauty (`reviews_Beauty_5.json.gz`, `meta_Beauty.json.gz`, checksum trong `configs/data_sources.json`), lọc 5-core, chia leave-last-out. Bộ chia `data/splits/leave_last_out.csv` và `manifest.json` là nguồn duy nhất cho mọi mô hình và module.
- Huấn luyện RPG bằng code gốc `RPG_KDD2025` (submodule trong `third_party/`) với encoder `sentence-t5-base`, m = 32, codebook 256.
- Ghép văn bản mô tả theo cách của code gốc, kèm quy tắc fallback cho item thiếu description (title + category).
- Adapter nạp semantic ID hoặc embedding do nhóm tạo vào code gốc.
- Tokenizer tự viết: OPQ, PQ, RQ, token ngẫu nhiên, ID có tín hiệu hành vi.
- Baseline: Popularity, ItemKNN, SASRec (RecBole), AR-Rec (tự viết, dựa trên FP-Growth).
- Module khai phá mẫu (FP-Growth, PrefixSpan, đối chiếu luật ↔ semantic ID), module phân cụm (PCA, k-means, phân cấp, DBSCAN), phân tích đồ thị (hubness, Louvain), đánh giá mở rộng (coverage, long-tail, cold-start, phân khúc).
- Báo cáo tổng kết, slide, repository chạy lại được từ bản clone sạch.

### 3.2. Ngoài phạm vi

- Cài đặt lại kiến trúc, hàm loss hoặc vòng huấn luyện của RPG.
- Encoder qua API trả phí (`text-embedding-3-large`); do đó không so với mốc 0,0464 ở Table 2 của paper.
- Dữ liệu ảnh (CLIP), liên kết mua kèm có sẵn trong metadata (`related`) và các tập Amazon khác (Sports, Toys, CDs).
- Lọc hoặc sửa dữ liệu chỉ để khớp một con số của paper.
- Suy đoán giá trị cho trường metadata bị thiếu (ví dụ brand).
- Tối ưu siêu tham số diện rộng cho RPG; chỉ dùng siêu tham số theo paper.
- Triển khai thực tế, phục vụ trực tuyến.

## 4. Sản phẩm bàn giao

| Sản phẩm | Phụ trách chính | Hạn | Trạng thái |
|---|---|---|---|
| Project Charter (tài liệu này) | C | 8/11 (v1) | Nháp v0.2 |
| Script tải dữ liệu, pipeline 5-core, bộ chia, data contract, notebook EDA, báo cáo chất lượng dữ liệu | B | 12/11 | Đã merge (PR #1, 13/11) |
| Văn bản mô tả sau fallback (941 item thiếu description, trong đó 7 item thiếu cả title) | A | Trước khi encode | Chưa thực hiện |
| `semantic_ids.npy` (OPQ) và báo cáo chất lượng ID | A | 12/11 | Chưa có trên `main` |
| `src/evaluate.py` có unit test | C (bản đầu), B (mở rộng) | 12/11 / 19/11 | Mới có docstring khung trên `main`, chưa có hàm |
| Kết quả RPG code gốc 3 seed; adapter | A | 19/11 (adapter: 18/11) | |
| Bảng RQ1: độ đo cụm của 4 cách tạo ID; bảng so sánh thuật toán phân cụm | A, C | 26/11 | |
| Bảng RQ2: luật, mẫu tuần tự, AR-Rec, đối chiếu luật ↔ token | C | 26/11 | |
| Bảng/hình RQ3, RQ4 | B | 26/11 | |
| Nâng cấp chính (ID có tín hiệu hành vi) 3 seed; nâng cấp phụ (đồ thị giảm hub) | A, C / B | 3/12 | |
| Báo cáo cuối (PDF), slide, repository hoàn chỉnh | C hợp nhất; cả nhóm | 10/12 | |

## 5. KPI

| # | KPI | Ngưỡng đạt | Cách đo | Bắt buộc | Trạng thái |
|---|---|---|---|---|---|
| K1 | Dữ liệu khớp paper | 22.363 user, 12.101 item; 198.502 tương tác sau 5-core, trong đó train 153.776, train + validation 176.139 (khớp mốc của kế hoạch). Mọi con số báo cáo ghi rõ phạm vi đếm. | `manifest.json`, `reports/eda_summary.json`; đối chiếu Table 1 | Có | Đạt về số lượng (13/11); còn đối chiếu định nghĩa thống kê trong Table 1 |
| K2 | RPG mô hình nền | NDCG@10 trong khoảng 0,038–0,045 (mốc 0,0429, Table 4, `sentence-t5-base`) | `src/evaluate.py` trên top-K của code gốc, mean ± std 3 seed | Có | Chưa đo |
| K3 | RPG so với SASRec | NDCG@10 của RPG cao hơn SASRec, paired t-test p < 0,05 | Paired t-test theo user trên NDCG@10 | Có (nếu không đạt: báo cáo kèm phân tích nguyên nhân) | Chưa đo |
| K4 | RQ1 – phân cụm | ≥ 3 thuật toán phân cụm và ≥ 3 cách tạo semantic ID được so bằng cùng bộ độ đo; có hình tương quan chất lượng cụm ↔ NDCG@10 | Bảng phụ 1, 2 của kế hoạch | Có | Chưa đo |
| K5 | RQ2 – khai phá mẫu | FP-Growth chạy ở 3 mức min_support; PrefixSpan chạy được; AR-Rec có trong bảng kết quả chung; có số đo độ khớp luật ↔ semantic ID | Bảng phụ 3 | Có | Chưa đo |
| K6 | RQ3 – đồ thị | Có số đo hubness (độ lệch bậc vào) và kiểm định hub ↔ gợi ý quá mức (tương quan Spearman) | Bảng phụ 4 | Có | Chưa đo |
| K7 | RQ4 – đánh giá mở rộng | Coverage, tỉ lệ long-tail, NDCG@10 theo nhóm cold-start cho mọi mô hình; nhóm head/long-tail theo định nghĩa của EDA (top 20% item theo tần suất train) | Bảng phụ 5 | Có | Chưa đo |
| K8 | Kết luận | Mỗi RQ có ít nhất một bảng hoặc hình và một câu kết luận kèm con số | Rà soát báo cáo | Có | |
| K9 | Tái lập | `run_all.sh` chạy lại từ bản clone sạch; `--verify-only` khớp `manifest.json`; mọi số trong báo cáo truy được về một dòng `experiments/log.csv` | Kiểm tra ngày 7/12 | Có | Phần dữ liệu đã đạt |
| K10 | Ngân sách GPU | Tổng ≤ 45 giờ GPU; cảnh báo ở 80% | `configs/gpu_budget.md` | Có | Chưa có lần train nào được ghi trong `gpu_budget.md` |

## 6. Definition of Done

### 6.1. DoD chung cho mọi đầu việc

Một đầu việc được coi là hoàn thành khi thỏa toàn bộ các điều kiện sau:

1. Code nằm trên `main` qua Pull Request, có ít nhất 1 approve và CI (pytest) chạy thành công.
2. Hàm trong `src/` có docstring tiếng Việt và unit test cho trường hợp tính tay được.
3. Thí nghiệm có file config trong `configs/`, seed cố định, một dòng trong `experiments/log.csv`.
4. Đọc dữ liệu qua các file trong `data/processed/` và bộ chia trong `data/splits/` theo [data_contract.md](data_contract.md); không ghép mapping hoặc bộ chia từ nguồn khác.
5. Chỉ dùng phần train để fit hoặc thống kê (OPQ trên embedding là ngoại lệ vì không dùng tương tác; FP-Growth, PrefixSpan, ItemKNN, Popularity, embedding đồng xuất hiện, phân nhóm long-tail đều lấy từ `train_sequences.jsonl`); test chỉ chạy cho cấu hình đã chọn trên validation.
6. Kết quả có bảng hoặc hình lưu trong `reports/figures/` và một câu kết luận kèm con số.
7. Notebook đã xóa output; không commit dữ liệu thô, embedding, checkpoint.

### 6.2. DoD theo sản phẩm

| Sản phẩm | Điều kiện hoàn thành bổ sung |
|---|---|
| Pipeline dữ liệu (đã đạt) | Đạt K1; chạy lại cho đầu ra giống nhau từng byte; `--verify-only` khớp checksum trong `manifest.json`; unit test xác nhận mỗi tương tác thuộc đúng một phần chia và nhãn validation/test không nằm trong lịch sử dùng để dự đoán chính nhãn đó. |
| Văn bản mô tả | Ghép theo thứ tự trường của code gốc; item thiếu description dùng title + category; ghi số item áp dụng fallback; không đổi thứ tự item so với `items.jsonl`. |
| `src/evaluate.py` | Recall@K, NDCG@K khớp ví dụ tính tay; loại item trong `seen_item_ids` khỏi danh sách ứng viên; khớp metric của RecBole trên ít nhất một mô hình (sai lệch < 1e-4). |
| RPG code gốc | Đạt K2; adapter nạp lại chính ID OPQ của code gốc cho NDCG@10 không đổi (trong phạm vi dao động seed). |
| Tokenizer | 4 file semantic ID cùng định dạng (12.101 × 32, giá trị 0–255), hàng `item_id − 1` ứng với item đó; nếu adapter dùng hàng 0 cho padding thì chuyển đổi tường minh; có tỉ lệ va chạm và entropy codebook. |
| Module khai phá mẫu | Giao dịch là tập item duy nhất của chuỗi train đầy đủ (không cắt 50); bảng số tập phổ biến, số luật, thời gian chạy ở 3 mức min_support, kèm số item đạt ngưỡng support; 20 luật lift cao nhất kèm tên sản phẩm và số đếm tuyệt đối; AR-Rec chấm bằng `src/evaluate.py`. |
| Mẫu tuần tự | PrefixSpan chạy trên `item_ids` nguyên thứ tự; nêu rõ cách xử lý các tương tác trùng timestamp và kiểm tra độ nhạy của kết quả với cách xử lý này. |
| Module phân cụm | Bảng silhouette, Davies–Bouldin, số cụm, tỉ lệ nhiễu, ARI với token đầu OPQ; ghi tỉ lệ phương sai giữ lại của PCA; purity/NMI theo brand chỉ tính trên item có brand và ghi độ phủ nhãn. |
| Phân tích đồ thị | Phân bố bậc vào, độ lệch, 20 hub, tương quan hub ↔ độ phổ biến (tần suất train) và ↔ tần suất trong top-10. |
| Báo cáo | Mỗi mục kỹ thuật mở đầu bằng câu dẫn về bài giảng và công thức; bảng mapping kiến thức ở phụ lục; số paper ghi rõ bảng nguồn; số tương tác ghi rõ phạm vi đếm. |

## 7. Scope tối thiểu (MVP) và phần mở rộng

**MVP (bắt buộc):** dữ liệu khớp Table 1 (đạt về số lượng); RPG chạy bằng code gốc; baseline Popularity, ItemKNN, SASRec, AR-Rec; RQ1 (≥ 3 thuật toán phân cụm, ≥ 3 cách tạo ID, độ đo cụm); RQ2 (FP-Growth, PrefixSpan, đối chiếu luật ↔ token); RQ3 (phân bố bậc, hubness); RQ4 (coverage, long-tail, cold-start); báo cáo; slide.

**Mở rộng:** tương quan cụm ↔ NDCG đủ 4 biến thể (cần train biến thể); nâng cấp chính; Louvain; phân khúc người dùng; đồ thị giảm hub; encoder mới (dự phòng).

## 8. Ràng buộc và danh sách cắt giảm

| Ràng buộc | Nội dung |
|---|---|
| Thời gian | 5 tuần, hạn chót 10/12 cố định; đóng băng thí nghiệm ngày 5/12. |
| Tính toán | Kaggle T4/P100 (30 giờ/tuần/tài khoản), Colab dự phòng; ước 3–4 giờ/lần train RPG; khoảng 11 lần train, 35–45 giờ GPU. |
| Chi phí | Không dùng API trả phí. |
| Mã nguồn | Code gốc giữ nguyên trong `third_party/`; mọi thay đổi đi qua `adapters/`. |
| Dữ liệu | `data/processed/` không commit, mỗi thành viên tự tạo lại bằng pipeline và kiểm tra bằng `--verify-only` trước khi dùng. |

Khi vượt 80% ngân sách GPU hoặc chậm tiến độ 2 ngày, cắt theo thứ tự:

1. Encoder mới.
2. Biến thể RQ (giữ OPQ, PQ, ngẫu nhiên).
3. Phát hiện cộng đồng Louvain.
4. Seed của nâng cấp chính từ 3 xuống 2 (cấu hình chính luôn giữ 3 seed).

Không cắt: EDA, đánh giá cụm của RQ1, khai phá mẫu của RQ2, báo cáo.

## 9. Giả định và điểm cần xác minh

| # | Nội dung | Kết quả / cách xác minh | Trạng thái |
|---|---|---|---|
| G1 | Số tương tác trong Table 1 | Pipeline cho 198.502 tương tác sau 5-core; 176.139 là số train + validation (bằng 198.502 − 22.363). Chưa xác định Table 1 dùng phạm vi đếm nào; báo cáo nêu cả hai số. | Đã xác minh số lượng; còn đối chiếu định nghĩa trong paper |
| G2 | Giao dịch cho FP-Growth | 22.363 giao dịch, độ dài trung bình 6,88, trung vị 4. Ngưỡng min_support 0,0005 / 0,001 / 0,002 tương ứng số đếm ≥ 12 / ≥ 23 / ≥ 45; với tần suất item trung vị 7 và p95 44, hơn một nửa item không đạt mức thấp nhất và ít nhất 95% item không đạt mức cao nhất. | Đã xác minh; rủi ro luật thưa được nâng lên mức Cao |
| G3 | Code gốc cho phép nạp semantic ID tự tạo | A đọc `genrec/models/RPG`; nếu không được, áp dụng phương án thay file embedding. | Chờ, hạn 18/11 |
| G4 | Nhãn để tính purity/NMI | categories có cho 100% item; brand thiếu 17,34%. Cần thống nhất cách chọn danh mục lá từ đường dẫn `categories` (một item có thể có nhiều đường dẫn). | Đã xác minh độ phủ; chờ quy tắc chọn danh mục lá |
| G5 | Giảng viên chấp nhận dùng code gốc và thư viện (mlxtend, scikit-learn, FAISS) thay vì tự cài thuật toán | Câu hỏi gửi giảng viên (`reports/decisions.md`). | Chưa trả lời; quyết định lại 20/11 |
| G6 | Thứ tự trong chuỗi phản ánh thứ tự mua | Có 77.402 tương tác dư khi gộp theo (user, timestamp), tức nhiều user có nhiều tương tác cùng timestamp; pipeline sắp các tương tác này theo số dòng file gốc. Thứ tự trong các nhóm này không mang thông tin thời gian. | Đã phát hiện; C xử lý trong PrefixSpan (mục 6.2) |
| G7 | Mọi item có tương tác train | Tần suất train nhỏ nhất bằng 0, tức có item chỉ xuất hiện ở validation/test. Các item này không có trong luật kết hợp và không có embedding đồng xuất hiện từ train. | Cần đếm số item này; ảnh hưởng AR-Rec, nâng cấp chính, cold-start |

## 10. Mốc kiểm tra

| Cổng | Ngày | Tiêu chí chính | Trạng thái |
|---|---|---|---|
| Cổng 1 | 12/11 | Dữ liệu khớp Table 1; `semantic_ids.npy` ít va chạm; baseline đã log; FP-Growth chạy ở 3 mức support; k-means có silhouette; mapping kiến thức điền xong. | Phần dữ liệu đạt (13/11); các tiêu chí còn lại chưa có trên `main` |
| Cổng 2 | 17/11 | RPG code gốc đạt K2; adapter nạp lại ID OPQ cho cùng kết quả. | |
| Cổng 3 | 26/11 | Mỗi RQ có kết luận sơ bộ kèm con số; chốt nâng cấp chính. | |
| Cổng 4 | 3/12 | Nâng cấp chính có bảng 3 seed và p-value; báo cáo v1 đủ mục. | |
| Nộp bài | 10/12 | Báo cáo PDF, link repository, slide. | |

Không qua cổng thì áp dụng phương án dự phòng của tuần tương ứng trong kế hoạch, không dồn việc sang tuần sau khi chưa điều chỉnh.

## 11. Vai trò

| TV | Vai trò | Phần báo cáo |
|---|---|---|
| A | Semantic ID & Mô hình (RQ1, nâng cấp chính); ghép văn bản mô tả và fallback; theo dõi ngân sách GPU | Nền tảng RPG, RQ1, Nâng cấp chính |
| B | Dữ liệu, Đánh giá & Đồ thị (RQ3, RQ4); duy trì pipeline và data contract | Dữ liệu, RQ3, RQ4, Nâng cấp phụ |
| C | Khai phá mẫu, Phân cụm & Tổng hợp (RQ1, RQ2); hợp nhất báo cáo và slide; mapping kiến thức | Khai phá mẫu, So sánh phân cụm, Thảo luận; hợp nhất |

Review chéo: A review B, B review C, C review A. Integration Owner luân phiên theo tuần (Tuần 1: B, Tuần 2: C, Tuần 3: A, Tuần 4: B, Tuần 5: C).

## 12. Rủi ro chính

| Rủi ro | Mức độ | Phương án |
|---|---|---|
| Không nạp được semantic ID tự tạo vào code gốc | Cao | Thay file embedding đầu vào; biến thể PQ, RQ, ngẫu nhiên chỉ đánh giá bằng độ đo cụm. |
| Hết ngân sách GPU | Cao | Danh sách cắt giảm mục 8; dùng 2–3 tài khoản. |
| Luật kết hợp quá thưa | Cao (theo G2) | Quét 3 mức min_support, báo số item đạt ngưỡng; phân tích luật ở cấp thương hiệu/danh mục; AR-Rec bổ sung Popularity khi thiếu luật. |
| Mẫu tuần tự phụ thuộc thứ tự giả trong nhóm trùng timestamp | Trung bình | Gộp các tương tác cùng timestamp thành một phần tử (itemset) hoặc chạy hai cách và so kết quả. |
| DBSCAN cho một cụm lớn + nhiễu | Trung bình | PCA trước, khoảng cách cosine, eps từ đồ thị k-distance; thất bại thì ghi thành nhận xét về giới hạn thuật toán. |
| Rò rỉ dữ liệu | Thấp (đã có kiểm tra ở pipeline) | Bộ chia dùng chung có checksum; mọi thống kê tần suất lấy từ train; mỗi module mới có test riêng. |
| So sai mốc paper hoặc lẫn phạm vi đếm | Thấp | Ghi bảng nguồn cho mọi số paper và phạm vi đếm cho mọi số tương tác; C kiểm tra khi hợp nhất. |

Danh sách đầy đủ: mục 7 của [project_plan.md](project_plan.md).

## 13. Nguyên tắc

- **Trung thực:** kết quả âm (nâng cấp không cải thiện, cụm không đi cùng NDCG, FP-Growth yếu, RPG lệch mốc paper) được trình bày thành phát hiện kèm lý giải, không loại khỏi báo cáo.
- **Tái lập:** seed {42, 43, 44}; config riêng cho từng thí nghiệm; môi trường theo `requirements.txt`; dữ liệu kiểm tra bằng checksum.
- **Hiểu toàn bộ dự án:** mỗi thành viên vẽ lại được luồng dữ liệu của hai phần còn lại (kiểm tra Tuần 2 và Tuần 5).
- **Ghi nguồn:** README phân biệt phần code của tác giả và phần tự viết.

## Lịch sử phiên bản

| Phiên bản | Ngày | Nội dung |
|---|---|---|
| v0.1 | 13/11 | Bản nháp đầu tiên; đã phản ánh quyết định dùng code gốc ghi trong `reports/decisions.md` (13/11). |
| v0.2 | 13/11 | Cập nhật theo pipeline dữ liệu và EDA (PR #1): thêm đặc điểm dữ liệu vào bối cảnh; K1 ghi rõ phạm vi đếm và trạng thái đạt; DoD bổ sung data contract, quy ước chỉ số item, xử lý trùng timestamp, độ phủ nhãn brand; G1, G2, G4 đã xác minh; thêm G6, G7; nâng rủi ro luật thưa lên Cao; thêm cột trạng thái cho sản phẩm, KPI và cổng. |
