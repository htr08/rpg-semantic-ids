# RQ3 — Ablation và đồ thị giảm hub

Thí nghiệm dùng phép cắt cạnh trên đồ thị kNN và đối chứng lấy mẫu ứng viên cùng ngân sách. Cả bốn cấu hình dùng checkpoint OPQ `main_s42` đã chọn trên validation, semantic ID OPQ và split test của Beauty. Lượt chấm giữ item đã xem. [Cấu hình thí nghiệm](../configs/rpg_graph_ablation.yaml) · [Protocol đánh giá](../docs/evaluation.md)

## Thiết kế

| Cấu hình | Cách tìm ứng viên |
|---|---|
| Đồ thị gốc | RPG gốc giữ 200 ứng viên mỗi hàng, 20 beam, 3 bước lan truyền. |
| Mutual kNN | Giữ cạnh giữa hai item khi cả hai cùng nằm trong danh sách láng giềng của nhau. |
| Giới hạn bậc vào | Giữ tối đa 200 cạnh vào mỗi item, ưu tiên hạng của item đích trong hàng nguồn, hòa thì ưu tiên ID nguồn nhỏ hơn. Chọn mức 200 trước khi chấm theo độ dài hàng của đồ thị gốc. |
| Bỏ đồ thị | Với từng user, lấy mẫu đều không hoàn lại đúng số item phân biệt mà đồ thị gốc đã thăm, rồi chấm tất cả ứng viên mẫu bằng trung bình logit token RPG. |

Hai đồ thị rút cạnh dùng nguyên hàm lan truyền của RPG, với cùng trạng thái RNG ban đầu như lượt đồ thị gốc. Các ô trống trong bảng láng giềng lặp lại item nguồn để giữ hình dạng tensor. RPG gộp trùng trước khi chấm. Số cạnh và bậc vào chỉ tính cạnh giữa hai item khác nhau.

Đối chứng ngẫu nhiên khớp số item phân biệt đã thăm theo từng user. Tổng số phép chấm và thời gian xử lý có thể khác vì lan truyền có thể ghé lại một item qua nhiều bước. Số liệu hub trong top-10 dùng cùng 20 item có bậc vào cao nhất của đồ thị gốc cho cả bốn cấu hình.

## Đối chiếu trước khi so sánh

Lượt đồ thị gốc lặp lại đúng thứ tự top-10 của cả 22.363 user trong PR 2. File ranking khớp SHA-256, Recall và NDCG của hai lượt giống nhau. So với adjacency CPU ở PR 1, adjacency GPU có cùng tập láng giềng ở 12.089/12.101 hàng. Có 1.010 hàng khác thứ tự và 12 hàng khác tập láng giềng. Thí nghiệm rút cạnh từ chính adjacency GPU đã dùng để chấm đối chứng. [Số liệu và checksum](rpg_graph_ablation.json)

## Kết quả gợi ý

| Cấu hình | Recall@10 | NDCG@10 | Coverage@10 | Long-tail@10 | Item đã thăm trung bình | p-value so với đồ thị gốc |
|---|---:|---:|---:|---:|---:|---:|
| Đồ thị gốc | 0,06851 | **0,03986** | 51,50% | 18,54% | 6.568,53 | — |
| Mutual kNN | 0,06837 | 0,03980 | 51,67% | 18,57% | 6.324,35 | 0,639 |
| Giới hạn bậc vào 200 | 0,06846 | 0,03984 | 51,72% | 18,62% | 6.395,19 | 0,791 |
| Bỏ đồ thị, lấy mẫu cùng ngân sách | 0,05156 | 0,03033 | **59,19%** | **23,10%** | 6.568,53 | 1,32 × 10⁻³⁰ |

Paired t-test hai phía dùng NDCG@10 theo user, với p-value chưa hiệu chỉnh. Khi lấy mẫu cùng ngân sách, NDCG@10 giảm **23,9%** so với dùng đồ thị. Coverage và long-tail tăng, cho thấy lượt lấy mẫu đưa nhiều sản phẩm vào top-10 hơn nhưng tìm item đích kém hơn. Hai cách rút cạnh chỉ làm NDCG@10 thay đổi dưới 0,00006 ở checkpoint này.

## Hubness và mức xuất hiện trong top-10

| Cấu hình | Cạnh có hướng | Độ lệch chuẩn bậc vào | P99 bậc vào | Bậc vào tối đa | Spearman bậc vào ↔ top-10 | Số vị trí của 20 hub gốc |
|---|---:|---:|---:|---:|---:|---:|
| Đồ thị gốc | 2.408.099 | 28,14 | 274 | 347 | 0,188 | 964 (0,431%) |
| Mutual kNN | 2.204.924 | 13,88 | 199 | 199 | 0,179 | 971 (0,434%) |
| Giới hạn bậc vào 200 | 2.279.483 | 15,65 | 200 | 200 | 0,138 | 973 (0,435%) |
| Bỏ đồ thị, lấy mẫu cùng ngân sách | — | — | — | — | — | 1.232 (0,551%) |

Mutual kNN và giới hạn bậc vào đều thu hẹp phân bố bậc, nhưng 20 hub gốc vẫn xuất hiện với tỉ lệ tương đương trong top-10. Giới hạn bậc vào làm tương quan toàn catalog giảm từ 0,188 xuống 0,138. Trên seed 42, cắt cạnh vì vậy làm giảm hubness theo cấu trúc, còn mức xuất hiện của các hub gốc thay đổi rất ít.

## Tái lập

Chuẩn bị cache RPG và thư mục dữ liệu chung theo hướng dẫn xuất ranking. Lệnh chạy cả bốn cấu hình và ghi `rankings.jsonl`, `visited.json` cùng `summary.json` ở thư mục của mỗi lượt. File `summary.json` cấp trên tổng hợp các lượt và ghi checksum đầu vào. Hai tham số đối chiếu lấy ranking của PR 2 và adjacency CPU của PR 1. [Hướng dẫn chuẩn bị cache RPG](../docs/evaluation.md#xuất-ranking-test-từ-checkpoint-rpg)

```bash
python -m scripts.run_graph_ablation \
  --repo experiments/rpg_upstream --root experiments/rpg_ranking_export \
  --checkpoint experiments/rpg_ranking_export/ckpt/main_s42/main_s42-run_rpg.py-Nov-14-2026_16-24-f3eab8.pth \
  --reference-rankings experiments/rpg_ranking_export/main_s42/test/rankings.jsonl \
  --reference-adjacency experiments/graph_hubness/adjacency.npy \
  --output-dir experiments/rpg_graph_ablation
```
