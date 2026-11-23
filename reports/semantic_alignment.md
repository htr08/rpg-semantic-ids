# Đối chiếu luật kết hợp và k-means với semantic ID (OPQ)

12101 item, 32 vị trí token, codebook 256. Luật sinh từ phần train. Số token chung của cặp (A, B) là số vị trí có cùng mã ở cùng vị trí. Sinh bởi `python -m src.semantic_alignment`.

## Cặp item trong luật lift cao nhất so với cặp ngẫu nhiên

Cặp là luật một item ⇒ một item, mỗi cặp chỉ tính một lần, xếp theo lift giảm dần. Cặp ngẫu nhiên: hai item khác nhau chọn đều trong catalog. Ngưỡng t ("nhiều token chung") là số nhỏ nhất để không quá 5% cặp ngẫu nhiên có ≥ t token chung. Các cặp dùng chung item (cột "Item") nên không độc lập; p-value lấy từ kiểm định hoán vị hàng ID giữa các item có mặt trong cặp, giữ nguyên bậc từng item, và giá trị nhỏ nhất đo được là 1/(1 + số lần hoán vị).

| min_support | Cặp dùng / có sẵn | Item | Count nhỏ nhất | Trung bình token chung: luật / ngẫu nhiên / hoán vị | Cặp không trùng token: luật / ngẫu nhiên | Tỉ lệ ≥ 1: luật / ngẫu nhiên | t | Tỉ lệ ≥ t: luật / ngẫu nhiên / hoán vị | p trung bình | p tỉ lệ ≥ t | Cặp trùng hoàn toàn ID |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0005 | 200 / 7015 | 84 | 12 | 0.650 / 0.145 / 0.245 | 0.660 / 0.866 | 0.340 / 0.134 | 2 | 0.110 / 0.010 / 0.031 | 0.0002 | 0.0002 | 0 |
| 0.001 | 200 / 1423 | 109 | 23 | 0.755 / 0.145 / 0.185 | 0.585 / 0.866 | 0.415 / 0.134 | 2 | 0.155 / 0.010 / 0.019 | 0.0002 | 0.0002 | 0 |
| 0.002 | 102 / 102 | 66 | 45 | 0.873 / 0.145 / 0.227 | 0.569 / 0.866 | 0.431 / 0.134 | 2 | 0.176 / 0.010 / 0.028 | 0.0002 | 0.0002 | 0 |

Tách theo thương hiệu. Cùng thương hiệu: cả hai item có brand và brand giống nhau (không phân biệt hoa thường); thiếu thương hiệu: ít nhất một item không có brand; khác thương hiệu: đủ brand nhưng khác nhau.

Với nhóm cùng thương hiệu, đối chứng là hoán vị hàng ID trong từng brand (giữa mọi item của brand đó trong catalog), nên cặp ngẫu nhiên vẫn cùng brand.

| min_support | Nhóm | Số cặp | Trung bình token chung | Tỉ lệ ≥ t | Trung bình khi hoán vị trong brand | Tỉ lệ ≥ t khi hoán vị trong brand | p trung bình | p tỉ lệ ≥ t |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0005 | same_brand | 34 | 0.735 | 0.147 | 0.524 | 0.100 | 0.13 | 0.23 |
| 0.0005 | different_brand | 9 | 0.444 | 0.111 | — | — | — | — |
| 0.0005 | missing_brand | 157 | 0.643 | 0.102 | — | — | — | — |
| 0.001 | same_brand | 89 | 1.112 | 0.247 | 0.530 | 0.091 | 0.0016 | 0.0002 |
| 0.001 | different_brand | 19 | 0.368 | 0.053 | — | — | — | — |
| 0.001 | missing_brand | 92 | 0.489 | 0.087 | — | — | — | — |
| 0.002 | same_brand | 37 | 1.676 | 0.378 | 0.741 | 0.148 | 0.0006 | 0.0008 |
| 0.002 | different_brand | 17 | 0.294 | 0.059 | — | — | — | — |
| 0.002 | missing_brand | 48 | 0.458 | 0.062 | — | — | — | — |

## ARI giữa k-means và token OPQ

k-means k = 256 trên PCA 50 chiều (giữ 0.69474 phương sai; PCA fit một lần), n_init = 5, seed [42, 43, 44]. Mỗi token OPQ mã hóa một không gian con riêng của vector, nên ARI giữa các token khác nhau cũng được nêu làm mốc.

| Chỉ số | Giá trị |
|---|---:|
| ARI với token đầu, từng seed | 0.0111, 0.0109, 0.0112 |
| ARI với token đầu, trung bình | 0.0111 |
| ARI với các vị trí 1..31, trung bình qua seed (nhỏ nhất / trung vị / lớn nhất theo vị trí) | 0.0094 / 0.0107 / 0.0136 (lớn nhất tại vị trí 28) |
| Mốc xáo trộn: ARI của token đầu đã xáo trộn với k-means seed đầu (20 lần, trung bình ± độ lệch chuẩn) | -1.5e-05 ± 9.9e-05 |
| ARI giữa token đầu và token ở vị trí khác (trung bình / lớn nhất) | 0.0013 / 0.0017 |
| ARI giữa hai lần k-means khác seed (seed đầu và seed thứ hai) | 0.5393 |
