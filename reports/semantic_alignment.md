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

Với nhóm cùng thương hiệu, đối chứng là hoán vị hàng ID trong từng (brand, nhóm tần suất train): chỉ các item có số đếm train ≥ min_count của brand đó (item hiếm không thể vào luật nên không dùng làm mẫu), chia thành 3 nhóm theo phân vị số đếm train của các item đủ điều kiện trong toàn catalog. Cặp ngẫu nhiên vẫn cùng brand và mỗi item giữ nhóm tần suất của nó, nên khác biệt về tần suất không bị tính là tín hiệu mua kèm, nhưng 3 nhóm là ghép thô nên có thể còn dư tác động của tần suất; p-value mang tính mô tả. Cặp có đối chứng bất biến (cột "Cặp đối chứng bất biến") là cặp mà cả hai item là phần tử duy nhất của nhóm, hoặc hai item là toàn bộ nhóm: hoán vị không đổi số token chung nên kiểm định bảo thủ hơn.

| min_support | Nhóm | Số cặp | Trung bình token chung | Tỉ lệ ≥ t | Trung bình khi hoán vị | Tỉ lệ ≥ t khi hoán vị | p trung bình | p tỉ lệ ≥ t | Cặp đối chứng bất biến |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0005 | same_brand | 34 | 0.735 | 0.147 | 0.705 | 0.136 | 0.41 | 0.48 | 2 / 34 |
| 0.0005 | different_brand | 9 | 0.444 | 0.111 | — | — | — | — | — |
| 0.0005 | missing_brand | 157 | 0.643 | 0.102 | — | — | — | — | — |
| 0.001 | same_brand | 89 | 1.112 | 0.247 | 0.604 | 0.124 | 0.0002 | 0.0024 | 0 / 89 |
| 0.001 | different_brand | 19 | 0.368 | 0.053 | — | — | — | — | — |
| 0.001 | missing_brand | 92 | 0.489 | 0.087 | — | — | — | — | — |
| 0.002 | same_brand | 37 | 1.676 | 0.378 | 1.088 | 0.227 | 0.0034 | 0.011 | 2 / 37 |
| 0.002 | different_brand | 17 | 0.294 | 0.059 | — | — | — | — | — |
| 0.002 | missing_brand | 48 | 0.458 | 0.062 | — | — | — | — | — |

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

## Đầu vào

Thứ tự hàng của semantic ID đã kiểm: 12101 hàng khớp từng hàng với `sentence-t5-base_OPQ32,IVF1,PQ32x8.sem_ids` (khóa theo asin) xếp theo `id_mapping.json`. Embedding so với semantic ID: láng giềng gần nhất theo cosine của 1500 item ngẫu nhiên chung trung bình 2.93 token, cặp ngẫu nhiên 0.14 (tỉ số 20.7, ngưỡng 5). Đây là kiểm thống kê, chỉ bắt được lệch thứ tự trên diện rộng và không chỉ ra từng hàng. Số hàng được kiểm bằng số item của `id_mapping.json`, và `items.jsonl` khớp mapping từng dòng. SHA-256 dưới đây để đối chiếu đúng phiên bản file đầu vào khi tái lập.

| Đầu vào | File | SHA-256 |
|---|---|---|
| semantic_ids | `semantic_ids.npy` | `e7b2c233ff44b652e735bde04e5f844ab301bb2a6e8fa2f873ef1f11acfc503b` |
| embeddings | `item_embeddings.npy` | `a2b542bdf8f34b1f6c81695b5cff7f3c3b98d75d06bab51721bd10b54628a836` |
| id_mapping | `id_mapping.json` | `6f2a054b369e1cac8e5c9ec65b70dbdc7e60c4f04578a430b8cb818adb70f068` |
| items | `items.jsonl` | `6c37776f8a25d4c63f066e6a02525e9984b71f5972033851c894d12e217f1378` |
| interactions | `interactions.csv` | `ca7536f499b5b752a13531197ccfa2e7accafe34331b5083c8979ee72a0db400` |
| splits | `leave_last_out.csv` | `76e6952986ac268e2ecf4d68475e3aaf24f7700b86ee5a26d35a18c14d101c1f` |
| asin_sem_ids | `sentence-t5-base_OPQ32,IVF1,PQ32x8.sem_ids` | `e85affbd11102542ea2be0d64aff1491575ec625dc67e9dc82efb0cc57ef140c` |
