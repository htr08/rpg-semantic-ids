# Độ thưa của luật kết hợp và ảnh hưởng đến AR-Rec (validation)

Phụ trách: C (RQ2), T7 14/11. Số liệu lấy từ [ar_rec_sweep.json](ar_rec_sweep.json) (bảng đầy đủ: [ar_rec_sweep.md](ar_rec_sweep.md)) và [eda_summary.json](eda_summary.json). Mọi kết quả trong tài liệu này là trên bộ chia validation; test chưa chạy.

## 1. Độ dài chuỗi

| Phạm vi | Trung bình | Trung vị | Nguồn |
|---|---:|---:|---|
| Toàn bộ chuỗi (train + validation + test) | 8,88 | 6 | `sequence_length_full` |
| Phần train | 6,88 | 4 | `sequence_length_train` |

- Giá trị chính xác của chuỗi toàn bộ là 198.502 / 22.363 = 8,8764, làm tròn 8,88. Con số 8,87 trong kế hoạch là giá trị cắt cụt.
- Luật chỉ được khai phá từ phần train (và lịch sử của user khi chấm validation cũng là phần train), nên đại lượng liên quan đến AR-Rec là 6,88, không phải 8,88.
- 50,9% user (11.383 / 22.363) có không quá 4 item khác nhau trong lịch sử train.

## 2. Từ chuỗi ngắn đến cặp item hiếm

Giao dịch là tập item duy nhất của phần train mỗi user (22.363 giao dịch, 12.101 item).

| Đại lượng | Giá trị |
|---|---:|
| Cặp item cùng xuất hiện ít nhất một lần | 834.194 |
| Trung vị số giao dịch chứa một cặp | 1 |
| Tần suất train của item: trung vị / p95 | 7 / 44 |
| Item không có tương tác train | 33 |

Số cặp đạt đếm tối thiểu theo min_support (đếm tối thiểu = ⌈min_support × 22.363⌉):

| min_support | Đếm tối thiểu | Item phổ biến | Cặp đạt đếm tối thiểu | Tỉ lệ trong 834.194 cặp |
|---:|---:|---:|---:|---:|
| 0,002 | 45 | 601 (5,0%) | 102 | 0,01% |
| 0,001 | 23 | 1.452 (12,0%) | 1.423 | 0,17% |
| 0,0005 | 12 | 3.218 (26,6%) | 7.070 | 0,85% |
| 0,0001 | 3 | 11.541 (95,4%) | 55.652 | 6,7% |

Ở các mức support của kế hoạch (0,0005–0,002), dưới 1% số cặp đồng xuất hiện đạt ngưỡng, và 73–95% item không phổ biến nên không thể xuất hiện trong luật.

## 3. Hệ quả lên số user được phủ bởi luật

Luật 1 item ⇒ 1 item (`max_len = 2`), `min_confidence` theo cấu hình tốt nhất của từng mức:

| min_support | Luật | User có luật khớp | Nhãn là item phổ biến | Nhãn nằm trong hậu quả của luật khớp |
|---:|---:|---:|---:|---:|
| 0,002 | 204 | 14,1% | 21,7% | 1,1% |
| 0,001 | 2.837 | 25,0% | 38,7% | 3,2% |
| 0,0005 | 14.089 | 44,0% | 56,8% | 6,3% |
| 0,0001 | 84.883 | 91,9% | 95,7% | 16,1% |

- "Nhãn là item phổ biến" là điều kiện cần để nhãn xuất hiện trong hậu quả của bất kỳ luật nào ở mức đó. "Nhãn nằm trong hậu quả" là cận trên của recall khi chỉ dùng luật; phần Popularity bổ sung vẫn có thể trúng nhãn.
- 51 user có nhãn validation chưa từng xuất hiện ở train và 958 user (4,3%) có nhãn với tần suất train dưới 3; các nhãn này không bao giờ nằm trong luật có đếm tối thiểu từ 3 trở lên.

Tỉ lệ user có luật khớp theo độ dài lịch sử (số item khác nhau trong train):

| min_support | ≤ 4 (11.383 user) | 5–9 (7.433) | 10–19 (2.528) | ≥ 20 (1.019) |
|---:|---:|---:|---:|---:|
| 0,002 | 7,0% | 13,7% | 30,1% | 56,3% |
| 0,001 | 14,5% | 26,3% | 48,9% | 73,9% |
| 0,0005 | 30,5% | 48,0% | 73,9% | 91,2% |
| 0,0001 | 86,1% | 96,9% | 99,8% | 100,0% |

Ở mọi mức, tỉ lệ có luật khớp tăng theo độ dài lịch sử; nhóm ≤ 4 item (một nửa số user) có tỉ lệ thấp nhất.

## 4. Ảnh hưởng đến chất lượng gợi ý

NDCG@10 giữ item đã xem (`exclude_seen = false`), `max_len = 2`, `min_confidence` tốt nhất của từng mức. Popularity thuần: 0,00785.

| min_support | 0,002 | 0,001 | 0,0005 | 0,0003 | 0,0002 | 0,00015 | **0,0001** | 0,00005 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Đếm tối thiểu | 45 | 23 | 12 | 7 | 5 | 4 | **3** | 2 |
| NDCG@10 | 0,01323 | 0,01867 | 0,02306 | 0,02843 | 0,03224 | 0,03517 | **0,03728** | 0,03551 |

- NDCG@10 tăng đơn điệu khi hạ min_support từ 0,002 đến 0,0001 rồi giảm ở 0,00005, nên cực đại nằm trong lưới đã quét (đếm tối thiểu 3), không ở biên.
- Cấu hình được chọn: `max_len = 2`, `min_support = 0,0001`, `min_confidence = 0,05` (84.883 luật; 91,9% user có luật khớp): Recall@10 0,06985, NDCG@10 0,03728, cao hơn Popularity 0,02943 (sai số chuẩn 0,00105 theo user).
- Hiệu so với cấu hình đứng thứ hai (cùng `max_len` và `min_support`, `min_confidence` 0,02) là 0,00030 (sai số chuẩn 0,00018), nên `min_confidence` trong khoảng 0,02–0,1 cho kết quả gần nhau; từ 0,2 trở lên NDCG@10 giảm rõ (0,03002 và 0,02407 ở 0,2 và 0,3).
- Lưới của `max_len = 3` và `4` dừng ở min_support 0,0005 (`max_len = 4` ở mức này tạo 683.794 luật ở `min_confidence` 0,02, đỉnh bộ nhớ của cả lưới khoảng 0,67 GB); chưa quét thấp hơn. Tại các mức support dùng chung, NDCG@10 của `max_len = 2` cao hơn hoặc xấp xỉ `max_len = 3, 4` (ví dụ 0,0005: 0,02306 / 0,02226 / 0,02175; riêng 0,001 thì `max_len = 3` cao hơn `max_len = 2` 0,00018). Phần lớn mức tăng đến từ việc hạ min_support, chỉ thực hiện được với `max_len = 2`.

## 5. Giới hạn của kết quả

- Cấu hình được chọn bằng argmax trên 105 cấu hình ở cùng tập validation; mức cải thiện so với Popularity và sai số chuẩn chưa hiệu chỉnh cho việc chọn (winner's curse). Ước lượng không chệch cần chạy test một lần cho cấu hình đã chọn.
- Với đếm tối thiểu 3, mỗi luật dựa trên ít nhất ba giao dịch. Trong 84.883 luật của cấu hình được chọn (cột `count` của `patterns.generate_rules`), 22.011 luật (25,9%) có đúng 3 giao dịch, trung vị là 5, 80,4% có không quá 10 và lớn nhất là 88. Phần lớn luật vì vậy dựa trên ít giao dịch, gần với thống kê đồng xuất hiện item–item hơn là luật mua kèm bền vững. Điểm `confidence × lift` ở support thấp bị chi phối bởi cặp hiếm (lift ≤ 1 / max(sup(X), sup(Y))).
- Lưới của `max_len` khác nhau không cùng tập min_support (xem `DEFAULT_GRID` trong `src/ar_rec_sweep.py`); so sánh giữa các `max_len` chỉ có ý nghĩa tại các mức support dùng chung.
- Chia dữ liệu là leave-last-out theo từng user, không theo thời gian toàn cục; luật được khai phá từ train của mọi user nên có thể chứa tương tác xảy ra sau thời điểm nhãn của một user khác. Quy ước này giống các mô hình khác trong nhóm.
- 11.383 / 22.363 user có ≤ 4 item trong lịch sử; với nhóm này, tỉ lệ có luật khớp ở các mức support của kế hoạch từ 7,0% đến 30,5%, phần còn lại được bổ sung bằng Popularity.

## 6. Tái lập

```powershell
.\venv\Scripts\python.exe -m src.ar_rec_sweep --rankings-output experiments/ar_rec_val_selected/rankings.jsonl
```

Lưới mặc định (105 cấu hình) chạy khoảng 17 phút trên máy 8 GB RAM (đo hai lần: 16 phút 56 giây và 17 phút 31 giây; đỉnh bộ nhớ của tiến trình 0,67 GB). `rankings.jsonl` là danh sách của cấu hình được chọn, theo định dạng bàn giao của bộ chấm chung (`user_id`, `ranked_items`, mỗi user `max(K) + len(seen_item_ids)` item).
