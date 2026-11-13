# Dữ liệu dùng chung sau tiền xử lý

Pipeline tự viết của B chạy bằng thư viện chuẩn Python:

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing
.\.venv\Scripts\python.exe -m src.preprocessing --verify-only
```

Cấu hình ở `configs/preprocessing.json`: `min_core = 5`, `max_history = 50`.
Đầu vào là hai file Beauty 2014 trong `data/raw/`; SHA-256 phải khớp `configs/data_sources.json`.
Chạy lại với cùng đầu vào và cấu hình tạo các file giống nhau từng byte, không phụ thuộc seed.

## Quy tắc tiền xử lý

1. Giữ `reviewerID`, `asin`, `unixReviewTime` và số dòng vật lý trong file gốc. Không dùng rating; giữ nguyên tương tác lặp.
2. Lọc đồng thời user và item có ít hơn 5 tương tác, lặp đến khi không còn thay đổi.
3. Duyệt user theo lần xuất hiện đầu trong file; với mỗi user, sắp theo timestamp tăng dần, trùng timestamp thì theo số dòng gốc.
4. User ID bắt đầu từ 1. Item ID bắt đầu từ 1 theo lần gặp đầu khi duyệt các chuỗi đã sắp xếp. ID 0 dành cho padding.
5. Với mỗi user, giữ item cuối làm test, item kế cuối làm validation, tất cả tương tác trước đó làm train.
6. Giữ toàn bộ chuỗi train cho khai phá mẫu. Chỉ lấy 50 item cuối khi tạo lịch sử đầu vào của mô hình.

K-core được tính trên toàn bộ tập gốc trước khi chia, theo kế hoạch. Tần suất Popularity, ItemKNN,
FP-Growth, PrefixSpan và embedding hành vi phải tính từ **train**. Vocab và metadata bao gồm toàn bộ catalog sau lọc.

## File trong `data/processed/`

Thư mục này đã gitignore; mỗi thành viên tự tạo lại bằng pipeline.

| File | Nội dung |
|---|---|
| `id_mapping.json` | `user2id`, `item2id`, `id2user`, `id2item`; phần tử 0 trong hai danh sách ngược là `null` |
| `interactions.csv` | `interaction_id,user_id,item_id,timestamp,source_line`; tương tác đã lọc và sắp theo user/thời gian |
| `train_sequences.jsonl` | Mỗi dòng gồm `user_id`, `item_ids`; chuỗi train đầy đủ, chưa cắt 50 |
| `validation.jsonl` | `user_id`, `history`, `target`, `seen_item_ids`, `target_in_history` |
| `test.jsonl` | Cùng cấu trúc validation; lịch sử test bao gồm tương tác validation |
| `items.jsonl` | `item_id`, `asin`, `metadata_missing`, `metadata`; theo item ID tăng dần |
| `preprocessing_stats.json` | Số lượng trước/sau lọc, từng vòng lọc, số mẫu chia, số lịch sử bị cắt và metadata thiếu |

`history` giữ tối đa 50 item gần nhất. `seen_item_ids` giữ mọi item đã thấy trong lịch sử tương ứng,
kể cả ngoài cửa sổ 50, để hàm đánh giá có đủ dữ liệu loại ứng viên đã xem.
Nhãn validation không nằm trong lịch sử validation; nhãn test không nằm trong lịch sử test **ở cùng vị trí tương tác**.
Nếu cùng một sản phẩm được mua nhiều lần, ID nhãn có thể từng xuất hiện trước đó;
`target_in_history` đánh dấu trường hợp này để hàm đánh giá xử lý rõ ràng, pipeline không tự xóa tương tác.

Metadata giữ nguyên các trường có mặt: `title`, `price`, `brand`, `feature`, `categories`, `description`.
Ảnh và các liên kết mua kèm không được xuất. Bước này chưa ghép mô tả hay encode:
A dùng cùng mapping để áp dụng cách ghép văn bản của code RPG và quy tắc fallback đã thống nhất.
Metadata thiếu vẫn có một dòng rỗng và cờ `metadata_missing`, giữ đúng thứ tự item.

## File trong `data/splits/`

`leave_last_out.csv` và `manifest.json` là hai file cần commit và dùng chung.

Mỗi dòng CSV có:

```text
user_id,train_start,train_stop,validation_index,test_index
```

Các chỉ số là giá trị `interaction_id` bắt đầu từ 0 trong `interactions.csv`, không tính header.
Train là khoảng nửa mở `[train_start, train_stop)`. `train_stop == validation_index`,
`test_index == validation_index + 1`. Không dùng số dòng vật lý của file raw để đọc bộ chia này;
`source_line` trong bảng tương tác chỉ dùng để truy ngược về raw.

`manifest.json` ghi quy ước chỉ số, cấu hình, checksum hai file raw và checksum tất cả đầu ra.
Lệnh `--verify-only` kiểm tra 8 file dữ liệu và bộ chia khớp manifest, không dùng mạng.
Các file phải cùng một lần sinh dữ liệu; không ghép mapping hoặc bộ chia từ nguồn khác.
`.gitattributes` giữ LF cho JSON/CSV để Git không đổi byte và làm lệch checksum khi checkout trên Windows.

## Cách dùng cho từng thành viên

- **A:** đọc `items.jsonl` theo thứ tự; hàng embedding thứ `item_id - 1` tương ứng item đó nếu không lưu hàng padding. Khi adapter dành hàng 0 cho padding, phải chuyển đổi tường minh theo mapping.
- **B:** Popularity/ItemKNN và thống kê long-tail chỉ dùng `train_sequences.jsonl`; chấm trên `validation.jsonl` hoặc `test.jsonl`. Lịch sử test được phép có validation, trọng số mô hình và thống kê train vẫn dùng train.
- **C:** FP-Growth lấy tập item duy nhất từ mỗi chuỗi train đầy đủ. PrefixSpan lấy danh sách `item_ids` nguyên thứ tự. Dùng `items.jsonl` để tra tên, brand và category.

Ví dụ đọc train và sinh tiền tố cho mô hình tuần tự:

```python
import json
from src.preprocessing import iter_training_examples

with open("data/processed/train_sequences.jsonl", encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        transaction = set(row["item_ids"])  # FP-Growth
        for example in iter_training_examples(row["item_ids"], max_history=50):
            history, target = example["history"], example["target"]
            # Đưa từng tiền tố train vào adapter/baseline; không thêm nhãn validation/test.
```

## Đối chiếu số lượng Beauty

Đã chạy pipeline trên file thô có 198.502 tương tác, 22.363 user, 12.101 item.
Không có tương tác bị loại thêm bởi 5-core; kết quả thực tế:

| Phạm vi đếm | Tương tác |
|---|---:|
| Toàn bộ sau 5-core | 198.502 |
| Train | 153.776 |
| Validation | 22.363 |
| Test | 22.363 |
| Train + validation | 176.139 |

Tất cả 12.101 item có metadata. Không có nhãn validation/test trùng item đã thấy trước đó
trong lịch sử của cùng user. Chuỗi train dài nhất có 202 tương tác và được giữ nguyên;
136 lịch sử validation và 144 lịch sử test cần cắt về 50 khi tạo đầu vào mô hình.
Đã kiểm tra toàn bộ 198.502 tương tác được chia đúng một lần, lịch sử theo thứ tự thời gian
và nhãn/seen-item khớp bộ chia. Các kiểm thử kèm pipeline đều đạt.

Mốc 176.139 trong kế hoạch khớp phép đếm train + validation. Chưa dùng phép đếm này để khẳng định
định nghĩa thống kê trong paper; khi báo cáo phải nêu rõ phạm vi và đối chiếu lại Table 1.
