# Đánh giá mở rộng của B

Phần này thêm coverage, tỉ lệ long-tail và đa dạng danh mục vào bộ chấm Recall/NDCG
của C. Mọi mô hình xuất ranking về cùng `item_id` của pipeline B rồi dùng một bộ
chấm. Không cần embedding, GPU hoặc adapter để tính các metric này.

## Ý nghĩa và công thức

Với user u, L(u,K) là tối đa K item đầu của ranking. Phép so sánh bằng bộ chấm
chung giữ item đã xem trong danh sách. Khi đối chiếu một lần chạy cũ có lọc, bộ chấm vẫn
giữ nhãn nếu đó là sản phẩm mua lại.

| Chỉ số | Cách tính | Câu hỏi trả lời |
|---|---|---|
| `recall@K` | Trung bình theo user của việc nhãn có trong top-K | Có gợi ý trúng sản phẩm tiếp theo không? |
| `ndcg@K` | Nhãn ở hạng r cho điểm 1/log2(r+1); trượt cho 0 | Sản phẩm đúng có đứng gần đầu không? |
| `coverage@K` | Số item khác nhau trong hợp L(u,K) / số item toàn catalog | Hệ thống có chỉ lặp lại một nhóm ít sản phẩm không? |
| `long_tail_ratio@K` | Trung bình theo user của số item tail trong L(u,K) / độ dài L(u,K) | Các sản phẩm ít phổ biến có cơ hội được gợi ý không? |
| `category_diversity@K` | Trung bình khoảng cách Jaccard giữa các cặp item có nhãn trong từng danh sách, rồi trung bình theo user đủ điều kiện | Một danh sách có nhiều loại sản phẩm khác nhau không? |

Coverage là độ phủ **sản phẩm trên toàn bộ user**, không phải phần trăm user được
gợi ý. Mẫu số gồm cả item có tần suất train bằng 0, lấy từ `items.jsonl`.
Khi so sánh coverage giữa mô hình phải dùng cùng tập user, catalog, K và quy tắc lọc.

### Head và long-tail

1. Đếm mọi tương tác, kể cả tương tác lặp, trong `train_sequences.jsonl`.
2. Sắp toàn catalog theo tần suất giảm dần; bằng tần suất thì theo `item_id` tăng dần.
3. Head là `ceil(0.2 * số item)` đầu; tail là phần còn lại, kể cả item chưa có trong train.

Quy tắc giống EDA: Beauty có 12.101 item, head 2.421 item và tail 9.680 item.
Không dùng nhãn hoặc lịch sử validation/test để đếm độ phổ biến, kể cả khi chấm test.

### Đa dạng danh mục

Metadata Amazon có thể có nhiều đường dẫn danh mục. Với một sản phẩm, S(i) là tập
các **đường dẫn đầy đủ đến lá** trong `metadata.categories`; nhãn trùng được gộp.
Ví dụ `['Beauty', 'Hair Care', 'Brushes']` khác
`['Beauty', 'Makeup', 'Brushes']`, dù tên lá đều là Brushes.

Khoảng cách giữa hai sản phẩm:

```text
d(i,j) = 1 - |S(i) giao S(j)| / |S(i) hợp S(j)|
diversity(u,K) = trung bình d(i,j) trên mọi cặp item có nhãn trong L(u,K)
```

Cùng tập danh mục: 0; không chung đường dẫn: 1; giao một phần: nằm giữa 0 và 1.
Quy tắc dùng nhiều nhãn này phục vụ đo đa dạng; không tự quyết định nhãn đơn để
C tính purity/NMI. Đây là khác biệt danh mục, không phải khoảng cách embedding,
và không đo mức độ gần nhau giữa hai nhánh trong cây danh mục.

User có ít hơn hai item có nhãn không đo được đa dạng: giá trị per-user là NaN
trong API, ô trống trong CSV. Tổng hợp chỉ lấy user đủ điều kiện; nếu không có
user nào thì JSON ghi `null`, không ghi điểm 0 hoặc JSON NaN.

### Chỉ số kiểm tra dữ liệu đi kèm

| Khóa | Ý nghĩa |
|---|---|
| `category_label_coverage@K` | Trung bình theo user của tỉ lệ item được gợi ý có nhãn danh mục hợp lệ |
| `category_diversity_users@K` | Số user có ít nhất hai item có nhãn để tính đa dạng |
| `recommendation_fill@K` | Trung bình theo user của số item còn lại trong danh sách / K |
| `counts.short_ranking_users` | Số user bị thiếu item tại từng K trong JSON báo cáo |

Danh sách ngắn vẫn được chấm và có cảnh báo. Long-tail và độ phủ nhãn chia cho
số item thực có; danh sách rỗng cho hai tỉ lệ này bằng 0. Vì vậy phải đọc chúng
cùng `recommendation_fill`, không so trực tiếp một mô hình trả thiếu item với
mô hình có đầy đủ top-K. Không tự bổ sung item cho mô hình.

Đa dạng hoặc coverage cao không tự chứng minh mô hình tốt: gợi ý ngẫu nhiên cũng
có thể trải rộng nhưng trượt nhãn. Luôn đọc cùng Recall/NDCG.

## Ví dụ tính tay

Catalog gồm 5 item; head = {1}. Danh mục: item 1 và 2 thuộc A, item 3 thuộc B,
item 4 thuộc cả A và B, item 5 thiếu danh mục.

- User 1: ranking `[1,2,3,4,5]`, đã thấy 1, nhãn 2 → top-2 `[2,3]`.
- User 2: ranking `[1,4,2,3,5]`, chưa thấy item nào, nhãn 4 → top-2 `[1,4]`.

Coverage = 4/5 = 0,8; long-tail = (1 + 1/2)/2 = 0,75; đa dạng = (1 + 1/2)/2 = 0,75.
Recall@2 = 1; NDCG@2 = (1 + 1/log2(3))/2 ≈ 0,8155. Ví dụ này nằm trong unit test.

## Định dạng bàn giao ranking của A và C

JSONL, một dòng cho một user của bộ chia được chấm:

```json
{"user_id": 1, "ranked_items": [103, 25, 84, 16, 92]}
```

- `user_id`, `item_id` là số nguyên dương của mapping B; không dùng ASIN, token,
  ID RecBole nội bộ hoặc padding 0. Mô hình phải chuyển đổi ID trước khi xuất.
- Item xếp theo điểm giảm dần, không lặp. Bộ chấm không đọc score hoặc tự sắp lại.
- Để chấm giữ item đã xem, xuất ít nhất `max(K)` item. Để chấm thêm chế độ có lọc,
  xuất ít nhất `max(K) + len(seen_item_ids)` item trước lọc nếu catalog đủ lớn.
- Mỗi user đúng một dòng. CLI yêu cầu tập user khớp chính xác bộ chia.
- API chỉ kiểm tra prefix đủ dùng để tránh duyệt 12.101 item cho từng user;
  CLI kiểm tra kiểu ID và item lặp trên toàn file. Các ID sử dụng để chấm phải có trong catalog.
- Dùng validation để chọn cấu hình; test dành cho cấu hình đã chọn. Cùng một ranking
  không mặc nhiên dùng được cho cả validation và test vì lịch sử đầu vào khác nhau.

## Chạy bộ chấm

Sau khi có môi trường ảo và dữ liệu khớp manifest:

```powershell
.\venv\Scripts\python.exe -m src.evaluate --rankings experiments/rpg_s42_validation/rankings.jsonl --model-name rpg_s42 --split validation --ks 5 10 --no-exclude-seen --output-dir experiments/rpg_s42_validation
```

Thay đường dẫn bằng file ranking thực tế; lệnh trên minh họa cách chấm RPG,
không có sẵn file RPG trong repo. CLI mặc định đọc `data/processed/` và `data/splits/`;
có thể đổi qua `--processed-dir` và `--splits-dir`. Dùng `--no-exclude-seen`
để so sánh bằng bộ chấm chung. Bỏ cờ này để tái chấm theo mặc định có lọc của CLI cũ.

Trước khi chấm, CLI kiểm tra checksum các đầu ra tiền xử lý theo manifest. Đầu ra:

- `metrics.json`: tên mô hình, bộ chia, điểm tổng hợp, quy tắc tính, quy mô dữ liệu,
  cold-start, số danh sách ngắn, đường dẫn và SHA-256 của ranking/train/items/split/manifest/code chấm.
- `per_user.csv`: mỗi user một dòng, sắp ID tăng dần; có Recall/NDCG và các tỉ lệ,
  đa dạng theo user. Coverage chỉ có ở JSON vì là chỉ số trên toàn tập.

CSV giữ điểm theo user để làm paired t-test sau này.

### Cold-start theo tần suất sản phẩm

Với mỗi user, lấy item đúng trong split đang chấm và đếm số tương tác của item
đó trong `train_sequences.jsonl`. Kết quả `cold_start.groups` ghi số user và
NDCG trung bình cho mỗi K được yêu cầu trong bốn nhóm **0–5, 6–10, 11–15,
16–20**. User có item đúng ngoài top-K đóng góp điểm 0. Nhóm trống có
NDCG là `null`. `users_above_20` ghi số user có item đúng xuất hiện hơn
20 lần trong train. Chấm các mô hình trên cùng split và cùng chế độ lọc để đối chiếu.

[Kết quả cold-start trên Beauty](../reports/cold_start_beauty.json).

## Dùng từ Python

```python
from src.evaluate import build_evaluation_context, evaluate, load_rankings, load_split
from src.preprocessing import read_records

context = build_evaluation_context(
    (row for _, row in read_records('data/processed/train_sequences.jsonl')),
    (row for _, row in read_records('data/processed/items.jsonl')),
)
result = evaluate(
    load_rankings('experiments/my_model/rankings.jsonl'),
    load_split('data/processed/validation.jsonl'),
    context=context,
)
```

Không truyền `context` thì `evaluate` và `per_user_metrics` giữ các khóa Recall/NDCG
ban đầu. API trực tiếp không kiểm tra manifest; CLI thực hiện bước này.

## Chạy Popularity và ItemKNN trên Beauty

```powershell
.\venv\Scripts\python.exe -m scripts.run_baselines --models Pop ItemKNN --splits validation test --output-dir experiments/baselines_no_seen_history50
.\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Ranking và bảng per-user nằm trong `experiments/`.
[JSON kết quả Popularity và ItemKNN](../reports/baselines_cpu_no_seen.json).

## Tái lập SASRec-BCE

Dùng dữ liệu Beauty và `run_finetune_full.py` từ commit S³-Rec đã pin.
Chạy lần lượt seed 42, 43 và 44 với thư mục đầu ra riêng. Ví dụ chạy seed 42
từ thư mục gốc của S³-Rec.

```bash
python run_finetune_full.py --data_name Beauty --ckp 0 --seed 42 --epochs 200 --batch_size 256 --lr 0.001 --max_seq_length 50 --output_dir results/seed_42
```

Dùng thư mục đầu ra mới để khởi tạo từ đầu. Script chọn checkpoint theo
validation NDCG@20 với patience 10. Bộ chấm gốc đưa điểm item đã xem về 0.
[Mã nguồn S³-Rec đã pin](https://github.com/aHuiWang/CIKM2020-S3Rec/tree/49a890dcb516c9ed3126564495234ea74d5cd566)
· [Kết quả theo seed](../reports/sasrec_paper_common.json).

## Đối chiếu với điểm RPG của A

Code RPG gốc tại commit `7dcf95c` giữ item đã xem trong `generate`/`evaluate`.
Để đối chiếu các run đã log, cần ranking của A theo cùng mapping và bộ chia.
Baseline SASRec dùng checkpoint từ S³-Rec, chọn theo validation NDCG@20 với
patience 10. Khi chấm bằng bộ chấm dự án, `scripts.evaluate_sasrec_paper` quy đổi
item ID, giữ item đã xem và xuất kết quả theo user theo mapping B.
Xem [JSON kết quả SASRec](../reports/sasrec_paper_common.json) và
[JSON kết quả Popularity và ItemKNN](../reports/baselines_cpu_no_seen.json).
