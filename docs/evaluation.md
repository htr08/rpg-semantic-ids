# Đánh giá mở rộng của B

Phần này thêm coverage, tỉ lệ long-tail và đa dạng danh mục vào bộ chấm Recall/NDCG
của C. Mọi mô hình xuất ranking về cùng `item_id` của pipeline B rồi dùng một bộ
chấm. Không cần embedding, GPU hoặc adapter để tính các metric này.

## Ý nghĩa và công thức

Với user u, L(u,K) là tối đa K item đầu của ranking. Bộ chấm chung mặc định
giữ item đã xem trong danh sách (`exclude_seen=False`). Khi chấm với bộ lọc, bộ chấm vẫn
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

## Ví dụ tính tay với `--exclude-seen`

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
.\venv\Scripts\python.exe -m src.evaluate --rankings experiments/rpg_s42_validation/rankings.jsonl --model-name rpg_s42 --split validation --ks 5 10 --output-dir experiments/rpg_s42_validation
```

Thay đường dẫn bằng file ranking thực tế. CLI mặc định đọc `data/processed/`
và `data/splits/`. Có thể đổi qua `--processed-dir` và `--splits-dir`.
Dùng `--exclude-seen` khi cần đối chiếu kết quả có lọc.

Trước khi chấm, CLI kiểm tra checksum các đầu ra tiền xử lý theo manifest. Đầu ra:

- `metrics.json`: tên mô hình, bộ chia, điểm tổng hợp, quy tắc tính, quy mô dữ liệu,
  cold-start, số danh sách ngắn, đường dẫn và SHA-256 của ranking/train/items/split/manifest/code chấm.
- `per_user.csv`: mỗi user một dòng, sắp ID tăng dần; có Recall/NDCG và các tỉ lệ,
  đa dạng theo user. Coverage chỉ có ở JSON vì là chỉ số trên toàn tập.

### Paired t-test theo user

Chạy kiểm định hai phía trên NDCG@10 của RPG và SASRec theo từng `user_id`.
Các thư mục phải có `metrics.json` và `per_user.csv` từ cùng bộ chia và cùng
chế độ giữ item đã xem. Với nhiều seed, lấy trung bình NDCG@10 của từng user
trong mỗi mô hình trước khi tính chênh lệch. Mỗi user tạo một cặp quan sát.

```bash
python -m scripts.paired_t_test \
  --left experiments/rpg_s42_test experiments/rpg_s43_test experiments/rpg_s44_test \
  --right experiments/sasrec_paper_pr10/seed_42/test experiments/sasrec_paper_pr10/seed_43/test experiments/sasrec_paper_pr10/seed_44/test
```

Đầu ra cho biết số user, chênh lệch trung bình (RPG trừ SASRec), thống kê t và p-value.
Nếu mọi chênh lệch bằng nhau, thống kê t là `null`. Khi hai bên giống hệt nhau,
p-value ghi 1. Khi chênh lệch là hằng số khác 0, p-value ghi `null`.

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
.\venv\Scripts\python.exe -m scripts.run_baselines --models Pop ItemKNN --splits validation test --output-dir experiments/baselines_no_seen_pinned
.\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Kết quả dùng Python 3.12.10, torch 2.4.1, NumPy 1.26.4, SciPy 1.13.1 và RecBole
1.2.0. Trên cùng ranking đầu vào, Recall/NDCG của bộ chấm chung và RecBole khớp
trong sai số dưới 1e-4 ở cả hai split và hai mô hình. JSON ghi điểm RecBole và sai số.
Đối chiếu với lượt Python 3.13.5, torch 2.8.0+cpu, NumPy 2.2.6 và SciPy 1.15.3,
Pop giữ nguyên số đo. ItemKNN lệch tối đa khoảng 4,5e-5 ở Recall/NDCG và 5,8e-4
ở coverage giữa hai môi trường.

Trên test với bộ chấm chung, ItemKNN đạt NDCG@10 **0,0232** và SASRec-BCE
đạt **0,0224 ± 0,0002** qua ba seed.
[JSON kết quả Popularity và ItemKNN](../reports/baselines_cpu_no_seen.json).

## Tái lập SASRec-BCE

Từ thư mục gốc của dự án, tải mã và dữ liệu Beauty của S³-Rec.
Script kiểm tra SHA-256 của từng file và lưu tại `experiments/sasrec_reference/`.

```bash
python -m scripts.reproduce_sasrec_paper --prepare-only
```

Để tái lập điểm đã ghi, tải ZIP đầu ra Kaggle của B từ GitHub Release
và đặt tại `experiments/sasrec_reference/`.
[Tải `sasrec_paper_beauty_outputs.zip`](https://github.com/htr08/rpg-semantic-ids/releases/download/sasrec-beauty-checkpoints/sasrec_paper_beauty_outputs.zip).
Bản gốc có SHA-256
`566ef7d072605503759a2f0fa6150adc8eb9d087e197c2fbb8007d0b21558f76`.
Kiểm tra bằng `sha256sum` trên Linux hoặc `Get-FileHash -Algorithm SHA256`
trên PowerShell trước khi chấm. Trong ZIP, `summary.json` ghi kết quả
ba seed và mỗi thư mục `seed_<seed>/` chứa checkpoint cùng log huấn luyện.

Để chạy lại thí nghiệm, có thể tạo ZIP mới từ repo trên Kaggle với GPU và
Internet đã bật. Điểm và trọng số của lượt mới có thể khác bản gốc. Với PyTorch,
NumPy, SciPy và tqdm trong runtime, chạy lệnh sau. Mỗi seed chạy từ khởi tạo
ngẫu nhiên trong thư mục riêng, dùng BCE, batch 256, learning rate 0,001,
lịch sử tối đa 50, giới hạn 200 epoch. S³-Rec chọn checkpoint bằng
validation NDCG@20 với patience 10 và đưa điểm item đã xem về 0 khi chấm.

```bash
python -m scripts.reproduce_sasrec_paper --archive experiments/sasrec_reference/retrained.zip
```

Sau khi chuẩn bị dữ liệu của dự án theo [hợp đồng dữ liệu](data_contract.md),
chấm ZIP gốc trên CPU với bộ chấm chung. Đổi `--archive` nếu chấm ZIP mới.

```bash
python -m scripts.evaluate_sasrec_paper \
  --archive experiments/sasrec_reference/sasrec_paper_beauty_outputs.zip \
  --output-dir experiments/sasrec_paper_pr10
```

Bộ chấm S³-Rec gốc cho test NDCG@10 **0,0317 ± 0,0001**, so với mốc
**0,0318** ở Table 2. Điểm này dùng để đối chiếu paper.
[Mã nguồn S³-Rec đã pin](https://github.com/aHuiWang/CIKM2020-S3Rec/tree/49a890dcb516c9ed3126564495234ea74d5cd566) ·
[Kết quả theo seed](../reports/sasrec_paper_common.json).

## Xuất ranking test từ checkpoint RPG

Chuẩn bị code RPG gốc tại commit `7dcf95c` và các phụ thuộc của repo đó như trong README.
Giải nén gói A bàn giao, giữ cấu trúc `cache/`, `ckpt/` và `results/` bên dưới thư mục
`--root`. Trong cache cần có hai file raw Beauty, cache dữ liệu đã xử lý và `.sem_ids`
tương ứng với checkpoint. Giữ `results/<run_gốc>.json` để script lấy best epoch và
điểm validation lúc chọn checkpoint. Chuẩn bị dữ liệu nhóm theo [hợp đồng dữ liệu](data_contract.md).

```bash
python -m scripts.run_rpg \
  --repo experiments/rpg_upstream --root experiments/rpg_ranking_export --seed 42 \
  --checkpoint experiments/rpg_ranking_export/ckpt/main_s42/main_s42-run_rpg.py-Nov-14-2026_16-24-f3eab8.pth \
  --run_id main_s42_rankings \
  --rankings-output experiments/rpg_ranking_export/main_s42/test/rankings.jsonl

python -m src.evaluate \
  --rankings experiments/rpg_ranking_export/main_s42/test/rankings.jsonl \
  --model-name main_s42 --split test --ks 5 10 \
  --output-dir experiments/rpg_ranking_export/main_s42/test
```

Với checkpoint hybrid, truyền thêm `--sent_emb_model=x/hyb` và chọn đúng file `.pth`,
seed và đường dẫn output của run. Chế độ export chạy test trên một tiến trình.
Script đối chiếu mapping, lịch sử và nhãn test trước khi chạy, lấy top-10 có giữ
item đã xem từ chính lượt chấm của code RPG gốc và kiểm tra Recall/NDCG bằng bộ
chấm chung trước khi ghi JSONL. File `results/<run_id>.json` của lượt export ghi
checksum đầu vào và ranking, thiết bị, phiên bản PyTorch và batch size.

Lượt chấm lại có thể lệch nhẹ với điểm A ghi lúc train. Đồ thị giải mã của RPG
lấy điểm bắt đầu ngẫu nhiên, nên trạng thái RNG ở lượt chấm checkpoint mới có
thể khác lúc vừa train xong. So sánh hai bộ chấm trên ranking của cùng một lượt chạy.
Xem [kết quả và checksum của sáu lượt xuất OPQ và hybrid](../reports/rpg_ranking_exports.json).

Xem [ablation và đồ thị giảm hub](../reports/rpg_graph_ablation.md) để chấm lại
checkpoint OPQ khi thay đổi cách tìm ứng viên.

Baseline SASRec dùng checkpoint từ S³-Rec, chọn theo validation NDCG@20 với
patience 10. Khi chấm bằng bộ chấm dự án, `scripts.evaluate_sasrec_paper` quy đổi
item ID, giữ item đã xem và xuất kết quả theo user theo mapping B.
Xem [JSON kết quả SASRec](../reports/sasrec_paper_common.json) và
[JSON kết quả Popularity và ItemKNN](../reports/baselines_cpu_no_seen.json).
