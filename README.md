# rpg-semantic-ids

Khai phá cấu trúc ẩn sau Semantic ID: phân cụm, mẫu mua kèm và hiện tượng hub trong hệ gợi ý sinh RPG (KDD 2025).

Dự án học phần *Khai phá & Phân tích dữ liệu*, nhóm 3 thành viên, thực hiện từ 6/11 đến 10/12/2026. Mô hình RPG (Hou và cộng sự, KDD 2025) được dùng làm công cụ để phân tích dữ liệu sản phẩm Amazon Beauty (2014, 5-core: 22.363 user, 12.101 item, 198.502 tương tác gốc; 176.139 tương tác trong train + validation).

Kế hoạch đầy đủ: [docs/project_plan.md](docs/project_plan.md).

## Câu hỏi nghiên cứu

| Mã | Nội dung |
|---|---|
| RQ1 | Semantic ID (OPQ, PQ, RQ, ngẫu nhiên) có phải là một phép phân cụm tốt không; chất lượng cụm có đi cùng NDCG không. |
| RQ2 | Mẫu mua kèm (FP-Growth, PrefixSpan) có được phản ánh trong semantic ID không; đưa tín hiệu hành vi vào semantic ID có cải thiện gợi ý không. |
| RQ3 | Đồ thị kNN dùng để giải mã có hiện tượng hub không; hub có bị gợi ý quá mức không; cộng đồng Louvain có trùng với danh mục không. |
| RQ4 | Coverage, long-tail, cold-start và phân khúc người dùng: mô hình tốt và kém với nhóm nào. |

## Phần của tác giả và phần tự viết

| Thành phần | Nguồn |
|---|---|
| `third_party/RPG_KDD2025/` | Code gốc của tác giả bài báo (git submodule, không sửa). Huấn luyện RPG chạy bằng `main.py` của repo này. |
| RecBole | Thư viện bên ngoài, dùng cho baseline Popularity, ItemKNN, SASRec. |
| `adapters/` | Tự viết: nạp semantic ID hoặc embedding do nhóm tạo vào code gốc. |
| `src/`, `scripts/`, `notebooks/` | Tự viết: pipeline dữ liệu, tokenizer, đánh giá, khai phá mẫu, phân cụm, phân tích đồ thị. |

## Cấu trúc thư mục

```text
.github/workflows/   CI: pytest cho src/ và adapters/
configs/             cấu hình từng thí nghiệm (*.yaml) và gpu_budget.md
data/raw/            dữ liệu thô (không commit)
data/processed/      chuỗi, embedding, semantic_ids.npy (không commit)
data/splits/         chỉ số leave-last-out dùng chung (commit)
third_party/         code gốc RPG_KDD2025 (submodule)
adapters/            adapter nạp semantic ID / embedding vào code gốc
notebooks/           EDA, semantic ID như phân cụm, khai phá mẫu, so sánh phân cụm, cold-start, hubness
scripts/             tải dữ liệu, tạo semantic ID, chạy baseline và RPG, run_all.sh
src/                 module tự viết (preprocessing, tokenizer, evaluate, patterns, graph, ...)
tests/               unit test
experiments/log.csv  nhật ký thí nghiệm (run, config, seed, giờ GPU, kết quả)
reports/             báo cáo, hình, bảng mapping kiến thức, quyết định của nhóm
docs/                tài liệu kế hoạch
```

## Thiết lập môi trường

Yêu cầu Python 3.10 trở lên. Mỗi thành viên dùng môi trường ảo riêng:

```bash
python -m venv venv
# Windows: venv/Scripts/activate      # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Trên Kaggle/Colab, chạy `pip install -r requirements.txt` ở cell đầu notebook.

Thêm code gốc làm submodule (URL lấy từ repo của tác giả):

```bash
git submodule add <URL repo RPG_KDD2025> third_party/RPG_KDD2025
git submodule update --init --recursive
```

Xóa output notebook trước khi commit:

```bash
nbstripout --install
```

## Tải dữ liệu Beauty 2014 (phần B)

Script tải dữ liệu chỉ dùng thư viện chuẩn Python, chạy được trước khi cài các thư viện mô hình:

```bash
python scripts/download_data.py
python scripts/download_data.py --verify-only
```

Hai file được lưu nguyên dạng nén trong `data/raw/` (đã gitignore):

| File | Dung lượng nén | Số bản ghi gốc |
|---|---:|---:|
| `reviews_Beauty_5.json.gz` | 44.819.276 byte | 198.502 review |
| `meta_Beauty.json.gz` | 99.156.148 byte | 259.204 sản phẩm |

Nguồn là [Amazon Reviews 2014 của McAuley Lab](https://mcauleylab.ucsd.edu/public_datasets/data/amazon/index_2014.html),
qua cùng URL Stanford mà [loader RPG gốc](https://github.com/facebookresearch/RPG_KDD2025/blob/main/genrec/datasets/AmazonReviews2014/dataset.py) sử dụng.
Đây là Beauty 2014, không phải All Beauty 2018/2023. Đã kiểm tra file reviews:
22.363 user, 12.101 item, 198.502 tương tác; mỗi user và item đều có ít nhất 5 tương tác.
Mốc 176.139 trong kế hoạch bằng 198.502 trừ 22.363 (bỏ một tương tác mỗi user).
Nếu giữ riêng cả validation và test theo kế hoạch, phần train đầy đủ còn 153.776 tương tác
trước khi giới hạn độ dài đầu vào. Cần ghi rõ phạm vi đếm khi đối chiếu Table 1;
không lọc thêm dữ liệu chỉ để ép số tương tác thô về 176.139.

Script kiểm tra kích thước, SHA-256, giải nén toàn bộ để kiểm tra CRC gzip và đếm bản ghi.
File đang tải có đuôi `.part`, chỉ đổi sang tên chính thức sau khi kiểm tra thành công;
file đã có được kiểm tra lại trước khi bỏ qua tải. Nếu file đã có bị lỗi, script dừng để người dùng kiểm tra.
Lỗi mạng/I/O được thử lại tối đa 3 lần; sai checksum, kích thước hoặc số bản ghi thì dừng ngay.
`--verify-only` kiểm tra cục bộ, không cần mạng.

Tham chiếu SHA-256 nằm trong `configs/data_sources.json`, được nhóm ghi nhận từ lần tải HTTPS đầu tiên,
không phải checksum do nhà phát hành công bố. Cờ `--record-checksums` chỉ dùng để ghi các checksum còn thiếu
khi thiết lập manifest lần đầu; không thay thế checksum đã ghi. Có thể đổi thư mục lưu bằng `--output-dir`.

## Tiền xử lý và bộ chia dùng chung (phần B)

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing
.\.venv\Scripts\python.exe -m src.preprocessing --verify-only
```

Có thể dùng `python -m src.preprocessing` khi đang kích hoạt môi trường ảo.
Pipeline lọc 5-core đến hội tụ, sắp timestamp ổn định, ánh xạ ID và tách train/validation/test.
Train đầy đủ được giữ cho khai phá mẫu; lịch sử đầu vào đánh giá lấy tối đa 50 item cuối.
Metadata được xuất theo cùng mapping item, chưa ghép văn bản hay encode.

- `data/processed/`: mapping, bảng tương tác, chuỗi train, validation/test, metadata và thống kê (gitignore).
- `data/splits/leave_last_out.csv`: chỉ số chia theo tương tác dùng chung (commit).
- `data/splits/manifest.json`: nguồn dữ liệu, cấu hình và checksum các đầu ra (commit).

Quy ước từng file, cách đọc bộ chia và đầu vào cho A/B/C: [docs/data_contract.md](docs/data_contract.md).
`scripts/run_all.sh` chạy ba bước: tải dữ liệu, tiền xử lý và EDA; các bước mô hình/khai phá chưa được kích hoạt.

## EDA tuần 1 của B

```powershell
.\venv\Scripts\python.exe -m src.eda
.\venv\Scripts\python.exe -m pytest -q
```

Môi trường EDA đã kiểm tra với Python 3.12; dùng các bản numpy/pandas/matplotlib/ipykernel đã pin trong `requirements.txt`.
Mở [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb), chọn kernel từ `venv` và Run All.
Notebook dùng chung `src/eda.py` với CLI; output được giữ trống khi commit.

- [Báo cáo EDA tuần 1](reports/data_quality.md): độ dài chuỗi, long-tail, độ thưa, thống kê thiếu metadata và ví dụ mô tả; hai hình trong `reports/figures/`, số liệu trong `reports/eda_summary.json`.

## Kiểm thử

```bash
pytest
```

## Quy ước làm việc

- Branch theo dạng `<role>/<task>`; không commit trực tiếp lên `main`; mọi thay đổi qua Pull Request, có ít nhất 1 approve và CI thành công.
- Commit message theo Conventional Commits (`feat:`, `fix:`, `exp:`, `docs:`).
- Tên file bằng tiếng Anh, không dấu, snake_case; nội dung báo cáo, comment và docstring bằng tiếng Việt.
- Seed cố định {42, 43, 44}; mọi thí nghiệm có file config riêng và một dòng trong `experiments/log.csv`.
- Không commit dữ liệu thô, embedding, checkpoint.

Chi tiết: mục 8 của [docs/project_plan.md](docs/project_plan.md).
