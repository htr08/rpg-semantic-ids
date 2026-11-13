# rpg-semantic-ids

Khai phá cấu trúc ẩn sau Semantic ID: phân cụm, mẫu mua kèm và hiện tượng hub trong hệ gợi ý sinh RPG (KDD 2025).

Dự án học phần *Khai phá & Phân tích dữ liệu*, nhóm 3 thành viên, thực hiện từ 6/11 đến 10/12/2026. Mô hình RPG (Hou và cộng sự, KDD 2025) được dùng làm công cụ để phân tích dữ liệu sản phẩm Amazon Beauty (2014, 5-core: 22.363 user, 12.101 item, 176.139 tương tác).

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
