#!/usr/bin/env bash
# Chạy lại toàn bộ pipeline từ clone sạch. Mỗi bước cập nhật khi script tương ứng hoàn thiện.
set -euo pipefail

# 1. Tải dữ liệu và kiểm tra checksum
# python scripts/download_data.py

# 2. Tiền xử lý, lưu file chỉ số chia vào data/splits/
# python -m src.preprocessing

# 3. Encode mô tả và tạo semantic ID (OPQ, PQ, RQ, random, behavior)
# python scripts/build_semantic_ids.py --config configs/rpg_main.yaml

# 4. Baseline (RecBole)
# python scripts/run_baselines.py --config configs/baselines_recbole.yaml

# 5. RPG (code gốc qua adapter), seed 42 43 44
# python scripts/run_rpg.py --config configs/rpg_main.yaml

# 6. Khai phá mẫu, phân cụm, đồ thị, đánh giá: chạy các notebook trong notebooks/ theo thứ tự đánh số

echo "run_all.sh: chưa có bước nào được kích hoạt."
