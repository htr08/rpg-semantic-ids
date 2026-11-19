# Nhật ký quyết định

Ghi lại câu trả lời của giảng viên và các quyết định của nhóm. Mọi câu hỏi cho giảng viên được gom lại và hỏi một lần.

| Ngày | Nội dung | Nguồn | Ảnh hưởng |
|---|---|---|---|
| 13/11/2026 | Huấn luyện RPG bằng code gốc `RPG_KDD2025` (submodule); không tự viết mô hình, loss, vòng train. Việc kỹ thuật phía mô hình chỉ gồm adapter nạp semantic ID/embedding (hạn 18/11). | Nhóm, Project Charter cập nhật | Trọng tâm chuyển sang khai phá và phân tích dữ liệu. |
| 13/11/2026 | Mốc tham chiếu là NDCG@10 = 0,0429 (Table 4, `sentence-t5-base`), không dùng 0,0464 (Table 2, `text-embedding-3-large`). | Nhóm | KPI mô hình nền: 0,038–0,045. |
| 15/11/2026 | Văn bản mô tả: giữ nguyên cách ghép của code gốc (`_extract_meta_sentences` trong `genrec/datasets/AmazonReviews2014/dataset.py`), không thêm bước fallback riêng. Code gốc ghép 6 trường title, price, brand, feature, categories, description và bỏ qua trường thiếu: 934 item thiếu description dùng title + price + brand + feature + categories; 7 item thiếu cả title và description dùng các trường còn lại (không item nào thiếu categories). Không item nào có văn bản rỗng. | A, trả lời câu hỏi của B về 7 item trong `reports/data_quality.md` | Đáp ứng quy tắc "thiếu description dùng title + category" của kế hoạch; giữ nguyên embedding, semantic ID và kết quả 3 seed RPG (NDCG@10 = 0,0420 ± 0,0018), không train lại. Báo cáo ghi ở phần hạn chế: 941 item (7,8%) chỉ có văn bản ngắn. |
| 15/11/2026 | Thứ tự item dùng chung: `id_mapping.json` của pipeline B là chuẩn. Đã kiểm tra `id2item` và `id2user` của code gốc khớp từng vị trí với B (chạy pipeline B trên chính file thô code gốc tải; `id_mapping.json` tái lập đúng checksum trong `manifest.json`). | A, [notebook Colab v4](https://colab.research.google.com/drive/1a386RxcIBFD25EqsLYHgjNCVTyf-KtmH), cell 8 | Embedding và semantic ID của A dùng thẳng cho B, C: hàng i ↔ `item_id` i + 1, không cần bảng chuyển đổi. |
| | _Chờ trả lời giảng viên: được dùng code gốc không; chấm theo kỹ thuật trong đề cương hay chất lượng chung._ Nếu chưa có phản hồi, quyết định lại ngày 20/11. | Giảng viên | |
