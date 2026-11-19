# Quy ước đánh giá — quyết định nhóm 19/11/2026

Nhóm thống nhất **không lọc sản phẩm đã tương tác khỏi danh sách ứng viên**.
Quyết định này thay thế yêu cầu loại item đã xem trong kế hoạch và Project Charter trước đó.

## Quy tắc dùng chung

1. Dùng mapping và bộ chia leave-last-out của B. Fit mô hình và thống kê độ phổ biến
   chỉ từ train; validation dùng chọn cấu hình/checkpoint, test dùng chấm cấu hình đã chọn.
2. Tập ứng viên là toàn catalog theo mapping, gồm cả item user đã thấy trong lịch sử.
   Không dùng `seen_item_ids` hoặc `history` để mask điểm hay xóa ứng viên trước khi lấy top-K.
   Padding ID 0 không phải sản phẩm và không được đưa vào danh sách gợi ý.
3. Chấm Recall/NDCG tại K = 5, 10 từ thứ tự dự đoán của mô hình. Các chỉ số mở rộng
   như coverage, long-tail, đa dạng cũng dùng chính danh sách top-K không lọc đó.
4. Áp dụng cùng quy tắc cho RPG, Popularity, ItemKNN, SASRec và AR-Rec, cả ở bước
   chọn checkpoint trên validation lẫn chấm test. Không lấy mẫu negative để chấm ranking.
5. Nhãn mua lại vẫn là ứng viên bình thường. Không thay đổi hoặc xóa tương tác train,
   nhãn, mapping hay bộ chia để phù hợp quy tắc đánh giá.

Ví dụ: user từng tương tác A, B; ranking là `[A, C, B, D, E]`, nhãn D. Top-3
được chấm là `[A, C, B]`, nên Recall@3 = 0. Không bỏ A/B để đẩy D vào top-3.

`seen_item_ids` và `target_in_history` vẫn giữ trong dữ liệu cho truy vết, phân tích
và kiểm tra chất lượng; quyết định này không yêu cầu sinh lại dữ liệu.

## Kết quả đã có trước khi chốt

- Giữ nguyên checkpoint, ranking, log và metadata của run cũ để truy được cách chạy.
- Kết quả chấm có lọc seen phải được ghi rõ là protocol cũ, không so trực tiếp với
  số không lọc như thể hai bên dùng cùng thiết lập.
- Có thể chấm lại ranking trước lọc nếu đủ top-K, hoặc xuất lại ranking từ checkpoint.
  Việc chấm lại không thay đổi quy tắc đã dùng để chọn checkpoint trước đó.
- Nếu checkpoint được chọn bằng validation có lọc, phải ghi rõ khi báo cáo điểm
  không lọc của checkpoint ấy; không ghi thành run đã chọn checkpoint theo protocol mới.
- Trước khi kết luận mô hình nào tốt hơn hoặc làm paired t-test, đối chiếu cùng
  bộ chia, tập ứng viên, quy tắc lọc và cách chọn checkpoint.

## Liên hệ với code tác giả và triển khai

Code RPG công khai tại commit `7dcf95c` không lọc item đã xem trong
[`generate()`](https://github.com/facebookresearch/RPG_KDD2025/blob/7dcf95c/genrec/models/RPG/model.py)
và [`Evaluator`](https://github.com/facebookresearch/RPG_KDD2025/blob/7dcf95c/genrec/evaluator.py).
Đây là hành vi code đã kiểm tra, không phải khẳng định paper mô tả rõ quy tắc này
hoặc mọi baseline trong bảng paper đều dùng cùng protocol.

Đây là cập nhật tài liệu. Việc nối quy ước vào code, config, notebook và chấm lại
kết quả được thực hiện trong PR công việc tương ứng; không suy ra hành vi của
các run đã có chỉ từ dòng cấu hình dự kiến. Giá trị mục tiêu khi triển khai là
`exclude_seen: false` ở cả chọn checkpoint và đánh giá.
