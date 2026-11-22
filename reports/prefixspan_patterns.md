# Mẫu tuần tự (PrefixSpan) so với tập phổ biến không thứ tự (FP-Growth)

22,363 chuỗi train có thứ tự; mẫu và tập phổ biến có độ dài ≤ 3; min_confidence cho luật 1 item ⇒ 1 item = 0.1. Sinh bằng `python -m src.patterns prefixspan`.

Ghi chú:

- Số đếm của mẫu là số user có mẫu là dãy con (không cần liền kề) của chuỗi train; mỗi user tính một lần. Ngưỡng đếm tối thiểu giống FP-Growth.
- Khi không user nào lặp item (đúng với train Beauty), count(a → b) + count(b → a) bằng số đếm của {a, b}: thứ tự chia số đếm của mỗi cặp thành hai phần. Một tập có thể cho tối đa 2 mẫu độ dài 2 và 6 mẫu độ dài 3, nên số mẫu tuần tự có thể lớn hơn hoặc nhỏ hơn số tập không thứ tự; một cặp có số đếm dưới 2 × ngưỡng không thể có cả hai hướng đạt ngưỡng. Các số trong bảng chịu ảnh hưởng của phép chia này, nên chưa tự nó cho biết thứ tự có thêm thông tin hay không.
- Luật 1 → 1 tuần tự dùng cùng mẫu số count(a) và tử số count(a → b) ≤ count({a, b}), nên là tập con của luật 1 ⇒ 1 không thứ tự; chênh lệch số luật giữa hai cột là hệ quả của định nghĩa, không phải bằng chứng về thông tin của thứ tự.
- Thứ tự theo `interactions.csv`: timestamp tăng dần, trùng timestamp theo số dòng gốc. Trong phần train, 85,971 / 153,776 tương tác (55.9%) thuộc nhóm trùng (user, timestamp), và 57,021 / 131,413 cặp liền kề trong chuỗi có cùng timestamp. Mọi timestamp là mốc ngày: có. Cột "chặt" chỉ tính user có timestamp của hai item khác nhau.
- Hướng chiếm ≥ 80% của một cặp: số cặp quan sát được so với số cặp kỳ vọng nếu mỗi user chọn thứ tự độc lập với xác suất 1/2 (phân phối nhị thức theo số user của từng cặp). Mốc chỉ cho biết kết quả khác thứ tự ngẫu nhiên hoàn toàn; không cho biết nguyên nhân (xem mục Hạn chế).
- Câu hỏi thứ tự mua có thêm thông tin hay không chưa được trả lời ở tài liệu này; phân tích sâu nằm ở kế hoạch 24/11.

| min_support | Đếm tối thiểu | Item phổ biến | Cặp phổ biến (không thứ tự) | Mẫu tuần tự độ dài 2 | Mẫu độ dài 2 (thứ tự chặt) | Tập 3 item phổ biến | Mẫu tuần tự độ dài 3 | FP-Growth (s) | PrefixSpan (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0005 | 12 | 3,218 | 7,070 | 6,086 | 5,172 | 19,097 | 6,289 | 1.9 | 1.5 |
| 0.001 | 23 | 1,452 | 1,423 | 936 | 682 | 1,030 | 190 | 1.0 | 0.7 |
| 0.002 | 45 | 601 | 102 | 48 | 23 | 12 | 2 | 0.8 | 0.3 |

## Độ nhạy với thứ tự các item trùng ngày

Hoán vị ngẫu nhiên thứ tự các item trùng timestamp trong từng chuỗi (seed 42, 43, 44) rồi chạy lại PrefixSpan; cột khoảng là nhỏ nhất–lớn nhất qua các seed.

| min_support | Mẫu độ dài 2: theo chuỗi | sau hoán vị | thứ tự chặt | Mẫu độ dài 3: theo chuỗi | sau hoán vị |
|---:|---:|---:|---:|---:|---:|
| 0.0005 | 6,086 | 5,936–5,974 | 5,172 | 6,289 | 3,989–4,027 |
| 0.001 | 936 | 839–844 | 682 | 190 | 91–102 |
| 0.002 | 48 | 29–31 | 23 | 2 | 1 |

## Cặp phổ biến theo hướng

| min_support | Cặp phổ biến | Cả hai hướng đạt ngưỡng | Một hướng đạt | Không hướng nào đạt | Phần số đếm cặp có timestamp trùng | Luật 1 ⇒ 1 không thứ tự | Luật 1 → 1 tuần tự |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0005 | 7,070 | 183 | 5,720 | 1,167 | 11.8% | 13,127 | 5,484 |
| 0.001 | 1,423 | 7 | 922 | 494 | 17.6% | 2,837 | 931 |
| 0.002 | 102 | 0 | 48 | 54 | 30.4% | 204 | 48 |

## Mức nghiêng về một hướng (hướng chiếm ≥ 80% số user của cặp)

| min_support | Cách xác định thứ tự | Cặp xét | Tỉ lệ hướng chiếm ưu thế (trung vị) | Cặp có hướng ≥ ngưỡng | Kỳ vọng nếu thứ tự ngẫu nhiên | Cặp có tổng ≥ đếm tối thiểu: hướng ≥ ngưỡng / số cặp |
|---:|---|---:|---:|---:|---:|---:|
| 0.0005 | theo chuỗi (trùng timestamp theo số dòng) | 7,070 | 1.00 | 5,282 | 119.8 | 5,282 / 7,070 |
| 0.0005 | chặt (chỉ timestamp khác nhau) | 7,048 | 1.00 | 5,205 | 242.8 | 4,798 / 6,171 |
| 0.001 | theo chuỗi (trùng timestamp theo số dòng) | 1,423 | 0.89 | 887 | 2.1 | 887 / 1,423 |
| 0.001 | chặt (chỉ timestamp khác nhau) | 1,422 | 0.89 | 859 | 15.1 | 720 / 968 |
| 0.002 | theo chuỗi (trùng timestamp theo số dòng) | 102 | 0.86 | 63 | 0.0 | 63 / 102 |
| 0.002 | chặt (chỉ timestamp khác nhau) | 102 | 0.75 | 45 | 0.7 | 29 / 40 |

## Tập 3 item phổ biến và thứ tự

| min_support | Tập 3 item phổ biến | Có ít nhất một thứ tự phổ biến | Mẫu tuần tự độ dài 3 |
|---:|---:|---:|---:|
| 0.0005 | 19,097 | 6,233 | 6,289 |
| 0.001 | 1,030 | 190 | 190 |
| 0.002 | 12 | 2 | 2 |

## Kiểm tra nhất quán giữa PrefixSpan, FP-Growth và bộ đếm cặp độc lập

| min_support | Số đếm item khớp | Mẫu độ dài 2 khớp bộ đếm cặp | Mẫu có tập item không phổ biến | Cặp có count(a→b) + count(b→a) ≠ số đếm {a, b} |
|---:|:---:|:---:|---:|---:|
| 0.0005 | có | có | 0 | 0 |
| 0.001 | có | có | 0 | 0 |
| 0.002 | có | có | 0 | 0 |

## min_support = 0.0005: 20 mẫu a → b có số đếm lớn nhất

Bảng xếp theo số đếm theo chuỗi và chỉ lấy mức min_support thấp nhất; JSON lưu top mẫu của từng mức. Một số mẫu đổi hướng hoặc hòa khi chỉ tính thứ tự chặt (so hai cột cuối).

| # | a | b | a → b | b → a | {a, b} | a → b (chặt) | b → a (chặt) |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | [10747] Dead Sea Mud Facial Mask and Deep Facial Cleanser by A… | [3023] Dead Sea Mud Shampoo - Fast, Natural Relief from Dandr… | 81 | 3 | 84 | 75 | 3 |
| 2 | [3023] Dead Sea Mud Shampoo - Fast, Natural Relief from Dandr… | [6466] Dead Sea Salt Deep Hair Conditioner - for Dry or Damag… | 79 | 1 | 80 | 64 | 1 |
| 3 | [301] Dotting 5 X 2 Way Marbleizing Dotting Pen Set for Nail… | [279] World Pride Nail Tape Stripe Decoration Sticker Hologr… | 78 | 10 | 88 | 18 | 10 |
| 4 | [6463] Vitamin C Serum - Antioxidant Treatment - Natural & Or… | [3023] Dead Sea Mud Shampoo - Fast, Natural Relief from Dandr… | 75 | 7 | 82 | 68 | 7 |
| 5 | [6466] Dead Sea Salt Deep Hair Conditioner - for Dry or Damag… | [9744] Adovia Facial Serum - Anti Aging, Skin Lifting, Facial… | 69 | 1 | 70 | 62 | 1 |
| 6 | [834] SHANY Professional 13-Piece Cosmetic Brush Set with Po… | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates… | 63 | 4 | 67 | 9 | 4 |
| 7 | [3023] Dead Sea Mud Shampoo - Fast, Natural Relief from Dandr… | [9744] Adovia Facial Serum - Anti Aging, Skin Lifting, Facial… | 63 | 0 | 63 | 63 | 0 |
| 8 | [279] World Pride Nail Tape Stripe Decoration Sticker Hologr… | [302] 15pcs Nail Art Painting Pen Brush | 62 | 13 | 75 | 13 | 13 |
| 9 | [919] Dove Men+Care Sensitive + Face Lotion 1.69 FL.Oz. | [1060] Dove Men+Care Sensitive + Face Wash, 5 Ounces | 62 | 14 | 76 | 29 | 14 |
| 10 | [706] Banana Boat Sport Coolzone Sunscreen Lotion Spf 50, 6 … | [707] Banana Boat Protect and Hydrate Sunscreen Lotion Spf 3… | 60 | 21 | 81 | 12 | 21 |
| 11 | [6463] Vitamin C Serum - Antioxidant Treatment - Natural & Or… | [6466] Dead Sea Salt Deep Hair Conditioner - for Dry or Damag… | 59 | 1 | 60 | 57 | 1 |
| 12 | [10747] Dead Sea Mud Facial Mask and Deep Facial Cleanser by A… | [6466] Dead Sea Salt Deep Hair Conditioner - for Dry or Damag… | 58 | 2 | 60 | 57 | 2 |
| 13 | [301] Dotting 5 X 2 Way Marbleizing Dotting Pen Set for Nail… | [302] 15pcs Nail Art Painting Pen Brush | 57 | 9 | 66 | 6 | 9 |
| 14 | [3023] Dead Sea Mud Shampoo - Fast, Natural Relief from Dandr… | [6464] Dead Sea Mud Facial Mask - Ancient Natural Facial Mask… | 57 | 3 | 60 | 57 | 1 |
| 15 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates… | [671] SHANY Stamping Nail Art Set (Nail Art Image Plate Hold… | 56 | 2 | 58 | 9 | 2 |
| 16 | [3023] Dead Sea Mud Shampoo - Fast, Natural Relief from Dandr… | [3548] Eye Cream For Dark Circles, Puffiness, Wrinkles & Bags… | 56 | 1 | 57 | 56 | 1 |
| 17 | [704] Cotz Pediatric Spf 40, 3.5 Ounce | [706] Banana Boat Sport Coolzone Sunscreen Lotion Spf 50, 6 … | 54 | 1 | 55 | 53 | 1 |
| 18 | [3548] Eye Cream For Dark Circles, Puffiness, Wrinkles & Bags… | [9744] Adovia Facial Serum - Anti Aging, Skin Lifting, Facial… | 54 | 0 | 54 | 50 | 0 |
| 19 | [6465] Nature's Best Vitamin C Serum - #1 For Face, Eyes & Ac… | [9744] Adovia Facial Serum - Anti Aging, Skin Lifting, Facial… | 54 | 3 | 57 | 54 | 3 |
| 20 | [6463] Vitamin C Serum - Antioxidant Treatment - Natural & Or… | [6464] Dead Sea Mud Facial Mask - Ancient Natural Facial Mask… | 53 | 3 | 56 | 53 | 0 |

## Hạn chế

- Thứ tự phá hòa theo số dòng gốc của file raw. Trong phần train, 57,021 / 57,021 cặp liền kề trùng timestamp có ASIN tăng dần, tức thứ tự "theo chuỗi" của các item trùng ngày trùng với thứ tự ASIN. Hướng của một cặp tính theo cột "theo chuỗi" vì vậy có thể phản ánh quy tắc phá hòa; cột "chặt" loại ảnh hưởng này.
- Số mẫu tuần tự trong bảng đầu phụ thuộc quy tắc phá hòa: xem mục "Độ nhạy với thứ tự các item trùng ngày"; các cột hướng, luật 1 → 1 tuần tự và bảng top đều dùng thứ tự theo chuỗi.
- Ở min_support 0.0005, 5,282 cặp có hướng chiếm ≥ 80% (theo chuỗi). 2,954 user có ít nhất một cặp như vậy; 224 user (1%) có nhiều cặp nhất chiếm 75.4% số lượt (user, cặp) của các cặp này và 73.8% số lượt của mọi cặp phổ biến. Nhóm này chọn theo chính số lượt đóng góp; độ dài train trung vị của nhóm là 41, của toàn tập là 4. Số đếm ≥ 12 user vì vậy không tương đương 12 user độc lập về hành vi.
- Các số trên chỉ mô tả; nguyên nhân của việc nghiêng về một hướng (thứ tự mua, thời điểm item được review, user review nhiều item, quy tắc phá hòa) chưa được tách riêng.
- Train Beauty không có item lặp trong cùng một user; với dữ liệu có lặp, mẫu a → a xuất hiện và đẳng thức count(a → b) + count(b → a) = số đếm {a, b} không còn đúng.
- Chưa thực hiện: luật tuần tự với lift, và gộp item lên cấp danh mục (kế hoạch mục 3d, khi luật cấp item quá thưa).
- Thời gian chạy là một lần đo, hai thư viện (mlxtend, prefixspan) không cùng điều kiện nên không dùng để so sánh tốc độ.
