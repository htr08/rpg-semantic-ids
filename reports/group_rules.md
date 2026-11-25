# Luật kết hợp cấp nhóm (thương hiệu, danh mục) và cấp item

22,363 giao dịch train; min_confidence = 0.1; |X ∪ Y| ≤ 3. Chấm trên 22,363 lịch sử validation. Sinh bởi `python -m src.group_rules`.

- Giao dịch cấp nhóm là tập các nhóm chứa item train của user. Luật và số đếm Popularity chỉ lấy từ train; validation chỉ dùng để chấm; tương tác test không được dùng.
- Coverage: tỉ lệ user (trên mọi user validation) có ít nhất một luật áp dụng được (tiền đề ⊆ nhóm của lịch sử). Hit rate: trong user được phủ và có item validation thuộc ít nhất một nhóm (cột "User chấm hit"), tỉ lệ nhóm của item validation nằm trong hợp hậu quả của luật áp dụng được. Popularity: chọn cùng số nhóm phổ biến nhất (theo số user train) cho mỗi user. Hậu quả được tính cả khi nhóm đã có trong lịch sử. Hit rate giữa các cấp không so thẳng được vì tập đích và kích thước tập dự đoán khác nhau; so sánh có ý nghĩa là luật với Popularity trong cùng một dòng.
- Danh mục độ sâu d: tiền tố d phần tử đầu của đường dẫn categories; một item có nhiều đường dẫn thuộc nhiều nhóm. Item thiếu brand không thuộc nhóm nào ở cấp brand.

| Cấp | Số nhóm | Item không có nhóm | min_support | Số đếm tối thiểu | Nhóm phổ biến | Luật | Luật 1⇒1 | Nhóm trong luật | Coverage | Số nhóm dự đoán TB | Hit rate: luật | Hit rate: Popularity cùng kích thước | User chấm hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| item | 12,101 | 0 | 0.0005 | 12 | 3,218 | 119,967 | 13,127 | 514 | 0.391 | 34.6 | 0.139 | 0.033 | 8,741 |
| item | 12,101 | 0 | 0.001 | 23 | 1,452 | 8,995 | 2,837 | 245 | 0.250 | 22.4 | 0.130 | 0.026 | 5,590 |
| item | 12,101 | 0 | 0.002 | 45 | 601 | 276 | 204 | 66 | 0.141 | 5.2 | 0.080 | 0.010 | 3,146 |
| brand | 2,041 | 2,118 | 0.005 | 112 | 183 | 978 | 368 | 48 | 0.676 | 12.0 | 0.251 | 0.241 | 13,197 |
| brand | 2,041 | 2,118 | 0.01 | 224 | 85 | 140 | 110 | 23 | 0.521 | 8.3 | 0.222 | 0.205 | 10,243 |
| brand | 2,041 | 2,118 | 0.02 | 448 | 40 | 12 | 12 | 7 | 0.348 | 2.7 | 0.108 | 0.108 | 6,837 |
| category depth 2 | 10 | 0 | 0.005 | 112 | 6 | 123 | 27 | 6 | 1.000 | 5.7 | 0.909 | 0.992 | 22,363 |
| category depth 2 | 10 | 0 | 0.01 | 224 | 6 | 123 | 27 | 6 | 1.000 | 5.7 | 0.909 | 0.992 | 22,363 |
| category depth 2 | 10 | 0 | 0.02 | 448 | 6 | 107 | 27 | 6 | 1.000 | 5.7 | 0.908 | 0.992 | 22,363 |
| category depth 3 | 54 | 0 | 0.005 | 112 | 38 | 3,740 | 449 | 36 | 1.000 | 19.7 | 0.900 | 0.919 | 22,363 |
| category depth 3 | 54 | 0 | 0.01 | 224 | 33 | 2,000 | 346 | 32 | 1.000 | 18.4 | 0.883 | 0.901 | 22,363 |
| category depth 3 | 54 | 0 | 0.02 | 448 | 29 | 873 | 244 | 24 | 0.998 | 17.4 | 0.863 | 0.884 | 22,323 |
| category depth 4 | 202 | 0 | 0.005 | 112 | 117 | 5,007 | 960 | 88 | 1.000 | 26.2 | 0.692 | 0.650 | 22,354 |
| category depth 4 | 202 | 0 | 0.01 | 224 | 88 | 1,798 | 498 | 55 | 0.994 | 21.3 | 0.621 | 0.586 | 22,233 |
| category depth 4 | 202 | 0 | 0.02 | 448 | 58 | 491 | 206 | 29 | 0.968 | 15.4 | 0.512 | 0.473 | 21,641 |

Cấu hình có hit rate của luật không vượt Popularity cùng kích thước: brand (min_support 0.02); category depth 2 (min_support 0.005); category depth 2 (min_support 0.01); category depth 2 (min_support 0.02); category depth 3 (min_support 0.005); category depth 3 (min_support 0.01); category depth 3 (min_support 0.02).

## brand: 10 luật lift cao nhất ở min_support = 0.01

| # | Tiền đề X | Hậu quả Y | Đếm X∪Y | Đếm X | Đếm Y | Confidence | Lift | Kulczynski |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Gelish | Harmony | 247 | 461 | 372 | 0.536 | 32.21 | 0.600 |
| 2 | AXE | Dove ; Suave | 244 | 718 | 450 | 0.340 | 16.89 | 0.441 |
| 3 | AXE | Clear ; Suave | 228 | 718 | 434 | 0.318 | 16.36 | 0.421 |
| 4 | AXE | Clear ; Dove | 264 | 718 | 520 | 0.368 | 15.81 | 0.438 |
| 5 | Clear ; Dove | Simple | 248 | 520 | 730 | 0.477 | 14.61 | 0.408 |
| 6 | Clear | Dove ; Suave | 307 | 1086 | 450 | 0.283 | 14.05 | 0.482 |
| 7 | Clear | Motions | 237 | 1086 | 376 | 0.218 | 12.98 | 0.424 |
| 8 | Clear | Simple | 386 | 1086 | 730 | 0.355 | 10.89 | 0.442 |
| 9 | AXE | Clear | 377 | 718 | 1086 | 0.525 | 10.81 | 0.436 |
| 10 | Banana Boat | Clear | 289 | 568 | 1086 | 0.509 | 10.48 | 0.387 |

## category depth 2: 10 luật lift cao nhất ở min_support = 0.01

| # | Tiền đề X | Hậu quả Y | Đếm X∪Y | Đếm X | Đếm Y | Confidence | Lift | Kulczynski |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Beauty > Fragrance ; Beauty > Hair Care | Beauty > Bath & Body | 441 | 1296 | 5247 | 0.340 | 1.45 | 0.212 |
| 2 | Beauty > Bath & Body ; Beauty > Makeup | Beauty > Fragrance | 337 | 2385 | 2246 | 0.141 | 1.41 | 0.146 |
| 3 | Beauty > Fragrance ; Beauty > Tools & Accessories | Beauty > Bath & Body | 226 | 696 | 5247 | 0.325 | 1.38 | 0.184 |
| 4 | Beauty > Fragrance ; Beauty > Skin Care | Beauty > Bath & Body | 515 | 1630 | 5247 | 0.316 | 1.35 | 0.207 |
| 5 | Beauty > Bath & Body | Beauty > Hair Care ; Beauty > Skin Care | 2966 | 5247 | 9740 | 0.565 | 1.30 | 0.435 |
| 6 | Beauty > Fragrance ; Beauty > Tools & Accessories | Beauty > Makeup | 477 | 696 | 11812 | 0.685 | 1.30 | 0.363 |
| 7 | Beauty > Makeup | Beauty > Tools & Accessories | 5696 | 11812 | 8499 | 0.482 | 1.27 | 0.576 |
| 8 | Beauty > Bath & Body | Beauty > Hair Care ; Beauty > Tools & Accessories | 1228 | 5247 | 4282 | 0.234 | 1.22 | 0.260 |
| 9 | Beauty > Bath & Body | Beauty > Skin Care ; Beauty > Tools & Accessories | 1422 | 5247 | 4985 | 0.271 | 1.22 | 0.278 |
| 10 | Beauty > Bath & Body | Beauty > Skin Care | 4352 | 5247 | 15664 | 0.829 | 1.18 | 0.554 |

## category depth 3: 10 luật lift cao nhất ở min_support = 0.01

| # | Tiền đề X | Hậu quả Y | Đếm X∪Y | Đếm X | Đếm Y | Confidence | Lift | Kulczynski |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Beauty > Bath & Body > Scrubs & Body Treatments ; Beauty > Hair Care > Conditioners | Beauty > Hair Care > Hair & Scalp Treatments | 265 | 536 | 2583 | 0.494 | 4.28 | 0.298 |
| 2 | Beauty > Hair Care > Conditioners ; Beauty > Skin Care > Sun | Beauty > Hair Care > Hair & Scalp Treatments | 326 | 728 | 2583 | 0.448 | 3.88 | 0.287 |
| 3 | Beauty > Hair Care > Conditioners ; Beauty > Skin Care > Sun | Beauty > Hair Care > Shampoos | 506 | 728 | 4078 | 0.695 | 3.81 | 0.410 |
| 4 | Beauty > Hair Care > Conditioners | Beauty > Hair Care > Hair & Scalp Treatments ; Beauty > Hair Care > Shampoos | 626 | 3777 | 989 | 0.166 | 3.75 | 0.399 |
| 5 | Beauty > Bath & Body > Scrubs & Body Treatments ; Beauty > Hair Care > Styling Products | Beauty > Hair Care > Hair & Scalp Treatments | 254 | 600 | 2583 | 0.423 | 3.67 | 0.261 |
| 6 | Beauty > Bath & Body > Cleansers ; Beauty > Hair Care > Conditioners | Beauty > Hair Care > Shampoos | 568 | 851 | 4078 | 0.667 | 3.66 | 0.403 |
| 7 | Beauty > Hair Care > Hair & Scalp Treatments ; Beauty > Skin Care > Sun | Beauty > Hair Care > Shampoos | 358 | 538 | 4078 | 0.665 | 3.65 | 0.377 |
| 8 | Beauty > Bath & Body > Scrubs & Body Treatments ; Beauty > Hair Care > Shampoos | Beauty > Hair Care > Hair & Scalp Treatments | 251 | 597 | 2583 | 0.420 | 3.64 | 0.259 |
| 9 | Beauty > Bath & Body > Scrubs & Body Treatments ; Beauty > Hair Care > Conditioners | Beauty > Hair Care > Shampoos | 353 | 536 | 4078 | 0.659 | 3.61 | 0.373 |
| 10 | Beauty > Bath & Body > Cleansers ; Beauty > Hair Care > Hair & Scalp Treatments | Beauty > Hair Care > Conditioners | 332 | 545 | 3777 | 0.609 | 3.61 | 0.349 |

## category depth 4: 10 luật lift cao nhất ở min_support = 0.01

| # | Tiền đề X | Hậu quả Y | Đếm X∪Y | Đếm X | Đếm Y | Confidence | Lift | Kulczynski |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Beauty > Makeup > Nails > Nail Art ; Beauty > Makeup > Nails > Nail Polish | Beauty > Tools & Accessories > Nail Tools > Nail Art Equipment | 311 | 392 | 2317 | 0.793 | 7.66 | 0.464 |
| 2 | Beauty > Makeup > Nails > Nail Art | Beauty > Tools & Accessories > Nail Tools > Nail Art Equipment | 746 | 954 | 2317 | 0.782 | 7.55 | 0.552 |
| 3 | Beauty > Tools & Accessories > Nail Tools > Nail Art Equipment | Beauty > Tools & Accessories > Nail Tools > Nail Brushes | 251 | 2317 | 385 | 0.108 | 6.29 | 0.380 |
| 4 | Beauty > Tools & Accessories > Nail Tools > False Nails | Beauty > Tools & Accessories > Nail Tools > Nail Art Equipment | 354 | 545 | 2317 | 0.650 | 6.27 | 0.401 |
| 5 | Beauty > Hair Care > Conditioners ; Beauty > Hair Care > Styling Products > Hair Sprays | Beauty > Hair Care > Styling Products > Creams, Gels & Lotions | 246 | 517 | 2035 | 0.476 | 5.23 | 0.298 |
| 6 | Beauty > Skin Care > Body > Moisturizers ; Beauty > Skin Care > Face > Exfoliators & Scrubs | Beauty > Skin Care > Face > Oils & Serums | 227 | 599 | 1667 | 0.379 | 5.08 | 0.258 |
| 7 | Beauty > Hair Care > Conditioners ; Beauty > Skin Care > Sun > Skin Protection | Beauty > Hair Care > Styling Products > Creams, Gels & Lotions | 260 | 568 | 2035 | 0.458 | 5.03 | 0.293 |
| 8 | Beauty > Bath & Body > Cleansers > Body Washes ; Beauty > Hair Care > Shampoos | Beauty > Skin Care > Sun > Skin Protection | 239 | 649 | 1641 | 0.368 | 5.02 | 0.257 |
| 9 | Beauty > Skin Care > Face > Creams & Moisturizers ; Beauty > Skin Care > Face > Oils & Serums | Beauty > Skin Care > Face > Exfoliators & Scrubs | 235 | 935 | 1120 | 0.251 | 5.02 | 0.231 |
| 10 | Beauty > Hair Care > Shampoos ; Beauty > Hair Care > Styling Products > Hair Sprays | Beauty > Hair Care > Styling Products > Creams, Gels & Lotions | 238 | 524 | 2035 | 0.454 | 4.99 | 0.286 |
