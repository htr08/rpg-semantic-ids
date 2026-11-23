# Luật kết hợp trên giao dịch train (FP-Growth)

22,363 giao dịch; min_confidence = 0.1; tập phổ biến và luật giới hạn |X ∪ Y| ≤ 4. Sinh bằng `python -m src.patterns rules`.

Ghi chú:

- Giao dịch là tập item duy nhất trong phần train của mỗi user; không dùng validation và test.
- Giới hạn |X ∪ Y| ≤ 4 chỉ cắt các tập phổ biến dài hơn; số item phổ biến và số đếm của mọi luật không đổi. Số tập phổ biến không giới hạn độ dài nằm trong `fpgrowth_0.0005.json`, `fpgrowth_0.001.json` và `fpgrowth_trial.json` (mức 0,002).
- Xếp hạng theo lift giảm dần, hòa thì theo số đếm X ∪ Y giảm dần, rồi theo thứ tự item. Luật A ⇒ B và B ⇒ A của cùng một cặp có cùng lift và số đếm, nên bảng thứ nhất chỉ giữ một luật cho mỗi tập X ∪ Y.
- Với support thấp, lift bị chặn trên bởi 1 / max(sup(X), sup(Y)); số đếm X ∪ Y ở các luật đứng đầu chỉ vài chục giao dịch (cột Đếm X∪Y) nên lift lớn không đồng nghĩa với quan hệ mua kèm bền vững. Cột Kulczynski dùng để đối chiếu.
- Cùng brand: mọi item trong X ∪ Y có brand giống nhau (không phân biệt hoa thường; brand rỗng hoặc "Unknown" tính là thiếu). Các biến thể ghi khác nhau của một thương hiệu (ví dụ "SHANY" và "Shany Cosmetics") được tính là khác brand.
- Chung danh mục lá: mọi item trong X ∪ Y có chung ít nhất một danh mục lá (phần tử cuối của một đường dẫn `categories`). Quy tắc chọn danh mục lá chính thức (G4) chưa được chốt; cờ này chỉ dùng để mô tả.

| min_support | Số giao dịch tối thiểu | Item phổ biến | Tập phổ biến (≤ max_len) | Luật | Luật 1 item ⇒ 1 item | Item có trong luật | Confidence TB | Lift trung vị | Thời gian FP-Growth (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0005 | 12 | 3,218 | 69,033 | 656,681 | 13,127 | 514 | 0.579 | 353.1 | 2.8 |
| 0.001 | 23 | 1,452 | 4,698 | 20,024 | 2,837 | 245 | 0.558 | 182.7 | 2.0 |
| 0.002 | 45 | 601 | 716 | 290 | 204 | 66 | 0.512 | 102.7 | 0.9 |

## min_support = 0.0005 (số đếm ≥ 12)

### 20 luật lift cao nhất (mỗi tập X ∪ Y chỉ giữ một luật)

| # | Tiền đề X | Hậu quả Y | Đếm X∪Y | Đếm X | Đếm Y | Confidence | Lift | Kulczynski | Cùng brand | Chung danh mục lá |
|---:|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 1 | [253] SHANY 180 Color Eyeshadow Palette (180 Color Eyeshadow Palette, Unite…<br>[3457] SHANY Cosmetics Eyelash Extend Assorted Reusable Eyelashes Thick and … | [10561] SHANY Cosmetics French Nail Set, 7 Ounce<br>[11212] SHANY Carry All Trunk Professional Makeup Kit - Eyeshadow,Pedicure,ma… | 12 | 12 | 12 | 1.000 | 1863.6 | 1.000 | có | không |
| 2 | [253] SHANY 180 Color Eyeshadow Palette (180 Color Eyeshadow Palette, Unite…<br>[3457] SHANY Cosmetics Eyelash Extend Assorted Reusable Eyelashes Thick and … | [11212] SHANY Carry All Trunk Professional Makeup Kit - Eyeshadow,Pedicure,ma…<br>[11987] SHANY Cosmetics Intense Eyes Palette 72 Color Eyeshadow Palette, 17 O… | 12 | 12 | 12 | 1.000 | 1863.6 | 1.000 | có | không |
| 3 | [253] SHANY 180 Color Eyeshadow Palette (180 Color Eyeshadow Palette, Unite…<br>[10561] SHANY Cosmetics French Nail Set, 7 Ounce | [11212] SHANY Carry All Trunk Professional Makeup Kit - Eyeshadow,Pedicure,ma…<br>[11987] SHANY Cosmetics Intense Eyes Palette 72 Color Eyeshadow Palette, 17 O… | 12 | 12 | 12 | 1.000 | 1863.6 | 1.000 | có | không |
| 4 | [136] SHANY Makeup Artists Must Have Pro Eyeshadow Palette, 96 Color<br>[4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [2981] SHANY Slick & Shine Lipstick Set - Set of 12 Famouse Colors<br>[4358] SHANY Foundation Brush - Premium collection -Ultra Firm and smooth | 12 | 12 | 13 | 1.000 | 1720.2 | 0.962 | không | không |
| 5 | [136] SHANY Makeup Artists Must Have Pro Eyeshadow Palette, 96 Color<br>[4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [2981] SHANY Slick & Shine Lipstick Set - Set of 12 Famouse Colors<br>[11060] SHANY Professional Multi Effect Velvet Touch Eyeshadow Palette, 24 Co… | 12 | 12 | 13 | 1.000 | 1720.2 | 0.962 | không | không |
| 6 | [136] SHANY Makeup Artists Must Have Pro Eyeshadow Palette, 96 Color<br>[4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [2981] SHANY Slick & Shine Lipstick Set - Set of 12 Famouse Colors<br>[11983] SHANY Eyeshadow Kit, 48 Color | 12 | 12 | 13 | 1.000 | 1720.2 | 0.962 | không | không |
| 7 | [136] SHANY Makeup Artists Must Have Pro Eyeshadow Palette, 96 Color<br>[4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [4358] SHANY Foundation Brush - Premium collection -Ultra Firm and smooth<br>[4579] SHANY UV Gel Light Nail Dryer, 36 Watts Pro Series | 12 | 12 | 13 | 1.000 | 1720.2 | 0.962 | không | không |
| 8 | [136] SHANY Makeup Artists Must Have Pro Eyeshadow Palette, 96 Color<br>[4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [4579] SHANY UV Gel Light Nail Dryer, 36 Watts Pro Series<br>[11060] SHANY Professional Multi Effect Velvet Touch Eyeshadow Palette, 24 Co… | 12 | 12 | 13 | 1.000 | 1720.2 | 0.962 | không | không |
| 9 | [136] SHANY Makeup Artists Must Have Pro Eyeshadow Palette, 96 Color<br>[4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [4579] SHANY UV Gel Light Nail Dryer, 36 Watts Pro Series<br>[11983] SHANY Eyeshadow Kit, 48 Color | 12 | 12 | 13 | 1.000 | 1720.2 | 0.962 | không | không |
| 10 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [2981] SHANY Slick & Shine Lipstick Set - Set of 12 Famouse Colors<br>[4358] SHANY Foundation Brush - Premium collection -Ultra Firm and smooth | 12 | 12 | 13 | 1.000 | 1720.2 | 0.962 | không | không |
| 11 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [2981] SHANY Slick & Shine Lipstick Set - Set of 12 Famouse Colors<br>[11060] SHANY Professional Multi Effect Velvet Touch Eyeshadow Palette, 24 Co… | 12 | 12 | 13 | 1.000 | 1720.2 | 0.962 | không | không |
| 12 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [2981] SHANY Slick & Shine Lipstick Set - Set of 12 Famouse Colors<br>[11983] SHANY Eyeshadow Kit, 48 Color | 12 | 12 | 13 | 1.000 | 1720.2 | 0.962 | không | không |
| 13 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [4358] SHANY Foundation Brush - Premium collection -Ultra Firm and smooth<br>[4579] SHANY UV Gel Light Nail Dryer, 36 Watts Pro Series | 12 | 12 | 13 | 1.000 | 1720.2 | 0.962 | không | không |
| 14 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [4579] SHANY UV Gel Light Nail Dryer, 36 Watts Pro Series<br>[11060] SHANY Professional Multi Effect Velvet Touch Eyeshadow Palette, 24 Co… | 12 | 12 | 13 | 1.000 | 1720.2 | 0.962 | không | không |
| 15 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [4579] SHANY UV Gel Light Nail Dryer, 36 Watts Pro Series<br>[11983] SHANY Eyeshadow Kit, 48 Color | 12 | 12 | 13 | 1.000 | 1720.2 | 0.962 | không | không |
| 16 | [754] SHANY Studio Quality Natural Cosmetic Brush Set with Leather Pouch, 2…<br>[4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [2981] SHANY Slick & Shine Lipstick Set - Set of 12 Famouse Colors<br>[4358] SHANY Foundation Brush - Premium collection -Ultra Firm and smooth | 12 | 12 | 13 | 1.000 | 1720.2 | 0.962 | không | không |
| 17 | [754] SHANY Studio Quality Natural Cosmetic Brush Set with Leather Pouch, 2…<br>[4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [2981] SHANY Slick & Shine Lipstick Set - Set of 12 Famouse Colors<br>[11060] SHANY Professional Multi Effect Velvet Touch Eyeshadow Palette, 24 Co… | 12 | 12 | 13 | 1.000 | 1720.2 | 0.962 | không | không |
| 18 | [754] SHANY Studio Quality Natural Cosmetic Brush Set with Leather Pouch, 2…<br>[4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [2981] SHANY Slick & Shine Lipstick Set - Set of 12 Famouse Colors<br>[11983] SHANY Eyeshadow Kit, 48 Color | 12 | 12 | 13 | 1.000 | 1720.2 | 0.962 | không | không |
| 19 | [754] SHANY Studio Quality Natural Cosmetic Brush Set with Leather Pouch, 2…<br>[4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [4358] SHANY Foundation Brush - Premium collection -Ultra Firm and smooth<br>[4579] SHANY UV Gel Light Nail Dryer, 36 Watts Pro Series | 12 | 12 | 13 | 1.000 | 1720.2 | 0.962 | không | không |
| 20 | [754] SHANY Studio Quality Natural Cosmetic Brush Set with Leather Pouch, 2…<br>[4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [4579] SHANY UV Gel Light Nail Dryer, 36 Watts Pro Series<br>[11060] SHANY Professional Multi Effect Velvet Touch Eyeshadow Palette, 24 Co… | 12 | 12 | 13 | 1.000 | 1720.2 | 0.962 | không | không |

Trong 20 luật: 3 luật cùng brand, 0 luật chung danh mục lá; số đếm X∪Y nhỏ nhất 12, lớn nhất 12.

### 20 luật lift cao nhất dạng một item ⇒ một item

| # | Tiền đề X | Hậu quả Y | Đếm X∪Y | Đếm X | Đếm Y | Confidence | Lift | Kulczynski | Cùng brand | Chung danh mục lá |
|---:|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 1 | [12042] Acne Sunscreen SPF 30 For Glamorous, Sensitive Skin - Oil Free Protec… | [12043] Acne Treatments - Hollywood acne treatment cream that works | 14 | 17 | 14 | 0.824 | 1315.5 | 0.912 | không | không |
| 2 | [11984] Multicolor Eyeshadow palette - All In One | [11986] 2011 Ultimate Makeup set - Eyeshadow/ Blush/ Lipgloss/ Mascara | 13 | 14 | 16 | 0.929 | 1297.9 | 0.871 | không | có |
| 3 | [11060] SHANY Professional Multi Effect Velvet Touch Eyeshadow Palette, 24 Co… | [11986] 2011 Ultimate Makeup set - Eyeshadow/ Blush/ Lipgloss/ Mascara | 16 | 18 | 16 | 0.889 | 1242.4 | 0.944 | không | có |
| 4 | [11983] SHANY Eyeshadow Kit, 48 Color | [11986] 2011 Ultimate Makeup set - Eyeshadow/ Blush/ Lipgloss/ Mascara | 16 | 18 | 16 | 0.889 | 1242.4 | 0.944 | không | không |
| 5 | [11060] SHANY Professional Multi Effect Velvet Touch Eyeshadow Palette, 24 Co… | [11984] Multicolor Eyeshadow palette - All In One | 14 | 18 | 14 | 0.778 | 1242.4 | 0.889 | không | có |
| 6 | [11983] SHANY Eyeshadow Kit, 48 Color | [11984] Multicolor Eyeshadow palette - All In One | 14 | 18 | 14 | 0.778 | 1242.4 | 0.889 | không | không |
| 7 | [4358] SHANY Foundation Brush - Premium collection -Ultra Firm and smooth | [11986] 2011 Ultimate Makeup set - Eyeshadow/ Blush/ Lipgloss/ Mascara | 16 | 19 | 16 | 0.842 | 1177.0 | 0.921 | không | không |
| 8 | [4358] SHANY Foundation Brush - Premium collection -Ultra Firm and smooth | [11984] Multicolor Eyeshadow palette - All In One | 14 | 19 | 14 | 0.737 | 1177.0 | 0.868 | không | không |
| 9 | [11060] SHANY Professional Multi Effect Velvet Touch Eyeshadow Palette, 24 Co… | [11983] SHANY Eyeshadow Kit, 48 Color | 17 | 18 | 18 | 0.944 | 1173.4 | 0.944 | không | không |
| 10 | [4358] SHANY Foundation Brush - Premium collection -Ultra Firm and smooth | [11060] SHANY Professional Multi Effect Velvet Touch Eyeshadow Palette, 24 Co… | 17 | 19 | 18 | 0.895 | 1111.6 | 0.920 | không | không |
| 11 | [4358] SHANY Foundation Brush - Premium collection -Ultra Firm and smooth | [11983] SHANY Eyeshadow Kit, 48 Color | 17 | 19 | 18 | 0.895 | 1111.6 | 0.920 | không | không |
| 12 | [11984] Multicolor Eyeshadow palette - All In One | [11985] SHANY Leopard Professional Rolling Makeup Case, Light Weight, 12 Poun… | 13 | 14 | 19 | 0.929 | 1092.9 | 0.806 | không | không |
| 13 | [11985] SHANY Leopard Professional Rolling Makeup Case, Light Weight, 12 Poun… | [11986] 2011 Ultimate Makeup set - Eyeshadow/ Blush/ Lipgloss/ Mascara | 14 | 19 | 16 | 0.737 | 1029.9 | 0.806 | không | không |
| 14 | [4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [11060] SHANY Professional Multi Effect Velvet Touch Eyeshadow Palette, 24 Co… | 12 | 15 | 18 | 0.800 | 993.9 | 0.733 | không | không |
| 15 | [4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | [11983] SHANY Eyeshadow Kit, 48 Color | 12 | 15 | 18 | 0.800 | 993.9 | 0.733 | không | có |
| 16 | [11060] SHANY Professional Multi Effect Velvet Touch Eyeshadow Palette, 24 Co… | [11985] SHANY Leopard Professional Rolling Makeup Case, Light Weight, 12 Poun… | 15 | 18 | 19 | 0.833 | 980.8 | 0.811 | không | không |
| 17 | [11983] SHANY Eyeshadow Kit, 48 Color | [11985] SHANY Leopard Professional Rolling Makeup Case, Light Weight, 12 Poun… | 15 | 18 | 19 | 0.833 | 980.8 | 0.811 | không | không |
| 18 | [10561] SHANY Cosmetics French Nail Set, 7 Ounce | [11987] SHANY Cosmetics Intense Eyes Palette 72 Color Eyeshadow Palette, 17 O… | 16 | 20 | 19 | 0.800 | 941.6 | 0.821 | có | không |
| 19 | [4358] SHANY Foundation Brush - Premium collection -Ultra Firm and smooth | [4704] Shany Cosmetics Everyday Travel Trio Eyeshadow Palette, Olive, Champa… | 12 | 19 | 15 | 0.632 | 941.6 | 0.716 | không | không |
| 20 | [3559] BelleSha Smooth Silky Glowing Hair No Tangles Brush | [5386] Bellesha Hands Free Foot Scrubber | 12 | 18 | 16 | 0.667 | 931.8 | 0.708 | không | không |

Trong 20 luật: 1 luật cùng brand, 4 luật chung danh mục lá; số đếm X∪Y nhỏ nhất 12, lớn nhất 17.

## min_support = 0.001 (số đếm ≥ 23)

### 20 luật lift cao nhất (mỗi tập X ∪ Y chỉ giữ một luật)

| # | Tiền đề X | Hậu quả Y | Đếm X∪Y | Đếm X | Đếm Y | Confidence | Lift | Kulczynski | Cùng brand | Chung danh mục lá |
|---:|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 1 | [253] SHANY 180 Color Eyeshadow Palette (180 Color Eyeshadow Palette, Unite…<br>[9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[11992] Cameo Carry All Beauty Case by Shany © 90pc Pro Make Up Set - Premium… | 23 | 23 | 23 | 1.000 | 972.3 | 1.000 | không | không |
| 2 | [253] SHANY 180 Color Eyeshadow Palette (180 Color Eyeshadow Palette, Unite…<br>[9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [754] SHANY Studio Quality Natural Cosmetic Brush Set with Leather Pouch, 2…<br>[11992] Cameo Carry All Beauty Case by Shany © 90pc Pro Make Up Set - Premium… | 23 | 23 | 23 | 1.000 | 972.3 | 1.000 | không | không |
| 3 | [253] SHANY 180 Color Eyeshadow Palette (180 Color Eyeshadow Palette, Unite…<br>[9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [9619] SHANY Eyeshadow Kit, Metallic Runway, 96 Color<br>[11992] Cameo Carry All Beauty Case by Shany © 90pc Pro Make Up Set - Premium… | 23 | 23 | 23 | 1.000 | 972.3 | 1.000 | không | không |
| 4 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [671] SHANY Stamping Nail Art Set (Nail Art Image Plate Holder, Scraper, St…<br>[9619] SHANY Eyeshadow Kit, Metallic Runway, 96 Color | 24 | 24 | 24 | 1.000 | 931.8 | 1.000 | có | không |
| 5 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [6436] SHANY 2012 Edition All In One Harmony Makeup Kit, 25 Ounce<br>[9619] SHANY Eyeshadow Kit, Metallic Runway, 96 Color | 24 | 24 | 24 | 1.000 | 931.8 | 1.000 | có | không |
| 6 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [9619] SHANY Eyeshadow Kit, Metallic Runway, 96 Color<br>[11212] SHANY Carry All Trunk Professional Makeup Kit - Eyeshadow,Pedicure,ma… | 24 | 24 | 24 | 1.000 | 931.8 | 1.000 | có | không |
| 7 | [253] SHANY 180 Color Eyeshadow Palette (180 Color Eyeshadow Palette, Unite…<br>[9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [671] SHANY Stamping Nail Art Set (Nail Art Image Plate Holder, Scraper, St…<br>[9619] SHANY Eyeshadow Kit, Metallic Runway, 96 Color | 23 | 23 | 24 | 1.000 | 931.8 | 0.979 | có | không |
| 8 | [253] SHANY 180 Color Eyeshadow Palette (180 Color Eyeshadow Palette, Unite…<br>[11992] Cameo Carry All Beauty Case by Shany © 90pc Pro Make Up Set - Premium… | [671] SHANY Stamping Nail Art Set (Nail Art Image Plate Holder, Scraper, St…<br>[9619] SHANY Eyeshadow Kit, Metallic Runway, 96 Color | 23 | 23 | 24 | 1.000 | 931.8 | 0.979 | không | không |
| 9 | [253] SHANY 180 Color Eyeshadow Palette (180 Color Eyeshadow Palette, Unite…<br>[9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [671] SHANY Stamping Nail Art Set (Nail Art Image Plate Holder, Scraper, St…<br>[11992] Cameo Carry All Beauty Case by Shany © 90pc Pro Make Up Set - Premium… | 23 | 23 | 24 | 1.000 | 931.8 | 0.979 | không | không |
| 10 | [253] SHANY 180 Color Eyeshadow Palette (180 Color Eyeshadow Palette, Unite…<br>[9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [834] SHANY Professional 13-Piece Cosmetic Brush Set with Pouch, Set of 12 …<br>[11992] Cameo Carry All Beauty Case by Shany © 90pc Pro Make Up Set - Premium… | 23 | 23 | 24 | 1.000 | 931.8 | 0.979 | không | không |
| 11 | [253] SHANY 180 Color Eyeshadow Palette (180 Color Eyeshadow Palette, Unite…<br>[9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [6436] SHANY 2012 Edition All In One Harmony Makeup Kit, 25 Ounce<br>[9619] SHANY Eyeshadow Kit, Metallic Runway, 96 Color | 23 | 23 | 24 | 1.000 | 931.8 | 0.979 | có | không |
| 12 | [253] SHANY 180 Color Eyeshadow Palette (180 Color Eyeshadow Palette, Unite…<br>[11992] Cameo Carry All Beauty Case by Shany © 90pc Pro Make Up Set - Premium… | [6436] SHANY 2012 Edition All In One Harmony Makeup Kit, 25 Ounce<br>[9619] SHANY Eyeshadow Kit, Metallic Runway, 96 Color | 23 | 23 | 24 | 1.000 | 931.8 | 0.979 | không | không |
| 13 | [253] SHANY 180 Color Eyeshadow Palette (180 Color Eyeshadow Palette, Unite…<br>[9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [6436] SHANY 2012 Edition All In One Harmony Makeup Kit, 25 Ounce<br>[11992] Cameo Carry All Beauty Case by Shany © 90pc Pro Make Up Set - Premium… | 23 | 23 | 24 | 1.000 | 931.8 | 0.979 | không | không |
| 14 | [253] SHANY 180 Color Eyeshadow Palette (180 Color Eyeshadow Palette, Unite…<br>[9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [9619] SHANY Eyeshadow Kit, Metallic Runway, 96 Color<br>[11212] SHANY Carry All Trunk Professional Makeup Kit - Eyeshadow,Pedicure,ma… | 23 | 23 | 24 | 1.000 | 931.8 | 0.979 | có | không |
| 15 | [253] SHANY 180 Color Eyeshadow Palette (180 Color Eyeshadow Palette, Unite…<br>[11992] Cameo Carry All Beauty Case by Shany © 90pc Pro Make Up Set - Premium… | [9619] SHANY Eyeshadow Kit, Metallic Runway, 96 Color<br>[11212] SHANY Carry All Trunk Professional Makeup Kit - Eyeshadow,Pedicure,ma… | 23 | 23 | 24 | 1.000 | 931.8 | 0.979 | không | không |
| 16 | [253] SHANY 180 Color Eyeshadow Palette (180 Color Eyeshadow Palette, Unite…<br>[9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [11212] SHANY Carry All Trunk Professional Makeup Kit - Eyeshadow,Pedicure,ma…<br>[11992] Cameo Carry All Beauty Case by Shany © 90pc Pro Make Up Set - Premium… | 23 | 23 | 24 | 1.000 | 931.8 | 0.979 | không | không |
| 17 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[11992] Cameo Carry All Beauty Case by Shany © 90pc Pro Make Up Set - Premium… | [671] SHANY Stamping Nail Art Set (Nail Art Image Plate Holder, Scraper, St…<br>[9619] SHANY Eyeshadow Kit, Metallic Runway, 96 Color | 23 | 23 | 24 | 1.000 | 931.8 | 0.979 | không | không |
| 18 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [754] SHANY Studio Quality Natural Cosmetic Brush Set with Leather Pouch, 2…<br>[11992] Cameo Carry All Beauty Case by Shany © 90pc Pro Make Up Set - Premium… | 23 | 24 | 23 | 0.958 | 931.8 | 0.979 | không | không |
| 19 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[11992] Cameo Carry All Beauty Case by Shany © 90pc Pro Make Up Set - Premium… | [6436] SHANY 2012 Edition All In One Harmony Makeup Kit, 25 Ounce<br>[9619] SHANY Eyeshadow Kit, Metallic Runway, 96 Color | 23 | 23 | 24 | 1.000 | 931.8 | 0.979 | không | không |
| 20 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [9619] SHANY Eyeshadow Kit, Metallic Runway, 96 Color<br>[11992] Cameo Carry All Beauty Case by Shany © 90pc Pro Make Up Set - Premium… | 23 | 24 | 23 | 0.958 | 931.8 | 0.979 | không | không |

Trong 20 luật: 6 luật cùng brand, 0 luật chung danh mục lá; số đếm X∪Y nhỏ nhất 23, lớn nhất 24.

### 20 luật lift cao nhất dạng một item ⇒ một item

| # | Tiền đề X | Hậu quả Y | Đếm X∪Y | Đếm X | Đếm Y | Confidence | Lift | Kulczynski | Cùng brand | Chung danh mục lá |
|---:|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 1 | [9933] Hair Tamer "No Knot" Pink Detangling Hair Brush Teezer | [9934] Hair Tamer Black Diamond Ultra Mist Multi-Use Spray Bottle | 31 | 31 | 31 | 1.000 | 721.4 | 1.000 | không | không |
| 2 | [9931] Comfy White Microfiber Salon Towels - 10 Pack , 29" x 16" | [9933] Hair Tamer "No Knot" Pink Detangling Hair Brush Teezer | 24 | 25 | 31 | 0.960 | 692.5 | 0.867 | không | không |
| 3 | [9931] Comfy White Microfiber Salon Towels - 10 Pack , 29" x 16" | [9934] Hair Tamer Black Diamond Ultra Mist Multi-Use Spray Bottle | 24 | 25 | 31 | 0.960 | 692.5 | 0.867 | không | không |
| 4 | [9932] Black Croc Hair Styling Clips - 6 Pack | [9933] Hair Tamer "No Knot" Pink Detangling Hair Brush Teezer | 31 | 33 | 31 | 0.939 | 677.7 | 0.970 | không | không |
| 5 | [9932] Black Croc Hair Styling Clips - 6 Pack | [9934] Hair Tamer Black Diamond Ultra Mist Multi-Use Spray Bottle | 31 | 33 | 31 | 0.939 | 677.7 | 0.970 | không | không |
| 6 | [9931] Comfy White Microfiber Salon Towels - 10 Pack , 29" x 16" | [9932] Black Croc Hair Styling Clips - 6 Pack | 25 | 25 | 33 | 1.000 | 677.7 | 0.879 | không | không |
| 7 | [9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [11992] Cameo Carry All Beauty Case by Shany © 90pc Pro Make Up Set - Premium… | 24 | 36 | 25 | 0.667 | 596.3 | 0.813 | không | không |
| 8 | [9619] SHANY Eyeshadow Kit, Metallic Runway, 96 Color | [11992] Cameo Carry All Beauty Case by Shany © 90pc Pro Make Up Set - Premium… | 23 | 35 | 25 | 0.657 | 587.8 | 0.789 | không | không |
| 9 | [2837] SHANY Zebra Rolling Trolly Makeup Case, Light Weight, 12 Pounds | [11976] SHANY Jet Black Professional Rolling Makeup Case Premium Collection, … | 23 | 27 | 34 | 0.852 | 560.3 | 0.764 | không | có |
| 10 | [9619] SHANY Eyeshadow Kit, Metallic Runway, 96 Color | [11156] SHANY Acrylic Nail Brush Sets with Pouch, 5 Ounce | 23 | 35 | 27 | 0.657 | 544.3 | 0.754 | có | không |
| 11 | [9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [11156] SHANY Acrylic Nail Brush Sets with Pouch, 5 Ounce | 23 | 36 | 27 | 0.639 | 529.2 | 0.745 | có | không |
| 12 | [11212] SHANY Carry All Trunk Professional Makeup Kit - Eyeshadow,Pedicure,ma… | [11992] Cameo Carry All Beauty Case by Shany © 90pc Pro Make Up Set - Premium… | 24 | 41 | 25 | 0.585 | 523.6 | 0.773 | không | có |
| 13 | [11156] SHANY Acrylic Nail Brush Sets with Pouch, 5 Ounce | [11212] SHANY Carry All Trunk Professional Makeup Kit - Eyeshadow,Pedicure,ma… | 23 | 27 | 41 | 0.852 | 464.6 | 0.706 | có | không |
| 14 | [9619] SHANY Eyeshadow Kit, Metallic Runway, 96 Color | [9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | 25 | 35 | 36 | 0.714 | 443.7 | 0.704 | có | có |
| 15 | [11212] SHANY Carry All Trunk Professional Makeup Kit - Eyeshadow,Pedicure,ma… | [11583] SHANY Cosmetics Urban Gal Collection Brush Kit (15 Piece Vegan Travel… | 26 | 41 | 34 | 0.634 | 417.1 | 0.699 | có | không |
| 16 | [9046] SHANY Cosmetics Feather Hair Extension Kit, 8 Ounce (Feather Hair, Ho… | [11976] SHANY Jet Black Professional Rolling Makeup Case Premium Collection, … | 26 | 43 | 34 | 0.605 | 397.7 | 0.685 | không | không |
| 17 | [1231] Shea Butter - 100% Natural African Ultra Rich Raw Shea Nuts Including… | [11771] Best Bentonite Clay - Top Healing Clay - Finest Azteca Gold Blended B… | 26 | 40 | 37 | 0.650 | 392.9 | 0.676 | không | không |
| 18 | [5382] Axe Styling Spiked Up Look Gel, 6 Ounce | [5384] Axe Styling Messy Look Matte Gel, 6 Ounce | 29 | 38 | 45 | 0.763 | 379.3 | 0.704 | có | có |
| 19 | [9739] SHANY Professional Eyeshadow Kit, 180 Color, 180 Grams | [11212] SHANY Carry All Trunk Professional Makeup Kit - Eyeshadow,Pedicure,ma… | 25 | 36 | 41 | 0.694 | 378.8 | 0.652 | có | không |
| 20 | [9619] SHANY Eyeshadow Kit, Metallic Runway, 96 Color | [11212] SHANY Carry All Trunk Professional Makeup Kit - Eyeshadow,Pedicure,ma… | 24 | 35 | 41 | 0.686 | 374.0 | 0.636 | có | không |

Trong 20 luật: 8 luật cùng brand, 4 luật chung danh mục lá; số đếm X∪Y nhỏ nhất 23, lớn nhất 31.

## min_support = 0.002 (số đếm ≥ 45)

### 20 luật lift cao nhất (mỗi tập X ∪ Y chỉ giữ một luật)

| # | Tiền đề X | Hậu quả Y | Đếm X∪Y | Đếm X | Đếm Y | Confidence | Lift | Kulczynski | Cùng brand | Chung danh mục lá |
|---:|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 1 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[6436] SHANY 2012 Edition All In One Harmony Makeup Kit, 25 Ounce | [671] SHANY Stamping Nail Art Set (Nail Art Image Plate Holder, Scraper, St…<br>[834] SHANY Professional 13-Piece Cosmetic Brush Set with Pouch, Set of 12 … | 47 | 58 | 52 | 0.810 | 348.5 | 0.857 | có | không |
| 2 | [1809] Motions Straight Finish Sealer, 3.3 Ounce | [1810] Motions Straight Finish Leave in Conditioner, 8 Ounce | 45 | 59 | 62 | 0.763 | 275.1 | 0.744 | có | không |
| 3 | [671] SHANY Stamping Nail Art Set (Nail Art Image Plate Holder, Scraper, St…<br>[834] SHANY Professional 13-Piece Cosmetic Brush Set with Pouch, Set of 12 … | [6436] SHANY 2012 Edition All In One Harmony Makeup Kit, 25 Ounce | 49 | 52 | 77 | 0.942 | 273.7 | 0.789 | có | không |
| 4 | [1808] Motions Straight Finish Cleanser, 8 Ounce | [1809] Motions Straight Finish Sealer, 3.3 Ounce | 47 | 67 | 59 | 0.701 | 265.9 | 0.749 | có | không |
| 5 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[6436] SHANY 2012 Edition All In One Harmony Makeup Kit, 25 Ounce | [671] SHANY Stamping Nail Art Set (Nail Art Image Plate Holder, Scraper, St… | 48 | 58 | 73 | 0.828 | 253.5 | 0.743 | có | không |
| 6 | [1808] Motions Straight Finish Cleanser, 8 Ounce | [1810] Motions Straight Finish Leave in Conditioner, 8 Ounce | 45 | 67 | 62 | 0.672 | 242.3 | 0.699 | có | không |
| 7 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[834] SHANY Professional 13-Piece Cosmetic Brush Set with Pouch, Set of 12 … | [6436] SHANY 2012 Edition All In One Harmony Makeup Kit, 25 Ounce | 54 | 67 | 77 | 0.806 | 234.1 | 0.754 | có | không |
| 8 | [670] SHANY 2012 Nail Art Polish Stamp Manicure Image Plates set of 25pcs<br>[834] SHANY Professional 13-Piece Cosmetic Brush Set with Pouch, Set of 12 … | [671] SHANY Stamping Nail Art Set (Nail Art Image Plate Holder, Scraper, St… | 50 | 67 | 73 | 0.746 | 228.6 | 0.716 | có | không |
| 9 | [2891] Just For Me Hair Milk Styling Creme, 8-Ounce Bottle | [2892] Just For Me Hair Milk Smoothing Edges Crème, 4-Ounce Container | 45 | 69 | 70 | 0.652 | 208.4 | 0.648 | có | có |
| 10 | [671] SHANY Stamping Nail Art Set (Nail Art Image Plate Holder, Scraper, St… | [6436] SHANY 2012 Edition All In One Harmony Makeup Kit, 25 Ounce | 51 | 73 | 77 | 0.699 | 202.9 | 0.680 | có | không |
| 11 | [1078] Organic Rosehip Oil 4OZ ★SALE★ 100% Pure Rosehip Seed Oil & USDA Cert… | [1079] Vitamin C Serum For Face 20% ★Weekend SALE★ Organic Serum + Hyaluroni… | 56 | 86 | 79 | 0.651 | 184.3 | 0.680 | không | có |
| 12 | [3025] Dead Sea Salt Soap - Face and Body Soap - 100% Natural Soap - with De…<br>[6466] Dead Sea Salt Deep Hair Conditioner - for Dry or Damaged Hair - Great… | [9744] Adovia Facial Serum - Anti Aging, Skin Lifting, Facial Serum with Vit… | 47 | 55 | 104 | 0.855 | 183.8 | 0.653 | không | không |
| 13 | [3548] Eye Cream For Dark Circles, Puffiness, Wrinkles & Bags - 1.7 OZ / Bes… | [3549] Anti-Aging Facial Moisturizer Cream 3.4 OZ For Face, Under Eye, Oily,… | 50 | 88 | 72 | 0.568 | 176.5 | 0.631 | không | không |
| 14 | [3025] Dead Sea Salt Soap - Face and Body Soap - 100% Natural Soap - with De… | [9744] Adovia Facial Serum - Anti Aging, Skin Lifting, Facial Serum with Vit… | 64 | 83 | 104 | 0.771 | 165.8 | 0.693 | không | không |
| 15 | [3515] Axe Spiking Glue Spiked Up Look, 3.2 Ounce | [3516] Axe Styling Gum Messy Look, 3.2 Ounce | 62 | 94 | 96 | 0.660 | 153.6 | 0.653 | có | có |
| 16 | [3023] Dead Sea Mud Shampoo - Fast, Natural Relief from Dandruff - Itchy, Dr… | [6466] Dead Sea Salt Deep Hair Conditioner - for Dry or Damaged Hair - Great…<br>[10747] Dead Sea Mud Facial Mask and Deep Facial Cleanser by Adovia - Deep Po… | 56 | 136 | 60 | 0.412 | 153.5 | 0.673 | không | không |
| 17 | [385] Clear Scalp and Hair Beauty Therapy Frizz Control Nourishing Shampoo,… | [386] Clear Scalp and Hair Beauty Therapy Frizz Control Nourishing Daily Co… | 57 | 92 | 91 | 0.620 | 152.3 | 0.623 | có | không |
| 18 | [3023] Dead Sea Mud Shampoo - Fast, Natural Relief from Dandruff - Itchy, Dr…<br>[6464] Dead Sea Mud Facial Mask - Ancient Natural Facial Mask and Skin Care … | [6466] Dead Sea Salt Deep Hair Conditioner - for Dry or Damaged Hair - Great… | 49 | 60 | 120 | 0.817 | 152.2 | 0.612 | không | không |
| 19 | [1079] Vitamin C Serum For Face 20% ★Weekend SALE★ Organic Serum + Hyaluroni… | [3021] ORGANIC Argan Oil 4OZ ★SALE★ 100% PURE Argan Oil for Hair, Face, Skin… | 60 | 79 | 114 | 0.759 | 149.0 | 0.643 | không | không |
| 20 | [375] Suave Professionals Moroccan Infusion Color Care Shampoo 28 oz | [376] Suave Professionals Moroccan Infusion Color Care Conditioner 28 oz | 58 | 99 | 88 | 0.586 | 148.9 | 0.622 | có | không |

Trong 20 luật: 13 luật cùng brand, 3 luật chung danh mục lá; số đếm X∪Y nhỏ nhất 45, lớn nhất 64.

### 20 luật lift cao nhất dạng một item ⇒ một item

| # | Tiền đề X | Hậu quả Y | Đếm X∪Y | Đếm X | Đếm Y | Confidence | Lift | Kulczynski | Cùng brand | Chung danh mục lá |
|---:|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 1 | [1809] Motions Straight Finish Sealer, 3.3 Ounce | [1810] Motions Straight Finish Leave in Conditioner, 8 Ounce | 45 | 59 | 62 | 0.763 | 275.1 | 0.744 | có | không |
| 2 | [1808] Motions Straight Finish Cleanser, 8 Ounce | [1809] Motions Straight Finish Sealer, 3.3 Ounce | 47 | 67 | 59 | 0.701 | 265.9 | 0.749 | có | không |
| 3 | [1808] Motions Straight Finish Cleanser, 8 Ounce | [1810] Motions Straight Finish Leave in Conditioner, 8 Ounce | 45 | 67 | 62 | 0.672 | 242.3 | 0.699 | có | không |
| 4 | [2891] Just For Me Hair Milk Styling Creme, 8-Ounce Bottle | [2892] Just For Me Hair Milk Smoothing Edges Crème, 4-Ounce Container | 45 | 69 | 70 | 0.652 | 208.4 | 0.648 | có | có |
| 5 | [671] SHANY Stamping Nail Art Set (Nail Art Image Plate Holder, Scraper, St… | [6436] SHANY 2012 Edition All In One Harmony Makeup Kit, 25 Ounce | 51 | 73 | 77 | 0.699 | 202.9 | 0.680 | có | không |
| 6 | [1078] Organic Rosehip Oil 4OZ ★SALE★ 100% Pure Rosehip Seed Oil & USDA Cert… | [1079] Vitamin C Serum For Face 20% ★Weekend SALE★ Organic Serum + Hyaluroni… | 56 | 86 | 79 | 0.651 | 184.3 | 0.680 | không | có |
| 7 | [3548] Eye Cream For Dark Circles, Puffiness, Wrinkles & Bags - 1.7 OZ / Bes… | [3549] Anti-Aging Facial Moisturizer Cream 3.4 OZ For Face, Under Eye, Oily,… | 50 | 88 | 72 | 0.568 | 176.5 | 0.631 | không | không |
| 8 | [3025] Dead Sea Salt Soap - Face and Body Soap - 100% Natural Soap - with De… | [9744] Adovia Facial Serum - Anti Aging, Skin Lifting, Facial Serum with Vit… | 64 | 83 | 104 | 0.771 | 165.8 | 0.693 | không | không |
| 9 | [3515] Axe Spiking Glue Spiked Up Look, 3.2 Ounce | [3516] Axe Styling Gum Messy Look, 3.2 Ounce | 62 | 94 | 96 | 0.660 | 153.6 | 0.653 | có | có |
| 10 | [385] Clear Scalp and Hair Beauty Therapy Frizz Control Nourishing Shampoo,… | [386] Clear Scalp and Hair Beauty Therapy Frizz Control Nourishing Daily Co… | 57 | 92 | 91 | 0.620 | 152.3 | 0.623 | có | không |
| 11 | [1079] Vitamin C Serum For Face 20% ★Weekend SALE★ Organic Serum + Hyaluroni… | [3021] ORGANIC Argan Oil 4OZ ★SALE★ 100% PURE Argan Oil for Hair, Face, Skin… | 60 | 79 | 114 | 0.759 | 149.0 | 0.643 | không | không |
| 12 | [375] Suave Professionals Moroccan Infusion Color Care Shampoo 28 oz | [376] Suave Professionals Moroccan Infusion Color Care Conditioner 28 oz | 58 | 99 | 88 | 0.586 | 148.9 | 0.622 | có | không |
| 13 | [923] Nexxus Youth Renewal Rejuvenating Shampoo, 13.5 Ounce | [1174] Nexxus Youth Renewal Rejuvenating Conditioner, 13.5 Ounce | 51 | 90 | 89 | 0.567 | 142.4 | 0.570 | có | không |
| 14 | [3549] Anti-Aging Facial Moisturizer Cream 3.4 OZ For Face, Under Eye, Oily,… | [9744] Adovia Facial Serum - Anti Aging, Skin Lifting, Facial Serum with Vit… | 46 | 72 | 104 | 0.639 | 137.4 | 0.541 | không | không |
| 15 | [3548] Eye Cream For Dark Circles, Puffiness, Wrinkles & Bags - 1.7 OZ / Bes… | [9744] Adovia Facial Serum - Anti Aging, Skin Lifting, Facial Serum with Vit… | 54 | 88 | 104 | 0.614 | 131.9 | 0.566 | không | không |
| 16 | [702] Axe Gravity Defying Dust Messy Look, 0.035 Ounce | [3516] Axe Styling Gum Messy Look, 3.2 Ounce | 53 | 94 | 96 | 0.564 | 131.3 | 0.558 | có | không |
| 17 | [3023] Dead Sea Mud Shampoo - Fast, Natural Relief from Dandruff - Itchy, Dr… | [5595] Konjac Sponge - Activated Charcoal - Facial Sponge, 100% Natural Spon… | 46 | 136 | 58 | 0.338 | 130.4 | 0.566 | không | không |
| 18 | [1078] Organic Rosehip Oil 4OZ ★SALE★ 100% Pure Rosehip Seed Oil & USDA Cert… | [3021] ORGANIC Argan Oil 4OZ ★SALE★ 100% PURE Argan Oil for Hair, Face, Skin… | 57 | 86 | 114 | 0.663 | 130.0 | 0.581 | không | không |
| 19 | [702] Axe Gravity Defying Dust Messy Look, 0.035 Ounce | [3515] Axe Spiking Glue Spiked Up Look, 3.2 Ounce | 50 | 94 | 94 | 0.532 | 126.5 | 0.532 | có | không |
| 20 | [6466] Dead Sea Salt Deep Hair Conditioner - for Dry or Damaged Hair - Great… | [9744] Adovia Facial Serum - Anti Aging, Skin Lifting, Facial Serum with Vit… | 70 | 120 | 104 | 0.583 | 125.4 | 0.628 | không | không |

Trong 20 luật: 11 luật cùng brand, 3 luật chung danh mục lá; số đếm X∪Y nhỏ nhất 45, lớn nhất 70.
