# Kế hoạch thực nghiệm bổ sung cho paper (theo nhận xét của giảng viên)

Nhận xét: "nhóm cần thêm minh chứng method của nhóm hiệu quả trên các dataset khác, phần test đang lặp lại có khi nào bị leak không, làm thêm thực nghiệm cho paper, bài method thêm ablation study để mạnh thêm".

## 1. Hiệu quả trên dataset khác

Repo này chấm zero-shot trên FakeAVCeleb, AV-Deepfake1M (val), LAV-DF (test), MAVOS-DD (test).

Dùng weight nào:
- **Model của mình (M3-R, M2, BL-\*)**: dùng weight của mình, train trên VN-AV-DF, không train lại. Đây là câu hỏi "method có tổng quát hoá không". Nếu train lại trên FakeAVCeleb rồi test FakeAVCeleb thì chỉ đo được in-domain, không trả lời được câu này.
- **Model đã công bố**: dùng weight của tác giả (AVH-Align checkpoint AV-1M; SpeechForensics không cần train). Tự train lại baseline thì người đọc sẽ hỏi mình có train kém đi không. Weight gốc tránh được nghi ngờ đó.
- **Số các paper khác báo cáo** (AVH-Align Bảng 2, MAVOS-DD Bảng 3): chỉ đưa vào bảng bối cảnh riêng (mục 10 của report), không xếp hạng chung, vì khác tập test và khác tiền xử lý.
- **Tuỳ chọn, chưa làm**: fine-tune M3-R trên train split của FakeAVCeleb rồi so với số in-domain của paper khác. Việc này cần chuyển nhãn sang 3 đầu (visual / audio / mismatch) và chạy train của repo chính trên dữ liệu ngoài.

Những điều cần nói thẳng trong paper (mục 11 của report): AV-HuBERT đã pre-train trên VoxCeleb2, nguồn của 3/4 dataset ngoài; Φ khởi tạo từ checkpoint AVH-Align train trên real của AV-1M; model của mình chỉ thấy tiếng Việt.

## 2. Test lặp lại có rò không

Đã chạy trên dữ liệu thật (`docs/vnavdf_audit.md`, 27/11/2026):
- 3 part, 23.242 file. Giữa train, validation và test **không trùng** người nói (120), video YouTube nguồn (164), clip real (2.400), hash file, nguồn audio donor, hay câu thoại (2.400).
- 1.734 clip mượn audio từ video khác; không clip nào mượn chéo split.
- `test-log.json` (có từ v005): 16 lần chấm test trong 2 phiên. Mỗi checkpoint đã train chỉ chấm test **một lần**. SpeechForensics không có weight nên xuất hiện ở cả 2 vòng; audit không tính nó.
- Lối tắt một đặc trưng: AUC toàn bộ 0,47-0,53 ở cả 3 part. Riêng ô giọng AI toàn câu (voice/full) so với real: khoảng lặng đầu 0,63-0,69, khoảng lặng cuối 0,34-0,58. Lối tắt yếu này chỉ có ở một ô, nên nêu trong paper.

Audit không trả lời được: các vòng v003-v004 chưa có test-log, và quyết định thiết kế (chọn kiến trúc giữa các vòng) có thể đã bị số test ảnh hưởng. Câu trả lời mạnh nhất cho phần này là kết quả trên dataset ngoài: mô hình chưa từng chạm vào các tập đó.

## 3. Ablation cho bài method

| Ablation | Cần train lại? | Trạng thái |
|---|---|---|
| Bỏ cặp sạch ↔ nén (`trihead_v5_no_pair`, cổng R6) | có | đã có, 2 seed |
| Tắt từng nhánh lúc test: trace, sync tương đối, sync tuyệt đối, kênh Φ, audio | không | repo này, trên mọi dataset |
| Cổng độ phân giải bật / tắt | không | repo này |
| Chỉ train bản sạch (`trihead_v5_clean`, R4) | có | có spec trong repo chính, chưa chạy |
| Thêm nhánh chất lượng (`trihead_v5_quality`, R5) | có | có spec, chưa chạy |
| Bỏ Φ AVH-Align, bỏ đầu loại thao tác, định tuyến tuyệt đối | có | spec v004 (`trihead_v4_no_phi`, `_no_type`, `_abs`); muốn dùng cho M3-R phải thêm spec v5 tương ứng |
| Bỏ sham làm âm khó (`sham_weight = 0`), bỏ layer drop XLS-R | có | chưa có spec |

Đề xuất thứ tự nếu còn quota GPU:
1. Chạy repo này: không tốn train, có ngay bảng tắt nhánh trên 5 tập.
2. `trihead_v5_clean` (đã có spec).
3. Thêm 2 spec v5 (bỏ Φ, bỏ sham) vào repo chính rồi train 1 seed.

Khác biệt giữa hai kiểu ablation phải ghi rõ trong paper:
- Tắt nhánh lúc test cho biết đầu ra **đang dựa** vào nhánh nào.
- Ablation train lại cho biết nhánh đó có **cần thiết** không.
