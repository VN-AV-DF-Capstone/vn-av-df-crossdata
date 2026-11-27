# Hướng dẫn chạy (tiếng Việt)

## 0. Dữ liệu: các tập sẽ test

Cả 4 tập đều có sẵn trên Kaggle; chỉ cần Add Input, không phải upload. Đã kiểm ngày 28/11/2026: thư mục, file nhãn và một video mẫu của từng tập.

| Key | Tập | Kaggle input | Lấy mẫu | Bị sửa gì | Độ phân giải |
|---|---|---|---|---|---|
| `fakeavceleb` | FakeAVCeleb v1.2: 500 real, 21.066 fake, tiếng Anh | `aicontentdetections/fakeavceleb-v1-2` | 500 / lớp × 4 (RVRA, FVRA, RVFA, FVFA) | face swap (FaceSwap, FSGAN), Wav2Lip, giọng clone SV2TTS | crop mặt VoxCeleb2 |
| `avdeepfake1m` | AV-Deepfake1M **val**: 57.340 video (test không công bố nhãn) | `aicontentdetections/av-deepfake1m` | 500 / lớp × 4 (real, visual, audio, both) | sửa vài từ: TalkLip + VITS / YourTTS | 224 × 224 |
| `lavdf` | LAV-DF **test**: 26.100 video | `elin75/localized-audio-visual-deepfake-dataset-lav-df` | 500 / lớp × 4 | sửa vài từ, hình và / hoặc tiếng | 224 × 224 |
| `polyglotfake` | PolyGlotFake: 762 real, 14.472 fake, 7 ngôn ngữ | `debajyatidey/polyglotfake` | 100 / (lớp × ngôn ngữ) | dịch + clone giọng (XTTS, Bark, MicroTTS, Vall-E-X, Tacotron) + lip-sync (VideoRetalking, Wav2Lip) | 720p |

Ghi chú:
- AV-Deepfake1M trên Kaggle để val dưới dạng 20 file zip 1 GB. `archive.py` đọc thẳng các phần zip và chỉ trích video được lấy mẫu, không giải nén cả bộ.
- MAVOS-DD không có bản trên Kaggle. Bản gốc trên HuggingFace cần chấp nhận điều khoản và nặng 308 GB, nên bỏ qua. PolyGlotFake thay vai trò tập đa ngôn ngữ.
- LAV-DF và AV-Deepfake1M ở 224 × 224, thấp hơn cổng 360p của M3-R. Khi bật cổng, M3-R chỉ còn đầu audio, nên báo cáo luôn có cả hai chế độ bật và tắt cổng.
- PolyGlotFake là tập ngoài duy nhất ở 720p. Kiểu giả của nó (giọng TTS + lip-sync) cũng gần VN-AV-DF nhất, nên đây là phép thử đầy đủ nhất cho M3-R.
- Các bản Kaggle do người khác upload. Điều khoản gốc vẫn áp dụng: chỉ dùng cho nghiên cứu, không phân phối lại. Trong paper phải trích dẫn bài gốc của từng tập.

## 1. Weight

- Hai seed: Kaggle dataset `linhxm/vn-av-df-weights-2seeds` (private, 138 MB), gom bằng `tools/export_weights.py`.
  - Seed 42 lấy từ Output test v005, seed 43 từ Output `vn-av-df-train-v006-test`.
  - Gồm: M3-R, M3-R bỏ cặp sạch ↔ nén, M2, BL-M/sup, BL-A2, BL-M, SpeechForensics, BL-V, BL-Ens-AV.
  - Ngưỡng BL-Ens-AV dựng lại khớp `metrics.json` ở cả hai seed: 3,3362 và 4,2606.
- AVH-Align bản công bố: notebook tự tải `AVH-Align_AV1M.pt` (commit đã pin trong `settings.py` của repo chính).

## 2. Chạy trên Kaggle (`notebooks/crossdata.ipynb`)

Code lấy từ GitHub public (`CROSS_URL`). Nếu clone lỗi, notebook dùng dataset `linhxm/vn-av-df-crossdata-code` khi dataset đó được attach.

Mỗi tập ngoài cần khoảng 11 GB cho 2.000 video, nên không gộp được mọi tập vào một Output 20 GB. Con số này ước từ kích thước mảng đặc trưng, chưa đo trên Kaggle. Chạy mỗi tập 2 phiên, rồi một phiên gộp báo cáo:

| Bước | SESSION | ONLY | Accelerator | Input thêm |
|---|---|---|---|---|
| 1 | `cpu` | `["fakeavceleb"]` (lần lượt từng tập) | None | tập đó + weights-2seeds |
| 2 | `gpu` | cùng tập | T4 ×2 | Output bước 1 (`PREVIOUS_WORK = [...]`) + weights-2seeds |
| 3 | `report` | `None` | None | Output mọi phiên bước 2 (`PREVIOUS_WORK = [..., ...]`) |
| riêng | `audit` | | None | 3 part VN-AV-DF |

Thứ tự nên chạy:
1. `polyglotfake`.
2. `fakeavceleb`.
3. `lavdf`.
4. `avdeepfake1m`.

Không chạy lại test set VN-AV-DF:
- Mỗi checkpoint đã chấm test đúng một lần (`docs/vnavdf_audit.md`). Cột "VN-AV-DF (in-domain)" của báo cáo lấy thẳng từ `metrics.json` của lần đó.
- Đường ống mới đã được kiểm một lần ở bước chạy thử ngày 27/11 (24 video, bên dưới).
- Adapter `vnavdf` vẫn còn nhưng mặc định tắt (`root: null`). Chỉ bật khi sửa đường ống, và không đưa số của nó vào paper.

Kiểm đường ống ngày 27/11/2026 (24 video test VN-AV-DF, cả CPU lẫn T4 ×2) không có lỗi. AUC của M3-R là 0,961 / 1,000 (seed 42 / 43); số công bố 0,973 / 0,971. Thời gian từng encoder cho 24 video, tính bằng giây, đã gồm lúc nạp model:

| Bước | CPU | GPU |
|---|---|---|
| landmarks | 212 | 158 |
| avhubert | 185 | 153 |
| mouth | 11 | 10 |
| xlsr | 88 | 57 |

Với số video ít, thời gian nạp model chiếm phần lớn, nên chưa suy ra được tốc độ cho 2.000 video. Lần chạy thật đầu tiên nên xem `extract/<encoder>/samples_per_min` trong log để chia phiên.

## 3. Đọc kết quả

- `results/report.md` (cũng hiện trong notebook, cell 5) và W&B project `vn-av-df-crossdata`.
- Mục 3 và 3b: AUC zero-shot, từng seed và trung bình 2 seed. Mục 6: so cặp với M3-R trên cùng video, CI bootstrap theo người.
- Mục 5: ngưỡng VN chuyển sang tập mới. FPR trên real thường lệch xa 5 % khi đổi ngôn ngữ hay nguồn video, nên báo thêm ngưỡng hiệu chỉnh lại.
- Mục 7: kiểm lối tắt khoảng lặng đầu. Mục 8: cổng độ phân giải. Mục 9: ablation tắt nhánh và ablation bỏ cặp (train lại, 2 seed).
- `audit.md`: câu trả lời cho câu hỏi "test lặp lại có rò không" (xem `docs/vnavdf_audit.md`).

## 4. Chạy local (máy có GPU, hoặc CPU để thử)

```bash
pip install -e ../vn-av-df-forensics[data,training] -e .[mavos,wandb,test]
python -m vnav_cross run --work work --main-root ../vn-av-df-forensics --datasets polyglotfake
pytest
```

Local cần môi trường worker AV-HuBERT của repo chính (`environments/`).
