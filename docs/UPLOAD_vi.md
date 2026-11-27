# Đưa repo lên GitHub và chạy trên Kaggle

## 1. Tạo repo GitHub

Trên GitHub, tạo repo **private** tên `vn-av-df-crossdata` (org `VN-AV-DF-Capstone` hoặc tài khoản `linhxm`). Không thêm README hay .gitignore vì repo đã có sẵn. Sau đó chạy:

```bash
cd D:/COMP/RESEARCH/Deepfake_VN/vn-av-df-crossdata
git init -b main
git add .
git status            # kiểm: không có work/, *.mp4, *.npz (đã nằm trong .gitignore)
git commit -m "Cross-dataset evaluation of VN-AV-DF detectors: adapters, scoring, metrics, report, leakage audit"
git remote add origin https://github.com/<org-hoặc-user>/vn-av-df-crossdata.git
git push -u origin main
```

Repo để public: notebook clone từ `CROSS_URL = "https://github.com/linhxm/vn-av-df-crossdata.git"`. Đặt tên hoặc tài khoản khác thì sửa dòng này trong `tools/make_notebook.py` rồi chạy lại `python tools/make_notebook.py`. Clone lỗi thì notebook dùng dataset `linhxm/vn-av-df-crossdata-code` (nếu có attach).

## 2. Dataset trên Kaggle (private)

| Dataset | Nội dung | Tạo / cập nhật |
|---|---|---|
| `linhxm/vn-av-df-crossdata-code` | code repo này | đã tạo 27/11. Sửa code xong thì cập nhật bằng lệnh bên dưới |
| `linhxm/vn-av-df-weights-2seeds` | 18 thư mục checkpoint, seed 42 (v005) + 43 (v006), 138 MB | đã tạo 27/11, bằng `tools/export_weights.py` |
| `linhxm/vn-av-df-data-part1-trim04-fake`, `-part2-fake` | cho phiên audit | đã có |
| `aicontentdetections/fakeavceleb-v1-2`, `aicontentdetections/av-deepfake1m`, `elin75/localized-audio-visual-deepfake-dataset-lav-df`, `debajyatidey/polyglotfake` | dataset ngoài (bản public trên Kaggle) | không cần upload: Add Input theo tên (xem RUNBOOK_vi.md mục 0) |

Cập nhật dataset code sau khi sửa (chạy từ thư mục `Deepfake_VN`):

```bash
rm -rf kaggle_upload/vn-av-df-crossdata-code/{src,configs,tools,tests,notebooks,docs}
tar -C vn-av-df-crossdata --exclude=work --exclude=__pycache__ --exclude=.ruff_cache -cf - . | tar -C kaggle_upload/vn-av-df-crossdata-code -xf -
cd kaggle_upload && kaggle datasets version -p vn-av-df-crossdata-code -m "update" --dir-mode zip
# chạy từ trong kaggle_upload: CLI 2.2.4 lỗi với file .gitignore khi -p có dấu "/"
```

## 3. Tạo notebook chạy thật (trên web Kaggle)

1. Vào Kaggle, chọn Create → Notebook → File → Import notebook, rồi chọn `notebooks/crossdata.ipynb`.
2. Trong Add Input, thêm:
   - `linhxm/vn-av-df-crossdata-code`;
   - `linhxm/vn-av-df-weights-2seeds`;
   - tập ngoài của phiên đó (RUNBOOK_vi.md mục 0).
3. Settings: bật Internet (để tải AV-HuBERT, XLS-R, checkpoint AVH-Align). Trong Add-ons → Secrets, bật `WANDB_API_KEY`.
4. Sửa ở cell 2 các dòng: `DATASETS`, `ONLY`, `PER_STRATUM`, `SESSION`.
5. Chạy theo thứ tự (mỗi lần dùng Save Version → Save & Run All):

| Lần | SESSION | Accelerator | Input thêm |
|---|---|---|---|
| 1 | `cpu` | None | |
| 2 | `gpu` | GPU T4 ×2 | Output lần 1 (Add Input → Your Work → notebook đó); đặt `PREVIOUS_WORK = Path("/kaggle/input/notebooks/linhxm/<tên notebook lần 1>/crossdata")` |
| 3 | `audit` | None | 3 dataset part VN-AV-DF |

Nên dùng mỗi lần một notebook riêng (vd. `vn-av-df-crossdata-cpu`, `-gpu`, `-audit`), để Output lần 1 còn nguyên khi chạy lần 2.

Kiểm quota GPU trước lần 2: 6 giờ mỗi tuần, reset 00:00 UTC thứ Bảy.
- Chạy thử 24 video (`vn-av-df-crossdata-smoke`) đo được thời gian từng bước; xem log của notebook đó.
- Có thể tạo notebook kiểu khác: `python tools/make_notebook.py --out <file> --set SESSION='"gpu"' --set ONLY='["fakeavceleb"]'`.

## 4. Kết quả

- Cell 5 của notebook hiển thị `report.md` (tiếng Anh) và mọi biểu đồ.
- Output `crossdata/results/` chứa `metrics.json`, `metrics.csv`, `tables/`, `figures/`. Tải về bằng `kaggle kernels output linhxm/<notebook> -p <thư mục>`.
- W&B: project `vn-av-df-crossdata`, có một run cho mỗi bước và mỗi dataset.
