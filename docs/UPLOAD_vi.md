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

Repo để public: notebook clone từ `CROSS_URL = "https://github.com/VN-AV-DF-Capstone/vn-av-df-crossdata.git"`. Đặt tên hoặc tài khoản khác thì sửa dòng này trong `tools/make_notebook.py` rồi chạy lại `python tools/make_notebook.py`. Clone lỗi thì notebook dùng dataset `linhxm/vn-av-df-crossdata-code` (nếu có attach).

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

## 3. Notebook trên Kaggle (đã tạo sẵn)

`tools/create_kaggle_notebooks.py` đã tạo 10 notebook private, mới chỉ lưu, chưa chạy:
- input, accelerator, Internet và cell 2 đã điền sẵn;
- code lấy bằng `git clone` từ GitHub, không cần dataset code.

| Notebook | Accelerator | Input đã gắn | Cell 2 |
|---|---|---|---|
| `vn-av-df-crossdata-<tập>-cpu` (polyglotfake, fakeavceleb, lavdf, avdeepfake1m) | None | tập đó + `linhxm/vn-av-df-weights-2seeds` | `SESSION = "cpu"`, `ONLY = ["<tập>"]` |
| `vn-av-df-crossdata-<tập>-gpu` | T4 ×2 | như trên + Output của notebook `-cpu` cùng tập | `SESSION = "gpu"`, `PREVIOUS_WORK` = Output đó |
| `vn-av-df-crossdata-report` | None | Output của 4 notebook `-gpu` | `SESSION = "report"` (tập nào chưa chạy thì tự bỏ qua) |
| `vn-av-df-crossdata-audit` | None | 3 part VN-AV-DF | `SESSION = "audit"` |

Mỗi notebook làm như sau:
1. Mở notebook → Add-ons → Secrets → bật `WANDB_API_KEY`. API không gắn secret được; thiếu secret thì notebook dừng ngay và báo rõ lý do.
2. Save Version → Save & Run All.
3. Thứ tự: 4 notebook `-cpu` (chạy song song được) → notebook `-gpu` của tập đã chạy xong `-cpu` → `report`.
4. Trước khi chạy `-gpu`, kiểm ô Input: Output của notebook `-cpu` phải là version mới nhất. Nếu Kaggle hiện nút Update thì bấm.

Tạo lại notebook, hoặc tạo cho tài khoản khác: `python tools/create_kaggle_notebooks.py --owner <user>` (thêm `--dry-run` để xem trước).

## 4. Kết quả

- Cell 5 của notebook hiển thị `report.md` (tiếng Anh) và mọi biểu đồ.
- Output `crossdata/results/` chứa `metrics.json`, `metrics.csv`, `tables/`, `figures/`. Tải về bằng `kaggle kernels output linhxm/<notebook> -p <thư mục>`.
- W&B: project `vn-av-df-crossdata`, có một run cho mỗi bước và mỗi dataset.
