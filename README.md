# VN-AV-DF Cross-dataset Evaluation

**Do the VN-AV-DF detectors (M3-R, M2) still work on other datasets, how do they compare with published detectors run with their authors' weights, and could the in-domain test numbers come from a leak?**

This repository answers three review questions on the VN-AV-DF method paper:

| Review question | What this repository runs | Output |
|---|---|---|
| Evidence that the method works on other datasets | zero-shot scoring of M3-R, M2 and the baselines on FakeAVCeleb, AV-Deepfake1M (val), LAV-DF (test), MAVOS-DD (test); same videos for every detector | Sections 1-6 of `report.md` |
| Comparison with published models | AVH-Align with the released checkpoint, SpeechForensics (training-free); numbers reported by other papers in a separate context table | Sections 3, 6, 10 |
| Is the repeated test set leaking? | split-overlap audit (speaker, source video, clip, hash, donor audio, sentence), test-use log, single-feature shortcut AUCs; leading-silence trimming on external sets | `audit.md`, Section 7 |
| Stronger ablation | test-time branch knockout of the trained M3-R (trace, sync relative / absolute, AVH-Align phi channel, audio) on every dataset, plus the resolution gate on / off | Sections 8-9 |

The detectors, encoders and checkpoint loader come from [vn-av-df-forensics](https://github.com/linhxm/vn-av-df-forensics) (installed next to this repository). Nothing is re-implemented: features are extracted with the same AV-HuBERT / XLS-R / mouth-crop encoders and loaded with the same `BenchData`, so a score here is the score the trained model gives.

## Protocols

| Protocol | Definition |
|---|---|
| Zero-shot | VN-AV-DF checkpoints of both training seeds (42 from v005, 43 from v006) applied unchanged; no target-domain training. Tables give each seed and the seed mean with its range |
| Published weights | `avh_align_official`: AVH-Align (Smeu et al., CVPR 2025) with the checkpoint trained on 45k AV-Deepfake1M real videos; SpeechForensics (Liang et al., NeurIPS 2024) needs no training |
| Common set | every detector is evaluated on the videos that all detectors could score; videos lost to decoding, face tracking or missing features are counted per class |
| Windows | 7 s windows, half overlap, video score = highest window (as in the demo / training clips) |
| Resolution gate | M3-R ships with a 360p gate (below it, the audio head decides). Reported on and off; BL-Ens-AV likewise |
| Transfer threshold | the VN-AV-DF threshold (5% FPR on its validation reals) applied unchanged: FPR on target reals shows calibration drift |
| Calibrated threshold | re-fitted at 5% FPR on the reals of a subject-disjoint 30% calibration part of the target; FPR / TPR on the other 70% |
| Paired comparison | AUC(M3-R) - AUC(other) on the same videos, 95% CI by bootstrap over subjects (1000 replicates) |
| Leading-silence check | the leading silence (|x| > 5e-4, rounded up to 1/25 s; AVH-Align's definition) removed from every frame stream before scoring, plus single-feature AUCs (silence, duration, resolution) |
| Test-time ablation | one M3-R branch output set to zero before the heads (trained weights unchanged) |
| Retrained ablation | `trihead_v5_no_pair` (M3-R without the clean <-> re-encoded pairing), both seeds, paired against M3-R of the same seed |
| Pipeline check | done once (27/11/2026, 24 VN-AV-DF test videos, CPU and GPU): scores through this normalize / extract / window path matched the published in-domain AUC. The `vnavdf` adapter stays OFF by default: the VN-AV-DF test split is not re-used, and the in-domain column of the report comes from the checkpoints' own `metrics.json` |

## Datasets

Public Kaggle mirrors, checked on 28/11/2026 (folders, label files, a sample video). The original terms still apply: research use only, no redistribution. Cite the original papers.

| Key | Dataset | Kaggle input | Sample | Manipulations | Tasks |
|---|---|---|---|---|---|
| `fakeavceleb` | FakeAVCeleb v1.2 (Khalid et al., NeurIPS 2021 D&B): 500 real, 21,066 fake | `aicontentdetections/fakeavceleb-v1-2` | 500 per class (RVRA, FVRA, RVFA, FVFA) | FaceSwap, FSGAN, Wav2Lip, SV2TTS voice cloning | AI?, visual, audio |
| `avdeepfake1m` | AV-Deepfake1M validation split (Cai et al., ACM MM 2024): 57,340 videos | `aicontentdetections/av-deepfake1m` (val in 20 zip volumes, read in place by `archive.py`) | 500 per class | word-level edits, visual / audio / both | AI?, visual, audio |
| `lavdf` | LAV-DF test split (Cai et al., DICTA 2022): 26,100 videos | `elin75/localized-audio-visual-deepfake-dataset-lav-df` | 500 per class | word-level edits, visual / audio / both | AI?, visual, audio |
| `polyglotfake` | PolyGlotFake (Hou et al., arXiv 2405.08838): 762 real, 14,472 fake, 7 languages | `debajyatidey/polyglotfake` | 100 per class x language | translated cloned voice (XTTS, Bark, MicroTTS, Vall-E-X, Tacotron) + lip-sync (VideoRetalking, Wav2Lip) | AI?, visual, audio |
| `mavosdd` | MAVOS-DD test (Croitoru et al., 2025) | none (HF gated, 308 GB) | 60 per cell | talking head, face swap, lip-sync; real audio | AI?, visual |
| `folder` | any `<root>/real/**`, `<root>/fake/**` | yours | 1000 per class | unknown | AI? |

LAV-DF and AV-Deepfake1M are 224 x 224, below the 360p gate of M3-R; PolyGlotFake (720p) is the only external set where the full M3-R (visual branch included) runs with the gate on.

## Run on Kaggle

`notebooks/crossdata.ipynb` (regenerate with `python tools/make_notebook.py`):

Inputs: `linhxm/vn-av-df-weights-2seeds` (checkpoints of both seeds, `tools/export_weights.py`) and the external datasets; the code is cloned from GitHub (`linhxm/vn-av-df-crossdata-code` as a fallback). Step-by-step upload guide: `docs/UPLOAD_vi.md`.

1. `SESSION = "cpu"` (Accelerator None): sample, normalize, leading silence, landmarks, mouth crops.
2. `SESSION = "gpu"` (T4 x2), `PREVIOUS_WORK` = Output of step 1: AV-HuBERT, XLS-R, scoring, evaluation, report.
3. `SESSION = "audit"` (Accelerator None): VN-AV-DF leakage audit from the three part datasets.

W&B project `vn-av-df-crossdata` (private): one run per step and dataset with configs, coverage, failures, architecture and parameter tables, metric tables, every figure and a `crossdata-results` artifact. Only numbers and charts are logged, never frames or audio.

## Run locally

```bash
pip install -e ../vn-av-df-forensics[data,training] -e .[mavos,wandb,test]
python -m vnav_cross run --work work --main-root ../vn-av-df-forensics --datasets fakeavceleb
python -m vnav_cross audit --work work --manifests <part>/manifest.jsonl ... \
       --test-logs ../vn-av-df-forensics/docs/v005_results/test-log.json ../vn-av-df-forensics/docs/v006_results/test-log.json
pytest
```

## Outputs (`<work>/results/`)

| File | Content |
|---|---|
| `report.md` | data coverage, detector architectures and parameters, zero-shot AUC with CIs, per-class / per-method / per-language AUC, ROC curves, score distributions, operating points, paired deltas, shortcut checks, gate, ablation, published numbers, caveats |
| `metrics.json`, `metrics.csv` | every metric per dataset x detector x condition x gate x task |
| `roster.json` | detectors, thresholds, in-domain AUC, architecture summaries |
| `figures/*.png`, `tables/*.csv` | the charts and tables of the report |
| `audit.md`, `audit.json` | VN-AV-DF leakage audit |

## Layout

| Path | Content |
|---|---|
| `configs/crossdata.yaml` | datasets, sampling, roster, protocols |
| `configs/published_results.yaml` | numbers reported by other papers, with their source table |
| `src/vnav_cross/datasets.py` | adapters to one row schema |
| `src/vnav_cross/sampling.py` | stratified sampling, calibration split |
| `src/vnav_cross/media.py` | normalization, leading silence |
| `src/vnav_cross/features.py` | encoder runs with per-video failure records |
| `src/vnav_cross/scoring.py` | roster, windowed scoring, trimming, branch knockout, BL-Ens-AV |
| `src/vnav_cross/metrics.py`, `evaluate.py` | metrics, common set, gate, paired bootstrap |
| `src/vnav_cross/figures.py`, `report.py` | charts and the Markdown report |
| `src/vnav_cross/audit.py` | VN-AV-DF leakage audit |
| `src/vnav_cross/tracking.py` | W&B (on top of `vn_av_df.tracking`) |
| `tools/export_weights.py` | packs the checkpoints of several seeds into one Kaggle dataset, checks the BL-Ens-AV threshold |
| `tools/make_notebook.py` | generates the Kaggle notebook (`--set NAME=VALUE` for session copies) |
| `docs/` | Vietnamese runbook and upload guide, experiment plan for the paper, the audit result on the released data |

## Caveats

- AV-HuBERT was pre-trained on LRS3 and VoxCeleb2, and FakeAVCeleb, LAV-DF and AV-Deepfake1M are built from VoxCeleb2.
- The released AVH-Align checkpoint, which also initializes the phi network of BL-M, M2 and M3-R, was trained on AV-Deepfake1M train reals.
- Our detectors saw Vietnamese only, so every external number is a cross-language transfer.
- Trimming slices cached features rather than re-encoding the video.
