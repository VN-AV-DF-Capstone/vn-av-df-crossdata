"""Synthetic tests: adapters, sampling, silence, metrics, audit, and score → evaluate → report
end to end on random features with small randomly initialized VN-AV-DF models."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from vnav_cross import audit, datasets, media, metrics, sampling

MAIN_ROOT = Path(os.environ.get("VNAVDF_ROOT", Path(__file__).resolve().parents[2] / "vn-av-df-forensics"))


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


# ---------------------------------------------------------------- adapters
def test_fakeavceleb_layout_and_methods(tmp_path):
    root = tmp_path / "FakeAVCeleb_v1.2"
    touch(root / "RealVideo-RealAudio/African/men/id00076/00109.mp4")
    touch(root / "FakeVideo-RealAudio/African/men/id00076/00109_faceswap.mp4")
    touch(root / "RealVideo-FakeAudio/African/men/id00076/00109_fake.mp4")
    touch(root / "FakeVideo-FakeAudio/African/men/id00076/00109_wavtolip.mp4")
    (root / "meta_data.csv").write_text(
        "source,target1,target2,method,category,type,race,gender,path\n"
        "id00076,,,wav2lip,D,FakeVideo-FakeAudio,African,men,00109_wavtolip.mp4\n", encoding="utf-8")
    rows = {r["category"]: r for r in datasets.fakeavceleb(root)}
    assert set(rows) == {"RVRA", "FVRA", "RVFA", "FVFA"}
    assert rows["RVRA"]["label"] == 0 and rows["RVFA"]["synthetic_audio"] and not rows["RVFA"]["synthetic_visual"]
    assert rows["FVFA"]["method"] == "wav2lip" and rows["FVRA"]["method"] == "unknown"
    assert rows["FVFA"]["subject"] == "id00076" and rows["FVFA"]["race"] == "African"
    assert len({r["sample_id"] for r in rows.values()}) == 4


def test_avdeepfake1m_and_lavdf_metadata(tmp_path):
    av = tmp_path / "av1m"
    meta = [{"file": "id00166/abc/00001.mp4", "modify_type": "real", "fake_segments": []},
            {"file": "id00166/abc/00001_x.mp4", "modify_type": "both_modified", "audio_model": "vits",
             "fake_segments": [[1.0, 1.5]], "visual_fake_segments": [[1.0, 1.5]], "audio_fake_segments": [[1.0, 1.5]]}]
    (av / "val").mkdir(parents=True)
    (av / "val_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    rows = datasets.avdeepfake1m(av)
    assert [r["category"] for r in rows] == ["real", "both_modified"]
    assert rows[1]["method"] == "vits" and rows[1]["audio_fake_intervals"] == [[1.0, 1.5]]
    assert rows[0]["subject"] == "id00166" and rows[0]["path"].replace("\\", "/").endswith("val/id00166/abc/00001.mp4")

    lav = tmp_path / "lavdf"
    lav.mkdir()
    (lav / "metadata.json").write_text(json.dumps([
        {"file": "test/000001.mp4", "split": "test", "modify_video": False, "modify_audio": False,
         "fake_periods": [], "original": None},
        {"file": "test/000002.mp4", "split": "test", "modify_video": True, "modify_audio": False,
         "fake_periods": [[0.2, 0.9]], "original": "test/000001.mp4"},
        {"file": "train/000003.mp4", "split": "train", "modify_video": True, "modify_audio": True,
         "fake_periods": [[0.2, 0.9]], "original": "train/000004.mp4"}]), encoding="utf-8")
    rows = datasets.lavdf(lav)
    assert [r["category"] for r in rows] == ["real", "visual_modified"]
    assert rows[0]["subject"] == rows[1]["subject"] == "000001"
    assert rows[1]["visual_fake_intervals"] == [[0.2, 0.9]] and rows[1]["audio_fake_intervals"] == []


def test_mavos_parts_and_test_sets(tmp_path):
    root = tmp_path / "mavos"
    root.mkdir()
    records = [
        {"video_path": "english/real/a_Po82BhllEjA_1.mp4", "label": "real", "split": "test",
         "open_set_model": False, "open_set_language": False, "language": "english", "generative_method": None},
        {"video_path": "english/memo/b.mp4", "label": "fake", "split": "test",
         "open_set_model": False, "open_set_language": False, "language": "english", "generative_method": "memo"},
        {"video_path": "german/roop/c.mp4", "label": "fake", "split": "test",
         "open_set_model": True, "open_set_language": True, "language": "german", "generative_method": "roop"},
        {"video_path": "hindi/memo/d.mp4", "label": "fake", "split": "train",
         "open_set_model": False, "open_set_language": False, "language": "hindi", "generative_method": "memo"},
    ]
    (root / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    rows = datasets.mavosdd(root, subsets=("in_domain",))
    assert len(rows) == 2 and {r["subset"] for r in rows} == {"core"}
    assert rows[0]["subject"] == "Po82BhllEjA" and not any(r["synthetic_audio"] for r in rows)
    assert len(datasets.mavosdd(root, subsets=("open_full",))) == 3


# ---------------------------------------------------------------- sampling, silence, metrics
def fake_rows(n_subjects=12, per=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        for i in range(per):
            fake = i % 2 == 1
            rows.append(datasets.row("folder", "/data", f"{'fake' if fake else 'real'}/s{s}_{i}.mp4",
                                     visual=fake, audio=fake and i % 3 == 0,
                                     category="fake" if fake else "real", subject=f"s{s}",
                                     method=f"g{int(rng.integers(2))}" if fake else "real"))
    return rows


def test_sampling_is_deterministic_and_capped():
    rows = fake_rows()
    spec = {"per_stratum": 10, "strata": ["category"], "max_per_subject": 1}
    a, b = sampling.sample(rows, spec, 7), sampling.sample(rows, spec, 7)
    assert [r["sample_id"] for r in a] == [r["sample_id"] for r in b]
    assert len(a) == 20
    for cat in ("real", "fake"):
        subjects = [r["subject"] for r in a if r["category"] == cat]
        assert len(subjects) == len(set(subjects))
    stats = sampling.sample_stats(rows, a, ["category"])
    assert stats["full"] == len(rows) and stats["sampled"] == 20
    split = sampling.calibration_split(list(a), 0.3, 7)
    by_subject = {}
    for r in split:
        by_subject.setdefault(r["subject"], set()).add(r["part"])
    assert all(len(v) == 1 for v in by_subject.values())


def test_leading_silence_rounds_up_to_frames():
    sr = 16000
    wave = np.zeros(sr, np.float32)
    wave[int(0.205 * sr):] = 0.1
    assert media.leading_silence(wave, sr) == pytest.approx(0.24)
    assert media.leading_silence(np.full(sr, 0.1, np.float32), sr) == 0.0
    assert media.leading_silence(np.zeros(sr, np.float32), sr) == pytest.approx(1.0)


def test_metrics_blocks_and_paired_bootstrap():
    rows = fake_rows(20, 6)
    for r in rows:
        r["part"] = "evaluation"
    rng = np.random.default_rng(1)
    labels = [r["label"] for r in rows]
    good = [y + rng.normal(scale=0.5) for y in labels]
    bad = [rng.normal() for _ in labels]
    block = metrics.task_block(rows, good, "ai", 0.5, (0.01, 0.05), 200, 0)
    assert 0.8 < block["auc"] <= 1 and block["auc_ci"][0] <= block["auc"] <= block["auc_ci"][1]
    assert block["transfer"]["fpr"] is not None and block["n_subjects"] == 20
    paired = metrics.paired(rows, labels, good, bad, 200, 0)
    assert paired["delta"] > 0.2 and paired["ci"][0] > 0 and paired["p_le_0"] == 0.0
    groups = metrics.group_auc(rows, good, "method")
    assert set(groups) <= {"g0", "g1"}


def test_audit_detects_shared_speaker(tmp_path):
    def manifest(name, rows):
        path = tmp_path / name / "manifest.jsonl"
        path.parent.mkdir()
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return path

    base = {"source_clip_id": "c", "sha256": "h", "cell": "real", "label": 0, "duration_s": 5.0}
    a = manifest("p1", [{**base, "split": "train", "speaker_id": "spk1", "source_id": "yt1", "sha256": "h1", "source_clip_id": "c1"},
                        {**base, "split": "test", "speaker_id": "spk2", "source_id": "yt2", "sha256": "h2", "source_clip_id": "c2",
                         "cell": "sham", "audio_source_id": "yt1"}])
    b = manifest("p2", [{**base, "split": "test", "speaker_id": "spk1", "source_id": "yt3", "sha256": "h3", "source_clip_id": "c3"}])
    result = audit.run_audit([a, b])
    assert result["overlap"]["speaker_id"]["shared"] == 1
    assert result["verdict"]["identity_or_source_leak"] == ["speaker_id"]
    assert result["donor_audio"]["crossing_splits"] == {"sham:test<-train": 1}


# ---------------------------------------------------------------- end to end on random features
def write_cache(cache, sid, n, fake, dim, groups, rng):
    for enc in ("avhubert", "xlsr", "mouth"):
        (cache / enc).mkdir(parents=True, exist_ok=True)
    a = rng.normal(size=(n, dim)).astype(np.float32)
    v = (rng.normal(size=(n, dim)) if fake else a + rng.normal(scale=0.3, size=(n, dim))).astype(np.float32)
    np.savez(cache / "avhubert" / f"{sid}.npz", audio=a, visual=v, audio_valid=np.ones(n, bool),
             visual_valid=np.ones(n, bool), landmarks=rng.normal(100, 2, size=(n, 68, 2)).astype(np.float16))
    np.savez(cache / "xlsr" / f"{sid}.npz", ssl=rng.normal(size=(n, groups, 1024)).astype(np.float16),
             valid=np.ones(n, bool))
    np.savez(cache / "mouth" / f"{sid}.npz", mouth=rng.integers(0, 255, size=(n, 96, 96)).astype(np.uint8),
             lips=rng.normal(scale=0.1, size=(n, 20, 2)).astype(np.float16),
             pose=rng.normal(scale=0.1, size=(n, 3)).astype(np.float32), valid=np.ones(n, bool), native_side=64.0)


@pytest.mark.skipif(not (MAIN_ROOT / "settings.py").is_file(), reason="vn-av-df-forensics checkout not found")
def test_score_evaluate_report_end_to_end(tmp_path):
    torch = pytest.importorskip("torch")
    from vn_av_df.training.models import build
    from vn_av_df.training.train import save

    from vnav_cross import pipeline
    from vnav_cross.evaluate import step_evaluate
    from vnav_cross.report import step_report

    dim, groups = 32, 3
    runs = tmp_path / "runs"
    for name in ("trihead_v5", "avh_align_plus", "avh_align_sup", "xlsr_frozen", "speechforensics"):
        torch.manual_seed(0)
        model = build(name, dim=dim, assets={"load_pretrained": False, "ssl_groups": groups})
        folder = runs / f"{name}_seed42"
        folder.mkdir(parents=True)
        save(folder / "best.pt", {"config": {"name": name, "dim": dim, "hidden": 128, "dropout": 0.1,
                                             "assets": model.rebuild_assets()}, "state": model.state_dict(), "epoch": 0})
        tasks = {t: {"auc": 0.9, "operating_points": {"fpr_0.05": {"threshold": 0.5}}} for t in ("ai", "visual", "audio")}
        (folder / "metrics.json").write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
    ens = runs / "bl_ens_av_seed42"
    ens.mkdir()
    (ens / "ensemble.json").write_text(json.dumps({"visual": "avh_align_sup", "audio": "xlsr_frozen",
                                                   "stats": {"visual": [0.0, 1.0], "audio": [0.5, 0.1]}}), encoding="utf-8")
    (ens / "metrics.json").write_text(json.dumps({"tasks": {"ai": {"auc": 0.93, "operating_points": {"fpr_0.05": {"threshold": 3.0}}}}}), encoding="utf-8")

    work = tmp_path / "work"
    cross = pipeline.load_config(work=str(work), main_root=str(MAIN_ROOT))
    cross["datasets"] = {"folder": {"root": str(tmp_path), "sample": {"per_stratum": 100}}}
    cross["models"] = {"runs": [str(runs)], "published": [], "reference": "trihead_v5",
                       "ours": ["trihead_v5", "avh_align_plus", "avh_align_sup", "xlsr_frozen", "speechforensics", "bl_ens_av"]}
    cross["protocols"]["bootstrap"] = 50
    rows = sampling.calibration_split(fake_rows(10, 4), 0.3, 1)
    pipeline.write_jsonl(work / "folder/manifest.jsonl", rows)
    rng = np.random.default_rng(0)
    media_rows = []
    for i, r in enumerate(rows):
        n = 150 + 40 * (i % 3)  # 6-7.6 s: one or two 7 s windows
        write_cache(work / "folder/cache", r["sample_id"], n, r["label"], dim, groups, rng)
        media_rows.append({"sample_id": r["sample_id"], "clip": "x.mp4", "short_side": 224 if i % 4 < 2 else 720,
                           "duration_s": n / 25, "lead_silence_s": 0.2 * (i % 3)})
    pipeline.write_jsonl(work / "folder/media.jsonl", media_rows)

    pipeline.step_score(cross, device="cpu", workers=0)
    scores = json.loads((work / "folder/scores/trihead_v5_seed42.json").read_text(encoding="utf-8"))
    assert {"base", "trim", "knockout:trace", "knockout:audio"} <= set(scores["conditions"])
    assert len(scores["scores"]) == len(rows)
    assert (work / "folder/scores/bl_ens_av_seed42.json").is_file()

    results = step_evaluate(cross)
    d = results["datasets"]["folder"]
    assert d["n_common"] == len(rows)
    m3 = d["models"]["trihead_v5_seed42"]["results"]
    assert "base|gate_on" in m3 and "base|gate_off" in m3 and m3["base|gate_off"]["ai"]["auc"] is not None
    assert "base|gate_on" not in d["models"]["speechforensics_seed42"]["results"]  # single-score: never gated
    assert "base|gate_on" in d["models"]["bl_ens_av_seed42"]["results"]
    assert "transfer" in m3["base|gate_on"]["ai"]
    assert any(k.startswith("trihead_v5_seed42|gate_off|") for k in d["paired"])
    assert set(d["shortcut_probes"]) == {"lead_silence_s", "duration_s", "short_side"}

    report = step_report(cross, results)
    text = report.read_text(encoding="utf-8")
    assert "Zero-shot AUC" in text and "Test-time ablation" in text
    assert (work / "results/figures/auc_heatmap_ai.png").is_file()
    assert (work / "results/figures/knockout_trihead_v5_seed42.png").is_file()


def test_polyglotfake_layout_and_json(tmp_path):
    root = tmp_path / "PolyGlotFake"
    touch(root / "real/es/es_1.mp4")
    touch(root / "fake/to_ru/es_1_to_ru_Xtts.mp4")
    touch(root / "fake/to_en/es_1_to_en_Bark.mp4")
    (root / "json_file/fake_Json_file").mkdir(parents=True)
    (root / "json_file/real_json_file").mkdir(parents=True)
    (root / "json_file/fake_Json_file/all_fake_video.json").write_text(json.dumps({"video": [
        {"filename": "es_1_to_ru_Xtts.mp4", "raw_video_name": "es_1.mp4", "tts_technique": "Xtts",
         "raw_lang": "es", "target_lang": "ru", "sync_tech": "video_retalking"}]}), encoding="utf-8")
    (root / "json_file/real_json_file/all_real_video.json").write_text(json.dumps({"videos": [
        {"filename": "es_1.mp4", "lang": "es"}]}), encoding="utf-8")
    rows = {r["relative"]: r for r in datasets.polyglotfake(root)}
    real, ru, en = rows["real/es/es_1.mp4"], rows["fake/to_ru/es_1_to_ru_Xtts.mp4"], rows["fake/to_en/es_1_to_en_Bark.mp4"]
    assert real["label"] == 0 and ru["synthetic_visual"] and ru["synthetic_audio"]
    assert ru["method"] == "Xtts+video_retalking" and ru["language"] == "ru"
    assert en["method"] == "Bark+unknown" and en["language"] == "en"  # not in the json: from the path
    assert real["subject"] == ru["subject"] == en["subject"] == "es_1"


def test_avdeepfake1m_reads_split_zip(tmp_path):
    import zipfile

    from vnav_cross import archive

    root = tmp_path / "av1m"
    (root / "val").mkdir(parents=True)
    whole = tmp_path / "val.zip"
    with zipfile.ZipFile(whole, "w") as z:
        z.writestr("val/id00012/abc/00002/real.mp4", b"r" * 70_000)
        z.writestr("val/id00012/abc/00002/fake_video_real_audio.mp4", b"f" * 70_000)
    data = whole.read_bytes()
    for i, k in enumerate(range(0, len(data), 40_000)):
        (root / "val" / f"val.zip.{i + 1:03d}").write_bytes(data[k:k + 40_000])
    (root / "val_metadata.json").write_text(json.dumps([
        {"file": "id00012/abc/00002/real.mp4", "modify_type": "real"},
        {"file": "id00012/abc/00002/fake_video_real_audio.mp4", "modify_type": "visual_modified",
         "fake_segments": [[1.0, 1.4]], "visual_fake_segments": [[1.0, 1.4]], "audio_fake_segments": []},
        {"file": "id99999/missing/00001/real.mp4", "modify_type": "real"}]), encoding="utf-8")
    rows = datasets.avdeepfake1m(root)
    assert [r["category"] for r in rows] == ["real", "visual_modified"]  # the row without a video is dropped
    assert rows[1]["member"] == "val/id00012/abc/00002/fake_video_real_audio.mp4"
    out = archive.extract(rows[1]["archive"], rows[1]["member"], tmp_path / "x.mp4")
    assert out.read_bytes() == b"f" * 70_000
