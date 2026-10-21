"""Evaluasi model ONNX vs baseline SOP pada split uji, tulis eval/metrics.md, dan terapkan gerbang rilis.

Jalankan: `python -m eval.evaluasi [--model models/kedaluwarsa-v1.onnx] [--gerbang]`
`--gerbang`: keluar dengan kode 1 bila model lebih buruk dari baseline (rilis diblokir).
Semua angka dihitung saat dijalankan; tidak ada nilai yang diketik manual.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sklearn
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score,
)

from eval import dataset
from src import baseline, fitur, versi
from src.latih import NAMA_MASUKAN

RAIB = Path(__file__).resolve().parent / "metrics.md"
HARI_ACUAN = date(2026, 10, 21)
AMBANG_MODEL = 0.5
# Baseline memprediksi "terbuang" untuk status >= SEGERA (SOP: butuh tindakan).
STATUS_POSITIF = {baseline.StatusSop.SEGERA, baseline.StatusSop.HARI_INI, baseline.StatusSop.KEDALUWARSA}


def skor_baseline_baris(baris: list[dict]) -> np.ndarray:
    return np.array(
        [baseline.skor_baseline(HARI_ACUAN + timedelta(days=r["sisa_hari"]), HARI_ACUAN) for r in baris]
    )


def prediksi_baseline_baris(baris: list[dict]) -> np.ndarray:
    return np.array(
        [
            int(baseline.status_sop(HARI_ACUAN + timedelta(days=r["sisa_hari"]), HARI_ACUAN) in STATUS_POSITIF)
            for r in baris
        ]
    )


def skor_model(jalur: Path, baris: list[dict]) -> np.ndarray:
    sesi = ort.InferenceSession(str(jalur), providers=["CPUExecutionProvider"])
    return sesi.run(None, {NAMA_MASUKAN: fitur.matriks(baris)})[1][:, 1]


def _ringkas(y, skor, pred) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, skor)),
        "pr_auc": float(average_precision_score(y, skor)),
        "presisi": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def hitung(jalur: Path) -> dict:
    _, uji = dataset.pisah(dataset.baca())
    y = np.array([r["terbuang"] for r in uji])
    s_model = skor_model(jalur, uji)
    s_base = skor_baseline_baris(uji)
    hasil = {
        "n_uji": len(uji),
        "prevalensi": float(y.mean()),
        "model": _ringkas(y, s_model, (s_model >= AMBANG_MODEL).astype(int)),
        "baseline": _ringkas(y, s_base, prediksi_baseline_baris(uji)),
    }
    hasil["model"]["brier"] = float(brier_score_loss(y, s_model))
    # Peringkat di dalam kelompok SEGERA (1-3 hari): baseline memberi skor sama => tak bisa memilah.
    segera = np.array([1 <= r["sisa_hari"] <= 3 for r in uji])
    hasil["segera"] = {
        "n": int(segera.sum()),
        "dipakai_bukan_dibuang": float(1 - y[segera].mean()),
        "auc_model": float(roc_auc_score(y[segera], s_model[segera])),
        "auc_baseline": float(roc_auc_score(y[segera], s_base[segera])),
    }
    return hasil


def gerbang_lolos(hasil: dict) -> tuple[bool, list[str]]:
    """Rilis hanya bila model tidak lebih buruk dari baseline pada ROC-AUC dan PR-AUC (metrik yang sama)."""
    gagal = [
        f"{m}: model {hasil['model'][m]:.4f} < baseline {hasil['baseline'][m]:.4f}"
        for m in ("roc_auc", "pr_auc")
        if hasil["model"][m] < hasil["baseline"][m]
    ]
    return not gagal, gagal


def render(hasil: dict, jalur: Path) -> str:
    m, b, s = hasil["model"], hasil["baseline"], hasil["segera"]
    lolos, gagal = gerbang_lolos(hasil)
    sha_data = hashlib.sha256(dataset.BERKAS.read_bytes()).hexdigest()[:16]
    sha_model = hashlib.sha256(jalur.read_bytes()).hexdigest()[:16]
    baris = lambda n, k, f="{:.3f}": f"| {n} | {f.format(b[k])} | {f.format(m[k])} |"  # noqa: E731
    return f"""# Metrik evaluasi — {versi.MODEL_NAMA} {versi.MODEL_VERSI}

> **Dihasilkan otomatis oleh `python -m eval.evaluasi`; jangan diedit tangan.**
> **Data: SINTETIS** (`eval/data/sintetis-v1.csv`, sha256 `{sha_data}`). Angka ini membuktikan pipeline
> latih → ONNX → evaluasi → gerbang berjalan, BUKAN kinerja di rumah tangga nyata (lihat Keterbatasan).

- Model: `{versi.BERKAS_ONNX}` sha256 `{sha_model}`, GradientBoosting (60 pohon, kedalaman 3) → ONNX, fitur: {", ".join(fitur.NAMA_FITUR)}
- Target prediksi: P(batch terbuang), yaitu >20% jumlah tersisa pada hari kedaluwarsa
- Split uji: {hasil["n_uji"]} baris, prevalensi terbuang {hasil["prevalensi"]:.3f} (split terstratifikasi 70/30, seed tetap)
- Lingkungan: scikit-learn {sklearn.__version__}, onnxruntime {ort.__version__}

## Model vs baseline (SOP: status SEGERA/HARI_INI/KEDALUWARSA = prediksi terbuang)

| Metrik (set uji) | Baseline SOP | Model |
|---|---|---|
{baris("ROC-AUC", "roc_auc")}
{baris("PR-AUC", "pr_auc")}
{baris("Presisi (model: p>=0,5)", "presisi")}
{baris("Recall", "recall")}
{baris("F1", "f1")}
| Brier (hanya model; skor baseline bukan probabilitas) | - | {m["brier"]:.3f} |

## Batch SEGERA (1-3 hari), n = {s["n"]}

| Ukuran | Nilai |
|---|---|
| Batch SEGERA yang dipakai (bukan dibuang), di dunia simulasi | {s["dipakai_bukan_dibuang"]:.1%} |
| ROC-AUC memilah batch terbuang di dalam SEGERA, baseline | {s["auc_baseline"]:.3f} |
| ROC-AUC memilah batch terbuang di dalam SEGERA, model | {s["auc_model"]:.3f} |

Baris pertama adalah properti dunia simulasi (sama untuk model maupun baseline), **bukan** pengukuran target
produk "≥50% batch SEGERA dipakai" (BR-303; `metrik/10` bab 3). Target itu butuh data produksi (event `buang_catat`
dan `Catat pakai`) dan belum dapat diukur secara offline. Nilai tambah model yang terukur di sini: memilah
prioritas di dalam kelompok SEGERA yang bagi baseline setara.

## Gerbang rilis (AGENTS.md: model tidak boleh lebih buruk dari baseline)

Aturan: ROC-AUC dan PR-AUC model >= baseline pada set uji yang sama. Hasil: **{"LOLOS" if lolos else "GAGAL, rilis diblokir"}**{"" if lolos else chr(10) + chr(10).join("- " + g for g in gagal)}

## Keterbatasan (jujur)

- Label dan fitur sama-sama berasal dari simulasi yang sama (pemakaian ~ `laju_pakai_harian`, `jumlah_aktif`); baseline tidak melihat kedua fitur itu, jadi keunggulan model di sini hampir pasti by-construction.
- `bobot_resep` mewakili sinyal `recipe_id` Pawonee; efeknya diasumsikan (x1 + 0,5·bobot), tidak diukur.
- Tidak ada validasi pada data produksi; ganti dataset (ADR) dan jalankan ulang sebelum klaim apa pun ke produk.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path("models") / versi.BERKAS_ONNX)
    ap.add_argument("--keluaran", type=Path, default=RAIB)
    ap.add_argument("--gerbang", action="store_true")
    a = ap.parse_args()
    hasil = hitung(a.model)
    a.keluaran.write_text(render(hasil, a.model), encoding="utf-8")
    lolos, gagal = gerbang_lolos(hasil)
    print(f"metrik tertulis: {a.keluaran}; gerbang {'LOLOS' if lolos else 'GAGAL'}")
    for g in gagal:
        print("  -", g, file=sys.stderr)
    return 1 if (a.gerbang and not lolos) else 0


if __name__ == "__main__":
    sys.exit(main())
