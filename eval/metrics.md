# Metrik evaluasi — risiko-kedaluwarsa 1.0.0

> **Dihasilkan otomatis oleh `python -m eval.evaluasi`; jangan diedit tangan.**
> **Data: SINTETIS** (`eval/data/sintetis-v1.csv`, sha256 `88c805bc8b72af25`). Angka ini membuktikan pipeline
> latih → ONNX → evaluasi → gerbang berjalan, BUKAN kinerja di rumah tangga nyata (lihat Keterbatasan).

- Model: `kedaluwarsa-v1.onnx` sha256 `7c87a63e693915cc`, GradientBoosting (60 pohon, kedalaman 3) → ONNX, fitur: sisa_hari, laju_pakai_harian, bobot_resep, hari_habis_dipakai, rasio_habis_vs_sisa
- Target prediksi: P(batch terbuang), yaitu >20% jumlah tersisa pada hari kedaluwarsa
- Split uji: 1200 baris, prevalensi terbuang 0.387 (split terstratifikasi 70/30, seed tetap)
- Lingkungan: scikit-learn 1.6.1, onnxruntime 1.21.0

## Model vs baseline (SOP: status SEGERA/HARI_INI/KEDALUWARSA = prediksi terbuang)

| Metrik (set uji) | Baseline SOP | Model |
|---|---|---|
| ROC-AUC | 0.709 | 0.966 |
| PR-AUC | 0.572 | 0.950 |
| Presisi (model: p>=0,5) | 0.521 | 0.870 |
| Recall | 0.688 | 0.853 |
| F1 | 0.593 | 0.862 |
| Brier (hanya model; skor baseline bukan probabilitas) | - | 0.072 |

## Batch SEGERA (1-3 hari), n = 436

| Ukuran | Nilai |
|---|---|
| Batch SEGERA yang dipakai (bukan dibuang), di dunia simulasi | 58.3% |
| ROC-AUC memilah batch terbuang di dalam SEGERA, baseline | 0.500 |
| ROC-AUC memilah batch terbuang di dalam SEGERA, model | 0.955 |

Baris pertama adalah properti dunia simulasi (sama untuk model maupun baseline), **bukan** pengukuran target
produk "≥50% batch SEGERA dipakai" (BR-303; `metrik/10` bab 3). Target itu butuh data produksi (event `buang_catat`
dan `Catat pakai`) dan belum dapat diukur secara offline. Nilai tambah model yang terukur di sini: memilah
prioritas di dalam kelompok SEGERA yang bagi baseline setara.

## Gerbang rilis (AGENTS.md: model tidak boleh lebih buruk dari baseline)

Aturan: ROC-AUC dan PR-AUC model >= baseline pada set uji yang sama. Hasil: **LOLOS**

## Keterbatasan (jujur)

- Label dan fitur sama-sama berasal dari simulasi yang sama (pemakaian ~ `laju_pakai_harian`, `jumlah_aktif`); baseline tidak melihat kedua fitur itu, jadi keunggulan model di sini hampir pasti by-construction.
- `bobot_resep` mewakili sinyal `recipe_id` Pawonee; efeknya diasumsikan (x1 + 0,5·bobot), tidak diukur.
- Tidak ada validasi pada data produksi; ganti dataset (ADR) dan jalankan ulang sebelum klaim apa pun ke produk.
