# Model card — risiko-kedaluwarsa 1.0.0

| | |
|---|---|
| Pemilik | Pedaree / `pedaree-ai-models` |
| Versi | `1.0.0` (`src/versi.py`; juga di metadata ONNX `model_versi`) |
| Artefak | `models/kedaluwarsa-v1.onnx` (dibangun `python -m src.latih`, tidak di-commit) |
| Keputusan | ADR-0002 |
| Metrik | `eval/metrics.md` (dihasilkan otomatis) |

## Tujuan dan pemakaian

Memperkirakan P(batch terbuang) = peluang >20% jumlah batch masih tersisa pada hari kedaluwarsa, agar
UI/server dapat memprioritaskan batch yang berisiko tinggi **di dalam** status SOP yang sama (mis. di antara batch
SEGERA). Dipakai lewat API Rust (`pedaree-backend-service`), bukan dilayani dari repo ini.

**Bukan untuk**: mengubah ambang SOP (AMAN/WASPADA/SEGERA/HARI_INI/KEDALUWARSA), urutan FEFO, atau status batch
(semuanya milik `src/baseline.py`/TownHall); membuat pesanan TitipO otomatis (sinyal hanya anjuran); menilai
batch KEDALUWARSA atau tanggal null (di luar SOP).

## Model

`GradientBoostingClassifier` (scikit-learn 1.6.1; 60 pohon, kedalaman 3, lr 0,1, seed 0) -> ONNX (opset 17, `zipmap=False`).
Masukan `masukan` float32 `[N,5]`: `sisa_hari`, `laju_pakai_harian`, `bobot_resep`, `hari_habis_dipakai`
(= `jumlah_aktif / laju_pakai_harian`, maks 365), `rasio_habis_vs_sisa` (= `hari_habis_dipakai / (sisa_hari + 1)`);
fitur dihitung oleh `src/fitur.py` yang sama untuk latih, evaluasi, dan inferensi. Keluaran 1 = probabilitas `[N,2]`
(kolom 1 = terbuang). `bobot_resep` (0..1) adalah proksi konteks `recipe_id` Pawonee; repo ini tidak menyalin/melatih
ulang data resep Pawonee.

## Data

**Sintetis sepenuhnya** (`eval/data/sintetis-v1.csv`, 4.000 baris, generator deterministik, `sumber=SINTETIS`).
Tidak ada data pengguna. Asumsi dunia simulasi dijelaskan di `eval/README.md`.

## Evaluasi

Lihat `eval/metrics.md` (set uji 1.200 baris dari split terstratifikasi 70/30). Ringkasan pada saat tulisan ini:
model ROC-AUC 0,966 vs baseline SOP 0,709; PR-AUC 0,950 vs 0,572. Gerbang rilis (tidak lebih buruk dari baseline)
LOLOS. Angka dapat berubah bila dilatih ulang; sumber kebenaran adalah `eval/metrics.md`.

## Keterbatasan dan risiko

- Label dan fitur dari simulasi yang sama; keunggulan atas baseline sebagian besar by-construction. **Belum ada bukti kinerja di rumah tangga nyata.**
- Target produk "≥50% batch SEGERA dipakai" tidak dapat diukur offline; butuh telemetri produksi.
- Tidak ada uji keadilan/segmen; satuan hitung dan 13 satuan baku tidak dimodelkan (laju dan jumlah diasumsikan satuan sama).
- Model tidak boleh dipakai di produksi sebelum dilatih ulang pada ekspor teranonimkan DB `pedaree` dan `eval/metrics.md` diperbarui (ADR).

## Perubahan versi

| Versi | Perubahan |
|---|---|
| 1.0.0 | Rilis awal pada data sintetis |
