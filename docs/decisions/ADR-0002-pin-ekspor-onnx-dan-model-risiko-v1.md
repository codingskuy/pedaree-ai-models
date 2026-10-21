# ADR-0002 — Pin ekspor ONNX dan model risiko-kedaluwarsa v1

Status: Diterima | Tanggal: 2026-10-21

## Konteks

Tiket PDR-AI-003 (issue #3). ADR-0001 mengunci `onnx==1.18.0` + `skl2onnx==1.18.0`, tetapi kombinasi itu **tidak dapat dipasang bersama**: `skl2onnx 1.18.0` mengimpor `onnx.helper.split_complex_to_pairs` yang sudah diganti nama di onnx 1.18 (`ImportError` saat `import skl2onnx`). Setelah itu, `protobuf` yang tak di-pin ikut naik ke 7.x dan konverter pohon skl2onnx gagal (`TypeError: Expected an int, got a boolean`). Jadi `src/latih.py` lama tidak pernah bisa menghasilkan artefak pada instalasi bersih. Ini temuan, dilaporkan di PR.

## Keputusan

1. `skl2onnx` naik ke **1.19.1** dan `protobuf` dikunci **5.29.5** di `requirements.txt` (versi lain tetap: onnx 1.18.0, onnxruntime 1.21.0, scikit-learn 1.6.1, numpy 2.0.2). Diverifikasi dengan instalasi bersih, `pip check`, latih, ekspor, dan validasi onnxruntime.
2. Model v1: `GradientBoostingClassifier` (60 pohon, kedalaman 3) yang memprediksi P(batch terbuang) dari 5 fitur (`sisa_hari`, `laju_pakai_harian`, `bobot_resep`, `hari_habis_dipakai`, `rasio_habis_vs_sisa`), diekspor ke ONNX (`zipmap=False`; keluaran 0 = label, keluaran 1 = probabilitas `[N,2]`), versi semver di `src/versi.py` dan metadata ONNX (`model_versi`).
3. Gerbang rilis: ROC-AUC dan PR-AUC model >= baseline SOP pada split uji yang sama; `python -m eval.evaluasi --gerbang` keluar kode 1 bila tidak. CI menjalankannya.
4. Artefak `models/*.onnx` tetap tidak di-commit (`.gitignore`); dibangun oleh `python -m src.latih` di CI/rilis. `eval/metrics.md` dihasilkan oleh perintah evaluasi, bukan diedit tangan.

## Konsekuensi

- Model hanya menambah skor risiko; ambang SOP dan urutan FEFO tetap milik `src/baseline.py`/TownHall.
- Perubahan fitur, urutan keluaran, atau format = bump `MODEL_VERSI` + ADR baru (kontrak I/O untuk API Rust: PDR-AI-004).
- Semua metrik saat ini berasal dari dataset SINTETIS; tidak boleh dikutip sebagai kinerja produksi.

## Addendum 2026-10-21 — kunci transitif

Keputusan manusia: seluruh dependensi (langsung dan transitif) dikunci dengan `==` di `requirements.txt`, sesuai `pip freeze` dari venv bersih Python 3.12 (versi CI) yang lolos `pip check`, `pytest` (33 lulus), `python -m src.latih`, dan `python -m eval.evaluasi --gerbang` (LOLOS). Daftar persis ada di `requirements.txt`; pemasangan ulang dari berkas itu menghasilkan `pip freeze` identik. Menaikkan versi apa pun = ulangi verifikasi yang sama dan catat di ADR baru.
