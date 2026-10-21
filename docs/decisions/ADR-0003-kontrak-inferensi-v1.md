# ADR-0003 — Kontrak inferensi model v1.0 untuk API Rust

Status: Diterima | Tanggal: 2026-10-21

## Konteks

Tiket PDR-AI-004 (issue #4). Serving model dilakukan oleh `pedaree-backend-service` (Rust), bukan repo ini; backend butuh kontrak masukan/keluaran yang tegas. Rujukan: `produk/20-kontrak-api-KMP-mobile.md` bab 2.5 (`/pantry/expiring`, urut FEFO). FSD/FRD/PRD/SRS v1.0.0 masih templat, jadi kontrak diturunkan dari `inventori/10`, `notifikasi/10`, dan AGENTS.md; bagian yang diasumsikan ditandai di bawah.

## Keputusan

1. Kontrak 1.0 didefinisikan oleh model pydantic di `src/kontrak.py` dan diterbitkan sebagai JSON Schema di `docs/kontrak/` (`masukan.schema.json`, `keluaran.schema.json`), lengkap dengan `contoh-*.json` dan `vektor-fitur.json` (vektor emas fitur untuk uji paritas Rust). Dokumen penjelasan: `docs/kontrak/INFERENSI.md`.
2. JSON camelCase mengikuti API Pedaree; 13 satuan baku (satuan lain ditolak); satuan non-kanonis dinormalkan (kg=1000 g, liter=1000 ml, sdm=15 ml, sdt=5 ml); satuan hitung harus bilangan bulat; maks 50.000 g/ml per batch; maks 2.000 batch per permintaan; properti tak dikenal ditolak.
3. Pembagian tanggung jawab: **status SOP dan urutan FEFO selalu dari aturan terkunci** (`src/baseline.py`); model hanya mengisi `skorRisiko` (P(batch terbuang)) untuk batch hidup bertanggal. Batch KEDALUWARSA / jumlah 0 `terkunci` (skor baseline 1,0; tidak ikut FEFO); tanggal null tanpa status, skor 0,0.
4. Perubahan skema (properti, enum, batas, makna) = bump `SKEMA_VERSI` dan `MODEL_VERSI`, ADR baru, dan regenerasi `python -m src.kontrak_tulis`; uji `test_skema_di_repo_identik_dengan_kode` memblokir drift. Masa dual-read 30 hari untuk perubahan yang tidak kompatibel (KONTEKS §9).
5. Rust dapat memuat ONNX langsung (`ort`: masukan `masukan` f32 `[N,5]`, keluaran indeks 1 = probabilitas `[N,2]`, kolom 1 = terbuang) asalkan mereproduksi `src/fitur.py` (diverifikasi dengan `vektor-fitur.json`, toleransi 1e-4), atau memanggil implementasi acuan Python. Pilihan ada pada tim backend.

## Asumsi (perlu konfirmasi manusia/backend)

- `lajuPakaiHarian` dihitung backend (mis. rerata pemakaian 14 hari dari MutasiStok PAKAI) dalam satuan batch; jendela dan penanganan item baru (laju 0) belum diputuskan dokumen.
- `bobotResep` 0..1 dihitung backend dari `recipe_id` Pawonee (cara menurunkan belum didefinisikan).
- Pemakaian di `/pantry/expiring`: usulan menambahkan `skorRisiko` opsional ke `BatchKedaluwarsa` di `openapi.yaml` backend; repo ini tidak mengubah kontrak backend.
- Tanggal acuan `hariIni` = tanggal kalender WIB yang dikirim pemanggil (tidak dibaca dari jam sistem).

## Konsekuensi

- Backend memiliki satu sumber skema; klien Rust dapat dibangkitkan dari JSON Schema.
- Model tidak dapat mengubah status/urutan/ambang; sesuai AGENTS.md butir 11 dan invarian FEFO.
- Skor model berasal dari data SINTETIS (lihat `eval/metrics.md`); jangan dipakai sebagai angka produk sebelum ada data nyata.
