# Kontrak inferensi v1.0 (pedaree-ai-models -> pedaree-backend-service)

Sumber kebenaran: `src/kontrak.py` -> `masukan.schema.json` / `keluaran.schema.json` (terkunci uji). Keputusan: ADR-0003.
Skema 1.0, model `risiko-kedaluwarsa` 1.0.0. **Skor berasal dari model berdata sintetis** (`eval/metrics.md`).

## Alur

1. Backend menyusun `PermintaanInferensi` dari StokBatch aktif satu rumah tangga (maks 2.000 batch).
2. Implementasi acuan: `python -m src.inferensi --masukan permintaan.json [--model models/kedaluwarsa-v1.onnx]`
   (atau memuat ONNX langsung di Rust, lihat bawah). `--contoh` menjalankan contoh resmi, `--skema` mencetak skema.
3. Hasil `HasilInferensi`: satu entri per batch, terurut FEFO global (terdekat dulu, tanggal null setelahnya, terkunci terakhir).

## Masukan (camelCase)

| Properti | Tipe | Aturan |
|---|---|---|
| `skemaVersi` | `"1.0"` | wajib cocok |
| `hariIni` | date `YYYY-MM-DD` | tanggal WIB acuan |
| `batch[]` | 1..2000 | `batchId` unik |
| `batch[].batchId` / `pantryItemId` | `btc_...` / `itm_...` | ULID berawalan |
| `nama` (3-60), `kategori` | enum 8 kategori | SAYUR BUMBU BUAH PROTEIN KERING CAIR BEKU SISA |
| `satuan` | 13 satuan baku | selain itu ditolak |
| `jumlahAktif` | >= 0 | satuan hitung bulat; gram/ml maks 50.000 per batch (setelah normalisasi kg/liter/sdm/sdt) |
| `tanggalMasuk`, `tanggalKedaluwarsa` | date, null | kedaluwarsa >= masuk; null = di luar SOP |
| `lajuPakaiHarian` | >= 0 (default 0) | ASUMSI: backend menghitung rerata pemakaian harian dalam `satuan` |
| `bobotResep` | 0..1 (default 0) | konteks resep Pawonee |
| `recipeId` | string/null | hanya rujukan |

Properti lain ditolak (`additionalProperties: false`).

## Keluaran

| Properti | Arti |
|---|---|
| `model.{nama,versi,sumberModel}` | `sumberModel=baseline` bila ONNX tak tersedia (semua skor dari aturan) |
| `hasil[].statusSop` | AMAN/WASPADA/SEGERA/HARI_INI/KEDALUWARSA, atau null (tanggal null). **Dari aturan terkunci, bukan model** |
| `hasil[].sisaHari` | `kedaluwarsa - hariIni`, negatif bila lewat; null bila tanggal null |
| `hasil[].skorRisiko` | 0..1, P(batch terbuang); KEDALUWARSA = 1,0; tanggal null = 0,0 |
| `hasil[].sumberSkor` | `model` atau `baseline` |
| `hasil[].urutanPakai` | peringkat FEFO (1 = dulu) di dalam item; null bila terkunci |
| `hasil[].terkunci` | KEDALUWARSA atau jumlah 0: hanya bisa dibuang, tidak ikut FEFO |

Skor tidak mengubah status, urutan, maupun ambang SOP. Sinyal ke TitipO tetap anjuran; skor ini bukan pemicu pesanan.

## Muat ONNX langsung di Rust (opsional)

- Masukan `masukan`: f32 `[N,5]` urutan `sisa_hari, laju_pakai_harian, bobot_resep, hari_habis_dipakai, rasio_habis_vs_sisa`.
- Fitur turunan (satuan kanonis): `laju = max(laju_pakai_harian, 0.001)`; `hari_habis_dipakai = min(jumlahAktif / laju, 365)`; `rasio_habis_vs_sisa = hari_habis_dipakai / (sisa_hari + 1)`.
- Keluaran indeks 1: probabilitas `[N,2]`, kolom 1 = terbuang. Vektor uji paritas: `vektor-fitur.json` (toleransi 1e-4).
- Hanya nilai batch hidup bertanggal (bukan KEDALUWARSA, jumlah > 0) yang dinilai model; sisanya skor aturan.

## Mengubah kontrak

Ubah `src/kontrak.py` -> bump `SKEMA_VERSI` + `MODEL_VERSI` (`src/versi.py`) -> ADR baru -> `python -m src.kontrak_tulis` -> `pytest`.
