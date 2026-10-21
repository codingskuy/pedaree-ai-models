# eval/ — dataset evaluasi dan metrik

> **DATASET INI SINTETIS.** Tidak ada data pengguna/rumah tangga nyata di repo ini
> (AGENTS.md: dilarang menyimpan data pengguna mentah). Setiap baris memuat
> `sumber=SINTETIS`. Ekspor teranonimkan DB `pedaree` belum tersedia; ganti dataset ini
> begitu ada (lihat "Langkah berikut").

## Berkas

| Berkas | Isi |
|---|---|
| `dataset.py` | Generator deterministik (seed `20261021`, `python -m eval.dataset`) |
| `data/sintetis-v1.csv` | 4.000 baris keluaran generator; diuji agar identik dengan hasil generator |
| `metrics.md` | Dibuat oleh PDR-AI-003 dari hasil evaluasi yang benar-benar dijalankan |

## Skema baris

| Kolom | Arti |
|---|---|
| `id` | `sint_NNNNN` |
| `nama`, `kategori`, `satuan` | Bahan dari `inventori/10-model-stok-dapur.md` bab 5 (tanpa Garam/Gula: tanggal null, di luar SOP) + `Sisa Masakan` (SISA) |
| `sisa_hari` | Hari tersisa sampai kedaluwarsa pada saat snapshot (0..min(umur,21)) |
| `jumlah_aktif`, `laju_pakai_harian` | Dalam `satuan` penyimpanan |
| `bobot_resep` | 0..1, seberapa sering bahan muncul di resep; fitur konteks pengganti `recipe_id` Pawonee |
| `recipe_id` | `rcp_sint_NNN` atau kosong; id karangan, BUKAN id resep Pawonee |
| `persen_terpakai` | Hasil simulasi pemakaian sampai hari kedaluwarsa |
| `terbuang` | Label: 1 bila >20% jumlah tersisa pada hari kedaluwarsa |

## Cara data dibuat (dan batasnya)

Satu batch per baris; pemakaian harian ~ Gamma(2) dengan rerata `laju_pakai_harian x (1 + 0.5 x bobot_resep)`.
Yang **ditetapkan dokumen**: nama bahan, umur simpan, satuan. Yang **asumsi karangan**: ukuran batch,
laju pakai, efek bobot resep, ambang label 20%, tidak ada interaksi FEFO antar-batch, tidak ada
pembelian ulang. Karena label diturunkan dari `laju_pakai_harian`/`jumlah_aktif` yang tidak
dilihat baseline SOP, model yang memakai fitur itu hampir pasti menang di dataset ini; itu
properti dunia simulasi, **bukan bukti kinerja produksi**.

## Langkah berikut

- Ganti/tambahkan dataset dari ekspor teranonimkan DB `pedaree` (tanpa `householdId`, tanpa nama pengguna) lewat ADR.
