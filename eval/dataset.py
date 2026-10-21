"""Generator dataset evaluasi SINTETIS (bukan data pengguna).

Setiap baris ditandai `sumber=SINTETIS`. Dunia simulasi: satu batch tunggal per
baris, pemakaian harian acak sampai tanggal kedaluwarsa; label `terbuang=1`
bila lebih dari 20% jumlah awal-snapshot tersisa pada hari kedaluwarsa.

Parameter bahan (umur simpan, minimum) diambil dari `inventori/10-model-stok-dapur.md`
bab 5; laju pakai, ukuran batch, dan efek bobot resep adalah ASUMSI karangan
untuk simulasi (tidak diukur dari rumah tangga nyata). Metrik di atas dataset
ini hanya membuktikan pipeline berfungsi, bukan kinerja di produksi.

Jalankan: `python -m eval.dataset` (menulis eval/data/sintetis-v1.csv).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SEED = 20261021
JUMLAH_BARIS = 4000
AMBANG_TERBUANG = 0.2
BERKAS = Path(__file__).resolve().parent / "data" / "sintetis-v1.csv"
KOLOM = [
    "id", "nama", "kategori", "satuan", "sisa_hari", "jumlah_aktif",
    "laju_pakai_harian", "bobot_resep", "recipe_id", "persen_terpakai", "terbuang", "sumber",
]

# nama, kategori, satuan, umur simpan (hari), batch tipikal, laju pakai tipikal/hari
# Garam & Gula (umur 0, tanggal null) di luar SOP sehingga tidak ikut dataset.
BAHAN = [
    ("Beras", "KERING", "gram", 180, 5000, 250),
    ("Telur Ayam", "PROTEIN", "pcs", 14, 12, 1.5),
    ("Bawang Merah", "BUMBU", "gram", 21, 500, 15),
    ("Bawang Putih", "BUMBU", "gram", 30, 250, 8),
    ("Cabai Merah", "BUMBU", "gram", 7, 250, 12),
    ("Tomat", "BUAH", "gram", 5, 600, 45),
    ("Bayam", "SAYUR", "ikat", 2, 2, 0.5),
    ("Kangkung", "SAYUR", "ikat", 2, 2, 0.5),
    ("Wortel", "SAYUR", "gram", 14, 600, 35),
    ("Minyak Goreng", "CAIR", "ml", 365, 1000, 30),
    ("Sisa Masakan", "SISA", "pcs", 2, 3, 0.8),
]
RESEP_POOL = 40  # recipe_id sintetis "rcp_sint_000".."rcp_sint_039"; BUKAN id resep Pawonee nyata


def buat_baris(n: int = JUMLAH_BARIS, seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    baris = []
    for i in range(n):
        nama, kategori, satuan, umur, batch_tipikal, laju_tipikal = BAHAN[rng.integers(len(BAHAN))]
        sisa_hari = int(rng.integers(0, min(umur, 21) + 1))
        jumlah = batch_tipikal * rng.uniform(0.1, 0.7)
        laju = laju_tipikal * rng.lognormal(0.0, 0.6)
        bobot = float(rng.uniform(0.0, 1.0))
        if satuan in ("pcs", "ikat"):
            jumlah = max(1.0, round(jumlah))
        else:
            jumlah = round(jumlah, 1)
        laju_efektif = laju * (1.0 + 0.5 * bobot)  # asumsi: bahan populer di resep lebih sering dipakai
        # pemakaian harian dari hari ini sampai hari kedaluwarsa (sisa_hari + 1 hari), noise Gamma(2)
        harian = rng.gamma(2.0, laju_efektif / 2.0, size=sisa_hari + 1)
        terpakai = min(jumlah, float(harian.sum()))
        persen = terpakai / jumlah
        terbuang = int((1.0 - persen) > AMBANG_TERBUANG)
        punya_resep = rng.uniform() < 0.7
        recipe_id = f"rcp_sint_{int(rng.integers(RESEP_POOL)):03d}" if punya_resep else ""
        baris.append({
            "id": f"sint_{i:05d}", "nama": nama, "kategori": kategori, "satuan": satuan,
            "sisa_hari": sisa_hari, "jumlah_aktif": round(jumlah, 3),
            "laju_pakai_harian": round(laju, 3), "bobot_resep": round(bobot, 3),
            "recipe_id": recipe_id, "persen_terpakai": round(persen, 3),
            "terbuang": terbuang, "sumber": "SINTETIS",
        })
    return baris


def tulis(path: Path = BERKAS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KOLOM, lineterminator="\n")
        w.writeheader()
        w.writerows(buat_baris())
    return path


def baca(path: Path = BERKAS) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return [
            {**r, "sisa_hari": int(r["sisa_hari"]), "jumlah_aktif": float(r["jumlah_aktif"]),
             "laju_pakai_harian": float(r["laju_pakai_harian"]), "bobot_resep": float(r["bobot_resep"]),
             "persen_terpakai": float(r["persen_terpakai"]), "terbuang": int(r["terbuang"])}
            for r in csv.DictReader(f)
        ]



def pisah(baris: list[dict], porsi_uji: float = 0.3, seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Pisah latih/uji deterministik, terstratifikasi menurut label."""
    rng = np.random.default_rng(seed + 1)
    latih: list[dict] = []
    uji: list[dict] = []
    for label in (0, 1):
        kelompok = [r for r in baris if r["terbuang"] == label]
        idx = rng.permutation(len(kelompok))
        n_uji = int(round(len(kelompok) * porsi_uji))
        uji += [kelompok[i] for i in idx[:n_uji]]
        latih += [kelompok[i] for i in idx[n_uji:]]
    return sorted(latih, key=lambda r: r["id"]), sorted(uji, key=lambda r: r["id"])


if __name__ == "__main__":
    print(f"dataset sintetis tertulis: {tulis()}")
