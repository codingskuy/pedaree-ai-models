"""Rekayasa fitur bersama (latih, evaluasi, inferensi) agar tidak ada train/serve skew."""
from __future__ import annotations

import numpy as np

NAMA_FITUR = [
    "sisa_hari",
    "laju_pakai_harian",
    "bobot_resep",
    "hari_habis_dipakai",  # jumlah_aktif / laju_pakai_harian (dibatasi 365)
    "rasio_habis_vs_sisa",  # hari_habis_dipakai / (sisa_hari + 1)
]
LAJU_MIN = 1e-3
HARI_MAKS = 365.0


def vektor(sisa_hari: float, jumlah_aktif: float, laju_pakai_harian: float, bobot_resep: float) -> list[float]:
    laju = max(laju_pakai_harian, LAJU_MIN)
    habis = min(jumlah_aktif / laju, HARI_MAKS)
    return [float(sisa_hari), float(laju_pakai_harian), float(bobot_resep), habis, habis / (sisa_hari + 1.0)]


def matriks(baris: list[dict]) -> np.ndarray:
    return np.array(
        [vektor(r["sisa_hari"], r["jumlah_aktif"], r["laju_pakai_harian"], r["bobot_resep"]) for r in baris],
        dtype=np.float32,
    )
