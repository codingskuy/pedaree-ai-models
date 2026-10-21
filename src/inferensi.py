"""Inferensi risiko kedaluwarsa memakai model ONNX ringan.

Kontrak masukan/keluaran resmi untuk API Rust ditetapkan di PDR-AI-004.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, Field

from src import fitur, versi
from src.baseline import skor_baseline

MODEL = Path(__file__).resolve().parent.parent / "models" / versi.BERKAS_ONNX
_sesi: dict[Path, ort.InferenceSession] = {}


class FiturItem(BaseModel):
    sisa_hari: int = Field(ge=0, description="hari tersisa sampai kedaluwarsa (batch KEDALUWARSA tidak dinilai model)")
    jumlah_aktif: float = Field(gt=0)
    laju_pakai_harian: float = Field(ge=0)
    bobot_resep: float = Field(default=0.0, ge=0.0, le=1.0)


def _sesi_onnx(jalur: Path) -> ort.InferenceSession:
    if jalur not in _sesi:
        _sesi[jalur] = ort.InferenceSession(str(jalur), providers=["CPUExecutionProvider"])
    return _sesi[jalur]


def skor_risiko(item: FiturItem, jalur: Path = MODEL) -> float:
    """P(batch terbuang) 0..1. Tanpa artefak model, jatuh ke skor baseline SOP (deterministik)."""
    if not jalur.exists():
        hari = date(2000, 1, 1)
        return skor_baseline(hari + timedelta(days=item.sisa_hari), hari)
    x = np.array(
        [fitur.vektor(item.sisa_hari, item.jumlah_aktif, item.laju_pakai_harian, item.bobot_resep)],
        dtype=np.float32,
    )
    return float(_sesi_onnx(jalur).run(None, {"masukan": x})[1][0][1])


if __name__ == "__main__":
    contoh = FiturItem(sisa_hari=2, jumlah_aktif=500, laju_pakai_harian=40.0, bobot_resep=0.9)
    print(f"skor risiko: {skor_risiko(contoh):.2f}")
