"""Inferensi risiko kedaluwarsa (model ONNX) di atas baseline SOP/FEFO terkunci.

Alur (docs/kontrak/INFERENSI.md): validasi `PermintaanInferensi` -> status SOP + urutan FEFO dari
`src/baseline.py` (model TIDAK boleh mengubahnya) -> skor risiko model untuk batch yang
masih hidup -> `HasilInferensi`. Serving produksi lewat API Rust; modul ini adalah implementasi
acuan dan sumber vektor uji, bukan layanan.

CLI: `python -m src.inferensi --contoh | --masukan berkas.json | --skema`.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, Field

from src import baseline, fitur, versi
from src.baseline import Batch, StatusSop
from src.kontrak import (
    BatchMasukan, HasilBatch, HasilInferensi, InfoModel, PermintaanInferensi, SumberSkor,
    skema_keluaran, skema_masukan,
)

MODEL = Path(__file__).resolve().parent.parent / "models" / versi.BERKAS_ONNX
KONTRAK_DIR = Path(__file__).resolve().parent.parent / "docs" / "kontrak"
_sesi: dict[Path, ort.InferenceSession] = {}


class FiturItem(BaseModel):
    """Fitur mentah satu batch (API tingkat rendah; dipakai uji/evaluasi)."""

    sisa_hari: int = Field(ge=0, description="hari tersisa sampai kedaluwarsa (batch KEDALUWARSA tidak dinilai model)")
    jumlah_aktif: float = Field(gt=0)
    laju_pakai_harian: float = Field(ge=0)
    bobot_resep: float = Field(default=0.0, ge=0.0, le=1.0)


def _sesi_onnx(jalur: Path) -> ort.InferenceSession:
    if jalur not in _sesi:
        _sesi[jalur] = ort.InferenceSession(str(jalur), providers=["CPUExecutionProvider"])
    return _sesi[jalur]


def _skor_model(jalur: Path, x: np.ndarray) -> np.ndarray:
    return _sesi_onnx(jalur).run(None, {"masukan": x})[1][:, 1]


def skor_risiko(item: FiturItem, jalur: Path = MODEL) -> float:
    """P(batch terbuang) 0..1. Tanpa artefak model, jatuh ke skor baseline SOP (deterministik)."""
    if not jalur.exists():
        hari = date(2000, 1, 1)
        return baseline.skor_baseline(hari + timedelta(days=item.sisa_hari), hari)
    x = np.array(
        [fitur.vektor(item.sisa_hari, item.jumlah_aktif, item.laju_pakai_harian, item.bobot_resep)],
        dtype=np.float32,
    )
    return float(_skor_model(jalur, x)[0])


def _ke_baseline(b: BatchMasukan) -> Batch:
    return Batch(b.batch_id, b.pantry_item_id, b.jumlah_kanonis(), b.tanggal_masuk, b.tanggal_kedaluwarsa)


def prediksi(permintaan: PermintaanInferensi, jalur: Path = MODEL) -> HasilInferensi:
    hari = permintaan.hari_ini
    pakai_model = jalur.exists()
    batches = [_ke_baseline(b) for b in permintaan.batch]
    urut = baseline.urutan_fefo(batches, hari)
    peringkat_global = {b.id: i for i, b in enumerate(urut)}
    per_item: dict[str, int] = {}
    urutan_item: dict[str, int] = {}
    for b in urut:
        per_item[b.pantry_item_id] = per_item.get(b.pantry_item_id, 0) + 1
        urutan_item[b.id] = per_item[b.pantry_item_id]

    status = {b.batch_id: baseline.status_sop(b.tanggal_kedaluwarsa, hari) for b in permintaan.batch}
    # Hanya batch hidup bertanggal yang dinilai model (KEDALUWARSA/kosong/tanpa tanggal = aturan tetap).
    dinilai = [
        b for b in permintaan.batch
        if status[b.batch_id] not in (None, StatusSop.KEDALUWARSA) and b.jumlah_aktif > 0
    ]
    skor_m: dict[str, float] = {}
    if pakai_model and dinilai:
        x = np.array(
            [
                fitur.vektor(
                    baseline.hari_tersisa(b.tanggal_kedaluwarsa, hari), b.jumlah_kanonis(), b.laju_kanonis(), b.bobot_resep
                )
                for b in dinilai
            ],
            dtype=np.float32,
        )
        for b, s in zip(dinilai, _skor_model(jalur, x), strict=True):
            skor_m[b.batch_id] = float(s)

    hasil: list[HasilBatch] = []
    for b in permintaan.batch:
        st = status[b.batch_id]
        terkunci = b.batch_id not in peringkat_global
        if b.batch_id in skor_m:
            skor, sumber = skor_m[b.batch_id], SumberSkor.MODEL
        else:
            skor, sumber = baseline.skor_baseline(b.tanggal_kedaluwarsa, hari), SumberSkor.BASELINE
        hasil.append(
            HasilBatch(
                batch_id=b.batch_id,
                pantry_item_id=b.pantry_item_id,
                status_sop=st,
                sisa_hari=None if b.tanggal_kedaluwarsa is None else baseline.hari_tersisa(b.tanggal_kedaluwarsa, hari),
                skor_risiko=round(min(max(skor, 0.0), 1.0), 4),
                sumber_skor=sumber,
                urutan_pakai=None if terkunci else urutan_item[b.batch_id],
                terkunci=terkunci,
            )
        )
    # urut FEFO global; batch terkunci di akhir (stabil menurut urutan masukan)
    hasil.sort(key=lambda h: (h.terkunci, peringkat_global.get(h.batch_id, 0)))
    return HasilInferensi(
        hari_ini=hari,
        model=InfoModel(nama=versi.MODEL_NAMA, versi=versi.MODEL_VERSI, sumber_model="onnx" if pakai_model else "baseline"),
        hasil=hasil,
    )


def contoh_permintaan() -> PermintaanInferensi:
    """Contoh resmi (tanggal tetap agar deterministik); sama dengan docs/kontrak/contoh-masukan.json."""
    return PermintaanInferensi.model_validate(
        {
            "skemaVersi": "1.0",
            "hariIni": "2026-10-21",
            "batch": [
                {"batchId": "btc_01J9Z8QK2M4C", "pantryItemId": "itm_01J9Z8QK2M4A", "nama": "Bawang Merah",
                 "kategori": "BUMBU", "satuan": "gram", "jumlahAktif": 650, "tanggalMasuk": "2026-09-28",
                 "tanggalKedaluwarsa": "2026-10-23", "lajuPakaiHarian": 15, "bobotResep": 0.9, "recipeId": None},
                {"batchId": "btc_01J9Z8QK2M4D", "pantryItemId": "itm_01J9Z8QK2M4A", "nama": "Bawang Merah",
                 "kategori": "BUMBU", "satuan": "gram", "jumlahAktif": 500, "tanggalMasuk": "2026-10-15",
                 "tanggalKedaluwarsa": "2026-11-05", "lajuPakaiHarian": 15, "bobotResep": 0.9, "recipeId": None},
                {"batchId": "btc_01J9Z8QK2M4E", "pantryItemId": "itm_01J9Z8QK2M4F", "nama": "Bayam",
                 "kategori": "SAYUR", "satuan": "ikat", "jumlahAktif": 2, "tanggalMasuk": "2026-10-18",
                 "tanggalKedaluwarsa": "2026-10-20", "lajuPakaiHarian": 0.5, "bobotResep": 0.4, "recipeId": None},
                {"batchId": "btc_01J9Z8QK2M4G", "pantryItemId": "itm_01J9Z8QK2M4H", "nama": "Garam",
                 "kategori": "KERING", "satuan": "gram", "jumlahAktif": 250, "tanggalMasuk": "2026-08-01",
                 "tanggalKedaluwarsa": None, "lajuPakaiHarian": 3, "bobotResep": 1.0, "recipeId": None},
            ],
        }
    )


def _json(model: BaseModel) -> str:
    return model.model_dump_json(by_alias=True, indent=2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--contoh", action="store_true", help="jalankan contoh resmi (masukan + keluaran)")
    g.add_argument("--masukan", type=Path, help="berkas JSON PermintaanInferensi")
    g.add_argument("--skema", action="store_true", help="cetak skema JSON masukan dan keluaran")
    ap.add_argument("--model", type=Path, default=MODEL)
    a = ap.parse_args(argv)
    if a.skema:
        print(json.dumps({"masukan": skema_masukan(), "keluaran": skema_keluaran()}, indent=2, ensure_ascii=False))
        return 0
    permintaan = contoh_permintaan() if a.contoh else PermintaanInferensi.model_validate_json(a.masukan.read_text())
    keluaran = prediksi(permintaan, a.model)
    if a.contoh:
        print("# masukan\n" + _json(permintaan) + "\n# keluaran")
    print(_json(keluaran))
    return 0


if __name__ == "__main__":
    sys.exit(main())
