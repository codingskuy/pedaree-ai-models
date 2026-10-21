"""Tulis ulang artefak kontrak di docs/kontrak/ dari kode (sumber kebenaran = src/kontrak.py).

Jalankan: `python -m src.kontrak_tulis`. Uji `tests/test_kontrak.py` memastikan berkas di repo
identik dengan hasil di sini, sehingga perubahan skema selalu terlihat di diff (dan wajib ADR + bump versi).
"""
from __future__ import annotations

import json
from pathlib import Path

from src import fitur, inferensi, versi
from src.kontrak import SKEMA_VERSI, skema_keluaran, skema_masukan

KASUS_FITUR = [
    {"sisaHari": 0, "jumlahAktif": 500, "lajuPakaiHarian": 40, "bobotResep": 0.9},
    {"sisaHari": 2, "jumlahAktif": 650, "lajuPakaiHarian": 15, "bobotResep": 0.9},
    {"sisaHari": 3, "jumlahAktif": 2, "lajuPakaiHarian": 0.5, "bobotResep": 0.4},
    {"sisaHari": 7, "jumlahAktif": 5000, "lajuPakaiHarian": 250, "bobotResep": 0.0},
    {"sisaHari": 14, "jumlahAktif": 100, "lajuPakaiHarian": 0, "bobotResep": 1.0},
]


def _tulis(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def bangun() -> dict[str, dict]:
    meta = {"x-skema-versi": SKEMA_VERSI, "x-model-versi": versi.MODEL_VERSI}
    permintaan = inferensi.contoh_permintaan()
    keluaran = inferensi.prediksi(permintaan)
    vektor = [
        {"masukan": k, "fitur": [round(v, 6) for v in fitur.vektor(k["sisaHari"], k["jumlahAktif"], k["lajuPakaiHarian"], k["bobotResep"])]}
        for k in KASUS_FITUR
    ]
    return {
        "masukan.schema.json": {**skema_masukan(), **meta},
        "keluaran.schema.json": {**skema_keluaran(), **meta},
        "contoh-masukan.json": json.loads(permintaan.model_dump_json(by_alias=True)),
        "contoh-keluaran.json": json.loads(keluaran.model_dump_json(by_alias=True)),
        "vektor-fitur.json": {"namaFitur": fitur.NAMA_FITUR, "toleransi": 1e-4, "kasus": vektor},
    }


def tulis(dir_: Path = inferensi.KONTRAK_DIR) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for nama, data in bangun().items():
        _tulis(dir_ / nama, data)


if __name__ == "__main__":
    tulis()
    print(f"artefak kontrak tertulis di {inferensi.KONTRAK_DIR}")
