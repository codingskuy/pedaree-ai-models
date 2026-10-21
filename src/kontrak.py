"""Kontrak masukan/keluaran inferensi model untuk API Rust (PDR-AI-004, skema 1.0).

Bentuk JSON mengikuti gaya API Pedaree (camelCase, ULID berawalan, tanggal `YYYY-MM-DD`,
13 satuan baku). Skema JSON diterbitkan di `docs/kontrak/` dan dijaga uji agar tidak
berubah diam-diam: perubahan skema = bump `SKEMA_VERSI` + `MODEL_VERSI` + ADR baru.
"""
from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

from src.baseline import StatusSop

SKEMA_VERSI = "1.0"
MAKS_BATCH = 2000  # batas Varian 1 per rumah tangga (inventori/10 bab 4)
MAKS_GRAM_ML = 50_000  # batas per batch (AGENTS.md)


class Kategori(StrEnum):
    SAYUR = "SAYUR"
    BUMBU = "BUMBU"
    BUAH = "BUAH"
    PROTEIN = "PROTEIN"
    KERING = "KERING"
    CAIR = "CAIR"
    BEKU = "BEKU"
    SISA = "SISA"


class Satuan(StrEnum):
    """13 satuan baku (sumber sah: inventori/20-satuan-hortikultura.md)."""

    GRAM = "gram"
    KG = "kg"
    ML = "ml"
    LITER = "liter"
    SDM = "sdm"
    SDT = "sdt"
    PCS = "pcs"
    SIUNG = "siung"
    BUAH = "buah"
    IKAT = "ikat"
    BATANG = "batang"
    BONGGOL = "bonggol"
    LEMBAR = "lembar"


# Satuan -> (satuan kanonis penyimpanan, pengali). kg=1000 g; liter=1000 ml; sdm=15 ml; sdt=5 ml.
KANONIS: dict[Satuan, tuple[Satuan, float]] = {
    Satuan.GRAM: (Satuan.GRAM, 1), Satuan.KG: (Satuan.GRAM, 1000),
    Satuan.ML: (Satuan.ML, 1), Satuan.LITER: (Satuan.ML, 1000),
    Satuan.SDM: (Satuan.ML, 15), Satuan.SDT: (Satuan.ML, 5),
    Satuan.PCS: (Satuan.PCS, 1), Satuan.SIUNG: (Satuan.SIUNG, 1), Satuan.BUAH: (Satuan.BUAH, 1),
    Satuan.IKAT: (Satuan.IKAT, 1), Satuan.BATANG: (Satuan.BATANG, 1),
    Satuan.BONGGOL: (Satuan.BONGGOL, 1), Satuan.LEMBAR: (Satuan.LEMBAR, 1),
}
SATUAN_HITUNG = {Satuan.PCS, Satuan.SIUNG, Satuan.BUAH, Satuan.IKAT, Satuan.BATANG, Satuan.BONGGOL, Satuan.LEMBAR}


class _Dasar(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class BatchMasukan(_Dasar):
    """Satu StokBatch aktif + fitur konteks yang dihitung backend."""

    batch_id: str = Field(pattern=r"^btc_[0-9A-Za-z]{4,40}$", examples=["btc_01J9Z8QK2M4C"])
    pantry_item_id: str = Field(pattern=r"^itm_[0-9A-Za-z]{4,40}$", examples=["itm_01J9Z8QK2M4A"])
    nama: str = Field(min_length=3, max_length=60)
    kategori: Kategori
    satuan: Satuan = Field(description="Satuan `jumlahAktif` dan `lajuPakaiHarian`; dinormalkan ke satuan kanonis.")
    jumlah_aktif: float = Field(ge=0, description="Sisa batch; 0 = habis (terkunci dari FEFO).")
    tanggal_masuk: date
    tanggal_kedaluwarsa: date | None = Field(default=None, description="null = tidak mudah busuk (di luar SOP).")
    laju_pakai_harian: float = Field(
        default=0.0, ge=0,
        description="ASUMSI: rerata pemakaian per hari item ini dalam `satuan`, dihitung backend dari MutasiStok PAKAI (mis. 14 hari terakhir).",
    )
    bobot_resep: float = Field(default=0.0, ge=0, le=1, description="0..1, seberapa sering bahan muncul di resep Pawonee (konteks).")
    recipe_id: str | None = Field(default=None, max_length=64, description="Rujukan resep Pawonee; hanya konteks, boleh null.")

    @model_validator(mode="after")
    def _validasi_domain(self) -> BatchMasukan:
        if self.tanggal_kedaluwarsa is not None and self.tanggal_kedaluwarsa < self.tanggal_masuk:
            raise ValueError("tanggalKedaluwarsa tidak boleh sebelum tanggalMasuk")
        kanonis, pengali = KANONIS[self.satuan]
        if kanonis in SATUAN_HITUNG and self.jumlah_aktif != int(self.jumlah_aktif):
            raise ValueError(f"satuan hitung '{self.satuan.value}' harus bilangan bulat")
        if kanonis in (Satuan.GRAM, Satuan.ML) and self.jumlah_aktif * pengali > MAKS_GRAM_ML:
            raise ValueError(f"jumlah per batch maksimal {MAKS_GRAM_ML} gram/ml (pecah menjadi beberapa batch)")
        return self

    def jumlah_kanonis(self) -> float:
        return self.jumlah_aktif * KANONIS[self.satuan][1]

    def laju_kanonis(self) -> float:
        return self.laju_pakai_harian * KANONIS[self.satuan][1]


class PermintaanInferensi(_Dasar):
    skema_versi: Literal["1.0"] = SKEMA_VERSI
    hari_ini: date = Field(description="Tanggal kalender WIB acuan perhitungan sisa hari.")
    batch: list[BatchMasukan] = Field(min_length=1, max_length=MAKS_BATCH)

    @model_validator(mode="after")
    def _id_unik(self) -> PermintaanInferensi:
        ids = [b.batch_id for b in self.batch]
        if len(ids) != len(set(ids)):
            raise ValueError("batchId harus unik dalam satu permintaan")
        return self


class SumberSkor(StrEnum):
    MODEL = "model"
    BASELINE = "baseline"


class InfoModel(_Dasar):
    nama: str
    versi: str
    sumber_model: Literal["onnx", "baseline"] = Field(description="`baseline` bila artefak ONNX tidak tersedia.")


class HasilBatch(_Dasar):
    batch_id: str
    pantry_item_id: str
    status_sop: StatusSop | None = Field(description="Dari baseline SOP (terkunci). null bila tanggalKedaluwarsa null.")
    sisa_hari: int | None = Field(description="Hari tersisa; null bila tanggalKedaluwarsa null; negatif bila lewat.")
    skor_risiko: float = Field(ge=0, le=1, description="P(batch terbuang). Hanya penambah informasi; tidak mengubah status/urutan.")
    sumber_skor: SumberSkor = Field(description="`baseline` untuk batch KEDALUWARSA (1.0), tanggal null (0.0), atau tanpa model.")
    urutan_pakai: int | None = Field(
        description="Peringkat FEFO (1 = pakai dulu) di dalam item yang sama; null bila terkunci (KEDALUWARSA atau jumlah 0).",
    )
    terkunci: bool = Field(description="true bila KEDALUWARSA atau jumlahAktif = 0: tidak ikut FEFO (hanya bisa dibuang).")


class HasilInferensi(_Dasar):
    skema_versi: Literal["1.0"] = SKEMA_VERSI
    hari_ini: date
    model: InfoModel
    hasil: list[HasilBatch] = Field(
        description="Terurut FEFO global: kedaluwarsa terdekat dulu, tanggal null (FIFO) setelahnya, batch terkunci terakhir.",
    )


def skema_masukan() -> dict:
    return PermintaanInferensi.model_json_schema(by_alias=True, mode="validation")


def skema_keluaran() -> dict:
    return HasilInferensi.model_json_schema(by_alias=True, mode="serialization")
