"""Baseline deterministik: ambang SOP kedaluwarsa dan urutan FEFO.

Sumber (dokumen non-beku Pedaree-TownHall, ditandai sebagai asumsi kerja
karena FSD/FRD/PRD/SRS v1.0.0 masih templat):
- `notifikasi/10-sop-kedaluwarsa.md` bab 2 (ambang), bab 6 (pengecualian);
- `inventori/10-model-stok-dapur.md` bab 2.4 (status batch) dan bab 3 butir 5 (FEFO).

Modul ini TIDAK boleh diubah oleh model: ambang dan urutan FEFO terkunci
(AGENTS.md, invarian domain). Model hanya menambah skor risiko di atasnya.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class StatusSop(StrEnum):
    AMAN = "AMAN"
    WASPADA = "WASPADA"
    SEGERA = "SEGERA"
    HARI_INI = "HARI_INI"
    KEDALUWARSA = "KEDALUWARSA"


# Skor risiko baseline per status: pembanding bagi model (PDR-AI-003).
# Nilai mengikuti heuristik cadangan lama di src/inferensi.py (1.0 / 0.85 / 0.6 / 0.2),
# ditambah HARI_INI yang sebelumnya menyatu dengan SEGERA (ASUMSI: 0.95).
SKOR_BASELINE: dict[StatusSop, float] = {
    StatusSop.KEDALUWARSA: 1.0,
    StatusSop.HARI_INI: 0.95,
    StatusSop.SEGERA: 0.85,
    StatusSop.WASPADA: 0.6,
    StatusSop.AMAN: 0.2,
}


def hari_tersisa(tanggal_kedaluwarsa: date, hari_ini: date) -> int:
    """Selisih hari kalender sampai kedaluwarsa (negatif bila sudah lewat).

    SOP bab 2 menulis selisih `hariIni - tanggalKedaluwarsa`, tetapi tabel ambang
    dinyatakan dalam "hari tersisa"; fungsi ini mengikuti tabel ambang.
    """
    return (tanggal_kedaluwarsa - hari_ini).days


def status_dari_sisa_hari(sisa_hari: int) -> StatusSop:
    """Petakan hari tersisa ke status SOP: >7 AMAN, 4-7 WASPADA, 1-3 SEGERA, 0 HARI_INI, <0 KEDALUWARSA."""
    if sisa_hari < 0:
        return StatusSop.KEDALUWARSA
    if sisa_hari == 0:
        return StatusSop.HARI_INI
    if sisa_hari <= 3:
        return StatusSop.SEGERA
    if sisa_hari <= 7:
        return StatusSop.WASPADA
    return StatusSop.AMAN


def status_sop(tanggal_kedaluwarsa: date | None, hari_ini: date) -> StatusSop | None:
    """Status SOP sebuah batch; `None` bila tanggal kedaluwarsa null (garam, gula: di luar SOP).

    Catatan SISA (SOP bab 6): umur default 2 hari sehingga batch SISA sudah
    SEGERA sejak hari masuk; tidak diperlukan kasus khusus.
    """
    if tanggal_kedaluwarsa is None:
        return None
    return status_dari_sisa_hari(hari_tersisa(tanggal_kedaluwarsa, hari_ini))


def skor_baseline(tanggal_kedaluwarsa: date | None, hari_ini: date) -> float:
    """Skor risiko baseline 0..1; batch tanpa tanggal kedaluwarsa berisiko 0."""
    status = status_sop(tanggal_kedaluwarsa, hari_ini)
    return 0.0 if status is None else SKOR_BASELINE[status]


@dataclass(frozen=True)
class Batch:
    """Subset StokBatch (inventori/10 bab 2.4) yang relevan untuk FEFO."""

    id: str
    pantry_item_id: str
    jumlah_aktif: float
    tanggal_masuk: date
    tanggal_kedaluwarsa: date | None


def layak_fefo(batch: Batch, hari_ini: date) -> bool:
    """Batch ikut FEFO bila masih ada jumlah dan belum KEDALUWARSA (batch kedaluwarsa terkunci, hanya bisa dibuang)."""
    if batch.jumlah_aktif <= 0:
        return False
    return status_sop(batch.tanggal_kedaluwarsa, hari_ini) != StatusSop.KEDALUWARSA


def urutan_fefo(batches: list[Batch], hari_ini: date) -> list[Batch]:
    """Urutan pemakaian FEFO: kedaluwarsa terdekat dulu; tanggal null = FIFO (masuk tertua dulu).

    ASUMSI (dokumen tidak eksplisit): dalam satu item, batch bertanggal diutamakan
    sebelum batch tanpa tanggal. Tie-break: tanggal masuk lalu id (deterministik).
    """
    layak = [b for b in batches if layak_fefo(b, hari_ini)]
    return sorted(
        layak,
        key=lambda b: (
            b.tanggal_kedaluwarsa is None,
            b.tanggal_kedaluwarsa or date.max,
            b.tanggal_masuk,
            b.id,
        ),
    )


def alokasi_pakai(batches: list[Batch], jumlah: float, hari_ini: date) -> list[tuple[str, float]]:
    """Potong `jumlah` dari batch menurut urutan FEFO; hasil: daftar (id batch, jumlah diambil).

    Mengangkat ValueError bila jumlah tidak positif atau stok layak-FEFO tidak cukup.
    """
    if jumlah <= 0:
        raise ValueError("jumlah pakai harus > 0")
    sisa = jumlah
    hasil: list[tuple[str, float]] = []
    for batch in urutan_fefo(batches, hari_ini):
        ambil = min(batch.jumlah_aktif, sisa)
        hasil.append((batch.id, ambil))
        sisa = round(sisa - ambil, 6)
        if sisa <= 0:
            return hasil
    raise ValueError("stok layak-FEFO tidak cukup")
