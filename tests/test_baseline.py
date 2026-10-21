from datetime import date, timedelta

import pytest

from src.baseline import (
    Batch,
    StatusSop,
    alokasi_pakai,
    hari_tersisa,
    skor_baseline,
    status_dari_sisa_hari,
    status_sop,
    urutan_fefo,
)

HARI_INI = date(2026, 10, 21)


@pytest.mark.parametrize(
    ("sisa", "status"),
    [
        (30, StatusSop.AMAN),
        (8, StatusSop.AMAN),
        (7, StatusSop.WASPADA),
        (4, StatusSop.WASPADA),
        (3, StatusSop.SEGERA),
        (1, StatusSop.SEGERA),
        (0, StatusSop.HARI_INI),
        (-1, StatusSop.KEDALUWARSA),
        (-30, StatusSop.KEDALUWARSA),
    ],
)
def test_ambang_sop(sisa, status):
    assert status_dari_sisa_hari(sisa) == status


def test_status_dari_tanggal():
    assert hari_tersisa(HARI_INI + timedelta(days=5), HARI_INI) == 5
    assert status_sop(HARI_INI + timedelta(days=5), HARI_INI) == StatusSop.WASPADA
    assert status_sop(HARI_INI - timedelta(days=1), HARI_INI) == StatusSop.KEDALUWARSA


def test_tanggal_null_di_luar_sop():
    assert status_sop(None, HARI_INI) is None
    assert skor_baseline(None, HARI_INI) == 0.0


def test_sisa_masakan_umur_2_hari_langsung_segera():
    masuk = HARI_INI
    kedaluwarsa = masuk + timedelta(days=2)
    assert status_sop(kedaluwarsa, masuk) == StatusSop.SEGERA
    assert status_sop(kedaluwarsa, masuk + timedelta(days=1)) == StatusSop.SEGERA


def test_skor_baseline_monoton_naik_dengan_urgensi():
    urut = [StatusSop.AMAN, StatusSop.WASPADA, StatusSop.SEGERA, StatusSop.HARI_INI, StatusSop.KEDALUWARSA]
    from src.baseline import SKOR_BASELINE

    skor = [SKOR_BASELINE[s] for s in urut]
    assert skor == sorted(skor) and len(set(skor)) == 5


def _b(id_, jumlah, masuk, kedaluwarsa):
    return Batch(id_, "itm_x", jumlah, masuk, kedaluwarsa)


def test_fefo_kedaluwarsa_terdekat_dulu():
    a = _b("a", 10, date(2026, 10, 1), HARI_INI + timedelta(days=9))
    b = _b("b", 10, date(2026, 10, 15), HARI_INI + timedelta(days=2))
    assert [x.id for x in urutan_fefo([a, b], HARI_INI)] == ["b", "a"]


def test_fefo_null_fifo_dan_setelah_bertanggal():
    tua = _b("tua", 5, date(2026, 8, 1), None)
    baru = _b("baru", 5, date(2026, 10, 1), None)
    bertanggal = _b("t", 5, date(2026, 10, 10), HARI_INI + timedelta(days=30))
    assert [x.id for x in urutan_fefo([baru, tua, bertanggal], HARI_INI)] == ["t", "tua", "baru"]


def test_fefo_mengunci_batch_kedaluwarsa_dan_kosong():
    lewat = _b("lewat", 5, date(2026, 9, 1), HARI_INI - timedelta(days=1))
    kosong = _b("kosong", 0, date(2026, 9, 1), HARI_INI + timedelta(days=1))
    ok = _b("ok", 5, date(2026, 9, 1), HARI_INI + timedelta(days=1))
    assert [x.id for x in urutan_fefo([lewat, kosong, ok], HARI_INI)] == ["ok"]


def test_hari_ini_masih_ikut_fefo():
    hi = _b("hi", 5, date(2026, 9, 1), HARI_INI)
    assert [x.id for x in urutan_fefo([hi], HARI_INI)] == ["hi"]


def test_alokasi_memotong_batch_tercepat_dulu():
    a = _b("a", 100, date(2026, 10, 1), HARI_INI + timedelta(days=9))
    b = _b("b", 30, date(2026, 10, 15), HARI_INI + timedelta(days=2))
    assert alokasi_pakai([a, b], 50, HARI_INI) == [("b", 30), ("a", 20)]


def test_alokasi_stok_tidak_cukup_dan_jumlah_tidak_valid():
    a = _b("a", 10, date(2026, 10, 1), HARI_INI + timedelta(days=9))
    lewat = _b("l", 100, date(2026, 9, 1), HARI_INI - timedelta(days=1))
    with pytest.raises(ValueError):
        alokasi_pakai([a, lewat], 11, HARI_INI)  # batch kedaluwarsa tidak dihitung
    with pytest.raises(ValueError):
        alokasi_pakai([a], 0, HARI_INI)
