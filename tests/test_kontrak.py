import json

import pytest
from pydantic import ValidationError

from src import baseline, fitur, inferensi, kontrak, kontrak_tulis, latih, versi
from src.kontrak import HasilInferensi, PermintaanInferensi

DIR = inferensi.KONTRAK_DIR


@pytest.fixture(scope="module")
def artefak(tmp_path_factory):
    return latih.latih(tmp_path_factory.mktemp("models"))


def _batch(**kw):
    dasar = {
        "batchId": "btc_AAAA0001", "pantryItemId": "itm_AAAA0001", "nama": "Tomat", "kategori": "BUAH",
        "satuan": "gram", "jumlahAktif": 300, "tanggalMasuk": "2026-10-18", "tanggalKedaluwarsa": "2026-10-23",
        "lajuPakaiHarian": 40, "bobotResep": 0.5,
    }
    dasar.update(kw)
    return dasar


def _minta(*batches, hari="2026-10-21"):
    return PermintaanInferensi.model_validate({"skemaVersi": "1.0", "hariIni": hari, "batch": list(batches)})


# ---- skema terbit dan terkunci ----
def test_skema_di_repo_identik_dengan_kode():
    bangun = kontrak_tulis.bangun()
    for nama in ("masukan.schema.json", "keluaran.schema.json", "vektor-fitur.json", "contoh-masukan.json"):
        assert json.loads((DIR / nama).read_text()) == bangun[nama], f"{nama} kedaluwarsa: jalankan python -m src.kontrak_tulis (+ ADR + bump versi)"


def test_skema_memuat_versi_kontrak_dan_model():
    for nama in ("masukan.schema.json", "keluaran.schema.json"):
        s = json.loads((DIR / nama).read_text())
        assert s["x-skema-versi"] == kontrak.SKEMA_VERSI == "1.0"
        assert s["x-model-versi"] == versi.MODEL_VERSI


def test_contoh_berkas_valid_menurut_pydantic():
    PermintaanInferensi.model_validate_json((DIR / "contoh-masukan.json").read_text())
    HasilInferensi.model_validate_json((DIR / "contoh-keluaran.json").read_text())


def test_vektor_fitur_emas_cocok_dengan_kode():
    emas = json.loads((DIR / "vektor-fitur.json").read_text())
    assert emas["namaFitur"] == fitur.NAMA_FITUR
    for k in emas["kasus"]:
        m = k["masukan"]
        got = fitur.vektor(m["sisaHari"], m["jumlahAktif"], m["lajuPakaiHarian"], m["bobotResep"])
        assert got == pytest.approx(k["fitur"], abs=emas["toleransi"])


# ---- validasi masukan ----
def test_menolak_properti_tak_dikenal_dan_versi_salah():
    with pytest.raises(ValidationError):
        _minta(_batch(rahasia=1))
    with pytest.raises(ValidationError):
        PermintaanInferensi.model_validate({"skemaVersi": "2.0", "hariIni": "2026-10-21", "batch": [_batch()]})


@pytest.mark.parametrize(
    "ubah",
    [
        {"satuan": "ons"},  # di luar 13 satuan baku
        {"satuan": "pcs", "jumlahAktif": 1.5},  # satuan hitung pecahan
        {"satuan": "gram", "jumlahAktif": 50_001},  # >50.000 g
        {"satuan": "kg", "jumlahAktif": 51},  # 51 kg = 51.000 g
        {"batchId": "x_123"},
        {"tanggalKedaluwarsa": "2026-10-01"},  # sebelum tanggalMasuk
        {"jumlahAktif": -1},
        {"bobotResep": 1.2},
        {"nama": "ab"},
    ],
)
def test_menolak_masukan_tidak_valid(ubah):
    with pytest.raises(ValidationError):
        _minta(_batch(**ubah))


def test_batch_id_duplikat_dan_daftar_kosong_ditolak():
    with pytest.raises(ValidationError):
        _minta(_batch(), _batch())
    with pytest.raises(ValidationError):
        PermintaanInferensi.model_validate({"skemaVersi": "1.0", "hariIni": "2026-10-21", "batch": []})


def test_batas_2000_batch():
    banyak = [_batch(batchId=f"btc_{i:08d}") for i in range(2001)]
    with pytest.raises(ValidationError):
        PermintaanInferensi.model_validate({"skemaVersi": "1.0", "hariIni": "2026-10-21", "batch": banyak})


def test_13_satuan_baku_tepat():
    assert [s.value for s in kontrak.Satuan] == [
        "gram", "kg", "ml", "liter", "sdm", "sdt", "pcs", "siung", "buah", "ikat", "batang", "bonggol", "lembar",
    ]


def test_normalisasi_satuan_kanonis():
    b = _minta(_batch(satuan="kg", jumlahAktif=2, lajuPakaiHarian=0.5)).batch[0]
    assert b.jumlah_kanonis() == 2000 and b.laju_kanonis() == 500
    assert _minta(_batch(satuan="sdm", jumlahAktif=2)).batch[0].jumlah_kanonis() == 30


# ---- perilaku inferensi ----
def test_contoh_cli_mengikuti_skema(artefak, capsys):
    assert inferensi.main(["--contoh", "--model", str(artefak)]) == 0
    keluar = capsys.readouterr().out
    bagian = keluar.split("# keluaran\n")
    PermintaanInferensi.model_validate_json(bagian[0].replace("# masukan\n", ""))
    HasilInferensi.model_validate_json(bagian[1])


def test_skema_cli_cocok_dengan_berkas(capsys):
    assert inferensi.main(["--skema"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["masukan"] == kontrak.skema_masukan() and s["keluaran"] == kontrak.skema_keluaran()


def test_status_dan_urutan_fefo_mengikuti_baseline_bukan_model(artefak):
    a = _batch(batchId="btc_AAAA000A", tanggalKedaluwarsa="2026-11-10", jumlahAktif=100)
    b = _batch(batchId="btc_AAAA000B", tanggalKedaluwarsa="2026-10-22", jumlahAktif=5000, lajuPakaiHarian=1)
    hasil = inferensi.prediksi(_minta(a, b), artefak)
    assert [h.batch_id for h in hasil.hasil] == ["btc_AAAA000B", "btc_AAAA000A"]
    assert [h.urutan_pakai for h in hasil.hasil] == [1, 2]
    assert [h.status_sop for h in hasil.hasil] == [baseline.StatusSop.SEGERA, baseline.StatusSop.AMAN]
    assert hasil.model.versi == versi.MODEL_VERSI and hasil.model.sumber_model == "onnx"


def test_batch_kedaluwarsa_dan_kosong_terkunci_di_akhir(artefak):
    lewat = _batch(batchId="btc_AAAA000C", tanggalKedaluwarsa="2026-10-20", tanggalMasuk="2026-10-01")
    kosong = _batch(batchId="btc_AAAA000D", jumlahAktif=0)
    ok = _batch(batchId="btc_AAAA000E")
    hasil = inferensi.prediksi(_minta(lewat, kosong, ok), artefak).hasil
    assert hasil[0].batch_id == "btc_AAAA000E" and not hasil[0].terkunci
    kunci = {h.batch_id: h for h in hasil[1:]}
    assert all(h.terkunci and h.urutan_pakai is None for h in kunci.values())
    assert kunci["btc_AAAA000C"].skor_risiko == 1.0 and kunci["btc_AAAA000C"].sumber_skor == "baseline"


def test_tanggal_null_tanpa_status_dan_skor_nol(artefak):
    h = inferensi.prediksi(_minta(_batch(tanggalKedaluwarsa=None, nama="Garam")), artefak).hasil[0]
    assert h.status_sop is None and h.sisa_hari is None and h.skor_risiko == 0.0 and h.urutan_pakai == 1


def test_skor_dalam_rentang_dan_model_membedakan_di_dalam_status_sama(artefak):
    cepat = _batch(batchId="btc_AAAA000F", jumlahAktif=100, lajuPakaiHarian=100)
    lambat = _batch(batchId="btc_AAAA000G", jumlahAktif=3000, lajuPakaiHarian=5)
    h = {x.batch_id: x for x in inferensi.prediksi(_minta(cepat, lambat), artefak).hasil}
    assert h["btc_AAAA000F"].status_sop == h["btc_AAAA000G"].status_sop
    assert 0 <= h["btc_AAAA000F"].skor_risiko < h["btc_AAAA000G"].skor_risiko <= 1


def test_tanpa_artefak_semua_skor_dari_baseline(tmp_path):
    hasil = inferensi.prediksi(_minta(_batch()), tmp_path / "tidak-ada.onnx")
    assert hasil.model.sumber_model == "baseline"
    assert hasil.hasil[0].sumber_skor == "baseline" and hasil.hasil[0].skor_risiko == 0.85


def test_keluaran_deterministik_dan_hasil_sama_dengan_contoh_berkas(artefak):
    a = inferensi.prediksi(inferensi.contoh_permintaan(), artefak)
    b = inferensi.prediksi(inferensi.contoh_permintaan(), artefak)
    assert a == b
    disk = HasilInferensi.model_validate_json((DIR / "contoh-keluaran.json").read_text())
    kunci = lambda h: [(x.batch_id, x.status_sop, x.sisa_hari, x.urutan_pakai, x.terkunci, x.sumber_skor) for x in h.hasil]  # noqa: E731
    assert kunci(a) == kunci(disk)  # skor numerik bergantung artefak terlatih; struktur harus tetap


def test_fitur_item_tingkat_rendah_masih_berfungsi(artefak):
    aman = inferensi.FiturItem(sisa_hari=2, jumlah_aktif=100, laju_pakai_harian=100)
    bahaya = inferensi.FiturItem(sisa_hari=2, jumlah_aktif=2000, laju_pakai_harian=10)
    assert inferensi.skor_risiko(bahaya, artefak) > inferensi.skor_risiko(aman, artefak)


def test_adr_kontrak_ada():
    adr = next((DIR.parent / "decisions").glob("ADR-0003-*.md")).read_text()
    assert "1.0" in adr and "bump" in adr.lower()


def test_cli_masukan_berkas_dan_galat_validasi(artefak, tmp_path, capsys):
    f = tmp_path / "req.json"
    f.write_text((DIR / "contoh-masukan.json").read_text())
    assert inferensi.main(["--masukan", str(f), "--model", str(artefak)]) == 0
    HasilInferensi.model_validate_json(capsys.readouterr().out)
    f.write_text('{"skemaVersi": "1.0", "hariIni": "2026-10-21", "batch": []}')
    with pytest.raises(ValidationError):
        inferensi.main(["--masukan", str(f), "--model", str(artefak)])
