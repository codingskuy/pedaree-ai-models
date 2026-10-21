import numpy as np
import onnx
import onnxruntime as ort
import pytest

from eval import dataset, evaluasi
from src import fitur, inferensi, latih, versi


@pytest.fixture(scope="module")
def artefak(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    return latih.latih(d)


def test_onnx_sah_dan_bermetadata_versi(artefak):
    m = onnx.load(str(artefak))
    onnx.checker.check_model(m)
    meta = {p.key: p.value for p in m.metadata_props}
    assert meta["model_versi"] == versi.MODEL_VERSI
    assert meta["fitur"].split(",") == fitur.NAMA_FITUR
    assert "SINTETIS" in meta["data_latih"]


def test_onnxruntime_memuat_dan_keluaran_probabilitas(artefak):
    sesi = ort.InferenceSession(str(artefak), providers=["CPUExecutionProvider"])
    assert sesi.get_inputs()[0].shape[1] == len(fitur.NAMA_FITUR)
    x = fitur.matriks(dataset.baca()[:50])
    p = sesi.run(None, {"masukan": x})[1]
    assert p.shape == (50, 2)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5) and (p >= 0).all()


def test_gerbang_model_tidak_lebih_buruk_dari_baseline(artefak):
    hasil = evaluasi.hitung(artefak)
    lolos, gagal = evaluasi.gerbang_lolos(hasil)
    assert lolos, gagal
    assert hasil["model"]["roc_auc"] > hasil["baseline"]["roc_auc"]


def test_gerbang_memblokir_model_yang_lebih_buruk():
    hasil = {
        "model": {"roc_auc": 0.60, "pr_auc": 0.60},
        "baseline": {"roc_auc": 0.70, "pr_auc": 0.55},
    }
    lolos, gagal = evaluasi.gerbang_lolos(hasil)
    assert not lolos and len(gagal) == 1 and "roc_auc" in gagal[0]


def test_gerbang_cli_keluar_1_bila_buruk(artefak, tmp_path, monkeypatch):
    buruk = {"n_uji": 1, "prevalensi": 0.5, "model": {"roc_auc": .5, "pr_auc": .5, "presisi": 0, "recall": 0, "f1": 0, "brier": 1},
             "baseline": {"roc_auc": .7, "pr_auc": .7, "presisi": 0, "recall": 0, "f1": 0},
             "segera": {"n": 1, "dipakai_bukan_dibuang": .5, "auc_model": .5, "auc_baseline": .5}}
    monkeypatch.setattr(evaluasi, "hitung", lambda _j: buruk)
    monkeypatch.setattr("sys.argv", ["x", "--model", str(artefak), "--keluaran", str(tmp_path / "m.md"), "--gerbang"])
    assert evaluasi.main() == 1
    assert "GAGAL" in (tmp_path / "m.md").read_text()


def test_metrics_md_dihasilkan_dan_menyebut_versi_serta_sintetis(artefak, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["x", "--model", str(artefak), "--keluaran", str(tmp_path / "m.md"), "--gerbang"])
    assert evaluasi.main() == 0
    teks = (tmp_path / "m.md").read_text()
    assert versi.MODEL_VERSI in teks and "SINTETIS" in teks and "LOLOS" in teks


def test_metrics_md_di_repo_ada_dan_bertanda_sintetis():
    teks = evaluasi.RAIB.read_text()
    assert versi.MODEL_VERSI in teks and "SINTETIS" in teks


def test_split_tidak_bocor_dan_deterministik():
    baris = dataset.baca()
    a_latih, a_uji = dataset.pisah(baris)
    b_latih, b_uji = dataset.pisah(baris)
    assert [r["id"] for r in a_uji] == [r["id"] for r in b_uji]
    assert not {r["id"] for r in a_latih} & {r["id"] for r in a_uji}
    assert len(a_latih) + len(a_uji) == len(baris)


def test_inferensi_berurutan_risiko_naik_saat_stok_besar_dan_waktu_singkat(artefak):
    aman = inferensi.FiturItem(sisa_hari=2, jumlah_aktif=100, laju_pakai_harian=100)
    bahaya = inferensi.FiturItem(sisa_hari=2, jumlah_aktif=2000, laju_pakai_harian=10)
    assert inferensi.skor_risiko(bahaya, artefak) > inferensi.skor_risiko(aman, artefak)


def test_inferensi_tanpa_artefak_jatuh_ke_baseline(tmp_path):
    s = inferensi.skor_risiko(inferensi.FiturItem(sisa_hari=2, jumlah_aktif=10, laju_pakai_harian=1), tmp_path / "tidak-ada.onnx")
    assert s == 0.85  # SEGERA
