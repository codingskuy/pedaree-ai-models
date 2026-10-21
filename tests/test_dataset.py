from eval import dataset


def test_semua_baris_ditandai_sintetis():
    baris = dataset.baca()
    assert len(baris) == dataset.JUMLAH_BARIS
    assert {r["sumber"] for r in baris} == {"SINTETIS"}


def test_berkas_cocok_dengan_generator_deterministik(tmp_path):
    baru = dataset.tulis(tmp_path / "x.csv")
    assert baru.read_text() == dataset.BERKAS.read_text()


def test_dataset_tidak_memuat_data_pengguna_dan_valid():
    baris = dataset.baca()
    assert all(r["recipe_id"] == "" or r["recipe_id"].startswith("rcp_sint_") for r in baris)
    assert all(r["sisa_hari"] >= 0 and r["jumlah_aktif"] > 0 and 0 <= r["bobot_resep"] <= 1 for r in baris)
    assert all("garam" not in r["nama"].lower() and "gula" not in r["nama"].lower() for r in baris)


def test_kelas_seimbang_cukup():
    rasio = sum(r["terbuang"] for r in dataset.baca()) / dataset.JUMLAH_BARIS
    assert 0.15 < rasio < 0.6
