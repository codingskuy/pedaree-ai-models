"""Latih model risiko kedaluwarsa (risiko terbuang) dan ekspor ke ONNX.

Jalankan: `python -m src.latih [--keluaran models]`.
Data: dataset SINTETIS eval/data/sintetis-v1.csv (lihat eval/README.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from sklearn.ensemble import GradientBoostingClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from eval import dataset
from src import fitur, versi

SEED_MODEL = 0
TOLERANSI_PARITAS = 1e-4
NAMA_MASUKAN = "masukan"


def buat_model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=60, max_depth=3, learning_rate=0.1, random_state=SEED_MODEL)


def ekspor_onnx(model: GradientBoostingClassifier) -> onnx.ModelProto:
    onx = convert_sklearn(
        model,
        initial_types=[(NAMA_MASUKAN, FloatTensorType([None, len(fitur.NAMA_FITUR)]))],
        options={id(model): {"zipmap": False}},
        target_opset=17,
    )
    onx.doc_string = f"{versi.MODEL_NAMA} {versi.MODEL_VERSI}: P(batch terbuang), fitur sintetis-v1"
    for k, v in {
        "model_nama": versi.MODEL_NAMA,
        "model_versi": versi.MODEL_VERSI,
        "fitur": ",".join(fitur.NAMA_FITUR),
        "data_latih": "SINTETIS eval/data/sintetis-v1.csv",
    }.items():
        entri = onx.metadata_props.add()
        entri.key, entri.value = k, v
    return onx


def validasi_onnx(onx: onnx.ModelProto, model: GradientBoostingClassifier, x: np.ndarray) -> float:
    """Pastikan ONNX sah dan keluarannya sama dengan sklearn; kembalikan selisih maksimum."""
    onnx.checker.check_model(onx)
    sesi = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    p_onnx = sesi.run(None, {NAMA_MASUKAN: x})[1][:, 1]
    selisih = float(np.max(np.abs(p_onnx - model.predict_proba(x)[:, 1])))
    if selisih > TOLERANSI_PARITAS:
        raise RuntimeError(f"ONNX tidak setara sklearn: selisih maks {selisih:.2e}")
    return selisih


def latih(keluaran: Path) -> Path:
    latih_baris, _ = dataset.pisah(dataset.baca())
    x = fitur.matriks(latih_baris)
    y = np.array([r["terbuang"] for r in latih_baris])
    model = buat_model().fit(x, y)
    onx = ekspor_onnx(model)
    selisih = validasi_onnx(onx, model, x[:500])
    keluaran.mkdir(parents=True, exist_ok=True)
    jalur = keluaran / versi.BERKAS_ONNX
    jalur.write_bytes(onx.SerializeToString())
    print(f"model {versi.MODEL_NAMA} {versi.MODEL_VERSI} tersimpan di {jalur} (selisih ONNX-sklearn maks {selisih:.1e})")
    return jalur


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keluaran", type=Path, default=Path("models"))
    latih(ap.parse_args().keluaran)
