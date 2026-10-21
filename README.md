# pedaree-ai-models — Model Ringan Kedaluwarsa

Divisi **Pedaree (Smart Pantry)**, org **Coding-Skuy**. Opsi A.

TownHall: https://github.com/Coding-Skuy/Pedaree-TownHall

## Ringkasan

Model ringan untuk memprediksi risiko kedaluwarsa bahan dan menyarankan urutan
pemakaian. Dirancang agar dapat berjalan di perangkat (format ONNX) maupun di
sisi backend.

## Modul pantry-resep

Kontrak lintas divisi (detail: `docs/MODUL-PANTRY-RESEP.md`):

- **Pemilik modul:** Pawonee.
- **Konsumen modul:** Pedaree (model memakai `recipe_id` Pawonee sebagai fitur
  konteks: bahan yang dipakai banyak resep mendapat prioritas pemakaian).
- Model tidak melatih ulang data resep Pawonee; hanya memakai rujukan resep
  yang sudah dipublikasikan.

## Teknologi (versi dikunci)

- Python 3.12.8
- scikit-learn 1.6.1
- skl2onnx 1.19.1, protobuf 5.29.5 (lihat ADR-0002)
- onnxruntime 1.21.0
- numpy 2.0.2
- pydantic 2.10.6

Lihat `requirements.txt` sebagai sumber kebenaran versi.

## Struktur

```text
src/               kode latih dan inferensi
models/            artefak .onnx hasil `src.latih` (tidak di-commit)
eval/              dataset sintetis, evaluasi, dan metrics.md
tests/             uji perilaku model
docs/              arsitektur dan kontrak modul
```

## Cara jalan

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m eval.dataset            # (opsional) regenerasi dataset sintetis
python -m src.latih               # -> models/kedaluwarsa-v1.onnx
python -m eval.evaluasi --gerbang # -> eval/metrics.md; kode 1 bila lebih buruk dari baseline
```

## CI

Workflow `.github/workflows/ci.yml` menjalankan uji Python dan validasi ekspor ONNX.
