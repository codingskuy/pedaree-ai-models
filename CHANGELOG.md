# Changelog — pedaree-ai-models

Format mengikuti Keep a Changelog; versi mengikuti TownHall.

## [Belum dirilis]

### Ditambahkan
- Kontrak inferensi 1.0 untuk API Rust: model pydantic `src/kontrak.py`, JSON Schema + contoh + vektor emas fitur di `docs/kontrak/`, CLI `python -m src.inferensi --contoh|--masukan|--skema`, ADR-0003, PDR-AI-004.
- Model risiko-kedaluwarsa 1.0.0 (GradientBoosting -> ONNX), model card, `eval/metrics.md`, evaluasi vs baseline dan gerbang rilis, PDR-AI-003.
- Baseline deterministik ambang SOP kedaluwarsa + urutan/alokasi FEFO (`src/baseline.py`) dan dataset evaluasi SINTETIS (`eval/`), PDR-AI-002.
- Fondasi AI-dev v1.0.0: `AGENTS.md`, `CLAUDE.md`, ADR-0001, `docs/ARSITEKTUR-AI-DEV.md` (2026-10-14).

### Diperbaiki
- Pin `skl2onnx 1.19.1` + `protobuf 5.29.5`: kombinasi lama tidak dapat dipasang/ekspor ONNX gagal (ADR-0002).
