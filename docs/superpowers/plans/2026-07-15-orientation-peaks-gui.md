# Plan — Orientation → Peaks (py4DSTEM GUI)

**Rama / worktree:** `peak-indexer-notebook`  
**Estado:** IMPLEMENTADO (O0–O4)  
**Fecha:** 2026-07-15  

## Objetivo

Ventana aparte (como Index BVM) que usa py4DSTEM `Crystal` para definir picos desde orientación — Path A (known generate) y Path B (ACOM match) — sin tocar el pipeline de strain, para comparar con nuestro Index BVM.

## Implementado

| Pieza | Archivo |
|-------|---------|
| Motor | `orientation_peaks.py` — match Å⁻¹↔px, Path A/B, compare, figure, CSV |
| Engine | `engine.run_orientation_peaks`, `apply_orientation_peaks_to_basis_params` |
| GUI | `qt_orientation.OrientationPeaksDialog` |
| Botón | Basis → **Orient. peaks…** (`qt_main.py`) |
| Tests | `tests/test_orientation_peaks.py` (matcher + Path A + Path B smoke) |

## Uso

1. Calibrar hasta BVM (origin / Q).
2. Basis → **Orient. peaks…**
3. Load CIF (o Si/Au), elegir Known o ACOM → **Run**.
4. Opcional **Compare vs Index BVM** (no escribe params).
5. Opcional **Send to Fast4D** → mismos `index_origin/g1/g2` + `manual_enabled`.

## Nota NumPy 2.x

`orientation_plan` en py4DSTEM 0.14.19 usa `.astype(np.integer)` (roto en NumPy 2). Se parchea en `orientation_peaks._patch_acom_numpy_integer` antes de Path B.
