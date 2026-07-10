# CHANGELOG — Memory Optimization & UI Follow-ups

**Fecha:** 2026-07-09
**Origen:** Investigación completa de arquitectura de memoria (`MEMORY_ARCHITECTURE_REPORT.md`), ejecutada como 3 planes vía subagent-driven-development, más varios pedidos de UI que surgieron en la misma sesión.
**Rama:** `claude/grafiphy-context-review-9a6ef4` (mergeada a `main` vía PR #4 + este commit final).
**Alcance:** memoria + UI/UX. **No** se modificó ningún algoritmo científico ni resultado numérico.

---

## 1. Memoria

| Cambio | Archivo(s) | Detalle |
|---|---|---|
| Liberación automática al cambiar de scan | `engine.py`, `qt_main.py` | `release_scans()` + `ResidentDataPolicy` (antes manual, solo con el botón "Free RAM"). Configurable en `Settings → Resident data (RAM)…` (default: 2 scans en RAM). |
| Liberación automática en batch | `driver.py` | `driver.compute_all` llama `engine.free_memory()` después de cada scan. |
| `gc.collect()` post-guardado | `pipeline.py` | Justo después del único punto donde datacube + braggpeaks recién detectados están ambos en RAM a la vez. |
| Memmap más agresivo | `pipeline.py` | Umbral bajado de 6 GiB → 2 GiB; nueva variable `FAST4D_FORCE_MEMMAP=1`. |
| Código muerto eliminado | `batch_common.py` | ~370 líneas de `BatchScanResult`/`BatchScanItem` sin ningún llamador en todo el repo. |
| Resolución perezosa de figuras | `qt_widgets.py`, `qt_params.py` | `ClickableFigureLabel` ya no cachea una referencia permanente — sigue a `FigurePolicy`. |
| Detección de Bragg en streaming (opt-in) | `bragg_stream.py` (nuevo) | Escribe picos a HDF5 posición-por-posición en vez de construir todo en RAM. **No conectado a ningún flujo todavía.** Validado con datos reales: 256 patrones de `256x256_Demo.mib`, 8,301 picos, 0 discrepancias vs. la llamada directa. |
| Infraestructura de tests | `tests/`, `requirements-dev.txt` | No existía antes de esta sesión. 26 tests, todos pasando. |

Detalle tarea-por-tarea en `docs/superpowers/plans/*.md`.

## 2. UI / UX

| Cambio | Detalle |
|---|---|
| Textboxes más anchos | `_LabeledSlider` (Origin/Ellipse/etc.) y los multiplicadores ×in/×out/×BF del diálogo "Create ADF/BF/DP" — antes recortaban a 1 dígito visible. |
| Sync Files panel ↔ tabla de parámetros | Clickear el **header** de una columna (nombre de archivo) en la tabla del medio ahora selecciona ese archivo en el panel Files — antes Play Calibration/Analysis podía operar silenciosamente sobre el archivo equivocado. |
| Auto-load antes del tuner de 6 puntos | "Tune detection (live)" carga datacube+probe automáticamente si falta, en vez de exigir Load manual + Update. |
| Alcance de estadística compartida (opt-in) | Nuevo checkbox en Análisis: las vistas "…across files" del Report (Lines/ROIs agrupadas, distribución/box/PCA/stress/stats cruzados) ya NO mezclan automáticamente archivos no relacionados que comparten un id de línea/ROI como "L1". Apagado por default; encendido solo para experimentos de reproducibilidad reales. |
| Auto-save configurable | El checkbox "Save" (antes siempre visible junto a Compute) se movió a `Settings → Auto-save on Compute`. Apagarlo evita la escritura de figuras/workspace en cada Calculate (File)/Calculate (All), ahorrando tiempo de análisis; el comportamiento por default (encendido) es idéntico al de antes. |
| Virtual ADF/BF | Selector de modo DP (Mean/Max/Both) + barra de progreso en vivo; el botón se promovió al toolbar principal del paso Probe (venía de otra rama, integrado sin conflictos). |
| Comentario aclaratorio | `pipeline.py` — se documentó que el orden `data=(bragg_rys, bragg_rxs)` en la detección de 6 puntos es intencional (confirmado por el dueño del proyecto), tras una revisión automatizada que lo marcó erróneamente como bug. |

## 3. Hallazgo separado (NO corregido en esta rama)

Durante la validación de `bragg_stream.py` se encontró y confirmó (reproducido dos veces, en `pipeline.py`, función `detect_selected_bragg_disks_step`, línea ~1382) un bug preexistente y **no relacionado**: el orden de argumentos `data=(state.bragg_rys, state.bragg_rxs)` está invertido respecto a la convención real de py4DSTEM — causa `IndexError` en scans no cuadrados y resultados silenciosamente incorrectos en cuadrados con x≠y. Se dejó **sin tocar** (fuera de alcance, afecta corrección científica) y se registró como tarea aparte para el dueño del proyecto.

## 4. Estado final

- 20 commits en `claude/grafiphy-context-review-9a6ef4`, revisados individualmente durante la ejecución (varias rondas de fix por hallazgos reales de revisión, documentadas en `docs/superpowers/plans/*.md`).
- 26/26 tests pasando; compila limpio en todos los archivos tocados.
- Confirmado en vivo por el dueño del proyecto: sync Files/tabla, textboxes, tuner auto-load, Virtual ADF/BF, y el toggle de Settings — todos funcionando.
- **Pendiente:** validación de `bragg_stream.py` a escala completa (512×512) con datos reales; smoke-test manual del toggle "Repro. exp." con dos archivos reales.

**Rollback:** revertir el merge commit de `main` (`git revert -m 1 <merge-sha>`), o volver al commit previo al PR #4 si es necesario.
