# Plan B — Report export: DOCX / PDF (y PPTX sano) sin cortar mapas

**Rama / worktree:** `peak-indexer-notebook`  
**Estado:** **COMPLETO (B0–B4).** `tools/report_export/` genera paneles por canal + calib/`report_*` enteros; writers PDF/DOCX/PPTX; GUI `ExportReportDialog` con categorías; `split_map_figures` deprecated. Tests en `tests/test_report_export.py`.  
**Paralelo con:** [Plan A — Report browser árbol](./2026-07-14-report-browser-tree.md)  
**Fecha:** 2026-07-14

---

## El problema (evidencia actual)

Hoy el “reporte exportable” es básicamente:

1. **Save** escribe PNGs compuestos (`strain_*.png`, `stress_*.png`, calibraciones en `summary/`).
2. **Export PPTX** (`tools/export_calibration_pptx.py`) **reparte esos PNG a ciegas con PIL**:
   - Strain: asume **siempre** layout horizontal (1 fila, ~5 paneles); recorta tiras **verticales** al 80% del ancho para εxx/εyy/εxy/θ.
   - Basis: 3 tiras de ancho igual.
   - Stress: 3 tiras (encaja con el 1×3 actual).

Si el usuario elige `strain_layout = vertical` (o `square`) en la tabla Strain:

- py4DSTEM genera un composite **apilado / en grid**.
- El exportador sigue cortando por **X a altura completa**.
- Resultado: “mapas a la mitad”, títulos partidos, colorbars atravesados, slides feos.

Además:

- `bbox_inches="tight"` + `suptitle` pueden comerse títulos al guardar.
- Las celdas del PPTX están pensadas landscape; un PNG vertical se encoge y se ve “cortado”.
- **No hay** export DOCX ni PDF todavía (solo PPTX + CSV/XLSX de tablas).

Eso es exactamente el enfoque “burdo”: figuras prefabricadas + geometría fija + tijeras.

## Objetivo del plan

Un export de informe **elegible por el usuario** (`DOCX` | `PDF` | `PPTX`) que:

1. **No dependa** de partir composites con coordenadas mágicas.
2. Respete el layout elegido (horizontal / vertical / square) **sin romper paneles**.
3. Conserve títulos, colorbars y tipografía legible.
4. Sea bonito lo bastante para lab notes / tesis / share con colaboradores.
5. Pueda crecer hacia “solo lo que el árbol marca” (Plan A) sin reescribir el motor otra vez.

## Principio de diseño (la mejora sustancial)

> **Exportar canales, no tijeras sobre un collage.**

En Save (o en un paso “Prepare export assets”):

- Guardar **por canal** (y metadatos): `exx.png`, `eyy.png`, … + JSON `{layout, vrange, cmap, scan, label}`.
- Opcional: seguir guardando el composite para la GUI, pero el **export nunca lo corta**.

Al armar DOCX/PDF/PPTX:

- El builder **compone páginas** con esos paneles (grid tipográfico), no adivina bounding boxes de un PNG viejo.
- Layout del documento ≠ layout del collage py4DSTEM: el usuario elige plantilla de informe (1 canal/página, 2×2, fila, etc.) **independiente** de `strain_layout` de visualización.

## Frontera con Plan A

| Plan A (browser) | Este plan (B) |
|---|---|
| Qué se ve en sesión / RAM | Qué se empaqueta al disco/compartir |
| No materializar lines/ROI hasta Send | Incluir en el doc solo core + Reports enviados |
| Tree + filtros | Plantillas DOCX/PDF/PPTX |

**Paralelo OK** si B no asume que lines/ROI siempre existen, y A no borra `strain_*/stress_*` hasta que B escriba assets por-canal.

## Objetivos

- Selector de formato: **DOCX / PDF / PPTX** (mismo diálogo de selección de contenido).
- Pipeline de assets **layout-aware** (o layout-agnostic por-canal).
- Páginas con márgenes, títulos de sección, caption por scan, sin clip de `suptitle`.
- Deprecar (o aislar) `split_map_figures` / `split_basis_images` como path legacy “horizontal-only”.
- Calidad visual: tipografía consistente, colorbars enteros, sin medias mapas.

## No-objetivos (v1)

- Editor WYSIWYG de informe dentro de Fast4D.
- Branding corporativo complejo / plantillas Word del usuario (puede ser v2: “use my .docx template”).
- Re-calcular ciencia en el export (solo ensambla lo ya computado).
- Sustituir el preview del Report tab (eso es Plan A).

## Diagnóstico → requisitos

| Fallo actual | Requisito |
|---|---|
| PIL X-strips con layout vertical | Prohibido cortar composites; usar paneles por canal |
| `0.80` ancho asume legend horizontal | No hardcodear geometría de py4DSTEM show_strain |
| PPT cells landscape + PNG alto | Plantillas portrait/landscape; 1 mapa por página o grid tipográfico |
| `bbox_inches=tight` come títulos | Save export assets con `pad_inches` / sin tight agresivo / título en caption del doc |
| Solo PPTX | Capa común → writers DOCX + PDF + PPTX |
| Basis split 3 tiras fijas | Exportar paneles basis por axes, o figura monopanel |

## Arquitectura propuesta

```
                  ┌─────────────────────────────┐
  scan arrays /   │  export_assets.prepare(...) │  escribe:
  figures vivos   │  (por canal + manifest JSON) │   export_assets/<scan>/strain_without_roi/exx.png
                  └──────────────┬──────────────┘   …/manifest.json  {layout, ranges, …}
                                 │
                  ┌──────────────▼──────────────┐
                  │  report_builder.build(...)  │  TOC + secciones
                  │  formato = docx|pdf|pptx    │
                  └──────────────┬──────────────┘
                     ┌───────────┼───────────┐
                     ▼           ▼           ▼
                 writer_docx  writer_pdf  writer_pptx
```

**Ubicación sugerida:** `tools/report_export/` (nuevo)  
Mantener `tools/export_calibration_pptx.py` como wrapper legacy o migrar y dejar shim.

### Stack tentativo (decidible en kickoff)

| Formato | Opción A (recomendada) | Opción B |
|---|---|---|
| DOCX | `python-docx` + imágenes PNG | — |
| PDF | matplotlib/`PdfPages` **o** reportlab | weasyprint desde HTML |
| PPTX | `python-pptx` (ya hay) pero alimentado por paneles | — |

Preferencia: **PNG vector-ish no**; raster a 150–300 dpi con colorbars incluidos en cada panel individual. Para PDF “bonito”, `PdfPages` con figuras matplotlib recién dibujadas desde arrays es más limpio que re-embeber PNG cortados.

## Fases

### B0 — Spike (1–2 días)
- Reproducir el bug vertical → PPTX (screenshot + PNG partido).
- Prototipo: 1 scan, 4 paneles ε desde arrays → 1 página PDF + 1 DOCX sin PIL crop.
- Decidir stack DOCX/PDF.

### B1 — Export assets por canal
- Al Save (flag o siempre en modo report): escribir `export_assets/…/{channel}.png` + `manifest.json` con `strain_layout`, vranges, cmap, paths.
- Basis/calibración: o paneles separados, o una imagen **completa sin split** por slide/sección.
- Tests: horizontal / vertical / square producen N archivos de canal válidos (no tiras).

### B2 — Builder común + PPTX nuevo
- `build_report(selection, format=…)` lee manifest + assets.
- Reemplazar uso de `split_map_figures` en el path feliz.
- Diálogo: formato + mismas categorías actuales (`calib_trends`, `strain_maps`, `basis_panels`) + futuro “Reports enviados” (Plan A).

### B3 — DOCX + PDF
- Writers con plantilla mínima: portada (fecha, scans), calibración, mapas (1 canal/página o 2×2), trends.
- Opción usuario: **página por canal** (más limpio) vs **grid compacto**.

### B4 — Integración GUI + deprecación
- Report: `Export…` abre formato (DOCX/PDF/PPTX) en vez de solo PPTX.
- Warning si faltan assets (pedir Save primero) — igual que hoy con `summary/calibrations`.
- Marcar `split_map_figures` como legacy; quitar del path default.
- Documentar en `ForGITHUB-Updates.md` / README corto.

### B5 (opcional) — Alineación Plan A
- Export solo nodos marcados en el tree / Reports enviados.
- No incluir lines/ROI salvo Send.

## Archivos probables

| Área | Archivos |
|---|---|
| Export legacy | `tools/export_calibration_pptx.py` (`split_map_figures` L193–236) |
| Nuevo | `tools/report_export/` (`prepare_assets`, `build`, `writers`) |
| GUI | `qt_main.py` (`_export_pptx_report`), `qt_widgets.py` (`ExportSelectionDialog`), `qt_report.py` |
| Strain layout | `engine.py` (`strain_layout`), `pipeline.py` (`update_strain_params`, `_apply_show_orientation_to_figures`) |
| Save | `engine.save_figures`, `save_summary`, `fast_artifacts` |
| Tests | `tests/test_report_export_*.py` (layouts × formatos) |

## Criterios de éxito

1. Con `strain_layout=vertical` (y `square`), export DOCX/PDF/PPTX muestra **mapas completos** por canal; cero tiras diagonales/medias.
2. Títulos de sección y captions legibles; colorbars no partidos.
3. Usuario elige formato en el diálogo.
4. Tests automatizados fallan si alguien reintroduce crop por fracciones de ancho fijas sobre el composite.
5. Workspaces viejos: si solo hay composite horizontal, path legacy o regeneración de assets al Save.

## Riesgos

- **Doble I/O:** assets por canal + composites GUI → más disco; mitigar con “prepare on export” (generar assets al exportar, no en cada Save) si el Save se vuelve lento.
- **py4DSTEM figure ownership:** mejor redibujar desde arrays (`strain_raw`) que parsear axes del composite.
- **Dependencias nuevas** (`python-docx`, quizás `reportlab`) → `requirements` / conda env `py4dstem-01419`.
- **Scope creep** de plantillas Word corporativas — dejarlo explícitamente en v2.

## Orden recomendado si ambos planes corren en paralelo

```
Semana 1:  A1 (tree UI stub)  ║  B0 spike + B1 assets por canal
Semana 2:  A2–A3 (lazy + Send) ║  B2 PPTX nuevo + B3 DOCX/PDF
Semana 3:  A4 migración       ║  B4 GUI unificada + tests
           (opcional A5)       ║  (opcional B5 ↔ tree selection)
```

Punto de sincronización único: **manifest de qué figuras son “core” vs “Reports”** (un JSON schema corto compartido).

## Kickoff checklist

- [x] Confirmar formatos v1: DOCX + PDF + PPTX.
- [x] Assets se generan **on Export** (`prepare_export_assets`), no en cada Save.
- [x] Estilo v1: 1 canal / figura completa por página (calidad).
- [x] No tocar `split_map_figures` para “arreglar vertical” con más ifs — deprecado; path nuevo en `tools.report_export`.

## Checklist de cierre (2026-07-14)

- [x] B1 assets por canal + calib + reports
- [x] B2/B3 writers PDF / DOCX / PPTX + portada
- [x] B4 GUI `ExportReportDialog` + flags include_*
- [x] Deprecation warning en `split_map_figures`
- [x] `python-docx` en requirements + tests DOCX/flags
