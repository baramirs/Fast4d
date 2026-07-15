# Plan A — Report browser: árbol por scan + materialización on-demand

**Rama / worktree:** `peak-indexer-notebook`  
**Estado:** A1–A3 implementados en código (árbol + Send → `report_*` + sin auto maps-with-lines). A4 migración legacy vía rama «Legacy» (opt-in). Plan B aún no empezado.  
**Paralelo con:** [Plan B — Export DOCX/PDF / layouts](./2026-07-14-report-export-docx-pdf.md)  
**Fecha:** 2026-07-14

---

## Por qué este plan

El Report actual es un conjunto de tabs + combos. Obliga a Refresh/Show (ya mitigado con auto-show), no escala con muchos scans, y mezcla **productos core** (calibración, strain/stress) con **derivados** (line profiles, maps with lines/ROIs) que hoy se materializan “por si acaso”.

Queremos un browser tipo árbol, lazy, y que lines/ROI solo existan cuando el usuario las manda (Live → Send).

## Frontera con Plan B (importante)

| Este plan (A) | Plan B |
|---|---|
| Cómo **navegar** y **qué figuras existen** en sesión | Cómo **exportar** un paquete bonito (DOCX/PDF/PPTX) |
| No auto-generar lines/ROI/maps-with-* | No partir composites con PIL a ciegas |
| Send to Report → rama `Reports` | Layout-aware pages, sin títulos cortados |

Pueden implementarse **en paralelo**: A toca `qt_report.py` + contratos de materialización; B toca `tools/export_*` + generación por-canal. Evitar que ambos reescriban `FIGURE_ORDER` / `save_figures` el mismo día sin coordinar.

## Objetivos

- Árbol por scan con filtros (qué ramas/tipos ver).
- Show perezoso al click (sin precargar todo en RAM).
- Sin auto-materializar `line_profiles`, `maps_with_lines`, `roi_*`, etc.
- Live Line / Live ROI → **Send to Report** → hoja bajo `Reports`.
- Cross-scan = acciones de sesión, no hojas repetidas por scan.

## No-objetivos

- Export DOCX/PDF/PPTX (Plan B).
- Cambiar algoritmos de strain/líneas.
- Unificar diálogos Live en uno solo.

## Estructura de árbol (propuesta)

```
Scan01
  Calibrations
    Probe / Detection / Origin / Ellipse / Q-pixel / Basis / …
  Maps
    Theoretical reference (without ROI)
      εxx / εyy / εxy / θ / σ… / ADF
    Experimental reference (with ROI)
      …
  Reports                    ← solo lo enviado por el usuario
    Live line L1 on εyy (2026-07-14 …)
    Live ROI set A …
Scan02
  …
Session                      ← no cuelga de un scan
  [acciones] Strain distribution… / Box-Violin… / PCA… / …
```

Filtro UI (checklist o chips): Calibrations / Maps / Reports / Solo disponibles / Texto.

## Fases

### A1 — Modelo + UI del árbol
- Sustituir (o complementar) combos por `QTreeWidget` / modelo.
- Nodos hoja = keys en `scan.figures` + entradas Reports.
- Click → auto-show (reutilizar `_show` lazy actual).

### A2 — Contrato de materialización
- Auditar y cortar builds automáticos de profiles/maps-with-* en load / Analysis / refresh.
- `∑ Analysis` = stress (+ lo mínimo); **no** registrar todas las line figures.

### A3 — Send to Report
- Live Line/ROI registran bajo namespace `reports/…` (o keys prefijadas).
- Tree salta al nodo nuevo.

### A4 — Migración workspaces viejos
- Si ya hay PNGs legacy de lines/ROI → mostrarlos (rama Reports o “Legacy”), no regenerar ausentes.
- Tests: open workspace no dispara builds masivos.

### A5 (opcional) — Persistencia metadatos Reports
- Manifest de entradas user-sent para sobrevivir reload.

## Archivos probables

`qt_report.py`, `qt_main.py` (Live Send), `engine.py` (`register_figure`, `build_*`), `analysis.py` (acciones sesión), tests.

## Criterios de éxito

- Abrir Report con N scans no construye line/ROI/maps-with-*.
- Localizar un mapa ≤ 2 interacciones típicas (filtro + click).
- Send desde Live aparece bajo **Reports**.
- Workspace legacy abre sin pico de RAM por auto-materialización.

## Dependencias / riesgos

- Coordinar con Plan B qué keys se consideran “exportables por defecto”.
- PPTX actual asume ciertas PNGs en `figures/` — no borrar strain/stress composites hasta que B tenga alternativa.
