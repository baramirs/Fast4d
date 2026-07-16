# ForGITHUB-Updates — borrador humano (antes de pulir con AI)

**Propósito:** este archivo es un borrador **detallado y en lenguaje humano** de todo lo que se hizo en la rama `peak-indexer-notebook` (Index BVM + fixes de UX + Report). Sirve como materia prima: luego puedes pasarlo por AI para acortar, ordenar por PR, y subirlo a GitHub (Release notes / PR body / CHANGELOG).

**Rama:** `peak-indexer-notebook`  
**Worktree:** `.claude/worktrees/peak-indexer-notebook`  
**Fecha de este borrador:** 2026-07-14  
**Estado:** cambios listos para commit local; **aún sin push**.

---

## Cómo usar este archivo

1. Léelo completo (está escrito para que un humano entienda el “por qué”).
2. Pídele a una AI: “convierte esto en un PR description limpio / release notes en inglés”.
3. No copies ciegamente rutas internas de debug (`debug-683296.log`, planes locales) al README público.

---

## 1. Index BVM — indexación del Bragg Vector Map desde la GUI

### Qué problema resolvía
Antes, sacar `g1`/`g2` (vectores de red en píxeles) para el paso **Basis** era manual o vía notebook. Queríamos el mismo flujo del notebook de demo (`indexing_bvm_demo`) dentro de Fast4D, **antes** de abrir el tuner de Basis, con un botón claro y un “Send to Fast4D” que deje los parámetros listos en la tabla.

### Qué se añadió

| Pieza | Archivo(s) | Qué hace |
|---|---|---|
| Motor de indexación | `bvm_indexing.py` (nuevo) | Maxima 2D del BVM, RANSAC para proponer lattice, asignación hkl con anclaje a zone axis + ejes reales H/V, resultado tipado (`IndexingResult`). Incluye manejo de **QR flip** (convención de componentes Q) al estilo py4DSTEM, sin crashear. |
| Pegamento engine | `engine.py`, `param_spec.py` | `engine.index_bvm(scan)`, `apply_indexing_to_basis_params`, campos nuevos en `CalibrationParams`: `zone_axis`, `real_axis_h`, `real_axis_v`, `indexing_tol_px`, `indexing_seed`. Figura clave `indexing` disponible para Report/Basis. `Scan.indexing_result` guarda el último run. |
| Diálogo GUI | `qt_indexer.py` (nuevo) | `IndexerDialog` modeless: crystal info, zone/real axes, tolerancia, seed, **Run indexing**, tabla de picos, overlay, **Set as g1/g2**, **Send to Fast4D**, **Export CSV/PNG**. |
| Botón en toolbar | `qt_main.py` | En el paso **Basis** → botón **Index BVM…** (junto a Setting / Apply / Reset). Abre el diálogo sin bloquear la ventana principal. |
| Tests | `tests/test_bvm_indexing.py` | Unit/regresión del motor (sintético + checks). |
| Validación E2E | `tools/validate_bvm_indexer_e2e.py` | Contra el Demo Path A: índices esperados, `g1`/`g2` vs manifest (Δ≈0). |
| Notebook de referencia | `notebooks/indexing_bvm_demo.ipynb`, `notebooks/run_indexing_nb.py` | Referencia científica / reproducibilidad del algoritmo fuera de la GUI. |
| `.gitignore` | `notebooks/output/` | No versionar salidas temporales del notebook. |

### Flujo de usuario (Index BVM)
1. Path A con braggpeaks cargado; Origin / Ellipse / Q-pixel en buen estado.
2. Paso **Basis** → **Index BVM…**.
3. Revisar crystal + zone axis `[uvw]` + ejes reales H (+ry) / V (+rx).
4. **Run indexing** → overlay + tabla.
5. (Opcional) elegir filas y **Set as g1 / g2**.
6. **Send to Fast4D** → escribe `index_origin` / `g1` / `g2` y `basis_manual_enabled=True`.
7. Abrir Basis normalmente: los vectores ya están en la tabla de parámetros.

### Ayuda en la app
- Menú **Help → Index BVM guide…** (HTML en `qt_main.py`).
- Quick Start menciona Index BVM en el paso Basis y en “Finding help”.

---

## 2. Figuras modeless (varias abiertas a la vez)

### Problema
Al maximizar o hacer click en un thumbnail de figura, el diálogo usaba `.exec()` (modal): bloqueaba la app y solo podías ver **una** figura a la vez. Incómodo para comparar mapas.

### Fix
- `qt_widgets.py`: `FigureDialog` / `from_png` → NonModal + flags de ventana; click en `ClickableFigureLabel` usa `.show()` y guarda refs en `host._figure_windows`.
- `qt_report.py`: Maximize del Report igual (modeless).

### Resultado
Puedes tener Origin, Strain, Basis, etc. abiertos en paralelo, moverlos, minimizar, sin congelar Fast4D.

---

## 3. Escala de orientación (theta) en mapas de strain

### Problema
El usuario ponía el rango de orientación en la GUI (p.ej. −2°…2°) pero el colorbar / clim terminaba mostrando algo tipo ±5.2°. Había dos causas:

1. Un “auto-widen” en `_effective_orientation_clims` que **ignoraba** el vrange del usuario cuando los datos estaban en grados.
2. py4DSTEM a veces deja los tick labels del colorbar desincronizados respecto al clim real (y a veces los datos están en radianes internamente).

### Fix (`pipeline.py`, y sync en `engine.py` donde aplica)
- Se eliminó el auto-widen: si el usuario pide −2…2, eso se respeta (con conversión deg↔rad solo cuando los datos están en radianes).
- Al refrescar colorbars de theta se **fuerzan** etiquetas en grados desde el vrange de la GUI.
- `compute_strain` asegura `update_strain_params` desde `scan.params` antes de `get_strain`, para no plotear con params viejos.

### Resultado
Lo que escribes en el selector de rango de orientación es lo que ves en el mapa y en la barra de color.

---

## 4. Crash / error de QR flip en Index BVM

### Problema
Con ciertas orientaciones, `anchor_hkl_with_real_axes` fallaba cuando hacía falta un “QR flip” (intercambio de componentes Q). Mensaje/traceback apuntaba a un `raise` que ya no debía existir si el proceso estaba actualizado; a veces el usuario veía el error por proceso Fast4D viejo (`.pyc` / sin reiniciar).

### Fix (`bvm_indexing.py`)
Implementación del swap de componentes Q al estilo py4DSTEM cuando `qr_flip` es necesario, en lugar de abortar.

### Resultado
Index BVM completa el anclaje hkl también en esos casos de convención QR.

---

## 5. Menú: View → Settings → Help (+ guía Index BVM)

### Problema
El orden del menú era View → Help → Settings (Help en medio). El usuario pidió **View, Settings, Help**. Además faltaba documentación de Index BVM en Help.

### Fix (`qt_main.py`, `qt_quickstart.py`)
- Orden de menú: **View → Settings → Help**.
- Help: Quick Start, Calibration guide, **Index BVM guide…**.
- Quick Start actualizado (Basis + lista de ayudas).

---

## 6. Report: auto-mostrar al seleccionar (sin ritual Refresh → Show)

### Problema
Cada vez que cargabas data nueva o un workspace viejo, en el tab Report había que pulsar **Refresh** y luego **Show**. Igual al cambiar Item/Map en el selector: elegir + Show. Muy incómodo.

### Decisión de diseño sobre RAM (importante)
**No** precargamos “todos los mapas a la vez” en el panel. Eso duplicaría figuras en memoria sin beneficio.

Plan bueno (el que quedó):
1. Al cargar workspace, las figuras PNG ya viven en `scan.figures` (como antes).
2. El Report solo **renderiza la selección actual** (lazy).
3. Cambias View / Scan / Item / Map / pestaña → se muestra solo eso.
4. Tras `report_refresh()` (load / Compute / etc.) también auto-muestra la selección actual.

### Fix (`qt_report.py`)
- Señal `changed` en `_ViewSelector`; Report escucha y llama `_auto_show()`.
- `refresh()` (llamado por la app al cargar) → reconstruye combos **y** auto-show.
- Placeholder: “Select a view — it shows automatically.”
- Botón **Refresh quitado de la UI** (el método `refresh()` sigue existiendo para llamadas internas desde `report_refresh` / ParamTable). Ya no hace falta el botón: load/Compute ya refrescan, y el selector dispara el show.
- Botón **Show** se mantiene solo como “re-render manual” por si algo externo cambió la figura sin tocar el combo (tooltip lo explica).

### Textos
Mensajes que decían “Profiles → Set up Lines” ahora dicen **Analysis → Set up Lines & ROI** (el paso de la toolbar se llama Analysis).

---

## 7. Live Line / Live ROI en Analysis

### Contexto
En un commit anterior de esta línea (`e8fd98c`) esos botones ya se habían sacado del Report y puesto en el toolbar del paso cuyo label es **Analysis** (clave interna `lines`).

### En este update
- Se **reordenó** el toolbar de Analysis para poner primero **Live Line Profile** y **Live ROI profile**, luego separador, luego Set up Lines & ROI / Analyse (file) / Analysis (all). Así se nota que las herramientas interactivas “viven” en Analysis, no en Strain ni en Report.
- Docstrings/comentarios que aún decían “Report → Live…” o “Profiles step” se actualizaron a **Analysis**.

### Dónde encontrarlos
Icon strip → **Analysis** (a la derecha de Strain) → botones Live al inicio del toolbar contextual.

---

## 8. Archivos tocados (checklist para el PR)

### Nuevos
- `bvm_indexing.py`
- `qt_indexer.py`
- `tests/test_bvm_indexing.py`
- `tools/validate_bvm_indexer_e2e.py`
- `notebooks/indexing_bvm_demo.ipynb`
- `notebooks/run_indexing_nb.py`
- `ForGITHUB-Updates.md` (este archivo)

### Modificados
- `engine.py` — index_bvm, params, figures
- `param_spec.py` — zone/real axes, list3, etc.
- `pipeline.py` — orientation clims / colorbar theta (sin auto-widen)
- `qt_main.py` — Index BVM button, Help/menu order, Analysis Live order, guías HTML
- `qt_indexer.py` — (nuevo diálogo)
- `qt_params.py` — figura indexing en Basis, report wiring
- `qt_report.py` — auto-show, sin botón Refresh, textos Analysis
- `qt_widgets.py` — FigureDialog modeless
- `qt_quickstart.py` — Index BVM en la guía
- `.gitignore` — `notebooks/output/`

---

## 9. Cómo probar (manual + automatizado)

### Automatizado
```text
# env py4dstem-01419
pytest tests/test_bvm_indexing.py
# con PYTHONUTF8=1 en Windows si hace falta
python tools/validate_bvm_indexer_e2e.py
```

### Manual GUI
1. `run_gui.bat` desde el worktree / env correcto.
2. Load Demo Path A (braggpeaks).
3. Basis → Index BVM… → Run → Send → abrir Basis y ver g1/g2.
4. Abrir varias figuras (click thumbnail / Maximize) → deben coexistir modeless.
5. Strain: fijar vrange theta −2…2 → colorbar debe decir ±2°, no ~±5°.
6. Menú: View | Settings | Help; abrir Index BVM guide.
7. Report: cargar workspace → sin pulsar Refresh, la vista aparece; cambiar Item cambia el mapa solo.
8. Analysis → Live Line / Live ROI al frente del toolbar.

---

## 10. Notas que NO deben ir crudas a GitHub (o hay que reescribir)

- Rutas absolutas de Windows del autor (`C:\Users\…`).
- IDs de sesión de debug / logs NDJSON temporales.
- Planes internos de Cursor (`.cursor/plans/…`).
- “Worktree de Claude” — en el PR público hablar de la rama `peak-indexer-notebook`.

---

## 12. Report browser (árbol) — 2026-07-14 (sesión siguiente)

- Panel Report rediseñado: árbol por scan con filtros (Calibrations / Maps / Reports / Legacy / Session) + búsqueda.
- Maps agrupados en **Theoretical reference (without ROI)** / **Experimental reference (with ROI)** (mismo naming en Live tools, Store, ParamTable).
- Show perezoso al click; sin botón Refresh.
- Live Line/ROI → Send escribe solo keys `report_*` (no `maps_with_lines` / `line_profiles` genéricos).
- Session analysis como nodo de sesión (distribución, PCA, tablas, …).
- Tests: `tests/test_report_tree.py`.

## 12b. Menú Tools — Analysis fuera del toolbar

- Orden del menú: **View → Tools → Settings → Help**.
- Tools: Live Line Profile, Live ROI Profile, Set up Lines & ROI, Analyse (file), Analysis (all).
- Quitados del toolbar del paso Analysis y del botón ∑ Analysis de la barra inferior.

## 13. Report export PDF/DOCX/PPTX sin tijeras — 2026-07-14 (Plan B cerrado)

- Nuevo `tools/report_export/`:
  - `prepare_export_assets(...)` — mapas **por canal** desde arrays + calibraciones / `report_*` como figura completa; `manifest.json`.
  - Writers PDF (`PdfPages`), DOCX (`python-docx`), PPTX (`python-pptx`) con portada (fecha, scans, conteos).
- GUI: **Report → Export…** abre `ExportReportDialog` (formato PDF/DOCX/PPTX + checkboxes Maps / Calibrations / Reports).
- `split_map_figures` queda **deprecated** (warning); ya no es el path feliz de la GUI.
- El layout vertical/square de strain deja de romper el export porque **no se corta el collage**.
- Tests: `tests/test_report_export.py` (PDF/DOCX/PPTX + flags include_*).
- Dep: `python-docx` en `requirements.txt` (env `py4dstem-01419`).

---

## 14. Crystal from CIF — Index BVM + Q-pixel compartidos — 2026-07-14

### Qué problema resolvía
Index BVM y Q-pixel usaban solo Si / Au / Custom (editor manual). Queríamos cargar un **CIF** (formato estándar de cristalografía) como única fuente de verdad para lattice + structure factors — reutilizable también desde GPA / 4DSTEM.

### Qué se añadió

| Pieza | Archivo(s) | Qué hace |
|---|---|---|
| Loader | `engine.load_crystal_from_cif`, `is_approximately_cubic`, `CifCrystalInfo` | `Crystal.from_CIF` (py4DSTEM + pymatgen); extrae `a_lat` / positions; **warning** si la celda no es cúbica (Index BVM v1 sigue cúbico). |
| Params | `CalibrationParams.cif_path`, `cal_crystal="CIF"` | Persistido en session JSON vía `to_dict` / `_overlay_params_dict`. |
| Shared build | `_build_crystal`, `cal_crystal_obj` | Mismo CIF alimenta Index (`lattice_a`) y Q-pixel (`calibrate_pixel_size`). |
| UI Index | `qt_indexer.py` | Botón **Load CIF…**, label con a + path, aviso naranja si no cúbico. |
| UI Crystal editor | `qt_main.CrystalEditorDialog` | Mismo **Load CIF…** (parity con Q-pixel). |
| Params table | `param_spec.py` | Enum Crystal incluye `CIF`; fila readonly `cif_path`. |
| Fixture + tests | `tests/fixtures/Si.cif`, `tests/test_crystal_cif.py` | Sin red; 9 tests (cúbico, no-cúbico, persistencia, `_build_crystal`). |
| Dep | `requirements.txt` | `pymatgen` explícito. |

### Flujo
1. Index BVM… (o Crystal editor) → **Load CIF…** → elige `.cif`.
2. `cal_crystal=CIF`, `cif_path` guardado.
3. Run indexing / Q-pixel refit usan el mismo cristal.
4. Si el CIF no es cúbico: warning en UI; Index usa `a` convencional como métrica efectiva (v1).

### No-objetivos (v1)
Editor CIF WYSIWYG; indexación no-cúbica completa; descarga automática de CIFs.

---

## 15. Orient. peaks (py4DSTEM) — GUI aparte vs Index BVM — 2026-07-15

### Qué problema resolvía
Querer usar la orientación (CIF + zone / ACOM) de py4DSTEM para **definir picos**, sin mezclarlo con nuestro Index BVM (RANSAC) ni tocar el pipeline de strain — y poder comparar ambos métodos en la misma sesión.

### Qué se añadió

| Pieza | Archivo(s) | Qué hace |
|---|---|---|
| Motor | `orientation_peaks.py` | Path A `generate_diffraction_pattern` + match; Path B `orientation_plan` + `match_single_pattern`; compare; figure; CSV. |
| Engine | `run_orientation_peaks`, `apply_orientation_peaks_to_basis_params` | Wrappers finos; reusa `_build_crystal` / calibraciones hasta basis. |
| GUI | `qt_orientation.OrientationPeaksDialog` | Run / Compare vs Index BVM / Send / Export. |
| Botón | `qt_main` Basis toolbar | **Orient. peaks…** junto a **Index BVM…**. |
| Tests | `tests/test_orientation_peaks.py` | Matcher unidades + Path A sintético + Path B smoke (Si.cif). |
| Plan | `docs/superpowers/plans/2026-07-15-orientation-peaks-gui.md` | Road O0–O4. |

### Flujo
1. Basis → **Orient. peaks…**
2. CIF + Known (zone/proj) o ACOM → **Run**
3. Opcional **Compare vs Index BVM** (no escribe params)
4. Opcional **Send** → mismos `index_origin/g1/g2` + `basis_manual_enabled`

### Notas
- **No** modifica `pipeline.py` strain steps.
- Parche NumPy 2 para `astype(np.integer)` en ACOM (`_patch_acom_numpy_integer`).
- Unidades: teórico en Å⁻¹; BVM en px vía `Q_pixel` (igual que Index BVM).

---

## 16. Indexing plugins — física + menú Plugin — 2026-07-15

### Diferencia física (qué camino elegir)

| Camino | Pregunta | Cristal | Zone / ejes | Cómo |
|---|---|---|---|---|
| **Index BVM Unknown** | ¿Qué red 2D forman los spots? | Opcional | No (zone solo → hkl relativo) | RANSAC → g1/g2 |
| **Index BVM Known** | ¿Miller **absolutos**? | Sí (`a`/CIF) | Zone + ejes H/V + QR | RANSAC + anclaje ±g |
| **Orient Path A** | ¿Dónde deberían estar los Bragg? | CIF completo | Zone + proj_x | Genera patrón → NN |
| **Orient Path B** | ¿Qué orientación del CIF encaja? | CIF completo | No (ACOM busca) | ACOM → regenera → NN |

**Index BVM** = medida → red → (opcional) etiquetas.  
**Orient. peaks** = CIF → orientación (dada o buscada) → picos teóricos → NN.

### Separación Plugin

| Pieza | Archivo(s) | Qué hace |
|---|---|---|
| Paquete | `plugins/indexing/` | `peaks` (upsample), `types`, `protocol`, 3 plugins, `registry`, `apply` |
| Menú | `qt_main` **Plugin** | Index BVM Unknown / Known / Orient. peaks (quitados del toolbar Basis) |
| Help | Indexing plugins guide… | Árbol “qué sabes” + workflow |
| Peak sampling | diálogos | 1× / 2× / 4× zoom BVM antes de máximos |

Dependencia cortada: `orientation_peaks` ya **no** importa `bvm_indexing` (ambos usan `plugins.indexing.peaks`).
