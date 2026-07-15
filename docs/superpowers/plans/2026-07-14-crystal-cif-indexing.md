# Plan — Crystal from CIF for Index BVM (+ shared Q-pixel)

**Rama / worktree:** `peak-indexer-notebook`  
**Estado:** PLAN — implementar y testear en chat dedicado (después del commit Report/Tools).  
**Fecha:** 2026-07-14  
**Depende de:** Index BVM ya en GUI (`qt_indexer.py`, `bvm_indexing.py`, `engine.index_bvm`).

---

## Contexto actual (qué cristal usa Index hoy)

Index BVM reutiliza el **mismo cristal que Q-pixel**:

| Fuente | Detalle |
|---|---|
| Default | **Si** diamond, `a = 5.431 Å` (`CAL_CRYSTALS["Si"]`) |
| Alternativa | **Au** FCC, `a = 4.078 Å` |
| Custom | `CrystalEditorDialog` → `params.custom_crystal` `{a_lat, atom_num, positions}` |

Flujo:

```
IndexerDialog → params.cal_crystal_obj()
             → engine.index_bvm(..., lattice_a=crystal.a_lat, zone_axis=...)
             → bvm_indexing.index_bvm (modelo cúbico: |g| ≈ |hkl| / a)
```

UI hoy solo muestra `Crystal: Si  a = 5.4310 Å` (lectura). No hay Load CIF.

## Objetivo

1. Cargar un **archivo CIF** (formato clásico de cristalografía) como cristal de referencia.
2. Usar **el mismo CIF** para Index BVM y, idealmente, Q-pixel / structure factors.
3. Permitir reutilizar el mismo fichero en **GPA** y **4DSTEM** (un solo source of truth).

## Stack disponible

py4DSTEM ya expone:

```python
from py4DSTEM.process.diffraction import Crystal
Crystal.from_CIF(path, primitive=True, conventional_standard_structure=True)
```

Depende de **pymatgen** (verificar en env `py4dstem-01419`; añadir a `requirements.txt` si falta).

## Diseño propuesto

### Params

```text
cal_crystal: "Si" | "Au" | "Custom" | "CIF"
cif_path: str | None          # ruta al .cif (persistir en session JSON)
# custom_crystal sigue para el editor manual
```

`cal_crystal_obj()` / `_build_crystal()`:

- Si `CIF` y `cif_path` válido → `Crystal.from_CIF(...)` (py4DSTEM) + cache de `a_lat` / name para Index.
- Fallbacks claros si pymatgen/CIF falla.

### UI

- **Index BVM** (`qt_indexer.py`): botón `Load CIF…` + label con nombre/a del CIF.
- **Crystal editor / Q-pixel** (opcional misma sesión): selector “From CIF” reutilizando `cif_path`.
- Al cargar CIF: set `cal_crystal="CIF"`, guardar path, refrescar label.

### Indexación

- **v1 (cúbico / métrica efectiva):** extraer `a` (o promedio de a≈b≈c) del CIF → `lattice_a` como hoy.
- **v1.5 (si no cúbico):** avisar en UI que el indexador BVM aún asume métrica cúbica; ofrecer “use conventional a” o bloquear con mensaje.
- **v2 (futuro):** extender `bvm_indexing` a celda general (matriz métrica / d-spacings del Crystal py4DSTEM).

### Persistencia

- Session / workspace: guardar `cif_path` (y opcionalmente copia relativa del CIF en `results/` o path absoluto).
- No embeber el CIF entero en JSON a menos que el path se rompa.

## Fases

### C0 — Spike
- En env `py4dstem-01419`: `Crystal.from_CIF("…Si.cif")` → imprimir lattice, positions count.
- Confirmar pymatgen instalado.

### C1 — Engine
- `load_crystal_from_cif(path) -> CalCrystal | py4DSTEM.Crystal` helper en `engine.py`.
- Extender `cal_crystal_obj` / `_build_crystal` / `index_bvm` para `CIF`.
- Tests unitarios con un CIF mínimo (Si o fixture en `tests/fixtures/`).

### C2 — Indexer UI
- `Load CIF…` en `IndexerDialog`.
- Mostrar path + a + warning si no cúbico.
- Run indexing con `lattice_a` del CIF.

### C3 — Q-pixel parity (mismo CIF)
- `_build_crystal` ya lee CIF → calibrate_pixel_size usa el mismo Crystal.
- Documentar en Help / Quick Start: “one CIF for GPA + Fast4D”.

### C4 — Docs + graphify / codebase-memory
- Actualizar `ForGITHUB-Updates.md`.
- `graphify update .`
- Re-index codebase-memory MCP del worktree si el proyecto está registrado.

## Criterios de éxito

1. Load CIF → Index BVM corre y propone g1/g2 sin usar Si hardcodeado.
2. Mismo `cif_path` alimenta Q-pixel structure factors cuando `cal_crystal=="CIF"`.
3. Test automatizado con fixture CIF (no red).
4. Mensaje claro si el CIF no es cúbico (v1).
5. Nada de secrets; CIF de ejemplo libre (p.ej. Si público) en `tests/fixtures/`.

## Archivos probables

| Área | Archivos |
|---|---|
| Motor | `engine.py` (`CalCrystal`, `_build_crystal`, `index_bvm`) |
| Index | `bvm_indexing.py`, `qt_indexer.py` |
| Params | `param_spec.py`, session save/load |
| Q-pixel | `CrystalEditorDialog` / `_build_crystal` |
| Tests | `tests/test_crystal_cif.py`, `tests/fixtures/*.cif` |
| Deps | `requirements.txt` (+ pymatgen si hace falta) |

## No-objetivos (v1)

- Editor CIF WYSIWYG.
- Indexación no-cúbica completa (solo warning).
- Descargar CIFs de la red automáticamente.

## Kickoff checklist

- [ ] Confirmar pymatgen en `py4dstem-01419`
- [ ] Elegir CIF de prueba (Si o material del lab)
- [ ] Decidir: CIF solo Index, o Index+Q-pixel desde el día 1 (recomendado: ambos vía `_build_crystal`)
- [ ] Usar **codebase-memory** + **graphify** al explorar call graph de `index_bvm` / `_build_crystal`
