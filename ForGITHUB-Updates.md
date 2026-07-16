# Fast4D — Release notes draft (next version)

**Audience:** GitHub PR / Release / CHANGELOG  
**Language:** English  
**Scope:** Workflow, memory, and Report improvements on `main`  
**Status:** Public tree cleaned — internal plan docs and BVM indexing modules are **not** shipped

---

## Highlights

| Area | One-liner |
|------|-----------|
| **RAM lifecycle** | Release datacube after Bragg save; optional CPU streaming; Free RAM can drop reloadable peaks |
| **Report** | Auto-show, tree browser, PDF/DOCX/PPTX export without collage cropping |
| **Tools menu** | Live Line / Live ROI / Set up Lines & ROI / Analyse under **Tools** |
| **UX** | Modeless figures; View → Tools → Settings → Help; strain theta clim respects user range |
| **EMD / virtual images** | More robust tutorial EMD load; save ADF/BF into the same `.h5` |
| **Crystal (Q-pixel)** | Optional CIF load for Q-pixel calibration crystal |

---

## What's new

### RAM lifecycle
Large memmapped datacubes are released after Bragg peaks are saved. Path A calibration/strain uses compact braggpeaks. Optional CPU streamed detection (`FAST4D_STREAM_BRAGG=1`). Free RAM can drop reloadable peaks. Detection science unchanged.

### Report
Auto-show on selection; per-scan tree (Calibrations / Maps / Reports / Legacy / Session); Export… → PDF / DOCX / PPTX with full figures (`python-docx`).

### Tools menu and UX
**View → Tools → Settings → Help.** Live/Analyse tools live under Tools (not the Analysis toolbar). Modeless figures; theta clim respects GUI vrange; Q-pixel origin guard; optional CIF for Q-pixel crystal.

### EMD / ADF-BF
More robust DataCube path discovery; virtual images can append ADF/BF into the same loaded `.h5`.

---

## Test plan

```text
pytest tests/test_report_tree.py tests/test_report_export.py -q
pytest tests/test_bragg_stream_finalize.py tests/test_free_memory.py tests/test_drop_braggpeaks_reload.py -q
pytest tests/test_emd_datapath.py tests/test_vc_save_same_h5.py tests/test_qpixel_origin_guard.py -q
pytest tests/test_crystal_cif.py -q
```

Manual: `run_gui.bat` → Detect → Origin→Strain without cube reload; Report export; Tools → Live Line; optional EMD + CIF for Q-pixel.

---

## Notes

- Public repo root intentionally stays a flat app layout for now; packaging under `fast4d/` is a later cleanup.
- Indexing / Plugin work lives on development branches only — not in this public tree.
