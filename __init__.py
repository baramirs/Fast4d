"""Fast4D — unified 4D-STEM strain/stress GUI (PySide6 / Qt6).

A clean, single-purpose rebuild of the Fast Analysis workflow:
  • One GUI for single and multi-scan (multi = template + serial compute).
  • COMPUTE (heavy, once) is separated from ANALYSIS (light, repeatable).
  • Grounded in 4Dstrain-analysis.ipynb; reuses pipeline.py / fast_artifacts.py.

Layers
------
UI-free (engine; reused as-is from the proven codebase):
  engine.py     — orchestration over pipeline / fast_artifacts / fast_batch / stress
  driver.py     — single==multi serial loop; Path A (braggpeaks) vs B (raw→detect)
  analysis.py   — scientific/statistical layer (scipy/pandas/sklearn/uncertainties)
  param_spec.py — framework-neutral parameter specification (no GUI dependency)

GUI (PySide6 / Qt6):
  qt_params.py  — Excel-like parameter table (QTabWidget of QTableWidget)
  qt_widgets.py — ResourceMonitor, CalStateStrip, ConsoleWidget, AdfView (pyqtgraph)
  qt_main.py    — dockable QMainWindow (icon strip → tabs, threaded Compute/Analysis)
  app.py        — entrypoint  (python app.py)

Run::

  cd C:\\Users\\jtapiaca.ASURITE\\Fast4d
  conda run -n py4dstem-01419 python app.py
"""
