# Fast4D

A desktop GUI and processing pipeline for **4D-STEM** (four-dimensional scanning
transmission electron microscopy) analysis, built on
[py4DSTEM](https://github.com/py4dstem/py4DSTEM) and PySide6 (Qt6).

Fast4D wraps the full workflow — from loading a datacube, calibrating the
diffraction geometry, detecting Bragg disks, to computing strain and stress
maps — behind a single dockable Qt interface, with the heavy computation running
off the GUI thread.

## Features

- **Interactive Qt6 GUI** with dockable panels, live previews, and a built-in
  Quick Start guide.
- **Calibration** of diffraction center, ellipse, and pixel size.
- **Bragg disk detection** and lattice/basis fitting.
- **Strain and stress mapping** (Hooke's law from the fitted strain).
- **Drift estimation** across ADF / strain map series.
- **Batch processing** for running the pipeline over multiple datasets.
- **Optional GPU acceleration** via CuPy (NVIDIA/CUDA).
- **Report export**, including PowerPoint (`python-pptx`).

## Requirements

- Python 3.10+
- The dependencies in [`requirements.txt`](requirements.txt)

A conda environment is recommended, since `py4DSTEM` and its scientific stack
are easiest to install that way.

## Installation

```bash
# Clone
git clone https://github.com/baramirs/Fast4d.git
cd Fast4d

# (recommended) create an environment, then install deps
pip install -r requirements.txt
```

> **GPU (optional):** CuPy and `pynvml` are commented out in `requirements.txt`.
> Install the CuPy wheel matching your CUDA version (e.g. `cupy-cuda12x`) to
> enable GPU-accelerated processing.

## Usage

Launch the GUI:

```bash
python app.py
```

On Windows you can also double-click **`run_gui.bat`** (edit the `CONDA_PREFIX`
and `PY` paths inside it to point at your own environment).

## Project layout

| Path | Purpose |
|------|---------|
| `app.py` | GUI entrypoint — creates the QApplication and main window. |
| `qt_main.py`, `qt_*.py` | Qt widgets, panels, loaders, splash, report, quick start. |
| `engine.py` | Core computation engine (calibration, detection, strain). |
| `pipeline.py` | End-to-end processing pipeline. |
| `driver.py` | Orchestrates pipeline steps off the GUI thread. |
| `analysis.py`, `stress_analysis.py` | Strain/stress analysis. |
| `drift_estimate.py` | Drift estimation across map series. |
| `fast_batch.py`, `batch_common.py`, `batch_figures.py` | Batch processing and figures. |
| `calib_params_io.py`, `param_spec.py`, `state.py` | Parameters, I/O, and app state. |
| `fast_artifacts.py` | Artifact/result handling. |
| `icons/` | GUI icons. |
| `tools/` | Helper scripts (e.g. calibration → PowerPoint export). |
| `notebooks/` | Exploratory Jupyter notebooks. |

## License

No license specified yet. All rights reserved by the author unless a license is
added.

## Author

**Jose Manuel Tapia Caceres** ([@baramirs](https://github.com/baramirs))
