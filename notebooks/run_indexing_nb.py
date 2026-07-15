"""Ejecuta indexing_bvm_demo.ipynb de punta a punta (headless) y falla si algún assert falla.

Requiere un Python con nbclient/nbformat (NO tiene que ser el del kernel):
    python notebooks/run_indexing_nb.py
El notebook corre en el kernel registrado "py4dstem-01419"
(registrar una vez: <env>/python.exe -m ipykernel install --user --name py4dstem-01419).
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import nbformat
    from nbclient import NotebookClient
except ImportError:
    sys.exit(
        "Falta nbclient/nbformat en este Python. Usa un env que los tenga, p.ej.:\n"
        '  & "C:\\Users\\jtapiaca.ASURITE\\.conda\\envs\\py4dstem\\python.exe" notebooks/run_indexing_nb.py'
    )

NB = Path(__file__).resolve().parent / "indexing_bvm_demo.ipynb"
KERNEL = "py4dstem-01419"

nb = nbformat.read(NB, as_version=4)
client = NotebookClient(
    nb,
    kernel_name=KERNEL,
    timeout=1800,
    resources={"metadata": {"path": str(NB.parent)}},
)
print(f"Ejecutando {NB.name} con kernel {KERNEL}...")
client.execute()
nbformat.write(nb, NB)
print("OK — notebook ejecutado sin errores (asserts de regresión incluidos).")
print("Salidas en:", NB.parent / "output")
