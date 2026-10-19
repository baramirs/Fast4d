"""Level-2 synthetic 4D-STEM data: drawn Merlin .mib patterns with known strain/rotation.

Adapted from strain_check_ronda2/B/ronda2_B.py (Jose Tapia, 2026-10-10): same disc model
(soft edge I / (1 + exp((r - R) / 0.5))), same Si [110] reflections and amplitudes, same
R = 0.039874 * alpha / q_px (200 kV), same Merlin header template (Scan01). Added here:
lattice rotation phi, a rotation gradient inside the map (bent lamella), an arbitrary 2x2
strain tensor, and the R-Q rotation / flip of the detector.

Geometry (as ronda 2): scan N x N; columns 0..N/2-1 = reference (unstrained), columns
N/2..N-1 = strained block. Crystal x || [-110], y || [001]. Strain tensor e (crystal frame,
physical): g' = (I + e)^-T g. The lattice at scan row i is rotated by
phi + phi_grad * (i - (N - 1) / 2) / (N - 1)  (deg, CCW in calibrated q).
Detector (raw) vs calibrated q (ronda-2 convention, task A): q_raw = [swap if flip] R(-qr) q_cal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

A_SI = 5.431                                     # Å
LAMBDA_200KV = 0.039874                          # 1/Å per mrad (q_alpha = alpha / lambda)
HKL = [(1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (2, -2, 0), (-2, 2, 0), (0, 0, 4), (0, 0, -4)]
AMP = [300, 300, 300, 300, 200, 200, 120, 120]   # per-pixel peak, relative to the beam 2500
I_CENTER = 2500.0
EDGE_PX = 0.5
FAM = ["000", "111", "111", "111", "111", "220", "220", "004", "004"]
TEMPLATES = Path(__file__).resolve().parent / "templates"


def g_crystal(hkl) -> np.ndarray:
    """(gx, gy) in 1/Å, crystal frame x || [-110], y || [001]."""
    h = np.asarray(hkl, float)
    return np.column_stack([(h[:, 1] - h[:, 0]) / (np.sqrt(2.0) * A_SI), h[:, 2] / A_SI])


def rot(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def expected_py4dstem(e_phys: np.ndarray) -> np.ndarray:
    """py4DSTEM reads e = I - (I + e_phys)^-1 (beta convention), crystal frame."""
    return np.eye(2) - np.linalg.inv(np.eye(2) + e_phys)


def _frame_header(tpl: bytes, k: int, det: int) -> bytes:
    t = bytearray(tpl)
    t[4:10] = f"{k:06d}".encode()
    s = bytes(t).replace(b",0256,0256,", f",{det:04d},{det:04d},".encode())
    if len(s) != len(tpl):
        raise RuntimeError("Merlin frame header template changed length")
    return s


def _write_hdr(path: Path, n: int, det: int) -> None:
    rows = []
    for line in (TEMPLATES / "merlin_scan.hdr").read_bytes().decode("latin-1").split("\n"):
        key = line.split(":")[0]
        if key == "ROI Rows":
            line = f"ROI Rows:\t{det}"
        elif key == "Detector Field Of View":
            line = f"Detector Field Of View:\t{det:.2f} x {det:.2f} scan"
        elif key in ("ScanX", "ScanY"):
            line = f"{key}:\t{n}"
        elif line.startswith("Frames in Acquisition"):
            line = f"Frames in Acquisition (Number):\t{n * n}"
        elif line.startswith("Frames per Trigger"):
            line = f"Frames per Trigger (Number):\t{n}"
        rows.append(line)
    path.write_bytes("\n".join(rows).encode("latin-1"))


def render(det: int, centers, amps, radius: float, dose, rng) -> np.ndarray:
    """Soft-edged discs; dose None -> beam peak 4000 counts, no noise; else Poisson with
    ``dose`` electrons in the pattern."""
    img = np.zeros((det, det))
    w = int(np.ceil(radius + 6 * EDGE_PX)) + 1
    off = np.arange(-w, w + 1)
    for (cx, cy), a in zip(centers, amps):
        ix = int(np.floor(cx)) + off[:, None]
        iy = int(np.floor(cy)) + off[None, :]
        r = np.hypot(ix - cx, iy - cy)
        v = a / (1.0 + np.exp((r - radius) / EDGE_PX))
        ok = (ix >= 0) & (ix < det) & (iy >= 0) & (iy < det)
        ixb, iyb = np.broadcast_arrays(ix, iy)
        np.add.at(img, (ixb[ok], iyb[ok]), v[ok])
    out = np.round(img * (4000.0 / I_CENTER)) if dose is None else rng.poisson(img * (dose / img.sum()))
    return np.minimum(out, 65535).astype(">u2")


def make_case(folder: Path, name: str, *, n: int, det: int, q_px: float, alpha_mrad: float,
              dose, strain_phys, phi_deg: float, phi_grad_deg: float, qr_deg: float,
              flip: bool, n_vacuum: int, vacuum_dose: float, seed: int) -> dict:
    """Write <name>.mib/.hdr + <name>_vacuum.mib/.hdr + truth.npz; return the truth dict."""
    folder.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    radius = LAMBDA_200KV * alpha_mrad / q_px
    center = np.array([det / 2 - 0.4, det / 2 + 0.3])
    e_phys = np.asarray(strain_phys, float)
    if e_phys.shape != (2, 2) or not np.allclose(e_phys, e_phys.T):
        raise ValueError("strain_phys must be a symmetric 2x2 tensor")
    g0 = g_crystal(HKL)
    gs = g0 @ np.linalg.inv(np.eye(2) + e_phys)            # rows: g'^T = g^T (I + e)^-1
    raw_of_cal = rot(-qr_deg)
    if flip:
        raw_of_cal = np.array([[0.0, 1.0], [1.0, 0.0]]) @ raw_of_cal
    amps = np.concatenate([[I_CENTER], AMP])
    tpl = (TEMPLATES / "merlin_frame_header384.bin").read_bytes()
    centers = np.zeros((n, n, len(HKL) + 1, 2))
    phi_map = np.zeros((n, n))
    with (folder / f"{name}.mib").open("wb") as f:
        k = 0
        for i in range(n):
            phi_i = phi_deg + phi_grad_deg * (i - (n - 1) / 2) / max(n - 1, 1)
            R = rot(phi_i)
            for j in range(n):
                g = gs if j >= n // 2 else g0
                d_cal = (g @ R.T) / q_px                    # px, calibrated frame
                d_raw = d_cal @ raw_of_cal.T
                cen = np.vstack([center, center + d_raw])
                centers[i, j], phi_map[i, j] = cen, phi_i
                k += 1
                f.write(_frame_header(tpl, k, det))
                f.write(render(det, cen, amps, radius, dose, rng).tobytes())
    _write_hdr(folder / f"{name}.hdr", n, det)
    with (folder / f"{name}_vacuum.mib").open("wb") as f:
        for k in range(1, n_vacuum * n_vacuum + 1):
            f.write(_frame_header(tpl, k, det))
            f.write(render(det, center[None], np.array([I_CENTER]), radius, vacuum_dose, rng).tobytes())
    _write_hdr(folder / f"{name}_vacuum.hdr", n_vacuum, det)
    truth = dict(centers=centers, phi_map=phi_map, radius_px=radius, q_px=q_px, center=center,
                 e_phys=e_phys, e_expected=expected_py4dstem(e_phys), qr_deg=qr_deg, flip=flip)
    np.savez_compressed(folder / "truth.npz", **truth)
    inside = np.all((centers > 0) & (centers < det - 1), axis=-1)
    truth["fraction_discs_inside"] = float(inside.mean())
    return truth
