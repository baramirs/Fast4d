"""Mini synthetic Si [110] dataset with known truth, for fast tests (no .mib, no detection).

Same physics as PARA TEST/Sintetico_Si110/make_synthetic.py, but the Bragg peak
positions go straight into a py4DSTEM ``BraggVectors`` (16x16 to 32x32 scan):

- Si diamond, a = 5.431 A, zone z || [110], x || [-110] at ``phi_deg`` in (qx, qy),
  y || [001]; kinematic intensities from py4DSTEM ``Crystal`` + a weak 002.
- Q pixel ``q_pix`` (1/A per px); beam origin on a known plane; optional ellipse.
- Strain: physical e_yy only (d' = (1 + e) d along [001]), a linear ramp along y of
  the crystal, 0 at the centre and +-``e_max`` at the ends. py4DSTEM (e = 1 - beta)
  reads (e - e_ref) / (1 + e) against a reference with physical strain e_ref.
- ``vacuum``: those positions get only the central beam, so the lattice fit fails
  there (py4DSTEM mask 0), as in real vacuum.

Geometry (py4DSTEM): qx = row, qy = column of the pattern; rx = row, ry = column of
the scan; no R-Q rotation / flip, so (rx, ry) || (qx, qy).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

A_LAT = 5.431
Q_PIX = 0.0200
PHI_DEG = 355.0
E_MAX = 0.10
I_CENTER = 2500.0
I_111 = 300.0
I_002_REL = 0.25
ORIGIN0 = (127.6, 127.9)
ORIGIN_SLOPE = ((1.5, 0.5), (-1.0, 0.8))
QSHAPE = (256, 256)
EDGE_PX = 10


@lru_cache(maxsize=4)
def reflections(g_max: float = 1.7):
    """(hkl, intensity, gx, gy) of the [110] zone (h + k = 0), g in 1/A along x=[-110], y=[001]."""
    import py4DSTEM
    pos = np.array([[0, 0, 0], [.25, .25, .25], [0, .5, .5], [.25, .75, .75],
                    [.5, 0, .5], [.75, .25, .75], [.5, .5, 0], [.75, .75, .25]], float)
    cr = py4DSTEM.process.diffraction.Crystal(pos, 14, A_LAT)
    cr.calculate_structure_factors(g_max)
    h = cr.hkl.T
    inten = np.asarray(cr.struct_factors_int, float)
    m = (h[:, 0] + h[:, 1] == 0) & np.any(h != 0, axis=1)
    h, inten = h[m], inten[m]
    i111 = inten[np.all(np.abs(h) == 1, axis=1)].max()
    rel = inten / i111 * I_111
    h = np.vstack([h, [[0, 0, 2], [0, 0, -2]]])          # 002: multiple scattering in Si [110]
    rel = np.concatenate([rel, [I_002_REL * I_111] * 2])
    gx = (h[:, 1] - h[:, 0]) / (np.sqrt(2.0) * A_LAT)
    gy = h[:, 2] / A_LAT
    return h.astype(int), rel, gx, gy


def axes(phi_deg: float = PHI_DEG):
    phi = np.radians(phi_deg)
    return np.array([np.cos(phi), np.sin(phi)]), np.array([-np.sin(phi), np.cos(phi)])


def strain_field(n: int, e_max: float = E_MAX, phi_deg: float = PHI_DEG) -> np.ndarray:
    """Physical e_yy (fraction): linear ramp along y of the crystal over the n x n scan."""
    _, yhat = axes(phi_deg)
    rx, ry = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    c = (n - 1) / 2.0
    s = (rx - c) * yhat[0] + (ry - c) * yhat[1]
    return e_max * s / np.abs(s).max()


def origin_field(n: int):
    u, v = np.meshgrid(np.linspace(-0.5, 0.5, n), np.linspace(-0.5, 0.5, n), indexing="ij")
    qx0 = ORIGIN0[0] + ORIGIN_SLOPE[0][0] * u + ORIGIN_SLOPE[0][1] * v
    qy0 = ORIGIN0[1] + ORIGIN_SLOPE[1][0] * u + ORIGIN_SLOPE[1][1] * v
    return qx0, qy0


def vacuum_columns(n: int, ncols: int) -> np.ndarray:
    """Vacuum band on the first ``ncols`` scan columns (ry < ncols)."""
    m = np.zeros((n, n), bool)
    m[:, :ncols] = True
    return m


def _ellipse_distort(d: np.ndarray, ellipse) -> np.ndarray:
    """Stretch offsets d (N, 2) from the origin: semi-axes (a, b) with a at ``theta_deg``
    from +qx (py4DSTEM p_ellipse convention), b = 1 keeps |g| along the minor axis."""
    a, b, theta_deg = ellipse
    t = np.radians(theta_deg)
    u = np.array([np.cos(t), np.sin(t)])
    w = np.array([-np.sin(t), np.cos(t)])
    return (d @ u)[:, None] * a * u[None] + (d @ w)[:, None] * b * w[None]


def make_braggvectors(n: int = 24, *, e_max: float = E_MAX, q_pix: float = Q_PIX,
                      phi_deg: float = PHI_DEG, g_max: float = 1.7, jitter_px: float = 0.03,
                      vacuum=None, ellipse=None, seed: int = 0):
    """(BraggVectors, truth). ``truth``: eyy_phys, eyy_py4dstem (reference e = 0), qx0,
    qy0 (px), vacuum (bool mask), and the generation parameters."""
    from py4DSTEM.braggvectors import BraggVectors
    rng = np.random.default_rng(seed)
    _, amp, gx, gy = reflections(float(g_max))
    xhat, yhat = axes(phi_deg)
    e = strain_field(n, e_max, phi_deg)
    qx0, qy0 = origin_field(n)
    vac = np.zeros((n, n), bool) if vacuum is None else np.asarray(vacuum, bool)
    bv = BraggVectors((n, n), QSHAPE)
    pla = bv._v_uncal
    for i in range(n):
        for j in range(n):
            if vac[i, j]:
                d, a = np.zeros((1, 2)), np.array([I_CENTER])
            else:
                gyp = gy / (1.0 + e[i, j])                       # g'_y = g_y / (1 + e)
                d = (gx[:, None] * xhat[None] + gyp[:, None] * yhat[None]) / q_pix
                if ellipse is not None:
                    d = _ellipse_distort(d, ellipse)
                d = np.vstack([[0.0, 0.0], d])
                a = np.concatenate([[I_CENTER], amp])
            q = d + np.array([qx0[i, j], qy0[i, j]]) + rng.normal(0.0, jitter_px, d.shape)
            keep = np.all((q > EDGE_PX) & (q < np.array(QSHAPE) - EDGE_PX), axis=1)
            data = np.zeros(int(keep.sum()), dtype=pla.dtype)
            data["qx"], data["qy"], data["intensity"] = q[keep, 0], q[keep, 1], a[keep]
            pla.get_pointlist(i, j).add(data)
    bv.set_raw_vectors(pla)
    truth = dict(n=n, eyy_phys=e, eyy_py4dstem=e / (1.0 + e), qx0=qx0, qy0=qy0, vacuum=vac,
                 q_pix=q_pix, phi_deg=phi_deg, e_max=e_max, ellipse=ellipse)
    return bv, truth


def expected_eyy(truth: dict, ref_mask=None) -> np.ndarray:
    """eyy that py4DSTEM must read against the median reference over ``ref_mask`` (valid
    crystal pixels only; None = whole crystal), as a fraction."""
    e = truth["eyy_phys"]
    ok = ~truth["vacuum"]
    m = ok if ref_mask is None else (np.asarray(ref_mask, bool) & ok)
    e_ref = float(np.median(e[m]))
    return (e - e_ref) / (1.0 + e)


def synthetic_adf(truth: dict, contrast: float = 0.05, level: float = 140000.0,
                  seed: int = 1) -> np.ndarray:
    """Low-contrast ADF (+-``contrast`` around ``level``) of a field with no dark areas;
    the vacuum positions (if any) are ~0."""
    rng = np.random.default_rng(seed)
    n = truth["n"]
    adf = level * (1.0 + contrast * (truth["eyy_phys"] / max(truth["e_max"], 1e-12)))
    adf = adf + rng.normal(0.0, 0.2 * contrast * level, (n, n))
    adf[truth["vacuum"]] = rng.normal(50.0, 10.0, int(truth["vacuum"].sum()))
    return adf


def make_scan(bv, *, name: str = "mini", roi_bounds=None, frame_mode: str = "manual",
              coordinate_rotation: float = PHI_DEG, q_guess: float = 0.0210):
    """Engine Scan fed with the synthetic BraggVectors (Path A without the file)."""
    import engine as E
    sc = E.Scan(name=name)
    st = sc.ensure_state()
    st.braggpeaks = bv
    p = sc.params
    p.center_guess = [128.0, 128.0]
    p.q_px = q_guess
    p.q_kmax = 1.6
    p.cal_crystal = "Si"
    p.zone_axis = [1, 1, 0]
    p.frame_mode = frame_mode
    p.frame_x_hkl = [-1, 1, 0]
    p.frame_y_hkl = [0, 0, 1]
    p.coordinate_rotation = coordinate_rotation
    if roi_bounds is not None:
        p.roi_bounds = list(roi_bounds)
    return sc


def run_flow(sc, *, with_roi: bool = True, log=None):
    """Calibration sequence + strain without (and with) ROI, no figures."""
    import engine as E
    E.run_calibration_sequence(sc, make_figures=False, log=log)
    E.compute_strain(sc, use_roi=False, log=log)
    if with_roi:
        E.compute_strain(sc, use_roi=True, log=log)
    return sc
