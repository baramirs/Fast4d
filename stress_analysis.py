"""
Stress analysis from 2D strain maps (linear elasticity, cubic crystal).

Strain components are unitless tensor strains ε_ij as produced by the workflow
(εyy, εxx, εxy channel order in the raw stack). Rotation θ is not used.

Stiffness C_ijkl in Voigt form for cubic: C11, C12, C44 in Pa (cube axes).
The stiffness is rotated into the axes of the strain tensor (crystal frame card,
see crystal_frame.py) before Hooke's law is applied.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

# Silicon at 300 K (approx.), Pa
ELASTIC_SILICON_PA = {
    "C11": 165.7e9,
    "C12": 63.9e9,
    "C44": 79.6e9,
}


def pa_to_gpa(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64) / 1.0e9


# --------------------------------------------------------------------------- #
# Stiffness in the frame of the strain tensor
# --------------------------------------------------------------------------- #
# Voigt order: 1 xx, 2 yy, 3 zz, 4 yz, 5 xz, 6 xy (engineering shear strains).
_VOIGT = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (2, 1): 3,
          (0, 2): 4, (2, 0): 4, (0, 1): 5, (1, 0): 5}


def cubic_stiffness_voigt(C11: float, C12: float, C44: float) -> np.ndarray:
    """6x6 Voigt stiffness of a cubic crystal in its CUBE axes [100],[010],[001]."""
    C = np.zeros((6, 6))
    C[:3, :3] = C12
    np.fill_diagonal(C[:3, :3], C11)
    C[3, 3] = C[4, 4] = C[5, 5] = C44
    return C


def _voigt_to_tensor(C6: np.ndarray) -> np.ndarray:
    C = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i, j, k, l] = C6[_VOIGT[(i, j)], _VOIGT[(k, l)]]
    return C


def _tensor_to_voigt(C: np.ndarray) -> np.ndarray:
    C6 = np.zeros((6, 6))
    for (i, j), a in _VOIGT.items():
        for (k, l), b in _VOIGT.items():
            C6[a, b] = C[i, j, k, l]
    return C6


def tensor_frame_axes(x_hkl, y_hkl, psi_deg: float = 0.0) -> np.ndarray:
    """3x3 matrix whose ROWS are the tensor axes (x, y, z) in cube coordinates.

    x starts along x_hkl, y along y_hkl (Gram-Schmidt), z = x cross y (the beam
    / zone direction); then x, y are turned by psi_deg about z (from x towards
    y). Cubic directions: [hkl] is parallel to (hkl)."""
    ex = np.asarray(x_hkl, float)
    ex = ex / np.linalg.norm(ex)
    ey = np.asarray(y_hkl, float)
    ey = ey - (ey @ ex) * ex
    ey = ey / np.linalg.norm(ey)
    t = np.radians(float(psi_deg))
    ex, ey = np.cos(t) * ex + np.sin(t) * ey, -np.sin(t) * ex + np.cos(t) * ey
    return np.vstack([ex, ey, np.cross(ex, ey)])


def rotate_stiffness(C6: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Voigt stiffness expressed in the axes given as rows of ``axes``."""
    R = np.asarray(axes, float)
    C = _voigt_to_tensor(C6)
    Cr = np.einsum("ai,bj,ck,dl,ijkl->abcd", R, R, R, R, C, optimize=True)
    return _tensor_to_voigt(Cr)


def inplane_stiffness(C11: float, C12: float, C44: float, *, mode: str = "plane_stress",
                      axes: np.ndarray | None = None) -> np.ndarray:
    """3x3 Q with (sigma_xx, sigma_yy, sigma_xy) = Q @ (eps_xx, eps_yy, 2 eps_xy).

    ``axes``: rows = tensor axes in cube coordinates (``tensor_frame_axes``);
    None = cube axes (the historical behaviour).
    plane_strain: eps_zz = eps_yz = eps_xz = 0.
    plane_stress: sigma_zz = sigma_yz = sigma_xz = 0 (thin lamella); the
    out-of-plane strains are eliminated exactly, including the couplings that
    appear when the axes are not cube axes.
    """
    m = str(mode).lower().replace("-", "_").replace(" ", "_")
    C = cubic_stiffness_voigt(C11, C12, C44)
    if axes is not None:
        C = rotate_stiffness(C, axes)
    p, o = [0, 1, 5], [2, 3, 4]
    Cpp = C[np.ix_(p, p)]
    if m == "plane_strain":
        return Cpp
    if m == "plane_stress":
        Coo = C[np.ix_(o, o)]
        if abs(np.linalg.det(Coo)) < 1e-30:
            raise ValueError("singular out-of-plane stiffness (check C11, C44)")
        return Cpp - C[np.ix_(p, o)] @ np.linalg.solve(Coo, C[np.ix_(o, p)])
    raise ValueError(f"mode must be 'plane_stress' or 'plane_strain', got {mode!r}")


def compute_stress(
    eps_xx: np.ndarray,
    eps_yy: np.ndarray,
    eps_xy: np.ndarray,
    C11: float,
    C12: float,
    C44: float,
    *,
    mode: str = "plane_stress",
    axes: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Hooke's law for a cubic crystal in the film (x-y) plane, IN THE AXES OF THE
    STRAIN TENSOR.

    The constants C11, C12, C44 are cube-axis constants. When the tensor is not
    written along cube axes (e.g. x = [-110], y = [001] for Si [110]) the
    stiffness is rotated first; ``axes`` gives the tensor axes (rows, cube
    coordinates). With axes=None the cube-axis formulas are used:

    * ``plane_strain``: sigma_xx = C11 exx + C12 eyy, sigma_xy = 2 C44 exy.
    * ``plane_stress``: C11' = C11 - C12^2/C11, C12' = C12 (1 - C12/C11).

    Returns sigma_ij in **Pa** (same shape as inputs). theta is not used.
    """
    exx = np.asarray(eps_xx, dtype=np.float64)
    eyy = np.asarray(eps_yy, dtype=np.float64)
    exy = np.asarray(eps_xy, dtype=np.float64)
    if exx.shape != eyy.shape or exx.shape != exy.shape:
        raise ValueError("eps_xx, eps_yy, eps_xy must have the same shape.")
    if abs(float(C11)) < 1e-30:
        raise ValueError("C11 must be non-zero.")
    Q = inplane_stiffness(float(C11), float(C12), float(C44), mode=mode, axes=axes)
    g = 2.0 * exy
    return {
        "sigma_xx": Q[0, 0] * exx + Q[0, 1] * eyy + Q[0, 2] * g,
        "sigma_yy": Q[1, 0] * exx + Q[1, 1] * eyy + Q[1, 2] * g,
        "sigma_xy": Q[2, 0] * exx + Q[2, 1] * eyy + Q[2, 2] * g,
    }


def _symmetric_gpa_limits(
    sxx: np.ndarray,
    syy: np.ndarray,
    sxy: np.ndarray,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)
    stack = np.concatenate(
        [
            np.asarray(sxx, dtype=np.float64).ravel(),
            np.asarray(syy, dtype=np.float64).ravel(),
            np.asarray(sxy, dtype=np.float64).ravel(),
        ]
    )
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.nanpercentile(finite, (2.0, 98.0))
    bound = float(max(abs(lo), abs(hi), 1e-9))
    return -bound, bound


def build_stress_maps_figure(
    sigma_pa: dict[str, np.ndarray],
    *,
    mode_label: str = "",
    vmin_gpa: float | None = None,
    vmax_gpa: float | None = None,
    units: str = "GPa",
    overlay_strain_percent: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    strain_vminmax_percent: tuple[float, float] = (-5.0, 5.0),
    line_segments: "dict | None" = None,
    line_profile_width: int = 3,
) -> Figure:
    """
    σ_xx, σ_yy, σ_xy in GPa with RdBu_r and labeled colorbars.
    If ``overlay_strain_percent`` is set (εyy, εxx, ε_xy unitless), add a top row in %.

    ``line_segments``: optional ``{lid: ((x0, y0), (x1, y1))}`` — drawn on every stress-map
    subplot in cycling colours with a semi-transparent width band.  Coordinates are in
    pixels (column=x, row=y) matching the *imshow* image axes.
    ``line_profile_width``: integration-band half-width in pixels (used for band visualisation).
    """
    # ── line-overlay helpers ──────────────────────────────────────────────────
    _LINE_COLORS_STRESS = [
        "#FFFF00", "#00FFFF", "#FF8C00", "#00FF7F", "#FF69B4", "#ADFF2F",
    ]

    def _draw_lines_on_ax(ax, segs: dict, width_px: int) -> None:
        """Overlay every segment from *segs* on *ax*."""
        if not segs:
            return
        for i, (lid, seg) in enumerate(sorted(segs.items())):
            try:
                p0, p1 = seg
                x0, y0 = float(p0[0]), float(p0[1])
                x1, y1 = float(p1[0]), float(p1[1])
                color = _LINE_COLORS_STRESS[i % len(_LINE_COLORS_STRESS)]
                # Thin outline for contrast, then bright centre line
                ax.plot([x0, x1], [y0, y1], color="black", linewidth=max(2, width_px + 1),
                        alpha=0.4, solid_capstyle="round", zorder=9)
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=max(1.5, width_px - 1),
                        alpha=0.9, solid_capstyle="round", zorder=10,
                        label=f"L{lid}" if str(lid)[0].upper() != "L" else str(lid))
                # Mark endpoints
                ax.scatter([x0, x1], [y0, y1], color=color, s=22, zorder=11,
                           edgecolors="black", linewidths=0.5)
            except Exception:
                pass

    segs: dict = {}
    if isinstance(line_segments, dict):
        segs = {k: v for k, v in line_segments.items() if v is not None}
    w_px = max(1, int(line_profile_width))

    # display units: GPa (default) or MPa (×1000). vmin/vmax are in display units.
    _scale = 1000.0 if str(units).upper() == "MPA" else 1.0
    _u = "MPa" if _scale != 1.0 else "GPa"
    sxx = pa_to_gpa(sigma_pa["sigma_xx"]) * _scale
    syy = pa_to_gpa(sigma_pa["sigma_yy"]) * _scale
    sxy = pa_to_gpa(sigma_pa["sigma_xy"]) * _scale
    vmin, vmax = _symmetric_gpa_limits(sxx, syy, sxy, vmin_gpa, vmax_gpa)

    if overlay_strain_percent is None:
        fig = Figure(figsize=(12.5, 3.8))
        axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
        maps_gpa = (sxx, syy, sxy)
        titles = (rf"$\sigma_{{xx}}$ ({_u})", rf"$\sigma_{{yy}}$ ({_u})", rf"$\sigma_{{xy}}$ ({_u})")
        for ax, Z, tit in zip(axes, maps_gpa, titles, strict=True):
            im = ax.imshow(Z, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="upper")
            ax.set_title(tit)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(_u)
            _draw_lines_on_ax(ax, segs, w_px)
        try:
            fig.suptitle(f"Stress maps — {mode_label}" if mode_label else "Stress maps", fontsize=13)
        except Exception:
            pass
        fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12)
        return fig

    fig = Figure(figsize=(12.5, 7.2))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
    eyy = np.asarray(overlay_strain_percent[0], dtype=float) * 100.0
    exx = np.asarray(overlay_strain_percent[1], dtype=float) * 100.0
    exy = np.asarray(overlay_strain_percent[2], dtype=float) * 100.0
    sv0, sv1 = strain_vminmax_percent
    axes_s = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axes_sig = [fig.add_subplot(gs[1, i]) for i in range(3)]
    st_maps = (eyy, exx, exy)
    st_titles = (r"$\varepsilon_{yy}$ (%)", r"$\varepsilon_{xx}$ (%)", r"$\varepsilon_{xy}$ (%)")
    for ax, Z, tit in zip(axes_s, st_maps, st_titles, strict=True):
        im = ax.imshow(Z, cmap="RdBu_r", vmin=sv0, vmax=sv1, origin="upper")
        ax.set_title(tit)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("%")
        _draw_lines_on_ax(ax, segs, w_px)
    sig_maps = (sxx, syy, sxy)
    sig_titles = (r"$\sigma_{xx}$ (GPa)", r"$\sigma_{yy}$ (GPa)", r"$\sigma_{xy}$ (GPa)")
    for ax, Z, tit in zip(axes_sig, sig_maps, sig_titles, strict=True):
        im = ax.imshow(Z, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(tit)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("GPa")
        _draw_lines_on_ax(ax, segs, w_px)
    try:
        fig.suptitle(
            f"Strain (%) + stress (GPa) — {mode_label}" if mode_label else "Strain + stress",
            fontsize=13,
        )
    except Exception:
        pass
    return fig


def _nearest_sample(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    xi = np.clip(np.rint(xs).astype(int), 0, arr.shape[1] - 1)
    yi = np.clip(np.rint(ys).astype(int), 0, arr.shape[0] - 1)
    return arr[yi, xi]


_PROFILE_ABS_INVALID = 1e12


def _sanitize_line_profile_map(arr: np.ndarray) -> np.ndarray:
    """Non-finite and absurd magnitudes → NaN before nanmean across line width."""
    a = np.asarray(arr, dtype=np.float64)
    out = a.copy()
    bad = ~np.isfinite(out) | (np.abs(out) > _PROFILE_ABS_INVALID)
    out[bad] = np.nan
    return out


def sample_line_profile(arr: np.ndarray, p0: tuple[float, float], p1: tuple[float, float], width: int = 3):
    arr = _sanitize_line_profile_map(np.asarray(arr, dtype=float))
    x0, y0 = map(float, p0)
    x1, y1 = map(float, p1)
    length = max(2, int(round(np.hypot(x1 - x0, y1 - y0))) + 1)
    xs = np.linspace(x0, x1, length)
    ys = np.linspace(y0, y1, length)
    dx = x1 - x0
    dy = y1 - y0
    norm = np.hypot(dx, dy) or 1.0
    nx = -dy / norm
    ny = dx / norm
    w = max(1, int(width))
    offsets = np.arange(-(w // 2), w // 2 + 1)
    samples = []
    for off in offsets:
        samples.append(_nearest_sample(arr, xs + off * nx, ys + off * ny))
    values = np.nanmean(np.vstack(samples), axis=0)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    distances = np.linspace(0, float(np.hypot(x1 - x0, y1 - y0)), length)
    return distances, values


def build_stress_line_profiles_figure(
    sigma_xx_pa: np.ndarray,
    sigma_yy_pa: np.ndarray,
    segments: dict[int, tuple[tuple[float, float], tuple[float, float]]],
    *,
    width: int = 3,
    title: str = "Stress along lines",
    y_stress_unit: str = "GPa",
) -> Figure:
    """σ_xx and σ_yy along each segment (literature-style lineouts).

    Internal storage is Pa; axis can be GPa (default) or MPa (values ×1000 vs GPa).
    """
    u = str(y_stress_unit).strip().upper()
    if u in ("MPA", "MEGAPASCAL"):
        scale = 1000.0
        ylabel = "MPa"
    else:
        scale = 1.0
        ylabel = "GPa"
    sxx = pa_to_gpa(np.asarray(sigma_xx_pa, dtype=float)) * scale
    syy = pa_to_gpa(np.asarray(sigma_yy_pa, dtype=float)) * scale
    nseg = len(segments)
    fig = Figure(figsize=(10.0, max(2.8, 2.9 * nseg)))
    if not segments:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No line segments", ha="center", va="center")
        return fig
    gs = fig.add_gridspec(nseg, 1, hspace=0.45)
    for row, (lid, (p0, p1)) in enumerate(sorted(segments.items())):
        ax = fig.add_subplot(gs[row, 0])
        d, vx = sample_line_profile(sxx, p0, p1, width=width)
        _, vy = sample_line_profile(syy, p0, p1, width=width)
        ax.plot(d, vx, label=r"$\sigma_{xx}$")
        ax.plot(d, vy, label=r"$\sigma_{yy}$")
        ax.set_xlabel("distance (px)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Line {lid}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    try:
        fig.suptitle(title, fontsize=13)
    except Exception:
        pass
    return fig