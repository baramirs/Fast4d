"""crystal_frame -- one declared, verifiable frame for the strain tensor.

The SAME file lives in Fast4D (``Fast4d/crystal_frame.py``) and in gpa4d
(``gpa4d/crystal_frame.py``). Keep them identical: it is the contract between
the two programs. Pure numpy, no GUI, no py4DSTEM.

The standard
------------
A strain map is only comparable with another one if both are written in the
same axes. Neither an angle typed by hand nor the pixel axes of a detector are
a frame; a crystal direction is. So every result carries a *frame card*::

    exx   normal strain along the crystal direction x_hkl   (e.g. [-110])
    eyy   normal strain along the crystal direction y_hkl   (e.g. [001])
    exy   tensor shear (symmetrised, NOT engineering), with y taken as
          x rotated +90 deg in the SCAN frame: row index down -> column
          index right, beam into the screen.

Why the handedness is tied to the scan and not to the crystal: in a 2mm
projection such as Si[110] the pattern cannot tell +[001] from -[001]
(the mirror is a symmetry of the projected lattice), so "crystal handedness"
is not measurable from the data. The scan is the common physical reference
of both techniques, so the sign of exy is defined there. exx and eyy do not
depend on it at all; they do not depend on the sign of x either (a 180 deg
turn leaves the whole tensor unchanged).

How each program meets the standard
-----------------------------------
* Fast4D (py4DSTEM): x is NOT typed. It is the combination of the fitted
  reference vectors that points along x_hkl (for Si[110] with
  g1 = (-1 1 1), g2 = (1 -1 1): x = g1 - g2 = (-2 2 0), y = g1 + g2 = (0 0 2)),
  and ``coordinate_rotation = atan2(x_qy, x_qx)``. py4DSTEM puts y at +90 deg of
  x in calibrated (qx, qy), and the QR calibration makes (qx, qy) parallel to
  the scan (rx, ry), so its handedness to the scan is +1 -- valid only if the
  QR rotation / flip are calibrated; the card records them.
* gpa4d: x is the carrier g(x_hkl) of the indexed FFT; y is +90 deg of x in
  the image (row, col). The image reaches the scan through the placement, whose
  flip is the handedness to the scan.
* To go from a program's native tensor to the standard one:
  rotate the axes by (x_angle_true - x_angle_applied) and multiply exy by the
  handedness. ``to_canonical`` does exactly that.
"""
from __future__ import annotations

import numpy as np

SCHEMA = "crystal_frame/1"

#: default for Si / diamond seen along [110] with the usual {111} basis
DEFAULT_SPEC = dict(mode="crystal", zone=[1, 1, 0], x_hkl=[-1, 1, 0], y_hkl=[0, 0, 1],
                    g1_hkl=[-1, 1, 1], g2_hkl=[1, -1, 1])

CONVENTION = ("exx along x_hkl, eyy along y_hkl; exy = tensor shear with y = x "
              "rotated +90 deg in the scan frame (row down -> column right, beam into "
              "the screen); strain as a fraction")

#: an angle mismatch below this is rounding, above it the frames differ
ANGLE_TOL_DEG = 0.5


# ------------------------------------------------------------------ helpers
def _v(hkl, B=None):
    """Cartesian direction of a reciprocal vector (hkl). B: 3x3 reciprocal
    basis (rows a*, b*, c*); None = cubic, where (hkl) is parallel to [hkl]."""
    v = np.asarray(hkl, float).ravel()[:3]
    return v if B is None else v @ np.asarray(B, float)


def hkl_str(hkl):
    return "".join(f"{int(round(x))}" if x >= 0 else f"-{int(round(-x))}"
                   for x in np.asarray(hkl, float).ravel()[:3])


def _ang(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _between(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _line_diff(a_deg, b_deg):
    """Smallest difference between two LINES (mod 180), signed, in (-90, 90]."""
    d = (float(a_deg) - float(b_deg) + 90.0) % 180.0 - 90.0
    return 90.0 if d == -90.0 else d


def combo(target_hkl, g1_hkl, g2_hkl, B=None):
    """(a, b, residual): a*g1 + b*g2 parallel-and-equal to target_hkl.
    residual is relative; > 1e-6 means the target is not in the g1,g2 plane."""
    M = np.column_stack([_v(g1_hkl, B), _v(g2_hkl, B)])
    t = _v(target_hkl, B)
    ab, *_ = np.linalg.lstsq(M, t, rcond=None)
    res = float(np.linalg.norm(M @ ab - t) / max(np.linalg.norm(t), 1e-12))
    return float(ab[0]), float(ab[1]), res


def check_spec(spec, B=None):
    """Problems with a frame spec, as a list of strings (empty = fine)."""
    out = []
    z = np.asarray(spec["zone"], float)
    for k in ("x_hkl", "y_hkl", "g1_hkl", "g2_hkl"):
        if k in spec and spec[k] is not None:
            if abs(float(np.asarray(spec[k], float) @ z)) > 1e-9:
                out.append(f"{k} ({hkl_str(spec[k])}) is not in zone [{hkl_str(z)}]")
    a = _between(_v(spec["x_hkl"], B), _v(spec["y_hkl"], B))
    if abs(a - 90.0) > 0.01:
        out.append(f"x_hkl and y_hkl are not perpendicular ({a:.2f} deg)")
    if spec.get("g1_hkl") is not None and spec.get("g2_hkl") is not None:
        for k in ("x_hkl", "y_hkl"):
            *_ab, r = combo(spec[k], spec["g1_hkl"], spec["g2_hkl"], B)
            if r > 1e-6:
                out.append(f"{k} is not a combination of g1_hkl, g2_hkl")
    return out


# ------------------------------------------------------ frame from a lattice
def frame_from_g(g1_qxy, g2_qxy, spec=None, B=None):
    """Where x_hkl and y_hkl point in a (qx, qy) plane, from the MEASURED
    reference vectors g1, g2 and their hkl labels.

    Returns angle_deg (x in (qx, qy), counter-clockwise from qx, i.e. the
    py4DSTEM ``coordinate_rotation``), y_sign (+1 when +y_hkl lies at +90 deg
    of x), the combinations used, and the label check: the angle and length
    ratio of g1, g2 measured versus predicted by the hkl -- a {111} labelled
    as {002} fails it by degrees, so it is reported, not assumed.
    """
    spec = dict(DEFAULT_SPEC if spec is None else spec)
    g1 = np.asarray(g1_qxy, float).ravel()[:2]
    g2 = np.asarray(g2_qxy, float).ravel()[:2]
    a, b, rx = combo(spec["x_hkl"], spec["g1_hkl"], spec["g2_hkl"], B)
    c, d, ry = combo(spec["y_hkl"], spec["g1_hkl"], spec["g2_hkl"], B)
    if max(rx, ry) > 1e-6:
        raise ValueError("x_hkl / y_hkl are not combinations of g1_hkl, g2_hkl: "
                         + "; ".join(check_spec(spec, B)))
    xq = a * g1 + b * g2
    yq = c * g1 + d * g2
    ang = _ang(xq) % 360.0
    t = np.radians(ang)
    yhat = np.array([-np.sin(t), np.cos(t)])
    y_sign = 1 if float(yhat @ yq) >= 0 else -1
    v1, v2 = _v(spec["g1_hkl"], B), _v(spec["g2_hkl"], B)
    g_ang_pred = _between(v1, v2)
    g_ang_meas = _between(g1, g2)
    ratio_pred = float(np.linalg.norm(v1) / np.linalg.norm(v2))
    ratio_meas = float(np.linalg.norm(g1) / np.linalg.norm(g2))
    return dict(angle_deg=float(ang), y_sign=int(y_sign),
                x_combo=[a, b], y_combo=[c, d],
                x_qxy=[float(xq[0]), float(xq[1])], y_qxy=[float(yq[0]), float(yq[1])],
                ortho_err_deg=float(_between(xq, yq) - 90.0),
                g_angle_meas_deg=g_ang_meas, g_angle_pred_deg=g_ang_pred,
                g_ratio_meas=ratio_meas, g_ratio_pred=ratio_pred,
                label_err_deg=float(g_ang_meas - g_ang_pred),
                label_ok=bool(abs(g_ang_meas - g_ang_pred) < 3.0
                              and abs(ratio_meas / ratio_pred - 1.0) < 0.05))


def combo_label(ab, names=("g1", "g2")):
    """'g1 - g2' style text for a combination (direction only: the largest
    coefficient is scaled to 1)."""
    ab = np.asarray(ab, float)
    ab = ab / max(np.abs(ab).max(), 1e-12)
    parts = []
    for coef, n in zip(ab, names):
        if abs(coef) < 1e-9:
            continue
        mag = abs(coef)
        s = n if abs(mag - 1) < 1e-9 else f"{mag:g}*{n}"
        parts.append(("- " if coef < 0 else "+ ") + s)
    txt = " ".join(parts)
    return txt[2:] if txt.startswith("+ ") else txt


# ---------------------------------------------------------------- the card
def make_card(tool, spec, native_space, x_angle_true_deg, x_angle_applied_deg,
              handedness_to_scan, **extra):
    """The frame card stored next to a strain result.

    x_angle_true_deg     where x_hkl really is in the native plane (from data)
    x_angle_applied_deg  the x axis the tensor was actually computed on
    handedness_to_scan   +1 / -1: is native y at +90 deg of x also in the scan?
    """
    spec = dict(spec)
    card = dict(schema=SCHEMA, tool=str(tool), mode=spec.get("mode", "crystal"),
                zone=[int(v) for v in spec["zone"]],
                x_hkl=[int(v) for v in spec["x_hkl"]],
                y_hkl=[int(v) for v in spec["y_hkl"]],
                convention=CONVENTION, native_space=str(native_space),
                x_angle_true_deg=float(x_angle_true_deg),
                x_angle_applied_deg=float(x_angle_applied_deg),
                residual_rotation_deg=float(_line_diff(x_angle_true_deg,
                                                       x_angle_applied_deg)),
                handedness_to_scan=int(handedness_to_scan))
    for k in ("g1_hkl", "g2_hkl"):
        if spec.get(k) is not None:
            card[k] = [int(v) for v in spec[k]]
    card.update(extra)
    card["canonical"] = bool(abs(card["residual_rotation_deg"]) <= ANGLE_TOL_DEG
                             and card["handedness_to_scan"] == 1)
    return card


def compatible(card_a, card_b):
    """Same zone and the same x / y LINES (sign of a direction is irrelevant).
    Returns (ok, [problems])."""
    probs = []
    if card_a is None or card_b is None:
        return False, ["one of the results has no frame card"]
    za, zb = np.asarray(card_a["zone"], float), np.asarray(card_b["zone"], float)
    if min(_between(za, zb), _between(za, -zb)) > 1e-3:
        probs.append(f"zone [{hkl_str(za)}] vs [{hkl_str(zb)}]")
    for k in ("x_hkl", "y_hkl"):
        a, b = np.asarray(card_a[k], float), np.asarray(card_b[k], float)
        if min(_between(a, b), _between(a, -b)) > 1e-3:
            probs.append(f"{k[0]}: [{hkl_str(a)}] vs [{hkl_str(b)}]")
    return (not probs), probs


# ------------------------------------------------------------ the tensor
def rotate_axes(exx, eyy, exy, dphi_deg):
    """Components in axes turned by dphi FROM x TOWARDS y (the native y).
    Identical to py4DSTEM get_rotated_strain_map with the x axis moved by
    +dphi, so rotate_axes(E, cr) == get_strain(coordinate_rotation=cr) of
    the unrotated map."""
    t = np.radians(float(dphi_deg))
    c, s = np.cos(t), np.sin(t)
    xx = c * c * exx + 2 * c * s * exy + s * s * eyy
    yy = s * s * exx - 2 * c * s * exy + c * c * eyy
    xy = c * s * (eyy - exx) + (c * c - s * s) * exy
    return xx, yy, xy


def to_canonical(exx, eyy, exy, card):
    """Native tensor -> standard frame, using the card: turn the axes by the
    residual rotation, then fix the handedness of exy to the scan."""
    d = float(card.get("residual_rotation_deg", 0.0))
    h = int(card.get("handedness_to_scan", 1))
    if abs(d) > 1e-9:
        exx, eyy, exy = rotate_axes(exx, eyy, exy, d)
    return exx, eyy, exy * h


def describe(card):
    """Short multi-line text of a card."""
    if not card:
        return "no frame card"
    L = [f"{card.get('tool', '?')}: x || [{hkl_str(card['x_hkl'])}], "
         f"y || [{hkl_str(card['y_hkl'])}], zone [{hkl_str(card['zone'])}]  "
         f"(mode {card.get('mode')})",
         f"  x in {card.get('native_space')}: true {card['x_angle_true_deg']:.2f} deg, "
         f"applied {card['x_angle_applied_deg']:.2f} deg -> residual "
         f"{card['residual_rotation_deg']:+.2f} deg",
         f"  handedness to the scan {card['handedness_to_scan']:+d}"]
    if "x_def" in card:
        L.append(f"  x = {card['x_def']},  y = {card.get('y_def', '?')}")
    if "label_err_deg" in card:
        L.append(f"  g1^g2 measured {card.get('g_angle_meas_deg', float('nan')):.2f} deg vs "
                 f"{card.get('g_angle_pred_deg', float('nan')):.2f} deg from the hkl "
                 f"({'labels OK' if card.get('label_ok') else 'LABELS DO NOT FIT'})")
    if card.get("assumptions"):
        L.append("  assumes: " + "; ".join(card["assumptions"]))
    return "\n".join(L)
