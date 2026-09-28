"""Read EMD/HDF5 *metadata only* — never the 4D payload.

Thermo Fisher Velox ``.emd`` files store STEM-image pixel size (scan step, R)
as ``BinaryResult.PixelSize`` in **metres**. Merlin / py4DSTEM Q-pixel is
reciprocal space (Å⁻¹/px) and must not be used as a real-space ruler.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

_MAX_META_BYTES = 256_000
_STEM_DETECTORS = ("haadf", "adf", "bf", "df4", "df2", "abf", "maidf")
_CAMERA_HINTS = ("merlin", "medipix", "ceta", "falcon", "quad", "ariva")
_R_KEY_HINTS = ("r_pixel", "pixelsize", "pixel_size", "pixel size", "scanstep", "scan_step")
_Q_KEY_HINTS = ("q_pixel", "a^-1", "a-1", "angstrom^-1", "reciprocal")


def extract_emd_metadata(path: str | Path) -> dict[str, Any]:
    """Walk an EMD/HDF5 file and collect pixel-size / detector metadata.

    Returns a JSON-serialisable report. Large datasets are listed by shape only.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)

    hits: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []

    with h5py.File(p, "r") as h5:
        def visit(name: str, obj) -> None:
            if isinstance(obj, h5py.Group):
                nodes.append({"path": name, "kind": "group",
                              "attrs": _attrs_preview(obj)})
                hits.extend(_hits_from_attrs(name, obj, kind_guess=_guess_group_kind(name, obj)))
                return
            if not isinstance(obj, h5py.Dataset):
                return
            shape = tuple(int(s) for s in obj.shape)
            nbytes = int(np.prod(shape) if shape else 1) * int(obj.dtype.itemsize)
            nodes.append({"path": name, "kind": "dataset", "shape": shape,
                          "dtype": str(obj.dtype), "nbytes": nbytes,
                          "attrs": _attrs_preview(obj)})
            hits.extend(_hits_from_attrs(name, obj, kind_guess=_guess_dataset_kind(name, shape)))
            leaf = name.split("/")[-1].lower()
            if nbytes <= _MAX_META_BYTES and (
                leaf in {"metadata", "version"} or "metadata" in leaf
            ):
                parsed, nread = _try_parse_small_payload(obj)
                if parsed is not None:
                    hits.extend(_hits_from_json(name, parsed, bytes_read=nread))

        h5.visititems(visit)
        hits.extend(_hits_from_attrs("/", h5, kind_guess="root"))

    kind = _classify_file(nodes, hits)
    rec = _recommend(hits, kind)
    return {
        "path": str(p),
        "kind": kind,
        "loaded_4d_payload": False,
        "image_shape": image_shape_from_nodes(nodes),
        "nodes": nodes,
        "hits": hits,
        "recommendation": rec,
    }


_EMD_SUFFIXES = (".emd", ".h5", ".hdf5")


def collect_emd_paths(
    sources: str | Path | list[str | Path],
    *,
    recursive: bool = True,
    suffixes: tuple[str, ...] = _EMD_SUFFIXES,
) -> list[Path]:
    """Resolve files from one path, a folder, or a list of either."""
    if isinstance(sources, (str, Path)):
        items = [sources]
    else:
        items = list(sources)
    found: list[Path] = []
    seen: set[Path] = set()
    for raw in items:
        p = Path(raw).expanduser()
        if p.is_file() and p.suffix.lower() in suffixes:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                found.append(rp)
            continue
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            for child in sorted(it):
                if child.is_file() and child.suffix.lower() in suffixes:
                    rp = child.resolve()
                    if rp not in seen:
                        seen.add(rp)
                        found.append(rp)
    return found


def _as_name_tokens(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v) for v in value if str(v).strip()]


def filter_emd_paths(
    paths: list[str | Path],
    *,
    contains: str | list[str] | tuple[str, ...] | None = None,
    exclude: str | list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    """Keep files whose *name* matches ``contains`` and does not match ``exclude``.

    Tokens are case-insensitive substrings. A list means OR. Empty ``contains``
    keeps every path (then ``exclude`` still applies).
    """
    keep = [t.lower() for t in _as_name_tokens(contains)]
    drop = [t.lower() for t in _as_name_tokens(exclude)]
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        name = p.name.lower()
        if keep and not any(t in name for t in keep):
            continue
        if drop and any(t in name for t in drop):
            continue
        out.append(p)
    return out


def image_shape_from_nodes(nodes: list[dict[str, Any]]) -> tuple[int, int] | None:
    """Largest STEM / map image (Ny, Nx). Trailing singleton dims are dropped."""
    preferred: list[tuple[int, tuple[int, int]]] = []
    fallback: list[tuple[int, tuple[int, int]]] = []
    for n in nodes:
        if n.get("kind") != "dataset":
            continue
        path = str(n.get("path") or "").lower()
        if "metadata" in path or "thumbnail" in path or path.endswith(".jpg"):
            continue
        raw_sh = tuple(int(s) for s in (n.get("shape") or ()))
        sh = tuple(s for s in raw_sh if s > 1)
        if len(sh) < 2:
            continue
        ny, nx = int(sh[0]), int(sh[1])
        area = ny * nx
        pair = (area, (ny, nx))
        leaf = path.rstrip("/").split("/")[-1]
        if "data/image" in path and leaf == "data":
            preferred.append(pair)
        elif len(raw_sh) == 4:
            fallback.append((int(raw_sh[0]) * int(raw_sh[1]),
                             (int(raw_sh[0]), int(raw_sh[1]))))
        elif len(sh) == 2 and min(ny, nx) >= 8:
            fallback.append(pair)
    pool = preferred or fallback
    if not pool:
        return None
    return max(pool)[1]


def scan_emd_sources(
    sources: str | Path | list[str | Path],
    *,
    recursive: bool = True,
    suffixes: tuple[str, ...] = _EMD_SUFFIXES,
) -> list[dict[str, Any]]:
    """Extract metadata for every EMD/HDF5 in ``sources`` (files and/or folders)."""
    out = []
    for p in collect_emd_paths(sources, recursive=recursive, suffixes=suffixes):
        try:
            out.append(extract_emd_metadata(p))
        except Exception as exc:
            out.append({"path": str(p), "kind": "error", "error": str(exc),
                        "loaded_4d_payload": False, "hits": [],
                        "recommendation": None})
    return out


def scan_emd_folder(folder: str | Path, *, suffixes=(".emd", ".h5", ".hdf5"),
                    recursive: bool = False) -> list[dict[str, Any]]:
    """Extract metadata for every EMD/HDF5 under ``folder``."""
    return scan_emd_sources(folder, recursive=recursive, suffixes=tuple(suffixes))


def comparison_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per file — the STEM R recommendation, for side-by-side compare."""
    rows = []
    for r in reports:
        rec = r.get("recommendation") or {}
        shape = r.get("image_shape")
        if shape is None:
            shape = image_shape_from_nodes(r.get("nodes") or [])
        ny = int(shape[0]) if shape else None
        nx = int(shape[1]) if shape else None
        nm = rec.get("nm_per_px")
        fov = None
        if nx is not None and nm is not None:
            try:
                fov = float(nx) * float(nm)   # PixelSize.width → FOV in x
            except (TypeError, ValueError):
                fov = None
        rows.append({
            "file": Path(r.get("path", "")).name,
            "kind": r.get("kind"),
            "shape": f"{ny}×{nx}" if ny is not None else None,
            "ny": ny,
            "nx": nx,
            "pm_per_px": rec.get("pm_per_px"),
            "nm_per_px": rec.get("nm_per_px"),
            "fov_nm": fov,
            "source": rec.get("source_kind"),
            "key": rec.get("key"),
            "error": r.get("error"),
            "path": r.get("path"),
        })
    return rows


def hits_as_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flat rows for a pandas table / notebook display."""
    rows = []
    for h in report.get("hits") or []:
        rows.append({
            "file": Path(report.get("path", "")).name,
            "kind": report.get("kind"),
            "group": h.get("group"),
            "source": h.get("source_kind"),
            "key": h.get("key"),
            "role": h.get("role"),
            "raw": h.get("raw"),
            "nm_per_px": h.get("nm_per_px"),
            "pm_per_px": h.get("pm_per_px"),
            "note": h.get("note"),
        })
    return rows


# ── internals ────────────────────────────────────────────────────────────────

def _attrs_preview(obj) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in obj.attrs.items():
        out[str(k)] = _jsonable(v)
    return out


def _jsonable(v: Any) -> Any:
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return v.hex()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        if v.size > 16:
            return f"ndarray{tuple(v.shape)}"
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _guess_group_kind(name: str, obj) -> str:
    low = name.lower()
    if "data/image" in low or low.startswith("data/image"):
        return "stem_image"
    py = str(obj.attrs.get("python_class", "") or "")
    if py == "DataCube" or str(obj.attrs.get("emd_group_type", "")) == "array":
        return "4d_cube"
    if "calibrat" in low:
        return "calibration"
    return "group"


def _guess_dataset_kind(name: str, shape: tuple[int, ...]) -> str:
    low = name.lower()
    if len(shape) == 4:
        return "4d_cube"
    if "data/image" in low and len(shape) == 2:
        return "stem_image"
    if "calibrat" in low:
        return "calibration"
    return "dataset"


def _hits_from_attrs(group: str, obj, *, kind_guess: str) -> list[dict[str, Any]]:
    hits = []
    for k, v in obj.attrs.items():
        key = str(k)
        hit = _hit_from_key_value(group, key, v, source_kind=kind_guess)
        if hit is not None:
            hits.append(hit)
    return hits


def _hits_from_json(group: str, data: Any, *, bytes_read: int) -> list[dict[str, Any]]:
    source = "stem_image" if "data/image" in group.lower() else "metadata"
    hits = []
    for key, val in _walk_json(data):
        hit = _hit_from_key_value(group, key, val, source_kind=source)
        if hit is None:
            continue
        hit["raw_bytes_read"] = int(bytes_read)
        det = _detector_from_json(data)
        if det:
            hit["note"] = (hit.get("note") or "") + f" detector={det}"
            if any(s in det.lower() for s in _STEM_DETECTORS):
                hit["source_kind"] = "stem_image"
        hits.append(hit)
    # Always surface detector / kV even without a pixel-size key
    det = _detector_from_json(data)
    if det and not any(h.get("key", "").endswith("Detector") for h in hits):
        hits.append({
            "group": group, "key": "BinaryResult.Detector", "raw": det,
            "role": "info", "source_kind": "stem_image",
            "nm_per_px": None, "pm_per_px": None,
            "raw_bytes_read": int(bytes_read),
            "note": f"STEM detector={det} (microscope image, not Merlin Q)",
        })
    kv = _nested_get(data, ("Optics", "AccelerationVoltage"))
    if kv is not None:
        hits.append({
            "group": group, "key": "Optics.AccelerationVoltage", "raw": kv,
            "role": "info", "source_kind": source,
            "nm_per_px": None, "pm_per_px": None,
            "raw_bytes_read": int(bytes_read),
            "note": "high tension (V)",
        })
    return hits


def _walk_json(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                out.extend(_walk_json(v, key))
            else:
                out.append((key, v))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:32]):
            out.extend(_walk_json(v, f"{prefix}[{i}]"))
    return out


def _nested_get(data: Any, path: tuple[str, ...]) -> Any:
    cur = data
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _detector_from_json(data: Any) -> str | None:
    for path in (("BinaryResult", "Detector"), ("Detector",), ("Instrument", "Detector")):
        v = _nested_get(data, path)
        if v:
            return str(v)
    return None


def _hit_from_key_value(group: str, key: str, raw: Any, *, source_kind: str) -> dict[str, Any] | None:
    key_l = key.lower().replace(" ", "")
    blob = f"{group} {key}".lower()
    role = _role_for_key(key_l, blob)
    if role == "skip":
        return None
    nm, pm, note = _to_nm_pm(raw, key=key, role=role)
    if role == "unknown" and nm is None:
        return None
    if any(h in blob for h in _CAMERA_HINTS) and role == "R":
        note = (note + " " if note else "") + "camera-related key — check before using as scan step"
    return {
        "group": group,
        "key": key,
        "raw": _jsonable(raw),
        "role": role,
        "source_kind": source_kind,
        "nm_per_px": nm,
        "pm_per_px": pm,
        "raw_bytes_read": 0,
        "note": note or "",
    }


def _role_for_key(key_l: str, blob: str) -> str:
    if any(h in key_l or h in blob for h in _Q_KEY_HINTS):
        return "Q"
    if any(h in blob for h in _CAMERA_HINTS):
        return "Q"
    if "pixelsize" in key_l or "pixel_size" in key_l or key_l.endswith("width") or key_l.endswith("height"):
        if "pixel" in key_l or "pixelsize" in blob:
            return "R"
    if any(h in key_l for h in _R_KEY_HINTS):
        return "R"
    if key_l.endswith("width") or key_l.endswith("height"):
        if "pixel" in blob:
            return "R"
    return "unknown"


def _to_nm_pm(raw: Any, *, key: str, role: str) -> tuple[float | None, float | None, str]:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None, None, ""
    if not np.isfinite(val) or val <= 0:
        return None, None, ""
    units = ""
    key_l = key.lower()
    if role == "Q":
        return None, None, f"Q-space value={val:g} (Å⁻¹/px typical — not a real-space ruler)"
    # Velox BinaryResult.PixelSize is metres. 1e-12..1e-7 m → 0.001..100 nm.
    if "pixelsize" in key_l.replace(".", "") or "pixel_size" in key_l:
        if 1e-13 < abs(val) < 1e-6:
            nm = val * 1e9
            return nm, nm * 1000.0, "Velox PixelSize in metres → scan step"
    # Already nm (py4DSTEM R_pixel_size)
    if "r_pixel" in key_l:
        unit_hint = "nm"
        if 1e-4 < abs(val) < 50:
            return val, val * 1000.0, f"R_pixel_size as {unit_hint}/px"
        if 0.05 < abs(val) < 5e4:
            # likely pm already
            return val / 1000.0, val, "R_pixel_size treated as pm/px"
    if 1e-13 < abs(val) < 1e-6:
        nm = val * 1e9
        return nm, nm * 1000.0, "value looks like metres → nm/px"
    if 1e-4 < abs(val) < 20:
        return val, val * 1000.0, "value looks like nm/px"
    return None, None, f"numeric {val:g} (units unclear)"


def _try_parse_small_payload(ds) -> tuple[Any | None, int]:
    try:
        nbytes = int(ds.size) * int(ds.dtype.itemsize)
        if nbytes > _MAX_META_BYTES:
            return None, 0
        raw = ds[()]
    except Exception:
        return None, 0
    nread = int(getattr(raw, "nbytes", nbytes))
    text = _decode_bytes(raw)
    if text is None:
        return None, nread
    text = text.strip().rstrip("\x00")
    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text), nread
        except json.JSONDecodeError:
            return None, nread
    return text, nread


def _decode_bytes(raw: Any) -> str | None:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        buf = bytes(raw)
    elif isinstance(raw, np.ndarray):
        buf = raw.tobytes()
    else:
        return None
    for enc in ("utf-8", "utf-16-le", "utf-16"):
        try:
            return buf.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def _classify_file(nodes: list[dict], hits: list[dict]) -> str:
    paths = " ".join(n.get("path", "") for n in nodes).lower()
    if "data/image" in paths or any("velox" in str(n.get("attrs", {})).lower() for n in nodes):
        return "velox"
    if any("emd_group_type" in (n.get("attrs") or {}) for n in nodes):
        return "py4dstem"
    if any("merlin" in str(n).lower() for n in nodes + hits):
        return "merlin_related"
    return "hdf5"


def _recommend(hits: list[dict], kind: str) -> dict[str, Any] | None:
    r_hits = [h for h in hits if h.get("role") == "R" and h.get("nm_per_px") is not None]
    if not r_hits:
        return None

    def score(h: dict) -> int:
        s = 0
        if h.get("source_kind") == "stem_image":
            s += 50
        if "pixelsize" in str(h.get("key", "")).lower().replace(".", ""):
            s += 20
        if kind == "velox":
            s += 10
        if h.get("source_kind") == "4d_cube":
            s -= 5
        if "camera-related" in (h.get("note") or ""):
            s -= 30
        return s

    best = max(r_hits, key=score)
    det_notes = [h.get("note", "") for h in hits if "detector=" in (h.get("note") or "").lower()]
    det = ""
    for n in det_notes:
        if "HAADF" in n.upper() or "ADF" in n.upper() or "BF" in n.upper():
            det = n
            break
    note = (
        f"STEM scan step from {best.get('source_kind')} "
        f"({best.get('key')}). This is the microscope raster, not Merlin Q-space."
    )
    if det:
        note = note.replace("STEM scan step", f"STEM scan step ({det})")
    return {
        "role": "R",
        "nm_per_px": float(best["nm_per_px"]),
        "pm_per_px": float(best["pm_per_px"]),
        "source_kind": best.get("source_kind"),
        "key": best.get("key"),
        "group": best.get("group"),
        "note": note,
    }
