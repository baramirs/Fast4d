"""Load calibration / strain export files (params.json, params.yaml) to a dict."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

# Filled on each load_params_dict (for GUI report): sibling yaml merged into json, etc.
LAST_LOAD_EXTRAS: str = ""


def _iter_sibling_params_yaml(json_dir: Path) -> list[Path]:
    """
    Same folder as the selected .json. Prefer params.yaml, but accept *any* .yaml
    that still looks like a notebook export (has origin or center_guess) so
    renames (e.g. my_params.yaml) still work.
    """
    if not json_dir.is_dir():
        return []
    def sort_key(p: Path) -> tuple:
        n = p.name.lower()
        prefer = 0 if n in ("params.yaml", "params.yml") else 1
        return (prefer, n, p.name)

    found: list[Path] = []
    for child in json_dir.iterdir():
        if not child.is_file() or child.suffix.lower() not in (".yaml", ".yml"):
            continue
        try:
            head = child.read_text(encoding="utf-8", errors="replace")[:48_000]
        except OSError:
            continue
        if "center_guess" not in head and "origin:" not in head:
            continue
        found.append(child)
    return sorted(found, key=sort_key)


def _merge_sibling_params_yaml(json_path: Path, data: dict[str, Any]) -> dict[str, Any]:
    """
    If the user picks gui_results/.../params.json (often without `origin`), look for
    params.yaml in the *same folder* (notebook export) and merge origin + pixel_size.
    """
    global LAST_LOAD_EXTRAS
    if _pair_int_from_parsed(((data.get("origin") or {}) or {}).get("center_guess")) is not None:
        return data
    sides = _iter_sibling_params_yaml(json_path.parent)
    if not sides:
        LAST_LOAD_EXTRAS = (
            f"No companion .yaml next to {json_path.name!r} with origin/center_guess "
            f"(looked in {json_path.parent})."
        )
        return data
    for side in sides:
        t2 = side.read_text(encoding="utf-8", errors="replace")
        try:
            import yaml  # type: ignore

            d2 = yaml.safe_load(t2)
        except Exception:
            d2 = {}
        if not isinstance(d2, dict):
            d2 = {}
        d2 = enrich_parsed_from_raw_text(t2, d2)
        out: dict[str, Any] = {**data}
        so = (d2.get("origin") or {}) or {}
        cg = so.get("center_guess")
        pair = _pair_int_from_parsed(cg) or extract_center_guess_yx_from_raw_text(t2)
        if pair is not None:
            o = {**((out.get("origin") or {}) or {}), "center_guess": [pair[0], pair[1]]}
            out["origin"] = o
        spx = (d2.get("pixel_size") or {}) or {}
        if spx:
            opx = (out.get("pixel_size") or {}) or {}
            merged_px = {**opx}
            for k, v in spx.items():
                if v is not None and merged_px.get(k) is None:
                    merged_px[k] = v
            out["pixel_size"] = merged_px
        origin_ok = _pair_int_from_parsed(((out.get("origin") or {}) or {}).get("center_guess")) is not None
        px_before = (data.get("pixel_size") or {}) or {}
        px_after = (out.get("pixel_size") or {}) or {}
        if origin_ok or px_after != px_before:
            LAST_LOAD_EXTRAS = (
                f"Note: merged from {side.name!r} next to {json_path.name!r} (JSON was missing that data)."
            )
            return out
        LAST_LOAD_EXTRAS = (
            f"Found {side.name!r} but could not read center_guess; check YAML layout next to {json_path.name!r}."
        )
    return data


def load_params_dict(path: str | Path) -> dict[str, Any]:
    global LAST_LOAD_EXTRAS
    LAST_LOAD_EXTRAS = ""
    p = Path(path).expanduser()
    if p.is_file():
        p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    text = p.read_text(encoding="utf-8")
    ext = p.suffix.lower()
    if ext in (".yaml", ".yml"):
        data = _load_yaml_str(text, path_label=str(p))
    elif ext == ".json" or text.lstrip().startswith(("{", "[")):
        data = json.loads(text)
    else:
        try:
            data = json.loads(text)
        except Exception:
            data = _load_yaml_str(text, path_label=str(p))
    if not isinstance(data, dict):
        return data
    data = enrich_parsed_from_raw_text(text, data)
    if p.suffix.lower() == ".json":
        data = _merge_sibling_params_yaml(p, data)
    return normalize_params_for_import(data)


def _pair_int_from_parsed(cg) -> tuple[int, int] | None:
    if cg is None:
        return None
    if isinstance(cg, (list, tuple)) and len(cg) == 2:
        try:
            return int(cg[0]), int(cg[1])
        except (TypeError, ValueError):
            return None
    return None


def extract_center_guess_yx_from_raw_text(text: str) -> tuple[int, int] | None:
    """
    Notebook / params.yaml layout (y native, x native) as two list items:
        origin:
          center_guess:
          - 161
          - 137
    also JSON: "center_guess": [161, 137]
    """
    for pat in (
        # Windows \\r\\n and Unix \\n: list under center_guess:
        r"center_guess:\s*[\r\n]+\s*-\s*(\d+)\s*[\r\n]+\s*-\s*(\d+)",
        r"center_guess:\s*(?:\n\s*-\s*(\d+)\s*\n\s*-\s*(\d+))",  # legacy
        # Flow-style in YAML: center_guess: [161, 137]
        r"center_guess:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]",
    ):
        m = re.search(pat, text)
        if m:
            return int(m.group(1)), int(m.group(2))
    for pat in (
        r'"center_guess"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',
        r"'center_guess'\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]",
        # one-line JSON
        r'"center_guess":\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',
    ):
        m = re.search(pat, text)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def enrich_parsed_from_raw_text(text: str, data: dict[str, Any]) -> dict[str, Any]:
    """
    YAML parsers can yield center_guess: null for the notebook's indented list form.
    The GUI / minimal params.json may omit `origin` entirely. Raw-text regex fixes both.
    """
    out: dict[str, Any] = copy.deepcopy(data)
    o = (out.get("origin") or {}) or {}
    if not _pair_int_from_parsed(o.get("center_guess")):
        from_raw = extract_center_guess_yx_from_raw_text(text)
        if from_raw:
            o2 = {**o, "center_guess": [from_raw[0], from_raw[1]]}
            out["origin"] = o2
    px = dict((out.get("pixel_size") or {}) or {})
    if px.get("pixel_size_ui_value", None) is None:
        m = re.search(r"(?m)^\s*pixel_size_ui_value:\s*([\d.eE+-]+)\s*$", text)
        if m:
            px["pixel_size_ui_value"] = float(m.group(1))
    if px.get("k_max_ui_value", None) is None and px.get("k_max", None) is None:
        m = re.search(r"(?m)^\s*k_max_ui_value:\s*([\d.eE+-]+)\s*$", text)
        if m:
            px["k_max_ui_value"] = float(m.group(1))
    if px.get("bragg_k_power", None) is None:
        m = re.search(r"(?m)^\s*bragg_k_power:\s*([\d.eE+-]+)\s*$", text)
        if m:
            px["bragg_k_power"] = float(m.group(1))
    # 4Dstrain notebook export (params dict): Q_pixel_size_calibration, k_max
    if px.get("pixel_size_ui_value", None) is None:
        m = re.search(
            r"(?m)^\s*Q_pixel_size_calibration:\s*([\d.eE+-]+|nan|inf)\s*$",
            text,
            re.IGNORECASE,
        )
        if m and m.group(1).lower() not in ("nan", "inf"):
            try:
                px["pixel_size_ui_value"] = float(m.group(1))
            except ValueError:
                pass
    if px.get("k_max_ui_value", None) is None and px.get("k_max", None) is None:
        m = re.search(r"(?m)^\s*k_max:\s*([\d.eE+-]+)\s*$", text)
        if m:
            v = float(m.group(1))
            px["k_max"] = v
            px["k_max_ui_value"] = v
    if px:
        out["pixel_size"] = px
    return out


def normalize_params_for_import(data: dict[str, Any]) -> dict[str, Any]:
    """
    Unify:
    - 4Dstrain **notebook** export: `Q_pixel_size_calibration`, `strain_settings`, …
    - **Workflow GUI** `save_final_figures_step` (pipeline.py): top-level
      `q_pixel_size`, `strain_params` / get_strain, `qr_rotation`, `qr_flip` — not the same keys
      as the import UI / report expect (`pixel_size`, `strain_settings`, `rotation`).
    """
    if not isinstance(data, dict):
        return data
    out = copy.deepcopy(data)
    # --- Q pixel: nested pixel_size, notebook calibration name, or GUI top-level
    px = dict((out.get("pixel_size") or {}) or {})
    if px.get("pixel_size_ui_value", None) is None and px.get(
        "Q_pixel_size_calibration", None
    ) is not None:
        try:
            v = float(px["Q_pixel_size_calibration"])
        except (TypeError, ValueError):
            pass
        else:
            if v == v:
                px["pixel_size_ui_value"] = v
    if px.get("pixel_size_ui_value", None) is None and out.get("q_pixel_size", None) is not None:
        try:
            v = float(out["q_pixel_size"])
        except (TypeError, ValueError):
            pass
        else:
            if v == v:
                px["pixel_size_ui_value"] = v
    if out.get("q_pixel_units") and px.get("Q_pixel_units", None) is None:
        px["Q_pixel_units"] = out["q_pixel_units"]
    if px.get("k_max_ui_value", None) is None and px.get("k_max", None) is not None:
        try:
            px["k_max_ui_value"] = float(px["k_max"])
        except (TypeError, ValueError):
            pass
    if px:
        out["pixel_size"] = px

    # Strain: GUI saves under strain_params / get_strain; notebook uses strain_settings
    sset = dict((out.get("strain_settings") or {}) or {})
    sp = out.get("strain_params")
    if isinstance(sp, dict):
        gs = (sp.get("get_strain") or {}) or {}
        for k, v in gs.items():
            if v is not None and sset.get(k) is None:
                sset[k] = v
        mps = (sp.get("set_max_peak_spacing") or {}) or {}
        if sset.get("max_peak_spacing") is None and mps.get("max_peak_spacing") is not None:
            sset["max_peak_spacing"] = mps["max_peak_spacing"]
    if sset:
        out["strain_settings"] = sset

    # QR: GUI top-level keys
    rot = dict((out.get("rotation") or {}) or {})
    if out.get("qr_rotation", None) is not None and rot.get("QR_rotation_calibration") is None:
        try:
            rot["QR_rotation_calibration"] = float(out["qr_rotation"])
        except (TypeError, ValueError):
            pass
    if out.get("qr_flip", None) is not None and rot.get("QR_flip") is None:
        rot["QR_flip"] = bool(out["qr_flip"])
    if rot:
        out["rotation"] = {**((out.get("rotation") or {}) or {}), **rot}

    # Map GUI params.json strain_basis_params into strain_basis for Step 12 import.
    sbp = out.get("strain_basis_params")
    if isinstance(sbp, dict) and isinstance(sbp.get("choose_basis_vectors"), dict):
        sb_existing = dict((out.get("strain_basis") or {}) or {})
        sbase = dict((sb_existing.get("STRAIN_BASIS_PARAMS") or {}) or {})
        sbase["choose_basis_vectors"] = sbp["choose_basis_vectors"]
        sb_existing["STRAIN_BASIS_PARAMS"] = sbase
        pick_prev = dict((sb_existing.get("STRAIN_PICK") or {}) or {})
        if sbp.get("qr_rotation") is not None:
            try:
                pick_prev["QR_rotation"] = float(sbp["qr_rotation"])
            except (TypeError, ValueError):
                pass
        if sbp.get("qr_flip") is not None:
            pick_prev["QR_flip"] = bool(sbp["qr_flip"])
        if sbp.get("manual_enabled") is not None:
            pick_prev["manual_enabled"] = bool(sbp["manual_enabled"])
        else:
            cbv0 = sbp.get("choose_basis_vectors") or {}
            if isinstance(cbv0, dict) and all(
                k in cbv0 and cbv0.get(k) is not None for k in ("index_origin", "index_g1", "index_g2")
            ):
                pick_prev["manual_enabled"] = True
        sb_existing["STRAIN_PICK"] = pick_prev
        out["strain_basis"] = sb_existing

    # Flat GUI keys (older / hand-edited JSON): fold into pixel_size for Step 11 import
    px2 = dict((out.get("pixel_size") or {}) or {})
    if out.get("bragg_k_power", None) is not None and px2.get("bragg_k_power", None) is None:
        try:
            v = float(out["bragg_k_power"])
            if v == v:
                px2["bragg_k_power"] = v
        except (TypeError, ValueError):
            pass
    if px2:
        out["pixel_size"] = px2

    return out


def _load_yaml_str(text: str, path_label: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("YAML must parse to a top-level object (dict).")
        return data
    except ImportError:
        parsed = _parse_params_export_fallback(text)
        if not parsed:
            raise RuntimeError(
                f"To read {path_label!r} install PyYAML: pip install pyyaml"
            )
        return parsed
    except Exception:
        # Corrupt YAML / odd encoding: try to salvage notebook-export fields.
        parsed = _parse_params_export_fallback(text)
        if parsed:
            return parsed
        raise


def _parse_params_export_fallback(text: str) -> dict[str, Any]:
    """
    Minimal parse when PyYAML is missing: extracts common keys from notebook exports.
    Not a full YAML parser; enough for origin + pixel_size + many flat blocks.
    """
    d: dict[str, Any] = {}
    m = re.search(
        r"center_guess:\s*[\r\n]+\s*-\s*(\d+)\s*[\r\n]+\s*-\s*(\d+)",
        text,
        re.MULTILINE,
    )
    if m:
        d.setdefault("origin", {})["center_guess"] = [int(m.group(1)), int(m.group(2))]
    pui = re.search(r"(?m)^\s*pixel_size_ui_value:\s*([\d.eE+-]+)\s*$", text)
    if pui:
        d.setdefault("pixel_size", {})["pixel_size_ui_value"] = float(pui.group(1))
    kmu = re.search(r"(?m)^\s*k_max_ui_value:\s*([\d.eE+-]+)\s*$", text)
    if kmu:
        d.setdefault("pixel_size", {})["k_max_ui_value"] = float(kmu.group(1))
    bkp = re.search(r"(?m)^\s*bragg_k_power:\s*([\d.eE+-]+)\s*$", text)
    if bkp:
        d.setdefault("pixel_size", {})["bragg_k_power"] = float(bkp.group(1))
    cr = re.search(r"(?m)^\s*coordinate_rotation:\s*([\d.eE+-]+|null)\s*$", text)
    if cr and cr.group(1) != "null":
        d.setdefault("strain_settings", {})["coordinate_rotation"] = float(cr.group(1))
    qrr = re.search(r"(?m)^\s*QR_rotation_calibration:\s*([\d.eE+-]+|null)\s*$", text)
    if qrr and qrr.group(1) != "null":
        d.setdefault("rotation", {})["QR_rotation_calibration"] = float(qrr.group(1))
    return d
