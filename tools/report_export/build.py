"""Build Fast4D reports (PDF / DOCX / PPTX) from per-channel / full-figure assets."""
from __future__ import annotations

from pathlib import Path

from .assets import (
    iter_report_images,
    load_manifest,
    prepare_export_assets,
)
from . import writers as W


def build_report(
    *,
    scans: list | None = None,
    assets_dir: Path | str | None = None,
    out_path: Path | str,
    fmt: str = "pdf",
    title: str = "Fast4D Strain / Stress Report",
    template: Path | str | None = None,
    dpi: int = 150,
    include_maps: bool = True,
    include_calib: bool = True,
    include_reports: bool = True,
) -> Path:
    """Prepare assets (if ``scans`` given) and write the chosen format.

    ``fmt`` ∈ {pdf, docx, pptx}. Never PIL-crops composite strain figures.
    """
    fmt = (fmt or "pdf").lower().lstrip(".")
    out_path = Path(out_path)
    if assets_dir is None:
        assets_dir = out_path.parent / "export_assets"
    assets_dir = Path(assets_dir)

    if scans:
        manifest = prepare_export_assets(
            scans, assets_dir, dpi=dpi,
            include_maps=include_maps,
            include_calib=include_calib,
            include_reports=include_reports,
        )
    else:
        manifest = load_manifest(assets_dir)

    rows = iter_report_images(manifest)
    if not rows:
        raise RuntimeError(
            "No export assets found. Compute strain (and optionally run calibrations / "
            "Send to Report), then export again.")

    if fmt == "pdf":
        if out_path.suffix.lower() != ".pdf":
            out_path = out_path.with_suffix(".pdf")
        return W.write_pdf(rows, out_path, title=title)
    if fmt == "docx":
        if out_path.suffix.lower() != ".docx":
            out_path = out_path.with_suffix(".docx")
        return W.write_docx(rows, out_path, title=title)
    if fmt == "pptx":
        if out_path.suffix.lower() != ".pptx":
            out_path = out_path.with_suffix(".pptx")
        tpl = Path(template) if template else None
        return W.write_pptx(rows, out_path, title=title, template=tpl)
    raise ValueError(f"Unsupported format: {fmt!r} (use pdf, docx, or pptx)")
