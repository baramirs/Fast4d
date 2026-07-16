"""Apply a BasisProposal into Fast4D scan.params (Send contract)."""
from __future__ import annotations

from typing import Any

from plugins.indexing.types import BasisProposal


def apply_proposal_to_scan(scan: Any, proposal: BasisProposal, *, log=None) -> None:
    """Write proposed index_origin/g1/g2 (+ optional QR/coord) into ``scan.params``."""
    if proposal is None:
        raise RuntimeError(f"[{getattr(scan, 'name', '?')}] no BasisProposal to apply")
    p = scan.params
    p.index_origin = int(proposal.index_origin)
    p.index_g1 = int(proposal.index_g1)
    p.index_g2 = int(proposal.index_g2)
    p.basis_manual_enabled = True
    if proposal.suggested_qr_rotation_deg is not None:
        p.qr_rotation = float(proposal.suggested_qr_rotation_deg)
    if proposal.suggested_coordinate_rotation_deg is not None:
        try:
            p.coordinate_rotation = float(proposal.suggested_coordinate_rotation_deg)
        except Exception:
            pass
    # Keep last raw result on scan for Compare / Report when available
    raw = proposal.raw_result
    if raw is not None:
        plugin_id = str(proposal.plugin_id)
        if plugin_id.startswith("index_bvm"):
            scan.indexing_result = raw
        elif plugin_id.startswith("orient"):
            scan.orientation_peaks_result = raw
    if log is not None:
        log(
            f"[{getattr(scan, 'name', '?')}] basis indices from {proposal.plugin_id}: "
            f"origin={p.index_origin} g1={p.index_g1} g2={p.index_g2}"
        )


def proposal_from_indexing_result(result: Any, *, plugin_id: str) -> BasisProposal:
    return BasisProposal(
        plugin_id=plugin_id,
        index_origin=int(result.index_origin),
        index_g1=int(result.index_g1),
        index_g2=int(result.index_g2),
        g1_px=result.g1_px,
        g2_px=result.g2_px,
        metrics=dict(getattr(result, "metrics", {}) or {}),
        suggested_qr_rotation_deg=None,
        suggested_coordinate_rotation_deg=None,
        raw_result=result,
        figure_key="indexing",
    )


def proposal_from_orientation_result(result: Any, *, plugin_id: str = "orient_peaks") -> BasisProposal:
    return BasisProposal(
        plugin_id=plugin_id,
        index_origin=int(result.index_origin),
        index_g1=int(result.index_g1),
        index_g2=int(result.index_g2),
        g1_px=result.g1_px,
        g2_px=result.g2_px,
        metrics=dict(getattr(result, "metrics", {}) or {}),
        suggested_qr_rotation_deg=getattr(result, "suggested_qr_rotation_deg", None),
        suggested_coordinate_rotation_deg=getattr(
            result, "suggested_coordinate_rotation_deg", None
        ),
        raw_result=result,
        figure_key="orientation_peaks",
    )
