"""Index BVM Known — RANSAC + absolute hkl via zone + real axes + QR."""
from __future__ import annotations

from plugins.indexing.apply import proposal_from_indexing_result
from plugins.indexing.peaks import find_peaks
from plugins.indexing.types import BasisProposal, BvmContext


class IndexBvmKnownPlugin:
    id = "index_bvm_known"
    label = "Index BVM (Known)"
    description = (
        "Same RANSAC lattice as Unknown, then absolute Miller indices using "
        "zone [uvw] + real axes H/V + QR (breaks ±g / Friedel ambiguity)."
    )

    def run(self, ctx: BvmContext, *, log=None) -> BasisProposal:
        import bvm_indexing as bix

        if not ctx.zone_axis:
            raise ValueError("Index BVM Known requires zone_axis [uvw]")
        maxima = find_peaks(
            ctx.bvm,
            min_spacing=ctx.min_spacing,
            min_absolute_intensity=ctx.min_absolute_intensity,
            max_num_peaks=ctx.max_num_peaks,
            edge_boundary=ctx.edge_boundary,
            image_upsample=int(ctx.image_upsample),
        )
        result = bix.index_bvm(
            ctx.bvm,
            ctx.origin_px,
            Q_pixel=float(ctx.Q_pixel),
            Q_units=str(ctx.Q_units),
            lattice_a=float(ctx.lattice_a if ctx.lattice_a is not None else 5.4309),
            zone_axis=ctx.zone_axis,
            real_axis_h=ctx.real_axis_h or [0, 0, -1],
            real_axis_v=ctx.real_axis_v or [-1, 1, 0],
            qr_rotation_deg=float(ctx.qr_rotation_deg),
            qr_flip=bool(ctx.qr_flip),
            tol_px=float(ctx.tol_px),
            seed=int(ctx.seed),
            min_spacing=float(ctx.min_spacing),
            min_absolute_intensity=float(ctx.min_absolute_intensity),
            max_num_peaks=int(ctx.max_num_peaks),
            edge_boundary=float(ctx.edge_boundary),
            maxima=maxima,
            orientation_mode="known",
        )
        if log is not None:
            log(
                f"Index BVM Known: {result.n_inliers}/{len(result.peaks)} inliers; "
                f"anchored={result.metrics.get('anchored')}"
            )
        return proposal_from_indexing_result(result, plugin_id=self.id)
