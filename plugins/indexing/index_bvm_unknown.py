"""Index BVM Unknown — RANSAC lattice → g1/g2 (no absolute Miller orientation)."""
from __future__ import annotations

from plugins.indexing.apply import proposal_from_indexing_result
from plugins.indexing.peaks import find_peaks
from plugins.indexing.types import BasisProposal, BvmContext


class IndexBvmUnknownPlugin:
    id = "index_bvm_unknown"
    label = "Index BVM (Unknown)"
    description = (
        "RANSAC 2D reciprocal lattice from BVM maxima; propose origin/g1/g2. "
        "Does not invent absolute orientation (optional relative hkl if zone given)."
    )

    def run(self, ctx: BvmContext, *, log=None) -> BasisProposal:
        import bvm_indexing as bix

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
            orientation_mode="unknown",
        )
        if log is not None:
            log(
                f"Index BVM Unknown: {result.n_inliers}/{len(result.peaks)} inliers; "
                f"g1={result.index_g1} g2={result.index_g2}"
            )
        return proposal_from_indexing_result(result, plugin_id=self.id)
