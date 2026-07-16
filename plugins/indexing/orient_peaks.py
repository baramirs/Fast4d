"""Orient. peaks — py4DSTEM Crystal Path A (known) / Path B (ACOM)."""
from __future__ import annotations

from plugins.indexing.apply import proposal_from_orientation_result
from plugins.indexing.peaks import find_peaks
from plugins.indexing.types import BasisProposal, BvmContext


class OrientPeaksPlugin:
    id = "orient_peaks"
    label = "Orient. peaks"
    description = (
        "CIF/crystal theory-first: Path A generate pattern (zone+proj) or "
        "Path B ACOM match_single_pattern, then NN-match to BVM maxima."
    )

    def run(self, ctx: BvmContext, *, log=None) -> BasisProposal:
        import orientation_peaks as op

        if ctx.crystal is None:
            raise ValueError("Orient. peaks requires a py4DSTEM Crystal (CIF)")
        mode = str(ctx.extras.get("mode", "known_generate") or "known_generate")
        maxima = find_peaks(
            ctx.bvm,
            min_spacing=ctx.min_spacing,
            min_absolute_intensity=ctx.min_absolute_intensity,
            max_num_peaks=ctx.max_num_peaks,
            edge_boundary=ctx.edge_boundary,
            image_upsample=int(ctx.image_upsample),
        )
        # orientation_peaks expects dict-like maxima
        maxima_dict = {
            "x": maxima["x"],
            "y": maxima["y"],
            "intensity": maxima["intensity"],
        }
        common = dict(
            crystal=ctx.crystal,
            crystal_name=ctx.crystal_name or getattr(ctx.crystal, "name", "crystal"),
            bvm=ctx.bvm,
            origin_px=ctx.origin_px,
            Q_pixel=float(ctx.Q_pixel),
            Q_units=str(ctx.Q_units),
            k_max=float(ctx.k_max),
            tol_px=float(ctx.tol_px),
            accel_voltage=float(ctx.accel_voltage),
            maxima=maxima_dict,
            min_spacing=float(ctx.min_spacing),
            min_absolute_intensity=float(ctx.min_absolute_intensity),
            max_num_peaks=int(ctx.max_num_peaks),
            edge_boundary=float(ctx.edge_boundary),
            log=log,
        )
        if mode == "acom_match":
            result = op.run_acom_match(
                crystal_key=ctx.crystal_key or ctx.crystal_name or "crystal",
                angle_step_zone_axis=float(ctx.angle_step_zone_axis),
                angle_step_in_plane=float(ctx.angle_step_in_plane),
                **common,
            )
        else:
            result = op.run_known_generate(
                zone_axis=ctx.zone_axis or [1, 1, 0],
                proj_x_lattice=ctx.proj_x_lattice or ctx.real_axis_h or [0, 0, -1],
                **common,
            )
        return proposal_from_orientation_result(result, plugin_id=self.id)
