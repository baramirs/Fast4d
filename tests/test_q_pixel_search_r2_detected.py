"""Q-pixel search on DETECTED disks (strain_check ronda 2, task B, CL160 3.0 mrad, 1e5 e).

Old behaviour: the fit from the guess (true 0.0130 x 1.03) went to 0.02075 (+60 %), the
crystal frame was indexed -35 deg off and e_yy came out -45 % (REPORTE_R2.md, H-R2-6).
With disk-detection bias the py4DSTEM profile fit also walks off the ring coincidence it
starts on, so the grid maximum must stay a candidate. Remaining e_yy bias (~ -17 %) is the
detection itself (same as with the true Q pixel)."""
from pathlib import Path

import numpy as np
import pytest

import engine as E

BP = Path(__file__).parent / "fixtures" / "r2" / "CL160_a3_d1e5_braggpeaks.h5"
TRUE_PX = 0.0130


@pytest.mark.skipif(not BP.is_file(), reason="ronda-2 fixture missing")
def test_detected_cl160_q_and_frame_are_recovered():
    sc = E.Scan(name="r2_cl160", raw_path="", braggpeaks_path=str(BP))
    p = sc.params
    p.center_guess = [128.0, 128.0]
    p.q_px = TRUE_PX * 1.03
    p.roi_bounds = [0, 6, 0, 12]
    p.cal_crystal, p.zone_axis, p.frame_mode = "Si", [1, 1, 0], "crystal"
    p.frame_x_hkl, p.frame_y_hkl = [-1, 1, 0], [0, 0, 1]
    E.load_braggpeaks(sc, log=lambda m: None)
    E.run_calibration_sequence(sc, make_figures=False, log=lambda m: None)
    assert abs(p.q_px_fitted / TRUE_PX - 1) < 0.02            # was +60 %
    assert p.q_px_check["status"] in ("differs", "ambiguous")  # the user is asked
    E.compute_strain(sc, use_roi=False, log=lambda m: None)
    E.compute_strain(sc, use_roi=True, log=lambda m: None)
    ang = float(sc.state.crystal_frame_cards["with_roi"]["x_angle_applied_deg"])
    assert min(ang % 180.0, 180.0 - ang % 180.0) < 0.5        # was 324.5 deg (-35.5)
    raw = np.asarray(sc.state.strain_raw["with_roi"], float)
    exy = np.nanmean(np.where(raw[4] > 0.5, raw[2], np.nan)[:, 6:12]) * 100
    assert abs(exy) < 0.05                                     # was -0.39 % spurious
