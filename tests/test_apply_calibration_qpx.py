"""'apply' calibration must use the FITTED Q pixel, not the guess.

PARA TEST (19-09-2026): q_px guess 0.0224 vs fitted 0.02157 A^-1/px -- applying the
guess put every |g| 3.8 % off (the indexing credibility flagged |x| vs Si +3.7 %).
"""
import engine as E


def _scan(refit, fitted):
    s = E.Scan(name="s")
    s.params.q_px = 0.0224
    s.params.q_refit = refit
    s.params.q_px_fitted = fitted
    return s


def test_apply_uses_the_fitted_q_pixel():
    cfg = E._params_to_batch_scan_cfg(_scan(True, 0.02157))
    assert cfg.step11["px_guess"] == 0.02157


def test_apply_uses_the_guess_when_no_refit_or_no_fit():
    assert E._params_to_batch_scan_cfg(_scan(False, 0.02157)).step11["px_guess"] == 0.0224
    assert E._params_to_batch_scan_cfg(_scan(True, None)).step11["px_guess"] == 0.0224
