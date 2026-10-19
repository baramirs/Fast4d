"""Q-pixel tool 'Test' button: global search by default (benchmark finding 2)."""
import pytest

import engine as E
import synthetic_si110 as S


@pytest.fixture
def host(qapp):
    from PySide6 import QtWidgets

    class Host(QtWidgets.QWidget):
        def __init__(self, sc):
            super().__init__()
            self._sc = sc
            self.logs = []
            self._console = type("C", (), {"log": staticmethod(self.logs.append)})()
            self._params = type("P", (), {"reload": staticmethod(lambda: None)})()

        def active_scan(self):
            return self._sc

        def _maybe_tidy_figures(self):
            pass

        def _update_active_views(self):
            pass

    bv, _ = S.make_braggvectors(24, e_max=0.02, jitter_px=0.03, seed=0)
    sc = S.make_scan(bv, q_guess=0.022)
    E.run_calibration_sequence(sc, make_figures=False, log=lambda m: None)
    return Host(sc)


def test_test_button_global_search_sets_the_guess_and_keeps_the_calibration(host):
    import qt_main
    dlg = qt_main.QPixelDialog(host)
    pl = E._pipeline()
    applied = pl._get_q_pixel_size(host._sc.state.braggpeaks.calibration)
    dlg._sp_px.setValue(0.022)
    assert dlg._test_mode.currentIndex() == 0          # global search is the default
    dlg._test()
    assert "Test error" not in dlg._status.text(), dlg._status.text()
    assert abs(dlg._sp_px.value() / S.Q_PIX - 1) < 2e-3
    assert pl._get_q_pixel_size(host._sc.state.braggpeaks.calibration) == applied
    dlg.close()
