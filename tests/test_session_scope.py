"""Session analysis acts only on the files ticked AND applied with Refresh."""
import numpy as np

import session_scope as SS


class S:
    def __init__(self, n):
        self.name = n


def test_nothing_applied_until_refresh():
    sel = SS.SessionSelection()
    sel.set_checked(["a", "b"])
    assert sel.applied == [] and sel.pending
    sel.refresh()
    assert sel.applied == ["a", "b"] and not sel.pending


def test_applied_scans_keep_session_order_and_ignore_unticked():
    sel = SS.SessionSelection()
    sel.set_checked(["c", "a"])
    sel.refresh()
    assert [s.name for s in sel.applied_scans([S("a"), S("b"), S("c")])] == ["a", "c"]


def test_changes_without_refresh_do_not_touch_applied():
    sel = SS.SessionSelection()
    sel.set_checked(["a"])
    sel.refresh()
    sel.set_checked(["a", "b"])
    assert sel.applied == ["a"] and sel.pending


def test_json_roundtrip_drops_missing_names():
    sel = SS.SessionSelection()
    sel.set_checked(["a", "x"])
    sel.refresh()
    other = SS.SessionSelection()
    other.load(sel.to_json(), ["a", "b"])
    assert other.applied == ["a"] and other.checked == ["a"]


def test_session_json_roundtrip(tmp_path):
    import engine as E
    SS.reset()
    scans = [E.Scan(name="a"), E.Scan(name="b")]
    SS.current().set_checked(["b"])
    SS.current().refresh()
    p = tmp_path / "s.json"
    E.save_session_json(scans, str(p))
    SS.reset()
    E.load_session_json(str(p))
    assert SS.current().applied == ["b"]


def test_pooled_correlation_uses_valid_pixels_of_all_scans():
    import analysis as A

    class Sc:
        def __init__(self, arr):
            self.name = "s"
            self.state = type("St", (), {"strain_raw": {"without_roi": arr}})()

    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 20, 3))
    a[:5] = 0.0                                   # failed-fit rows (Fast4D writes 0)
    b = rng.normal(size=(20, 20, 3))
    df = A.pooled_channel_correlation([Sc(a), Sc(b)], "without_roi")
    assert df.shape == (3, 3)
    assert np.allclose(np.diag(df.values), 1.0)
    assert df.attrs["n_pixels"] == 15 * 20 + 20 * 20
