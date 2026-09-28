"""The transient "computing" status must never be saved into (or restored from)
the scan metadata, or a reopened file keeps showing "computing…" forever."""
import engine as E


def test_computing_is_not_persisted():
    sc = E.Scan(name="s")
    sc.status = "computing"
    assert E._scan_metadata_dict(sc)["status"] == "done"


def test_other_status_is_kept():
    sc = E.Scan(name="s")
    sc.status = "error"
    assert E._scan_metadata_dict(sc)["status"] == "error"


def test_stale_computing_in_metadata_is_healed():
    sc = E.Scan(name="s")
    E.apply_metadata_to_scan(sc, {"status": "computing"})
    assert sc.status == "done"
