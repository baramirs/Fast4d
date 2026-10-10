"""find_sidecar_braggpeaks must not pair a scan with ANOTHER scan's braggpeaks.

18A/4DSTEM had ``Scan02-...-4ms.mib`` (130x130) and ``Scan02-...-4ms_.mib`` (128x128)
with ``...-4ms_braggpeaks.h5`` and ``...-4ms__braggpeaks.h5``. The prefix glob
``...-4ms*braggpeaks*.h5`` sorted ``-4ms__braggpeaks`` first, so the 130x130 scan got
the 128x128 braggpeaks (ROI mask (130,130) vs scan (128,128)).
"""
import engine as E


def _touch(folder, *names):
    for n in names:
        (folder / n).write_bytes(b"")


def test_underscore_sibling_does_not_cross_link(tmp_path):
    a, b = "Scan02-SpotSize9-FOV160nm-4ms", "Scan02-SpotSize9-FOV160nm-4ms_"
    _touch(tmp_path, f"{a}.mib", f"{b}.mib",
           f"{a}_braggpeaks.h5", f"{b}_braggpeaks.h5")
    assert E.find_sidecar_braggpeaks(str(tmp_path / f"{a}.mib")) == \
        str(tmp_path / f"{a}_braggpeaks.h5")
    assert E.find_sidecar_braggpeaks(str(tmp_path / f"{b}.mib")) == \
        str(tmp_path / f"{b}_braggpeaks.h5")


def test_no_own_braggpeaks_does_not_take_a_siblings(tmp_path):
    a, b = "Scan02", "Scan02_"
    _touch(tmp_path, f"{a}.mib", f"{b}.mib", f"{b}_braggpeaks.h5")
    assert E.find_sidecar_braggpeaks(str(tmp_path / f"{a}.mib")) is None


def test_existing_conventions_still_found(tmp_path):
    _touch(tmp_path, "Scan04_512.mib", "Scan04_512braggpeaks.h5",
           "Scan05.mib", "Scan05_braggpeaks_v2.h5")
    assert E.find_sidecar_braggpeaks(str(tmp_path / "Scan04_512.mib")) == \
        str(tmp_path / "Scan04_512braggpeaks.h5")
    assert E.find_sidecar_braggpeaks(str(tmp_path / "Scan05.mib")) == \
        str(tmp_path / "Scan05_braggpeaks_v2.h5")
