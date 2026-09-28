"""Serial Path-B braggpeaks queue: copy tuned detect params, load+probe, detect, free RAM."""
import driver
import engine as E


def _path_b_scan(name, tmp_path, **kw):
    raw = tmp_path / f"{name}.mib"
    raw.write_bytes(b"raw")
    vac = tmp_path / "vac.mib"
    if not vac.exists():
        vac.write_bytes(b"vac")
    return E.Scan(name=name, raw_path=str(raw), vacuum_path=str(vac), **kw)


def test_copy_detection_settings_copies_detect_and_vacuum_not_six_points():
    src = E.Scan(name="tuned", vacuum_path=r"C:\vac.mib")
    src.params.detect_min_absolute_intensity = 42
    src.params.detect_cuda = False
    src.params.probe_source = "bf_roi"
    src.params.probe_vacuum_roi_bounds = [1, 2, 3, 4]
    src.params.six_points = [(10.0, 20.0)] * 6
    dst = E.Scan(name="other")
    dst.params.six_points = [(9.0, 9.0)]
    E.copy_detection_settings(dst, src)
    assert dst.params.detect_min_absolute_intensity == 42
    assert dst.params.detect_cuda is False
    assert dst.params.probe_source == "bf_roi"
    assert dst.vacuum_path == r"C:\vac.mib"
    assert dst.params.probe_vacuum_roi_bounds == [1, 2, 3, 4]
    assert dst.params.six_points == [(9.0, 9.0)]
    src.params.probe_vacuum_roi_bounds.append(99)
    assert dst.params.probe_vacuum_roi_bounds == [1, 2, 3, 4]


def test_copy_detection_settings_is_noop_when_src_is_dst():
    sc = E.Scan(name="a")
    sc.params.detect_min_absolute_intensity = 7
    E.copy_detection_settings(sc, sc)
    assert sc.params.detect_min_absolute_intensity == 7


def test_ensure_datacube_and_probe_skips_when_already_loaded(monkeypatch):
    sc = E.Scan(name="a")
    st = sc.ensure_state()
    st.datacube = "cube"
    st.probe = "probe"
    called = []
    monkeypatch.setattr(E, "load_datacube", lambda *a, **k: called.append("load"))
    monkeypatch.setattr(E, "compute_probe", lambda *a, **k: called.append("probe"))
    E.ensure_datacube_and_probe(sc)
    assert called == []


def test_ensure_datacube_and_probe_uses_shared_probe(monkeypatch):
    sc = E.Scan(name="a")
    st = sc.ensure_state()
    st.datacube = "cube"
    applied = []
    monkeypatch.setattr(E, "load_datacube", lambda *a, **k: applied.append("load"))
    monkeypatch.setattr(
        E, "apply_probe",
        lambda scan, probe, **k: applied.append(("apply", probe)))
    monkeypatch.setattr(E, "compute_probe", lambda *a, **k: applied.append("compute"))
    E.ensure_datacube_and_probe(sc, shared_probe="SHARED")
    assert applied == [("apply", "SHARED")]


def test_compute_braggpeaks_all_runs_serially_and_frees_after_each(monkeypatch, tmp_path):
    scans = [_path_b_scan("a", tmp_path), _path_b_scan("b", tmp_path)]
    order = []
    monkeypatch.setattr(E, "compute_shared_probe", lambda *a, **k: "PROBE")
    monkeypatch.setattr(
        E, "ensure_datacube_and_probe",
        lambda sc, **k: order.append(("load", sc.name)))
    monkeypatch.setattr(
        E, "compute_braggpeaks",
        lambda sc, **k: order.append(("bp", sc.name)))
    monkeypatch.setattr(
        E, "free_memory",
        lambda scans_arg, **k: order.append(("free", scans_arg[0].name)))

    result = driver.compute_braggpeaks_all(scans)

    assert [(k, n) for k, n in order] == [
        ("load", "a"), ("bp", "a"), ("free", "a"),
        ("load", "b"), ("bp", "b"), ("free", "b"),
    ]
    assert result.n_ok == 2
    assert result.n_failed == 0


def test_compute_braggpeaks_all_copies_detect_params_from_source(monkeypatch, tmp_path):
    src = _path_b_scan("tuned", tmp_path)
    src.params.detect_min_absolute_intensity = 77
    other = _path_b_scan("other", tmp_path)
    monkeypatch.setattr(E, "compute_shared_probe", lambda *a, **k: None)
    monkeypatch.setattr(E, "ensure_datacube_and_probe", lambda *a, **k: None)
    monkeypatch.setattr(E, "compute_braggpeaks", lambda *a, **k: None)
    monkeypatch.setattr(E, "free_memory", lambda *a, **k: None)

    driver.compute_braggpeaks_all([other], source=src)

    assert other.params.detect_min_absolute_intensity == 77
    assert other.vacuum_path == src.vacuum_path


def test_compute_braggpeaks_all_skips_path_a(monkeypatch, tmp_path):
    bp = tmp_path / "already.h5"
    bp.write_bytes(b"x")
    path_a = E.Scan(name="done", braggpeaks_path=str(bp))
    path_b = _path_b_scan("need", tmp_path)
    called = []
    monkeypatch.setattr(E, "compute_shared_probe", lambda *a, **k: None)
    monkeypatch.setattr(E, "ensure_datacube_and_probe",
                        lambda sc, **k: called.append(sc.name))
    monkeypatch.setattr(E, "compute_braggpeaks", lambda sc, **k: called.append("bp:" + sc.name))
    monkeypatch.setattr(E, "free_memory", lambda *a, **k: None)

    result = driver.compute_braggpeaks_all([path_a, path_b])

    assert called == ["need", "bp:need"]
    assert result.n_ok == 2
    skipped = next(o for o in result.outcomes if o.scan is path_a)
    assert skipped.ok and "skipped" in (skipped.labels or [])


def test_compute_braggpeaks_all_continues_after_error(monkeypatch, tmp_path):
    scans = [_path_b_scan("a", tmp_path), _path_b_scan("b", tmp_path),
             _path_b_scan("c", tmp_path)]
    monkeypatch.setattr(E, "compute_shared_probe", lambda *a, **k: None)
    monkeypatch.setattr(E, "ensure_datacube_and_probe", lambda *a, **k: None)

    def _bp(sc, **k):
        if sc.name == "b":
            raise RuntimeError("boom")

    monkeypatch.setattr(E, "compute_braggpeaks", _bp)
    monkeypatch.setattr(E, "free_memory", lambda *a, **k: None)

    result = driver.compute_braggpeaks_all(scans)

    assert result.n_ok == 2
    assert result.n_failed == 1
    assert [o.scan.name for o in result.outcomes] == ["a", "b", "c"]


def test_compute_braggpeaks_all_cancel_stops_before_next_scan(monkeypatch, tmp_path):
    scans = [_path_b_scan("a", tmp_path), _path_b_scan("b", tmp_path)]
    started = []
    monkeypatch.setattr(E, "compute_shared_probe", lambda *a, **k: None)
    monkeypatch.setattr(E, "ensure_datacube_and_probe",
                        lambda sc, **k: started.append(sc.name))
    monkeypatch.setattr(E, "compute_braggpeaks", lambda *a, **k: None)
    monkeypatch.setattr(E, "free_memory", lambda *a, **k: None)

    n_done = {"n": 0}

    def on_done(_out):
        n_done["n"] += 1

    def cancel():
        return n_done["n"] >= 1

    result = driver.compute_braggpeaks_all(scans, cancel=cancel, on_scan_done=on_done)

    assert started == ["a"]
    assert result.n_ok == 1


def test_compute_braggpeaks_all_passes_shared_probe(monkeypatch, tmp_path):
    scans = [_path_b_scan("a", tmp_path), _path_b_scan("b", tmp_path)]
    seen = []
    monkeypatch.setattr(E, "compute_shared_probe", lambda *a, **k: "SHARED")
    monkeypatch.setattr(
        E, "ensure_datacube_and_probe",
        lambda sc, **k: seen.append(k.get("shared_probe")))
    monkeypatch.setattr(E, "compute_braggpeaks", lambda *a, **k: None)
    monkeypatch.setattr(E, "free_memory", lambda *a, **k: None)

    driver.compute_braggpeaks_all(scans)

    assert seen == ["SHARED", "SHARED"]


def test_compute_braggpeaks_all_does_not_share_bf_roi_probe(monkeypatch, tmp_path):
    src = _path_b_scan("tuned", tmp_path)
    src.params.probe_source = "bf_roi"
    src.ensure_state().probe = "FILE1_PROBE"
    other = _path_b_scan("other", tmp_path)
    seen = []
    monkeypatch.setattr(E, "compute_shared_probe", lambda *a, **k: "SHOULD_NOT")
    monkeypatch.setattr(
        E, "ensure_datacube_and_probe",
        lambda sc, **k: seen.append(k.get("shared_probe")))
    monkeypatch.setattr(E, "compute_braggpeaks", lambda *a, **k: None)
    monkeypatch.setattr(E, "free_memory", lambda *a, **k: None)

    driver.compute_braggpeaks_all([other], source=src)

    assert seen == [None]
    assert other.params.probe_source == "bf_roi"
