from types import SimpleNamespace

import pipeline


def test_compute_braggpeaks_step_collects_garbage_after_save(monkeypatch, tmp_path):
    fake_peaks = object()
    fake_datacube = SimpleNamespace(find_Bragg_disks=lambda **kw: fake_peaks)
    state = SimpleNamespace(
        use_existing_braggpeaks=False, braggpeaks=None, datacube=fake_datacube,
        probe=object(), detect_params={}, braggpeaks_path=None,
    )
    monkeypatch.setattr(pipeline, "normalize_subpixel_keyword", lambda x: x)
    monkeypatch.setattr(pipeline, "cupy_device_summary", lambda: "")
    monkeypatch.setattr(pipeline, "_ensure_cupy_current_device_for_thread", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "format_probe_template_log_line", lambda st: "")
    monkeypatch.setattr(pipeline, "probe_kernel_template_ndarray", lambda p: object())
    monkeypatch.setattr(pipeline, "save_braggpeaks_datacube_notebook_style", lambda *a, **k: None)

    gc_calls = []
    monkeypatch.setattr("gc.collect", lambda: gc_calls.append(1))

    result = pipeline.compute_braggpeaks_step(state, save_path=tmp_path / "bp.h5")

    assert gc_calls, "gc.collect() should run right after the notebook-style save completes"
    # Lifecycle contract: the raw datacube is released once peaks are detected AND
    # persisted (braggpeaks_path set), but the detected peaks themselves survive.
    assert state.datacube is None, "raw datacube should be released after a successful bragg save"
    assert result is fake_peaks, "detected braggpeaks must be returned unchanged"
    assert state.braggpeaks is fake_peaks, "detected braggpeaks must remain resident"
    assert str(state.braggpeaks_path) == str(tmp_path / "bp.h5")


def test_compute_braggpeaks_step_keeps_datacube_when_save_fails(monkeypatch, tmp_path):
    """If persistence fails, the datacube must NOT be released — the caller needs it
    to retry the save."""
    fake_peaks = object()
    fake_datacube = SimpleNamespace(find_Bragg_disks=lambda **kw: fake_peaks)
    state = SimpleNamespace(
        use_existing_braggpeaks=False, braggpeaks=None, datacube=fake_datacube,
        probe=object(), detect_params={}, braggpeaks_path=None,
    )
    monkeypatch.setattr(pipeline, "normalize_subpixel_keyword", lambda x: x)
    monkeypatch.setattr(pipeline, "cupy_device_summary", lambda: "")
    monkeypatch.setattr(pipeline, "_ensure_cupy_current_device_for_thread", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "format_probe_template_log_line", lambda st: "")
    monkeypatch.setattr(pipeline, "probe_kernel_template_ndarray", lambda p: object())

    def _boom(*a, **k):
        raise RuntimeError("notebook save failed")

    monkeypatch.setattr(pipeline, "save_braggpeaks_datacube_notebook_style", _boom)
    monkeypatch.setattr(pipeline, "save_braggpeaks_file", _boom)

    try:
        pipeline.compute_braggpeaks_step(state, save_path=tmp_path / "bp.h5")
    except RuntimeError:
        pass

    assert state.datacube is fake_datacube, "datacube must survive a failed save for retry"
