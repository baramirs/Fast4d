import numpy as np
import py4DSTEM

import pipeline
from bragg_stream import detect_braggpeaks_streaming, finalize_stream_to_braggvectors


def _tiny_synthetic_datacube():
    rng = np.random.default_rng(0)
    data = rng.normal(scale=1.0, size=(2, 2, 32, 32)).astype(np.float32)
    for rx in range(2):
        for ry in range(2):
            data[rx, ry, 16, 16] = 500.0
    return py4DSTEM.DataCube(data)


_KWARGS = dict(minAbsoluteIntensity=1.0, minPeakSpacing=5, edgeBoundary=2, subpixel="pixel")


def test_finalize_builds_braggvectors_matching_full_scan(tmp_path):
    dc = _tiny_synthetic_datacube()
    template = np.zeros((32, 32), dtype=np.float32)
    template[15:18, 15:18] = 1.0

    reference = dc.find_Bragg_disks(template=template, **_KWARGS)

    stream_path = tmp_path / "streamed.h5"
    detect_braggpeaks_streaming(dc, template, stream_path, batch_size=1, **_KWARGS)

    bv = finalize_stream_to_braggvectors(stream_path, Qshape=(32, 32))

    assert tuple(int(v) for v in bv.Rshape) == (2, 2)
    for rx in range(2):
        for ry in range(2):
            got = bv.raw[rx, ry]
            ref = reference.raw[rx, ry]
            assert len(got.qx) == len(ref.qx)
            np.testing.assert_allclose(sorted(got.qx), sorted(ref.qx), atol=1e-3)
            np.testing.assert_allclose(sorted(got.qy), sorted(ref.qy), atol=1e-3)


def test_finalized_braggvectors_round_trips_through_path_a_loader(tmp_path):
    """The whole point of finalize: the streamed result must be readable by the
    normal Path-A loader (pipeline.load_braggpeaks_file -> py4DSTEM.read)."""
    dc = _tiny_synthetic_datacube()
    template = np.zeros((32, 32), dtype=np.float32)
    template[15:18, 15:18] = 1.0

    reference = dc.find_Bragg_disks(template=template, **_KWARGS)

    stream_path = tmp_path / "streamed.h5"
    detect_braggpeaks_streaming(dc, template, stream_path, batch_size=2, **_KWARGS)
    bv = finalize_stream_to_braggvectors(stream_path, Qshape=(32, 32))

    out_path = tmp_path / "braggpeaks.h5"
    pipeline.save_braggpeaks_file(bv, out_path)

    reloaded = pipeline.load_braggpeaks_file(out_path, expected_rshape=(2, 2))

    assert tuple(int(v) for v in reloaded.Rshape) == (2, 2)
    for rx in range(2):
        for ry in range(2):
            got = reloaded.raw[rx, ry]
            ref = reference.raw[rx, ry]
            assert len(got.qx) == len(ref.qx)
            np.testing.assert_allclose(sorted(got.qx), sorted(ref.qx), atol=1e-3)
            np.testing.assert_allclose(sorted(got.qy), sorted(ref.qy), atol=1e-3)


def test_compute_braggpeaks_step_streaming_route_is_reloadable(tmp_path, monkeypatch):
    """End-to-end: FAST4D_STREAM_BRAGG=1 routes compute_braggpeaks_step through the
    streaming path, releases the datacube, and still leaves a Path-A-readable file."""
    from state import WorkflowState

    dc = _tiny_synthetic_datacube()
    reference = None

    template = np.zeros((32, 32), dtype=np.float32)
    template[15:18, 15:18] = 1.0
    reference = dc.find_Bragg_disks(template=template, **_KWARGS)

    state = WorkflowState()
    state.datacube = _tiny_synthetic_datacube()
    state.probe = object()
    state.detect_params = dict(_KWARGS)
    monkeypatch.setattr(pipeline, "probe_kernel_template_ndarray", lambda p: template)
    monkeypatch.setenv("FAST4D_STREAM_BRAGG", "1")

    out_path = tmp_path / "bp.h5"
    result = pipeline.compute_braggpeaks_step(state, save_path=out_path)

    assert state.datacube is None, "streaming path must release the datacube"
    assert result is state.braggpeaks
    assert out_path.exists()
    assert not out_path.with_name(out_path.stem + ".stream.h5").exists(), "temp stream file cleaned up"

    reloaded = pipeline.load_braggpeaks_file(out_path, expected_rshape=(2, 2))
    for rx in range(2):
        for ry in range(2):
            got = reloaded.raw[rx, ry]
            ref = reference.raw[rx, ry]
            assert len(got.qx) == len(ref.qx)
