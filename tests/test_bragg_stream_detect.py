import numpy as np
import py4DSTEM
import h5py

from bragg_stream import detect_braggpeaks_streaming


def _tiny_synthetic_datacube():
    # 2x2 scan, 32x32 detector, one bright pixel per pattern so detection has
    # something unambiguous to find without needing a realistic template.
    rng = np.random.default_rng(0)
    data = rng.normal(scale=1.0, size=(2, 2, 32, 32)).astype(np.float32)
    for rx in range(2):
        for ry in range(2):
            data[rx, ry, 16, 16] = 500.0  # single strong disk at center
    return py4DSTEM.DataCube(data)


def test_streaming_matches_full_scan_find_bragg_disks(tmp_path):
    dc = _tiny_synthetic_datacube()
    template = np.zeros((32, 32), dtype=np.float32)
    template[15:18, 15:18] = 1.0  # crude disk-shaped probe kernel

    # NOTE: py4DSTEM==0.14.19's own `get_maxima_2D` only accepts
    # subpixel in ('pixel', 'poly', 'multicorr') -- 'none' (used in the
    # original plan draft) raises AssertionError. 'pixel' is the equivalent
    # "no subpixel fit" option in this installed version.
    kwargs = dict(minAbsoluteIntensity=1.0, minPeakSpacing=5, edgeBoundary=2, subpixel="pixel")

    # Reference: py4DSTEM's own full-scan public API (what compute_braggpeaks_step calls today).
    # Returns a BraggVectors instance; its `.raw[rx, ry]` accessor (braggvectors.py:124-134)
    # returns a BVects with .qx/.qy/.I fields -- confirmed against the installed source and by
    # a throwaway script cross-checking these values against the position-batched call below.
    reference = dc.find_Bragg_disks(template=template, **kwargs)

    out_path = tmp_path / "streamed.h5"
    detect_braggpeaks_streaming(dc, template, out_path, batch_size=1, **kwargs)

    with h5py.File(out_path, "r") as f:
        for rx in range(2):
            for ry in range(2):
                streamed_peaks = f["peaks"][f"{ry}_{rx}"][()]
                ref_pl = reference.raw[rx, ry]
                assert streamed_peaks.shape[0] == len(ref_pl.qx)
                assert streamed_peaks.shape[0] > 0  # sanity: this synthetic scan does detect peaks
                np.testing.assert_allclose(sorted(streamed_peaks[:, 0]), sorted(ref_pl.qx), atol=1e-3)
                np.testing.assert_allclose(sorted(streamed_peaks[:, 1]), sorted(ref_pl.qy), atol=1e-3)
