import numpy as np
import h5py

from bragg_stream import StreamingBraggWriter


def test_writer_creates_one_dataset_per_position(tmp_path):
    path = tmp_path / "stream.h5"
    with StreamingBraggWriter(path, r_shape=(2, 2)) as w:
        w.write(0, 0, qx=np.array([1.0, 2.0]), qy=np.array([1.5, 2.5]), intensity=np.array([10.0, 20.0]))
        w.write(1, 1, qx=np.array([3.0]), qy=np.array([3.5]), intensity=np.array([30.0]))

    with h5py.File(path, "r") as f:
        assert tuple(f["r_shape"][()]) == (2, 2)
        peaks_0_0 = f["peaks"]["0_0"][()]
        assert peaks_0_0.shape == (2, 3)
        np.testing.assert_allclose(peaks_0_0[:, 0], [1.0, 2.0])   # qx
        np.testing.assert_allclose(peaks_0_0[:, 2], [10.0, 20.0])  # intensity
        peaks_1_1 = f["peaks"]["1_1"][()]
        assert peaks_1_1.shape == (1, 3)
        assert "0_1" not in f["peaks"]  # positions never written simply don't exist


def test_writer_handles_zero_peaks_at_a_position(tmp_path):
    path = tmp_path / "stream.h5"
    with StreamingBraggWriter(path, r_shape=(1, 1)) as w:
        w.write(0, 0, qx=np.array([]), qy=np.array([]), intensity=np.array([]))

    with h5py.File(path, "r") as f:
        assert f["peaks"]["0_0"][()].shape == (0, 3)
