import numpy as np

from bragg_stream import StreamingBraggWriter, read_streamed_peaks


def test_read_streamed_peaks_returns_one_positions_arrays(tmp_path):
    path = tmp_path / "stream.h5"
    with StreamingBraggWriter(path, r_shape=(1, 2)) as w:
        w.write(0, 0, qx=np.array([1.0]), qy=np.array([2.0]), intensity=np.array([9.0]))
        w.write(0, 1, qx=np.array([]), qy=np.array([]), intensity=np.array([]))

    peaks = read_streamed_peaks(path, 0, 0)
    assert peaks["qx"].tolist() == [1.0]
    assert peaks["qy"].tolist() == [2.0]
    assert peaks["intensity"].tolist() == [9.0]

    empty = read_streamed_peaks(path, 0, 1)
    assert empty["qx"].tolist() == []


def test_read_streamed_peaks_missing_position_returns_empty(tmp_path):
    path = tmp_path / "stream.h5"
    with StreamingBraggWriter(path, r_shape=(1, 1)) as w:
        w.write(0, 0, qx=np.array([1.0]), qy=np.array([1.0]), intensity=np.array([1.0]))

    # (0, 5) was never written (out of the declared r_shape, or simply skipped)
    result = read_streamed_peaks(path, 0, 5)
    assert result["qx"].tolist() == []
