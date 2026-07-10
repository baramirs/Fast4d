import pipeline


class _FakeStat:
    def __init__(self, size):
        self.st_size = size


class _FakePath:
    """Duck-types the one method _pick_mib_mem_mode actually calls."""
    def __init__(self, size):
        self._size = size

    def stat(self):
        return _FakeStat(self._size)


def test_force_memmap_env_var_wins(monkeypatch):
    monkeypatch.setenv("FAST4D_FORCE_MEMMAP", "1")
    assert pipeline._pick_mib_mem_mode(_FakePath(1024)) == "memmap"


def test_small_file_under_thresholds_returns_none(monkeypatch):
    monkeypatch.delenv("FAST4D_FORCE_MEMMAP", raising=False)
    assert pipeline._pick_mib_mem_mode(_FakePath(10 * 1024**2)) is None


def test_file_at_2gib_returns_memmap(monkeypatch):
    monkeypatch.delenv("FAST4D_FORCE_MEMMAP", raising=False)
    assert pipeline._pick_mib_mem_mode(_FakePath(2 * 1024**3)) == "memmap"
