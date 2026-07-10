import engine as E


def test_default_analysis_scope_is_not_shared():
    assert E.get_analysis_scope().shared_stats is False


def test_set_analysis_scope_updates_the_shared_instance():
    E.set_analysis_scope(shared_stats=True)
    try:
        assert E.get_analysis_scope().shared_stats is True
    finally:
        E.set_analysis_scope(shared_stats=False)  # restore default for other tests


def test_set_analysis_scope_ignores_none():
    E.set_analysis_scope(shared_stats=True)
    try:
        E.set_analysis_scope(shared_stats=None)
        assert E.get_analysis_scope().shared_stats is True  # unchanged
    finally:
        E.set_analysis_scope(shared_stats=False)
