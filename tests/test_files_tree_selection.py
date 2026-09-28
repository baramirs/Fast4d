"""Files dock: Ctrl+click multi-select without changing the active (current) file."""


def _rows(n=3):
    return [{"label": f"scan{i:02d}", "h5path": "", "tags": []} for i in range(n)]


def test_selected_scan_indices_starts_with_active(qapp):
    from qt_widgets import FilesTree
    tree = FilesTree()
    tree.refresh(_rows(), active=0)
    assert tree.selected_scan_indices() == [0]


def test_selected_scan_indices_accumulates_without_clearing_current(qapp):
    from qt_widgets import FilesTree
    tree = FilesTree()
    tree.refresh(_rows(), active=0)
    tree.topLevelItem(2).setSelected(True)
    assert tree.currentItem() is tree.topLevelItem(0)
    assert tree.selected_scan_indices() == [0, 2]


def test_toggle_scan_selected_adds_without_changing_current(qapp):
    from qt_widgets import FilesTree
    tree = FilesTree()
    tree.refresh(_rows(), active=0)
    current = tree.currentItem()
    tree.toggle_scan_selected(1)
    assert tree.currentItem() is current
    assert tree.selected_scan_indices() == [0, 1]


def test_toggle_scan_selected_can_drop_a_file_from_the_queue(qapp):
    from qt_widgets import FilesTree
    tree = FilesTree()
    tree.refresh(_rows(), active=0)
    tree.toggle_scan_selected(1)
    tree.toggle_scan_selected(1)
    assert tree.currentItem() is tree.topLevelItem(0)
    assert tree.selected_scan_indices() == [0]


def test_plain_select_scan_replaces_selection(qapp):
    from qt_widgets import FilesTree
    tree = FilesTree()
    tree.refresh(_rows(), active=0)
    tree.toggle_scan_selected(2)
    tree.select_scan(1)
    assert tree.currentItem() is tree.topLevelItem(1)
    assert tree.selected_scan_indices() == [1]
