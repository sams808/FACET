"""The window's frame as rebuilt for 2.0: the toolbar that reflows instead of
squeezing, the five names the right column's ten pages are chosen from, the
card on the empty window, the projection under View > Along an axis.

Each is checked against the complaint it answers: a combo box narrower than
its own text ('bon...nce'), a tab bar cut off after 'Ov...', a blank window
that does not say how to begin.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from facet.core import theme as T
from facet.ui import chrome, shell

EXAMPLES = Path(__file__).resolve().parent.parent / "facet" / "data" / "examples"
BIPO4 = EXAMPLES / "bismuth_phosphate_BiPO4_cod9008088.cif"
TEN = ["Site", "Tools", "Diffraction", "PDF", "EXAFS", "Planes", "Styles",
       "Volume", "Disorder", "Theme"]


@pytest.fixture(scope="module")
def styled(qapp):
    """The application's own style sheet, put on once for the module: the
    measurements below are of FACET as it is drawn, and setting a style
    sheet polishes every widget alive, so once is what it costs."""
    chrome.apply(qapp, T.Theme())
    yield
    qapp.setStyleSheet("")


@pytest.fixture
def window(qapp, styled):
    from facet.ui import preview

    win = preview.PreviewWindow()
    win.resize(1420, 900)
    win.show()
    qapp.processEvents()
    yield win
    win.close()
    win.deleteLater()
    qapp.processEvents()


# -- the toolbar ---------------------------------------------------------------------

def test_the_toolbar_reflows_and_never_squeezes_a_control(window, qapp):
    bar = window.toolbar
    rows = {}
    for width in (1100, 1420, 1920, 2560):
        window.resize(width, 900)
        qapp.processEvents()
        rows[width] = bar.rows_in_use()
        for name in ("style_box", "poly_box", "atom_label_box",
                     "bond_label_box", "list_threshold", "cell_check",
                     "label_scope_button"):
            widgets = getattr(window, name)
            for widget in (widgets if isinstance(widgets, list) else [widgets]):
                assert widget.width() >= widget.minimumSizeHint().width(), (
                    name, width, widget.width(),
                    widget.minimumSizeHint().width())
        # every group is inside the bar, none pushed past its right edge
        for group in bar._all():
            right = group.mapTo(bar, group.rect().topRight()).x()
            assert right <= bar.width(), (width, right, bar.width())
    assert rows[2560] == 1, rows
    assert rows[1420] == 2, rows
    assert rows[1100] >= rows[1420] >= rows[1920] >= rows[2560], rows


def test_the_toolbar_asks_for_one_groups_width_not_all_of_them(window):
    bar = window.toolbar
    widest = max(g.sizeHint().width() for g in bar._all())
    assert bar.minimumSizeHint().width() <= widest + 40
    # which is what lets the window be narrower than its toolbar's one row
    one_row = sum(g.sizeHint().width() for g in bar._all())
    assert window.minimumSizeHint().width() < one_row


def test_a_group_whose_width_changes_is_laid_out_again(window, qapp):
    bar = window.toolbar
    before = bar._assignment
    window.probability_label.show()
    window.probability_box.show()
    bar.reflow()
    qapp.processEvents()
    assert len(bar._assignment) == len(before)
    for group in bar._all():
        assert group.mapTo(bar, group.rect().topRight()).x() <= bar.width()


# -- the column's pages ---------------------------------------------------------------

def test_ten_pages_are_reached_from_five_names_that_always_fit(window, qapp):
    column = window.crystal_nav
    tabs = window.crystal_tabs
    assert sorted(column.titles()) == sorted(TEN)
    assert [tabs.tabText(i) for i in range(tabs.count())] == TEN
    assert tabs.tabBar().isHidden()
    assert len(column.buttons) == 5
    for width in (1100, 1366, 1420, 1920):
        window.resize(width, 900)
        qapp.processEvents()
        assert column.bar.sizeHint().width() <= column.width(), (
            width, column.bar.sizeHint().width(), column.width())
        for button in column.buttons:
            assert button.width() >= button.sizeHint().width()


def test_a_group_button_carries_the_name_of_its_page_on_screen(window, qapp):
    column = window.crystal_nav
    tabs = window.crystal_tabs
    geometry = column.button_for("Volume")
    assert geometry is column.button_for("Planes")
    assert geometry.text().startswith("Geometry") and not geometry.isChecked()
    assert column.button_for("Site").isChecked()
    # choosing from the group's list
    action = next(a for a in geometry.menu().actions() if a.text() == "Volume")
    action.trigger()
    qapp.processEvents()
    assert tabs.tabText(tabs.currentIndex()) == "Volume"
    assert geometry.text().startswith("Volume") and geometry.isChecked()
    assert action.isChecked()
    assert not column.button_for("Site").isChecked()
    # and from the code that has always driven the tab widget
    tabs.setCurrentIndex(TEN.index("Tools"))
    qapp.processEvents()
    assert column.button_for("Tools").isChecked()
    assert geometry.text().startswith("Geometry") and not geometry.isChecked()
    column.button_for("Site").click()
    assert tabs.currentIndex() == 0


def test_a_paged_column_names_a_page_no_group_lists(qapp):
    from PySide6.QtWidgets import QLabel, QTabWidget

    tabs = QTabWidget()
    for name in ("One", "Two", "Three", "Four"):
        tabs.addTab(QLabel(name), name)
    column = shell.PagedColumn(tabs, ("One", ("More", ("Two", "Three"))))
    assert column.titles() == ["One", "Two", "Three", "Four"]
    assert [b.menu() is not None for b in column.buttons] == [False, True, False]


# -- the empty window -----------------------------------------------------------------

def test_the_welcome_card_is_up_exactly_while_nothing_is_open(window, qapp):
    card = window.welcome
    assert card.isVisible() and window.view.quiet_placeholder
    assert [b.text().split()[0] for b in card.buttons] == ["Open", "Examples",
                                                           "Recent"]
    assert card.buttons[1].menu().actions(), "no example is offered"
    # centred on the view, and inside it
    centre = card.geometry().center()
    assert abs(centre.x() - window.view.width() // 2) <= 2
    assert card.width() <= window.view.width()
    window.load_many([str(BIPO4)])
    qapp.processEvents()
    assert not card.isVisible() and not window.view.quiet_placeholder
    window._on_remove(0)
    qapp.processEvents()
    assert card.isVisible()


def test_an_example_opens_from_the_card(window, qapp):
    menu = window.welcome.buttons[1].menu()
    menu.actions()[0].trigger()
    qapp.processEvents()
    assert window.project.entries and not window.welcome.isVisible()


# -- the projection, the status bar ------------------------------------------------------

def test_the_projection_is_in_the_view_menu_and_keeps_one_state(window, qapp):
    assert not window.ortho_check.isChecked()
    window.ortho_action.trigger()
    assert window.ortho_check.isChecked() and window.ortho_action.isChecked()
    assert window.view.camera.orthographic
    window.ortho_check.setChecked(False)            # what a session restore does
    assert not window.ortho_action.isChecked()
    assert not window.view.camera.orthographic


def test_the_window_opens_narrower_than_before(window):
    # 1.x needed 1434 px before anything gave way; the reflowing toolbar is
    # what the width was held up by
    minimum = window.minimumSizeHint()
    assert minimum.width() <= 1200 and minimum.height() <= 730, minimum


def test_the_tools_pages_are_six_names_whose_bar_never_changes_width(window, qapp):
    window.load_many([str(BIPO4)])
    qapp.processEvents()
    window.crystal_tabs.setCurrentIndex(TEN.index("Tools"))
    qapp.processEvents()
    tools = window.utilities
    pages = tools.pages
    assert len(pages.buttons) == 6 and len(pages.titles()) == tools.tabs.count() == 12
    widths = set()
    for index in range(tools.tabs.count()):
        tools.tabs.setCurrentIndex(index)
        qapp.processEvents()
        widths.add(pages.bar.sizeHint().width())
        assert pages.button_for(tools.tabs.tabText(index)).isChecked()
    assert len(widths) == 1, widths          # a group's name does not shift the bar
    # and it fits the column at the column's minimum width, scroll bar included
    assert widths.pop() <= window.crystal_tabs.minimumWidth() - 12
