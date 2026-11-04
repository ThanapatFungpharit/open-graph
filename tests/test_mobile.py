"""Touch devices: the layout, gestures, the soft keyboard, saving and the app's lifecycle."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import flet as ft
import numpy as np
import pytest

from opengraph import preferences
from opengraph.ui import dialogs, fonts, keypad, runtime
from opengraph.ui.geometry import Point
from opengraph.ui.main_window import COMPACT_WIDTH, PANEL_MIN_HEIGHT, MainWindow
from opengraph.ui.mathedit.widget import _FIELD_PADDING, MathEdit
from tests.support import first_id, make_window, wait, wait_until

PHONE = (390.0, 800.0)


@pytest.fixture
def touch() -> None:
    runtime.device.touch = True


@pytest.fixture
def phone(touch: None) -> Iterator[MainWindow]:
    win = make_window()
    win.resize(*PHONE)
    yield win
    win.shutdown()


def editor_of(window: MainWindow) -> MathEdit:
    edit: MathEdit = window.panel.row(first_id(window)).edit
    edit.focus()
    return edit


def soft_type(edit: MathEdit, value: str) -> None:
    """What the client reports when the hidden field's text becomes ``value``."""
    edit.input.value = value
    edit._on_input_change()


# -- layout --------------------------------------------------------------------------------


def test_a_narrow_window_puts_the_graph_above_the_expressions(window: MainWindow) -> None:
    assert not window.compact and isinstance(window.body, ft.Row)
    window.resize(*PHONE)
    assert window.compact and isinstance(window.body, ft.Column)
    assert window.body_host.content is window.body
    assert window.body.controls == [window.graph_area, window.row_splitter, window.panel.control]
    assert window.panel.control.width is None
    assert window.panel.control.height == round(800 * 0.42)
    assert window.coords.visible is False


def test_a_wide_window_returns_to_its_layout(window: MainWindow) -> None:
    width = window.panel.control.width
    window.resize(*PHONE)
    window.resize(COMPACT_WIDTH + 100, 500)
    assert not window.compact and isinstance(window.body, ft.Row)
    assert window.panel.control.width == width and window.panel.control.height is None
    assert window.coords.visible is True
    assert window.settings_dialog.control in window.graph_area.controls


def test_a_landscape_phone_keeps_the_side_by_side_layout(window: MainWindow) -> None:
    window.resize(844, 390)
    assert not window.compact


def test_the_floating_panels_fill_a_narrow_window(window: MainWindow) -> None:
    window.resize(*PHONE)
    settings = window.settings_dialog.control
    assert settings in window.overlay.controls and settings not in window.graph_area.controls
    assert (settings.left, settings.right, settings.top, settings.bottom) == (8, 8, 8, 8)
    assert settings.content.width is None
    assert window.tasks_dialog.control in window.overlay.controls
    assert all(row.wrap for row in window.settings_dialog._axis_rows if isinstance(row, ft.Row))
    window.resize(1200, 800)
    assert settings.left is None and settings.content.width == 520
    assert settings not in window.overlay.controls


def test_the_soft_keyboard_shrinking_the_window_leaves_room_for_the_graph(
    window: MainWindow,
) -> None:
    window.resize(390, 420)
    height = window.panel.control.height
    assert height == PANEL_MIN_HEIGHT or height <= 420 - 120 - 110


def test_the_row_splitter_changes_the_share_and_keeps_it(window: MainWindow) -> None:
    window.resize(*PHONE)
    before = window.panel.control.height

    class Drag:
        primary_delta = -50.0

    window._on_row_splitter_drag(Drag())  # type: ignore[arg-type]
    assert window.panel.control.height == before + 50
    window.resize(390, 700)  # the share of the height stays
    assert window.panel.control.height == round(700 * (before + 50) / 800)


def test_the_row_splitter_stops_at_the_limits(window: MainWindow) -> None:
    window.resize(*PHONE)

    class Drag:
        primary_delta = 5000.0

    window._on_row_splitter_drag(Drag())  # type: ignore[arg-type]
    assert window.panel.control.height == PANEL_MIN_HEIGHT


def test_the_same_controls_work_after_changing_layout(window: MainWindow) -> None:
    window.controller.set_text(first_id(window), "y = x")
    window.controller.recompile()
    window.resize(*PHONE)
    window.canvas.resize(390, 400)
    wait_until(lambda: not window.canvas.is_rendering)
    assert window.canvas.render()


# -- touch-only details --------------------------------------------------------------------


def test_row_actions_are_always_visible_on_touch(phone: MainWindow) -> None:
    row = phone.panel.row(first_id(phone))
    row.current = row.hover = False
    row._style()
    assert row.menu_button.opacity == 1.0 and row.delete_button.opacity == 1.0


def test_row_actions_still_wait_for_the_pointer_on_a_desktop(window: MainWindow) -> None:
    row = window.panel.row(first_id(window))
    row.current = row.hover = False
    row._style()
    assert row.menu_button.opacity == 0.0


def test_menus_show_no_shortcuts_on_touch(phone: MainWindow) -> None:
    file_menu = phone.menu_bar.controls[0]
    assert isinstance(file_menu, ft.SubmenuButton)
    for item in file_menu.controls:
        if isinstance(item, ft.MenuItemButton):
            assert isinstance(item.content, ft.Text)


def test_the_apps_have_no_quit_item(touch: None) -> None:
    runtime.device.native_mobile = True
    win = make_window()
    file_menu = win.menu_bar.controls[0]
    assert isinstance(file_menu, ft.SubmenuButton)
    labels = [
        item.content.value
        for item in file_menu.controls
        if isinstance(item, ft.MenuItemButton) and isinstance(item.content, ft.Text)
    ]
    assert "Save" in labels and "Quit" not in labels
    win.shutdown()


def test_a_finger_scrolls_the_list_instead_of_selecting_text(phone: MainWindow) -> None:
    edit = editor_of(phone)
    assert edit.detector.on_pan_down is None
    assert edit.detector.on_tap_down is not None
    assert edit.menu.primary_trigger == ft.ContextMenuTrigger.LONG_PRESS
    assert edit.menu.primary_items


def test_a_mouse_still_selects_by_dragging(window: MainWindow) -> None:
    edit = editor_of(window)
    assert edit.detector.on_pan_down is not None and edit.menu.items


# -- the soft keyboard ---------------------------------------------------------------------


def test_the_hidden_field_starts_full_so_backspace_can_be_seen(phone: MainWindow) -> None:
    edit = editor_of(phone)
    assert edit.input.value == _FIELD_PADDING
    assert edit.input.keyboard_type == ft.KeyboardType.VISIBLE_PASSWORD
    assert edit.input.capitalization == ft.TextCapitalization.NONE


def test_soft_keyboard_characters_are_typed(phone: MainWindow) -> None:
    edit = editor_of(phone)
    soft_type(edit, _FIELD_PADDING + "x")
    soft_type(edit, _FIELD_PADDING + "x+")
    assert edit.text() == "x+"


def test_soft_keyboard_backspace_deletes(phone: MainWindow) -> None:
    edit = editor_of(phone)
    edit.type_text("abc")
    soft_type(edit, _FIELD_PADDING[:-1])
    assert edit.text() == "ab"
    soft_type(edit, _FIELD_PADDING[:-3])  # several at once, when events bunch up
    assert edit.text() == ""


def test_backspace_works_after_typing_on_the_soft_keyboard(phone: MainWindow) -> None:
    edit = editor_of(phone)
    soft_type(edit, _FIELD_PADDING + "xy")
    soft_type(edit, _FIELD_PADDING + "x")
    assert edit.text() == "x"


def test_the_field_is_refilled_before_it_runs_out(phone: MainWindow) -> None:
    edit = editor_of(phone)
    edit.type_text("a" * 100)
    soft_type(edit, _FIELD_PADDING[:3])  # held down: nearly everything is gone
    assert edit.input.value == _FIELD_PADDING
    assert edit.text() == "a" * (100 - 61)
    soft_type(edit, _FIELD_PADDING[:-1])  # and it goes on working
    assert edit.text() == "a" * (100 - 62)


def test_the_field_does_not_grow_without_end(phone: MainWindow) -> None:
    edit = editor_of(phone)
    soft_type(edit, _FIELD_PADDING + "1" * 600)
    assert edit.input.value == _FIELD_PADDING
    assert edit.text() == "1" * 600


def test_the_keyboard_s_enter_key_adds_a_row(phone: MainWindow) -> None:
    edit = editor_of(phone)
    before = len(phone.controller.document.items)
    assert edit.input.on_submit is not None
    edit.input.on_submit(None)  # type: ignore[arg-type]
    assert len(phone.controller.document.items) == before + 1


def test_desktop_typing_is_unchanged(window: MainWindow) -> None:
    edit = editor_of(window)
    assert edit.input.value == ""
    soft_type(edit, "xy")
    assert edit.text() == "xy"
    soft_type(edit, "x")  # on a desktop Backspace arrives as a key, not through the field
    assert edit.text() == "xy"


# -- the row of keys -----------------------------------------------------------------------


def test_the_keypad_shows_while_an_expression_is_edited(phone: MainWindow) -> None:
    edit = editor_of(phone)
    assert phone.keypad.control.visible
    edit._set_focus(False)
    assert not phone.keypad.control.visible


def test_the_keypad_stays_away_on_a_desktop(window: MainWindow) -> None:
    editor_of(window)
    assert not window.keypad.control.visible


@pytest.mark.parametrize(
    ("actions", "source"),
    [
        (["x", "^", "2"], "x^2"),
        (["x", "^", "2", "Arrow Right", "+", "y"], "x^2+y"),
        (["sqrt(", "x"], "sqrt(x)"),
        (["x", "Backspace"], ""),
        (["π"], "π"),
        (["θ"], "θ"),
        (["z", "=", "x", "*", "y"], "z=x*y"),
    ],
)
def test_keys_act_on_the_editor(phone: MainWindow, actions: list[str], source: str) -> None:
    edit = editor_of(phone)
    for action in actions:
        if action == "2":  # the soft keyboard's job
            edit.type_text(action)
        else:
            assert keypad.press(action)
    assert edit.text().replace(" ", "") == source.replace(" ", "")


def test_keys_do_nothing_without_an_editor(phone: MainWindow) -> None:
    editor_of(phone)._set_focus(False)
    assert not keypad.press("x")


def test_every_key_acts_without_error(phone: MainWindow) -> None:
    editor_of(phone)
    for _, action in keypad.KEYS:
        assert keypad.press(action)


# -- gestures on the graph -----------------------------------------------------------------


def test_touch_uses_the_scale_gesture(phone: MainWindow) -> None:
    detector = phone.canvas.detector
    assert detector.on_scale_update is not None and detector.on_pan_update is None
    assert detector.on_long_press_start is not None


def test_a_mouse_keeps_the_pan_gesture(window: MainWindow) -> None:
    detector = window.canvas.detector
    assert detector.on_pan_update is not None and detector.on_scale_update is None
    assert detector.on_hover is not None and detector.on_scroll is not None


def test_one_finger_pans_the_plane(phone: MainWindow) -> None:
    canvas = phone.canvas
    center = canvas.viewport.center_x
    canvas.touch_start(Point(100, 100), 1)
    canvas.touch_move(Point(150, 100), 1.0, 1)
    canvas.touch_end()
    assert canvas.viewport.center_x < center


def test_a_pinch_zooms_about_the_fingers(phone: MainWindow) -> None:
    canvas = phone.canvas
    viewport = canvas.viewport
    scale = viewport.scale_x
    at = Point(300, 200)
    world = viewport.to_world(at.x, at.y)
    canvas.touch_start(at, 2)
    canvas.touch_move(at, 1.5, 2)
    canvas.touch_move(at, 2.0, 2)
    canvas.touch_end()
    assert viewport.scale_x == pytest.approx(scale * 2)
    after = viewport.to_world(at.x, at.y)
    assert float(after[0]) == pytest.approx(float(world[0]))  # what is under them stays
    assert float(after[1]) == pytest.approx(float(world[1]))


def test_two_fingers_moving_pan_the_plane(phone: MainWindow) -> None:
    canvas = phone.canvas
    center = canvas.viewport.center_x
    canvas.touch_start(Point(100, 100), 2)
    canvas.touch_move(Point(160, 100), 1.0, 2)
    canvas.touch_end()
    assert canvas.viewport.center_x < center


def test_a_finger_can_drag_a_point(phone: MainWindow) -> None:
    item = first_id(phone)
    phone.controller.set_text(item, "point[1, 2]")
    phone.controller.recompile()
    canvas = phone.canvas
    x, y = canvas.viewport.to_screen(1.0, 2.0)
    # Not quite on it: a finger is less exact than a mouse.
    canvas.touch_start(Point(float(x) + 18, float(y)), 1)
    canvas.touch_move(Point(float(x) + 118, float(y)), 1.0, 1)
    canvas.touch_end()
    phone.controller.flush()
    assert phone.controller.document.get(item).text != "point[1, 2]"


def test_a_tap_reads_a_curve_and_clears_on_empty_space(phone: MainWindow) -> None:
    phone.controller.set_text(first_id(phone), "y = x")
    phone.controller.recompile()
    canvas = phone.canvas
    wait_until(lambda: not canvas.is_rendering)
    canvas.render()
    x, y = canvas.viewport.to_screen(1.0, 1.0)
    canvas.tap(Point(float(x), float(y) + 15))  # a mouse would have to be within 12 pixels
    assert canvas._trace is not None
    canvas.tap(Point(float(x) + 300, float(y)))
    assert canvas._trace is None


def test_a_long_press_follows_the_curve(phone: MainWindow) -> None:
    phone.controller.set_text(first_id(phone), "y = x")
    phone.controller.recompile()
    canvas = phone.canvas
    wait_until(lambda: not canvas.is_rendering)
    canvas.render()
    x, y = canvas.viewport.to_screen(1.0, 1.0)
    canvas.read_at(Point(float(x), float(y)))
    first = canvas._trace
    x, y = canvas.viewport.to_screen(2.0, 2.0)
    canvas.read_at(Point(float(x), float(y)))
    assert first is not None and canvas._trace is not None and canvas._trace != first


def show_3d(window: MainWindow) -> None:
    window.controller.set_text(first_id(window), "z = x")
    window.controller.recompile()


def test_one_finger_turns_a_3d_view(phone: MainWindow) -> None:
    show_3d(phone)
    canvas = phone.canvas
    camera = canvas.view3d.camera
    canvas.touch_start(Point(100, 100), 1)
    canvas.touch_move(Point(160, 130), 1.0, 1)
    canvas.touch_end()
    moved = canvas.view3d.camera
    assert moved.azimuth != camera.azimuth and moved.box == camera.box


def test_a_pinch_zooms_a_3d_view(phone: MainWindow) -> None:
    show_3d(phone)
    canvas = phone.canvas
    azimuth = canvas.view3d.camera.azimuth
    canvas.touch_start(Point(100, 100), 2)
    canvas.touch_move(Point(100, 100), 2.0, 2)
    canvas.touch_end()
    assert canvas.view3d.camera.box.x_max == pytest.approx(2.5)
    assert canvas.view3d.camera.azimuth == azimuth


def test_two_fingers_pan_a_3d_view_without_a_shift_key(phone: MainWindow) -> None:
    show_3d(phone)
    canvas = phone.canvas
    camera = canvas.view3d.camera
    canvas.touch_start(Point(100, 100), 2)
    canvas.touch_move(Point(160, 100), 1.0, 2)
    canvas.touch_end()
    moved = canvas.view3d.camera
    assert moved.box != camera.box and moved.azimuth == camera.azimuth
    assert moved.box.x_max - moved.box.x_min == pytest.approx(10.0)


def test_finer_frames_wait_for_the_fingers_to_lift(phone: MainWindow) -> None:
    show_3d(phone)
    canvas = phone.canvas
    wait_until(lambda: not canvas.view3d.loading, 10_000)
    canvas.touch_start(Point(100, 100), 2)
    canvas.touch_move(Point(120, 100), 1.0, 2)
    wait(400)
    assert canvas.view3d.frame is None or not canvas.view3d.frame.still
    canvas.touch_end()
    wait_until(lambda: not canvas.view3d.loading, 10_000)
    assert canvas.view3d.frame is not None and canvas.view3d.frame.still


def test_a_tap_reads_a_surface(phone: MainWindow) -> None:
    show_3d(phone)
    canvas = phone.canvas
    wait_until(lambda: not canvas.view3d.loading, 10_000)
    frame = canvas.view3d.frame
    assert frame is not None
    x, y, _ = frame.camera.project(np.zeros((1, 3)), frame.width, frame.height)
    canvas.tap(Point(float(x[0]), float(y[0])))
    assert canvas._trace is not None and canvas._trace.label.startswith("(")


# -- saving --------------------------------------------------------------------------------


class FakePicker:
    """A client's file chooser: remembers what it is given, and how it answers."""

    def __init__(self, answer: str | None) -> None:
        self.answer = answer
        self.saved: list[dict[str, Any]] = []
        self.asked: list[dict[str, Any]] = []

    async def save_file(self, **arguments: Any) -> str | None:
        self.saved.append(arguments)
        return self.answer

    async def pick_files(self, **arguments: Any) -> list[Any]:
        self.asked.append(arguments)
        return []


class FakePage:
    web = False

    def show_dialog(self, _dialog: object) -> None:
        pass


@pytest.fixture
def app(phone: MainWindow, monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    """A phone window whose jobs on the event loop run at once."""
    runtime.device.native_mobile = True
    dialogs.host.page = FakePage()  # type: ignore[assignment]

    def spawn(make: Any) -> bool:
        coroutine = make()
        try:
            while True:
                coroutine.send(None)
        except StopIteration:
            pass
        return True

    monkeypatch.setattr(runtime, "spawn", spawn)
    yield phone


def test_saving_hands_the_document_to_the_file_chooser(app: MainWindow) -> None:
    picker = dialogs.host.files = FakePicker("/storage/graph.ograph.json")  # type: ignore[assignment]
    app.controller.set_text(first_id(app), "z = x y")
    app.controller.flush()
    app.controller.recompile()
    assert app.controller.modified
    done: list[bool] = []
    app.save_file(lambda: done.append(True))
    arguments = picker.saved[0]  # type: ignore[attr-defined]
    assert arguments["file_name"] == "graph.ograph.json"
    assert b"z = x y" in arguments["src_bytes"]
    assert not app.controller.modified and done == [True]
    assert app.path == Path("/storage/graph.ograph.json")


def test_cancelling_the_file_chooser_keeps_the_changes_unsaved(app: MainWindow) -> None:
    dialogs.host.files = FakePicker(None)  # type: ignore[assignment]
    app.controller.set_text(first_id(app), "y = x")
    app.controller.flush()
    app.controller.recompile()
    done: list[bool] = []
    app.save_file(lambda: done.append(True))
    assert app.controller.modified and done == []


def test_saving_again_asks_again(app: MainWindow) -> None:
    """An opened file's path is a temporary copy there: writing to it would lose the work."""
    picker = dialogs.host.files = FakePicker("/storage/a.ograph.json")  # type: ignore[assignment]
    app.path = Path("/cache/copy.ograph.json")
    app.save_file()
    assert len(picker.saved) == 1  # type: ignore[attr-defined]
    assert not Path("/cache/copy.ograph.json").exists()


def test_exporting_hands_the_image_to_the_file_chooser(
    app: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    picker = dialogs.host.files = FakePicker("/storage/graph.png")  # type: ignore[assignment]

    async def capture() -> bytes:
        return b"PNGDATA"

    monkeypatch.setattr(app, "capture", capture)
    app.export_image()
    wait_until(lambda: picker.saved)  # type: ignore[attr-defined]
    assert picker.saved[0]["src_bytes"] == b"PNGDATA"  # type: ignore[attr-defined]
    assert "Exported graph.png" in app.current_message()


def test_opening_shows_every_file_on_touch(app: MainWindow) -> None:
    """Android and iOS don't know .ograph.json as a type to filter by."""
    picker = dialogs.host.files = FakePicker(None)  # type: ignore[assignment]
    app.controller.mark_saved()
    app.open_file()
    arguments = picker.asked[0]  # type: ignore[attr-defined]
    assert arguments["file_type"] == ft.FilePickerFileType.ANY
    assert arguments["allowed_extensions"] is None


def test_desktops_still_save_to_a_path(window: MainWindow, tmp_path: Path) -> None:
    assert not dialogs.host.saves_through_client
    window.path = tmp_path / "a.ograph.json"
    assert window.save_file()
    assert window.path.exists()


# -- the app's life ------------------------------------------------------------------------


def test_going_to_the_background_keeps_the_work(app: MainWindow) -> None:
    app.controller.set_text(first_id(app), "z = x + y")
    app.controller.flush()
    app.controller.recompile()
    app.on_lifecycle(ft.AppLifecycleState.PAUSE)
    path = preferences.config_dir() / "autosave.ograph.json"
    assert "z = x + y" in path.read_text(encoding="utf-8")

    restored = MainWindow(preferences=app.preferences)
    assert restored.restore_autosave()
    assert restored.controller.document.items[0].text == "z = x + y"
    assert restored.controller.modified  # it still hasn't been saved anywhere
    restored.shutdown()


def test_coming_back_does_not_write_anything(app: MainWindow) -> None:
    app.on_lifecycle(ft.AppLifecycleState.RESUME)
    assert not (preferences.config_dir() / "autosave.ograph.json").exists()


def test_desktops_do_not_autosave(window: MainWindow) -> None:
    window.on_lifecycle(ft.AppLifecycleState.PAUSE)
    assert not (preferences.config_dir() / "autosave.ograph.json").exists()


def test_a_damaged_autosave_is_ignored(app: MainWindow) -> None:
    path = preferences.config_dir() / "autosave.ograph.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    assert not app.restore_autosave()


def test_nothing_to_restore(app: MainWindow) -> None:
    assert not app.restore_autosave()


def test_the_app_s_data_folder_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENGRAPH_CONFIG_DIR")
    monkeypatch.setenv("FLET_APP_STORAGE_DATA", "/data/app")
    assert preferences.config_dir() == Path("/data/app")


def test_opening_the_app_does_not_pop_up_the_keyboard(phone: MainWindow) -> None:
    phone.controller.new_document()
    assert not phone.keypad.control.visible
    assert phone.panel.row(first_id(phone)).edit.has_focus is False


def test_the_client_finds_the_fonts_where_the_app_is_packaged() -> None:
    """flet build takes a client's assets from the "assets" folder of the app's path."""
    root = Path(__file__).resolve().parent.parent / "src"
    assert (root / "main.py").is_file()
    for name in fonts.FILES.values():
        assert (root / "assets" / name).is_file()
        assert (fonts.ASSETS_DIR / name).is_file()


def test_the_reference_explains_touch() -> None:
    from opengraph.ui.main_window import syntax_reference

    assert "pinch" in syntax_reference()
