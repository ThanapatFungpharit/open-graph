from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengraph.document import PALETTE, Document, DocumentError, SliderRange
from opengraph.engine.compiler import compile_document


def test_add_remove_move() -> None:
    doc = Document()
    a = doc.add("y = x")
    b = doc.add("y = 2x")
    c = doc.add("y = 3x", index=0)
    assert [i.id for i in doc.items] == [c.id, a.id, b.id]
    assert len({a.id, b.id, c.id}) == 3
    assert doc.remove(a.id) == 1
    doc.move(b.id, 0)
    assert [i.id for i in doc.items] == [b.id, c.id]
    with pytest.raises(KeyError):
        doc.get(a.id)


def test_colors_cycle_through_palette_least_used_first() -> None:
    doc = Document()
    colors = [doc.add().color for _ in range(len(PALETTE))]
    assert colors == list(PALETTE)
    doc.remove(doc.items[2].id)  # frees the third colour
    assert doc.add().color == PALETTE[2]


def test_independent_visibility() -> None:
    doc = Document()
    a, b = doc.add("y = x"), doc.add("y = -x")
    b.visible = False
    results = compile_document(doc.rows())
    drawn = [i.id for i in doc.items if i.visible and results[i.id].plot is not None]
    assert drawn == [a.id]
    assert results[b.id].plot is not None  # hidden rows still compile


def test_json_round_trip(tmp_path: Path) -> None:
    doc = Document()
    doc.add("f(x) = x^2", color="#123456")
    doc.add("y = f(x) + 1", visible=False)
    doc.view = {"center_x": 1.0, "center_y": 2.0, "scale_x": 30.0, "scale_y": 30.0}
    path = tmp_path / "graph.ograph.json"
    doc.save(path)
    loaded = Document.load(path)
    assert [(i.text, i.color, i.visible) for i in loaded.items] == [
        ("f(x) = x^2", "#123456", True),
        ("y = f(x) + 1", doc.items[1].color, False),
    ]
    assert loaded.view == doc.view


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps([1, 2]),
        json.dumps({"format": "other"}),
        json.dumps({"format": "opengraph", "version": 99}),
        json.dumps({"format": "opengraph", "version": 1, "expressions": [{"text": 5}]}),
    ],
)
def test_invalid_files_raise_document_error(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(DocumentError):
        Document.load(path)


def test_bad_colors_are_replaced_not_trusted() -> None:
    data = {
        "format": "opengraph",
        "version": 1,
        "expressions": [{"text": "x", "color": "red; background: url(x)"}],
    }
    assert Document.from_dict(data).items[0].color in PALETTE


def test_unknown_fields_do_not_invalidate_document_data() -> None:
    data = {
        "format": "opengraph",
        "version": 1,
        "expressions": [{"text": "y = x", "future_option": object()}],
        "settings": {"degrees": True, "future_option": object()},
        "future_option": object(),
    }
    doc = Document.from_dict(data)
    assert doc.items[0].text == "y = x"
    assert doc.settings.degrees


def test_boolean_document_version_is_not_an_integer_version() -> None:
    with pytest.raises(DocumentError):
        Document.from_dict({"format": "opengraph", "version": True})


def test_slider_range_round_trips_and_is_optional(tmp_path: Path) -> None:
    doc = Document()
    doc.add("k = 3", slider=SliderRange(0.0, 5.0, 0.25))
    doc.add("y = kx")
    data = doc.to_dict()
    assert data["version"] == 1  # optional key: no format bump
    assert data["expressions"][0]["slider"] == {"min": 0.0, "max": 5.0, "step": 0.25}
    assert "slider" not in data["expressions"][1]
    path = tmp_path / "s.ograph.json"
    doc.save(path)
    loaded = Document.load(path)
    assert [i.slider for i in loaded.items] == [SliderRange(0.0, 5.0, 0.25), None]


def test_slider_expressions_round_trip() -> None:
    doc = Document()
    doc.add("k = 1", slider=SliderRange(0.0, 6.25, 0.25, max_expr="2.5^2"))
    data = doc.to_dict()
    assert data["expressions"][0]["slider"] == {
        "min": 0.0,
        "max": 6.25,
        "step": 0.25,
        "max_expr": "2.5^2",
    }
    loaded = Document.from_dict(data).items[0].slider
    assert loaded == SliderRange(0.0, 6.25, 0.25, max_expr="2.5^2")
    assert loaded.texts == ("0", "2.5^2", "0.25")


def test_slider_range_from_texts() -> None:
    r = SliderRange.from_texts(["-1", " 2π ", "0.5"], [-1.0, 6.28, 0.5])
    assert r == SliderRange(-1, 6.28, 0.5, max_expr="2π")
    with pytest.raises(ValueError, match="Slider step: Unknown name 'q'"):
        SliderRange.from_texts(["0", "1", "q"], [0.0, 1.0, "Unknown name 'q'"])
    assert r.with_values([-2.0, 3.0, 0.5]) == SliderRange(-2, 3, 0.5, max_expr="2π")


def test_expression_bounds_are_not_widened() -> None:
    r = SliderRange(0, 2, 0.1, max_expr="2a")
    assert r.including(5) is r
    assert r.including(-1) == SliderRange(-1, 2, 0.1, max_expr="2a")


@pytest.mark.parametrize(
    "slider",
    [
        "fast",
        {"min": 1, "max": 0, "step": 1},
        {"min": 0, "max": 1, "step": 0},
        {"min": 0, "max": 1},
        {"min": True, "max": 2, "step": 1},
        {"min": 0, "max": 1, "step": 1e-9},
    ],
)
def test_malformed_slider_ranges_are_dropped(slider: object) -> None:
    data = {
        "format": "opengraph",
        "version": 1,
        "expressions": [{"text": "k = 1", "slider": slider}],
    }
    assert Document.from_dict(data).items[0].slider is None


def test_slider_default_ranges() -> None:
    assert SliderRange.default_for(3) == SliderRange(-10, 10, 0.1)
    assert SliderRange.default_for(-10) == SliderRange(-10, 10, 0.1)
    assert SliderRange.default_for(25) == SliderRange(15, 35, 0.1)
    huge = SliderRange.default_for(-4e12)
    assert huge.min < -4e12 < huge.max and huge.steps == 200


def test_slider_range_includes_values_by_moving_one_bound() -> None:
    r = SliderRange(0, 1, 0.1)
    assert r.including(0.5) is r
    assert r.including(3) == SliderRange(0, 3, 0.1)
    assert r.including(-2) == SliderRange(-2, 1, 0.1)
    assert r.including(1e9) == SliderRange.default_for(1e9)  # too many steps otherwise


def test_slider_positions_map_to_clean_values() -> None:
    r = SliderRange(-0.3, 1, 0.1)
    assert r.steps == 13
    assert r.value_at(3) == 0.0  # not 5.55e-17
    assert r.value_at(13) == 1.0
    assert r.position_of(0.71) == 10
    assert r.position_of(50) == r.steps and r.position_of(-50) == 0
    odd = SliderRange(0, 1, 0.3)  # span isn't a whole number of steps
    assert odd.steps == 4 and odd.value_at(4) == 1.0
    assert SliderRange(0, 1e-5, 2.5e-7).value_at(1) == 2.5e-7


@pytest.mark.parametrize(
    ("lo", "hi", "step", "message"),
    [
        (1, 1, 0.1, "less than"),
        (0, 1, -1, "positive"),
        (0, 1, 1e-7, "too small"),
        (0, float("inf"), 1, "finite"),
    ],
)
def test_slider_range_validation(lo: float, hi: float, step: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SliderRange.validated(lo, hi, step)
