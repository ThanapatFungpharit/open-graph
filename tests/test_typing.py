"""Plain-text (linear) input, LaTeX-style commands and LaTeX in/out, incomplete input."""

from __future__ import annotations

import pytest

from opengraph.engine.compiler import compile_expression
from opengraph.engine.errors import ParseError
from opengraph.engine.parser import parse
from opengraph.ui import runtime
from opengraph.ui.main_window import MainWindow
from opengraph.ui.mathedit.editing import EditorModel
from opengraph.ui.mathedit.latex import latex_to_source, to_latex
from opengraph.ui.mathedit.nodes import BigOp, Char, Frac, Position, Root, Sup, Symbol
from opengraph.ui.mathedit.source import from_source
from opengraph.ui.mathedit.widget import MathEdit, _inserted
from tests.support import key as press_key
from tests.support import key_clicks


def typed(keys: str, model: EditorModel | None = None) -> EditorModel:
    model = model or EditorModel()
    for key in keys:
        if key in "→←":
            model.move(1 if key == "→" else -1)
        else:
            model.type_text(key)
    return model


@pytest.mark.parametrize("text", ["pi+1", "sqrt(x)+1", "sum ", r"\alpha ", "sum(k,k,1,3)"])
def test_batched_typing_is_one_complete_undo_step(text: str) -> None:
    model = EditorModel("a+")
    model.type_text(text)
    result, caret = model.source, model.describe()
    assert model.undo()
    assert model.source == "a+" and model.describe() == "a+|"
    assert not model.can_undo
    assert model.redo()
    assert (model.source, model.describe()) == (result, caret)


@pytest.mark.parametrize("command", ["backspace", "delete_forward", "delete_selection"])
def test_no_op_edit_preserves_redo(command: str) -> None:
    model = typed("x")
    assert model.undo()
    assert not getattr(model, command)()
    assert model.can_redo and not model.can_undo
    assert model.redo() and model.source == "x"


def test_typing_after_a_no_op_that_moves_the_caret_has_its_own_undo_step() -> None:
    model = typed("1/2")
    model.move_to_edge(1)
    assert not model.backspace()  # enters the denominator without deleting anything
    model.type_text("3")
    assert model.undo()
    assert model.source == "1/2"
    assert model.describe() == r"\frac{1}{2|}"


def test_no_op_at_history_limit_keeps_all_undo_steps() -> None:
    model = EditorModel()
    for _ in range(200):
        model.move_to_edge(1)
        model.type_text("x")
    assert not model.delete_forward()
    for _ in range(200):
        assert model.undo()
    assert model.source == "" and not model.can_undo


@pytest.mark.parametrize("text", ["", "\n\t "])
def test_empty_paste_keeps_the_selection_and_history(text: str) -> None:
    model = typed("abc")
    model.select_all()
    assert not model.paste(text)
    assert model.source == "abc" and model.selected_source() == "abc"
    assert model.undo() and model.source == ""


@pytest.mark.parametrize(("name", "prefix"), [("sqrtvalue", 4), ("pix", 2), ("summit", 3)])
def test_delimiter_inside_a_name_does_not_convert_its_prefix(name: str, prefix: int) -> None:
    model = EditorModel(name)
    model.set_cursor(Position(model.root, prefix))
    model.type_text("(")
    assert not any(isinstance(node, Root | Symbol | BigOp) for node in model.root.nodes)
    assert model.source == name[:prefix] + "()" + name[prefix:]


def test_tab_visits_fraction_fields_and_exits_nested_structures() -> None:
    model = typed("1/(x^2")
    assert model.tab()  # exponent -> parentheses
    assert model.cursor.lst.parent is not None
    assert model.tab()  # parentheses -> denominator
    assert model.tab(-1)  # denominator -> numerator
    assert model.cursor.lst is model.root.nodes[0].blocks[0]
    assert model.tab(-1)  # numerator -> before fraction
    assert model.cursor == Position(model.root, 0)
    assert not model.tab()


def test_tab_completes_and_fills_big_operator_in_the_widget() -> None:
    edit = MathEdit()
    edits: list[str] = []
    edit.text_edited.connect(edits.append)
    key_clicks(edit, "sum")
    assert press_key(edit, "Tab")
    op = edit.model.root.nodes[0]
    assert isinstance(op, BigOp) and edit.model.cursor.lst is op.var
    for text, target in [("k", op.lower), ("1", op.upper), ("n", op.body)]:
        key_clicks(edit, text)
        before = len(edits)
        assert press_key(edit, "Tab")
        assert edit.model.cursor.lst is target and len(edits) == before
    key_clicks(edit, "k")
    assert edit.text() == "sum(k, k, 1, n)"
    assert press_key(edit, "Tab", shift=True)
    assert edit.model.cursor.lst is op.upper
    assert press_key(edit, "Tab")
    assert press_key(edit, "Tab")  # leave operator
    assert edit.has_focus
    assert not press_key(edit, "Tab")  # leave the editor: the client moves the focus on


def test_tab_in_linear_input_keeps_normal_focus_navigation() -> None:
    edit = MathEdit("sum")
    edit.set_linear(True)
    assert not press_key(edit, "Tab")
    assert edit.text() == "sum"


@pytest.mark.parametrize(
    ("old", "new", "typed"),
    [
        ("", "a", "a"),
        ("ab", "abc", "c"),
        ("abc", "xabc", "x"),  # typed after Home moved the field's own caret
        ("abc", "ab", ""),  # Backspace: handled as a key instead
        ("abc", "aXc", "X"),  # a selection in the field replaced
        ("", "@", "@"),  # AltGr text arrives like any other character
    ],
)
def test_text_typed_into_the_hidden_field_is_recovered(old: str, new: str, typed: str) -> None:
    assert _inserted(old, new) == typed


def test_hidden_field_text_reaches_the_editor() -> None:
    edit = MathEdit()
    edit.focus()
    edits: list[str] = []
    edit.text_edited.connect(edits.append)
    for value in ("@", "@x", "@x+"):  # what the field reports, keystroke by keystroke
        edit.input.value = value
        edit._on_input_change()
    assert edit.text() == "@x+" and len(edits) == 3
    # Command shortcuts are editing keys, never text.
    assert press_key(edit, "X", ctrl=True)
    assert not press_key(edit, "X", alt=True)
    assert edit.text() == "@x+"


def test_batched_input_undoes_and_redoes_as_one_edit() -> None:
    edit = MathEdit("a+")
    edit.end()
    edits: list[str] = []
    edit.text_edited.connect(edits.append)
    edit.type_text("sqrt(x)+1")
    result = edit.text()
    assert result == "a+sqrt(x)+1" and edits == [result]
    edit.undo()
    assert edit.text() == "a+"
    edit.redo()
    assert edit.text() == result


def test_field_redo_after_ctrl_y_is_not_typed_again() -> None:
    edit = MathEdit()
    edit.focus()
    edit.input.value = "ab"
    edit._on_input_change()
    assert press_key(edit, "Z", ctrl=True)
    edit.input.value = ""  # the field undoes its own text too
    edit._on_input_change()
    assert press_key(edit, "Y", ctrl=True)
    edit.input.value = "ab"  # ...and redoes it: not new typing
    edit._on_input_change()
    assert edit.text() == "ab"


# -- incomplete expressions ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["x^", "sqrt(", "1 +", "(x + 1", "[1, 2", "|x", "sin(x", "y = x^", "f(x) = "]
)
def test_unfinished_text_is_flagged_incomplete(text: str) -> None:
    result = compile_expression(text)
    assert result.is_error and result.incomplete


@pytest.mark.parametrize("text", ["x)", "1 2", "$", "x + * 2", "()", "sqrt(x))"])
def test_wrong_text_is_not_incomplete(text: str) -> None:
    result = compile_expression(text)
    assert result.is_error and not result.incomplete


def test_parse_error_carries_the_flag() -> None:
    with pytest.raises(ParseError) as info:
        parse("2 *")
    assert info.value.incomplete


def test_unfinished_row_keeps_its_curve_and_stays_calm(window: MainWindow) -> None:
    c = window.controller
    row_id = c.document.items[0].id
    c.set_text(row_id, "y = x^2")
    c.flush()
    row = window.panel.row(row_id)
    assert len(window.canvas._layers) == 1
    c.set_text(row_id, "y = x^2 +")  # typing on
    c.flush()
    assert len(window.canvas._layers) == 1  # the last drawing stays
    assert row.error is False  # no red state
    assert "incomplete" in (row.feedback.value or "").lower()
    assert row.edit._error_span is None  # nothing underlined
    c.set_text(row_id, "y = x^2 + 1")
    c.flush()
    assert not row.feedback.visible


def test_a_real_error_still_removes_the_curve_and_underlines(window: MainWindow) -> None:
    c = window.controller
    row_id = c.document.items[0].id
    c.set_text(row_id, "y = x^2")
    c.flush()
    c.set_text(row_id, "y = x^2 )")
    c.flush()
    row = window.panel.row(row_id)
    assert row.error is True
    assert not window.canvas._layers


# -- plain-text (linear) mode ----------------------------------------------------------------


def test_linear_mode_types_characters_verbatim() -> None:
    model = EditorModel(linear=True)
    typed("x^2/(3+y)", model)
    assert model.source == "x^2/(3+y)"
    assert all(isinstance(n, Char) for n in model.root.nodes)  # no fractions, no exponents


def test_switching_modes_keeps_the_expression() -> None:
    model = typed("x^2/3")
    source = model.source
    model.move_to_edge(1)
    model.set_linear(True)
    assert model.source == source and model.linear
    assert all(isinstance(n, Char) for n in model.root.nodes)
    typed("+1", model)
    model.set_linear(False)
    assert model.source == source + "+1"
    assert any(isinstance(n, Frac | Sup) for n in model.root.nodes)


def test_switching_keeps_undo_history() -> None:
    model = typed("ab")
    model.set_linear(True)
    typed("c", model)
    assert model.source == "abc"
    model.set_linear(False)
    assert model.undo()
    assert model.source == "ab"
    assert model.undo()
    assert model.source in ("a", "")


def test_linear_caret_follows_the_typeset_caret() -> None:
    model = typed("12+x")
    model.move(-1)  # between + and x
    model.set_linear(True)
    typed("*", model)
    assert model.source == "12+*x"


def test_linear_editing_deletes_and_selects_characters() -> None:
    model = EditorModel("sin(x)", linear=True)
    model.backspace()
    assert model.source == "sin(x"
    model.select_all()
    typed("y", model)
    assert model.source == "y"


def test_linear_copy_is_the_source_text() -> None:
    model = EditorModel("a^2+b", linear=True)
    model.select_all()
    assert model.selected_source() == "a^2+b"


def test_widget_modes_render_and_report() -> None:
    edit = MathEdit("1/2 + sqrt(x)")
    typeset_height = edit.content_height()
    edit.set_linear(True)
    edit.render()
    assert edit.is_linear and edit.text() == "1/2 + sqrt(x)"
    assert edit.content_height() < typeset_height  # one flat line is shorter than ½
    edit.set_linear(False)
    assert not edit.is_linear and edit.text() == "1/2 + sqrt(x)"


def test_menu_toggle_applies_to_every_row_and_new_rows(window: MainWindow) -> None:
    c = window.controller
    second = c.add_expression("x/2")
    window.set_linear_input(True)
    first = c.document.items[0].id
    assert window.panel.row(first).edit.is_linear and window.panel.row(second).edit.is_linear
    third = c.add_expression("y")
    assert window.panel.row(third).edit.is_linear
    assert window.panel.row(second).edit.text() == "x/2"
    window.set_linear_input(False)
    assert not window.panel.row(second).edit.is_linear
    assert window.panel.row(second).edit.text() == "x/2"


def test_linear_input_edits_still_reach_the_controller(window: MainWindow) -> None:
    window.set_linear_input(True)
    row_id = window.controller.document.items[0].id
    edit = window.panel.row(row_id).edit
    key_clicks(edit, "y=x^2")
    window.controller.flush()
    assert window.controller.document.get(row_id).text == "y=x^2"
    assert len(window.canvas._layers) == 1


# -- LaTeX-style commands --------------------------------------------------------------------


def test_backslash_greek_and_relations() -> None:
    model = typed(r"\alpha ")
    assert [type(n) for n in model.root.nodes] == [Symbol] and model.source == "alpha"
    assert typed(r"x\le 3").source == "x<=3"
    assert typed(r"2\cdot 3").source == "2*3"


def test_backslash_structures() -> None:
    model = typed(r"\frac 1→2")
    assert isinstance(model.root.nodes[0], Frac) and model.source == "1/2"
    model = typed(r"\sqrt x")
    assert isinstance(model.root.nodes[0], Root) and model.source == "sqrt(x)"
    model = typed(r"\sum ")
    assert isinstance(model.root.nodes[0], BigOp)


def test_backslash_function_and_terminator() -> None:
    model = typed(r"\sin(x)")
    assert model.source == "sin(x)"
    assert typed(r"\pi+1").source == "pi+1"  # a non-space terminator is typed afterwards


def test_backslash_command_is_one_undo_step() -> None:
    model = typed(r"\alpha ")
    assert model.undo()
    assert model.source == "\\alpha" or model.source == "\\alpha "
    assert not any(isinstance(n, Symbol) for n in model.root.nodes)


def test_unknown_backslash_command_is_left_alone() -> None:
    model = typed(r"\foo x")
    assert model.source.startswith("\\foo")


def test_letters_after_a_backslash_do_not_auto_convert() -> None:
    model = typed(r"\pi")  # pending until ended
    assert all(isinstance(n, Char) for n in model.root.nodes)
    typed(" ", model)
    assert isinstance(model.root.nodes[0], Symbol)


# -- LaTeX in and out ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("latex", "source"),
    [
        (r"\frac{1}{2}", "1/2"),
        (r"\frac{x+1}{2}+3", "(x+1)/2+3"),
        (r"2\frac{1}{x}", "2(1/x)"),
        (r"\frac{1}{2}^2", "(1/2)^2"),
        (r"\sqrt{x^2+1}", "sqrt(x^2+1)"),
        (r"\sqrt[3]{x}", "cbrt(x)"),
        (r"\alpha^{2}+y_{1}", "alpha^2+y_1"),
        (r"\pi^{n+1}", "pi^(n+1)"),
        (r"\left(x+1\right)^2", "(x+1)^2"),
        (r"\alpha\cdot\pi", "alpha*pi"),
        (r"x \le 3", "x<=3"),
        (r"\sin\left(x\right)", "sin(x)"),
        (r"\sin x", "sin(x)"),
        (r"\sum_{k=1}^{n} k^2", "sum(k^2, k, 1, n)"),
        (r"\int_0^1 x^2\,dx", "int(x^2, x, 0, 1)"),
        (r"\lim_{x\to 0}\frac{\sin x}{x}", "limit(sin(x)/x, x, 0)"),
        (r"$\frac{a}{b}$", "a/b"),
    ],
)
def test_latex_to_source(latex: str, source: str) -> None:
    assert latex_to_source(latex) == source


@pytest.mark.parametrize("text", ["x^2", r"\foo{x}", r"\frac{1}", "", r"x^"])
def test_latex_to_source_declines_other_text(text: str) -> None:
    assert latex_to_source(text) is None


@pytest.mark.parametrize(
    "latex",
    [
        r"\frac{1}{2}",
        r"\frac{x+1}{2}+3",
        r"\sqrt[3]{x^2}",
        r"\sum_{k=1}^{n} k^2",
        r"\int_0^1 x^2\,dx",
        r"\left(a+b\right)^2",
        r"\sin\left(x\right)+\alpha",
    ],
)
def test_latex_converted_expressions_parse(latex: str) -> None:
    source = latex_to_source(latex)
    assert source is not None
    parse(source)  # valid OpenGraph syntax


@pytest.mark.parametrize(
    ("source", "latex"),
    [
        ("(x+1)/2", r"\frac{x + 1}{2}"),
        ("sqrt(x^2+1)", r"\sqrt{x^{2} + 1}"),
        ("cbrt(x)", r"\sqrt[3]{x}"),
        ("a_1^2", "a_{1}^{2}"),
        ("2*pi", r"2 \cdot \pi"),
        ("x<=3", r"x \le 3"),
        ("sum(k^2, k, 1, n)", r"\sum_{k=1}^{n} k^{2}"),
        ("int(x^2, x, 0, 1)", r"\int_{0}^{1} x^{2}\, dx"),
    ],
)
def test_to_latex(source: str, latex: str) -> None:
    assert to_latex(from_source(source)[0].nodes) == latex


def test_latex_round_trips_through_source() -> None:
    for source in ["(x+1)/2", "sqrt(x^2+1)", "2*pi*x^2", "x<=3", "sum(k^2, k, 1, n)"]:
        back = latex_to_source(to_latex(from_source(source)[0].nodes))
        assert back is not None
        assert parse(back) == parse(source) or compile_expression(back).message == (
            compile_expression(source).message
        )


def test_pasting_latex_inserts_typeset_maths() -> None:
    model = EditorModel()
    model.paste(r"\frac{x+1}{2}")
    assert isinstance(model.root.nodes[0], Frac)
    assert model.source == "(x+1)/2"
    model = EditorModel()
    model.paste("x^2 + 1")  # ordinary source still pastes as before
    assert model.source == "x^2 + 1"


def test_copy_as_latex() -> None:
    edit = MathEdit("(x+1)/2")
    edit.copy_latex()
    assert runtime.clipboard.text == r"\frac{x + 1}{2}"
    edit.set_linear(True)
    edit.copy_latex()  # plain-text mode exports the same maths
    assert runtime.clipboard.text == r"\frac{x + 1}{2}"


@pytest.mark.parametrize(
    "text",
    [
        "seq[k for k in sets[N]]",
        "seq[k for k in sets[Z]]",
        "seq[k^2 for k in sets[Z]]",
        "seq[1/k for k in sets[N]]",
    ],
)
def test_set_syntax_round_trips_through_the_editor_and_latex(text: str) -> None:
    root, _ = from_source(text)
    from opengraph.ui.mathedit.source import to_source

    assert parse(to_source(root)[0]) == parse(text)
    latex = to_latex(root.nodes)
    back = latex_to_source(latex)
    assert back is not None, latex
    assert parse(back) == parse(text)


def test_latex_number_domains_in_explicit_generators() -> None:
    text = r"\operatorname{seq}[k \operatorname{for} k \in \mathbb{N}]"
    source = latex_to_source(text)
    assert source is not None
    assert parse(source) == parse("seq[k for k in sets[N]]")


@pytest.mark.parametrize("text", ["seq[k^2 for k in sets[N]]", "seq[1/k for k in sets[N]]"])
def test_typing_for_leaves_the_generator_body(text: str) -> None:
    model = typed(text)
    assert parse(model.source) == parse(text)


@pytest.mark.parametrize(
    "keys",
    [
        "k in sets[N]",
        "m in sets[Z]",
        "S = seq[2k + 1 for k in sets[N]]",
        "seq[k^2 for k in sets[Z]]",
    ],
)
def test_typing_membership_and_sequences_key_by_key(keys: str) -> None:
    """Every state typing passes through is a hint, never a hard error, once the
    construct has started; the finished text parses to what was typed."""
    from opengraph.engine.compiler import compile_document

    model = EditorModel()
    started = False
    for key in keys:
        model.type_text(key)
        started = started or model.source.lstrip("S= ").startswith("seq[")
        result = compile_document([(0, model.source)])[0]
        hard = result.is_error and not result.incomplete
        generic = "Unknown name" in result.message or result.message in (
            "Unexpected ']'",
            "Empty parentheses",
        )
        if started and hard and not generic:
            raise AssertionError(f"{model.source!r}: {result.message}")
    assert parse(model.source) == parse(keys.replace("^2", "^(2)"))


def test_in_shows_as_membership_only_before_a_number_set() -> None:
    from opengraph.ui.mathedit.layout import _membership_in

    for source, expected in (
        ("k in sets[N]", True),
        ("k in sets[Z]", True),
        ("k in", True),
        ("e^(in x)", False),
    ):
        root, _ = from_source(source)
        lst = root.nodes[1].body if source.startswith("e^") else root  # e^(in x): the exponent
        index = next(i for i, n in enumerate(lst.nodes) if getattr(n, "text", "") == "in")
        assert _membership_in(lst, index) is expected, source


def test_only_n_inside_sets_is_drawn_as_a_number_set() -> None:
    from opengraph.ui.mathedit.layout import _set_letter
    from opengraph.ui.mathedit.nodes import Char, Paren

    def letters(source: str) -> list[bool]:
        root, _ = from_source(source)
        found = []
        for i, node in enumerate(root.nodes):
            if isinstance(node, Char) and node.text == "N":
                found.append(_set_letter(root, i))
            if isinstance(node, Paren):
                body = node.body
                found += [
                    _set_letter(body, j)
                    for j, n in enumerate(body.nodes)
                    if isinstance(n, Char) and n.text == "N"
                ]
        return found

    assert letters("k in sets[N]") == [True]
    assert letters("y = N x") == [False]  # a variable N stays a plain italic N
    assert letters("k in N") == [False]
    assert letters("L[N]") == [False]  # an index named N is not a set
