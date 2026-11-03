"""Coordinates the document, the math engine, the expression panel and the canvas.

Data flow::

    panel intents ──► Document (pure model) ──► compile_document ──► RowResults
                                                                      │
                          panel feedback ◄────────────────────────────┤
                          canvas layers  ◄────────────────────────────┘

Text edits are debounced and compiled on a background thread (the result is shown
only if nothing changed meanwhile; :meth:`Controller.flush` compiles at once);
visibility and colour changes only rebuild the layer list.
Slider drags rewrite the row text (``k = 1.3``) like typing would, but recompile at
most once per frame so the graph follows the handle. Dragging a point on the canvas
does the same to what its coordinates are: the ``q = …`` rows of ``(q, w)``, or the
numbers in ``(3.8, 5.5)`` itself.
Plot objects are reused when a row's resolved expression is unchanged so the canvas
keeps its cached geometry. Symbolic operations are tasks (see ``opengraph.tasks``) run
in worker processes, so SymPy never stalls the interface and any task can be cancelled;
their results come back as callbacks on the interface's event loop.

Equations solved for an unknown name (``2a + 3 = 7``) first show the compiler's
numerical solutions; SymPy then solves them exactly in a separate lane, and when it
finds a finite set of real roots the document is recompiled with them, so the row shows
(and other rows use) the exact solutions. Otherwise (or if that task is cancelled)
the numerical ones stay.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Hashable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

from opengraph import events
from opengraph.document import Document, ExpressionItem, SliderRange, TableData
from opengraph.engine.compiler import (
    CasFailure,
    CasRequest,
    ExactRoots,
    GeneralSolution,
    RowResult,
    SolutionFamily,
    SymbolicTarget,
    TableColumn,
    TableInput,
    compile_with_values,
    split_column_header,
)
from opengraph.engine.errors import MathError, Span
from opengraph.engine.formatting import format_value
from opengraph.graph.background import LatestTask
from opengraph.graph.canvas import GraphCanvas, Layer
from opengraph.graph.settings import GraphSettings
from opengraph.tasks import TaskManager
from opengraph.ui import runtime, theme
from opengraph.ui.expression_panel import CURVE_OPERATIONS, ExpressionPanel
from opengraph.ui.theme import Theme

log = logging.getLogger(__name__)

COMPILE_DELAY_MS = 120
LIVE_COMPILE_MS = 16  # slider drags: about one recompile per frame
_EXACT_CACHE_LIMIT = 500  # exact solutions remembered before the cache is reset


class Controller:
    def __init__(self, panel: ExpressionPanel, canvas: GraphCanvas) -> None:
        self.status_message = events.Signal()  # str
        self.modified_changed = events.Signal()  # bool
        self.busy_changed = events.Signal()  # bool: a symbolic job runs; new ones are refused
        self.settings_changed = events.Signal()  # GraphSettings, after a change or new document
        self.panel = panel
        self.canvas = canvas
        self.document = Document()
        self.results: dict[int, RowResult] = {}
        # Exact values from SymPy, shown with a row's value while its text is unchanged.
        self._exact: dict[int, tuple[str, str]] = {}  # item id -> (text, exact value)
        self._modified = False
        # The point being dragged on the canvas: its row, and its text and what the drag
        # rewrites (see _drag_targets) as they were when the drag began.
        self._point_drag: tuple[int, str, list[_DragTarget]] | None = None

        self._compile_timer = events.Timer(COMPILE_DELAY_MS, single_shot=True)
        self._compile_timer.timeout.connect(self._compile_in_background)
        self._live_timer = events.Timer(LIVE_COMPILE_MS, single_shot=True)
        self._live_timer.timeout.connect(self._compile_in_background)
        # Edits (typing, slider drags) compile on a thread of their own, so a slow row
        # (a long sum, a root scan) never freezes the interface; a result is applied
        # only if nothing changed since it was requested. flush() compiles at once.
        self._compiler = LatestTask(_compile_pool())
        self._compiler.ready.connect(self._compiled)
        self._revision = 0  # bumped by every change the next compile must see
        self._compile_requested = False

        # Symbolic work runs in worker processes: the ⋮ menu's operations in one lane,
        # exact solving for unknowns in another, so neither waits for the other.
        self.tasks = TaskManager()
        self._busy = False
        self._generation = 0  # bumped per document so stale job results are dropped
        # Exact solutions of equations solved for an unknown;
        # usable finite set (the numerical solutions are kept). Keyed by content, so it
        # stays valid across documents.
        self._exact_solutions: dict[SymbolicTarget, ExactRoots | None] = {}
        self._exact_pending: set[SymbolicTarget] = set()
        # Computer-algebra results (diff, factor…), keyed by content like the above.
        self._cas: dict[CasRequest, str | CasFailure] = {}
        self._cas_pending: set[CasRequest] = set()

        panel.add_requested.connect(lambda: self.add_expression(focus=True))
        panel.text_edited.connect(self._on_text_edited)
        panel.visibility_toggled.connect(self._on_visibility_toggled)
        panel.color_chosen.connect(self._on_color_chosen)
        panel.delete_requested.connect(self.delete_expression)
        panel.symbolic_requested.connect(self.run_symbolic)
        panel.solve_for_requested.connect(self.solve_for)
        panel.sliders_requested.connect(self.add_sliders)
        panel.submitted.connect(self._on_submitted)
        panel.navigate.connect(self._on_navigate)
        panel.slider_moved.connect(self.set_slider_value)
        panel.slider_range_edited.connect(self.set_slider_range)
        panel.evaluate_slider_range = self.evaluate_slider_range
        panel.add_table_requested.connect(lambda: self.add_table(focus=True))
        panel.table_changed.connect(self._on_table_changed)
        panel.row_focused.connect(canvas.set_focus)
        panel.row_moved.connect(self.move_expression)
        canvas.point_dragged.connect(self.drag_point)
        canvas.point_drag_finished.connect(self._on_point_drag_finished)
        canvas.view_changed.connect(lambda: self._set_modified(True))

    # -- document lifecycle ----------------------------------------------------------
    @property
    def modified(self) -> bool:
        return self._modified

    def _set_modified(self, modified: bool) -> None:
        if modified != self._modified:
            self._modified = modified
            self.modified_changed.emit(modified)

    def set_document(self, document: Document) -> None:
        self._compile_timer.stop()
        self._live_timer.stop()
        self._revision += 1
        self._generation += 1
        self._point_drag = None
        self.document = document
        self.results = {}
        self.panel.clear()
        self._exact.clear()
        if not document.items:
            document.add()
        for index, item in enumerate(document.items):
            self._insert_widget(index, item)
        # The saved view is in axis units, so the axis types must be set before it.
        settings = document.settings.normalized()
        viewport = self.canvas.viewport
        viewport.x_log, viewport.y_log = settings.x_axis.log, settings.y_axis.log
        if document.view:
            viewport.restore(document.view)
        else:
            viewport.reset()
        self.canvas.view3d.restore(document.view)
        self.canvas.set_settings(settings)
        self.settings_changed.emit(settings)
        self.recompile()
        self._set_modified(False)

    def _insert_widget(self, index: int, item: ExpressionItem) -> None:
        if item.table is not None:
            self.panel.insert_table_row(index, item.id, item.table, item.color, item.visible)
        else:
            self.panel.insert_row(index, item.id, item.text, item.color, item.visible)

    # -- settings and theme ------------------------------------------------------------
    @property
    def settings(self) -> GraphSettings:
        return self.document.settings

    def set_settings(self, settings: GraphSettings) -> None:
        """Apply new graph settings: the canvas updates at once; text is recompiled if
        its meaning changed (angle unit, complex mode, polar grid)."""
        settings = settings.normalized()
        old = self.document.settings
        if settings == old:
            return
        self.document.settings = settings
        self.canvas.set_settings(settings)
        if settings.compile_options() != old.compile_options():
            self.recompile()
        self._set_modified(True)
        self.settings_changed.emit(settings)

    def set_theme(self, current: Theme) -> None:
        """Recolour everything for ``current`` (the panel reads theme.current())."""
        theme.set_current(current)
        self.canvas.set_colors(current.canvas)
        self.panel.apply_theme()
        self._push_layers()

    def new_document(self) -> None:
        self.set_document(Document())
        if not runtime.device.touch:  # focusing would pop up the soft keyboard
            self.panel.focus_row(self.document.items[0].id)

    def open_document(self, path: str | Path) -> None:
        self.set_document(Document.load(path))  # raises DocumentError

    def save_document(self, path: str | Path) -> None:
        self.document_data()
        self.document.save(path)
        self._set_modified(False)

    def document_data(self) -> bytes:
        """The document as the bytes of a file, with the current view."""
        self.document.view = self.canvas.view_state()
        return self.document.to_json().encode("utf-8")

    def mark_saved(self) -> None:
        self._set_modified(False)

    def mark_modified(self) -> None:
        self._set_modified(True)

    # -- expressions ------------------------------------------------------------------
    def add_expression(
        self, text: str = "", after: int | None = None, *, focus: bool = False
    ) -> int:
        index = len(self.document) if after is None else self.document.index_of(after) + 1
        item = self.document.add(text, index=index)
        self.panel.insert_row(index, item.id, item.text, item.color, item.visible)
        if focus:
            self.panel.focus_row(item.id)
        if text:
            self.recompile()
        self._set_modified(True)
        return item.id

    def add_table(
        self, data: TableData | None = None, after: int | None = None, *, focus: bool = False
    ) -> int:
        """Add a data table row (empty x_n/y_n columns unless ``data`` is given)."""
        index = len(self.document) if after is None else self.document.index_of(after) + 1
        if data is None:
            data = TableData(*self._free_column_names())
        item = self.document.add(index=index, table=data)
        self._insert_widget(index, item)
        if focus:
            self.panel.focus_row(item.id)
        self.recompile()
        self._set_modified(True)
        return item.id

    def _free_column_names(self) -> tuple[list[str], list[list[str]]]:
        """``x_n``/``y_n`` with the smallest n not yet used by any table."""
        used = {
            split_column_header(name)[0]
            for _, table in self.document.tables()
            for name in table.names
        }
        n = 1
        while f"x_{n}" in used or f"y_{n}" in used:
            n += 1
        return [f"x_{n}", f"y_{n}"], [[], []]

    def _on_table_changed(self, item_id: int, data: TableData) -> None:
        self.document.get(item_id).table = data
        self._revision += 1
        self._set_modified(True)
        self._compile_timer.start()

    def delete_expression(self, item_id: int) -> None:
        index = self.document.remove(item_id)
        self._exact.pop(item_id, None)
        self.panel.remove_row(item_id)
        if not self.document.items:
            self.add_expression()
        ids = [item.id for item in self.document.items]
        self.panel.focus_row(ids[max(0, index - 1)])
        self.recompile()
        self._set_modified(True)

    def move_expression(self, item_id: int, index: int) -> None:
        """Move a row to ``index`` (rows are drawn in list order, later ones on top)."""
        if self.document.index_of(item_id) == index:
            return
        self.document.move(item_id, index)
        if self.panel.row_ids() != [item.id for item in self.document.items]:
            self.panel.move_row(item_id, index)
        self._push_layers()
        self._set_modified(True)

    def set_text(self, item_id: int, text: str) -> None:
        """Programmatic edit (used by tests and tools); same path as typing."""
        self.panel.row(item_id).replace_text(text)

    def _on_text_edited(self, item_id: int, text: str) -> None:
        self.document.get(item_id).text = text
        self._revision += 1
        self._set_modified(True)
        self._compile_timer.start()

    def _on_visibility_toggled(self, item_id: int, visible: bool) -> None:
        self.document.get(item_id).visible = visible
        self._set_modified(True)
        self._push_layers()

    def _on_color_chosen(self, item_id: int, color: str) -> None:
        self.document.get(item_id).color = color
        self._set_modified(True)
        self._push_layers()

    def _on_submitted(self, item_id: int) -> None:
        self.flush()
        self.add_expression(after=item_id, focus=True)

    def _on_navigate(self, item_id: int, direction: int) -> None:
        ids = [item.id for item in self.document.items]
        index = ids.index(item_id) + direction
        if 0 <= index < len(ids):
            self.panel.focus_row(ids[index])

    # -- sliders -----------------------------------------------------------------------
    def set_slider_value(self, item_id: int, value: float) -> None:
        """Rewrite a ``k = <number>`` row to ``value``; dependents follow within a frame."""
        name = self.results[item_id].defines if item_id in self.results else None
        if name is None:
            return
        self.panel.row(item_id).replace_text(f"{name} = {format_value(value)}")
        # Throttle rather than debounce: a continuous drag must not starve the timer.
        if not self._live_timer.is_active():
            self._live_timer.start()

    # -- dragging points ------------------------------------------------------------------
    def _drag_targets(self, result: RowResult) -> list[_DragTarget]:
        """What a point row's drag rewrites, per axis (0 = x, 1 = y): the id of the
        ``name = <number>`` row defining a coordinate, or the span of a number in the
        point's own text. Coordinates that are computed are left out."""
        if result.drag is None:
            return []
        sliders = {
            r.defines: item_id
            for item_id, r in self.results.items()
            if r.defines is not None and r.literal_value is not None
        }
        targets: list[_DragTarget] = []
        for axis, coordinate in enumerate(result.drag):
            if isinstance(coordinate, str):
                if coordinate in sliders and all(t != sliders[coordinate] for _, t in targets):
                    targets.append((axis, sliders[coordinate]))
            elif coordinate is not None:
                targets.append((axis, coordinate))
        return targets

    def drag_point(self, item_id: int, x: float, y: float) -> None:
        """Move a dragged point to (x, y) by rewriting the numbers it is made of."""
        if self._point_drag is None or self._point_drag[0] != item_id:
            result = self.results.get(item_id)
            if result is None:
                return
            # Spans refer to the text as compiled now, so every step of the drag
            # rewrites that text rather than the previous step's.
            text = self.document.get(item_id).text
            self._point_drag = (item_id, text, self._drag_targets(result))
        _, text, targets = self._point_drag
        values = (x, y)
        literals = sorted(
            ((t, values[axis]) for axis, t in targets if isinstance(t, tuple)), reverse=True
        )
        for (start, end), value in literals:  # right to left keeps earlier spans valid
            text = f"{text[:start]}{format_value(value)}{text[end:]}"
        if literals and text != self.document.get(item_id).text:
            self.panel.row(item_id).replace_text(text)
        for axis, target in targets:
            if isinstance(target, int):
                name = self.results[target].defines
                if name is not None:
                    self.panel.row(target).replace_text(f"{name} = {format_value(values[axis])}")
        if not self._live_timer.is_active():
            self._live_timer.start()

    def _on_point_drag_finished(self, item_id: int) -> None:
        self._point_drag = None
        self.flush()

    def evaluate_slider_range(self, item_id: int, texts: Sequence[str]) -> SliderRange:
        """The range typed for a row's slider (min, max and step may be expressions that
        use other rows); raises ValueError with a readable message."""
        _, values = self._compile([(i, text, item_id) for i, text in enumerate(texts)])
        return SliderRange.from_texts(texts, [_number_or_message(values[i]) for i in range(3)])

    def set_slider_range(
        self, item_id: int, lo: float | str, hi: float | str, step: float | str
    ) -> None:
        item = self.document.get(item_id)
        result = self.results.get(item_id)
        value = result.literal_value if result is not None else None
        texts = [t if isinstance(t, str) else format_value(t) for t in (lo, hi, step)]
        try:
            slider_range = self.evaluate_slider_range(item_id, texts)
        except ValueError as exc:
            self.status_message.emit(str(exc))
            self.panel.row(item_id).set_slider(value, item.slider)  # restore the fields
            return
        item.slider = slider_range
        self._set_modified(True)
        if value is None:
            return
        lo, hi = slider_range.min, slider_range.max
        if not lo <= value <= hi:  # keep the value inside the range it is edited with
            self.set_slider_value(item_id, min(max(value, lo), hi))
            self.flush()
        else:
            self.panel.row(item_id).set_slider(value, slider_range)

    # -- compilation -------------------------------------------------------------------
    def flush(self) -> None:
        """Apply any pending (debounced or background) edit immediately."""
        if (
            self._compile_timer.is_active()
            or self._live_timer.is_active()
            or self._compile_requested
        ):
            self.recompile()

    @property
    def compiling(self) -> bool:
        """Whether an edit is waiting to be compiled (or compiling in the background)."""
        return (
            self._compile_timer.is_active()
            or self._live_timer.is_active()
            or self._compile_requested
        )

    def recompile(self) -> None:
        """Compile the document now, on this thread, and show the results."""
        self._compile_timer.stop()
        self._live_timer.stop()
        self._revision += 1
        self._compile_requested = False
        self._compiler.cancel()
        bounds = self._slider_bounds()
        self._show_results(*self._compile(bounds))

    def _compile_in_background(self) -> None:
        self._revision += 1
        self._compile_requested = True
        tables = [(item_id, _table_input(table)) for item_id, table in self.document.tables()]
        known = {t: roots for t, roots in self._exact_solutions.items() if roots is not None}
        job = partial(
            compile_with_values,
            self.document.rows(),
            self._slider_bounds(),
            self.document.settings.compile_options(),
            tables,
            known,
            dict(self._cas),
        )
        self._compiler.request(self._revision, job)

    def _compiled(self, revision: object, outcome: object) -> None:
        if revision != self._revision or not self._compile_requested:
            return  # the document changed since: a newer compile is coming
        if not isinstance(outcome, tuple):
            self.recompile()  # it failed on the worker: report it from here
            return
        self._compile_requested = False
        self._show_results(*outcome)

    def _slider_bounds(self) -> list[tuple[Hashable, str, int]]:
        return [
            ((item.id, i), text, item.id)
            for item in self.document.items
            if item.slider is not None and any(item.slider.exprs)
            for i, text in enumerate(item.slider.texts)
        ]

    def _show_results(
        self,
        results: dict[int, RowResult],
        bound_values: Mapping[Hashable, float | MathError],
    ) -> None:
        previous = self.results
        for item_id, result in results.items():
            old = previous.get(item_id)
            if old is None or old.plot is None:
                continue
            if result.incomplete and result.plot is None:
                # Unfinished typing (``x^``): keep the last drawing instead of flickering.
                results[item_id] = replace(result, plot=old.plot)
            elif old.plot == result.plot:
                # Same drawing: keep the old object so the canvas cache stays valid.
                results[item_id] = replace(result, plot=old.plot)
        self.results = results
        for item_id, result in results.items():
            row = self.panel.widget(item_id)
            message = result.message
            exact = self._exact.get(item_id)
            if exact is not None and exact[0] == self.document.get(item_id).text:
                message = f"{message}  (exact: {exact[1]})"
            error = result.is_error
            if result.literal_value is not None:
                problem = self._refresh_slider_range(item_id, bound_values)
                if problem:
                    message, error = problem, True
            row.set_feedback(
                message,
                error=error and not result.incomplete,
                span=result.error_span,
                has_graph=result.plot is not None,
            )
            row.set_symbolic_operations(self._operations(result))
            row.set_solve_for(self._solve_names(result), enabled=not self._busy)
            row.set_slider_suggestions(result.undefined)
            if self.document.get(item_id).table is not None:
                table_row = self.panel.table(item_id)
                table_row.set_cell_errors(dict(result.cell_errors))
                table_row.set_table_values(dict(result.table_values))
            self._update_slider(item_id, result.literal_value)
        self._push_layers()
        self._solve_exactly(results)
        self._compute_algebra(results)

    def _compile(
        self, values: Sequence[tuple[Hashable, str, int]] = ()
    ) -> tuple[dict[int, RowResult], dict[Hashable, float | MathError]]:
        """Compile the document, and evaluate ``values`` in it (see compile_with_values).

        Runs on the interface thread; :meth:`_compile_in_background` is the same work
        on the compile thread."""
        tables = [(item_id, _table_input(table)) for item_id, table in self.document.tables()]
        known = {t: roots for t, roots in self._exact_solutions.items() if roots is not None}
        return compile_with_values(
            self.document.rows(),
            values,
            self.document.settings.compile_options(),
            tables,
            known,
            self._cas,
        )

    def _refresh_slider_range(
        self, item_id: int, values: Mapping[Hashable, float | MathError]
    ) -> str:
        """Re-evaluate a slider's expression bounds; returns why they can't be, or ""."""
        item = self.document.get(item_id)
        if item.slider is None or not any(item.slider.exprs):
            return ""
        try:
            # Not an edit: the range follows the rows it depends on.
            item.slider = item.slider.with_values(
                [_number_or_message(values[(item_id, i)]) for i in range(3)]
            )
        except ValueError as exc:
            return str(exc)  # keep the last good range meanwhile
        return ""

    def _solve_exactly(self, results: Mapping[int, RowResult]) -> None:
        """Start exact solves for equations solved for an unknown that have none yet."""
        for item_id, result in results.items():
            target = result.symbolic
            if result.solved is None or target is None:
                continue
            if target in self._exact_solutions or target in self._exact_pending:
                continue
            if len(self._exact_solutions) > _EXACT_CACHE_LIMIT:
                self._exact_solutions.clear()
            self._exact_pending.add(target)
            self.tasks.submit(
                "exact",
                f"Solve exactly: {self.document.get(item_id).text}",
                "solve",
                target,
                partial(self._on_exact_outcome, target),
                # Cancelled: keep the numerical solutions and don't try again.
                partial(self._on_exact_solved, target, None),
            )

    def _compute_algebra(self, results: Mapping[int, RowResult]) -> None:
        """Start the computer-algebra calls (``diff(…)``, ``factor(…)``) rows wait for."""
        for item_id, result in results.items():
            for request in result.pending:
                if request in self._cas or request in self._cas_pending:
                    continue
                if len(self._cas) > _EXACT_CACHE_LIMIT:
                    self._cas.clear()
                self._cas_pending.add(request)
                self.tasks.submit(
                    "cas",
                    f"{request.operation.capitalize()}: {self.document.get(item_id).text}",
                    "cas",
                    request,
                    partial(self._on_algebra_done, request),
                    partial(self._on_algebra_done, request, MathError("Cancelled")),
                )

    def _on_algebra_done(self, request: CasRequest, outcome: object) -> None:
        self._cas_pending.discard(request)
        if isinstance(outcome, MathError):
            self._cas[request] = CasFailure(outcome.message)
        else:
            assert isinstance(outcome, str)
            self._cas[request] = outcome
        if any(request in r.pending for r in self.results.values()):
            self.recompile()

    def _on_exact_outcome(self, target: SymbolicTarget, outcome: object) -> None:
        self._on_exact_solved(target, _exact_roots(outcome))

    def _on_exact_solved(self, target: SymbolicTarget, roots: ExactRoots | None) -> None:
        self._exact_pending.discard(target)
        self._exact_solutions[target] = roots
        if roots is not None and any(r.symbolic == target for r in self.results.values()):
            self.recompile()

    def _operations(self, result: RowResult) -> frozenset[str]:
        """The symbolic operations a row's ⋮ menu offers right now."""
        if self._busy or result.symbolic is None:
            return frozenset()
        if result.symbolic.value:
            return frozenset({"exact"})
        if result.symbolic.variable not in ("x", "y"):  # an equation solved for a name
            return frozenset({"solve"})
        return CURVE_OPERATIONS

    def _update_slider(self, item_id: int, value: float | None) -> None:
        row = self.panel.widget(item_id)
        if value is None:
            row.set_slider(None)
            return
        item = self.document.get(item_id)
        # The range is stored as soon as the slider appears so it stays put while the
        # value moves; a typed value outside it widens it. Neither counts as a user
        # edit by itself (opening a file must not mark it modified).
        if item.slider is None:
            item.slider = SliderRange.default_for(value)
        else:
            item.slider = item.slider.including(value)
        row.set_slider(value, item.slider)

    def _push_layers(self) -> None:
        layers = []
        for item in self.document.items:
            result = self.results.get(item.id)
            if item.visible and result is not None and result.plot is not None:
                # Shown in a shade that contrasts with the canvas in the current theme.
                color = theme.current().curve_color(item.color)
                draggable = bool(self._drag_targets(result))
                layers.append(Layer(item.id, result.plot, color, draggable))
        self.canvas.set_layers(layers)

    # -- symbolic tools ------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._busy

    def _set_busy(self, busy: bool) -> None:
        if busy == self._busy:
            return
        self._busy = busy
        for item_id, result in self.results.items():
            widget = self.panel.widget(item_id)
            widget.set_symbolic_operations(self._operations(result))
            widget.set_solve_for(self._solve_names(result), enabled=not busy)
        self.busy_changed.emit(busy)

    def _solve_names(self, result: RowResult) -> tuple[str, ...]:
        """Undefined names the row offers to be solved for (rearranging needs SymPy)."""
        return result.solve_for if result.rearrange is not None else ()

    def add_sliders(self, item_id: int, names: list[str]) -> None:
        """Define undefined ``names`` as ``name = 1`` rows (with sliders) below a row."""
        self.flush()
        result = self.results.get(item_id)
        wanted = [n for n in names if result is not None and n in result.undefined]
        if not wanted:
            return
        for name in reversed(wanted):
            self.add_expression(f"{name} = 1", after=item_id)
        self.status_message.emit(
            f"Added slider{'s' if len(wanted) > 1 else ''} for {', '.join(wanted)}"
        )

    def solve_for(self, item_id: int, name: str) -> None:
        """Rearrange a row that uses undefined names for ``name``; results go below it."""
        if self._busy:
            self.status_message.emit("Please wait for the current computation to finish")
            return
        self.flush()
        result = self.results.get(item_id)
        if result is None or result.rearrange is None or name not in self._solve_names(result):
            self.status_message.emit(f"This row can't be solved for {name}")
            return
        self.status_message.emit(f"Solving for {name}…")
        target = replace(result.rearrange, variable=name)
        self._start_task(item_id, "rearrange", f"Solve for {name}", target)

    def run_symbolic(self, item_id: int, operation: str) -> None:
        if self._busy:
            self.status_message.emit("Please wait for the current computation to finish")
            return
        self.flush()
        result = self.results.get(item_id)
        if result is None or result.symbolic is None:
            self.status_message.emit("This row has nothing to operate on symbolically")
            return
        if operation not in self._operations(result):
            self.status_message.emit("That operation doesn't apply to this row")
            return
        self.status_message.emit(f"Computing {_OPERATION_NAMES[operation]}…")
        title = _OPERATION_NAMES[operation].capitalize()
        self._start_task(item_id, operation, title, result.symbolic)

    def _start_task(self, item_id: int, operation: str, title: str, target: SymbolicTarget) -> None:
        """Run a ⋮-menu operation as a cancellable task; one at a time."""
        self._set_busy(True)
        generation = self._generation
        self.tasks.submit(
            "symbolic",
            f"{title}: {self.document.get(item_id).text}",
            operation,
            target,
            lambda outcome: self._on_symbolic_finished(item_id, operation, outcome, generation),
            self._on_symbolic_cancelled,
        )

    def wait_for_jobs(self, timeout_ms: int = 30_000) -> bool:
        return self.tasks.wait(timeout_ms)

    def _on_symbolic_cancelled(self) -> None:
        self._set_busy(False)
        self.status_message.emit("Cancelled")

    def _on_symbolic_finished(
        self, item_id: int, operation: str, outcome: object, generation: int
    ) -> None:
        self._set_busy(False)
        if generation != self._generation:
            return  # the document was replaced while computing
        if isinstance(outcome, MathError):
            self.status_message.emit(outcome.message)
            return
        anchor = item_id if any(i.id == item_id for i in self.document.items) else None
        if operation == "rearrange":
            assert isinstance(outcome, str)
            name, expression = outcome.split(" = ", 1)
            self.add_expression(f"formula({name}, {expression})", after=anchor)
            self.status_message.emit(f"Added {outcome.split(' =')[0]} = …")
            return
        if operation == "exact":
            assert isinstance(outcome, str)
            if anchor is not None:
                self._exact[item_id] = (self.document.get(item_id).text, outcome)
                self.recompile()
            self.status_message.emit(f"Exact value: {outcome}")
            return
        target = self.results[item_id].symbolic if item_id in self.results else None
        variable = target.variable if target is not None else "x"
        if operation == "solve" and isinstance(outcome, GeneralSolution):
            general = outcome.text(variable)
            if variable in ("x", "y"):
                rows = [text for text, _ in outcome.roots]
                if rows:
                    rows = [f"list[{', '.join(rows)}]" if len(rows) > 1 else rows[0]]
                rows += [_family_points(family) for family in outcome.families]
                rows += [
                    f"{generator.source(outcome.index(variable))}[1...100]"
                    for generator in outcome.generators
                ]
                for roots in reversed(rows):
                    point = f"point[{roots}, 0]" if variable == "x" else f"point[0, {roots}]"
                    self.add_expression(point, after=anchor)
            self.status_message.emit(f"Infinitely many solutions: {general}")
            return
        if operation == "solve":
            assert isinstance(outcome, list)
            if not outcome:
                self.status_message.emit("No real solutions")
                return
            if variable not in ("x", "y"):
                exact = ", ".join(root.exact for root in outcome)
                if len(outcome) > 1:
                    exact = f"list[{exact}]"
                self.status_message.emit(f"Exact solutions: {variable} = {exact}")
                return
            roots = ", ".join(root.exact for root in outcome)
            if len(outcome) > 1:
                roots = f"list[{roots}]"
            point = f"point[{roots}, 0]" if variable == "x" else f"point[0, {roots}]"
            self.add_expression(point, after=anchor)
            self.status_message.emit(
                f"Found {len(outcome)} real solution{'s' if len(outcome) != 1 else ''}"
            )
            return
        assert isinstance(outcome, str)
        dependent = "y" if variable == "x" else "x"
        if operation == "simplify" and target is not None and target.equation:
            self.add_expression(f"{outcome} = 0", after=anchor)
        else:
            self.add_expression(f"{dependent} = {outcome}", after=anchor)
        added = {"derivative": "derivative", "integral": "integral"}
        self.status_message.emit(f"Added {added.get(operation, 'simplified form')}")


# What dragging a point rewrites on one axis (0 = x, 1 = y): a slider row's id, or the
# span of a number in the point's own text.
_DragTarget = tuple[int, int | Span]


_OPERATION_NAMES = {
    "derivative": "derivative",
    "integral": "integral",
    "simplify": "simplified form",
    "solve": "roots",
    "exact": "exact value",
}


_pool: ThreadPoolExecutor | None = None


def _compile_pool() -> ThreadPoolExecutor:
    """One thread for document compiles (shared by every controller)."""
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(1, thread_name_prefix="opengraph-compile")
    return _pool


def _exact_roots(outcome: object) -> ExactRoots | None:
    """Finite roots or exact generators from a solve task; None for unusable results."""
    if isinstance(outcome, GeneralSolution):
        return outcome
    if not isinstance(outcome, list):
        return None
    return tuple((root.exact, root.value) for root in outcome)


# How many members of each side of a periodic family ⋮ ▸ Solve plots: enough to cover
# about ±_FAMILY_REACH, at least one and at most _FAMILY_MEMBERS per side.
_FAMILY_REACH = 60.0
_FAMILY_MEMBERS = 50


def _family_points(family: SolutionFamily) -> str:
    """``pi/6 + 2*pi*list[-10...10]``: members of a periodic family as a list."""
    count = min(_FAMILY_MEMBERS, max(1, math.ceil(_FAMILY_REACH / family.period_value)))
    members = f"{family.period}*list[-{count}...{count}]"
    return members if family.base == "0" else f"{family.base} + {members}"


def _number_or_message(value: float | MathError) -> float | str:
    return value.message if isinstance(value, MathError) else value


def _table_input(table: TableData) -> TableInput:
    """The engine's view of a document table."""
    columns = tuple(
        TableColumn(name, tuple(cells))
        for name, cells in zip(table.names, table.columns, strict=True)
    )
    return TableInput(columns, markers=table.style != "lines", lines=table.style != "points")
