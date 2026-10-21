# Design notes

OpenGraph separates the toolkit-free math engine, document model, controller and canvas.
The source string is the document and compiler input. The structured editor is a
view and editing model over that source, and ordinary typing uses the same operator
precedence as pasted text.

## Engine

* `lexer.py` and `parser.py`: source spans, atomic identifiers and a Pratt grammar.
* `ast.py`: immutable trees, explicit scoped reductions/sequences, domain guards and
  interval restrictions. Structural equality supplies numeric/geometry cache keys.
* `compiler.py`: collect explicit definitions, reject duplicates/cycles, inline
  functions, resolve local bindings, enforce equal-length lists and classify plots.
  Equations never add definitions. Affine rearrangement requires structural proof.
* `evaluator.py`: vectorized NumPy closures with constant folding. Guards retain
  original domains; bounded expressions mask open/closed interval endpoints.
  Variable-free sub-trees are folded in 256-bit MPFR (`precise.py`), rounded once.
* `precise.py`: arbitrary-precision evaluation with gmpy2. A tree of gmpy2 closures
  (folding, Σ/Π, fallback) and a register bytecode for the compiled MPFR machine
  (`native/_mpvm.pyx`) implement the float evaluator's real semantics (NaN outside a
  domain, real odd roots, guards). `PreciseFunction` evaluates `f(origin + offset) -
  out_origin` for float64 offsets, which is how deep-zoom tiles stay exact.
  `precision_bits` decides when float64 no longer resolves a pixel.
* `symbolic.py`: explicit AST-to-SymPy conversion, verified source output and symbolic
  operations. Algebra/derivatives preserve domains. Antiderivative requests include
  current parameter values, so exceptional values cannot reuse a generic formula.
  Rational operations (`cancel`, `together`, `apart`), collection and trigonometric
  simplification use the same CAS request/cache/worker path as existing operations.
  Algebra, derivatives, limits and Taylor polynomials accept free parameters in
  standalone rows; definitions continue to require explicit parameters or bindings.
  The compiler retains guards through nested transformations and resolves cached
  formulas against current definitions before deciding whether to plot or display.
  Partial fractions also include current parameter values in their requests, since
  coincident poles need a different decomposition. Free coefficients are accepted
  only when the decomposition introduces no additional parameter exclusions.
* `source.py`: source serialization that preserves the original expression structure
  rather than simplifying away domain restrictions.
* `plots.py`, `sampling.py`, `analysis.py`: adaptive sampling, normalized implicit
  zero sets, inequality masks, numerical root scans and points of interest. A
  `ViewRect` with `bits > 0` has bounds that are offsets from an exact origin; plots
  then sample through `precise.py` and return geometry relative to that origin.

## Native kernels

`native/_kernels.pyx` (segment chaining, Douglas–Peucker, screen placement) and
`native/_mpvm.pyx` (the MPFR machine) are Cython, built by `setup.py` against gmpy2's
bundled MPFR/GMP. Both run without the GIL. Each has a pure Python/NumPy reference
(`rendering._chain_segments`, `rendering._simplify_numpy`, `canvas.place_points`,
`precise.scalar_closure`) that gives identical results; tests compare them, and
`OPENGRAPH_PURE_PYTHON=1` runs everything on the references.

## Application

The interface uses Flet. Everything runs on one asyncio event loop (Flet's):
`events.py` provides observer signals, timers and `post()`, which background threads
use to hand results back to that loop. Interface classes own their Flet controls
(composition) and work without a page, so the tests drive the same objects the window
shows; `ui/runtime.py` sends changes to the client once a control is on a page.

The graph canvas and the typeset editor build lists of Flet canvas shapes in Python.
The editor's layout is computed with the metrics of the bundled DejaVu fonts (read by
fontTools), the same files the client draws, so the caret and selection line up with
the glyphs. Characters reach the editor through a hidden text field (keyboard
layouts, AltGr); editing keys (arrows, Backspace, Tab, Ctrl+Z...) come from the page's
keyboard events, which the window routes to the focused editor, then to the menu
shortcuts, then to the graph. Input methods that compose text in the field (CJK) are
not supported by the editor.

Flet diffs canvas shapes by object identity, so the graph canvas keeps its shapes
between redraws: each tile's shapes are reused until its screen transform or colour
changes, and the grid, axes and labels until the view changes. These shapes are marked
frozen (as Flet marks a component's output), which makes a reused shape free to diff
and is why the canvas never modifies a shape after building it; the one exception is
an inequality's fill image, which is moved in place so the client keeps it decoded.
A hover, a marker update or an arriving tile therefore sends only what changed. Each
tile also carries an *outline* built on the worker thread: implicit segments joined
into polylines and every line simplified to within 0.2 px, which is what gets stroked.
Tracing still uses the full geometry. Workers also convert both to screen pixels
measured from the tile's own corner (`rendering.ScreenGeometry`), valid for any pan,
and compress inequality shading into a 1-bit PNG body; the canvas adds a two-colour
palette for the layer's colour, so recolouring never re-encodes the raster.

Deep zoom: `graph/viewport.py` gives a linear axis an exact gmpy2 *anchor* once
float64 can't resolve its pixels; the canvas's world coordinates are then float64
offsets from it, and tiles have exact origins (multiples of the tile size). Ticks are
exact multiples of their decimal step, and labels and traced coordinates are formatted
from exact values. Points of interest and point dragging are off at that depth.

The controller compiles debounced edits on a single compile thread (`LatestTask`),
applying a result only if the document's revision is unchanged; `flush()` and the
other synchronous paths compile on the interface thread as before.

The panel emits intents. The controller changes the document, compiles it and sends
plot layers and feedback to the UI. The canvas consumes plots, never source text.
Settings for angle units and number mode affect compilation; grid appearance does
not. Polar coordinate names have fixed meaning on every grid.

SymPy jobs run in cancellable worker processes managed by `tasks.py`. Content-keyed
requests prevent stale results from changing unrelated rows. Exact equation answers
update feedback only. Solve for creates `formula(...)` rows that can represent free
symbols without trying to evaluate or bind them.

Infinite sequences are lazy exact generators. Display previews are labelled and
never substitute for infinite mathematical values. Direct term indexing and explicit
finite slices are the only route from a sequence to finite list arithmetic/plots.
Finite lists broadcast scalars and require matching lengths with other lists.

The document remains JSON with expression source, tables and graph settings. The
project is unreleased; this cleanup intentionally does not migrate old spellings.
See [semantic changes](SEMANTICS.md) for the review-to-implementation mapping.
