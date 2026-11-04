# OpenGraph

A desktop symbolic graphing calculator built with Flet, NumPy, SymPy and gmpy2.
Enter mathematics in a typeset or plain-text editor, plot curves live, and compute
symbolic derivatives, antiderivatives, limits, Taylor polynomials and exact roots.

## Run and check

Python 3.13 or newer is required. OpenGraph runs on Windows, macOS and Linux; CI
tests all three (`.github/workflows/ci.yml`). The same commands work in PowerShell
and POSIX shells:

```bash
uv sync
uv run opengraph
uv run pytest
uv run ruff check src tests
uv run mypy
```

`uv sync` compiles the Cython kernels in `src/opengraph/native` (geometry loops and
an MPFR evaluator linked against the MPFR library bundled with gmpy2). This needs a C
compiler: on Windows, Visual Studio Build Tools with the "Desktop development with
C++" workload; on macOS, the Xcode Command Line Tools (`xcode-select --install`); on
Linux, `gcc` (for example `sudo apt install build-essential` or
`sudo dnf install gcc`). The kernels link against the MPFR and GMP libraries inside
the gmpy2 wheel, so no system MPFR is needed. Without a compiler the install still
succeeds and the same results come from the Python fallbacks, more slowly; set
`OPENGRAPH_REQUIRE_NATIVE=1` to make a failed build an error instead. After editing a `.pyx` file, `uv sync` rebuilds it.
`OPENGRAPH_PURE_PYTHON=1` runs without the compiled kernels (the test suite passes
both ways).

The application accepts an optional `.ograph.json` document path. F1 opens the
syntax reference, which includes the function registry. Ctrl+Shift+L switches the
editor between typeset and plain-text input. On macOS every Ctrl shortcut uses the
Command key instead (⇧⌘L), and the menus and reference show it that way; Ctrl+wheel
and Shift+wheel zoom keep the Control key. Preferences are stored in
`%APPDATA%\OpenGraph`, `~/Library/Application Support/OpenGraph` or
`$XDG_CONFIG_HOME/opengraph`.

`uv run opengraph --web` serves the same interface at http://localhost:8550 instead
of opening a window (`--port=N` picks another port); this needs the `flet-web`
development dependency. It is meant for development: one browser tab at a time.

The interface is drawn with bundled DejaVu fonts (`src/opengraph/ui/assets/fonts`,
see `LICENSE-DejaVu.txt`). The typeset editor lays out every glyph itself, so it
measures the same font files the client draws with.

Graph geometry loads in the background as the visible area changes. Explicit
functions use reusable strips; implicit curves and inequalities use 512-pixel tiles,
so large canvases do not require one enormous sampling grid. Panning reuses completed
tiles without sampling them again. Obsolete
requests are cancelled when you edit or navigate. Parametric and
polar curves use adaptive sampling of their parameter range for the current view.
The reusable geometry cache holds up to 256 tiles or 64 MiB, in addition to the
current visible geometry and its canvas shapes. Automatic points of interest also
run in the background, and image export waits for the visible geometry to finish.

The interface thread only places shapes. Geometry workers sample each tile, simplify
its outline, convert it to screen pixels and compress inequality shading; their
compiled loops release the GIL, so tiles are computed in parallel. Edits are
compiled on a thread of their own, so a slow row (a long sum, a root scan) doesn't
freeze typing or slider drags.

## Precision and deep zoom

Plotting uses float64 while it can resolve a pixel. Zooming further on linear axes
(up to 10⁴⁰ pixels per unit, against float64's 10¹²) switches to arbitrary precision:
the view keeps its position as an exact gmpy2 value, and tiles are evaluated in MPFR
with as many bits as the zoom needs (plus guard bits), then rounded once relative to
the tile. So `y = 4 + 1e-21 sin(1e21 x)` shows its wiggles at (2, 4), which float64 can't
represent. Axis labels and traced coordinates show every digit a pixel resolves.
Zooming back out returns to float64. At such depths, points of interest and point
dragging (float64 analyses) are not offered, complex-mode expressions are evaluated in
float64, and a saved view keeps a float64 centre.

Constant sub-expressions are folded once, in 256-bit arithmetic, and rounded once:
`1e16 + 1 - 1e16` is 1, `0.1 + 0.2` is 0.3, `sin(30)` in degrees is exactly 0.5, and
`sum(1/k^2 for k from 1 to 100000)` is correctly rounded. With gmpy2 installed, SymPy
and mpmath also use GMP for exact integers, rationals and arbitrary precision.

## Phones and tablets

The same application runs as an Android or iOS app. [Flet](https://flet.dev) builds them;
the first build downloads Flutter and, for Android, its SDK. iOS needs a Mac with Xcode.

```bash
uv run --with flet-cli==1.0.4 flet build apk --python-version 3.13   # build/apk
uv run --with flet-cli==1.0.4 flet build ipa --python-version 3.13   # needs a signing identity
```

`[tool.flet]` in `pyproject.toml` holds the app's name and identifier. The version of
`flet-cli` has to be the one `pyproject.toml` pins for `flet`.

The Actions workflow **Application builds** builds for Windows, Linux, macOS (Intel and
Apple silicon) and Android (APK and AAB), and keeps the results as artifacts of the run. It
runs by hand from the Actions tab, and for a tag `v1.2.3`, which must match the version in
`pyproject.toml`. Android builds use a debug signature unless the repository has the secrets
`ANDROID_KEY_STORE_BASE64`, `ANDROID_KEY_STORE_PASSWORD`, `ANDROID_KEY_PASSWORD` and
`ANDROID_KEY_ALIAS`. These builds don't compile the Cython kernels, so they use the Python
fallbacks, which give the same results more slowly.

On a touch screen one finger pans the graph (or turns a 3D view), and two fingers pinch to
zoom and move it. Tap, or press and hold and slide, to read coordinates. Below 720 pixels
wide the graph goes above the expressions, with a handle to divide the space between them.
A row of keys above the keyboard has what soft keyboards lack: arrow keys to move through
exponents and fractions, `^`, `/`, brackets, `√`, `π` and `θ`. Press and hold an expression
for undo, copy and paste. The file chooser does the saving, so Save always asks where. The
current graph is kept when the app goes to the background and is back on the next launch.

Three things differ from the desktop:

* Android and iOS have no gmpy2, so exact arithmetic and deep zoom run on mpmath, which
  gives the same results more slowly. The Cython kernels aren't built either.
* Apps can't start processes, so symbolic work runs on a thread, and the interface may
  stutter during long computations. Cancel stops a computation by raising an exception in
  that thread, which works within moments unless one operation in C holds it up, such as
  multiplying integers of millions of digits.
* Opening a graph by a file passed on the command line is for desktops only.

## Predictable expression syntax

| Purpose | Example |
|---|---|
| Explicit curve | `y = x^2`, `x = sin(y)`, or bare `x^2` |
| Implicit curve | `x^2 + y^2 = 1` |
| Inequality | `y < x^2`, `x + y >= 1`, `1 < x <= 3` |
| Polar curve | `r = 1 + cos(theta)` |
| Surface | `z = x^2 - y^2`, `x = y z`, `x + y + z = 1` |
| Implicit surface | `x^2 + y^2 + z^2 = 9` |
| Point/curve in space | `point[1, 2, 3]`, `parametric(cos(u), sin(u), u for u in [0, 6pi])` |
| Definition | `a = 2`, `f(t) = t^2`, `g(u, v) = u + v` |
| Restricted curve | `plot(t^2 for t in (0, 2])` |
| Parametric curve | `parametric(cos(u), sin(u) for u in [0, 2pi])` |
| Continuous curve | `t = [0, 2pi]`, then `point[cos(t), sin(t)]` |
| Point/vector | `point[1, 2]`, `p = point[1, 2]`, `2p`, `p + point[3, 4]` |
| Finite list/range | `L = list[1, 2, 3]`, `list[1...10]` |
| Index/slice | `L[2]`, `L[2...3]`, `L[list[3, 1]]` (1-based) |
| Number sets | `sets[N]` (1, 2, 3, …) and `sets[Z]` (0, 1, -1, 2, -2, …), only after `in` |
| Infinite sequence | `S = seq[k^2 for k in sets[N]]`; `k in sets[N]` (or `k ∈ sets[Z]`) is `k = seq[k for k in sets[N]]` |
| Explicit sequence selection | `S[1000]`, `total(S[1...10])`, `point[S[1...20], 0]` |
| Interval | `[0, 1)`, `(0, 1]`, `[2, 2]`, `(2, 2)` (empty) |
| Finite exact roots | `A = roots(a^2 - 4 for a)`, then `y = A x` |
| Symbolic display | `formula(a, (y-b)/x)` |

**Typed data is written `type[data]`.** Values that need a type say so with a keyword
and brackets: `point[…]`, `list[…]`, `seq[…]`, `sets[N]`. Operations keep parentheses
(`plot(…)`, `roots(…)`), and plain names are always ordinary variables. Intervals are
the one bare form: brackets alone mean nothing else, so `[0, 1)` needs no keyword.

**Operations name their parts.** Instead of remembering argument order, write the
expression, then `for` and the variable, then its clauses in any order:

| Operation | Keyword form | Positional form (still accepted) |
|---|---|---|
| Sum / product | `sum(k^2 for k from 1 to 10)`, `prod(k for k from 1 to 5)` | `sum(k^2, k, 1, 10)` |
| Definite integral | `int(sin(t) for t from 0 to x)` | `int(sin(t), t, 0, x)` |
| Limit | `limit(sin(x)/x for x to 0)`, `limit(1/x for x to 0 from right)` | `limit(1/x, x, 0, right)` |
| Taylor polynomial | `taylor(e^x for x at 0 degree 3)` | `taylor(e^x, x, 0, 3)` |
| Derivative | `diff(x^3 for x)`, `diff(x^3 for x order 2)` | `diff(x^3, x, 2)` |
| Antiderivative, roots, collect, apart | `integrate(x^2 for x)`, `roots(a^2 - 4 for a)` | `roots(a^2 - 4, a)` |
| Plot range / parametric | `plot(t^2 for t in (0, 2])`, `parametric(cos(u), sin(u) for u in [0, 2pi])` | `plot(t^2, t, (0, 2])` |

The clause words (`for`, `from`, `to`, `at`, `degree`, `order`, `in`) are only keywords
inside an operation's parentheses, so they stay usable as names elsewhere. In the
typeset editor, `sum(`, `int(` and `limit(` open a template; typing `for`, `from` and
`to` followed by a space moves to the variable, lower and upper fields (`to` goes to the
point of a limit, `from` to its side), so you can type the keyword form in reading order.

**Names are stable.** Adjacent letters form one name: `ab` is an identifier,
independent of whether a or b is defined elsewhere. Write `a b` or `a*b` to multiply.
Numbers can touch names: `2pi x` is `2*pi*x`. Functions require parentheses.

Complex expressions also accept compact imaginary products: `ix` means `i*x`, and
`it` means `i*t`, unless that complete name is explicitly defined or is a function
parameter. This shorthand applies in Complex Mode and inside `re(...)`, `im(...)`
and `arg(...)`, which evaluate complex arguments in either mode. For example,
`simplify(re(e^ix))` gives `cos(x)` without assigning a value to x. Explicit bindings
for `i` in real mode retain their meaning.

Operator precedence, from low to high, is `+ -`, `* /` and implicit multiplication,
unary minus, factorial, then powers (right associative). Thus `-x^2 = -(x^2)`,
`1/2x = (1/2)x`, `x^2/3 = (x^2)/3`, and `3^2! = (3^2)!`.
Ordinary typing and pasted source follow the same rules. Use parentheses to extend
an exponent or denominator; arrows still allow editing inside a selected structure.
Greek and root shortcuts convert complete words at delimiters, or use backslash
commands such as `\pi`, `\sqrt` and `\frac`. Substrings in identifiers such as
`spin`, `pix` and `sqrtvalue` are never converted. LaTeX copy/paste preserves names.

In typeset input, **Tab** after `sum`, `prod`, `int` or `limit` creates the operator's
editable fields. Tab moves to the next field, and **Shift+Tab** moves back; for sums
and products the order is variable, start, end, then body. Inside a fraction, Tab
moves from numerator to denominator, then outside. Tab at the expression's outer
level moves focus to the next control, as it does in plain-text input. Space still
completes operator names, and arrow keys can visit every character and field.

Undo keeps a committed batch of text together, including shortcut conversions.
Individual shortcut completion can still be undone back to its original letters.
An edit that changes nothing preserves Redo, and pasting empty text preserves the
current selection. International keyboards can enter printable AltGr characters.

**Equations do not introduce bindings.** `2a = 4` reports a solution but does not
define a for another row. A background exact solution changes the feedback only.
Define `a = 2` explicitly, or bind a finite root list with `roots`. Multiple roots
always form a list, even if there is only one, so their type is stable.

**Sequences are not sets or finite lists.** They retain order and repeats. `sets[N]`
enumerates 1, 2, 3, ...; `sets[Z]` enumerates 0, 1, -1, 2, -2, ... . The number sets
have their own bracket syntax, like `list[…]` and `seq[…]`, so a plain `N` or `Z` is
always an ordinary variable: `N = 5` and `k in sets[N]` can sit side by side. Writing
`k in N` (or `ℕ`) is an error that points to `sets[N]`; the editor draws the set as
sets[ℕ] and the variable as an italic N, and LaTeX `\mathbb{N}` pastes as `sets[N]`. The displayed preview
is labelled and never participates in arithmetic. Select finite terms explicitly
before plotting or aggregating. `k in sets[N]` is shorthand for
`k = seq[k for k in sets[N]]`:
k is the whole lazy sequence, so `k[5]` is 5 and `y = k[1...3] x` draws three lines,
while `total(k)` asks for a finite slice. Only a single name may come before `in`, and
not x or y. `set[...]` is rejected with instructions for the sequence syntax.

**Lists have equal-length pairing.** Scalars broadcast over a list; two lists must
have equal lengths. Mismatches are errors rather than silently discarded elements.
This also applies to point coordinates and curve families.

**Intervals are continuous lists.** Use them directly in arithmetic, functions,
and point coordinates. For example:

```text
t = [0, 2pi]
point[cos(t), sin(t)]
u = [-2, 2]
point[u, u^2]
point[2 [0, 1], 3]
```

These draw a circle, a parabola segment, and a horizontal segment. Values remain
continuous; no finite list of samples is substituted into the expression. Reusing
an interval preserves pairing, so `point[u, u-u]` stays on the x-axis. Two intervals
pair at the same fractional position: `point[[0, 2], [10, 20]]` draws the segment
from (0, 10) to (2, 20). Finite lists can broadcast alongside an interval to draw
several curves, as in `point[u, list[1, 2] u]`.

In Complex Mode, a standalone expression with interval values draws a curve in
the complex plane: the horizontal coordinate is its real part and the vertical
coordinate is its imaginary part. For example, `e^(i t)` with `t = [0, 2pi]`
draws a circle. `e^(t(ln(sqrt(a^2+b^2))+i*atan(b,a)))` with `t = [0, 100]`,
`a = 1`, and `b = 1/2` draws an outward spiral. Use radians for these examples.

Open endpoints remain excluded. `[a, a]` gives one value; an equal-endpoint interval
with an open end is empty. Arithmetic feedback labels mapped expressions as
continuous values. Finite indexing and `total`, `mean`, and `length` require finite
lists. Continuous curve families that are linear in their interval values, such as
`y = [1, 2) x`, shade the exact band. For nonlinear mappings, use point coordinates,
such as `point[u, sin(u)]`.

**Plot ranges and parameters are local.** `parametric(..., u, [a, b])` binds u inside
that curve; a definition of u elsewhere does not change it. `point[...]` always
means a point and never infers a parameter. `tmin` and `tmax` are ordinary names.
Every function of one parameter plots consistently, regardless of its parameter
name. Intervals can be named and used as ranges; an open endpoint is excluded.
When typing an interval, enter either `)` or `]` to choose its closing endpoint;
the editor replaces its automatic closer, so `[0, 1)` and `(0, 1]` work directly.
A constant assignment such as `r = 1` defines a reusable variable on every grid.
Use `x^2 + y^2 = r^2` for a circle with that radius. Assignments to `r`
that use `theta` describe polar curves; `theta` remains a reserved coordinate. Changing
the displayed grid does not change the meaning of a row.

## 3D graphs

Any equation that uses `z` is a surface, and a point or curve with a third coordinate
lies in space. While such a row is visible, the graph is a 3D view; hide or delete the
last one and it returns to the plane, which keeps its own position.

```text
z = sin(x) cos(y)
x^2 + y^2 + z^2 = 9
y = z^2
z = list[1, 2, 3] x
point[1, 2, 3]
parametric(4cos(u), 4sin(u), u/2 for u in [0, 6pi])
t = [0, 6pi]
point[cos(t), sin(t), t]
```

Three list columns of a table plot as points in space with `point[x_1, y_1, z_1]`.

**`z` is a coordinate, like `x` and `y`.** It is reserved everywhere, also in Complex
Mode, so `z = 5` is a plane rather than a definition with a slider; pick another name
for a value. Function parameters and bound variables may still be called z, as in
`f(z) = z^2`. A function of two names, such as `g(u, v) = u v`, draws nothing until a
row like `z = g(x, y)` uses it.

**Rows keep their meaning in a 3D view.** A row without z stays flat: `y = x^2` is a
curve in the plane z = 0, not a sheet above it, and `x = 3` is a line. Write `x = 3 + 0 z`
for the plane. Inequalities show their boundary there, without shading.

Drag to turn the view, and Shift+drag to pan. The wheel zooms, Ctrl+wheel only x and y,
and Shift+wheel only z (the last two need an unlocked aspect ratio). Arrow keys turn the
view, and Ctrl+0 resets it. The box is always drawn as a cube, so stretching one axis
shows the shape with a different scale on that axis. Graph Settings has all three ranges.
Hover over a surface, curve or point to read a point on it. The view is saved with the
document.

Surfaces are sampled on a grid: 72 cells a side for `z = f(x, y)`, and 40 for implicit
surfaces by marching tetrahedra. So, as with implicit curves, features smaller than a
cell can be missed; zoom in to see them. Surfaces follow the edge of their domain,
and jumps and poles leave a gap rather than a false wall.

Not available in 3D: inequalities using z, surfaces given by two parameters, continuous
families such as `z = [0, 1) x`, logarithmic axes, deep zoom, points of interest, and
dragging points. Intersecting surfaces are drawn correctly but are always opaque.

## Algebra and calculus

`diff`, `integrate`, `simplify`, `expand`, `factor`, `cancel`, `together`, `apart`,
`collect`, `trigsimp`, `limit` and `taylor` work in expressions and definitions.
Examples:

```text
g(t) = diff(t^3)
diff(x^5 for x order 3)
limit(sin(x)/x for x to 0)
limit(abs(x)/x for x to 0 from right)
taylor(e^x for x at 0 degree 4)
sum(k^2 for k from 1 to 10)
prod(k, k, 1, 5)
int(t^2, t, 0, 1)
```

Standalone algebra and derivative rows accept unassigned symbols. For example,
`expand((a+b)^2)` displays `a^2 + 2*a*b + b^2`, `factor(a^2-b^2)` displays
`(a-b)*(a+b)`, and `diff(a*x^2, x)` displays `2*a*x`. Partial derivatives compose:
`diff(diff(x^2*y^3, x), y)` gives `6*x*y^2`. Limits and Taylor polynomials can also
retain free parameters, such as `limit(sin(a*x)/x, x, 0)` giving `a`.
These rows display formulas without defining their free names. Once the remaining
parameters have definitions, a formula in x plots automatically using the cached
symbolic result. Function definitions still need explicit parameters or defined
constants for every name they use.

Choose an algebra operation for the form you need:

| Operation | Example | Result |
|---|---|---|
| Cancel common factors | `cancel((x^2-1)/(x-1))` | `x+1`, with x=1 still excluded |
| Combine fractions | `together(1/a + 1/b)` | `(a+b)/(a*b)` |
| Partial fractions | `apart(1/(x^2-1), x)` | `-1/(2*(x+1)) + 1/(2*(x-1))` |
| Collect powers | `collect(a*x + b*x + c*x^2, x)` | `c*x^2 + x*(a+b)` |
| Trigonometric identities | `trigsimp(2*sin(x)*cos(x))` | `sin(2*x)` |

`apart(f)` infers its variable using the same rules as `diff(f)`; an explicit
second argument chooses it. `collect` always requires the variable. The new
operations compose with each other and respect the document's angle units. Algebra
keeps user constant names, except that partial fractions use current parameter values
and recompute when those values change. This handles coincident poles, such as
`a = 0` in `apart(1/(x^2-a^2), x)`. Free parameters remain allowed when the
decomposition needs no additional conditions; otherwise define them first. Partial
fractions require a rational expression in the chosen variable; unsupported forms
produce a diagnostic.

Sums/products always require the explicit index. A three-argument definite integral
binds x; use four arguments for another integration variable. This binding never
depends on which names are defined in other rows.

Algebra and derivatives preserve the original domain. Simplifying `x/x` gives a
value of 1 away from zero while keeping zero undefined. Menu-generated source may
use `guard(value, original_expression)` to retain exclusions across save/copy/paste.
A guard also remains meaningful to later plotting and numerical root validation.
Generated results of restricted curves may use `restrict(value, coordinate, interval)`
to preserve the range when reused as a function.

Antiderivatives use the current values of user parameters and are recomputed when
those values change. This handles `a = 0` in `integrate(e^(a x))` and `b = -1` in
`integrate(x^b)` without discarding exceptional cases. Parameter-dependent results
that still need unsupported conditional expressions produce an explicit diagnostic.
Real antiderivatives use `ln(abs(x))` for the reciprocal case. The integration
constant is zero. Derivatives and algebra keep user parameter names where possible.

The row menu also offers symbolic operations. Solve for inserts a `formula(...)`
display row, which may contain free names without being an error or a definition.
Copy/edit the expression into an explicit definition when desired. Infinite equation
solutions are displayed exactly where supported; generated plotting rows use an
explicit finite selection. SymPy runs in cancellable background worker processes;
Ctrl+J opens Tasks.

## Graphs, data and interaction

The canvas supports adaptive grids, cursor-centred zoom, pan, tracing, intercepts,
stationary points and supported intersections. Ctrl+wheel and Shift+wheel zoom
individual axes when the aspect ratio is unlocked. Graph Settings controls ranges,
linear/log axes, grid appearance, radians/degrees, complex mode and aspect ratio.
Themes can follow the system or use Light/Dark explicitly.

Expression rows can be reordered, hidden, recoloured and deleted. Unfinished typing
keeps the last valid drawing. Literal numeric definitions have sliders; bounds and
steps can be expressions. Undefined names offer explicit Add slider buttons.
Suitable literal points and slider-controlled point coordinates can be dragged.

Tables support named list columns, formula columns, point/line styles and UTF-8 CSV
import. Formula columns can refer to other columns and definitions without cycles.
Documents save as `.ograph.json`; plots export as PNG/JPEG. The syntax may still change
before a first release, and old spellings aren't migrated.

## Numerical limits

* Numerical root feedback states its search interval `[-1e8, 1e8]` and explicitly
  warns that roots may be missed. It never claims exactness from a zero floating-point
  residual. Use Solve for exact roots, or `roots` for a finite exact root list.
* Implicit contours are sampled numerically. Recognizable positive outer powers and
  repeated identical factors are reduced to their underlying zero set, retaining the
  original domain. Tiny features and unrecognized tangent zero sets can still be
  missed; an empty numerical drawing is not a proof that no solutions exist.
* Chained inequalities shade the intersection of their comparisons, preserving
  strict/dashed and non-strict/solid boundaries. Plot intervals provide another
  way to restrict one curve without shading a region.
* Default arithmetic is real-valued; undefined values are not drawn. Complex mode
  uses principal branches, so a negative fractional power can differ from the real
  odd-root convention. Complex curves are drawn where their values are real.
* Quadrature, large sums, long lists and curve families have bounded work budgets.
  Definite integrals are numerical; divergent or difficult interior singularities
  require care. Exact Value asks SymPy for a closed result.
* General symbolic solution families may require conditions or special functions
  outside the supported language. Such results are reported explicitly.

See [design notes](docs/DESIGN.md) and the
[semantic changes](docs/SEMANTICS.md) for implementation details.

## License

OpenGraph is released under the [MIT License](LICENSE). The bundled DejaVu fonts have their
own [license](src/opengraph/ui/assets/fonts/LICENSE-DejaVu.txt).
