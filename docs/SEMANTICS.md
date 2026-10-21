# Semantics cleanup

This unreleased syntax revision resolves the calculator review's correctness and
predictability issues. Document migration was intentionally excluded.

| Concern | Resolution |
|---|---|
| Sample-based coefficient proof | Structural affine decomposition; varying coefficients remain implicit |
| Conditional antiderivatives | Current parameter values form the request/cache key; never discard Piecewise cases |
| Typed/pasted precedence | Delimiter and implicit-product rules match the text parser |
| Simplification changes domains | Guard transformed values with the original numeric expression; serialize guards |
| Sets used as sequences | Explicit lazy `seq[...]`, ordered terms and labelled previews |
| Membership used as generation | `k in sets[N]` is shorthand for the lazy sequence `k = seq[k for k in sets[N]]`; never truncated |
| Set N vs variable N | Number sets have their own syntax, `sets[N]` / `sets[Z]`; a plain N or Z is always a variable |
| Equations introduce variables | Equation feedback never mutates the definition environment; use definitions or `roots` |
| Names split by namespace | Identifiers are atomic; spaces/operators express multiplication. Complex contexts also accept compact imaginary products such as ix, with explicit bindings taking priority |
| Word-substring conversions | Only full shortcut tokens convert at delimiters; LaTeX also preserves names |
| Grid changes semantics | Grid appearance is excluded from compile options; polar coordinates are consistent |
| Magic parameter names/ranges | Explicit `parametric` binding and interval; all unary functions plot consistently |
| Intervals lack utility | Intervals act as continuous lists in arithmetic, functions, and point coordinates; named intervals draw curves directly and linear families shade exact bands |
| Shortest-list truncation | Reject unequal lengths in both expansion and vectorized values |
| Bounded root scan looks complete | State bounds and possible omissions; numeric roots use approximate feedback |
| Squared implicit zero sets disappear | Normalize recognizable outer powers/factors and preserve the original domain |
| Rearranged formulas become errors | Insert valid symbolic-only `formula` display rows |
| Symbolic operations require numeric assignments | Standalone algebra, derivatives, limits and Taylor rows accept free symbols without defining them |
| Rational algebra lacks explicit result forms | `cancel`, `together`, `apart`, `collect` and `trigsimp` preserve original domains and compose through the CAS cache |

Limits remain numerical: general contour completeness is not decidable from a pixel
grid, and a bounded numerical root scan is not an exhaustive solver. The UI reference
and README describe these limits rather than presenting an empty drawing as proof.
