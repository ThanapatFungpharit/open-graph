"""Pratt parser with stable identifiers and explicit point/list/sequence syntax.

Relations are statement-level. Products and division (including implicit products)
associate left; powers associate right and bind above factorial and unary signs.
Thus 1/2x = (1/2)x and 3^2! = (3^2)!. Adjacent letters are one identifier.
Points use point[a,b], finite lists list[...], infinite sequences
seq[... for k in sets[N]] or sets[Z], and intervals [a,b], (a,b), [a,b), (a,b].
``k in sets[N]`` is shorthand for ``k = seq[k for k in sets[N]]``; a plain N is a variable.
"""

from __future__ import annotations

from functools import lru_cache

from opengraph.engine.ast import (
    BinOp,
    Call,
    ComparisonChain,
    Index,
    IntervalExpr,
    ListExpr,
    Name,
    Node,
    Num,
    RangeExpr,
    Relation,
    SequenceExpr,
    Statement,
    TupleExpr,
    UnaryOp,
)
from opengraph.engine.errors import ParseError, Span
from opengraph.engine.lexer import Kind, Token, tokenize

RELATIONS = frozenset({"=", "<", ">", "<=", ">="})
_BINARY_BP = {"+": 10, "-": 10, "*": 20, "/": 20, "^": 40}
_IMPLICIT_BP = 20
_PREFIX_BP = 30
_FACTORIAL_BP = 35
# Number sets are written sets[N] (1, 2, 3, …) and sets[Z] (0, 1, -1, …), so a plain N
# or Z is always an ordinary variable. ℕ and ℤ are reserved and point to that spelling.
_SET_NAMES = {"N": "N", "Z": "Z"}
_BLACKBOARD = {"ℕ": "N", "ℤ": "Z"}
_EXPECTED_SET = "Expected sets[N] or sets[Z] after 'in'"
_ONLY_SETS = "Only sets[N] (natural numbers) and sets[Z] (integers) are supported after 'in'"
# "in" and "for" are keywords only where they can be: "in" before N/Z/ℕ/ℤ, "for" inside
# seq[…]. Elsewhere they are ordinary names, so complex-mode "3in" is still 3·i·n.
_SET_WORDS = frozenset({"in", "for"})

# Operations can name their parts instead of relying on argument order:
# sum(k^2 for k from 1 to 10) is sum(k^2, k, 1, 10). Each form lists the clauses it
# requires and allows (in positional order), the leading expressions, and an example.
_CLAUSE_WORDS = frozenset({"in", "from", "to", "at", "degree", "order"})


class _Form:
    def __init__(
        self,
        required: tuple[str, ...],
        optional: tuple[str, ...],
        items: int,
        example: str,
        max_items: int | None = None,
    ) -> None:
        self.required, self.optional, self.items, self.example = required, optional, items, example
        self.max_items = items if max_items is None else max_items


_RANGE = ("from", "to")
_FORMS: dict[str, _Form] = {
    **{
        name: _Form(_RANGE, (), 1, f"{name}(k^2 for k from 1 to 10)")
        for name in ("sum", "Σ", "Sigma", "prod", "Π", "Pi")
    },
    **{
        name: _Form(_RANGE, (), 1, f"{name}(sin(t) for t from 0 to x)")
        for name in ("int", "integral", "∫")
    },
    "limit": _Form(("to",), ("from",), 1, "limit(sin(x)/x for x to 0), or … to 0 from right"),
    "taylor": _Form(("at", "degree"), (), 1, "taylor(e^x for x at 0 degree 3)"),
    "diff": _Form((), ("order",), 1, "diff(x^3 for x), or diff(x^3 for x order 2)"),
    "integrate": _Form((), (), 1, "integrate(x^2 for x)"),
    "roots": _Form((), (), 1, "roots(a^2 - 4 for a)"),
    "collect": _Form((), (), 1, "collect(a x + b x for x)"),
    "apart": _Form((), (), 1, "apart(1/(x^2 - 1) for x)"),
    "plot": _Form(("in",), (), 1, "plot(t^2 for t in (0, 2])"),
    "parametric": _Form(("in",), (), 2, "parametric(cos(u), sin(u) for u in [0, 2pi])", 3),
}


def _join(a: Span, b: Span) -> Span:
    return (a[0], b[1])


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = tokenize(text)
        self.pos = 0
        # Number of open |...| groups in the current parenthesis level; a "|" in infix
        # position closes a group when this is positive, otherwise it starts a new one.
        self.abs_depth = 0
        # Open seq[…] bodies: there, "for" ends the body instead of multiplying.
        self.seq_depth = 0
        # Words that end the expression being parsed instead of multiplying it: "for"
        # in an operation's arguments, the clause words in a clause's value.
        self.stop_words: frozenset[str] = frozenset()

    # -- token helpers -------------------------------------------------------------
    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def at_op(self, *ops: str) -> bool:
        token = self.peek()
        return token.kind is Kind.OP and token.text in ops

    def describe(self, token: Token) -> str:
        return "end of input" if token.kind is Kind.END else f"'{token.text}'"

    # -- grammar ---------------------------------------------------------------------
    def parse_statement(self) -> Statement:
        self.check_membership_shape()
        left = self.parse_expr(0)
        if isinstance(left, TupleExpr) and self.at_op(","):
            return self.parse_point_list(left)
        if self.at_in():
            return self.parse_membership(left)
        result: Statement = left
        if self.peek().kind is Kind.OP and self.peek().text in RELATIONS:
            op_token = self.advance()
            right = self.parse_expr(0)
            comparisons = [Relation(op_token.text, left, right, _join(left.span, right.span))]
            while self.peek().kind is Kind.OP and self.peek().text in RELATIONS:
                if op_token.text == "=" or self.peek().text == "=":
                    raise ParseError("Only one '=' is allowed", self.peek().span)
                left = right
                op_token = self.advance()
                right = self.parse_expr(0)
                comparisons.append(
                    Relation(op_token.text, left, right, _join(left.span, right.span))
                )
            result = (
                comparisons[0]
                if len(comparisons) == 1
                else ComparisonChain(
                    tuple(comparisons), _join(comparisons[0].span, comparisons[-1].span)
                )
            )
        self.expect_end()
        return result

    def at_in(self) -> bool:
        token = self.peek()
        return (token.kind is Kind.NAME and token.text == "in") or (
            token.kind is Kind.OP and token.text == "∈"
        )

    def check_membership_shape(self) -> None:
        """A row that is just ``k in``, ``k in N`` or ``k in R`` means membership, even
        though ``in`` is only a keyword before ``sets[…]``: say what is missing or how to
        write the set, instead of reading it as the product k·in·N."""
        kinds = [t.kind for t in self.tokens]
        if kinds[:2] != [Kind.NAME, Kind.NAME] or self.tokens[1].text != "in":
            return
        if self.tokens[0].text in ("x", "y"):  # never a membership; complex x·i·n
            return
        if kinds[2:] == [Kind.END]:
            raise ParseError(_EXPECTED_SET, self.tokens[2].span, incomplete=True)
        if kinds[2:] == [Kind.NAME, Kind.END]:
            self.reject_set_name(self.tokens[2], 3)

    def reject_set_name(self, token: Token, after: int) -> None:
        """``token`` stands where a number set belongs but isn't ``sets[…]``: a hint
        while it may still become ``sets``, otherwise how to write the set. Names that
        don't look like a set are left alone (they may be ordinary products)."""
        text = token.text
        if "sets".startswith(text) and self.unfinished(self.tokens[after]):
            raise ParseError(_EXPECTED_SET, token.span, incomplete=True)
        if text in _SET_NAMES or text in _BLACKBOARD:
            letter = _BLACKBOARD.get(text, text)
            kind = "natural numbers" if letter == "N" else "integers"
            bare = f"; a bare {text} is a variable" if text in _SET_NAMES else ""
            raise ParseError(f"Write sets[{letter}] for the {kind}{bare}", token.span)
        if (len(text) == 1 and text.isupper()) or text in "ℝℚℂ":
            raise ParseError(_ONLY_SETS, token.span)

    def starts_set(self, index: int) -> bool:
        """``sets[`` starts at token ``index``."""
        if index + 1 >= len(self.tokens):  # past END
            return False
        token, following = self.tokens[index], self.tokens[index + 1]
        return (
            token.kind is Kind.NAME
            and token.text == "sets"
            and following.kind is Kind.OP
            and following.text == "["
        )

    def is_keyword(self, index: int) -> bool:
        """Whether the name token at ``index`` is the keyword ``in`` (``in sets[N]``)
        or ``for`` (inside ``seq[…]``) rather than an ordinary name."""
        token = self.tokens[index]
        if token.text == "in":
            return self.starts_set(index + 1)
        return token.text == "for" and self.seq_depth > 0

    def unfinished(self, token: Token) -> bool:
        """``token`` is the end of the row, or one of the closing brackets the editor
        typed ahead of the caret (``seq[2k ]``, ``seq[k for k in sets[]]``): more typing
        may still fix the error."""
        index = self.tokens.index(token)
        for following in self.tokens[index:]:
            if following.kind is Kind.END:
                return True
            if following.text not in ("]", ")"):
                return False
        return True  # pragma: no cover - the token list always ends with END

    def parse_domain(self) -> str:
        """The number set after ``in``: ``sets[N]`` or ``sets[Z]``."""
        if self.unfinished(self.peek()):
            raise ParseError(_EXPECTED_SET, self.peek().span, incomplete=True)
        if not self.starts_set(self.pos):
            token = self.peek()
            if token.kind is Kind.NAME:
                self.reject_set_name(token, self.pos + 1)
            raise ParseError(_ONLY_SETS, token.span)
        self.advance()  # sets
        opener = self.advance()
        inner = self.peek()
        if self.unfinished(inner):
            raise ParseError("Write N or Z inside sets[…]", inner.span, incomplete=True)
        if inner.kind is not Kind.NAME or inner.text not in _SET_NAMES:
            raise ParseError(_ONLY_SETS, inner.span)
        self.advance()
        if not self.at_op("]"):
            if self.peek().kind is Kind.END:
                raise ParseError("Missing closing ']'", opener.span, incomplete=True)
            raise ParseError(
                f"Expected ']' but found {self.describe(self.peek())}", self.peek().span
            )
        self.advance()
        return _SET_NAMES[inner.text]

    def parse_membership(self, left: Node) -> Statement:
        """``k in sets[N]``: shorthand for ``k = seq[k for k in sets[N]]`` (still lazy)."""
        if not isinstance(left, Name):
            raise ParseError(
                "Write one name before 'in', like k in sets[N], or define a sequence with "
                "seq[2k for k in sets[N]]",
                left.span,
            )
        if left.name in ("x", "y"):
            raise ParseError(f"'{left.name}' is a plot axis; use another name, like k", left.span)
        self.advance()
        domain = self.parse_domain()
        self.expect_end()
        span = _join(left.span, self.tokens[self.pos - 1].span)
        sequence = SequenceExpr(left, left.name, domain, left.name, span)
        return Relation("=", left, sequence, span)

    def parse_point_list(self, first: TupleExpr) -> Statement:
        """``(1, 2), (3, 4), ...``: several points in one row, as a list of tuples."""
        points: list[Node] = [first]
        while self.at_op(","):
            self.advance()
            point = self.parse_expr(0)
            if not isinstance(point, TupleExpr):
                raise ParseError(
                    "Separate points with commas, like point[1, 2], point[3, 4]", point.span
                )
            points.append(point)
        self.expect_end()
        return ListExpr(tuple(points), _join(first.span, points[-1].span))

    def expect_end(self) -> None:
        token = self.peek()
        if token.kind is Kind.END:
            return
        if token.text in (")", "]", "}"):
            raise ParseError(f"Unmatched '{token.text}'", token.span)
        if token.text == ",":
            raise ParseError(
                "Unexpected ','; write several points as point[1, 2], point[3, 4]", token.span
            )
        raise ParseError(f"Unexpected {self.describe(token)}", token.span)

    def parse_expr(self, rbp: int) -> Node:
        left = self.parse_prefix()
        while True:
            token = self.peek()
            if token.kind is Kind.OP and token.text in _BINARY_BP:
                lbp = _BINARY_BP[token.text]
                if lbp <= rbp:
                    break
                self.advance()
                right = self.parse_expr(lbp - 1 if token.text == "^" else lbp)
                left = BinOp(token.text, left, right, _join(left.span, right.span))
            elif token.kind is Kind.OP and token.text == "!":
                if rbp >= _FACTORIAL_BP:
                    break
                self.advance()
                span = _join(left.span, token.span)
                left = Call("factorial", (left,), span, token.span)
            elif self.starts_index(token, left):
                left = self.parse_index(left)
            elif self.starts_implicit_operand(token):
                if rbp >= _IMPLICIT_BP:
                    break
                right = self.parse_expr(_IMPLICIT_BP)
                left = BinOp("*", left, right, _join(left.span, right.span))
            else:
                break
        return left

    def starts_index(self, token: Token, left: Node) -> bool:
        """``L[2]``: a bracket touching a list-valued operand indexes it (``a [1, 2]``
        with a space, or after a number, still multiplies)."""
        return (
            token.kind is Kind.OP
            and token.text == "["
            and token.span[0] == left.span[1]
            and isinstance(left, Name | Call | ListExpr | RangeExpr | SequenceExpr | Index)
        )

    def parse_index(self, target: Node) -> Node:
        """``[i]``, ``[a...b]`` (a slice) or ``[list[i, j]]`` after ``target``."""
        opener = self.advance()
        saved_depth, self.abs_depth = self.abs_depth, 0
        index = self.parse_expr(0)
        if self.at_op(","):
            raise ParseError(
                "An index takes one position; pick several with L[list[1, 3]]", self.peek().span
            )
        if self.at_op("..."):
            self.advance()
            index = RangeExpr(index, self.parse_expr(0), _join(index.span, self.peek().span))
        if not self.at_op("]"):
            if self.peek().kind is Kind.END:
                raise ParseError("Missing closing ']'", opener.span, incomplete=True)
            raise ParseError(
                f"Expected ']' but found {self.describe(self.peek())}", self.peek().span
            )
        closer = self.advance()
        self.abs_depth = saved_depth
        return Index(target, index, _join(target.span, closer.span))

    def starts_implicit_operand(self, token: Token) -> bool:
        if token.kind is Kind.NAME:
            return token.text not in self.stop_words and not self.is_keyword(self.pos)
        if token.kind is Kind.NUMBER:
            # "x2" and "(x)2" multiply, but "2 3" and "x + 2 3" are typos.
            return self.tokens[self.pos - 1].kind is not Kind.NUMBER
        if token.kind is Kind.OP:
            return token.text in ("(", "{", "[") or (token.text == "|" and self.abs_depth == 0)
        return False

    def parse_prefix(self) -> Node:
        token = self.advance()
        if token.kind is Kind.NUMBER:
            try:
                return Num(float(token.text), token.span)
            except ValueError:  # pragma: no cover - regex only admits valid floats
                raise ParseError(f"Invalid number '{token.text}'", token.span) from None
        if token.kind is Kind.NAME:
            if token.text == "set" and self.at_op("["):
                raise ParseError("Use seq[...] for an ordered sequence", token.span)
            if token.text == "sets" and self.at_op("["):
                raise ParseError(
                    "A number set goes after 'in', like k in sets[N] or seq[2k for k in sets[N]]",
                    token.span,
                )
            if token.text == "point":
                if self.at_op("["):
                    return self.parse_point(token)
                if self.at_op("("):
                    raise ParseError("Write a point as point[x, y]", token.span)
            if token.text == "list" and self.at_op("["):
                return self.parse_bracket_list(token)
            if token.text == "seq" and self.at_op("["):
                return self.parse_set(token)
            if self.at_op("("):
                return self.parse_call(token)
            return Name(token.text, token.span)
        if token.kind is Kind.OP:
            if token.text == "-":
                operand = self.parse_expr(_PREFIX_BP)
                return UnaryOp("-", operand, _join(token.span, operand.span))
            if token.text == "+":
                return self.parse_expr(_PREFIX_BP)
            if token.text in ("(", "{"):
                return self.parse_group(token)
            if token.text == "|":
                return self.parse_abs(token)
            if token.text == "[":
                return self.parse_bracket(token)
        if token.kind is Kind.END:
            end = len(self.text)
            raise ParseError("Expression is incomplete", (max(0, end - 1), end), incomplete=True)
        raise ParseError(f"Unexpected {self.describe(token)}", token.span)

    def parse_list(self, opener: Token) -> tuple[list[Node], Token]:
        """Parse comma-separated expressions and the opener's matching closer."""
        close = {"{": "}", "[": "]", "(": ")"}[opener.text]
        saved_depth, self.abs_depth = self.abs_depth, 0
        items = [self.parse_expr(0)]
        while self.at_op(","):
            self.advance()
            items.append(self.parse_expr(0))
        if not self.at_op(close):
            if self.peek().kind is Kind.END:
                raise ParseError(f"Missing closing '{close}'", opener.span, incomplete=True)
            raise ParseError(
                f"Expected '{close}' but found {self.describe(self.peek())}", self.peek().span
            )
        closer = self.advance()
        self.abs_depth = saved_depth
        return items, closer

    def parse_call(self, name_token: Token) -> Node:
        opener = self.advance()
        if self.at_op(")"):
            raise ParseError(
                f"'{name_token.text}()' needs an argument", _join(name_token.span, self.peek().span)
            )
        form = _FORMS.get(name_token.text)
        if form is None:
            args, closer = self.parse_list(opener)
        else:
            args, closer = self.parse_operation(name_token.text, form, opener)
        return Call(
            name_token.text, tuple(args), _join(name_token.span, closer.span), name_token.span
        )

    def parse_operation(self, name: str, form: _Form, opener: Token) -> tuple[list[Node], Token]:
        """An operation's arguments: positional (``sum(k^2, k, 1, 10)``) or, after
        ``for``, named clauses in any order (``sum(k^2 for k from 1 to 10)``), returned
        in the positional order the compiler expects."""
        saved_depth, self.abs_depth = self.abs_depth, 0
        saved_stops, self.stop_words = self.stop_words, frozenset({"for"})
        try:
            items = [self.parse_expr(0)]
            while self.at_op(","):
                self.advance()
                items.append(self.parse_expr(0))
        finally:
            self.stop_words = saved_stops
        keyword = self.peek()
        if not (keyword.kind is Kind.NAME and keyword.text == "for"):
            self.abs_depth = saved_depth
            return items, self.close_arguments(opener)
        self.advance()
        variable = self.advance()
        if self.unfinished(variable):
            raise ParseError("Name the variable after 'for'", variable.span, incomplete=True)
        if variable.kind is not Kind.NAME or variable.text in _SET_WORDS | _CLAUSE_WORDS:
            raise ParseError(f"Name the variable after 'for', like {form.example}", variable.span)
        clauses: dict[str, Node] = {}
        while self.peek().kind is Kind.NAME and self.peek().text in _CLAUSE_WORDS:
            word = self.advance()
            if word.text in clauses:
                raise ParseError(f"'{word.text}' is given twice", word.span)
            if word.text not in form.required + form.optional:
                raise ParseError(
                    f"{name} has no '{word.text}' part; write {form.example}", word.span
                )
            if self.unfinished(self.peek()):
                raise ParseError(
                    f"Expected a value after '{word.text}'", self.peek().span, incomplete=True
                )
            saved_stops, self.stop_words = self.stop_words, _CLAUSE_WORDS
            try:
                clauses[word.text] = self.parse_expr(0)
            finally:
                self.stop_words = saved_stops
        following = self.peek()
        if (
            following.kind is Kind.NAME
            and self.unfinished(self.tokens[self.pos + 1])
            and any(w.startswith(following.text) for w in form.required + form.optional)
        ):  # "diff(x^3 for x ord": the clause word is still being typed
            raise ParseError(f"Keep typing; write {form.example}", following.span, incomplete=True)
        missing = [word for word in form.required if word not in clauses]
        if missing:
            parts = " and ".join(f"'{word}'" for word in missing)
            raise ParseError(
                f"{name} needs {parts}; write {form.example}",
                following.span,
                incomplete=self.unfinished(following),
            )
        if not form.items <= len(items) <= form.max_items:
            count = "one expression" if form.items == 1 else f"{form.items} expressions"
            if form.max_items != form.items:
                count = f"{form.items} or {form.max_items} expressions"
            raise ParseError(
                f"{name} takes {count} before 'for'; write {form.example}", opener.span
            )
        if name == "limit" and "from" in clauses:
            side = clauses["from"]
            if not (isinstance(side, Name) and side.name in ("left", "right")):
                raise ParseError("limit: write 'from left' or 'from right'", side.span)
        self.abs_depth = saved_depth
        closer = self.close_arguments(opener)
        var = Name(variable.text, variable.span)
        values = [clauses[word] for word in (*form.required, *form.optional) if word in clauses]
        return [*items, var, *values], closer

    def close_arguments(self, opener: Token) -> Token:
        """The ``)`` that ends an argument list."""
        if not self.at_op(")"):
            if self.peek().kind is Kind.END:
                raise ParseError("Missing closing ')'", opener.span, incomplete=True)
            raise ParseError(
                f"Expected ')' but found {self.describe(self.peek())}", self.peek().span
            )
        return self.advance()

    def parse_point(self, name_token: Token) -> Node:
        """``point[x, y]`` or ``point[x, y, z]``."""
        opener = self.advance()
        if self.at_op("]"):
            raise ParseError(
                "point[] needs two or three coordinates, like point[1, 2]",
                _join(name_token.span, self.peek().span),
            )
        items, closer = self.parse_list(opener)
        span = _join(name_token.span, closer.span)
        if len(items) not in (2, 3):
            raise ParseError("A point needs two or three coordinates, like point[1, 2]", span)
        return TupleExpr(tuple(items), span)

    def parse_group(self, opener: Token) -> Node:
        """``(expr)``, ``{expr}`` or an interval ``(a, b)`` / ``(a, b]``."""
        close = "}" if opener.text == "{" else ")"
        if self.at_op(close):
            message = "Empty braces" if opener.text == "{" else "Empty parentheses"
            raise ParseError(message, _join(opener.span, self.peek().span))
        if opener.text == "(":
            return self.parse_enclosure(opener)
        items, closer = self.parse_list(opener)
        if len(items) != 1:
            raise ParseError(
                "Braces only group; write a point as point[x, y] and a list as list[a, b]",
                _join(opener.span, closer.span),
            )
        return items[0]

    def parse_bracket(self, opener: Token) -> Node:
        """A bare ``[``: only an interval ``[a, b]`` / ``[a, b)`` is allowed."""
        if self.at_op("]"):
            raise ParseError(
                "Empty brackets; write an empty list as list[]",
                _join(opener.span, self.peek().span),
            )
        return self.parse_enclosure(opener)

    def parse_enclosure(self, opener: Token) -> Node:
        """After ``(`` or ``[``: a parenthesised expression or an interval.

        One expression closed by ``)`` is a group; two separated by a comma and closed
        by ``)`` or ``]`` are an interval. Anything else gets a pointer to the right
        spelling of a point or a list.
        """
        saved_depth, self.abs_depth = self.abs_depth, 0
        items = [self.parse_expr(0)]
        while self.at_op(","):
            self.advance()
            items.append(self.parse_expr(0))
        token = self.peek()
        if token.kind is not Kind.OP or token.text not in (")", "]"):
            self.abs_depth = saved_depth
            if len(items) == 2:
                expected = "')' or ']'"
            else:
                expected = "')'" if opener.text == "(" else "']'"
            if token.kind is Kind.END:
                raise ParseError(f"Missing closing {expected}", opener.span, incomplete=True)
            if token.text == "...":
                raise ParseError("Ranges are written list[a...b]", token.span)
            raise ParseError(f"Expected {expected} but found {self.describe(token)}", token.span)
        closer = self.advance()
        self.abs_depth = saved_depth
        span = _join(opener.span, closer.span)
        if len(items) == 1:
            if opener.text == "(" and closer.text == ")":
                return items[0]
            if opener.text == "[":
                raise ParseError(
                    "Brackets alone are an interval like [1, 2]; write a list as list[...]", span
                )
            raise ParseError(f"Expected ')' but found '{closer.text}'", closer.span)
        if len(items) > 2:
            raise ParseError(
                "Too many values; write a point as point[x, y], a list as list[a, b, c] "
                "and an interval with two endpoints like [a, b]",
                span,
            )
        return IntervalExpr(items[0], items[1], opener.text == "[", closer.text == "]", span)

    def parse_bracket_list(self, name_token: Token) -> Node:
        """``list[]``, ``list[a, b, ...]`` or the integer range ``list[a...b]``."""
        opener = self.advance()
        saved_depth, self.abs_depth = self.abs_depth, 0
        items: list[Node] = []
        stop: Node | None = None
        if not self.at_op("]"):
            items.append(self.parse_expr(0))
            if self.at_op("..."):
                self.advance()
                stop = self.parse_expr(0)
            else:
                while self.at_op(","):
                    self.advance()
                    items.append(self.parse_expr(0))
        if not self.at_op("]"):
            if self.peek().kind is Kind.END:
                raise ParseError("Missing closing ']'", opener.span, incomplete=True)
            raise ParseError(
                f"Expected ']' but found {self.describe(self.peek())}", self.peek().span
            )
        closer = self.advance()
        self.abs_depth = saved_depth
        span = _join(name_token.span, closer.span)
        if stop is not None:
            return RangeExpr(items[0], stop, span)
        return ListExpr(tuple(items), span)

    def parse_set(self, name_token: Token) -> Node:
        """``seq[body for k in sets[N]]``: ``body`` for every ``k`` in ℕ (or ℤ)."""
        opener = self.advance()
        saved_depth, self.abs_depth = self.abs_depth, 0
        if self.at_op("]"):
            raise ParseError(
                "A sequence needs a generator, like seq[2k for k in sets[N]]",
                _join(name_token.span, self.peek().span),
                incomplete=self.unfinished(self.peek()),
            )
        self.seq_depth += 1
        try:
            body = self.parse_expr(0)
        finally:
            self.seq_depth -= 1
        keyword = self.peek()
        if keyword.kind is Kind.END:
            raise ParseError("Missing closing ']'", opener.span, incomplete=True)
        if keyword.kind is not Kind.NAME or keyword.text != "for":
            raise ParseError(
                f"Expected 'for' but found {self.describe(keyword)}; "
                "write seq[2k for k in sets[N]]",
                keyword.span,
                incomplete=self.unfinished(keyword),
            )
        self.advance()
        variable = self.advance()
        if self.unfinished(variable):
            raise ParseError("Name the variable after 'for'", variable.span, incomplete=True)
        if variable.kind is not Kind.NAME or variable.text in _SET_WORDS:
            raise ParseError("Name the variable after 'for', like k", variable.span)
        if not self.at_in():
            raise ParseError(
                f"Expected 'in' after '{variable.text}'",
                self.peek().span,
                incomplete=self.unfinished(self.peek()) or self.typing_in(),
            )
        self.advance()
        domain = self.parse_domain()
        if not self.at_op("]"):
            if self.peek().kind is Kind.END:
                raise ParseError("Missing closing ']'", opener.span, incomplete=True)
            raise ParseError(
                f"Expected ']' but found {self.describe(self.peek())}", self.peek().span
            )
        closer = self.advance()
        self.abs_depth = saved_depth
        text = " ".join(self.text[opener.span[1] : keyword.span[0]].split())
        return SequenceExpr(body, variable.text, domain, text, _join(name_token.span, closer.span))

    def typing_in(self) -> bool:
        """``seq[k for k i]``: the next name is the start of ``in``, at the row's end."""
        token = self.peek()
        return (
            token.kind is Kind.NAME
            and "in".startswith(token.text)
            and self.unfinished(self.tokens[self.pos + 1])
        )

    def parse_abs(self, opener: Token) -> Node:
        self.abs_depth += 1
        inner = self.parse_expr(0)
        if not self.at_op("|"):
            raise ParseError(
                "Missing closing '|'", opener.span, incomplete=self.peek().kind is Kind.END
            )
        closer = self.advance()
        self.abs_depth -= 1
        span = _join(opener.span, closer.span)
        return Call("abs", (inner,), span, opener.span)


@lru_cache(maxsize=1024)
def parse(text: str) -> Statement:
    """Parse one expression row. Raises :class:`ParseError` on invalid input.

    Results are cached by text: rows are re-parsed whenever any row in the document
    changes, and most of them have not.
    """
    if not text.strip():
        raise ParseError("Empty expression", (0, 0))
    return _Parser(text).parse_statement()
