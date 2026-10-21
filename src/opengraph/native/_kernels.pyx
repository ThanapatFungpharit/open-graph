# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Compiled loops for the geometry workers and the canvas (float64).

Each function reproduces a reference implementation in Python/NumPy exactly (the
fallbacks in :mod:`opengraph.graph.rendering` and :mod:`opengraph.graph.canvas`) and
releases the GIL while it runs, so tile workers no longer stall the interface thread.
"""

import numpy as np


cdef extern from "<math.h>" nogil:
    double rint(double x)


cdef Py_ssize_t _walk(
    const long long[:, ::1] pairs,
    const long long[::1] degree,
    const long long[::1] start,
    const long long[::1] incident,
    unsigned char[::1] used,
    long long[::1] nodes,
    Py_ssize_t count,
    Py_ssize_t index,
    long long at,
) noexcept nogil:
    """Follow segments from ``at`` along ``index`` until a node where other than two
    segments meet; appends the visited nodes and returns the new count."""
    cdef long long a, b, first
    nodes[count] = at
    count += 1
    while not used[index]:
        used[index] = 1
        a = pairs[index, 0]
        b = pairs[index, 1]
        at = b if a == at else a
        nodes[count] = at
        count += 1
        if degree[at] != 2:
            break
        first = incident[start[at]]
        index = incident[start[at] + 1] if first == index else first
    return count


def chain_segments(const long long[:, ::1] pairs, Py_ssize_t n_nodes):
    """Walk segments (node index pairs, no zero-length ones) into chains.

    Returns ``(nodes, lengths)``: every chain's node indices one after another, and
    each chain's length. Chains start at nodes where other than two segments meet (in
    the order the nodes first appear in ``pairs``); the closed loops that remain follow.
    """
    cdef Py_ssize_t m = pairs.shape[0], i, j, k, index, count = 0, chains = 0, n_order = 0
    cdef long long a, at
    cdef long long[::1] degree = np.zeros(n_nodes, dtype=np.int64)
    cdef long long[::1] start = np.zeros(n_nodes + 1, dtype=np.int64)
    cdef long long[::1] incident = np.empty(max(2 * m, 1), dtype=np.int64)
    cdef long long[::1] fill = np.zeros(n_nodes, dtype=np.int64)
    cdef long long[::1] order = np.empty(max(n_nodes, 1), dtype=np.int64)
    cdef unsigned char[::1] seen = np.zeros(n_nodes, dtype=np.uint8)
    cdef unsigned char[::1] used = np.zeros(max(m, 1), dtype=np.uint8)
    # A chain of k segments visits k + 1 nodes, and there are at most m chains.
    cdef long long[::1] nodes = np.empty(2 * m + 1, dtype=np.int64)
    cdef long long[::1] lengths = np.empty(m + 1, dtype=np.int64)
    cdef Py_ssize_t before
    with nogil:
        for i in range(m):
            for k in range(2):
                a = pairs[i, k]
                degree[a] += 1
                if not seen[a]:
                    seen[a] = 1
                    order[n_order] = a
                    n_order += 1
        for i in range(n_nodes):
            start[i + 1] = start[i] + degree[i]
        for i in range(m):  # each node's segments in the order they appear
            for k in range(2):
                a = pairs[i, k]
                incident[start[a] + fill[a]] = i
                fill[a] += 1
        for j in range(n_order):
            at = order[j]
            if degree[at] == 2:
                continue
            for k in range(degree[at]):
                index = incident[start[at] + k]
                if used[index]:
                    continue
                before = count
                count = _walk(pairs, degree, start, incident, used, nodes, count, index, at)
                lengths[chains] = count - before
                chains += 1
        for index in range(m):  # closed loops remain
            if used[index]:
                continue
            before = count
            count = _walk(
                pairs, degree, start, incident, used, nodes, count, index, pairs[index, 0]
            )
            lengths[chains] = count - before
            chains += 1
    return np.asarray(nodes[:count]).copy(), np.asarray(lengths[:chains]).copy()


cdef void _split(
    const double[:, ::1] p,
    unsigned char[::1] keep,
    Py_ssize_t lo,
    Py_ssize_t hi,
    double tol2,
    int depth,
    int passes,
) noexcept nogil:
    """Douglas–Peucker between the kept vertices ``lo`` and ``hi``; ``depth`` passes
    have split this span's ancestors."""
    cdef Py_ssize_t i, worst_i
    cdef double ax, ay, cx, cy, length2, t, ox, oy, dist, worst
    while hi - lo >= 2:
        ax = p[lo, 0]
        ay = p[lo, 1]
        cx = p[hi, 0] - ax
        cy = p[hi, 1] - ay
        length2 = cx * cx + cy * cy
        worst = 0.0
        worst_i = -1
        for i in range(lo + 1, hi):
            if length2 == 0:
                t = 0.0
            else:
                t = ((p[i, 0] - ax) * cx + (p[i, 1] - ay) * cy) / length2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
            ox = p[i, 0] - (ax + t * cx)
            oy = p[i, 1] - (ay + t * cy)
            dist = ox * ox + oy * oy
            if dist > worst:  # the first farthest vertex wins ties
                worst = dist
                worst_i = i
        if not worst > tol2:
            return
        if depth >= passes - 1:  # still unresolved after the last pass: keep it all
            for i in range(lo + 1, hi):
                keep[i] = 1
            return
        keep[worst_i] = 1
        depth += 1
        _split(p, keep, lo, worst_i, tol2, depth, passes)
        lo = worst_i  # the right half continues here


def simplify(
    const double[:, ::1] points,
    double tolerance,
    const long long[::1] starts,
    Py_ssize_t span,
    int passes,
):
    """Indices of the vertices to keep; the same result as ``rendering._simplify_numpy``.

    Splitting one span never changes another, so each span between the initial
    vertices (every ``span``-th, and each line's ends) is simplified on its own.
    """
    cdef Py_ssize_t n = points.shape[0], i, prev = -1
    if n <= 2:
        return np.arange(n)
    cdef unsigned char[::1] keep = np.zeros(n, dtype=np.uint8)
    cdef double tol2 = tolerance * tolerance
    with nogil:
        i = 0
        while i < n:
            keep[i] = 1
            i += span
        keep[n - 1] = 1
        for i in range(starts.shape[0] - 1):
            keep[starts[i]] = 1
            keep[starts[i + 1] - 1] = 1
        for i in range(n):
            if keep[i]:  # later splits only mark vertices before i
                if prev >= 0:
                    _split(points, keep, prev, i, tol2, 0, passes)
                prev = i
    return np.flatnonzero(np.asarray(keep))


def place(const double[:, ::1] points, double kx, double ky, double dx, double dy):
    """``[(x * kx + dx, y * ky + dy)]`` rounded to 0.01 px, as a list of tuples."""
    cdef Py_ssize_t n = points.shape[0], i
    cdef list out = [None] * n
    cdef double x, y
    for i in range(n):
        x = rint((points[i, 0] * kx + dx) * 100.0) / 100.0
        y = rint((points[i, 1] * ky + dy) * 100.0) / 100.0
        out[i] = (x, y)
    return out
