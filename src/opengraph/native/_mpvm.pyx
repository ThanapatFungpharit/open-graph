# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Register machine evaluating :class:`opengraph.engine.precise.Program` in MPFR.

One sample at a time: the inputs are ``origin + offset`` (exact at the program's
precision), the code runs over a small set of MPFR registers, and the result minus the
output origin is rounded to a double. The loop holds no Python objects, so it runs
without the GIL and geometry workers evaluate tiles in parallel. Opcodes must match
:class:`opengraph.engine.precise.Op`.
"""

from libc.stdlib cimport malloc, free

from gmpy2 cimport import_gmpy2, mpfr, MPFR, mpfr_t, mpfr_ptr, mpfr_prec_t, mpfr_rnd_t, MPFR_RNDN

cdef extern from "mpfr.h" nogil:
    void mpfr_init2(mpfr_ptr x, mpfr_prec_t prec)
    void mpfr_clear(mpfr_ptr x)
    int mpfr_set(mpfr_ptr rop, mpfr_ptr op, mpfr_rnd_t rnd)
    int mpfr_set_d(mpfr_ptr rop, double op, mpfr_rnd_t rnd)
    int mpfr_set_si(mpfr_ptr rop, long op, mpfr_rnd_t rnd)
    void mpfr_set_nan(mpfr_ptr x)
    double mpfr_get_d(mpfr_ptr op, mpfr_rnd_t rnd)
    int mpfr_add(mpfr_ptr rop, mpfr_ptr a, mpfr_ptr b, mpfr_rnd_t rnd)
    int mpfr_add_d(mpfr_ptr rop, mpfr_ptr a, double b, mpfr_rnd_t rnd)
    int mpfr_sub(mpfr_ptr rop, mpfr_ptr a, mpfr_ptr b, mpfr_rnd_t rnd)
    int mpfr_mul(mpfr_ptr rop, mpfr_ptr a, mpfr_ptr b, mpfr_rnd_t rnd)
    int mpfr_div(mpfr_ptr rop, mpfr_ptr a, mpfr_ptr b, mpfr_rnd_t rnd)
    int mpfr_neg(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_abs(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_pow(mpfr_ptr rop, mpfr_ptr a, mpfr_ptr b, mpfr_rnd_t rnd)
    int mpfr_sqrt(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_cbrt(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_sin(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_cos(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_tan(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_sec(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_csc(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_cot(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_asin(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_acos(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_atan(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_atan2(mpfr_ptr rop, mpfr_ptr y, mpfr_ptr x, mpfr_rnd_t rnd)
    int mpfr_sinh(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_cosh(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_tanh(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_asinh(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_acosh(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_atanh(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_exp(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_log(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_log10(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_gamma(mpfr_ptr rop, mpfr_ptr a, mpfr_rnd_t rnd)
    int mpfr_floor(mpfr_ptr rop, mpfr_ptr a)
    int mpfr_ceil(mpfr_ptr rop, mpfr_ptr a)
    int mpfr_fmod(mpfr_ptr rop, mpfr_ptr a, mpfr_ptr b, mpfr_rnd_t rnd)
    int mpfr_nan_p(mpfr_ptr x)
    int mpfr_number_p(mpfr_ptr x)
    int mpfr_integer_p(mpfr_ptr x)
    int mpfr_zero_p(mpfr_ptr x)
    int mpfr_sgn(mpfr_ptr x)
    int mpfr_cmp(mpfr_ptr a, mpfr_ptr b)
    int mpfr_lessequal_p(mpfr_ptr a, mpfr_ptr b)
    int mpfr_less_p(mpfr_ptr a, mpfr_ptr b)
    int mpfr_greaterequal_p(mpfr_ptr a, mpfr_ptr b)

import_gmpy2()

# Opcodes (opengraph.engine.precise.Op).
cdef enum:
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    NEG = 5
    POW = 6
    ODDPOW = 7
    SIN = 10
    COS = 11
    TAN = 12
    SEC = 13
    CSC = 14
    COT = 15
    ASIN = 16
    ACOS = 17
    ATAN = 18
    ATAN2 = 19
    SINH = 20
    COSH = 21
    TANH = 22
    ASINH = 23
    ACOSH = 24
    ATANH = 25
    SQRT = 26
    CBRT = 27
    ABS = 28
    EXP = 29
    LN = 30
    LOG10 = 31
    GAMMA = 32
    FLOOR = 33
    CEIL = 34
    SIGN = 35
    MOD = 36
    MIN = 37
    MAX = 38
    GUARD = 40
    BOUND = 41


cdef int execute(const int[:, ::1] code, mpfr_ptr regs, mpfr_ptr tmp) noexcept nogil:
    """Run the program once over ``regs`` (inputs already set)."""
    cdef Py_ssize_t k
    cdef int op, sign
    cdef mpfr_ptr dst, a, b, c, d
    cdef bint inside
    for k in range(code.shape[0]):
        op = code[k, 0]
        dst = &regs[code[k, 1]]
        a = &regs[code[k, 2]]
        b = &regs[code[k, 3]]
        if op == ADD:
            mpfr_add(dst, a, b, MPFR_RNDN)
        elif op == SUB:
            mpfr_sub(dst, a, b, MPFR_RNDN)
        elif op == MUL:
            mpfr_mul(dst, a, b, MPFR_RNDN)
        elif op == DIV:
            mpfr_div(dst, a, b, MPFR_RNDN)
        elif op == NEG:
            mpfr_neg(dst, a, MPFR_RNDN)
        elif op == POW:
            mpfr_pow(dst, a, b, MPFR_RNDN)
        elif op == ODDPOW:
            # |a|^b, times sign(a) when the root's numerator is odd (real odd roots).
            sign = mpfr_sgn(a)
            mpfr_abs(tmp, a, MPFR_RNDN)
            mpfr_pow(dst, tmp, b, MPFR_RNDN)
            if mpfr_nan_p(a):
                mpfr_set_nan(dst)
            elif code[k, 6] and sign < 0:
                mpfr_neg(dst, dst, MPFR_RNDN)
            elif code[k, 6] and sign == 0 and mpfr_number_p(dst):
                mpfr_set_si(dst, 0, MPFR_RNDN)  # |0|^b * sign(0)
            elif code[k, 6] and sign == 0:
                mpfr_set_nan(dst)  # inf * 0
        elif op == SIN:
            mpfr_sin(dst, a, MPFR_RNDN)
        elif op == COS:
            mpfr_cos(dst, a, MPFR_RNDN)
        elif op == TAN:
            mpfr_tan(dst, a, MPFR_RNDN)
        elif op == SEC:
            mpfr_sec(dst, a, MPFR_RNDN)
        elif op == CSC:
            mpfr_csc(dst, a, MPFR_RNDN)
        elif op == COT:
            mpfr_cot(dst, a, MPFR_RNDN)
        elif op == ASIN:
            mpfr_asin(dst, a, MPFR_RNDN)
        elif op == ACOS:
            mpfr_acos(dst, a, MPFR_RNDN)
        elif op == ATAN:
            mpfr_atan(dst, a, MPFR_RNDN)
        elif op == ATAN2:
            mpfr_atan2(dst, a, b, MPFR_RNDN)
        elif op == SINH:
            mpfr_sinh(dst, a, MPFR_RNDN)
        elif op == COSH:
            mpfr_cosh(dst, a, MPFR_RNDN)
        elif op == TANH:
            mpfr_tanh(dst, a, MPFR_RNDN)
        elif op == ASINH:
            mpfr_asinh(dst, a, MPFR_RNDN)
        elif op == ACOSH:
            mpfr_acosh(dst, a, MPFR_RNDN)
        elif op == ATANH:
            mpfr_atanh(dst, a, MPFR_RNDN)
        elif op == SQRT:
            mpfr_sqrt(dst, a, MPFR_RNDN)
        elif op == CBRT:
            mpfr_cbrt(dst, a, MPFR_RNDN)
        elif op == ABS:
            mpfr_abs(dst, a, MPFR_RNDN)
        elif op == EXP:
            mpfr_exp(dst, a, MPFR_RNDN)
        elif op == LN:
            mpfr_log(dst, a, MPFR_RNDN)
        elif op == LOG10:
            mpfr_log10(dst, a, MPFR_RNDN)
        elif op == GAMMA:
            # Poles at 0, -1, -2, ... are undefined (NaN), as in the float evaluator.
            if mpfr_integer_p(a) and mpfr_sgn(a) <= 0:
                mpfr_set_nan(dst)
            else:
                mpfr_gamma(dst, a, MPFR_RNDN)
        elif op == FLOOR:
            mpfr_floor(dst, a)
        elif op == CEIL:
            mpfr_ceil(dst, a)
        elif op == SIGN:
            if mpfr_nan_p(a):
                mpfr_set_nan(dst)
            else:
                mpfr_set_si(dst, mpfr_sgn(a), MPFR_RNDN)
        elif op == MOD:
            # The remainder takes the sign of the divisor (NumPy's mod).
            mpfr_fmod(tmp, a, b, MPFR_RNDN)
            if not mpfr_zero_p(tmp) and mpfr_number_p(tmp) and (mpfr_sgn(tmp) < 0) != (mpfr_sgn(b) < 0):
                mpfr_add(dst, tmp, b, MPFR_RNDN)
            else:
                mpfr_set(dst, tmp, MPFR_RNDN)
        elif op == MIN or op == MAX:
            if mpfr_nan_p(a) or mpfr_nan_p(b):
                mpfr_set_nan(dst)
            elif (mpfr_cmp(a, b) <= 0) == (op == MIN):
                mpfr_set(dst, a, MPFR_RNDN)
            else:
                mpfr_set(dst, b, MPFR_RNDN)
        elif op == GUARD:
            if mpfr_number_p(b):
                mpfr_set(dst, a, MPFR_RNDN)
            else:
                mpfr_set_nan(dst)
        elif op == BOUND:
            c = &regs[code[k, 4]]
            d = &regs[code[k, 5]]
            if code[k, 6] & 1:
                inside = mpfr_greaterequal_p(b, c)
            else:
                inside = mpfr_less_p(c, b)
            if inside:
                if code[k, 6] & 2:
                    inside = mpfr_lessequal_p(b, d)
                else:
                    inside = mpfr_less_p(b, d)
            if inside:
                mpfr_set(dst, a, MPFR_RNDN)
            else:
                mpfr_set_nan(dst)
        else:
            return -1
    return 0


def run(
    const int[:, ::1] code,
    list constants,
    list origins,
    const double[:, ::1] offsets,
    int registers,
    int result,
    mpfr out_origin,
    long bits,
    double[::1] out,
    Py_ssize_t start,
    Py_ssize_t stop,
):
    """Evaluate samples ``start:stop``: ``out[i] = f(origins + offsets[:, i]) - out_origin``."""
    cdef Py_ssize_t n_const = len(constants), n_input = len(origins)
    cdef Py_ssize_t i, j, total = registers + n_input + 1
    cdef mpfr_ptr regs = <mpfr_ptr>malloc(total * sizeof(mpfr_t))
    if regs == NULL:
        raise MemoryError()
    cdef mpfr_ptr inputs_origin = &regs[registers]
    cdef mpfr_ptr tmp = &regs[registers + n_input]
    cdef mpfr value
    cdef mpfr_ptr out_ref = MPFR(out_origin)
    cdef int status = 0
    for j in range(total):
        mpfr_init2(&regs[j], bits)
    try:
        for j in range(n_const):
            value = constants[j]
            mpfr_set(&regs[j], MPFR(value), MPFR_RNDN)
        for j in range(n_input):
            value = origins[j]
            mpfr_set(&inputs_origin[j], MPFR(value), MPFR_RNDN)
        with nogil:
            for i in range(start, stop):
                for j in range(n_input):
                    mpfr_add_d(&regs[n_const + j], &inputs_origin[j], offsets[j, i], MPFR_RNDN)
                status = execute(code, regs, tmp)
                if status:
                    break
                mpfr_sub(tmp, &regs[result], out_ref, MPFR_RNDN)
                out[i] = mpfr_get_d(tmp, MPFR_RNDN)
    finally:
        for j in range(total):
            mpfr_clear(&regs[j])
        free(regs)
    if status:
        raise ValueError("unknown opcode")
