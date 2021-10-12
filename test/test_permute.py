import ffcx.codegeneration.C.cnodes as Language
from ffcx.codegeneration.permute import permute_in_place
import numpy as np
import os
import sys
from cffi import FFI


def test_permute():
    n = 12
    perm = np.zeros(n, dtype=int)
    for i in range(4):
        perm[3 * i] = i
        perm[3 * i + 1] = 4 + i
        perm[3 * i + 2] = 8 + i

    A = Language.ArrayDecl("double", Language.Symbol("A"), sizes=(n,))
    A = Language.FlattenedArray(A, dims=(n,))

    # Build code with cffi
    code_f = permute_in_place(Language, perm, A, "forward")
    code_r = permute_in_place(Language, perm, A, "reverse")
    fn = f"""typedef double ufc_scalar_t;
    void permute_fwd(double *A)
    {{\n{Language.StatementList(code_f)}\n}}
    void permute_rev(double *A)
    {{\n{Language.StatementList(code_r)}\n}}"""
    ffibuilder = FFI()
    ffibuilder.cdef("void permute_fwd(double *); void permute_rev(double *);")
    ffibuilder.set_source("_permute", fn)
    ffibuilder.compile(verbose=True)
    sys.path.append(os.getcwd())

    import _permute
    Avec = np.arange(n, dtype=np.float64)

    _permute.lib.permute_fwd(_permute.ffi.cast("double *", Avec.ctypes.data))
    assert np.allclose(Avec.reshape(4, 3).transpose().flatten(), np.arange(n, dtype=np.float64))

    _permute.lib.permute_rev(_permute.ffi.cast("double *", Avec.ctypes.data))
    assert np.allclose(Avec, np.arange(n, dtype=np.float64))
