import ffcx.codegeneration.C.cnodes as Language
from ffcx.codegeneration.permute import permute_in_place
import numpy as np
import os
from cffi import FFI


def test_permute():
    # Example permutation, reverses a range
    n = 21
    r = np.arange(n - 1, -1, -1, dtype=int)
    A = Language.ArrayDecl("double", Language.Symbol("A"), sizes=(n,))
    A = Language.FlattenedArray(A, dims=(n,))

    # Build code with cffi
    code_f = permute_in_place(Language, r, A, "forward")
    code_r = permute_in_place(Language, r, A, "reverse")
    fn = f"""typedef double ufc_scalar_t;
    void permute_fwd(double *A)
    {{\n{Language.StatementList(code_f)}\n}}
    void permute_rev(double *A)
    {{\n{Language.StatementList(code_r)}\n}}"""
    ffibuilder = FFI()
    ffibuilder.cdef("void permute_fwd(double *); void permute_rev(double *);")
    ffibuilder.set_source("_permute", fn)
    ffibuilder.compile(verbose=True)
    print(os.listdir())

    import _permute
    Avec = np.arange(n, dtype=np.float64)

    _permute.lib.permute_fwd(_permute.ffi.cast("double *", Avec.ctypes.data))
    assert np.allclose(np.flip(Avec), np.arange(n, dtype=np.float64))

    _permute.lib.permute_rev(_permute.ffi.cast("double *", Avec.ctypes.data))
    assert np.allclose(Avec, np.arange(n, dtype=np.float64))
