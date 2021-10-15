import ffcx.codegeneration.C.cnodes as Language
from ffcx.codegeneration.permute import permute_in_place
import numpy as np
import os
import sys
from cffi import FFI
import pytest
import importlib

@pytest.mark.parametrize("scalar_type, np_type",
                         [("double _Complex", "complex128"), ("double", "float64")])
def test_permute(scalar_type, np_type):

    n = 12
    perm = np.zeros(n, dtype=int)
    for i in range(4):
        perm[3 * i] = i
        perm[3 * i + 1] = 4 + i
        perm[3 * i + 2] = 8 + i

    A = Language.ArrayDecl(f"{scalar_type}", Language.Symbol("A"), sizes=(n,))
    A = Language.FlattenedArray(A, dims=(n,))

    # Build code with cffi
    code_f = permute_in_place(Language, scalar_type, perm, A, "forward")
    code_r = permute_in_place(Language, scalar_type, perm, A, "reverse")
    fn = f"""
    void permute_fwd({scalar_type} *A)
    {{\n{Language.StatementList(code_f)}\n}}
    void permute_rev({scalar_type} *A)
    {{\n{Language.StatementList(code_r)}\n}}"""
    ffibuilder = FFI()
    ffibuilder.cdef(f"void permute_fwd({scalar_type} *); void permute_rev({scalar_type} *);")
    ffibuilder.set_source(f"_permute_{np_type}", fn)
    ffibuilder.compile(verbose=True)
    sys.path.append(os.getcwd())

    importlib.import_module(f'_permute_{np_type}')
    _permute = sys.modules[f'_permute_{np_type}']

    Avec = np.arange(n, dtype=np_type)

    _permute.lib.permute_fwd(_permute.ffi.cast(f"{scalar_type} *", Avec.ctypes.data))
    assert np.allclose(Avec.reshape(4, 3).transpose().flatten(), np.arange(n, dtype=np_type))

    _permute.lib.permute_rev(_permute.ffi.cast(f"{scalar_type} *", Avec.ctypes.data))
    assert np.allclose(Avec, np.arange(n, dtype=np_type))
