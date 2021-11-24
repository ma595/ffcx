# Copyright (C) 2021 Chris Richardson
#
# This file is part of FFCx.(https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np


def permute_in_place(L, scalar_type, perm, A, direction):
    """Permute the flattened array A with the permutation given in perm."""
    # Marker for indices which still need adding to the list
    n = len(perm)

    if direction not in ("forward", "reverse"):
        raise RuntimeError(f"Invalid permutation direction: {direction}")

    mark = np.ones(n, dtype=bool)
    # List of lists, representing the permuting groups
    chains = [[]]
    idx = 0
    mark_count = 0
    while(mark_count < n):
        # Mark off this index as used, and attach to current chain
        mark[idx] = False
        mark_count += 1
        chains[-1].append(idx)
        idx = perm[idx]

        # If new index is already used, we need to start afresh
        if not mark[idx] and mark_count < n:
            # Find first unused entry in markers
            idx = mark.nonzero()[0][0]
            chains.append([])

    # Flatten and get sizes
    c_values = []
    size_values = []
    for ch in chains:
        if len(ch) > 1:
            if direction == "forward":
                c_values.extend(ch)
            else:
                c_values.extend(reversed(ch))
            size_values.append(len(ch))

    # Code generation
    w = A.array
    rank = len(A.dims)
    ncols = A.dims[-1].value
    nrows = 1
    if rank == 2:
        nrows = A.dims[0].value

    assert ncols == n, f"Incorrect number of permutation values for array: {ncols}, {n}"
    sizes = L.Symbol("perm_sizes")
    c = L.Symbol("perm_values")
    code = [L.ArrayDecl("const int", sizes, len(size_values), values=size_values),
            L.ArrayDecl("const int", c, len(c_values), values=c_values)]
    i = L.Symbol("i")
    j = L.Symbol("j")
    k = L.Symbol("k")
    p = L.Symbol("p")
    wtmp = L.Symbol("w_tmp")
    wrow = L.Symbol("w_row")

    body = [L.VariableDecl(f"const {scalar_type}", wtmp, wrow[c[p]]),
            L.ForRange(j, 1, sizes[i],
                       body=[L.Assign(wrow[c[p]], wrow[c[p + 1]]),
                             L.PreIncrement(p)]),
            L.Assign(wrow[c[p]], wtmp), L.PreIncrement(p)]
    code += [L.ForRange(k, 0, nrows, body=[L.VariableDecl(f"{scalar_type}*", wrow, w + k * 30),
                                           L.VariableDecl("int", p, 0), L.ForRange(i, 0, len(size_values), body=body)])]

    return code
