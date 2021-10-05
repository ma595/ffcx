# Copyright (C) 2021 Chris Richardson
#
# This file is part of FFCx.(https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np


def permute_in_place(L, perm, A, direction):
    """Permute the flattened array A with the permutation given in perm."""
    # Marker for indices which still need adding to the list
    n = len(perm)
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
            c_values.extend(ch)
            size_values.append(len(ch))

    # Code generation
    w = A.array
    len_A = np.product([v.value for v in A.dims])
    assert len_A == n, "Incorrect number of permutation values for array"
    sizes = L.Symbol("perm_sizes")
    c = L.Symbol("perm_values")
    code = [L.ArrayDecl("const int", sizes, len(size_values), values=size_values),
            L.ArrayDecl("const int", c, len(c_values), values=c_values)]
    i = L.Symbol("i")
    j = L.Symbol("j")
    p = L.Symbol("p")
    wtmp = L.Symbol("wtmp")

    if direction == "forward":
        body = [L.VariableDecl("const ufc_scalar_t", wtmp, w[c[p]]),
                L.ForRange(j, 1, sizes[i],
                body=[L.Assign(w[c[p]], w[c[p + 1]]), L.PreIncrement(p)]),
                L.Assign(w[c[p]], wtmp), L.PreIncrement(p)]
        code += [L.VariableDecl("int", p, 0), L.ForRange(i, 0, len(size_values), body=body)]
    elif direction == "reverse":
        body = [L.VariableDecl("const ufc_scalar_t", wtmp, w[c[p]]),
                L.ForRange(j, 1, sizes[(len(size_values) - 1) - i],
                body=[L.Assign(w[c[p - 1]], w[c[p]]), L.PreDecrement(p)]),
                L.Assign(w[c[p]], wtmp), L.PreDecrement(p)]
        code += [L.VariableDecl("int", p, len(c_values) - 1), L.ForRange(i, 0, len(size_values), body=body)]
    else:
        raise RuntimeError(f"Invalid permutation direction: {direction}")

    print(L.StatementList(code))
    return code
