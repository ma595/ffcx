# Copyright (C) 2021 Chris Richardson
#
# This file is part of FFCx.(https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np


def permute_in_place(L, perm, A, direction):
    """Permute the flattened array A with the permutation given in perm"""

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

    # Flatten and get offsets
    c_values = np.array([item for sublist in chains for item in sublist])
    offset_values = np.cumsum([0] + [len(ch) for ch in chains])

    # Code generation
    w = A.array
    len_A = np.product([v.value for v in A.dims])
    print(len_A, n)
    assert len_A == n, "Incorrect number of permutation values for array"
    offsets = L.Symbol("perm_offsets")
    c = L.Symbol("perm_values")
    code = [L.ArrayDecl("const int", offsets, len(offset_values), values=offset_values),
            L.ArrayDecl("const int", c, len(c_values), values=c_values)]
    i = L.Symbol("i")
    j = L.Symbol("j")
    wtmp = L.Symbol("wtmp")

    if direction == "forward":
        body = [L.VariableDecl("const ufc_scalar_t", wtmp, w[c[offsets[i]]]),
                L.ForRange(j, 1, offsets[i + 1] - offsets[i],
                body=[L.Assign(w[c[offsets[i] + j - 1]], w[c[offsets[i] + j]])]),
                L.Assign(w[c[offsets[i + 1] - 1]], wtmp)]
        code += [L.ForRange(i, 0, len(offset_values) - 1, body=body)]
    elif direction == "reverse":
        body = [L.VariableDecl("const ufc_scalar_t", wtmp, w[c[offsets[i + 1] - 1]]),
                L.ForRange(j, 1, offsets[i + 1] - offsets[i],
                body=[L.Assign(w[c[offsets[i + 1] - j]], w[c[offsets[i + 1] - j - 1]])]),
                L.Assign(w[c[offsets[i]]], wtmp)]
        code += [L.ForRange(i, 0, len(offset_values) - 1, body=body)]
    else:
        raise RuntimeError(f"Invalid permutation direction: {direction}")

    print(L.StatementList(code))
    return code
