import numpy as np

def locate(table, value):
    """
    Locate the index in a sorted table such that:
    - table[i] <= value < table[i+1] if table is ascending
    - For descending, returns (n-1 - ascending result)
    """
    table = np.asarray(table)
    value = np.asarray(value)
    ascending = table[0] <= table[-1]
    n = table.size

    # Flatten for processing and track original shape
    value_flat = value.ravel()
    jl = np.full(value_flat.shape, -1, dtype=int)
    ju = np.full(value_flat.shape, n, dtype=int)

    while np.max(ju - jl) > 1:
        jm = (jl + ju) // 2
        t_jm = table[jm]
        if ascending:
            mask = value_flat >= t_jm
        else:
            mask = value_flat <= t_jm
        jl = np.where(mask, jm, jl)
        ju = np.where(~mask, jm, ju)

    result = jl.reshape(value.shape)

    return result
