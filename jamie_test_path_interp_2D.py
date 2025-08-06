# Test path_interp_2D for some simple cases

import numpy as np
from path_interp_2d import path_interp_2d

# 1) Lookup tables (log‐space)
Te_table = np.log(np.array([5, 20, 100]))                          # 3‐point Te grid (eV)
Ne_table = np.log(np.array([1e14,1e17,1e18,1e19,1e20,1e21,1e22]))  # 7-point Ne grid (m^-3)

fctr_Table = np.zeros((7, 3))
fctr_Table[:, 0] = np.array([2.2, 2.2, 2.1, 1.9, 1.2, 1.1, 1.05]) / 5.3
fctr_Table[:, 1] = np.array([5.1, 5.1, 4.3, 3.1, 1.5, 1.25, 1.25]) / 10.05
fctr_Table[:, 2] = np.array([1.3, 1.3, 1.1, 0.8, 0.38, 0.24, 0.22]) / 2.1

# 2) Pick a few distinct test points
Te = np.array([10.0, 20.0, 50.0, 100.0])     # eV
n  = np.array([1e15, 1e17, 1e19, 1e21])      # m^-3

# 3) Interpolate in log‐space
fctr = path_interp_2d(
    fctr_Table,
    Ne_table,
    Te_table,
    np.log(n),
    np.log(Te)
)

# 4) Print with labels so you can compare directly
print("   n (m^-3):", n)
print("   Te (eV) :", Te)
print("fctr (PY) :", fctr)
