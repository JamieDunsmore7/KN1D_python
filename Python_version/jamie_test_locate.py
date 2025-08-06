# Test the locate() function for ascending and descending tables

import numpy as np
from locate import locate

def main():
    # 1) Ascending table tests
    table1 = np.array([1, 2, 3, 4])
    values = np.array([3.5, 3.0, 0.5, 4.5])
    idx1 = locate(table1, values)
    print("Ascending table:", table1)
    print("Values:          ", values)
    print("locate (PY)   :", idx1)

    # 2) Descending table tests
    table2 = np.array([4, 3, 2, 1])
    idx2 = locate(table2, values)
    print("\nDescending table:", table2)
    print("Values:           ", values)
    print("locate (PY)     :", idx2)

    # 3) Scalar lookup tests
    scalar_val = 2.7
    idx_s1 = locate(table1, scalar_val)
    idx_s2 = locate(table2, scalar_val)
    print(f"\nScalar {scalar_val} in ascending => {idx_s1}  (expect 1)")
    print(f"Scalar {scalar_val} in descending=> {idx_s2}  (expect 2)")

if __name__ == "__main__":
    main()
