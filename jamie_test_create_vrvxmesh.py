# Test Create_VrVxMesh in three scenarios mirroring the IDL tests.

import numpy as np
from create_vrvxmesh import Create_VrVxMesh

def print_array(name, arr):
    print(f"{name} =", np.array2string(arr, precision=4, separator=', '))


nv = 5
Ti1 = np.array([10.0, 20.0, 30.0])

# 1) baseline: no E0, no Tmax
vx1, vr1, Tnorm1, ix1, ir1 = Create_VrVxMesh(nv, Ti1)
print("\n--- Test 1: no E0, no Tmax ---")
print(f"nv = {nv}, Ti = {Ti1}")
print_array("vx1", vx1)
print_array("vr1", vr1)
print(f"Tnorm1 = {Tnorm1}")

# 2) with E0 = 5 eV
E0 = np.array([5.0])
vx2, vr2, Tnorm2, ix2, ir2 = Create_VrVxMesh(nv, Ti1, E0=E0)
print("\n--- Test 2: E0 =", E0, "---")
print_array("vx2", vx2)
print_array("vr2", vr2)
print(f"ixE0 = {ix2}, irE0 = {ir2}")

# 3) with Tmax = 25 eV
Tmax = 25.0
vx3, vr3, Tnorm3, ix3, ir3 = Create_VrVxMesh(nv, Ti1, Tmax=Tmax)
used_Ti = Ti1[Ti1 < Tmax]
print("\n--- Test 3: Tmax =", Tmax, "---")
print(f"Ti used = {used_Ti}")
print_array("vx3", vx3)
print_array("vr3", vr3)
print(f"Tnorm3 = {Tnorm3}")


