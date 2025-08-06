#  Create_VrVxMesh.py
#
# Sets up optimum Vr and Vx velocity space Mesh for Kinetic_Neutrals procedure

import numpy as np

def Create_VrVxMesh(nv, Ti, E0=None, Tmax=0.0):
    '''
    Input:
	nv - integer, number of elements desired in vr mesh
	Ti  - fltarr(*), Ti profile
    
    Keyword:
	E0  - fltarr, energy where a velocity bin is desired (optional)
	Tmax- float, ignore Ti above this value
    
    Output:
	vx  - dblarr(2*nvr+1), x velocity grid
	vr  - dblarr(nvr), r velocity grid
	Tnorm - float, optimum normalization temperature
	ixE0 - intarr, returns array elements of vx corresponding to energy E0
	irE0 - integer, returns array element of vr corresponding to energy E0
    '''
    if E0 is None:
        E0 = np.array([0.0])
    else:
        E0 = np.atleast_1d(E0)
    
    _Ti = list(np.atleast_1d(Ti).astype(float))

    for e in E0:
        if e > 0.0:
            _Ti.append(e)
    _Ti = np.array(_Ti, dtype=float)

    if Tmax > 0.0:
        _Ti = _Ti[_Ti < Tmax]


    maxTi = _Ti.max()
    minTi = _Ti.min()
    Tnorm = _Ti.mean()

    Vmax = 3.5

    if (maxTi - minTi) <= 0.1 * maxTi:
        v = np.arange(nv+1, dtype=float) * (Vmax / nv) # direct translation of the dindgen in IDL
    else:
        G = 2 * nv * np.sqrt(minTi/maxTi) / (1 - np.sqrt(minTi/maxTi))
        b = Vmax / (nv * (nv + G))
        a = G * b
        i = np.arange(nv+1, dtype=float)
        v = a * i + b * i**2

    last_v0 = None
    for e in E0:
        if e > 0.0:
            v0 = np.sqrt(e / Tnorm)
            # find first index where v > v0
            idx = np.searchsorted(v, v0, side='right')
            if idx < len(v):
                v = np.concatenate((v[:idx], [v0], v[idx:]))
                insert_idx = idx
            else:
                v = np.append(v, v0)
                insert_idx = len(v)-1
            last_v0 = v0

    vr = v[1:].copy()
    vx = np.concatenate((-vr[::-1], vr))


    ixE0 = irE0 = None
    if last_v0 is not None:
        ixE0 = np.where(np.isclose(np.abs(vx), last_v0))[0]
        if ixE0.size == 1:
            ixE0 = int(ixE0[0])

        irE0 = np.where(np.isclose(vr, last_v0))[0]
        if irE0.size == 1:
            irE0 = int(irE0[0])

    return vx, vr, Tnorm, ixE0, irE0