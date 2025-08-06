
# There are no python routines which exactly copy the behaviour of the IDL routines. The closest thing is to use
# scipy.interpolate.RectBivariateSpline and then use these slightly awkward functions to manually save and load the spline parameters. 
# TODO: verify that this produces the same results as the IDL code.

import numpy as np
from scipy.interpolate._fitpack2 import RectBivariateSpline

def save_rbspline(filename, rbspline, kx=3, ky=3):
    np.savez(filename,
             DKnot=rbspline.tck[0],
             TKnot=rbspline.tck[1],
             Coeff=rbspline.tck[2],
             kx=kx,
             ky=ky)

def load_rbspline(filename):
    data = np.load(filename)
    spline = RectBivariateSpline.__new__(RectBivariateSpline)
    spline.tck = (data['DKnot'], data['TKnot'], data['Coeff'])
    spline.kx = int(data['kx'])
    spline.ky = int(data['ky'])
    spline.degrees = (spline.kx, spline.ky)
    return spline
