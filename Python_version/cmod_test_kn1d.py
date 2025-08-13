import numpy as np
from kn1d import KN1D
import pickle as pkl
import matplotlib.pyplot as plt


input_file = 'cmod_test_in.pkl'
with open(input_file, 'rb') as f:
    input_data = pkl.load(f)

cmod_results_file = 'cmod_test_out_for_comparison.pkl'
with open(cmod_results_file, 'rb') as f:
    cmod_results = pkl.load(f)


d_pipe = input_data['d_pipe']
lc = input_data['lc']
mu = input_data['mu']
n_e = input_data['n_e'] * 1e20  # Convert from 1e20m^-3 to 1e20 m^-3
p_wall = input_data['p_wall']
t_e = input_data['t_e'] * 1000 # Convert from keV to eV
t_i = input_data['t_i'] * 1000 # Convert from keV to eV
vx = input_data['vx']
x = input_data['x']
x_lim = input_data['x_lim']
x_sep = input_data['x_sep']

File = 'test_kn1d_cmod'

# Use the default collisions
# H_H_EL   = False
# H2_H2_EL = False
# H_P_EL     = True
# H_H2_EL    = True
# H_P_CX     = True
# H2_P_EL    = True
# H2_H_EL    = True
# H2_HP_CX   = True
# Simple_CX  = True

compute_errors = True
Hdebrief = True
H2debrief = True
NewFile = True
ReadInput = False
refine = 0

KN1D(
    x          = x,
    xlimiter   = x_lim,
    xsep       = x_sep,
    GaugeH2    = p_wall,
    mu         = mu,
    Ti         = t_i,
    Te         = t_e,
    n          = n_e,
    vxi        = vx,
    LC         = lc,
    PipeDia    = d_pipe,
    compute_errors=compute_errors,
    H2debrief=H2debrief,
    Hdebrief=Hdebrief,
    NewFile=NewFile,
    ReadInput=ReadInput,
    File=File,
    refine=refine,
    plot=True
)




