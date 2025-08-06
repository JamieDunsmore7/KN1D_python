KN1D (Kinetic Neutral 1D Transport Code) is a fast, kinetic code for calculating atomic (H or D) and molecular (H2 or D2) profiles in the edge of fusion plasmas. It was originally written in IDL by Brian LaBombard (MIT PSFC) in 2000/2001. This repository contains a version of the original IDL code along with a full Python translation and some tests. 

The Python version has been validated against the IDL for most of the test cases provided by Brian LaBombard and for an Alcator C-Mod case provided by M.A. Miller, but there are likely still some bugs and certainly still some improvements to be made. In that spirit, please let me know if something doesn't work for you or if you have suggestions!!! You can email me at `jduns@mit.edu`, or you can report an issue on this page.

# Where do I start?
- Read the KN1D manual: https://www-internal.psfc.mit.edu/~labombard/KN1D_Source_Info.html
- Run the code! The best one to start with is probably `cmod_test_kn1d.py`. It reads in some inputs from a `.pkl` file for an example Alcator C-Mod case. This should give an idea of the required inputs and how to run the code. The `test_KN1D.py` script can also be used, but the input set-up is a bit more complex for this one.
- Compare the results to the IDL version. For the C-Mod example, you just need to open IDL and navigate to the `/IDL_version` directory. Then type `.R cmod_test_kn1d.pro` and it should (hopefully!) run.


# General notes
- I've probably left in some `breakpoint` commands in the Python and some `stop` commands in the IDL. To continue running the script if you hit one of these, you need to press `c` in python and `.continue` in IDL

- Some of these IDL scripts may be different slightly from the versions available online here: https://www-internal.psfc.mit.edu/~labombard/KN1D_Source_Info.html. This is because they are taken from Brian LaBombard's directory which is slightly more recent than the online version

- Brian LaBombard's directory has some old/new versions of different scripts suffixed with '_old' and '_new'. In this repo, I haven't used any of the files suffixed with '_old' or '_new'. Any differences should be small/negligable.

- You need an IDL licence to run the IDL version. It's also possible that you may run into some FORTRAN compliation issues due to IDL/FORTRAN version incompatibility. I will probably be of limited help in trying to fix these issues, but if you come across an error like this and know how to fix it then please tell me!

- The KN1D common blocks haven't been implemented in the python version yet. This means that the python/IDL versions can sometimes give different results if there are common block IDL inputs in the memory at the start of the IDL run. This can usually be fixed by starting again in a new IDL terminal

# Potential future improvements

- The common block inputs/outputs have been impelemented in python for `kinetic_h.py` and `kinetic_h2.py`, but not yet for `KN1D.py`. I should sort this to resolve any discrepancies that arise when KN1D common inputs are already stored in memory at the start of an IDL run.

- Some scripts (including `Balmer_alpha.py`) use 1e32 for null values as the IDL does, but it might be better to switch these all to NaNs

- The output plots from `KN1D.py` (and also `kinetic_h.py` and kinetic_h2.py`) are not well formatted

- The D3D test is not working yet!

- There are a few more tests included in the IDL version (e.g one for TCV) that I haven't implemented in python yet

- The `warn` keyword argument in `interp_fvrvxx.py` hasn't been implemented yet

- I'm not 100% sure that that `scipy.interp1d` translation I have for IDL's `interpol` is completely correctly (especially with regards to out of bounds/extrapolation logic). It would be good to double check this.

- Would be nice to turn this into a real package at some point (with an `__init__.py` file etc.) so that people can use it easily without having to mess around with system paths

- I've translated the IDL `#` command in many different ways (`@`, broadcasting arrays, `np.sum`, `np.dot`, `np.matmul` etc.). I'm sure the consistency of the code could be improved by settling on a single translation

- The code can probably be tidied up with some more sub-directories (e.g for all the test scripts)

# Main code inputs/outputs
See KN1D manual for more information on all of these
## Required Inputs

- x (m): x-grid on which the profiles are defined
- x_lim (m): location of x-limiter position
- x_sep (m): location of the separatrix
- n (m^-3): electron density profile (defined on the x-grid)
- Te (eV): electron temperature profile (defined on the x-grid)
- Ti (eV): ion temperature profile (defined on the x-grid)
- vx (ms^-1): velocity profile (defined on the x-grid)
- H2Gauge (mTorr): neutral pressure at the wall
- - D_pipe (m): optional input to account for the effect of atoms/molecules striking the side wall.
- L_C (m): connection length to nearest limiters
- mu: isotope mass (1 for hydrogen and 2 for deuterium)

## Outputs

Essentially the H (or D) and H2 (or D2) distributions and anything that can be derived from these quanities. For example

- n_H profiles
- n_H2 profiles
- Ionisation Rate profiles (Sion)
- Lyman-alpha emissivity profiles
- Balmer-alpha emissivity profiles

And much more...



