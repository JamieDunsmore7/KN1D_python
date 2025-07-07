import numpy as np

def NHSaha(Density, Te, p):
    """
    Evaluates the Saha equilibrium population density (m^-3)
    for atomic hydrogen level p.

    Parameters
    ----------
    Density : ndarray
        Electron density in m^-3
    Te : ndarray
        Electron temperature in eV
    p : int
        Hydrogen energy level (must be a scalar ≥ 1)

    Returns
    -------
    Result : ndarray
        Saha equilibrium population density (m^-3)
    """
    # Input checks
    Density = np.asarray(Density, dtype=np.float64)
    Te = np.asarray(Te, dtype=np.float64)

    if Density.shape != Te.shape:
        raise ValueError("Number of elements of Density and Te are different!")
    if not np.isscalar(p):
        raise ValueError('"p" must be a scalar')
    if p < 0:
        raise ValueError('"p" must be greater than 0')
    

    # Original IDL code contains a commented derivation of the Saha coefficient, but we just use the result here
    a = 3.310e-28  # [m^3 eV^1.5]

    # Initialize output with null value
    Result = np.full_like(Density, 1.0e32)

    # Compute only for valid entries
    mask = (
        (Density < 1.0e32) & (Density > 0.0) &
        (Te < 1.0e32) & (Te > 0.0)
    )

    if np.any(mask):
        D = Density[mask]
        T = Te[mask]
        Result[mask] = D * (a * D) * p**2 * np.exp(13.6057 / (p**2 * T)) / T**1.5

    return Result