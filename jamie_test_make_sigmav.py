import numpy as np
from make_sigma_v import Make_SigmaV

def main():
    # 1) Define the same test grid
    mE, Emin, Emax = 3, 1.0, 100.0
    E_particle = np.logspace(np.log10(Emin), np.log10(Emax), mE)

    nT, Tmin, Tmax = 2, 1.0, 100.0
    T_target = np.logspace(np.log10(Tmin), np.log10(Tmax), nT)

    mu_particle = mu_target = 1.0

    print("E_particle (eV):", E_particle)

    # 2) Loop over T_target using np.sqrt as σ(E)
    for T in T_target:
        print(" T_target (eV):", T)
        sigma_v = Make_SigmaV(
            E_particle, mu_particle,
            T, mu_target,
            sigma_function=np.sqrt
        )
        print(" SigmaV (PY)   [m^2/s]:", sigma_v)

if __name__ == "__main__":
    main()
