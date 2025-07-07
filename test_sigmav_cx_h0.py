import numpy as np
import matplotlib.pyplot as plt
from sigmav_cx_h0 import SigmaV_CX_H0 


def main_test_sigmav_cx_h0():
    """
    Generates and plots reaction rates <sigma v> for p + H(1s) -> H(1s) + p
    at various E0 values.
    """
    
    # Define Ti (Ion Temperature in eV) range
    # Equivalent to: Ti=10.0^(-1+(5+alog10(2))*findgen(101)/100)
    Ti = 10.0**(-1 + (5 + np.log10(2)) * np.arange(101) / 100)

    # --- Plot Setup ---
    # Equivalent of the IDL 'plot, /nodata, ...' command for the first plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Set scales and limits
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 2e4)
    ax.set_ylim(1.0e-9, 2.0e-6)
    
    # Set titles and labels
    ax.set_title(r'p + H(1s) $\rightarrow$ H(1s) + p')
    ax.set_xlabel('Ti (eV)')
    ax.set_ylabel(r'<$\sigma$v> (cm$^3$/s)') 
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', length=6)
    ax.tick_params(axis='both', which='minor', length=3)
    
    # --- Loop and Plot for different E0 values ---
    # IDL color indices: 0=black, 1=white, 2=red, 3=green, 4=blue, 5=cyan, 6=magenta, 7=yellow
    mpl_colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'orange'] 

    for i in range(6): # i from 0 to 5
        e0_scalar = 10.0**(i - 1)
        
        # Create an array `E` that has the same shape as `Ti` and is filled with `e0_scalar`.
        # This matches the expectation of your SigmaV_CX_H0 function.
        E_array = np.full_like(Ti, e0_scalar) 
        
        # Call the actual SigmaV_CX_H0 function
        sigv = SigmaV_CX_H0(Ti, E_array) # Pass the now-matching E_array
        
        # Oplot for subsequent curves
        # The 1e6 factor converts from m^3/s to cm^3/s (1 m^3 = 1e6 cm^3)
        ax.plot(Ti, sigv * 1e6, color=mpl_colors[i], linewidth=3)
        
        # Add E0 annotation for each curve
        ax.text(0.15, 0.7 + i * 0.04, f'E0 = {e0_scalar:.1e} eV', 
                transform=ax.transAxes, fontsize=9, color=mpl_colors[i])

    # --- Global Text annotations ---
    ax.text(0.15, 0.18, 'Output from python function which evaluates the reaction rates using:',
            transform=ax.transAxes, fontsize=9)
    ax.text(0.15, 0.15, 'Data from Janev et al., "Elementary Processes in Hydrogen-Helium Plasmas", p 272.',
            transform=ax.transAxes, fontsize=9)

    # Add a grid for better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Display the plot
    plt.show()

# This ensures that main_test_sigmav_cx_h0() is called only when the script is executed directly
if __name__ == '__main__':
    main_test_sigmav_cx_h0()