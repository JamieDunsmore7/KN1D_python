import numpy as np
import matplotlib.pyplot as plt
from sigma_cx_hh import Sigma_CX_HH

def main_test_sigma_cx_hh():
    """
    Generates and plots the charge-exchange cross-section for H2+ + H -> H2 + H+.
    """
    
    # --- Data Generation ---
    # Equivalent of: E=10.0^(-1+(5+alog10(2))*findgen(101)/100)
    E = 10.0**(-1 + (5 + np.log10(2)) * np.arange(101) / 100)

    # --- Calculate Cross Section ---
    # Call the (assumed) Sigma_CX_HH function
    # The 1e4 factor converts from m^2 to cm^2, as in the IDL script
    sig = Sigma_CX_HH(E) * 1e4 
    
    # --- Plot Setup ---
    # Equivalent of the IDL 'plot, /nodata, ...' command
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set scales and limits
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 2e4)
    ax.set_ylim(1.0e-16, 1.0e-14) # Note: IDL's 1.e-14 was likely a typo and should be 1.0e-14
    
    # Set titles and labels
    # H!D2!N!U+!N -> H₂⁺  (using LaTeX for proper subscript/superscript)
    ax.set_title(r'H$_2^+$ + H $\rightarrow$ H$_2$ + H$^+$') 
    ax.set_xlabel('E (eV)')
    ax.set_ylabel(r'Sigma (cm$^2$)') # Using LaTeX for superscript 2
    
    # Set tick parameters (Matplotlib defaults are usually sufficient)
    ax.tick_params(axis='both', which='major', length=6)
    ax.tick_params(axis='both', which='minor', length=3)
    
    # --- Plotting ---
    # Equivalent of IDL 'oplot,E,sig*1e4,color=4,thick=3.'
    ax.plot(E, sig, color='blue', linewidth=3)
    
    # --- Text annotations (equivalent to xyouts) ---
    # The 'transform=ax.transAxes' makes coordinates relative to the axes (0,0 to 1,1)
    ax.text(0.15, 0.20, 'Output from python function which evaluates the CX Cross-Section using:',
            transform=ax.transAxes, fontsize=9)
    ax.text(0.15, 0.17, 'Polynomial fit from Janev et al., "Elementary Processes in Hydrogen-Helium Plasmas", p 253.',
            transform=ax.transAxes, fontsize=9, color='blue')

    # Add a grid for better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Display the plot
    plt.show()


if __name__ == '__main__':
    main_test_sigma_cx_hh()