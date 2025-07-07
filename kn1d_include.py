import numpy as np

def kn1d_include(ax, x, xH, xH2, GaugeH2, File=None, HH_label="H₂"):
    """
    Compute mid‐indices and annotate gauge pressure (and optional filename).

    Returns
    -------
    mid, midH, midH2 : lists of int
        Indices into x, xH, xH2 nearest to each normalized xloc.
    """
    fig = ax.figure
    x_min, x_max = ax.get_xlim()

    # normalized positions in [0,1]
    xloc = np.array([.15, .3, .45, .6, .75, .9])

    # map each normalized xloc → data‐coordinate
    x_data = x_min + xloc * (x_max - x_min)

    # find nearest index in each array
    mid  = [int(np.argmin(np.abs(x  - xd))) for xd in x_data]
    midH = [int(np.argmin(np.abs(xH - xd))) for xd in x_data]
    midH2= [int(np.argmin(np.abs(xH2- xd))) for xd in x_data]

    # annotate gauge pressure and (optionally) filename in figure coords
    # put it at (0.15, 0.9) and (0.15, 0.86) in figure fractions
    fig.text(0.15, 0.90,
             rf"{HH_label} Gauge Pressure: {GaugeH2:.5f} mtorr",
             transform=fig.transFigure,
             fontsize=8, color="C2")
    if File:
        fig.text(0.15, 0.86,
                 f"FILE: {File}",
                 transform=fig.transFigure,
                 fontsize=8, color="C4")

    return mid, midH, midH2
