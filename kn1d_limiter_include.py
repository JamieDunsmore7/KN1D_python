import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

def kn1d_limiter_include(ax, xlimiter, xsep):
    """
    Draws the limiter‐SOL‐CORE schematic at the bottom of the panel
    and vertical dashed lines at xlimiter and xsep.

    Parameters
    ----------
    ax : matplotlib Axes
    xlimiter : float
        Data‐coordinate of the limiter.
    xsep : float
        Data‐coordinate of the separatrix.
    """

    fig = ax.figure
    # get figure coords for the little box
    # we'll put it at normalized [0.15, 0.05] with height .05
    left, bottom, width, height = 0.15, 0.05, 0.75, 0.05

    # Draw the box polygon in figure coordinates
    rect = plt.Polygon(
        [(left, bottom),
         (left+width, bottom),
         (left+width, bottom+height),
         (left, bottom+height)],
        closed=True,
        fill=False,
        transform=fig.transFigure,
        clip_on=False,
        edgecolor="black"
    )
    fig.add_artist(rect)

    # Labels inside the box
    fig.text(left + 0.25*width, bottom + 0.5*height, "LIMITER",
             ha="center", va="center", transform=fig.transFigure, fontsize=8)
    fig.text(left + 0.50*width, bottom + 0.5*height, "SOL",
             ha="center", va="center", transform=fig.transFigure, fontsize=8)
    fig.text(left + 0.75*width, bottom + 0.5*height, "CORE",
             ha="center", va="center", transform=fig.transFigure, fontsize=8)

    # Hash marks along bottom edge of the box
    nhash = 11
    xs = np.linspace(left, left+width, nhash)
    for xh in xs:
        fig.add_artist(plt.Line2D([xh, xh], [bottom, bottom+0.02],
                                  transform=fig.transFigure, color="black"))

    # Finally, vertical dashed lines at the limiter & separatrix
    ax.axvline(x=xlimiter, linestyle="--", color="black")
    ax.axvline(x=xsep,     linestyle="--", color="black")
