"""
plot_sets.py is a Python routine that includes some of the settings used
in the generation of plots for the cosmoGW project.

Adapted from the original plot_sets in GW_turbulence
(https://github.com/AlbertoRoper/GW_turbulence),
created January 2021

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/development/src/cosmoGW/plot_sets.py

Author
------
- **Alberto Roper Pol**
  (`alberto.roperpol@unige.ch <mailto:alberto.roperpol@unige.ch>`_)

Dates
-----
- Created: **01/01/2021** (*GW_turbulence*)
- Updated: **13/03/2025**
  (release **cosmoGW 1.0**: https://pypi.org/project/cosmoGW)
"""

import os
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import numpy as np

# Some common settings for the general plots

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=20)
plt.rcParams.update({'axes.labelsize': 'x-large',
                     'axes.titlesize': 'x-large',
                     'xtick.labelsize': 'x-large',
                     'ytick.labelsize': 'x-large',
                     'legend.fontsize': 'x-large'})


def axes_lines(ax=[], both=True):

    """
    Function that includes some default settings for the lines in the loglog
    plots.
    """

    if ax == []:
        ax = plt.gca()
    ax.tick_params(axis='y', direction='in', length=12)
    ax.tick_params(axis='x', direction='in', length=12, pad=10)
    ax.tick_params(which='minor', direction='in', length=6)
    if both:
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')


def ensure_dir(path, quiet=True):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        if not quiet:
            print(f"Directory already exists: {path}")
    os.makedirs(path, exist_ok=True)


def save_fig(dirr='', name='fig', form='png', axes=True, test=True):

    """
    Function that saves the current plot as a PDF using the generic
    settings described in axes_lines.

    Arguments:
        - dirr: directory where to save the figure (default is current dir)
        - name: name of the figure (default is 'fig')
        - form: format of the figure file (default is 'pdf')
        - axes: option to call axes_lines for axes formatting
                (default is True)
    """

    if axes:
        axes_lines()

    plts = os.path.join(dirr, 'plots')
    ensure_dir(plts)

    figg = f"{name}.{form}"
    figg_path = os.path.join(plts, figg)
    save_needed = True
    if os.path.exists(figg_path) and test:
        print(f"Figure already exists: {figg}")
        print("Comparing output to existing figure")
        figg_tmp = figg_path.replace('.png', '_tmp.png')
        plt.savefig(figg_tmp, bbox_inches='tight')
        save_needed = image_difference(figg_tmp, figg_path)
        os.remove(figg_tmp)
    if save_needed:
        print(f"Saving figure in {figg_path}")
        plt.savefig(figg_path, bbox_inches='tight')


def image_difference(img1_path, img2_path, tolerance=0.01):
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    diff = ImageChops.difference(img1, img2)
    np_diff = np.array(diff)
    mean_diff = np.mean(np_diff)
    # print(f"Mean pixel difference: {mean_diff}")
    within_tolerance = mean_diff < tolerance
    if not within_tolerance:
        print(f"Images differ more than allowed tolerance ({tolerance})")
    else:
        print(f"Images are within the allowed tolerance ({tolerance})")
    return not within_tolerance
