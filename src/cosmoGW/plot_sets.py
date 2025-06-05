"""
plot_sets.py is a Python routine that includes some of the settings used
in the generation of plots for the cosmoGW project.

Adapted from the original plot_sets in GW_turbulence
(https://github.com/AlbertoRoper/GW_turbulence),
created January 2021

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/development/src/cosmoGW/plot_sets.py

Author: Alberto Roper Pol
Created: 01/01/2021 (GW_turbulence)
Updated: 13/03/2025 (release cosmoGW 1.0: https://pypi.org/project/cosmoGW)
"""

import matplotlib.pyplot as plt

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

    if ax == []: ax = plt.gca()
    ax.tick_params(axis='y', direction='in', length=12)
    ax.tick_params(axis='x', direction='in', length=12, pad=10)
    ax.tick_params(which='minor', direction='in', length=6)
    if both:
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')

def save_fig(dirr='', name='fig', form='pdf', axes=True):

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

    import os

    if axes: axes_lines()
    plts = dirr + 'plots/'
    try:
        os.mkdir(plts)
    except:
        print('Not possible to create directory', plts)

    figg = name + '.' + form
    print('Saving figure in', plts, figg)
    plt.savefig(plts + figg, bbox_inches='tight')
