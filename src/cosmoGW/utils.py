'''
Author
------
- **Alberto Roper Pol**
  (`alberto.roperpol@unige.ch <mailto:alberto.roperpol@unige.ch>`_)

Dates
-----
- Created: **21/08/2025**
  (release **cosmoGW 1.0**: https://pypi.org/project/cosmoGW)

References
----------
Used in cosmoGW scientific routines for output reshaping
and general utilities.
'''

import numpy as np
import astropy.units as u
import pandas as pd
from cosmoGW import COSMOGW_HOME

# Reference values and constants
Tref = 100 * u.GeV  # EWPT
gref = 100          # EWPT
Neff_ref = 3        # reference Neff = 3

# Reference values for cosmology, from Planck:2018vyg
T0K = 2.72548 * u.K
H0_ref = 100 * u.km / u.s / u.Mpc
H0_ref = H0_ref.to(u.Hz)
OmL0_ref = 0.6841
OmM0_ref = 1. - OmL0_ref
h0_ref = 0.6732

# Reference slopes for broken power-law spectra (GW_templates)
a_ref = 5            # Batchelor spectrum k^5
b_ref = 2/3          # Kolmogorov spectrum k^(-2/3)
alp_ref = 2          # reference smoothness of broken power-law transition
alp_turb = 6 / 17    # von Karman smoothness parameter
bPi_vort = 3         # spectral slope found in the anisotropic stresses
alpPi = 2.15         # smoothness parameter for the fit of the anisotropic
fPi = 2.2            # break frequency of the anisotropic stresses
tdecay_ref = 'eddy'  # decaying time used in the constant-in-time model

# Reference values for sound waves templates (GW_templates)
OmGW_sw_ref = 1e-2   # normalized amplitude, based on Hindmarsh:2017gnf
a_sw_ref = 3         # low frequency slope f^3
b_sw_ref = 1         # intermediate frequency slope f
c_sw_ref = 3         # high frequency slope f^(-3)

# first and second peak, and smoothness parameters (GW_templates)
zpeak1_LISA_old = 10.
alp1_sw_ref = 1.5     # used in RoperPol:2023bqa
alp2_sw_ref = 0.5     # used in RoperPol:2023bqa
alp1_ssm = 4.0        # used in Hindmarsh:2019phv
alp2_ssm = 2.0        # used in Hindmarsh:2019phv
alp1_HL = 3.6         # found in Caprini:2024gyk
alp2_HL = 2.4         # found in Caprini:2024gyk
peak1_HL = 0.4        # found in Caprini:2024gyk
peak2_HL_weak = 0.5   # found in Caprini:2024gyk
peak2_HL_interm = 1.  # found in Caprini:2024gyk
peak2_HL_str = 0.5    # found in Caprini:2024gyk
alp1_LISA = 2.0       # used in Caprini:2024hue
alp2_LISA = 4.0       # used in Caprini:2024hue
peak1_LISA = 0.2      # used in Caprini:2024hue
peak2_LISA = 0.5      # used in Caprini:2024hue

# Reference values for models (GW_models)
Oms_ref = 0.1         # Source amplitude (fraction to radiation energy)
lf_ref = 0.01         # Source length scale (normalized by Hubble radius)
beta_ref = 100        # Nucleation rate beta/H_ast
N_turb = 2            # Source duration/eddy turnover time ratio
Nk_ref = 1000         # Wave number discretization
Nkconv_ref = 1000     # Wave number discretization for convolution
Np_ref = 3000         # Wave number discretization for convolution
NTT_ref = 5000        # Lifetimes discretization
dt0_ref = 11          # Numerical parameter for fit (Caprini:2024gyk)
bs_HL_eff = 20        # box-size L/vw used for UV in Caprini:2024yk
bs_k1HL = 40          # box-size L/vw used for IR in Caprini:2024gyk
tini_ref = 1.0        # Initial time of GW production (normalized)
tfin_ref = 1e4        # Final time of GW production in cit model


# Reference values for hydro_bubbles
cs2_ref = 1.0 / 3  # speed of sound squared
Nxi_ref = 10000    # reference discretization in xi
Nxi2_ref = 10      # reference discretization in xi out of the profiles
Nvws_ref = 20      # reference discretization in vwall
tol_ref = 1e-5     # reference tolerance on shooting algorithm
it_ref = 30        # reference number of iterations

# Reference values for a deflagration
vw_def = 0.5
alpha_def = 0.263

# Reference values for a hybrid
vw_hyb = 0.7
alpha_hyb = 0.052

# Reference values for a detonation
vw_det = 0.77
alpha_det = 0.091

# Reference set of colors
cols_ref = [
    'black', 'darkblue', 'blue', 'darkgreen', 'green',
    'purple', 'darkred', 'red', 'darkorange', 'orange', 'violet'
]

# Reference values for LISA and Taiji interferometers
L_LISA = 2.5e6 * u.km
P_LISA = 15
A_LISA = 3
L_Taiji = 3e6 * u.km
P_Taiji = 8
A_Taiji = 3
SNR_PLS = 10
T_PLS = 4
v_dipole = 1.23e-3
dir_sens = 'resources/detectors_sensitivity/'

# Reference frequency and beta arrays (interferometry)
f_ref = np.logspace(-4, 0, 5000) * u.Hz
beta_ref = np.linspace(-20, 20, 3000)


def safe_trapezoid(y, x, axis=-1):
    """
    Safely compute the trapezoidal integral of y with respect to x.

    Parameters
    ----------
    y : array_like
        Values to integrate.
    x : array_like
        Integration variable.
    axis : int, optional
        Axis along which to integrate (default: -1).

    Returns
    -------
    float or ndarray
        The integral result.
    """
    try:
        return np.trapezoid(y, x, axis=axis)
    except AttributeError:
        return np.trapz(y, x, axis=axis)


def reshape_output(arr, mult_a=True, mult_b=True, mult_c=True, mult_d=True,
                   mult_e=True, skip=0):
    """
    Reshape a 2D output array to 1D or scalar depending on input flags.

    Parameters
    ----------
    Omegas : ndarray
        The array to reshape (2D).
    mult_alp : bool
        True if the first dimension is multi-valued (array), False if scalar.
    mult_vw : bool
        True if the second dimension is multi-valued (array), False if scalar.

    Returns
    -------
    ndarray or scalar
        The reshaped output array or scalar.
    """
    axes = []
    # axis 1: vws, axis 2: alphas, axis 3: betas
    if arr.ndim - skip > 0 and not mult_a:
        axes.append(skip+0)
    if arr.ndim - skip > 1 and not mult_b:
        axes.append(skip+1)
    if arr.ndim - skip > 2 and not mult_c:
        axes.append(skip+2)
    if arr.ndim - skip > 3 and not mult_d:
        axes.append(skip+3)
    if arr.ndim - skip > 4 and not mult_e:
        axes.append(skip+4)
    return np.squeeze(arr, axis=tuple(axes))


def read_csv(file, dir0=dir_sens, dir_HOME=None, a='f', b='Omega'):
    """
    Read a CSV file and return a pandas DataFrame.

    Parameters
    ----------
    file : str
        Filename of the CSV file.
    dir0 : str, optional
        Default directory for sensitivity files.
    dir_HOME : str, optional
        Home directory override.
    a, b : str, optional
        Column names to use for frequency and Omega.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the CSV data.
    """

    if dir_HOME is None:
        dir_HOME = COSMOGW_HOME

    dirr = dir_HOME + dir0 + file + '.csv'
    df = pd.read_csv(dirr)
    x = np.array(df[a])
    y = np.array(df[b])
    return x, y


def check_temperature_MeV(T, func=''):
    """
    Check that the temperature T is in MeV units.

    Parameters
    ----------
    T : float or astropy.units.Quantity
        Temperature value.
    func : str, optional
        Name of the calling function (for error messages).

    Returns
    -------
    float
        Temperature in MeV.
    """

    try:
        T = T.to(u.MeV)
    except Exception:
        print(
            'The input temperature in ', func,
            ' needs to be given in energy units using astropy.units \n'
            ' setting T to default value T =', Tref
        )
        T = Tref.to(u.MeV)
    return T


def check_Hubble_Hz(H, func=''):
    """
    Check that the Hubble parameter H is in Hz units.

    Parameters
    ----------
    H : float or astropy.units.Quantity
        Hubble parameter value.
    func : str, optional
        Name of the calling function (for error messages).

    Returns
    -------
    float
        Hubble parameter in Hz.
    """

    try:
        H = H.to(u.Hz)
    except Exception:
        print(
            'The input Hubble rate in ', func,
            ' needs to be given in frequency units using astropy.units \n'
            ' setting H to default value H =', H0_ref
        )
        H = H0_ref
    return H


def complete_beta(a, b):
    """
    Compute the complete beta function for given arguments.

    Parameters
    ----------
    a : float
        First argument.
    b : float
        Second argument.

    Returns
    -------
    float
        Value of the beta function.
    """
    import math as m

    if a > 0 and b > 0:
        B = m.gamma(a)*m.gamma(b)/m.gamma(a + b)
    else:
        print('arguments of beta function need to be positive')
        B = 0

    return B


def Kron_delta(a, b):
    """
    Compute the Kronecker delta for two arguments.

    Parameters
    ----------
    a : int
        First argument.
    b : int
        Second argument.

    Returns
    -------
    int
        1 if a == b, else 0.
    """
    if a == b:
        return 1
    else:
        return 0
