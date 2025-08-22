"""
interferometry.py computes the response and sensitivity functions of space-based
interferometer GW detectors (e.g., LISA, Taiji) for the detection of
stochastic gravitational wave backgrounds (SGWB).

Adapted from the original interferometry in GW_turbulence
(https://github.com/AlbertoRoper/GW_turbulence),
created in May 2022

.. moduleauthor:: Alberto Roper Pol
.. currentmodule:: cosmoGW.interferometry

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/interferometry.py

.. note::
   For full documentation, visit `Read the Docs
   <https://cosmogw-manual.readthedocs.io/en/latest/interferometry.html>`_

.. note::
    See ``tutorials/interferometry/interferometry.ipynb`` for usage examples.

To use it, first install `cosmoGW <https://pypi.org/project/cosmoGW>`_::

    pip install cosmoGW

Author
------
Alberto Roper Pol

Dates
-----
Created: 01/05/2022

Updated: 01/11/2023 (preparation of cosmoGW code, included tutorial)

Updated: 21/08/2025 (release cosmoGW 1.0: https://pypi.org/project/cosmoGW)

References
----------

Caprini:2019pxz  - C. Caprini, D. Figueroa, R. Flauger, G. Nardini, M. Peloso,
M. Pieroni, A. Ricciardone, G. Tassinato, "*Reconstructing the spectral shape
of a stochastic gravitational wave background with LISA,*" JCAP **11** (2019),
017, `arXiv:1906.09244 <https://arxiv.org/abs/1906.09244>`_.

Schmitz:2020syl  - K. Schmitz "*New Sensitivity Curves for
Gravitational-Wave Signals from Cosmological Phase Transitions,*"
JHEP **01**, 097 (2021), `arXiv:2002.04615 <https://arxiv.org/abs/2002.04615>`_.

Orlando:2020oko  - G. Orlando, M. Pieroni, A. Ricciardone, "*Measuring
Parity Violation in the Stochastic Gravitational Wave Background with the
LISA-Taiji network,*" JCAP **03**, 069 (2021),
`arXiv:2011.07059 <https://arxiv.org/abs/2011.07059>`_.

RoperPol:2021xnd - A. Roper Pol, S. Mandal, A. Brandenburg, T. Kahniashvili,
"Polarization of gravitational waves from helical MHD turbulent sources,"
JCAP **04** (2022), 019, `arXiv:2107.05356 <https://arxiv.org/abs/2107.05356>`_.

RoperPol:2022iel - A. Roper Pol, C. Caprini, A. Neronov,
D. Semikoz, "The gravitational wave signal from primordial
magnetic fields in the Pulsar Timing Array frequency band,"
Phys. Rev. D **105**, 123502 (2022),
`arXiv:2201.05630 <https://arxiv.org/abs/2201.05630>`_.
"""

import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd
from cosmoGW import cosmology, COSMOGW_HOME

dir0 = 'resources/detectors_sensitivity/'

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

# Reference frequency and beta arrays
f_ref = np.logspace(-4, 0, 5000) * u.Hz
beta_ref = np.linspace(-20, 20, 3000)


# SENSITIVITIES AND NOISE POWER SPECTRAL DENSITY
# READING FUNCTIONS FROM FILES ON SENSITIVITY
def read_response_LISA_Taiji(dir0=dir0, dir_HOME=None, TDI=True, interf='LISA'):

    """
    Read response functions for LISA or Taiji interferometers.
    TDI channels are defined following RoperPol:2021xnd, appendix B.

    Parameters
    ----------
    dir0 : str, optional
        Directory containing sensitivity files
        (default: 'resources/detector_sensitivity').
    TDI : bool, optional
        If True, read response functions for TDI channels; otherwise XYZ.
    interf : str, optional
        Interferometer name ('LISA' or 'Taiji').

    Returns
    -------
    fs : astropy.units.Quantity
        Frequency array in Hz.
    MAs : np.ndarray
        Response function MA (or MX for XYZ).
    MTs : np.ndarray
        Response function MT (or MXY for XYZ).
    """

    if dir_HOME is None:
        dir_HOME = COSMOGW_HOME

    if interf not in ['LISA', 'Taiji']:
        raise ValueError("Unknown interferometer: {}".format(interf))

    if TDI:
        dirr = dir_HOME + dir0 + interf + '_response_f_TDI.csv'
        df = pd.read_csv(dirr)
        fs = np.array(df['frequency']) * u.Hz
        MAs = np.array(df['MA'])
        MTs = np.array(df['MT'])

    else:
        dirr = dir_HOME + dir0 + interf + '_response_f_X.csv'
        df = pd.read_csv(dirr)
        fs = np.array(df['frequency']) * u.Hz
        MAs = np.array(df['MX'])
        MTs = np.array(df['MXY'])

    # Note: for interferometry channels we have MA -> MX, MT -> MXY, DAE -> DXY
    return fs, MAs, MTs


def read_sens(dir0=dir0, SNR=SNR_PLS, T=T_PLS, interf='LISA', Xi=False,
              TDI=True, chan='A'):

    """
    Read sensitivity curve for a given interferometer.

    For LISA and Taiji the sensitivity can be given for each chanel X, Y,
    Z or on the TDI chanels A=E, T (A chanel is the default option, and
    the relevant for Omega sensitivity).

    Parameters
    ----------
    dir0 : str, optional
        Directory containing sensitivity files
        (default: 'detector_sensitivity').
    SNR : float, optional
        Signal-to-noise ratio threshold (default: 10).
    T : float, optional
        Observation time in years (default: 4).
    interf : str, optional
        Interferometer name ('LISA', 'Taiji', 'comb', 'muAres').
    Xi : bool, optional
        If True, include helical sensitivity and PLS.
    TDI : bool, optional
        If True, use TDI channels.
    chan : str, optional
        Specific channel ('A', 'X', 'Y', 'Z', 'T').

    Returns
    -------
    fs : astropy.units.Quantity
        Frequency array in Hz.
    Omega : np.ndarray
        Sensitivity curve array.
    OmegaPLS : np.ndarray
        Power law sensitivity curve array.
    Xi : np.ndarray, optional
        Helical sensitivity curve array (if Xi is True).
    XiPLS : np.ndarray, optional
        Helical power law sensitivity curve array (if Xi is True).
    """

    fact = SNR / np.sqrt(T)

    if interf == 'LISA':
        fs, LISA_Om = read_csv('LISA_Omega', dir0=dir0)
        fs, LISA_OmPLS = read_csv('LISA_OmegaPLS', dir0=dir0)
        LISA_OmPLS *= fact
        if Xi:
            fs, LISA_Xi = read_csv('LISA_Xi', dir0=dir0, b='Xi')
            fs, LISA_XiPLS = read_csv('LISA_XiPLS', dir0=dir0, b='Xi')
            LISA_XiPLS *= fact
            return fs, LISA_Om, LISA_OmPLS, LISA_Xi, LISA_XiPLS
        return fs, LISA_Om, LISA_OmPLS

    if interf == 'Taiji':
        fs, Taiji_Om = read_csv('Taiji_Omega', dir0=dir0)
        fs, Taiji_OmPLS = read_csv('Taiji_OmegaPLS', dir0=dir0)
        Taiji_OmPLS *= fact
        if Xi:
            fs, Taiji_Xi = read_csv('Taiji_Xi', dir0=dir0, b='Xi')
            fs, Taiji_XiPLS = read_csv('Taiji_XiPLS', dir0=dir0, b='Xi')
            Taiji_XiPLS *= fact
            return fs, Taiji_Om, Taiji_OmPLS, Taiji_Xi, Taiji_XiPLS
        return fs, Taiji_Om, Taiji_OmPLS

    if interf == 'comb':
        fs, LISA_Taiji_Xi = read_csv('LISA_Taiji_Xi', dir0=dir0, b='Xi')
        fs, LISA_Taiji_XiPLS = read_csv('LISA_Taiji_XiPLS', dir0=dir0, b='Xi')
        LISA_Taiji_XiPLS *= fact
        return fs, LISA_Taiji_Xi, LISA_Taiji_XiPLS

    if interf == 'muAres':
        fs, muAres_Om = read_csv('muAres_Omega', dir0=dir0)
        fs, muAres_OmPLS = read_csv('muAres_OmegaPLS', dir0=dir0)
        muAres_OmPLS *= fact
        return fs, muAres_Om, muAres_OmPLS


def read_csv(file, dir0=dir0, dir_HOME=None, a='f', b='Omega'):

    """
    Read a CSV file with two arrays and return them.

    Parameters
    ----------
    file : str
        Name of the CSV file (without extension).
    dir0 : str, optional
        Directory containing the file
        (default: 'resources/detector_sensitivity').
    a : str, optional
        Identifier in pandas dataframe for the first array (default: 'f').
    b : str, optional
        Identifier in pandas dataframe for the second array (default: 'Omega').

    Returns
    -------
    x : np.ndarray
        First array from the file (column `a`).
    y : np.ndarray
        Second array from the file (column `b`).
    """

    if dir_HOME is None:
        dir_HOME = COSMOGW_HOME

    dirr = dir_HOME + dir0 + file + '.csv'
    df = pd.read_csv(dirr)
    x = np.array(df[a])
    y = np.array(df[b])
    return x, y


def read_detector_PLIS_Schmitz(
    dirr0='/power-law-integrated_sensitivities/',
    dir_HOME=None,
    det='BBO', SNR=SNR_PLS, T=T_PLS
):

    """
    Read power law integrated sensitivities from Schmitz:2020syl.

    Parameters
    ----------
    dir0 : str, optional
        Directory where the PLS are stored
        (default: 'detector_sensitivity/power-law-integrated_sensitivities').
    det : str, optional
        GW detector name (check available detectors in default directory).
    SNR : float, optional
        Signal-to-noise ratio (SNR) of the resulting PLS (default: 10).
    T : float, optional
        Duration of the mission in years (default: 4).

    Returns
    -------
    tuple
        f : np.ndarray
            Frequency array.
        Omega : np.ndarray
            Power law integrated sensitivity curve.
    """

    frac = SNR / np.sqrt(T)
    if dir_HOME is None:
        dir_HOME = COSMOGW_HOME + dir0
    dirr = dir_HOME + dirr0 + 'plis_' + det + '.dat'
    df = pd.read_csv(
        dirr, header=14, delimiter='\t',
        names=['f', 'Omega (log)', 'hc (log)', 'Sh (log)']
    )
    f = 10 ** np.array(df['f'])
    Omega = 10 ** np.array(df['Omega (log)'])
    return f, Omega * frac


def read_MAC(dirr0='/LISA_Taiji/', dir_HOME=None, M='MAC', V='V'):

    """
    Read the V response functions of the cross-correlated channels
    of the LISA-Taiji network.

    Reference
    ---------
    RoperPol:2021xnd, see figure 18.
    Data from Orlando:2020oko, see figure 2.

    Parameters
    ----------
    dir0 : str, optional
        Directory where to save the results
        (default: 'detector_sensitivity/LISA_Taiji/').
    M : str, optional
        Channel to be read ('MAC', 'MAD', 'MEC', 'MED'; default: 'MAC').
    V : str, optional
        Stokes parameter. Use 'I' to read the intensity response
        (default: 'V').

    Returns
    -------
    tuple
        f : astropy.units.Quantity
            Frequency array in Hz.
        MAC : np.ndarray
            Response function array.
    """

    if dir_HOME is None:
        dir_HOME = COSMOGW_HOME + dir0
    dirr = dir_HOME + dirr0 + M + '_' + V + '.csv'
    df = pd.read_csv(dirr)
    f = np.array(df['f'])
    MAC = np.array(df['M'])
    inds = np.argsort(f)
    f = f[inds]
    MAC = MAC[inds]
    f = f * u.Hz

    return f, MAC


def read_all_MAC(V='V'):

    """
    Read all relevant TDI cross-correlated response functions between LISA
    and Taiji (AC, AD, EC, ED channels) using read_MAC.

    Reference
    ---------
    RoperPol:2021xnd, see figure 18.
    Data from Orlando:2020oko, see figure 2.

    Parameters
    ----------
    V : str, optional
        Stokes parameter. Use 'I' to read the intensity response
        (default: 'V').

    Returns
    -------
    tuple
        fs : astropy.units.Quantity
            Frequency array in Hz.
        M_AC : np.ndarray
            Response function for AC channel.
        M_AD : np.ndarray
            Response function for AD channel.
        M_EC : np.ndarray
            Response function for EC channel.
        M_ED : np.ndarray
            Response function for ED channel.
    """
    f_AC, M_AC = read_MAC(M='MAC', V=V)
    f_AD, M_AD = read_MAC(M='MAD', V=V)
    f_EC, M_EC = read_MAC(M='MEC', V=V)
    f_ED, M_ED = read_MAC(M='MED', V=V)

    min_f = np.max([
        np.min(f_AC.value), np.min(f_AD.value),
        np.min(f_EC.value), np.min(f_ED.value)
    ])
    max_f = np.min([
        np.max(f_AC.value), np.max(f_AD.value),
        np.max(f_EC.value), np.max(f_ED.value)
    ])

    fs = np.logspace(np.log10(min_f), np.log10(max_f), 1000) * u.Hz
    M_AC = np.interp(fs, f_AC, M_AC) * 2
    M_AD = np.interp(fs, f_AD, M_AD) * 2
    M_EC = np.interp(fs, f_EC, M_EC) * 2
    M_ED = np.interp(fs, f_ED, M_ED) * 2

    return fs, M_AC, M_AD, M_EC, M_ED


# NOISE POWER SPECTRAL DENSITY FUNCTIONS FOR SPACE-BASED INTERFEROMETERS
def Poms_f(f=f_ref, P=P_LISA, L=L_LISA):

    """
    Compute the power spectral density (PSD) of the optical metrology
    system (OMS) noise for a space-based interferometer.

    Reference
    ---------
    RoperPol:2021xnd, equation B.24.

    Parameters
    ----------
    f : astropy.units.Quantity, optional
        Frequency array (default: f_ref).
    P : float, optional
        OMS noise parameter (default: 15 for LISA, 8 for Taiji).
    L : astropy.units.Quantity, optional
        Length of the interferometer arm (default: 2.5e6 km for LISA).

    Returns
    -------
    astropy.units.Quantity
        OMS PSD noise array (units: 1/Hz).
    """

    f_mHz = f.to(u.mHz)
    L_pm = L.to(u.pm)
    Poms = P ** 2 / L_pm.value ** 2 * (1 + (2 / f_mHz.value) ** 4) / u.Hz

    return Poms


def Pacc_f(f=f_ref, A=A_LISA, L=L_LISA):

    """
    Compute the power spectral density (PSD) of the mass acceleration noise
    for a space-based interferometer channel.

    This function implements the analytical formula for the acceleration noise
    PSD as described in RoperPol:2021xnd, equation B.25.

    Parameters
    ----------
    f : astropy.units.Quantity, optional
        Frequency array (default: `f_ref`, units of Hz).
    A : float, optional
        Acceleration noise parameter (default: 3 for LISA; Taiji also uses 3).
    L : astropy.units.Quantity, optional
        Length of the interferometer arm (default: 2.5e6 km for LISA).

    Returns
    -------
    Pacc : astropy.units.Quantity
        Mass acceleration PSD noise (units of 1/Hz).

    References
    ----------
    RoperPol2021xnd
    """

    f_mHz = f.to(u.mHz)
    L_fm = L.to(u.fm)
    c = const.c
    Loc = L / c
    Loc = Loc.to(u.s)

    fsinv = (c / 2 / np.pi / f / L)
    fsinv = fsinv.to(1)

    Pacc = (
        A ** 2 * Loc.value ** 4 / L_fm.value ** 2
        * (1 + (0.4 / f_mHz.value) ** 2)
        * (1 + (f_mHz.value / 8) ** 4)
        * fsinv.value ** 4 / u.Hz
    )

    return Pacc


def Pn_f(f=f_ref, P=P_LISA, A=A_LISA, L=L_LISA, TDI=True):
    """
    Compute the noise power spectral density (PSD) for a space-based
    interferometer channel and its cross-correlation.

    This function calculates the PSD for a single channel (X) and the
    cross-correlation (XY), or the TDI channels (A and T) if TDI is True.

    Implements the analytical formulas from RoperPol:2021xnd,
    equations B.23 and B.26.

    Parameters
    ----------
    f : astropy.units.Quantity, optional
        Frequency array (default: `f_ref`, units of Hz).
    P : float, optional
        Optical metrology system noise parameter (default: 15 for LISA).
    A : float, optional
        Acceleration noise parameter (default: 3 for LISA).
    L : astropy.units.Quantity, optional
        Length of the interferometer arm (default: 2.5e6 km for LISA).
    TDI : bool, optional
        If True, compute TDI channel PSDs (A and T). If False, compute
        single channel and cross-correlation PSDs.

    Returns
    -------
    If TDI is True:
        PnA : astropy.units.Quantity
            Noise PSD of the TDI channel A.
        PnT : astropy.units.Quantity
            Noise PSD of the TDI channel T.
    If TDI is False:
        Pn : astropy.units.Quantity
            Noise PSD of the single channel.
        Pn_cross : astropy.units.Quantity
            Noise PSD of the cross-correlation channel.

    References
    ----------
    Roper Pol et al., JCAP 04 (2022) 019, https://arxiv.org/abs/2107.05356
    """
    Poms = Poms_f(f=f, P=P, L=L)
    Pacc = Pacc_f(f=f, A=A, L=L)
    c = const.c
    f0 = c / (2 * np.pi * L)
    f_f0 = f.to(u.Hz) / f0.to(u.Hz)

    Pn = Poms + (3 + np.cos(2 * f_f0.value)) * Pacc
    Pn_cross = -0.5 * np.cos(f_f0.value) * (Poms + 4 * Pacc)

    if TDI:
        PnA = 2 * (Pn - Pn_cross) / 3
        PnT = (Pn + 2 * Pn_cross) / 3
        return PnA, PnT
    else:
        return Pn, Pn_cross


# INTERFEROMETRY CALCULATIONS
# ANALYTICAL FIT FOR LISA SENSITIVITY
def R_f(f=f_ref, L=L_LISA):

    """
    Compute the analytical fit of the response function for a space-based
    interferometer channel.

    Implements the analytical formula from RoperPol:2021xnd, equation B.15.

    Parameters
    ----------
    f : astropy.units.Quantity, optional
        Frequency array (default: `f_ref`, units of Hz).
    L : astropy.units.Quantity, optional
        Length of the interferometer arm (default: 2.5e6 km for LISA).

    Returns
    -------
    Rf : astropy.units.Quantity
        Analytical fit of the response function.

    References
    ----------
    RoperPol2021xnd
    """

    c = const.c
    f0 = c / (2 * np.pi * L)
    f_f0 = f.to(u.Hz) / f0.to(u.Hz)
    Rf = 0.3 / (1 + 0.6 * f_f0.value ** 2)

    return Rf


def Sn_f_analytical(f=f_ref, P=P_LISA, A=A_LISA, L=L_LISA):

    """
    Compute the strain sensitivity using the analytical fit for a
    space-based interferometer channel.

    This function uses the analytical noise PSD and response function
    to calculate the strain sensitivity.

    Parameters
    ----------
    f : astropy.units.Quantity, optional
        Frequency array (default: `f_ref`, units of Hz).
    P : float, optional
        Optical metrology system noise parameter (default: 15 for LISA).
    A : float, optional
        Acceleration noise parameter (default: 3 for LISA).
    L : astropy.units.Quantity, optional
        Length of the interferometer arm (default: 2.5e6 km for LISA).

    Returns
    -------
    Sn : astropy.units.Quantity
        Strain sensitivity array (units: 1/Hz).

    References
    ----------
    RoperPol2021xnd
    """

    Pn = Pn_f(f=f, P=P, A=A, L=L)
    Rf = R_f(f=f, L=L)
    Sn = Pn / Rf

    return Sn


# NUMERICAL COMPUTATION OF RESPONSE FUNCTIONS
def compute_interferometry(f=f_ref, L=L_LISA, TDI=True, order=1, comp_all=False,
                           comp_all_rel=True):

    """
    Numerically compute the monopole (order=1) or dipole (order=2) response
    functions of a space-based GW interferometer channel.

    This function integrates over sky directions and computes the response
    functions for interferometer channels (or TDI channels if TDI is True).
    It can compute all relevant response functions, including those for
    geometric symmetries.

    Implements the formalism from RoperPol:2021xnd, appendix B
    (eqs. B.13, B.16).

    Parameters
    ----------
    f : astropy.units.Quantity, optional
        Frequency array (default: `f_ref`, units of Hz).
    L : astropy.units.Quantity, optional
        Length of the interferometer arm (default: 2.5e6 km for LISA).
    TDI : bool, optional
        If True, compute TDI channel response functions (A, E, T).
        If False, compute XYZ channel response functions.
    order : int, optional
        Moment of the response function (1 for monopole, 2 for dipole).
    comp_all : bool, optional
        If True, compute all response functions
        (monopole and dipole, X and A channels).
    comp_all_rel : bool, optional
        If True, compute only relevant response functions
        (not identically zero or equal).

    Returns
    -------
    tuple
        Monopole or dipole response functions, depending on `order` and options.
        See function body for details.

    References
    ----------
    RoperPol2021xnd
    """

    if comp_all_rel:
        comp_all = True

    c = const.c

    # Integration over sky directions (theta, phi)
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2 * np.pi, 50)

    # Array of wave numbers
    k = 2 * np.pi * f / c
    kL = L * k
    kL = kL.to(1)

    kLij, th, ph = np.meshgrid(kL, theta, phi, indexing='ij')

    kx1 = 0
    kx2 = np.cos(th)
    kx3 = 0.5 * (np.sqrt(3) * np.cos(ph) * np.sin(th) + np.cos(th))

    kU1 = kx2 - kx1
    kU2 = kx3 - kx2
    kU3 = kx1 - kx3

    # Detector transfer functions (eq. B.3)
    TkU1 = (
        (
            np.exp(-1j * kLij * (1 + kU1) / 2)
            * np.sinc(kLij.value * (1 - kU1) / 2 / np.pi)
            + np.exp(1j * kLij * (1 - kU1) / 2)
            * np.sinc(kLij.value * (1 + kU1) / 2 / np.pi)
        )
    )
    TkU2 = (
        np.exp(-1j * kLij * (1 + kU2) / 2)
        * np.sinc(kLij.value * (1 - kU2) / 2 / np.pi)
        + np.exp(1j * kLij * (1 - kU2) / 2)
        * np.sinc(kLij.value * (1 + kU2) / 2 / np.pi)
    )
    TkU3 = (
        np.exp(-1j * kLij * (1 + kU3) / 2)
        * np.sinc(kLij.value * (1 - kU3) / 2 / np.pi)
        + np.exp(1j * kLij * (1 - kU3) / 2)
        * np.sinc(kLij.value * (1 + kU3) / 2 / np.pi)
    )
    TkmU1 = (
        np.exp(-1j * kLij * (1 - kU1) / 2)
        * np.sinc(kLij.value * (1 + kU1) / 2 / np.pi)
        + np.exp(1j * kLij * (1 + kU1) / 2)
        * np.sinc(kLij.value * (1 - kU1) / 2 / np.pi)
    )
    TkmU2 = (
        np.exp(-1j * kLij * (1 - kU2) / 2)
        * np.sinc(kLij.value * (1 + kU2) / 2 / np.pi)
        + np.exp(1j * kLij * (1 + kU2) / 2)
        * np.sinc(kLij.value * (1 - kU2) / 2 / np.pi)
    )
    TkmU3 = (
        np.exp(-1j * kLij * (1 - kU3) / 2)
        * np.sinc(kLij.value * (1 + kU3) / 2 / np.pi)
        + np.exp(1j * kLij * (1 + kU3) / 2)
        * np.sinc(kLij.value * (1 - kU3) / 2 / np.pi)
    )

    U1 = np.array([0, 0, 1])
    U2 = 0.5 * np.array([np.sqrt(3), 0, -1])
    U3 = -0.5 * np.array([np.sqrt(3), 0, 1])

    cmat = np.matrix([[2, -1, -1], [0, -np.sqrt(3), np.sqrt(3)], [1, 1, 1]]) / 3

    # Initialize response function arrays
    if TDI or comp_all:
        QA = np.zeros((3, 3, len(kL), len(theta), len(phi)), dtype=complex)
        QE = np.zeros((3, 3, len(kL), len(theta), len(phi)), dtype=complex)
        QT = np.zeros((3, 3, len(kL), len(theta), len(phi)), dtype=complex)
    if not TDI or comp_all:
        QX = np.zeros((3, 3, len(kL), len(theta), len(phi)), dtype=complex)
        QY = np.zeros((3, 3, len(kL), len(theta), len(phi)), dtype=complex)
        if not comp_all_rel:
            QZ = np.zeros((3, 3, len(kL), len(theta), len(phi)), dtype=complex)

    for i in range(3):
        for j in range(3):
            Q1 = (
                0.25 * np.exp(-1j * kLij * kx1)
                * (TkU1 * U1[i] * U1[j] - TkmU3 * U3[i] * U3[j])
            )
            Q2 = (
                0.25 * np.exp(-1j * kLij * kx2)
                * (TkU2 * U2[i] * U2[j] - TkmU1 * U1[i] * U1[j])
            )
            Q3 = (
                0.25 * np.exp(-1j * kLij * kx3)
                * (TkU3 * U3[i] * U3[j] - TkmU2 * U2[i] * U2[j])
            )
            if TDI or comp_all:
                QA[i, j, :, :, :] = (
                    Q1 * cmat[0, 0] + Q2 * cmat[0, 1] + Q3 * cmat[0, 2]
                )
                QE[i, j, :, :, :] = (
                    Q1 * cmat[1, 0] + Q2 * cmat[1, 1] + Q3 * cmat[1, 2]
                )
                QT[i, j, :, :, :] = (
                    Q1 * cmat[2, 0] + Q2 * cmat[2, 1] + Q3 * cmat[2, 2]
                )
            if not TDI or comp_all:
                QX[i, j, :, :, :] = Q1
                QY[i, j, :, :, :] = Q2
                if not comp_all_rel:
                    QZ[i, j, :, :, :] = Q3

    k1 = np.cos(ph) * np.sin(th)
    k2 = np.sin(ph) * np.sin(th)
    k3 = np.cos(th)

    # polarization tensors (eq. B.14)
    e1ab = np.zeros((3, 3, len(kL), len(theta), len(phi)), dtype=complex)
    for i in range(3):
        ki = [k1, k2, k3][i]
        for j in range(3):
            kj = [k1, k2, k3][j]
            e1ab[i, j, :, :, :] = delta(i, j) - ki * kj
            if i == 0:
                if j == 1:
                    e1ab[i, j, :, :, :] += -1j * k3
                elif j == 2:
                    e1ab[i, j, :, :, :] += 1j * k2
            elif i == 1:
                if j == 0:
                    e1ab[i, j, :, :, :] += 1j * k3
                elif j == 2:
                    e1ab[i, j, :, :, :] += -1j * k1
            else:
                if j == 0:
                    e1ab[i, j, :, :, :] += -1j * k2
                elif j == 1:
                    e1ab[i, j, :, :, :] += 1j * k1

    if TDI or comp_all:
        if order == 1 and not comp_all:
            print('Computing TDI monopole response functions')
        if order == 2 and not comp_all:
            print('Computing TDI dipole response functions')
        if comp_all:
            print('Computing TDI monopole and dipole response functions')
        FAA = np.zeros((len(kL), len(theta), len(phi)), dtype=complex)
        FAE = np.zeros((len(kL), len(theta), len(phi)), dtype=complex)
        FTT = np.zeros((len(kL), len(theta), len(phi)), dtype=complex)
        if not comp_all_rel:
            FEE = np.zeros((len(kL), len(theta), len(phi)), dtype=complex)
            FAT = np.zeros((len(kL), len(theta), len(phi)), dtype=complex)
            FET = np.zeros((len(kL), len(theta), len(phi)), dtype=complex)
    if not TDI or comp_all:
        if order == 1 and not comp_all:
            print('Computing interferometer monopole response functions')
        if order == 2 and not comp_all:
            print('Computing interferometer dipole response functions')
        if comp_all:
            print(
                ('Computing interferometer monopole and dipole response '
                 'functions')
            )
        FXX = np.zeros((len(kL), len(theta), len(phi)), dtype=complex)
        FXY = np.zeros((len(kL), len(theta), len(phi)), dtype=complex)
        if not comp_all_rel:
            FYY = np.zeros((len(kL), len(theta), len(phi)), dtype=complex)
            FZZ = np.zeros((len(kL), len(theta), len(phi)), dtype=complex)
            FXZ = np.zeros((len(kL), len(theta), len(phi)), dtype=complex)
            FYZ = np.zeros((len(kL), len(theta), len(phi)), dtype=complex)

    for a in range(0, 3):
        for b in range(0, 3):
            for c in range(0, 3):
                for d in range(0, 3):
                    eabcd = 0.25 * e1ab[a, c, :, :, :] * e1ab[b, d, :, :, :]
                    if TDI or comp_all:
                        FAA += (
                            eabcd * QA[a, b, :, :, :]
                            * np.conjugate(QA[c, d, :, :, :])
                        )
                        FAE += (
                            eabcd * QA[a, b, :, :, :]
                            * np.conjugate(QE[c, d, :, :, :])
                        )
                        FTT += (
                            eabcd * QT[a, b, :, :, :]
                            * np.conjugate(QT[c, d, :, :, :])
                        )
                        if not comp_all_rel:
                            FEE += (
                                eabcd * QE[a, b, :, :, :]
                                * np.conjugate(QE[c, d, :, :, :])
                            )
                            FAT += (
                                eabcd * QA[a, b, :, :, :]
                                * np.conjugate(QT[c, d, :, :, :])
                            )
                            FET += (
                                eabcd * QE[a, b, :, :, :]
                                * np.conjugate(QT[c, d, :, :, :])
                            )
                    if not TDI or comp_all:
                        FXX += (
                            eabcd * QX[a, b, :, :, :]
                            * np.conjugate(QX[c, d, :, :, :])
                        )
                        FXY += (
                            eabcd * QX[a, b, :, :, :]
                            * np.conjugate(QY[c, d, :, :, :])
                        )
                        if not comp_all_rel:
                            FYY += (
                                eabcd * QY[a, b, :, :, :]
                                * np.conjugate(QY[c, d, :, :, :])
                            )
                            FZZ += (
                                eabcd * QZ[a, b, :, :, :]
                                * np.conjugate(QZ[c, d, :, :, :])
                            )
                            FXZ += (
                                eabcd * QX[a, b, :, :, :]
                                * np.conjugate(QZ[c, d, :, :, :])
                            )
                            FYZ += (
                                eabcd * QY[a, b, :, :, :]
                                * np.conjugate(QZ[c, d, :, :, :])
                            )

    # Monopole (eq. B.13) and dipole (eq. B.16) response functions of LISA
    # for TDI channels
    if TDI or comp_all:
        if order == 1 or comp_all:
            MAA1 = safe_trapezoid(FAA * np.sin(th), th, axis=1)
            MAA = safe_trapezoid(MAA1, phi, axis=1) / np.pi
            MTT1 = safe_trapezoid(FTT * np.sin(th), th, axis=1)
            MTT = safe_trapezoid(MTT1, phi, axis=1) / np.pi
            if not comp_all_rel:
                MEE1 = safe_trapezoid(FEE * np.sin(th), th, axis=1)
                MEE = safe_trapezoid(MEE1, phi, axis=1) / np.pi
                MAE1 = safe_trapezoid(FAE * np.sin(th), th, axis=1)
                MAE = safe_trapezoid(MAE1, phi, axis=1) / np.pi
                MAT1 = safe_trapezoid(FAT * np.sin(th), th, axis=1)
                MAT = safe_trapezoid(MAT1, phi, axis=1) / np.pi
                MET1 = safe_trapezoid(FET * np.sin(th), th, axis=1)
                MET = safe_trapezoid(MET1, phi, axis=1) / np.pi
            if not comp_all:
                return MAA, MEE, MTT, MAE, MAT, MET

        if order == 2 or comp_all:
            DAE1 = 1j * safe_trapezoid(
                FAE * np.sin(th) ** 2 * np.sin(ph), th, axis=1
            )
            DAE = safe_trapezoid(DAE1, phi, axis=1) / np.pi
            if not comp_all_rel:
                DAA1 = 1j * safe_trapezoid(
                    FAA * np.sin(th) ** 2 * np.sin(ph), th, axis=1
                )
                DAA = safe_trapezoid(DAA1, phi, axis=1) / np.pi
                DEE1 = 1j * safe_trapezoid(
                    FEE * np.sin(th) ** 2 * np.sin(ph), th, axis=1)
                DEE = safe_trapezoid(DEE1, phi, axis=1) / np.pi
                DTT1 = 1j * safe_trapezoid(
                    FTT * np.sin(th) ** 2 * np.sin(ph), th, axis=1
                )
                DTT = safe_trapezoid(DTT1, phi, axis=1) / np.pi
                DAT1 = 1j * safe_trapezoid(
                    FAT * np.sin(th) ** 2 * np.sin(ph), th, axis=1
                )
                DAT = safe_trapezoid(DAT1, phi, axis=1) / np.pi
                DET1 = 1j * safe_trapezoid(
                    FET * np.sin(th) ** 2 * np.sin(ph), th, axis=1
                )
                DET = safe_trapezoid(DET1, phi, axis=1) / np.pi
            if not comp_all:
                return DAA, DEE, DTT, DAE, DAT, DET

    if not TDI or comp_all:
        if order == 1 or comp_all:
            MXX1 = safe_trapezoid(FXX * np.sin(th), th, axis=1)
            MXX = safe_trapezoid(MXX1, phi, axis=1) / np.pi
            MXY1 = safe_trapezoid(FXY * np.sin(th), th, axis=1)
            MXY = safe_trapezoid(MXY1, phi, axis=1) / np.pi
            if not comp_all_rel:
                MYY1 = safe_trapezoid(FYY * np.sin(th), th, axis=1)
                MYY = safe_trapezoid(MYY1, phi, axis=1) / np.pi
                MZZ1 = safe_trapezoid(FZZ * np.sin(th), th, axis=1)
                MZZ = safe_trapezoid(MZZ1, phi, axis=1) / np.pi
                MXZ1 = safe_trapezoid(FXZ * np.sin(th), th, axis=1)
                MXZ = safe_trapezoid(MXZ1, phi, axis=1) / np.pi
                MYZ1 = safe_trapezoid(FYZ * np.sin(th), th, axis=1)
                MYZ = safe_trapezoid(MYZ1, phi, axis=1) / np.pi
            if not comp_all:
                return MXX, MYY, MZZ, MXY, MXZ, MYZ

        if order == 2 or comp_all:
            DXY1 = 1j * safe_trapezoid(
                FXY * np.sin(th) ** 2 * np.sin(ph), th, axis=1
            )
            DXY = safe_trapezoid(DXY1, phi, axis=1) / np.pi
            if not comp_all_rel:
                DXX1 = 1j * safe_trapezoid(
                    FXX * np.sin(th) ** 2 * np.sin(ph), th, axis=1
                )
                DXX = safe_trapezoid(DXX1, phi, axis=1) / np.pi
                DYY1 = 1j * safe_trapezoid(
                    FYY * np.sin(th) ** 2 * np.sin(ph), th, axis=1
                )
                DYY = safe_trapezoid(DYY1, phi, axis=1) / np.pi
                DZZ1 = 1j * safe_trapezoid(
                    FZZ * np.sin(th) ** 2 * np.sin(ph), th, axis=1
                )
                DZZ = safe_trapezoid(DZZ1, phi, axis=1) / np.pi
                DXZ1 = 1j * safe_trapezoid(
                    FXZ * np.sin(th) ** 2 * np.sin(ph), th, axis=1
                )
                DXZ = safe_trapezoid(DXZ1, phi, axis=1) / np.pi
                DYZ1 = 1j * safe_trapezoid(
                    FYZ * np.sin(th) ** 2 * np.sin(ph), th, axis=1
                )
                DYZ = safe_trapezoid(DYZ1, phi, axis=1) / np.pi
            if not comp_all:
                return DXX, DYY, DZZ, DXY, DXZ, DYZ

    if comp_all:
        if comp_all_rel:
            return MAA, MTT, DAE, MXX, MXY, DXY
        else:
            return (
                MAA, MEE, MTT, MAE, MAT, MET,
                DAA, DEE, DTT, DAE, DAT, DET,
                MXX, MYY, MZZ, MXY, MXZ, MYZ,
                DXX, DYY, DZZ, DXY, DXZ, DYZ
            )


def refine_M(f, M, A=.3, exp=0):

    """
    Refine the response function by appending a low-frequency term.

    Parameters
    ----------
    f : np.ndarray or astropy.units.Quantity
        Frequency array (should be in units of Hz).
    M : np.ndarray
        Response function to be refined at low frequencies.
    A : float, optional
        Amplitude of the response function at low frequencies
        (default: 0.3 for LISA monopole).
    exp : int, optional
        Exponent of the response function at low frequencies (default: 0).

    Returns
    -------
    fs : np.ndarray
        Refined array of frequencies.
    Ms : np.ndarray
        Refined response function.
    """

    ff0 = np.logspace(-6, np.log10(f[0].value), 1000) * u.Hz
    fs = np.append(ff0, f)
    Ms = np.append(A * ff0.value ** exp, np.real(M))

    return fs, Ms


def compute_response_LISA_Taiji(f=f_ref, dir0=dir0, dir_HOME=None,
                                save=True, ret=False):

    """
    Compute LISA and Taiji's monopole and dipole response functions.

    Uses the :func:`compute_interferometry` routine and refines the response
    functions at low frequencies.

    Parameters
    ----------
    f : np.ndarray or astropy.units.Quantity, optional
        Frequency array (default: f_ref).
    dir0 : str, optional
        Directory to save results (default: dir0).
    save : bool, optional
        If True, saves results as output files (default: True).
    ret : bool, optional
        If True, returns the results from the function (default: False).

    Returns
    -------
    MAA : np.ndarray
        Monopole response function of the TDI channel A.
    MTT : np.ndarray
        Monopole response function of the TDI channel T (Sagnac channel).
    DAE : np.ndarray
        Dipole response function of the TDI correlation of the
        channels A and E.
    MXX : np.ndarray
        Monopole response function of the interferometer channel X.
    MXY : np.ndarray
        Monopole response function of the correlation of interferometer
        channels X and Y.
    DXY : np.ndarray
        Dipole response functions of the correlation of the interferometer
        channels X and Y.
    """

    f = np.logspace(-4, 0, 5000) * u.Hz

    if dir_HOME is None:
        dir_HOME = COSMOGW_HOME + dir0

    # LISA response functions
    print('Calculating LISA response functions')
    MAA, MTT, DAE, MXX, MXY, DXY = compute_interferometry(
        f=f, L=L_LISA, comp_all_rel=True
    )
    # Taiji response functions
    print('Calculating Taiji response functions')
    MAA_Tai, MTT_Tai, DAE_Tai, MXX_Tai, MXY_Tai, DXY_Tai = \
        compute_interferometry(f=f, L=L_Taiji, comp_all_rel=True)

    # refine response functions at low frequencies (from known results)
    fs, MAs = refine_M(f, MAA)
    fs, MAs_Tai = refine_M(f, MAA_Tai)
    fs, MTs = refine_M(f, MTT, A=1.709840e6, exp=6)
    fs, MTs_Tai = refine_M(f, MTT_Tai, A=5.105546e6, exp=6)
    fs, DAEs = refine_M(f, DAE, A=0.2)
    fs, DAEs_Tai = refine_M(f, DAE_Tai, A=0.2)
    fs, MXs = refine_M(f, MXX, A=MXX[0])
    fs, MXs_Tai = refine_M(f, MXX_Tai, A=MXX_Tai[0])
    fs, MXYs = refine_M(f, MXY, A=MXY[0])
    fs, MXYs_Tai = refine_M(f, MXY_Tai, A=MXY_Tai[0])
    fs, DXYs = refine_M(f, DXY, A=DXY[0])
    fs, DXYs_Tai = refine_M(f, DXY_Tai, A=DXY_Tai[0])

    # Write response functions in csv files
    if save:
        df = pd.DataFrame({
            'frequency': fs, 'MX': np.real(MXs), 'MXY': np.real(MXYs),
            'DXY': np.real(DXYs)
        })
        df.to_csv(dir_HOME + 'LISA_response_f_X.csv')
        df = pd.DataFrame({
            'frequency': fs, 'MA': MAs, 'MT': MTs, 'DAE': DAEs
        })
        df.to_csv(dir_HOME + 'LISA_response_f_TDI.csv')
        print('saved response functions of channels X, Y of LISA in ',
              dir_HOME + 'LISA_response_f_X.csv')
        print('saved response functions of TDI channels of LISA in ',
              dir_HOME + 'LISA_response_f_TDI.csv')
        df_Tai = pd.DataFrame({
            'frequency': fs, 'MX': np.real(MXs_Tai), 'MXY': np.real(MXYs_Tai),
            'DXY': np.real(DXYs_Tai)
        })
        df_Tai.to_csv(dir_HOME + 'Taiji_response_f_X.csv')
        df_Tai = pd.DataFrame({
            'frequency': fs, 'MA': MAs_Tai, 'MT': MTs_Tai, 'DAE': DAEs_Tai
        })
        df_Tai.to_csv(dir_HOME + 'Taiji_response_f_TDI.csv')
        print('saved response functions of channels X, Y of Taiji in ',
              dir_HOME + 'Taiji_response_f_X.csv')
        print('saved response functions of TDI channels of Taiji in ',
              dir_HOME + 'Taiji_response_f_TDI.csv')

    if ret:
        return (
            fs, MAs, MTs, DAEs, MXs, MXYs, DXYs,
            MAs_Tai, MTs_Tai, DAEs_Tai, MXs_Tai, MXYs_Tai, DXYs_Tai
        )


# SENSITIVITIES AND SNR
def Sn_f(fs=f_ref, v=v_dipole, interf='LISA', TDI=True, M='MED', Xi=False):

    """
    Compute the strain sensitivity using the analytical fit for an
    interferometer channel.

    Parameters
    ----------
    fs : np.ndarray or astropy.units.Quantity, optional
        Frequency array (default: f_ref).
    v : float, optional
        Dipole velocity of the solar system (default: 1.23e-3).
    interf : str, optional
        Interferometer ('LISA', 'Taiji', 'comb'; default: 'LISA').
    TDI : bool, optional
        If True, use TDI channels (default: True).
    M : str, optional
        Cross-correlated channel (default: 'MED').
    Xi : bool, optional
        If True, compute sensitivity to polarized GW backgrounds
        (default: False).

    Returns
    -------
    tuple
        Frequency array and strain sensitivities (see function body).
    """

    # read LISA and Taiji TDI response functions
    if interf != 'comb':
        fs, MAs, MTs, DAEs = read_response_LISA_Taiji(TDI=TDI, interf=interf)
    else:
        if not Xi:
            f_ED_I, M_ED_I = read_MAC(M=M, V='I')
            if M == 'MED':
                f_ED_I, M_ED_I = refine_M(
                    f_ED_I, M_ED_I, A=0.028277196782809974
                )
            M_ED_I *= 2
            fs = np.logspace(
                np.log10(f_ED_I[0].value), np.log10(f_ED_I[-1].value), 1000
            ) * u.Hz
            MAs = np.interp(fs, f_ED_I, M_ED_I)
        else:
            fs, M_AC, M_AD, M_EC, M_ED = read_all_MAC(V='V')
            if M == 'MAC':
                MAs = abs(M_AC)
            if M == 'MAD':
                MAs = abs(M_AD)
            if M == 'MEC':
                MAs = abs(M_EC)
            if M == 'MED':
                MAs = abs(M_ED)
        MTs = MAs ** 0

    # power spectral density of the noise
    if interf == 'LISA':
        PnA, PnT = Pn_f(f=fs, TDI=TDI, P=P_LISA, A=A_LISA, L=L_LISA)
    if interf == 'Taiji':
        PnA, PnT = Pn_f(f=fs, TDI=TDI, P=P_Taiji, A=A_Taiji, L=L_Taiji)
    if interf == 'comb':
        PnA, PnT = Pn_f(f=fs, TDI=TDI, P=P_LISA, A=A_LISA, L=L_LISA)
        PnC, PnS = Pn_f(f=fs, TDI=TDI, P=P_Taiji, A=A_Taiji, L=L_Taiji)
        PnA = np.sqrt(PnA * PnC)
        PnT = np.sqrt(PnT * PnS)

    # if Xi is True, it only returns the strain sensitivity to cross-correlated
    # channels A and E
    if Xi:
        if interf != 'comb':
            SnA = PnA / v / abs(DAEs)
        else:
            SnA = PnA / MAs
        return fs, SnA

    # if not TDI, then PnA -> PnX, PnT -> Pncross, MAs -> MXs, MTs -> MXYs
    return fs, PnA / MAs, PnT / MTs


def Oms(f, S, h0=1.0, comb=False, S2=None, S3=None, S4=None, Xi=False):

    """
    Return the sensitivity Sh(f) in terms of the GW energy density Om(f).

    Reference for Omega is RoperPol:2021xnd, equation B.18
    (seems like it might have a typo, need to investigate this!).
    Final PLS sensitivites are again correct for a single channel,
    since the factor of 2 in the SNR compensates for the 1/2 factor here.
    Reference for combined sensitivities is RoperPol:2021xnd, equations B.37
    (GW energy density, combining LISA and Taiji TDI channels A and C), and B.41
    (polarization, combining 4 cross-correlation between LISA and Taiji TDI
    channels AE, AD, CE, CD).

    Strain sensitivities, Omega sensitivities, and OmGW PLS agree with those of
    reference Caprini:2019pxz, see equation 2.14.

    References
    ----------
    RoperPol:2021xnd, equations B.18, B.21, B.37, B.41.
    Caprini:2019pxz, equation 2.14.

    Parameters
    ----------
    f : np.ndarray or astropy.units.Quantity
        Frequency array (should be in units of Hz).
    S : np.ndarray
        Strain sensitivity function.
    h0 : float, optional
        Hubble rate uncertainty parameter (default: 1).
    comb : bool, optional
        If True, combine two sensitivities (default: False).
    S2, S3, S4 : np.ndarray, optional
        Additional sensitivities for combination (default: None).
    Xi : bool, optional
        If True, compute sensitivity to polarized GW backgrounds
        (default: False).

    Returns
    -------
    Omega : np.ndarray
        GW energy density sensitivity.
    """
    if S2 is None:
        S2 = []
    if S3 is None:
        S3 = []
    if S4 is None:
        S4 = []

    H0 = cosmology.H0_ref * h0
    A = 4 * np.pi ** 2 / 3 / H0 ** 2
    if Xi:
        A /= 2
    Omega = S * A * f ** 3

    if comb:
        Omega2 = S2 * A * f ** 3
        Omega = 1 / np.sqrt(1 / Omega ** 2 + 1 / Omega2 ** 2)
        if Xi:
            Omega3 = S3 * A * f ** 3
            Omega4 = S4 * A * f ** 3
            Omega = 1 / np.sqrt(
                1 / Omega ** 2 + 1 / Omega2 ** 2 +
                1 / Omega3 ** 2 + 1 / Omega4 ** 2
            )

    return Omega


def compute_Oms_LISA_Taiji(interf='LISA', TDI=True, h0=1.0):
    """
    Read response functions for LISA and/or Taiji, compute strain sensitivities,
    and from those, the sensitivity to the GW energy density spectrum Omega_s.

    Parameters
    ----------
    interf : str, optional
        Interferometer ('LISA', 'Taiji', 'comb'; default: 'LISA').
    TDI : bool, optional
        If True, use TDI channels (default: True).
    h0 : float, optional
        Hubble rate uncertainty parameter (default: 1.0).

    Returns
    -------
    tuple
        fs : frequency array
        OmSA : GW energy density sensitivity for channel A
        OmST : GW energy density sensitivity for channel T
    """

    # read LISA and Taiji strain sensitivities Sn_f(f)
    fs, SnA, SnT = Sn_f(interf=interf, TDI=TDI)

    # Sn is the sensitivity of the channel A (if TDI) or X (if not TDI) for LISA
    # or Taiji (depending on what is interf)
    OmSA = Oms(fs, SnA, h0=h0, comb=False, Xi=False)
    OmST = Oms(fs, SnT, h0=h0, comb=False, Xi=False)
    return fs, OmSA, OmST


def OmPLS(Oms, f=f_ref, beta=beta_ref, SNR=1, T=1, Xi=0):

    """
    Compute the power law sensitivity (PLS).

    Reference
    ---------
    RoperPol:2021xnd, appendix B (equation B.31).

    Parameters
    ----------
    Oms : np.ndarray
        GW energy density sensitivity.
    f : np.ndarray or astropy.units.Quantity, optional
        Frequency array (default: f_ref).
    beta : np.ndarray, optional
        Array of slopes (default: beta_ref from -20 to 20).
    SNR : float, optional
        Signal-to-noise ratio threshold (default: 1).
    T : float, optional
        Duration of the observation in years (default: 1).
    Xi : float, optional
        For polarization signals using the dipole response (default: 0).

    Returns
    -------
    Omega : np.ndarray
        GW energy density power law sensitivity (PLS).
    """

    Cbeta = np.zeros(len(beta))
    T_sec = (T * u.yr).to(u.s)
    for i in range(len(beta)):
        aux = f.value ** (2 * beta[i])
        aa = abs(1 - Xi * beta[i])
        Cbeta[i] = SNR / aa / np.sqrt(
            safe_trapezoid(aux / Oms ** 2, f.value) * T_sec.value
        )

    funcs = np.zeros((len(f), len(beta)))
    for i in range(len(beta)):
        funcs[:, i] = f.value ** beta[i] * Cbeta[i]
    Omega = np.zeros(len(f))
    for j in range(len(f)):
        Omega[j] = np.max(funcs[j, :])

    return Omega


def SNR(f, OmGW, fs, Oms, T=1.):

    r"""
    Compute the signal-to-noise ratio (SNR) of a GW signal.

    .. math ::
       {\rm SNR} = \sqrt{T \int \left( \frac{\Omega_{\rm GW} (f)}
       {\Omega_{\rm sens} (f)} \right)^2 df}

    Reference
    ---------
    RoperPol:2021xnd, appendix B (equation B.30).

    Parameters
    ----------
    f : np.ndarray or astropy.units.Quantity
        Frequency array of the GW signal.
    OmGW : np.ndarray
        GW energy density spectrum of the GW signal.
    fs : np.ndarray or astropy.units.Quantity
        Frequency array of the GW detector sensitivity.
    Oms : np.ndarray
        GW energy density sensitivity of the GW detector.
    T : float, optional
        Duration of observations in years (default: 1.0).

    Returns
    -------
    float
        SNR of the GW signal.
    """

    T_sec = (T * u.yr).to(u.s).value
    f_hz = f.to(u.Hz).value
    OmGW_interp = np.interp(fs, f_hz, OmGW)
    OmGW_interp[np.where(fs < f_hz[0])] = 0
    OmGW_interp[np.where(fs > f_hz[-1])] = 0
    integ = safe_trapezoid((OmGW_interp / Oms) ** 2, fs)
    SNR = np.sqrt(T_sec * integ)
    return SNR


def safe_trapezoid(y, x, axis=-1):
    """
    Safely compute the trapezoidal integral of y with respect to x.

    Uses numpy.trapezoid (Numpy>=1.20.0) or trapz function (older versions).

    Parameters
    ----------
    y : np.ndarray
        Array to integrate.
    x : np.ndarray
        Array of integration variable.
    axis : int, optional
        Axis along which to integrate (default: -1).

    Returns
    -------
    float or np.ndarray
        Result of the integration.
    """
    try:
        return np.trapezoid(y, x, axis=axis)
    except AttributeError:
        return np.trapz(y, x, axis=axis)


def delta(a, b):
    """
    Kronecker delta function.

    Parameters
    ----------
    a : int or float
        First value.
    b : int or float
        Second value.

    Returns
    -------
    int
        1 if a == b, 0 otherwise.
    """
    if a == b:
        return 1
    else:
        return 0
