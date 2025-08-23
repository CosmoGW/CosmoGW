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
- **Alberto Roper Pol**
  (`alberto.roperpol@unige.ch <mailto:alberto.roperpol@unige.ch>`_)

Dates
-----
- Created: **01/05/2022**
- Updated: **21/08/2025**
  (release **cosmoGW 1.0**: https://pypi.org/project/cosmoGW)

References
----------
- [**Caprini:2019pxz**]: C. Caprini, D. Figueroa, R. Flauger, G. Nardini,
  M. Peloso, M. Pieroni, A. Ricciardone, G. Tassinato,
  "*Reconstructing the spectral shape of a stochastic gravitational wave
  background with LISA,*" JCAP **11** (2019),
  017, `arXiv:1906.09244 <https://arxiv.org/abs/1906.09244>`_.

- [**Schmitz:2020syl**]: K. Schmitz "*New Sensitivity Curves for
  Gravitational-Wave Signals from Cosmological Phase Transitions,*"
  JHEP **01**, 097 (2021),
  `arXiv:2002.04615 <https://arxiv.org/abs/2002.04615>`_.

- [**Orlando:2020oko**]: G. Orlando, M. Pieroni, A. Ricciardone, "*Measuring
  Parity Violation in the Stochastic Gravitational Wave Background with the
  LISA-Taiji network,*" JCAP **03**, 069 (2021),
  `arXiv:2011.07059 <https://arxiv.org/abs/2011.07059>`_.

- [**RoperPol:2021xnd**]: A. Roper Pol, S. Mandal, A. Brandenburg,
  T. Kahniashvili, "Polarization of gravitational waves from helical MHD
  turbulent sources," JCAP **04** (2022), 019,
  `arXiv:2107.05356 <https://arxiv.org/abs/2107.05356>`_.

- [**RoperPol:2022iel**]: A. Roper Pol, C. Caprini, A. Neronov,
  D. Semikoz, "The gravitational wave signal from primordial
  magnetic fields in the Pulsar Timing Array frequency band,"
  Phys. Rev. D **105**, 123502 (2022),
  `arXiv:2201.05630 <https://arxiv.org/abs/2201.05630>`_.
"""

import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd
from cosmoGW import COSMOGW_HOME
# import reference values and utils functions
from cosmoGW.utils import (
    dir_sens, SNR_PLS, T_PLS, f_ref, P_LISA, L_LISA, A_LISA, P_Taiji,
    L_Taiji, A_Taiji, H0_ref, beta_ref, v_dipole,
    Kron_delta, safe_trapezoid, read_csv
)


# SENSITIVITIES AND NOISE POWER SPECTRAL DENSITY
# READING FUNCTIONS FROM FILES ON SENSITIVITY
def read_response_LISA_Taiji(dir0=dir_sens, dir_HOME=None,
                             TDI=True, interf='LISA'):

    """
    Read response functions for LISA or Taiji interferometers.
    TDI channels are defined following RoperPol:2021xnd, appendix B.

    Parameters
    ----------
    dir0 : str, optional
        Directory containing sensitivity files
        (default: dir_sens).
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


def read_sens(dir0=dir_sens, SNR=SNR_PLS, T=T_PLS, interf='LISA', Xi=False,
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


def read_detector_PLIS_Schmitz(
    dirr0='/power-law-integrated_sensitivities/',
    dir_HOME=None,
    det='BBO', SNR=SNR_PLS, T=T_PLS
):

    """
    Read power law integrated sensitivities from Schmitz:2020syl.

    Parameters
    ----------
    dirr0 : str, optional
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
        dir_HOME = COSMOGW_HOME + dir_sens
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
    dirr0 : str, optional
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
        dir_HOME = COSMOGW_HOME + dir_sens
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

    # Integration over sky directions (theta, phi)
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2 * np.pi, 50)

    # Array of wave numbers
    k = 2 * np.pi * f / const.c
    kL = L * k
    kL = kL.to(1)

    kLij, th, ph = np.meshgrid(kL, theta, phi, indexing='ij')
    kx1 = 0
    kx2 = np.cos(th)
    kx3 = 0.5 * (np.sqrt(3) * np.cos(ph) * np.sin(th) + np.cos(th))
    kU1 = kx2 - kx1
    kU2 = kx3 - kx2
    kU3 = kx1 - kx3

    TkU1, TkU2, TkU3, TkmU1, TkmU2, TkmU3 = \
        _compute_transfer_funcs(kLij, kU1, kU2, kU3)

    shape = (3, 3, len(kL), len(theta), len(phi))
    QA, QE, QT, QX, QY, QZ = _compute_response_functions_Q(
        TDI, comp_all, comp_all_rel, shape, kLij, kx1, kx2, kx3,
        TkU1, TkU2, TkU3, TkmU1, TkmU2, TkmU3)

    k1 = np.cos(ph) * np.sin(th)
    k2 = np.sin(ph) * np.sin(th)
    k3 = np.cos(th)
    e1ab = _compute_polarization_tensor(shape, k1, k2, k3)
    shape2 = (len(kL), len(theta), len(phi))
    FAA, FAE, FEE, FAT, FET, FTT, FXX, FXY, FYY, FZZ, FXZ, FYZ = \
        _compute_response_functions_F(TDI, comp_all, order, comp_all_rel,
                                      shape2, e1ab, QA, QE, QT, QX, QY, QZ)

    return _compute_response_return(FAA, FAE, FEE, FAT, FET, FTT, FXX, FXY,
                                    FYY, FZZ, FXZ, FYZ, TDI, comp_all, order,
                                    comp_all_rel, th, ph, phi)


def _compute_transfer_funcs(kLij, kU1, kU2, kU3):
    '''
    Compute the detector transfer functions for the given wave numbers,
    called from `compute_interferometry`.
    '''

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

    return TkU1, TkU2, TkU3, TkmU1, TkmU2, TkmU3


def _compute_response_functions_Q(TDI, comp_all, comp_all_rel, shape, kLij,
                                  kx1, kx2, kx3, TkU1, TkU2, TkU3,
                                  TkmU1, TkmU2, TkmU3):

    '''
    Compute the detector response functions Q for the given wave numbers,
    called from `compute_interferometry`.
    '''

    def calc_Q(kLij, kx, TkU, U, TkmU_other, U_other, i, j):
        return (
            0.25 * np.exp(-1j * kLij * kx)
            * (TkU * U[i] * U[j] - TkmU_other * U_other[i] * U_other[j])
        )

    # Initialize response function arrays
    QA = np.zeros(shape, dtype=complex)
    QE = np.zeros(shape, dtype=complex)
    QT = np.zeros(shape, dtype=complex)
    QX = np.zeros(shape, dtype=complex)
    QY = np.zeros(shape, dtype=complex)
    QZ = np.zeros(shape, dtype=complex)

    cmat = np.matrix([[2, -1, -1], [0, -np.sqrt(3), np.sqrt(3)], [1, 1, 1]]) / 3
    U1 = np.array([0, 0, 1])
    U2 = 0.5 * np.array([np.sqrt(3), 0, -1])
    U3 = -0.5 * np.array([np.sqrt(3), 0, 1])

    for i in range(3):
        for j in range(3):
            Q1 = calc_Q(kLij, kx1, TkU1, U1, TkmU3, U3, i, j)
            Q2 = calc_Q(kLij, kx2, TkU2, U2, TkmU1, U1, i, j)
            Q3 = calc_Q(kLij, kx3, TkU3, U3, TkmU2, U2, i, j)
            if TDI or comp_all:
                QA[i, j] = Q1 * cmat[0, 0] + Q2 * cmat[0, 1] + Q3 * cmat[0, 2]
                QE[i, j] = Q1 * cmat[1, 0] + Q2 * cmat[1, 1] + Q3 * cmat[1, 2]
                QT[i, j] = Q1 * cmat[2, 0] + Q2 * cmat[2, 1] + Q3 * cmat[2, 2]
            if not TDI or comp_all:
                QX[i, j, :, :, :] = Q1
                QY[i, j, :, :, :] = Q2
                if not comp_all_rel:
                    QZ[i, j, :, :, :] = Q3

    return QA, QE, QT, QX, QY, QZ


def _compute_polarization_tensor(shape, k1, k2, k3):
    '''
    Compute the polarization tensor for the given wave numbers,
    called from `compute_interferometry`.
    '''
    def imag_component(i, j, k1, k2, k3):
        # Returns the imaginary component to add for indices (i, j)
        if i == 0 and j == 1:
            return -1j * k3
        if i == 0 and j == 2:
            return 1j * k2
        if i == 1 and j == 0:
            return 1j * k3
        if i == 1 and j == 2:
            return -1j * k1
        if i == 2 and j == 0:
            return -1j * k2
        if i == 2 and j == 1:
            return 1j * k1
        return 0
    e1ab = np.zeros(shape, dtype=complex)
    kvec = [k1, k2, k3]
    for i in range(3):
        ki = kvec[i]
        for j in range(3):
            kj = kvec[j]
            e1ab[i, j, :, :, :] = Kron_delta(i, j) - ki * kj
            e1ab[i, j, :, :, :] += imag_component(i, j, k1, k2, k3)

    return e1ab


def _einsum_response(eabcd, Q1, Q2):
    '''
    Compute the response using Einstein summation convention,
    called from _compute_response_functions_F.
    '''
    return np.einsum(
        'abcd... , ab... , cd... -> ...', eabcd, Q1, np.conjugate(Q2)
    )


def _print_response_type(TDI, comp_all, order):
    '''
    Print the type of response being computed,
    called from _compute_response_functions_F.
    '''
    if TDI or comp_all:
        if order == 1 and not comp_all:
            print('Computing TDI monopole response functions')
        elif order == 2 and not comp_all:
            print('Computing TDI dipole response functions')
        elif comp_all:
            print('Computing TDI monopole and dipole response functions')
    if not TDI or comp_all:
        if order == 1 and not comp_all:
            print('Computing interferometer monopole response functions')
        elif order == 2 and not comp_all:
            print('Computing interferometer dipole response functions')
        elif comp_all:
            print('Computing interferometer monopole and dipole',
                  ' response functions')


def _compute_response_functions_F(TDI, comp_all, order, comp_all_rel, shape2,
                                  e1ab, QA, QE, QT, QX, QY, QZ):
    '''
    Compute the response functions for the given wave numbers,
    called from `compute_interferometry`.
    '''
    FAA = np.zeros(shape2, dtype=complex)
    FAE = np.zeros(shape2, dtype=complex)
    FEE = np.zeros(shape2, dtype=complex)
    FAT = np.zeros(shape2, dtype=complex)
    FET = np.zeros(shape2, dtype=complex)
    FTT = np.zeros(shape2, dtype=complex)
    FXX = np.zeros(shape2, dtype=complex)
    FXY = np.zeros(shape2, dtype=complex)
    FYY = np.zeros(shape2, dtype=complex)
    FZZ = np.zeros(shape2, dtype=complex)
    FXZ = np.zeros(shape2, dtype=complex)
    FYZ = np.zeros(shape2, dtype=complex)

    eabcd = 0.25 * np.einsum('ac... , bd... -> abcd...', e1ab, e1ab)
    _print_response_type(TDI, comp_all, order)

    if TDI or comp_all:
        FAA = _einsum_response(eabcd, QA, QA)
        FAE = _einsum_response(eabcd, QA, QE)
        FTT = _einsum_response(eabcd, QT, QT)
        if not comp_all_rel:
            FEE = _einsum_response(eabcd, QE, QE)
            FAT = _einsum_response(eabcd, QA, QT)
            FET = _einsum_response(eabcd, QE, QT)

    if not TDI or comp_all:
        FXX = _einsum_response(eabcd, QX, QX)
        FXY = _einsum_response(eabcd, QX, QY)
        if not comp_all_rel:
            FYY = _einsum_response(eabcd, QY, QY)
            FZZ = _einsum_response(eabcd, QZ, QZ)
            FXZ = _einsum_response(eabcd, QX, QZ)
            FYZ = _einsum_response(eabcd, QY, QZ)

    return FAA, FAE, FEE, FAT, FET, FTT, FXX, FXY, FYY, FZZ, FXZ, FYZ


def _compute_response_return(FAA, FAE, FEE, FAT, FET, FTT, FXX, FXY, FYY,
                             FZZ, FXZ, FYZ, TDI, comp_all, order, comp_all_rel,
                             th, ph, phi):
    '''
    Return the integrated response functions, called
    from `compute_interferometry`.
    '''
    def integrate(arr, weight=np.sin(th), axis=1):
        return safe_trapezoid(arr * weight, th, axis=axis)

    def integrate_phi(arr):
        return safe_trapezoid(arr, phi, axis=1) / np.pi

    def tdi_monopole():
        MAA = integrate_phi(integrate(FAA))
        MTT = integrate_phi(integrate(FTT))
        if not comp_all_rel:
            MEE = integrate_phi(integrate(FEE))
            MAE = integrate_phi(integrate(FAE))
            MAT = integrate_phi(integrate(FAT))
            MET = integrate_phi(integrate(FET))
            return (MAA, MEE, MTT, MAE, MAT, MET)
        return (MAA, MTT)

    def tdi_dipole():
        DAE = integrate_phi(1j * integrate(FAE, np.sin(th)**2 * np.sin(ph)))
        if not comp_all_rel:
            weight = np.sin(th)**2 * np.sin(ph)
            DAA = integrate_phi(1j * integrate(FAA, weight))
            DEE = integrate_phi(1j * integrate(FEE, weight))
            DTT = integrate_phi(1j * integrate(FTT, weight))
            DAT = integrate_phi(1j * integrate(FAT, weight))
            DET = integrate_phi(1j * integrate(FET, weight))
            return (DAA, DEE, DTT, DAE, DAT, DET)
        return (DAE,)

    def xyz_monopole():
        MXX = integrate_phi(integrate(FXX))
        MXY = integrate_phi(integrate(FXY))
        if not comp_all_rel:
            MYY = integrate_phi(integrate(FYY))
            MZZ = integrate_phi(integrate(FZZ))
            MXZ = integrate_phi(integrate(FXZ))
            MYZ = integrate_phi(integrate(FYZ))
            return (MXX, MYY, MZZ, MXY, MXZ, MYZ)
        return (MXX, MXY)

    def xyz_dipole():
        DXY = integrate_phi(1j * integrate(FXY, np.sin(th)**2 * np.sin(ph)))
        if not comp_all_rel:
            weight = np.sin(th)**2 * np.sin(ph)
            DXX = integrate_phi(1j * integrate(FXX, weight))
            DYY = integrate_phi(1j * integrate(FYY, weight))
            DZZ = integrate_phi(1j * integrate(FZZ, weight))
            DXZ = integrate_phi(1j * integrate(FXZ, weight))
            DYZ = integrate_phi(1j * integrate(FYZ, weight))
            return (DXX, DYY, DZZ, DXY, DXZ, DYZ)
        return (DXY,)

    if TDI or comp_all:
        if order == 1 or comp_all:
            result = tdi_monopole()
            if not comp_all:
                return result
        if order == 2 or comp_all:
            result = tdi_dipole()
            if not comp_all:
                return result

    if not TDI or comp_all:
        if order == 1 or comp_all:
            result = xyz_monopole()
            if not comp_all:
                return result
        if order == 2 or comp_all:
            result = xyz_dipole()
            if not comp_all:
                return result

    if comp_all:
        if comp_all_rel:
            return (tdi_monopole()[0], tdi_monopole()[1], tdi_dipole()[0],
                    xyz_monopole()[0], xyz_monopole()[1], xyz_dipole()[0])
        else:
            return tdi_monopole() + tdi_dipole() + xyz_monopole() + xyz_dipole()


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


def compute_response_LISA_Taiji(f=f_ref, dir0=dir_sens, dir_HOME=None,
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
        Directory to save results (default: dir_sens).
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


def _get_MAs(fs, interf, TDI, M, Xi):
    if interf != 'comb':
        fs, MAs, MTs, DAEs = read_response_LISA_Taiji(TDI=TDI, interf=interf)
        return fs, MAs, MTs, DAEs
    if not Xi:
        f_ED_I, M_ED_I = read_MAC(M=M, V='I')
        if M == 'MED':
            f_ED_I, M_ED_I = refine_M(f_ED_I, M_ED_I, A=0.028277196782809974)
        M_ED_I *= 2
        fs = (
            np.logspace(
                np.log10(f_ED_I[0].value),
                np.log10(f_ED_I[-1].value),
                1000
            ) * u.Hz
        )
        MAs = np.interp(fs, f_ED_I, M_ED_I)
        MTs = MAs ** 0
        return fs, MAs, MTs, None
    fs, M_AC, M_AD, M_EC, M_ED = read_all_MAC(V='V')
    M_map = {'MAC': abs(M_AC), 'MAD': abs(M_AD),
             'MEC': abs(M_EC), 'MED': abs(M_ED)}
    MAs = M_map.get(M, abs(M_ED))
    MTs = MAs ** 0
    return fs, MAs, MTs, None


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
    fs, MAs, MTs, DAEs = _get_MAs(fs, interf, TDI, M, Xi)

    # Power spectral density of the noise
    noise_map = {
        'LISA': (P_LISA, A_LISA, L_LISA),
        'Taiji': (P_Taiji, A_Taiji, L_Taiji),
        'comb': (P_LISA, A_LISA, L_LISA)
    }
    P, A, L = noise_map.get(interf, (P_LISA, A_LISA, L_LISA))
    PnA, PnT = Pn_f(f=fs, TDI=TDI, P=P, A=A, L=L)
    if interf == 'comb':
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

    H0 = H0_ref * h0
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
