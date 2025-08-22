"""
cosmology.py is a Python routine that contains functions relevant
for cosmological calculations, including a solver of Friedmann equations.

Adapted from the original cosmology in GW_turbulence
(https://github.com/AlbertoRoper/GW_turbulence),
created in Nov. 2022

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/cosmology.py

.. note::
   For full documentation, visit `Read the Docs
   <https://cosmogw-manual.readthedocs.io/en/latest/cosmology.html>`_

To use it, first install `cosmoGW <https://pypi.org/project/cosmoGW>`_::

    pip install cosmoGW

Author
------
- **Alberto Roper Pol**
  (`alberto.roperpol@unige.ch <mailto:alberto.roperpol@unige.ch>`_)

Dates
-----
- Created: **27/11/2022** (*GW_turbulence*)
- Updated: **21/08/2025** (release **cosmoGW 1.0**:
  https://pypi.org/project/cosmoGW)

References
----------
 - [**RoperPol:2018sap**]: A. Roper Pol, A. Brandenburg, T. Kahniashvili,
   A. Kosowsky, S. Mandal, "*The timestep constraint in solving the
   gravitational wave equations sourced by hydromagnetic turbulence,*".
   Geophys. Astrophys. Fluid Dynamics **114**, 1, 130 (2020),
   `arXiv:1807.05479 <https://arxiv.org/abs/1807.05479>`_.

 - [**RoperPol:2019wvy**]: A. Roper Pol, S. Mandal, A. Brandenburg,
   T. Kahniashvili, A. Kosowsky, "*Numerical simulations of gravitational
   waves from early-universe turbulence,*" Phys. Rev. D **102**, 083512 (2020),
   `arXiv:1903.08585 <https://arxiv.org/abs/1903.08585>`_.

 - [**RoperPol:2022iel**]: A. Roper Pol, C. Caprini, A. Neronov, D. Semikoz,
   "*The gravitational wave signal from primordial magnetic fields in the
   Pulsar Timing Array frequency band,*" Phys. Rev. D **105**, 123502 (2022),
   `arXiv:2201.05630 <https://arxiv.org/abs/2201.05630>`_.

 - [**He:2022qcs**]: Y. He, A. Roper Pol, A. Brandenburg,
   "*Modified propagation of gravitational waves from the early
   radiation era,*" JCAP **06** (2023), 025,
   `arXiv:2212.06082 <https://arxiv.org/abs/2212.06082>`_.

 - [**Planck:2018vyg**]: [Planck], "*Planck 2018 results. VI.
   Cosmological parameters,*" Astron. Astrophys. **641** (2020) A6,
   Astron. Astrophys. **652** (2021) C4 (erratum),
   `arXiv:1807.06209 <https://arxiv.org/abs/1807.06209>`_.

 - [**Rasanen**]: Lecture notes on Cosmology by Syksy Räsänen
   (https://www.mv.helsinki.fi/home/syrasane/cosmo/)
"""

import astropy.constants as const
import astropy.units as u
import pandas as pd
import numpy as np
from cosmoGW import COSMOGW_HOME
from cosmoGW.utils import (
    T0K, H0_ref, Neff_ref, gref, Tref, OmL0_ref, OmM0_ref, h0_ref,
    check_temperature_MeV, check_Hubble_Hz
)


def values_0(h0=1., neut=False, Neff=Neff_ref, ret_rad=False):
    """
    Return cosmological parameters at present time.

    Parameters
    ----------
    h0 : float
        Hubble rate scaling (default 1).
    neut : bool
        If True, include neutrinos in relativistic dofs.
    Neff : float
        Effective number of neutrino species (default 3).
    ret_rad : bool
        If True, also return radiation energy density.

    Returns
    -------
    g0 : float
        Relativistic degrees of freedom (2 if massive neutrinos are assumed).
    g0s : float
        Entropic/adiabatic degrees of freedom (3.91 including neutrinos)
    T0 : astropy.units.Quantity
        Present temperature in MeV, corresponding to 2.72548 K
    H0 : astropy.units.Quantity
        Present Hubble rate in Hz, corresponding to H0 = 100 h0 km/s/Mpc.
    rho_rad0 : astropy.units.Quantity, optional
        Radiation energy density at present time.
    Om_rad0 : float, optional
        Radiation density ratio to critical at present time.


    References
    ----------
    Rasanen - Lecture notes on Cosmology.
    """
    g0 = 2.
    if neut:
        g0 = 2 * (1 + Neff * 7 / 8 * (4. / 11) ** (4. / 3))
    g0s = 2 * (1 + Neff * 7 / 8 * 4. / 11)
    T0 = (T0K * const.k_B).to(u.MeV)
    H0 = H0_ref * h0

    if ret_rad:
        rho_rad0 = rho_radiation(g=g0, T=T0)
        rho_c = rho_critical(H=H0)
        Om_rad0 = rho_rad0 / rho_c
        return g0, g0s, T0, H0, rho_rad0, Om_rad0
    else:
        return g0, g0s, T0, H0


def Hs_fact():

    r"""
    Compute factor for Hubble rate at generation time (radiation era).

    The factor is given in units of Hz/:math:`{\rm MeV}^2`,
    such that the actual Hubble rate
    is given after multiplying fact x :math:`\sqrt{g}`
    x :math:`T^2`, being
    g the number of dof and
    T the temperature scale (in MeV).

    Returns
    -------
    fact : astropy.units.Quantity
        Factor in Hz/:math:`{\rm MeV}^2`.

    Reference
    ---------
    RoperPol:2022iel, eq. 29
    """
    fact = np.sqrt(
        4 * np.pi ** 3 * const.G / 45 / const.c ** 5 / const.hbar ** 3
    )
    fact = fact.to(u.Hz / u.MeV ** 2)
    return fact


def Hs_val(g=gref, T=Tref):
    """
    Compute Hubble rate at generation time (radiation era).

    Parameters
    ----------
    g : float
        Relativistic degrees of freedom (default 100).
    T : astropy.units.Quantity
        Temperature scale (default 100 GeV).

    Returns
    -------
    Hs : astropy.units.Quantity
        Hubble rate in frequency units (Hz).

    Reference
    ---------
    RoperPol:2022iel, eq. 29
    """
    Hs_f = Hs_fact()
    T = check_temperature_MeV(T, func='Hs_val')
    Hs = Hs_f * T ** 2 * np.sqrt(g)
    return Hs


def as_fact(Neff=Neff_ref):
    """
    Compute factor for scale factor ratio at generation time
    assuming adiabatic expansion of the universe.

    The factor is in units of MeV and the ratio is obtained by multiplying
    fact x :math:`g^{-1/3}/T`, being :math:`g` the number of dof
    and :math:`T` the temperature scale (in MeV).

    Parameters
    ----------
    Neff : float
        Effective number of neutrino species (default is 3).

    Returns
    -------
    fact : astropy.units.Quantity
        Ratio to present-day scale factor, :math:`T_0 g_{0s}^{1/3}`.

    Reference
    ---------
    RoperPol:2022iel, eq. 28
    """
    _, g0s, T0, _ = values_0(Neff=Neff)
    fact = T0 * g0s ** (1 / 3)
    return fact


def as_a0_rat(g=gref, T=Tref, Neff=Neff_ref):
    """
    Compute ratio of scale factors (generation/present)
    assuming adiabatic expansion of the universe.

    Parameters
    ----------
    g : float
        Number of relativistic degrees of freedom (dof) at the
        time of generation (default is 100).
    T : astropy.units.Quantity
        Temperature scale at the time of generation (default is 100 GeV).
    Neff : float
        Effective number of neutrino species (default is 3).

    Returns
    -------
    as_a0 : astropy.units.Quantity
        Ratio of scale factors (a*/a0).

    Reference
    ---------
    RoperPol:2022iel, eq. 28
    """
    as_f = as_fact(Neff=Neff)
    T = check_temperature_MeV(T, func='as_a0_rat')
    as_a0 = as_f * g ** (-1 / 3) / T
    return as_a0


def rho_radiation(g=gref, T=Tref):
    """
    Compute radiation energy density at a given epoch.

    Parameters
    ----------
    g : float
        Relativistic degrees of freedom (default 100).
    T : astropy.units.Quantity
        Temperature scale (default 100 GeV).

    Returns
    -------
    rho_rad : astropy.units.Quantity
        Radiation energy density in GeV/m^3 units.

    Reference
    ---------
    RoperPol:2019wvy, eq. 3
    """
    T = check_temperature_MeV(T, func='rho_radiation')
    rho_rad = np.pi ** 2 / 30 * g * T ** 4 / (const.hbar * const.c) ** 3
    rho_rad = rho_rad.to(u.GeV / u.m ** 3)
    return rho_rad


def rho_critical(H=H0_ref):
    """
    Compute critical energy density at a given epoch.

    Parameters
    ----------
    H : astropy.units.Quantity
        Hubble rate in units of frequency.

    Returns
    -------
    rho_c : astropy.units.Quantity
        Critical energy density in GeV/m^3 units.

    Reference
    ---------
    RoperPol:2019wvy, eq. 3
    """
    H = check_Hubble_Hz(H, func='rho_critical')
    rho_c = 3 * H ** 2 * const.c ** 2 / 8 / np.pi / const.G
    rho_c = rho_c.to(u.GeV / u.m ** 3)
    return rho_c


def thermal_g(dir0='', T=Tref, s=0, file=True, Neff=Neff_ref):
    """
    Return relativistic or adiabatic degrees of freedom as a function of T
    according to the thermal history of the Universe.

    Note that for T < 0.5 MeV, after neutrino decoupling, entropic
    and relativistic g are approximately equal.

    Parameters
    ----------
    dir0 : str, optional
        Directory for dof file (default resources/cosmology/).
    T : astropy.units.Quantity, optional
        Temperature scale (default 100 GeV).
    s : int, optional
        0 for relativistic, 1 for adiabatic dof.
    file : bool, optional
        If True, read dof from file. It reads the file
        'dir0/cosmology/T_gs.csv' and should be a
        pandas file with columns:
        ['T [GeV]', 'g_*', 'gS']
    Neff : float, optional
        Effective number of neutrino species (default 3).

    Returns
    -------
    g : float or ndarray
        Relativistic or adiabatic degrees of freedom.

    Reference
    ---------
    Rasanen - Lecture notes on Cosmology.
    """
    g0, g0s, _, _ = values_0(neut=True, Neff=Neff)
    if file:
        try:
            return _thermal_g_from_file(dir0, T, s, g0, g0s)
        except Exception:
            file = False
            print(
                'thermal_g reads the file %sT_gs.csv, which does not exist! \n'
                % dir0
            )
            print('using piecewise approximated function to approximate g')
    if not file:
        return _thermal_g_piecewise(T, s, g0, g0s)
    return None


def _thermal_g_from_file(dir0, T, s, g0, g0s):
    '''
    Read thermal degrees of freedom from a CSV file.
    '''
    dirr = dir0 + 'T_gs.csv'
    if dir0 == '':
        dirr = COSMOGW_HOME + '/resources/cosmology/T_gs.csv'
    df = pd.read_csv(dirr)
    Ts = np.array(df['T [GeV]'])
    gs = np.array(df['g_*']) if s == 0 else np.array(df['gS'])
    T = T.to(u.GeV)
    g = np.interp(T.value, np.sort(Ts), np.sort(gs))
    T = check_temperature_MeV(T, func='thermal_g')
    if not isinstance(T.value, (list, tuple, np.ndarray)):
        if T.value < 0.1:
            g = g0 if s == 0 else g0s
    else:
        if s == 0:
            g[T.value < 0.1] = g0
        if s == 1:
            g[T.value < 0.1] = g0s
    return g


def _thermal_g_piecewise(T, s, g0, g0s):
    '''
    Compute thermal degrees of freedom using a piecewise function.
    '''
    T = check_temperature_MeV(T, func='thermal_g')
    T = T.value
    if not isinstance(T, (list, tuple, np.ndarray)):
        T = [T]
    g = np.array([_piecewise_g_value(t, s, g0, g0s) for t in T])
    if len(g) == 1:
        return g[0]
    return g


def _piecewise_g_value(T, s, g0, g0s):
    thresholds = [
        (0.1, g0 if s == 0 else g0s),
        (0.5, 7.25),
        (100, 10.75),
        (150, 17.25),
        (1e3, 61.75),
        (4e3, 75.75),
        (8e4, 86.25),
        (1.7e5, 96.25),
    ]
    for threshold, value in thresholds:
        if T < threshold:
            return value
    return 106.75


def RD_dofs(dir0='', Neff=Neff_ref):
    """
    Compute relativistic and adiabatic dofs during the RD era.

    Parameters
    ----------
    dir0 : str, optional
        Directory where the file of dof is stored
        ('resources/cosmology/' directory by default)
    Neff : float, optional
        Effective number of neutrino species (default is 3)

    Returns
    -------
    T : ndarray
        Array of temperatures within RD era
    as_T : ndarray
        Array of scale factors
    gs : ndarray
        Array of relativistic dofs
    gS : ndarray
        Array of adiabatic dofs

    References
    ----------
    See values_0 and thermal_g.
    """
    T = np.logspace(-4, 8, 1000) * u.MeV
    gS = thermal_g(T=T, s=1, file=True, dir0=dir0)
    gs = thermal_g(T=T, s=0, file=True, dir0=dir0)
    g0, g0s, T0, _ = values_0(neut=True, Neff=Neff)
    as_T = (T0.to(u.MeV) / T) * (g0s / gs) ** (1 / 3)
    return T, as_T, gs, gS


def dofs_vs_a(a, dir0='', Neff=Neff_ref):
    """
    Compute relativistic and adiabatic dofs for scale factors.

    Parameters
    ----------
    a : ndarray
        Array of scale factors
    dir0 : str, optional
        Directory where the file of dof is stored
        ('/cosmology/' directory by default)
    Neff : float, optional
        Effective number of neutrino species (default is 3)

    Returns
    -------
    gs : ndarray
        Array of relativistic dofs
    gS : ndarray
        Array of adiabatic dofs

    References
    ----------
    See values_0 and thermal_g.
    """
    _, as_T, gs, gS = RD_dofs(dir0=dir0, Neff=Neff)
    inds = np.argsort(as_T)
    as2 = as_T[inds]
    gs = gs[inds]
    gS = gS[inds]
    gs = np.interp(a, as2, gs)
    gS = np.interp(a, as2, gS)
    return gs, gS


def Hs_from_a(a, dir0='', Neff=Neff_ref):

    r"""
    Compute Hubble rate during RD era from scale factor,
    assuming adiabatic expansion:

    .. math ::

        a^3 g_S T^3 = {\rm \ constant}.

    Parameters
    ----------
    a : ndarray
        Array of scale factors
    dir0 : str, optional
        Directory where the file of dof is stored
        (resources/cosmology/ default).
    Neff : float, optional
        Effective number of neutrino species (default is 3)

    Returns
    -------
    Hs : ndarray
        Array of Hubble rates during RD era
        Hs   -- Hubble rate during RD era

    References
    ----------
    See values_0 and thermal_g.
    """

    # compute the dofs
    gs, _ = dofs_vs_a(a, Neff=Neff, dir0=dir0)
    # compute the temperature scale from adiabatic expansions
    _, g0s, T0, _ = values_0(neut=True, Neff=Neff)
    T = T0.to(u.MeV)/a*(g0s/gs)**(1./3.)
    # get the Hubble rate
    Hs = Hs_val(T=T, g=gs)

    return Hs


def Omega_rad_dof(a, dir0='', Neff=Neff_ref):
    """
    Compute radiation energy density ratio due to varying dofs.

    Parameters
    ----------
    a : ndarray
        Array of scale factors
    dir0 : str, optional
        Directory where the file of dof is stored
        (resources/cosmology/ default)
    Neff : float, optional
        Effective number of neutrino species (default is 3)

    Returns
    -------
    Om_rat_dof : ndarray
        Ratio of radiation energy density due to accounting for
        T depending dofs

    References
    ----------
    He:2022qcs, eq.A.2
    """
    gs, gS = dofs_vs_a(a, dir0=dir0, Neff=Neff)
    g0, g0s, T0, H0 = values_0(neut=True, Neff=Neff)
    Om_rat_dof = (gs / g0) * (gS / g0s) ** (-4 / 3)
    return Om_rat_dof


def Omega_vs_a(a, dir0='', a0=1, h0=h0_ref, OmL0=OmL0_ref,
               dofs=True, Neff=Neff_ref):
    r"""
    Compute energy density ratios as a function of scale factor
    for a universe composed by matter, radiation, and dark energy:

    .. math ::

        \Omega (a) = \Omega_{R}^0 \times a^{-4} + \Omega_{M}^0
        \times a^{-3} + \Omega_{L}^0

    Parameters
    ----------
    a : ndarray
        Array of scale factors.
    dir0 : str, optional
        Directory for dof file (default resources/cosmology/).
    a0 : float, optional
        Present scale factor (default 1).
    h0 : float, optional
        Present Hubble rate scaling (default 0.6732).
    OmL0 : float, optional
        Present dark energy content (default 0.6841).
    dofs : bool, optional
        If True, compensate radiation energy density using dofs.
    Neff : float, optional
        Effective number of neutrino species (default 3).

    Returns
    -------
    Om_tot : ndarray
        Total energy density (normalized).
    Om_rad : ndarray
        Radiation energy density (normalized).
    Om_mat : ndarray
        Matter energy density (normalized).

    References
    ----------
    He:2022qcs, eq.A.2
    """
    g0, g0s, T0, H0, rho_rad0, OmR0 = values_0(
        h0=h0, neut=True, Neff=Neff, ret_rad=True
    )
    if dofs:
        Om_rat_dof = Omega_rad_dof(a, dir0=dir0, Neff=Neff)
        OmR0 = OmR0 * Om_rat_dof
    OmM0 = 1 - OmL0 - OmR0
    Om_mat = (a / a0) ** (-3) * OmM0
    Om_rad = (a / a0) ** (-4) * OmR0
    Om_tot = OmL0 + Om_rad + Om_mat
    return Om_tot, Om_rad, Om_mat


def friedmann(a, dir0='', a0=1, h0=h0_ref, OmL0=OmL0_ref,
              dofs=True, Neff=Neff_ref):
    """
    Use Friedmann equations to compute EOS and time derivatives.

    Parameters
    ----------
    a : ndarray
        Array of scale factors.
    dir0 : str, optional
        Directory for dof file (default resources/cosmology/).
    a0 : float, optional
        Present scale factor (default 1).
    h0 : float, optional
        Present Hubble rate scaling (default 0.6732).
    OmL0 : float, optional
        Present dark energy content (default 0.6841).
    dofs : bool, optional
        If True, compensate radiation energy density using dofs.
    Neff : float, optional
        Effective number of neutrino species (default 3).

    Returns
    -------
    w : float
        Equation of state (p = w rho).
    ad : float
        Cosmic time derivative of the scale factor.
    add : float
        Second cosmic time derivative of the scale factor.
    ap : float
        Conformal time derivative of the scale factor.
    app : float
        Second conformal time derivative of the scale factor.

    References
    ----------
    He:2022qcs, eqs. A.5-A.6
    """
    g0, g0s, T0, H0 = values_0(h0=h0, neut=True, Neff=Neff)
    Om_tot, Om_rad, _ = Omega_vs_a(
        a, a0=a0, h0=h0, OmL0=OmL0, dofs=dofs, dir0=dir0, Neff=Neff
    )
    w = (1 / 3 * Om_rad - OmL0) / Om_tot
    add = -.5 * Om_tot * H0 ** 2 * (a / a0) * (1 + 3 * w)
    ad = (a / a0) * np.sqrt(Om_tot) * H0
    app = .5 * a ** 3 * Om_tot * (1 - 3 * w) * H0 ** 2
    ap = ad * a
    return w, ad, add, ap, app


def friedmann_solver(a, dir0='', a0=1., h0=h0_ref, OmL0=OmL0_ref,
                     dofs=True, Neff=Neff_ref,
                     return_all=False, save=True, nm_fl=''):
    """
    Numerically solve Friedmann equations for a(eta) and a(t).

    A tutorial is available under cosmology/cosmology.ipynb

    Parameters
    ----------
    a : ndarray
        Array of scale factors.
    dir0 : str, optional
        Directory for dof file (default resources/cosmology/).
    a0 : float, optional
        Present scale factor (default 1).
    h0 : float, optional
        Present Hubble rate scaling (default 0.6732).
    OmL0 : float, optional
        Present dark energy content (default 0.6841).
    dofs : bool, optional
        If True, compensate radiation energy density using dofs.
    Neff : float, optional
        Effective number of neutrino species (default 3).
    return_all : bool, optional
        If True, return all variables used in the Friedmann solver.
    save : bool, optional
        If True, save the solutions in the file
        'friedmann/solution#nm_fl.csv' where #nm_fl is the input file name.
    nm_fl : str, optional
        File name suffix.

    Returns
    -------
    t : ndarray
        Cosmic time.
    eta : ndarray
        Conformal time.

    (other variables if return_all is True)

    References
    ----------
    He:2022qcs, appendix A.
    """
    Om_tot, Om_rad, Om_mat = Omega_vs_a(
        a, a0=a0, h0=h0, OmL0=OmL0, dir0=dir0, dofs=dofs, Neff=Neff
    )
    difft = np.zeros(len(a))
    diffeta = np.zeros(len(a))
    if a[0] > 1e-18:
        print('minimum a given is ', a[0])
        print('note that the solver assumes that a[0] corresponds to',
              'times t = eta = 0, so make sure to use an array of a with',
              'small enough values (a[0] ~ 1e-20)')
    difft[0] = 0
    diffeta[0] = 0
    print('Entering Friedmann solver')
    for i in range(1, len(a)):
        aas = np.logspace(np.log10(a[0]), np.log10(a[i]), 10000)
        Om_as = np.interp(aas, a, Om_tot)
        fft = 1 / aas / np.sqrt(Om_as)
        ffeta = fft / aas
        try:
            difft[i] = np.trapezoid(fft, aas)
            diffeta[i] = np.trapezoid(ffeta, aas)
        except Exception:
            difft[i] = np.trapz(fft, aas)
            diffeta[i] = np.trapz(ffeta, aas)
    g0, g0s, T0, H0 = values_0(h0=h0)
    t = difft / H0
    eta = diffeta / H0
    w, ad, add, ap, app = friedmann(
        a, dir0=dir0, h0=h0, OmL0=OmL0, dofs=dofs, Neff=Neff
    )
    if save:
        df = pd.DataFrame({
            'a': a, 't': t, 'eta': eta,
            'ap/a': ap / a, 'app/a': app / a
        })
        df.to_csv('friedmann/solution' + nm_fl + '.csv')
        with open('friedmann/README' + nm_fl + '.txt', 'w') as f:
            f.write("The file solution" + nm_fl + ".csv contains the ")
            f.write(
                "solutions from the Friedmann solver using the "
                "parameters:\n"
            )
            f.write("a0 = %.4e, h0 = %.4f, OmL0 = %.4f, Neff = %.4f \n"
                    % (a0, h0, OmL0, Neff))
            f.write("The solution contains 'a', 't', 'eta', 'ap/a' and 'app/a'")
        print(
            'The results are saved in the file friedmann/solution'
            + nm_fl + '.csv'
        )
        print('The input parameters used are stored in friedmann/README'
              + nm_fl + '.txt')
    print('Leaving Friedmann solver')
    if return_all:
        return t, eta, Om_tot, Om_rad, Om_mat, w, ad, add, ap, app
    else:
        return t, eta


def normalized_variables(a, eta, ap_a, app_a, dir0='', T=Tref, h0=h0_ref):
    """
    Compute normalized variables for GW generation initial time,
    which are required to be used in the Pencil Code.

    A tutorial is available under cosmology/cosmology_PC.ipynb

    Parameters
    ----------
    a : ndarray
        Scale factors, normalized to present-time a_0 = 1.
    eta : ndarray
        Conformal times, normalized to present-time a_0 = 1.
    ap_a : ndarray
        Conformal Hubble time a'/a, normalized to present-time a_0 = 1.
    app_a : ndarray
        a''/a, normalized to present-time a_0 = 1.
    dir0 : str, optional
        Directory for dof file (default resources/cosmology/).
    T : astropy.units.Quantity, optional
        Temperature scale (default 100 GeV).
    h0 : float, optional
        Present Hubble rate scaling (default 0.6732).

    Returns
    -------
    a_n : ndarray
        Normalized scale factor a/a_*.
    eta_n : ndarray
        Normalized conformal time eta/eta_*.
    HH_n : ndarray
        Normalized conformal Hubble rate H/H_*.
    app_a_n : ndarray
        Normalized second conformal time derivative of a.
    Omega : ndarray
        Ratio of total energy to present-time critical energy density.
    w : ndarray
        Equation of state.
    eta_n_0 : float
        Normalized conformal present time.
    aEQ_n : float
        Normalized equipartition scale factor.
    aL_n : float
        Normalized dark energy domination scale factor.
    a_acc_n : float
        Normalized scale factor when acceleration starts.
    eta_n_EQ : float
        Normalized conformal time at equipartition.
    eta_n_L : float
        Normalized conformal time at dark energy domination.
    eta_n_acc : float
        Normalized conformal time when acceleration starts.

    References
    ----------
    He:2022qcs, appendix A.
    RoperPol:2018sap.
    """
    g = thermal_g(T=T, s=0, dir0=dir0)
    gS = thermal_g(T=T, s=1, dir0=dir0)
    ast = as_a0_rat(T=T, g=gS)
    Hs = Hs_val(T=T, g=g)
    a_n = a / ast
    a0 = 1 / ast
    eta_ast = np.interp(ast, a, eta)
    eta_n = eta / eta_ast
    eta_n_0 = np.interp(a0, a_n, eta_n)
    HH_n = ap_a / Hs / ast
    app_a_n = app_a / Hs ** 2 / ast ** 2
    H0 = h0 * H0_ref
    Omega = (HH_n.value * Hs) ** 2 / a_n ** 2 / H0 ** 2
    w = 1 / 3 * (1 - app_a_n * 2 / HH_n ** 2)
    inds = np.argsort(w)
    aEQ_n = np.interp(1 / 6, w[inds], a_n[inds])
    aL_n = np.interp(-.5, w[inds], a_n[inds])
    a_acc_n = np.interp(-1 / 3, w[inds], a_n[inds])
    eta_n_EQ = np.interp(aEQ_n, a_n, eta_n)
    eta_n_L = np.interp(aL_n, a_n, eta_n)
    eta_n_acc = np.interp(a_acc_n, a_n, eta_n)
    return (
        a_n, eta_n, HH_n, app_a_n, Omega, w, eta_n_0,
        aEQ_n, aL_n, a_acc_n, eta_n_EQ, eta_n_L, eta_n_acc
    )


def ratio_app_a_n_factor(a, dir0='', a0=1, h0=h0_ref, OmL0=OmL0_ref,
                         dofs=True, Neff=Neff_ref):
    """
    Compute ratio of a''/a (normalized) to conformal Hubble rate H (normalized)
    times a_*/a_0 during RD era.

    Parameters
    ----------
    a : ndarray
        Array of scale factors.
    dir0 : str, optional
        Directory for dof file (default resources/cosmology/).
    a0 : float, optional
        Present scale factor (default 1).
    h0 : float, optional
        Present Hubble rate scaling (default 0.6732).
    OmL0 : float, optional
        Present dark energy content (default 0.6841).
    dofs : bool, optional
        If True, compensate radiation energy density using dofs.
    Neff : float, optional
        Effective number of neutrino species (default 3).


    Returns
    -------
    factor : ndarray
        Ratio of a''/a to conformal Hubble rate H times a_*/a_0.

    References
    ----------
    He:2022qcs, eq. 3.11
    """
    g0, g0s, T0, H0, rho_rad0, OmR0 = values_0(
        h0=h0, neut=True, Neff=Neff, ret_rad=True
    )
    OmM0 = 1 - OmL0 - OmR0
    Om_rat_dof = Omega_rad_dof(a, Neff=Neff, dir0=dir0)
    factor = .5 * OmM0 / OmR0 / Om_rat_dof
    return factor


def norm_variables_cut(eta_n, HH_n, a_n, Omega, Omega_mat, eta_n_0, dir0='',
                       T=Tref, OmM0=OmM0_ref, h0=h0_ref):
    """
    Cut normalized variables between initial time and present time.

    Parameters
    ----------
    eta_n : ndarray
        Normalized conformal time eta/eta_*.
    HH_n : ndarray
        Normalized conformal Hubble rate H/H_*.
    a_n : ndarray
        Normalized scale factor a/a_*.
    Omega : ndarray
        Ratio of total energy to present-time critical energy density.
    Omega_mat : ndarray
        Matter energy density (normalized).
    eta_n_0 : float
        Normalized conformal present time.
    Hs : float
        Hubble rate at the initial time.
    ast : float
        Scale factor at the initial time.
    dir0 : str, optional
        Directory for dof file (default resources/cosmology/).
    T : float, optional
        Temperature scale at the initial time in energy units
        (default is 100 GeV).
    OmM0 : float, optional
        Present-time content of matter (default is 0.3159).
    h0 : float, optional
        Present-time value of the Hubble rate H0 = h0 x 100 km/s/Mpc
        (default is 67.32 km/s/Mpc based on CMB observations).

    Returns
    -------
    eta_nn : ndarray
        Normalized conformal time.
    HH_nn : ndarray
        Normalized conformal Hubble rate.
    a_nn : ndarray
        Normalized scale factor.
    Omega_nn : ndarray
        Ratio of total energy to present-time critical energy density.
    Omega_mat_nn : ndarray
        Matter energy density (normalized).
    app_nn : ndarray
        Second time derivative of the scale factor.
    w_nn : ndarray
        Equation of state p/rho.
    """
    H0 = h0 * H0_ref
    g = thermal_g(T=T, s=0, dir0=dir0)
    gS = thermal_g(T=T, s=1, dir0=dir0)
    ast = as_a0_rat(T=T, g=gS)
    Hs = Hs_val(T=T, g=g)
    inds = np.where(eta_n > 1)[0]
    inds2 = np.where(eta_n[inds] < eta_n_0)[0]
    eta_nn = cut_var(eta_n, 1, eta_n_0, inds, inds2)
    HH_n0 = np.interp(1, eta_n, HH_n.value)
    HH_nn = cut_var(HH_n.value, HH_n0, H0 / Hs / ast, inds, inds2)
    a_nn = cut_var(a_n, 1, 1 / ast, inds, inds2)
    Omega_nn = cut_var(Omega, (Hs / H0) ** 2, 1, inds, inds2)
    Omega_mat_nn = cut_var(Omega_mat, OmM0 * ast ** (-3), OmM0, inds, inds2)
    Omega_rad_nn = Omega_nn - Omega_mat_nn - (1 - OmM0)
    w_nn = (1 / 3 * Omega_rad_nn - (1 - OmM0)) / Omega_nn
    app_nn = .5 * HH_nn ** 2 * (1 - 3 * w_nn)
    return eta_nn, HH_nn, a_nn, Omega_nn, Omega_mat_nn, app_nn, w_nn


def cut_var(x, x0, xf, inds, inds2):
    """
    Cut variable x using indices and add initial/final values.
    Used in norm_variables_cut function.

    Parameters
    ----------
    x : ndarray
        Input array.
    x0 : float
        Initial value to add.
    xf : float
        Final value to add.
    inds : ndarray
        Indices for first cut.
    inds2 : ndarray
        Indices for second cut.

    Returns
    -------
    y : ndarray
        Cut and extended array.
    """
    y = x[inds][inds2]
    y = np.append(x0, y)
    y = np.append(y, xf)
    return y
