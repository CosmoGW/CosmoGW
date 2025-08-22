"""
GW_back.py is a Python routine that contains functions relevant for
cosmological stochastic gravitational wave backgrounds (SGWB).

Adapted from the original cosmoGW in GW_turbulence
(https://github.com/AlbertoRoper/GW_turbulence),
created in Dec. 2021

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/GW_back.py

.. note::
   For full documentation, visit `Read the Docs
   <https://cosmogw-manual.readthedocs.io/en/latest/GW_back.html>`_.

To use it, first install `cosmoGW <https://pypi.org/project/cosmoGW>`_::

    pip install cosmoGW

Author
------
- **Alberto Roper Pol** \
(`alberto.roperpol@unige.ch <mailto:alberto.roperpol@unige.ch>`_)

Dates
-----
- Created: **01/12/2021** (*GW_turbulence*)
- Updated: **21/08/2025** \
    (release **cosmoGW 1.0**: https://pypi.org/project/cosmoGW)

References
----------
- [**Maggiore:1999vm**]: M. Maggiore, "*Gravitational wave experiments and\
early universe cosmology,*" Phys.Rept. **331** (2000) 283-367,\
`arXiv:gr-qc/9909001 <https://arxiv.org/abs/gr-qc/9909001>`_.

- [**RoperPol:2018sap**]: A. Roper Pol, A. Brandenburg, T. Kahniashvili,\
A. Kosowsky, S. Mandal, "*The timestep constraint in solving the\
gravitational wave equations sourced by hydromagnetic turbulence,*"\
Geophys. Astrophys. Fluid Dynamics **114**, 1, 130 (2020),\
`arXiv:1807.05479 <https://arxiv.org/abs/1807.05479>`_.

- [**RoperPol:2021xnd**]: A. Roper Pol, S. Mandal, A. Brandenburg,\
T. Kahniashvili, "*Polarization of gravitational waves from helical\
MHD turbulent sources,*" JCAP **04** (2022), 019,\
`arXiv:2107.05356 <https://arxiv.org/abs/2107.05356>`_.

- [**RoperPol:2022iel**]: A. Roper Pol, C. Caprini, A. Neronov, D. Semikoz,\
"*The gravitational wave signal from primordial magnetic fields in the\
Pulsar Timing Array frequency band,*" Phys. Rev. D **105**, 123502 (2022),\
`arXiv:2201.05630 <https://arxiv.org/abs/2201.05630>`_.
"""

import astropy.units as u
import numpy as np
from cosmoGW import cosmology
from cosmoGW.utils import H0_ref, gref, Neff_ref, Tref, check_temperature_MeV


def fac_hc_OmGW(d=1, h0=1.):

    r"""
    Returns the factor to transform the strain function :math:`h_c (f)` to
    the GW energy density :math:`\Omega_{\rm GW}(f)` away from the source.

    Parameters
    ----------
    d : int
        Option to give the factor to convert from energy density to strain if
        set to -1 (default 1).
    h0 : float
        Parameterizes the uncertainties (Hubble tension) in the value of the
        Hubble rate (default 1).

    Returns
    -------
    fac : float
        Factor to convert from the strain function hc(f) to the GW energy
        density OmGW(f) in frequency units (Hz).

    Reference
    ---------
    Maggiore:1999vm, eq. 17
    """

    # compute values at present day
    fac = H0_ref*h0*np.sqrt(3/2)/np.pi
    if d == -1:
        fac = 1/fac**2

    return fac


def _check_frequency(f, func=''):

    """
    Checks and converts the input frequency to Hz using astropy units.

    Parameters
    ----------
    f : astropy.units.Quantity
        Input frequency.
    func : str, optional
        Name of the calling function for error messages.

    Returns
    -------
    f : astropy.units.Quantity
        Frequency converted to Hz.
    """

    try:
        f = f.to(u.Hz)
    except Exception:
        print('Error: the input frequency in ', func,
              ' should be given in frequency units',
              ' using astropy.units, setting f to 1/year')
        f = 1/u.yr
        f = f.to(u.Hz)

    return f


def hc_OmGW(f, OmGW, d=1, h0=1.):

    r"""
    Transforms the GW energy density :math:`\Omega_{\rm GW}(f)` to the
    characteristic strain spectrum function :math:`h_c(f)` away from the source.

    .. note::

        Careful with the different notations (can be confusing!!), see below.

    .. math::

        \Omega_{\rm GW}(f) = \frac{2 \pi^2}{3 H_0^2} f^2 h_c^2(f)

    Parameters
    ----------
    f : array_like
        Frequency array (in units of frequency, e.g. Hz).
    OmGW : array_like
        GW energy density spectrum OmGW(f).
    d : int
        Option to convert from energy density to strain if set to -1
        (default 1).
    h0 : float
        Parameterizes the uncertainties (Hubble tension) in the value of the
        Hubble rate (default 1).

    Returns
    -------
    hc : array_like
        Strain spectrum.

    References
    ----------
    Maggiore:1999vm, eq. 17

    Maggiore defines :math:`S_h(f)` in eq. B12 such that

    .. math::

        h_c^2(f) = 2 f S_h^{\mathrm{Mag}}(f)

    Note that this is different than the notation in RoperPol:2021xnd,
    eq. B.18, used in interferometry.py, where

    .. math::

        h_c^2(f) = f S_h^{\pm}(f)

    such that

    .. math::

        S_h^{\pm}(f) = 2 S_h^{\mathrm{Mag}}(f)

    Hence, from :math:`h_c^2(f)` we can compute

    .. math::

        S_h^{\mathrm{Mag}}(f) = \frac{h_c^2(f)}{2f}
        \qquad
        S_h^{\pm}(f) = \frac{h_c^2(f)}{f}
    """

    f = _check_frequency(f, func='hc_OmGW')
    fac = fac_hc_OmGW(d=d, h0=h0)
    hc = fac / f * np.sqrt(OmGW)
    if d == -1:
        hc = fac * f ** 2 * OmGW ** 2
    return hc


def _check_Sf(Sf, func=''):

    """
    Checks and converts the input spectral density to 1/Hz^3 using
    astropy units.

    Parameters
    ----------
    Sf : astropy.units.Quantity
        Input spectral density.
    func : str, optional
        Name of the calling function for error messages.

    Returns
    -------
    Sf : astropy.units.Quantity
        Spectral density converted to 1/Hz^3.
    """

    try:
        Sf = Sf.to(1/u.Hz**3)
    except Exception:
        print(
            'Error: the input spectral density in', func,
            'should be given in 1/frequency^3 using astropy.units,',
            'setting Sf to 1/Hz^3'
        )
        Sf = 1. / u.Hz ** 3
    return Sf


def hc_Sf(f, Sf, d=1):

    """
    Transforms the power spectral density :math:`S_f(f)` to the
    characteristic strain spectrum function :math:`h_c(f)`.

    Parameters
    ----------
    f : array_like
        Frequency array (in units of frequency, e.g. Hz).
    Sf : array_like
        Power spectral density :math:`S_f(f)` (in units of 1/Hz^3).
    d : int
        Option to convert from strain to power spectral density if set to -1
        (default 1).

    Returns
    -------
    hc : array_like
        Strain spectrum.

    Reference
    ---------
    RoperPol:2022iel, eq. 42
    """
    f = _check_frequency(f, func='hc_Sf')
    Sf = _check_Sf(Sf, func='hc_Sf')
    hc = np.sqrt(12 * np.pi ** 2 * Sf * f ** 3)
    if d == -1:
        hc = Sf ** 2 / 12 / np.pi ** 2 / f ** 3
    return hc


def Omega_A(A=1., fref=0, beta=0, h0=1.):

    r"""
    Returns the amplitude of the SGWB energy density spectrum, expressed as a
    power law (PL), given the amplitude A of the characteristic strain, also
    expressed as a PL.

    A is always given for the reference frequency of :math:`f_{\rm yr} =
    1/(1 {\rm yr})` and is used in
    the common process reported by PTA collaborations.

    The GW energy density and characteristic amplitude can be expressed as:

    .. math::
        \Omega_{\rm GW} = \Omega_{\rm ref} * (f/f_{\rm ref})^\beta
        h_c = A * (f/f_{\rm yr})^\alpha

    Parameters
    ----------
    A : float
        Amplitude of the characteristic strain PL using 1yr as the reference
        frequency.
    fref : float
        Reference frequency used for the PL expression of the GW background
        given in units of frequency (default 1 yr^(-1)).
    beta : float
        Slope of the PL.
    h0 : float
        Parameterizes the uncertainties (Hubble tension) in the value of the
        Hubble rate (default 1).

    Returns
    -------
    Omref : float
        Amplitude of the GW energy density PL.

    Reference
    ---------
    RoperPol:2022iel, eq. 44
    """
    fac = fac_hc_OmGW(d=-1, h0=h0)
    fyr = (1. / u.yr).to(u.Hz)
    Omref = fac * fyr ** 2 * A ** 2
    if fref != 0:
        fref = _check_frequency(fref, func='Omega_A')
        Omref *= (fref.value / fyr.value) ** beta
    return Omref


def shift_onlyOmGW_today(OmGW, g=gref, gS=0., d=1, h0=1.,
                         Neff=Neff_ref):

    """
    Shifts the GW energy density spectrum from the time of generation to the
    present time (assumed to be within the RD era).

    Parameters
    ----------
    OmGW : array_like
        GW energy density spectrum per logarithmic interval (normalized by the
        radiation energy density).
    g : float
        Number of relativistic degrees of freedom (dof) at the time of
        generation (default is 100).
    gS : float
        Number of adiabatic dof (default is gS = g).
    d : int
        Option to reverse the transformation if set to -1 (default 1).
    h0 : float
        Parameterizes the uncertainties (Hubble tension) in the value of the
        Hubble rate (default 1).
    Neff : float
        Effective number of neutrino species (default is 3).

    Returns
    -------
    OmGW0 : array_like
        Shifted spectrum OmGW to present time.

    Reference
    ---------
    RoperPol:2022iel, eq. 27
    """
    Hs_f = cosmology.Hs_fact() * u.MeV ** 2
    as_f = cosmology.as_fact(Neff=Neff) / u.MeV
    if gS == 0:
        gS = g
    OmGW_f = (
        Hs_f ** 2 / H0_ref ** 2 / h0 ** 2 * as_f ** 4 *
        g / gS ** (4. / 3.)
    )
    if d == 1:
        OmGW0 = OmGW * OmGW_f
    if d == -1:
        OmGW0 = OmGW / OmGW_f
    return OmGW0


def shift_frequency_today(k, g=gref, gS=0., T=Tref, d=1,
                          kk=True, Neff=Neff_ref):
    r"""
    Transforms the normalized wave number at the time of generation by the
    Hubble rate :math:`H_\ast` to the present time frequency.

    Parameters
    ----------
    k : array_like
        Array of wave numbers (normalized by the Hubble scale).
    g : float
        Number of relativistic degrees of freedom (dof) at the time of
        generation (default is 100).
    gS : float
        Number of adiabatic dof (default is gS = g).
    T : float
        Temperature scale at the time of generation in energy units
        (convertible to MeV) (default is 100 GeV).
    d : int
        Option to reverse the transformation if set to -1 (default 1).
    kk : bool
        If True, kf corresponds to :math:`k_\ast {\cal H}_\ast`, otherwise
        refers to the length in terms of the Hubble size
        :math:`{\cal H}_\ast l_\ast = 2 \pi {\cal H}_\ast/k_\ast`.
    Neff : float
        Effective number of neutrino species (default is 3).

    Returns
    -------
    f : array_like
        Shifted wave number to frequency as a present time observable (in Hz).

    Reference
    ---------
    RoperPol:2022iel, eq. 32
    """

    if gS == 0:
        gS = g
    HHs = cosmology.Hs_val(g=g, T=T) * cosmology.as_a0_rat(g=gS, T=T, Neff=Neff)
    if d == 1:
        if not kk:
            k = 2 * np.pi * k
        f = k * HHs / 2 / np.pi
    if d == -1:
        f = 2 * np.pi * k.to(u.Hz) / HHs
        if not kk:
            f = 2 * np.pi / f
    return f


def shift_OmGW_today(k, OmGW, g=gref, gS=0., T=Tref, d=1,
                     h0=1., kk=True, Neff=Neff_ref):

    r"""
    Shifts the GW energy density spectrum from the time of generation to the
    present time. Assumes the time of generation is within the radiation
    dominated era. Test.

    Parameters
    ----------
    k : array_like
        Array of wave numbers (normalized by the Hubble scale).
    OmGW : array_like
        GW energy density spectrum per logarithmic interval (normalized by the
        radiation energy density).
    g : float
        Number of relativistic degrees of freedom (dof) at the time of
        generation (default is 100).
    gS : float
        Number of adiabatic dof (default is gS = g).
    T : float
        Temperature scale at the time of generation in energy units
        (convertible to MeV) (default is 100 GeV).
    d : int
        Option to reverse the transformation if set to -1 (default 1).
    h0 : float
        Parameterizes the uncertainties (Hubble tension) in the value of the
        Hubble rate (default 1).
    kk : bool
        If True, kf corresponds to k_* HH_*, otherwise refers to the length in
        terms of the Hubble size :math:`{\cal H}_\ast l_\ast
        = 2 \pi {\cal H}_\ast/k_\ast`.
    Neff : float
        Effective number of neutrino species (default is 3).

    Returns
    -------
    f : array_like
        Shifted wave number to frequency as a present time observable (in Hz).
    OmGW0 : array_like
        Shifted spectrum OmGW to present time.

    Reference
    ---------
    See functions shift_onlyOmGW_today and shift_frequency_today.
    """

    # shift Omega_GW
    OmGW0 = shift_onlyOmGW_today(OmGW, g=g, gS=gS, d=d, h0=h0, Neff=Neff)
    # shift frequency
    f = shift_frequency_today(k, g=g, gS=gS, T=T, d=d, kk=kk, Neff=Neff)
    return f, OmGW0


def shift_hc_today(k, hc, g=gref, gS=0., T=Tref,
                   d=1, kk=True, Neff=Neff_ref):

    r"""
    Shifts the characteristic amplitude spectrum from the time of generation to
    the present time.

    Assumes the time of generation is within the radiation dominated era.

    Parameters
    ----------
    k : array_like
        Array of wave numbers (normalized by the Hubble scale).
    hc : array_like
        Spectrum of GW characteristic amplitude per logarithmic interval.
    g : float
        Number of relativistic degrees of freedom (dof) at the time of
        generation (default is 100).
    gS : float
        Number of adiabatic dof (default is gS = g).
    T : float
        Temperature scale at the time of generation in energy units
        (convertible to MeV) (default is 100 GeV).
    d : int
        Option to reverse the transformation if set to -1 (default 1).
    kk : bool
        If True, kf corresponds to :math:`k_* \mathcal{H}_*`.
        Otherwise, refers to the length in terms of the Hubble size,
        i.e. :math:`\mathcal{H}_* l_* = 2\pi {\cal H}_\ast/k_*`.
    Neff : float
        Effective number of neutrino species (default is 3).

    Returns
    -------
    f : array_like
        Shifted wave number to frequency as a present time observable (in Hz).
    hc0 : array_like
        Shifted :math:`h_c(f)` spectrum to present time.

    Reference
    ---------
    RoperPol:2018sap, eq. B.12
    """

    as_f = cosmology.as_fact(Neff=Neff)
    T = check_temperature_MeV(T, func='GW_back.shift_hc_today')
    hc0 = hc * as_f * g ** (-1 / 3) / T
    if d == -1:
        hc0 = hc / as_f / g ** (-1 / 3) * T
    f = shift_frequency_today(k, g=g, gS=gS, T=T, d=d, kk=kk, Neff=Neff)
    return f, hc0
