"""
cosmoGW.py is a Python routine that contains functions relevant for
cosmological stochastic gravitational wave backgrounds (SGWB).

Adapted from the original cosmoGW in GW_turbulence
(https://github.com/AlbertoRoper/GW_turbulence),
created in Dec. 2021

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/cosmoGW.py

Author:  Alberto Roper Pol
Created: 01/12/2021 (GW_turbulence)
Updated: 13/03/2025 (release cosmoGW 1.0: https://pypi.org/project/cosmoGW)

Main references used in this library are:

Maggiore:1999vm  - M. Maggiore, "Gravitational wave experiments and
early universe cosmology," Phys.Rept. 331 (2000) 283-367,
arXiv:gr-qc/9909001.

RoperPol:2018sap - A. Roper Pol, A. Brandenburg, T. Kahniashvili,
A. Kosowsky, S. Mandal, "The timestep constraint in solving the
gravitational wave equations sourced by hydromagnetic turbulence,"
Geophys. Astrophys. Fluid Dynamics 114, 1, 130 (2020),
arXiv:1807.05479.

RoperPol:2021xnd - A. Roper Pol, S. Mandal, A. Brandenburg,
T. Kahniashvili, "Polarization of gravitational waves from helical
MHD turbulent sources," JCAP (2022) 04, 019, arXiv:2107.05356.

RoperPol:2022iel - A. Roper Pol, C. Caprini, A. Neronov, D. Semikoz,
"The gravitational wave signal from primordial magnetic fields in the
Pulsar Timing Array frequency band," Phys. Rev. D 105, 123502 (2022),
arXiv:2201.05630.
"""

import astropy.units as u
import numpy as np

def find_path():

    ## find the directory where cosmoGW is installed within sys.path
    import sys
    import os
    found = False
    paths = sys.path
    for path in paths:
      subdirs = os.walk(path)
      subdirs = list(subdirs)
      for j in subdirs:
        if not 'test' in j[0] and not 'env' in j[0]:
            if 'cosmoGW' in j[0]:
                pth = j[0]
                found = True
                break
      if found: break
    return pth + '/'

HOME = find_path()

import cosmoGW.cosmology as co

########################## SGWB at present time ##########################

def fac_hc_OmGW(d=1, h0=1.):

    """
    Function that returns the factor to transform the strain function
    hc(f) to the GW energy density OmGW (f) away from the source.

    Arguments:
        d  -- option to give the factor to convert from energy density to
             strain if set to -1 (default 1)
        h0 -- parameterizes the uncertainties (Hubble tension) in the value
              of the Hubble rate (default 1)

    Returns:
        fac -- factor to convert from the strain function hc(f) to the GW
               energy density OmGW (f) in frequency units (Hz)

    Reference: Maggiore:1999vm, eq. 17
    """

    # compute values at present day
    fac = co.H0_ref*h0*np.sqrt(3/2)/np.pi
    if d == -1: fac = 1/fac**2

    return fac

def check_frequency(f, func=''):

    try:
        f = f.to(u.Hz)
    except:
        print('Error: the input frequency in ', func,
              ' should be given in frequency units',
              ' using astropy.units, setting f to 1/year')
        f = 1/u.yr
        f = f.to(u.Hz)

    return f

def hc_OmGW(f, OmGW, d=1, h0=1.):

    """
    Function that transforms the  GW energy density OmGW (f) to the
    characteristic strain spectrum function hc(f) away from the source.
    Careful with the different notations (can be confusing!!), see below.

    Arguments:
        f    -- frequency array (in units of frequency, e.g. Hz)
        OmGW -- GW energy density spectrum OmGW (f)
        d    -- option to convert from energy density to strain if set
                to -1 (default 1)
        h0   -- parameterizes the uncertainties (Hubble tension) in the value
                of the Hubble rate (default 1)

    Returns:
        hc -- strain spectrum

    References (comment on notations):

    Main reference here is Maggiore:1999vm, eq. 17:

    OmGW(f) = (2 pi^2)/(3 H0^2) f^2 hc^2 (f).

    hc2 (f) is then defined as the spectrum in log(f) of h+^2 + hx^2 and
    Maggiore defines Sh(f) in eq. B12 such that hc^2 (f) = 2 f Sh_Mag (f).

    Note that this is different than the notation in  RoperPol:2021xnd,
    eq. B.18, used in interferometry.py, where hc^2 (f) = f Sh^pm (f),
    such that this is

    Sh^pm (f) = 2 Sh_Mag (f).

    Hence, from hc^2 (f) we can compute Sh_Mag (f) = hc^2 (f)/2f and
    Sh^pm (f) = hc^2 (f)/f.
    """

    f   = check_frequency(f, func='hc_OmGW')
    fac = fac_hc_OmGW(d=d, h0=h0)
    hc  = fac/f*np.sqrt(OmGW)
    if d==-1: hc = fac*f**2*OmGW**2

    return hc

def check_Sf(Sf, func=''):

    try:
        Sf = Sf.to(1/u.Hz**3)
    except:
        print('Error: the input spectral density in', func,
              'should be given in 1/frequency^3 using astropy.units,',
              ' setting Sf to 1/Hz^3')
        Sf = 1./u.Hz**3

    return Sf

def hc_Sf(f, Sf, d=1):

    """
    Function that transforms the power spectral density Sf (f) to the
    characteristic strain spectrum function hc(f).

    Arguments:
        f  -- frequency array (in units of frequency, e.g. Hz)
        Sf -- power spectral density Sf (f) (in units of 1/Hz^3)
        d  -- option to convert from strain to power spectral density if set
             to -1 (default 1)

    Returns:
        hc -- strain spectrum

    Reference: RoperPol:2022iel, eq. 42
    """

    f  = check_frequency(f, func='hc_Sf')
    Sf = check_Sf(Sf, func='hc_Sf')
    hc = np.sqrt(12*np.pi**2*Sf*f**3)
    if d==-1: hc = Sf**2/12/np.pi**2/f**3

    return hc

def Omega_A(A=1., fref=0, beta=0, h0=1.):

    """
    Function that returns the amplitude of the SGWB energy density
    spectrum, expressed as a power law (PL), given the amplitude A of the
    characteristic strain, also expressed as a PL.

    Note that A is always given for the reference frequency of 1/(1 year)
    and it is used in the common process reported by PTA collaborations.

    The GW energy density and characteristic amplitude can be expressed as:

    OmGW = Omref (f/fref)^beta
    hc = A (f/fyr)^alpha

    Arguments:
        A    -- amplitude of the characteristic strain PL using 1yr as the
                reference frequency
        fref -- reference frequency used for the PL expression of the
                GW background given in units of frequency (default
                1 yr^(-1))
        beta -- slope of the PL
        h0   -- parameterizes the uncertainties (Hubble tension) in the value
                of the Hubble rate (default 1)

    Returns:
        Omref -- amplitude of the GW energy density PL

    Reference: RoperPol:2022iel, eq. 44
    """

    fac = fac_hc_OmGW(d=-1, h0=h0)
    fyr = (1./u.yr).to(u.Hz)
    Omref = fac*fyr**2*A**2
    if fref != 0:
        fref = check_frequency(fref, func='Omega_A')
        Omref *= (fref.value/fyr.value)**beta

    return Omref

################ SGWB from the radiation-dominated era ################

def shift_onlyOmGW_today(OmGW, g=co.gref, gS=0., d=1, h0=1., Neff=co.Neff_ref):

    """
    Function that shifts the GW energy density spectrum from the time of
    generation to the present time (assumed to be within the RD era)

    Arguments:
        OmGW -- GW energy density spectrum per logarithmic interval
                (normalized by the radiation energy density)
        g    -- number of relativistic degrees of freedom (dof) at the time of generation
                (default is 100)
        gS   -- number of adiabatic dof (default is gS = g)
        d    -- option to reverse the transformation if set to -1, i.e.,
                to return OmGW(k) from the shifted to present time OmGW(f)
                (default 1)
        h0   -- parameterizes the uncertainties (Hubble tension) in the value
                of the Hubble rate (default 1)
        Neff -- effective number of neutrino species (default is 3)


    Returns:
        OmGW0 -- shifted spectrum OmGW to present time

    Reference: RoperPol:2022iel, eq. 27
    """

    Hs_f = co.Hs_fact()*u.MeV**2
    as_f = co.as_fact(Neff=Neff)/u.MeV
    # if gS is not specified, it is assumed to be equal to g*
    if gS==0: gS = g
    OmGW_f = Hs_f**2/co.H0_ref**2/h0**2*as_f**4*g/gS**(4./3.)
    if d==1:  OmGW0 = OmGW*OmGW_f
    if d==-1: OmGW0 = OmGW/OmGW_f

    return OmGW0

def shift_frequency_today(k, g=co.gref, gS=0., T=co.Tref, d=1, kk=True,
                          Neff=co.Neff_ref):

    """
    Function that transforms the normalized wave number at the time of
    generation by the Hubble rate H_* to the present time frequency.

    Arguments:
        k    -- array of wave numbers (normalized by the Hubble scale)
        g    -- number of relativistic degrees of freedom (dof) at the time of generation
                (default is 100)
        gS   -- number of adiabatic dof (default is gS = g)
        T    -- temperature scale at the time of generation in energy units
                (convertible to MeV) (default is 100 GeV)
        d    -- option to reverse the transformation if set to -1, i.e.,
                to return the normalized k from the frequency shifted to present
                time f (default 1)
        kk   -- if kk is True, then kf corresponds to k_* HH_*, otherwise it refers
                to the length in terms of the Hubble size HH_* l_* = 2 pi/(k_*/HH_*)
        Neff -- effective number of neutrino species (default is 3)

    Returns:
        f -- shifted wave number to frequency as a present time observable
             (in Hz)

    Reference: RoperPol:2022iel, eq. 32
    """

    # if gS is not specified, it is assumed to be equal to g*
    if gS==0: gS = g
    HHs = co.Hs_val(g=g, T=T)*co.as_a0_rat(g=gS, T=T, Neff=Neff)
    if d == 1:
        if kk == False: k = 2*np.pi*k
        f = k*HHs/2/np.pi
    if d==-1:
        f = 2*np.pi*k.to(u.Hz)/HHs
        if kk == False: f = 2*np.pi/f

    return f

def shift_OmGW_today(k, OmGW, g=co.gref, gS=0., T=co.Tref, d=1, h0=1.,
                     kk=True, Neff=co.Neff_ref):

    """
    Function that shifts the GW energy density spectrum from the time of
    generation to the present time. It assumes that the time of generation
    is within the radiation dominated era.

    Arguments:
        k    -- array of wave numbers (normalized by the Hubble scale)
        OmGW -- GW energy density spectrum per logarithmic interval
                (normalized by the radiation energy density)
        g    -- number of relativistic degrees of freedom (dof) at the time of generation
                (default is 100)
        gS   -- number of adiabatic dof (default is gS = g)
        T    -- temperature scale at the time of generation in energy units
                (convertible to MeV) (default is 100 GeV)
        d    -- option to reverse the transformation if set to -1, i.e.,
                to return the normalized k from the frequency shifted to present
                time f (default 1)
        h0   -- parameterizes the uncertainties (Hubble tension) in the value
                of the Hubble rate (default 1)
        kk   -- if kk is True, then kf corresponds to k_* HH_*, otherwise it refers
                to the length in terms of the Hubble size HH_* l_* = 2 pi/(k_*/HH_*)
        Neff -- effective number of neutrino species (default is 3)

    Returns:
        f     -- shifted wave number to frequency as a present time observable
                 (in Hz)
        OmGW0 -- shifted spectrum OmGW to present time

    Reference: see functions shift_onlyOmGW_today and shift_frequency_today
    """

    # shift Omega_GW
    OmGW0 = shift_onlyOmGW_today(OmGW, g=g, gS=gS, d=d, h0=h0, Neff=Neff)
    # shift frequency
    f     = shift_frequency_today(k,   g=g, gS=gS, T=T, d=d, kk=kk, Neff=Neff)

    return f, OmGW0

def shift_hc_today(k, hc, g=co.gref, gS=0., T=co.Tref, d=1, kk=True, Neff=co.Neff_ref):

    """
    Function that shifts the characteristic amplitude spectrum from the time
    of generation to the present time.

    It assumes that the time of generation is within the radiation dominated
    era.

    Arguments:
        k    -- array of wave numbers (normalized by the Hubble scale)
        hc   -- spectrum of GW characteristic amplitude per logarithmic interval
        g    -- number of relativistic degrees of freedom (dof) at the time of generation
                (default is 100)
        gS   -- number of adiabatic dof (default is gS = g)
        T    -- temperature scale at the time of generation in energy units
                (convertible to MeV) (default is 100 GeV)
        d    -- option to reverse the transformation if set to -1, i.e.,
                to return the normalized k from the frequency shifted to present
                time f, and hc(k) from the shifted to present time hc0(f) (default 1)
        kk   -- if kk is True, then kf corresponds to k_* HH_*, otherwise it refers
                to the length in terms of the Hubble size HH_* l_* = 2 pi/(k_*/HH_*)
        Neff -- effective number of neutrino species (default is 3)

    Returns:
        f    -- shifted wave number to frequency as a present time observable
                (in Hz)
        hc0  -- shifted hc spectrum to present time

    Reference: RoperPol:2018sap, eq. B.12
    """

    as_f = co.as_fact(Neff=Neff)
    T    = co.check_temperature_MeV(T, func='cosmoGW.shift_hc_today')
    hc0  = hc*as_f*g**(-1/3)/T
    if d == -1: hc0 = hc/as_f/g**(-1/3)*T
    f    = shift_frequency_today(k, g=g, gS=gS, T=T, d=d, kk=kk, Neff=Neff)

    return f, hc0
