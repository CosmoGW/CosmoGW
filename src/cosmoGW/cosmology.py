"""
cosmology.py is a Python routine that contains functions relevant
for cosmological calculations, including a solver of Friedmann equations.

Adapted from the original cosmology in GW_turbulence
(https://github.com/AlbertoRoper/GW_turbulence),
created in Nov. 2022

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/cosmology.py

Author: Alberto Roper Pol
Created: 27/11/2022 (GW_turbulence)
Updated: 13/03/2025 (release cosmoGW 1.0: https://pypi.org/project/cosmoGW)

Main references are:

RoperPol:2018sap - A. Roper Pol, A. Brandenburg, T. Kahniashvili,
A. Kosowsky, S. Mandal, "The timestep constraint in solving the
gravitational wave equations sourced by hydromagnetic turbulence,"
Geophys. Astrophys. Fluid Dynamics 114, 1, 130 (2020),
arXiv:1807.05479.

RoperPol:2019wvy - A. Roper Pol, S. Mandal, A. Brandenburg,
T. Kahniashvili, A. Kosowsky, "Numerical simulations of gravitational
waves from early-universe turbulence," Phys. Rev. D 102, 083512 (2020),
arXiv:1903.08585.

RoperPol:2022iel - A. Roper Pol, C. Caprini, A. Neronov, D. Semikoz,
"The gravitational wave signal from primordial magnetic fields in the
Pulsar Timing Array frequency band," Phys. Rev. D 105, 123502 (2022),
arXiv:2201.05630.

He:2022qcs       - Y. He, A. Roper Pol, A. Brandenburg,
"Modified propagation of gravitational waves from the early
radiation era," JCAP 06 (2023), 025, arXiv:2212.06082.

Planck:2018vyg   - [Planck], "Planck 2018 results. VI.
Cosmological parameters," Astron.Astrophys. 641 (2020) A6,
Astron.Astrophys. 652 (2021) C4 (erratum),
arXiv:1807.06209.

Rasanen - Lecture notes on Cosmology by Syksy Räsänen
(https://www.mv.helsinki.fi/home/syrasane/cosmo/)

"""

import astropy.constants as const
import astropy.units as u
import pandas as pd
import numpy  as np
from cosmoGW.cosmoGW import HOME

# reference values and constants
Tref     = 100*u.GeV  # EWPT
gref     = 100        # EWPT
Neff_ref = 3          # reference Neff = 3

# values at present time from Planck:2018vyg
T0K      = 2.72548*u.K
H0_ref   = 100*u.km/u.s/u.Mpc
H0_ref   = H0_ref.to(u.Hz)
OmL0_ref = 0.6841
OmM0_ref = 1. - OmL0_ref
h0_ref   = 0.6732

######################### Values at present time #########################

def values_0(h0=1., neut=False, Neff=Neff_ref, ret_rad=False):

    """
    Function that returns the values of cosmological parameters at present time.

    Arguments:
        h0      -- parameterizes the uncertainties (Hubble tension)
                   on the value of the Hubble rate (default 1)
        neut    -- option to add neutrinos in the calculation of today's
                   relativistic dofs
        Neff    -- effective number of neutrino species (default is 3)
        ret_rad -- option to return radiation energy density at present time

    Returns:
        g0      -- 2 relativistic degrees of freedom (assumes massive neutrinos
                   unless ret_rad=True)
        g0s     -- ~3.91 entropic/adiabatic degrees of freedom (including neutrinos)
        T0      -- temperature 2.72548 K, returned in energy units (MeV)
        H0      -- Hubble rate at the present time H0 = 100 h0 km/s/Mpc in frequency
                   units (Hz)
        rho_rad0, Om_rad0 -- radiation energy density (and ratio to critial density)
                             at present time

    Reference: Rasanen.
    """

    # relativistic dofs from photons
    g0  = 2
    # contribution from neutrinos if neut is True
    if neut: g0 = 2*(1 + Neff*7/8*(4/11)**(4/3))
    # adiabatic dofs
    g0s = 2*(1 + Neff*7/8*4/11)
    T0  = (T0K*const.k_B).to(u.MeV)
    H0  = H0_ref*h0

    if ret_rad:
        rho_rad0 = rho_radiation(g=g0, T=T0)
        rho_c    = rho_critical(H=H0)
        Om_rad0  = rho_rad0/rho_c
        return g0, g0s, T0, H0, rho_rad0, Om_rad0
    else: return g0, g0s, T0, H0

########################  Radiation-dominated era ########################

def Hs_fact():

    """
    Function that computes the factor used in the calculation of the Hubble
    parameter at the time of generation (within the radiation-dominated era).
    The factor is given in units of Hz/MeV^2, such that the actual Hubble rate
    is given after multiplying fact*sqrt(g)*T^2, being g the number of dof and
    T the temperature scale (in MeV).

    Reference: RoperPol:2022iel, eq. 29
    """

    fact = np.sqrt(4*np.pi**3*const.G/45/const.c**5/const.hbar**3)
    fact = fact.to(u.Hz/u.MeV**2)

    return fact

def check_temperature_MeV(T, func=''):

    try:
        T = T.to(u.MeV)
    except:
        print('The input temperature in ', func,
              ' needs to be given in energy units using astropy.units \n'
              ' setting T to default value T =', Tref)
        T = Tref.to(u.MeV)

    return T

def Hs_val(g=gref, T=Tref):

    """
    Function that computes the Hubble parameter at the time of generation
    (within the radiation-dominated era).

    Arguments:
        g  -- number of relativistic degrees of freedom (dof) at the
              time of generation (default is 100)
        T  -- temperature scale at the time of generation in energy units
              (convertible to MeV) (default is 100 GeV)

    Returns:
        Hs -- Hubble rate in frequency units (Hz)

    Reference: RoperPol:2022iel, eq. 29
    """

    Hs_f = Hs_fact()
    T    = check_temperature_MeV(T, func='Hs_val')
    Hs   = Hs_f*T**2*np.sqrt(g)

    return Hs

def as_fact(Neff=Neff_ref):

    """
    Function that computes the factor used in the calculation of the ratio
    between the scale factor at the time of generation and the present
    time assuming adiabatic expansion of the universe.

    The factor is in units of MeV and the ratio is obtained by multiplying
    fact*g^(-1/3)/T, being g the number of dof and T the temperature scale
    (in MeV).

    Arguments:
        Neff -- effective number of neutrino species (default is 3)

    Returns:
        fact -- ratio to present-day scale factor, T0 g0s^(1/3)

    Reference: RoperPol:2022iel, eq. 28
    """

    _, g0s, T0, _ = values_0(Neff=Neff)
    fact = T0*g0s**(1/3)

    return fact

def as_a0_rat(g=gref, T=Tref, Neff=Neff_ref):

    """
    Function that computes the ratio between the scale factor at the time
    of generation (a*) and the present time (a_0) assuming adiabatic
    expansion of the universe.

    Arguments:
        g     -- number of relativistic degrees of freedom (dof) at the
                 time of generation (default is 100)
        T     -- temperature scale at the time of generation in energy units
                 (convertible to MeV) (default is 100 GeV)
        Neff  -- effective number of neutrino species (default is 3)

    Returns:
        as_a0 -- ratio of scale factors (a*/a0)

    Reference: RoperPol:2022iel, eq. 28
    """

    as_f  = as_fact(Neff=Neff)
    T     = check_temperature_MeV(T, func='as_a0_rat')
    as_a0 = as_f*g**(-1/3)/T

    return as_a0

############################ COSMOLOGY CALCULATIONS ############################

def rho_radiation(g=gref, T=Tref):

    """
    Function that computes the radiation energy density at different
    epochs of the universe.

    Arguments:
        g -- number of relativistic degrees of freedom (dof) at
             the time of generation (default is 100)
        T -- temperature scale at the time of generation in energy units
             (default is 100 GeV)

    Returns:
        rho_rad -- energy density in GeV/m^3 units

    Reference: RoperPol:2019wvy, eq. 3
    """

    T       = check_temperature_MeV(T, func='rho_radiation')
    rho_rad = np.pi**2/30*g*T**4/(const.hbar*const.c)**3
    rho_rad = rho_rad.to(u.GeV/u.m**3)

    return rho_rad

def check_Hubble_Hz(H, func=''):

    try:
        H = H.to(u.Hz)
    except:
        print('The input Hubble rate in ', func,
              ' needs to be given in frequency units using astropy.units \n'
              ' setting H to default value H =', H0_ref)
        H = H0_ref

    return H

def rho_critical(H=H0_ref):

    """
    Function that computes the critical energy density at different
    epochs of the universe.

    Arguments:
        H -- Hubble rate in units of frequency

    Returns:
        rho_c -- energy density in GeV/m^3 units

    Reference: RoperPol:2019wvy, eq. 3
    """

    H     = check_Hubble_Hz(H, func='rho_critical')
    rho_c = 3*H**2*const.c**2/8/np.pi/const.G
    rho_c = rho_c.to(u.GeV/u.m**3)

    return rho_c

def thermal_g(dir0='', T=Tref, s=0, file=True, Neff=Neff_ref):

    """
    Returns the relativistic dof g_* as a function of T_* according
    to the thermal history of the Universe.
    Note that for T < 0.5 MeV, after neutrino decoupling, entropic
    and relativistic g are approximately equal.

    Arguments:
        dir0 -- directory where the file of dof is stored
                ('resources/cosmology/' directory by default)
        T    -- temperature scale at the time of generation in energy units
                (default is 100 GeV)
        s    -- option to return adiabatic (s=1) dof instead of relativistic
                (default 0)
        file -- option to read g_* or gS from a file with numerical values,
                based on numerical calculations (default True)
                It reads the file 'dir0/cosmology/T_gs.csv' and should be a
                pandas file with columns: ['T [GeV]', 'g_*', 'gS']
        Neff -- effective number of neutrino species (default is 3)

    Returns:
        g    -- relativistic or adiabatic degrees of freedom

    Reference: Rasanen.
    """

    # to make sure that the values at present time correspond to those at the
    # end of RD we take relativistic neutrinos
    g0, g0s, _, _ = values_0(neut=True, Neff=Neff)

    # read the file if it exists
    if file:
        try:
            dirr = dir0 + 'T_gs.csv'
            if dir0 == '':
                dirr = HOME + '/resources/cosmology/T_gs.csv'
            df = pd.read_csv(dirr)
            Ts = np.array(df['T [GeV]'])
            if s == 0: gs = np.array(df['g_*'])
            if s == 1: gs = np.array(df['gS'])
            T  = T.to(u.GeV)    # values of T from file are in GeV
            g  = np.interp(T.value, np.sort(Ts), np.sort(gs))
            T  = check_temperature_MeV(T, func='thermal_g')
            # check if T is a single value or an array
            if not isinstance(T.value, (list, tuple, np.ndarray)):
                if T.value < 0.1:
                    if s == 0: g = g0
                    if s == 1: g = g0s
            else:
                if s == 0: g[T.value < 0.1] = g0
                if s == 1: g[T.value < 0.1] = g0s

        except:
            file = False
            print('thermal_g reads the file %sT_gs.csv, which does not exist! \n'%dir0)
            print('using piecewise approximated function to approximate g')

    # if a file is not given, one can use an approximate piecewise function
    if not file:
        T = check_temperature_MeV(T, func='thermal_g')
        T = T.value
        # check if T is an array or a single value
        if not isinstance(T, (list, tuple, np.ndarray)):
            len_T = 1; T = [T]
        else: len_T = len(T)
        g = np.zeros(len_T)
        for i in range(0, len_T):
            # Check value of T in MeV and assign g_*
            if T[i] < 0.1:
                if s == 0: g[i] = g0
                if s == 1: g[i] = g0s
            elif T[i] < 0.5: g[i] = 7.25
            elif T[i] <= 100: g[i] = 10.75
            elif T[i] <= 150: g[i] = 17.25
            elif T[i] < 1e3: g[i] = 61.75
            elif T[i] < 4e3: g[i] = 75.75
            elif T[i] < 8e4: g[i] = 86.25
            elif T[i] < 1.7e5: g[i] = 96.25
            else: g[i] = 106.75
        if len_T == 1: g = g[0]

    return g

############################### FRIEDMANN EQUATIONS ###############################

"""
The code has been developed and used for the results of
He:2022qcs (see appendix A).

Friedmann solver included to GW_turbulence in June 2022,
adapted and added to cosmoGW for v 1.1.0 release in March 2025.

"""

def RD_dofs(dir0='', Neff=Neff_ref, scalef=False):

    """
    Function that computes the degrees of freedom (relativistic and adiabatic)
    during the RD era.

    Arguments:
        dir0 -- directory where the file of dof is stored
                ('resources/cosmology/' directory by default)'
        Neff -- effective number of neutrino species (default is 3)

    Returns:
        T      -- array of temperatures within RD era
        as_T   -- array of scale factors
        gs, gS -- array of relativistic and adiabatic dofs

    Calls functions 'values_0' and 'thermal_g'; see references therein.
    """

    #### compute the relativistic degrees of freedom as a function
    #### of T during the RD epoch from a stored file using the function
    #### thermal_g

    # array of temperatures during RD era
    T  = np.logspace(-4, 8, 1000)*u.MeV
    gS = thermal_g(T=T, s=1, file=True, dir0=dir0)
    gs = thermal_g(T=T, s=0, file=True, dir0=dir0)

    g0, g0s, T0, _ = values_0(neut=True, Neff=Neff)

    ### convert temperature and degrees of freedom into an array
    ### of scale factors using adiabatic expansion of the universe
    as_T = (T0.to(u.MeV)/T)*(g0s/gs)**(1/3)

    return T, as_T, gs, gS

def dofs_vs_a(a, dir0='', Neff=Neff_ref):

    """
    Function that computes the degrees of freedom (relativistic and adiabatic)
    for an array of scale factors.

    Arguments:
        a    -- array of scale factors
        dir0 -- directory where the file of dof is stored ('/cosmology/' directory by default)
        Neff -- effective number of neutrino species (default is 3)

    Returns:
        gs, gS -- array of relativistic and adiabatic dofs

    Calls functions 'values_0' and 'thermal_g'; see references therein.
    """

    #### Read the dofs during RD era
    T, as_T, gs, gS = RD_dofs(dir0=dir0, Neff=Neff)

    #### convert arrays of degrees of freedom and interpolate to original array of scale factors a
    #### (needs to be sorted first for interp to work)
    inds = np.argsort(as_T)
    as2  = as_T[inds]
    gs   = gs[inds]
    gS   = gS[inds]
    gs   = np.interp(a, as2, gs)
    gS   = np.interp(a, as2, gS)

    return gs, gS

def Hs_from_a(a, dir0='', Neff=Neff_ref):

    """
    Function that computes the Hubble rate H_* during the RD era given only the scale
    factor and assuming adiabatic expansion

        a^3 g_S T^3 = constant

    Arguments:
        a    -- array of scale factors
        dir0 -- directory where the file of dof is stored ('/cosmology/' directory by default)
        Neff -- effective number of neutrino species (default is 3)

    Returns:
        Hs   -- Hubble rate during RD era

    Calls functions 'values_0' and 'thermal_g'; see references therein.
    """

    # compute the dofs
    gs, gS = dofs_vs_a(a, Neff=Neff, dir0=dir0)
    # compute the temperature scale from adiabatic expansions
    g0, g0s, T0, _ = values_0(neut=True, Neff=Neff)
    T = T0.to(u.MeV)/a*(g0s/gs)**(1/3)
    # get the Hubble rate
    Hs = Hs_val(T=T, g=gs)

    return Hs

def Omega_rad_dof(a, dir0='', Neff=Neff_ref):

    """
    Function that computes the factor that takes into account the radiation
    energy density ratio variation due to the inclusion of varying dofs during
    the RD era.

    Arguments:
        a    -- array of scale factors
        dir0 -- directory where the file of dof is stored ('/cosmology/' directory by default)
        Neff -- effective number of neutrino species (default is 3)

    Returns:
        Om_rat_dof -- ratio of radiation energy density due to accounting for
                      T depending dofs

    Reference: He:2022qcs, eq.A.2
    """

    gs, gS = dofs_vs_a(a, dir0=dir0, Neff=Neff)
    g0, g0s, T0, H0 = values_0(neut=True, Neff=Neff)
    Om_rat_dof = (gs/g0)*(gS/g0s)**(-4/3)

    #### the numerical results for the factor (gs/g0)*(gS/g0s)**(-4/3)
    #### present some numerical inacuracy, so it requires some smoothing
    amax, Emax = sp.local_max(a[a>1e-10], Om_rat_dof[a>1e-10])
    Emax[Emax > 1] = 1.
    aa = np.append(a[a<=1e-10], amax)
    Om_rat_dof = np.append(Om_rat_dof[a<=1e-10], Emax)
    Om_rat_dof = np.interp(a, aa, Om_rat_dof)

    return Om_rat_dof

def Omega_vs_a(a, dir0='', a0=1, h0=h0_ref, OmL0=OmL0_ref, dofs=True, Neff=Neff_ref):

    """
    Function that computes the energy density ratio to present-time critical energy
    density as a function of the scale factor a, for a universe composed by matter,
    radiation, and dark energy:

    Omega (a) = OmR0 x a^(-4) + OmM0 x a^(-3) + OmL0

    Arguments:
        a    -- array of scale factors
        dir0 -- directory where the file of dof is stored ('/cosmology/' directory by default)
        a0   -- reference value of the scale factor at present time (default is 1)
        h0   -- present-time value of the Hubble rate H0 = h0 x 100 km/s/Mpc
                (default is 67.32 km/s/Mpc based on CMB observations)
        OmL0 -- present-time content of dark energy
        dofs -- option to compensate the rad energy density using dofs during RD era
        dir0 -- directory where file for dofs during RD era is stored

    Returns:
        Om_tot  -- total energy density (normalized to present-time critical)
        Om_rad  -- radiation energy density (normalized)
        Om_matt -- matter energy density (normalized)

    Reference: He:2022qcs, eq.A.2
    """

    ########## compute Om_rad0 ##########
    g0, g0s, T0, H0, rho_rad0, OmR0 = values_0(h0=h0, neut=True,
                                               Neff=Neff, ret_rad=True)

    #### compute dofs during RD era if dofs is chosen
    #### to compensate the radiation energy density dependence on dofs
    #### during RD era

    if dofs:
        Om_rat_dof=Omega_rad_dof(a, dir0=dir0, Neff=Neff)
        OmR0 = OmR0*Om_rat_dof

    ##### compute other contributions Om_Lam0 and Om_mat0, and total Om_tot
    OmM0   = 1 - OmL0 - OmR0
    Om_mat = (a/a0)**(-3)*OmM0
    Om_rad = (a/a0)**(-4)*OmR0
    Om_tot = OmL0 + Om_rad + Om_mat

    return Om_tot, Om_rad, Om_mat

def friedmann(a, dir0='', a0=1, h0=h0_ref, OmL0=OmL0_ref, dofs=True, Neff=Neff_ref):

    """
    Function that uses Friedmann equations to compute the eos (w) and the time derivatives
    of the scale factor for the radiation energy density described in Omega_vs_a().

    Arguments:
        a    -- array of scale factors
        dir0 -- directory where the file of dof is stored ('/cosmology/' directory by default)
        a0   -- reference value of the scale factor at present time (default is 1)
        h0   -- present-time value of the Hubble rate H0 = h0 x 100 km/s/Mpc
                (default is 67.32 km/s/Mpc based on CMB observations)
        OmL0 -- present-time content of dark energy
        dofs -- option to compensate the rad energy density using dofs during RD era
        dir0 -- directory where file for dofs during RD era is stored

    Returns:
        w   -- equation of state (p = w rho)
        ad  -- cosmic time derivative of the scale factor
        add -- second cosmic time derivative of the scale factor
        ap  -- conformal time derivative of the scale factor
        app -- second conformal time derivative of the scale factor

    Reference: He:2022qcs, eqs. A.5-A.6
    """

    g0, g0s, T0, H0   = values_0(h0=h0, neut=True, Neff=Neff)
    Om_tot, Om_rad, _ = Omega_vs_a(a, a0=a0, h0=h0, OmL0=OmL0, dofs=dofs,
                                   dir0=dir0, Neff=Neff)

    # equation of state
    w   = (1/3*Om_rad - OmL0)/Om_tot
    # time derivatives (cosmic time)
    add = -.5*Om_tot*H0**2*(a/a0)*(1 + 3*w)
    ad  = (a/a0)*np.sqrt(Om_tot)*H0
    # time derivatives (conformal time)
    app = .5*a**3*Om_tot*(1 - 3*w)*H0**2
    ap  = ad*a

    return w, ad, add, ap, app

def friedmann_solver(a, dir0='', a0=1., h0=h0_ref, OmL0=OmL0_ref,
                     dofs=True, Neff=Neff_ref,
                     return_all=False, save=True, nm_fl=''):

    """
    Function that uses Friedmann equations and solve them numerically to obtain
    the evolution of a(eta) and a(t) using the Omega distribution obtained from
    the function Omega_vs_a.

    A tutorial is available under cosmology/cosmology.ipynb

    Arguments:
        a    -- array of scale factors
        dir0 -- directory where the file of dof is stored ('/cosmology/' directory by default)
        a0   -- reference value of the scale factor at present time (default is 1)
        h0   -- present-time value of the Hubble rate H0 = h0 x 100 km/s/Mpc
                (default is 67.32 km/s/Mpc based on CMB observations)
        OmL0 -- present-time content of dark energy (default is 0.6841)
        dofs -- option to compensate the rad energy density using dofs during RD era
        Neff -- effective number of neutrino species (default is 3)
        return_all -- option to return all variables used in the Friedmann solver
        save -- option to save the solutions in the file
                'friedmann/solution#nm_fl.csv' where #nm_fl is the inpute file name
                The input variables used in the solver are stored in
                'friedmann/README#nm_fl.csv'

    Reference: He:2022qcs, appendix A.
    """

    Om_tot, Om_rad, Om_mat = Omega_vs_a(a, a0=a0, h0=h0, OmL0=OmL0, dir0=dir0,
                                         dofs=dofs, Neff=Neff)

    #### numerically compute the arrays of cosmic and conformal times
    #### from the energy density ratio
    difft   = np.zeros(len(a))
    diffeta = np.zeros(len(a))
    if a[0] > 1e-18:
        print('minimum a given is ', a[0])
        print('note that the solver assumes that a[0] corresponds to',
              'times t = eta = 0, so make sure to use an array of a with',
              'small enough values (a[0] ~ 1e-20)')
    difft[0]   = 0
    diffeta[0] = 0

    print('Entering Friedmann solver')

    for i in range(1, len(a)):
        aas   = np.logspace(np.log10(a[0]), np.log10(a[i]), 10000)
        Om_as = np.interp(aas, a, Om_tot)
        fft   = 1/aas/np.sqrt(Om_as)
        ffeta = fft/aas
        try:
            difft[i]   = np.trapezoid(fft, aas)
            diffeta[i] = np.trapezoid(ffeta, aas)
        except:
            difft[i]   = np.trapz(fft, aas)
            diffeta[i] = np.trapz(ffeta, aas)

    g0, g0s, T0, H0 = values_0(h0=h0)
    t   = difft/H0
    eta = diffeta/H0

    # Using Friedmann equations, we compute the eos (w), and the time derivatives of the
    # scale factor
    w, ad, add, ap, app = friedmann(a, dir0=dir0, h0=h0, OmL0=OmL0, dofs=dofs, Neff=Neff)

    if save:
        df = pd.DataFrame({'a' : a, 't' : t, 'eta': eta,
                           'ap/a': ap/a, 'app/a' : app/a})
        df.to_csv('friedmann/solution' + nm_fl + '.csv')
        with open('friedmann/README' + nm_fl + '.txt', 'w') as f:
            f.write("The file solution" + nm_fl + ".csv contains the ")
            f.write("solutions from the Friedmann solver using the parameters:\n")
            f.write("a0 = %.4e, h0 = %.4f, OmL0 = %.4f, Neff = %.4f \n"%(a0, h0, OmL0, Neff))
            f.write("The solution contains 'a', 't', 'eta', 'ap/a' and 'app/a'")
        print('The results are saved in the file friedmann/solution' + nm_fl + '.csv')
        print('The input parameters used are stored in friedmann/README' + nm_fl + '.txt')

    print('Leaving Friedmann solver')

    if return_all: return t, eta, Om_tot, Om_rad, Om_mat, w, ad, add, ap, app
    else: return t, eta

def normalized_variables(a, eta, ap_a, app_a, dir0='', T=Tref, h0=h0_ref):

    """
    Function that computes the normalized a, eta, HH, a'' for a given specific
    initial time of GW generation, which are required to be used in the Pencil Code.

    A tutorial is available under cosmology/cosmology_PC.ipynb

    Arguments:
        a     -- scale factors, normalized to present-time a_0 = 1
        eta   -- conformal times, normalized to present-time a_0 = 1
        ap_a  -- conformal Hubble time a'/a, normalized to present-time a_0 = 1
        app_a -- a''/a, normalized to present-time a_0 = 1
        dir0  -- directory where the file of dof is stored ('/cosmology/' directory by default)
        T     -- temperature scale at the time of generation in energy units
                 (default is 100 GeV)
        h0    -- present-time value of the Hubble rate H0 = h0 x 100 km/s/Mpc
                 (default is 67.32 km/s/Mpc based on CMB observations)

    Returns:
        a_n       -- normalized scale factor a/a_*
        eta_n     -- normalized conformal time eta/eta_*
        HH_n      -- normalized conformal Hubble rate H/H_*
        app_a_n   -- normalized second conformal time derivative of a, (a''/a)/H_*^2
        Omega     -- ratio of total energy to present-time critical energy denstiy
        w         -- equation of state
        eta_n_0   -- normalized conformal present time
        aEQ_n     -- normalized equipartition scale factor
        aL_n      -- normalized dark energy domination scale factor
        a_acc_n   -- normalized scale factor when acceleration starts
        eta_n_EQ  -- normalized conformal time at equipartition
        eta_n_L   -- normalized conformal time at dark energy domination
        eta_n_acc -- normalized conformal time when acceleration starts

    Reference: He:2022qcs, appendix A.

    The normalization follows that from RoperPol:2018sap.
    """

    g   = thermal_g(T=T, s=0, dir0=dir0)
    gS  = thermal_g(T=T, s=1, dir0=dir0)
    ast = as_a0_rat(T=T, g=gS)
    Hs  = Hs_val(T=T, g=g)

    # normalized scale rates
    a_n = a/ast
    a0  = 1/ast

    ## Find value of eta at the initial time of generation
    eta_ast = np.interp(ast, a, eta)
    # normalized conformal times
    eta_n   = eta/eta_ast
    eta_n_0 = np.interp(a0, a_n, eta_n)

    # normalized conformal Hubble rate
    HH_n = ap_a/Hs/ast

    # normalized acceleration
    app_a_n = app_a/Hs**2/ast**2

    ### we can recover the values of Omega and w
    H0    = h0*H0_ref
    Omega = (HH_n.value*Hs)**2/a_n**2/H0**2
    w     = 1/3*(1 - app_a_n*2/HH_n**2)

    # compute aEQ, aL, and a_acc
    inds      = np.argsort(w)
    aEQ_n     = np.interp(1/6, w[inds], a_n[inds])
    aL_n      = np.interp(-.5, w[inds], a_n[inds])
    a_acc_n   = np.interp(-1/3, w[inds], a_n[inds])
    eta_n_EQ  = np.interp(aEQ_n, a_n, eta_n)
    eta_n_L   = np.interp(aL_n, a_n, eta_n)
    eta_n_acc = np.interp(a_acc_n, a_n, eta_n)

    return a_n, eta_n, HH_n, app_a_n, Omega, w, eta_n_0, aEQ_n, \
           aL_n, a_acc_n, eta_n_EQ, eta_n_L, eta_n_acc

def ratio_app_a_n_factor(a, dir0='', a0=1, h0=h0_ref, OmL0=OmL0_ref,
                         dofs=True, Neff=Neff_ref):

    """
    Function that computes the ratio of a''/a (normalized) to conformal Hubble rate H (normalized)
    times a_*/a_0 using an approximation valid during the RD era.

    Reference: He:2022qcs, eq. 3.11

    Arguments:
        a    -- array of scale factors
        dir0 -- directory where the file of dof is stored ('/cosmology/' directory by default)
        a0   -- reference value of the scale factor at present time (default is 1)
        h0   -- present-time value of the Hubble rate H0 = h0 x 100 km/s/Mpc
                (default is 67.32 km/s/Mpc based on CMB observations)
        OmL0 -- present-time content of dark energy (default is 0.6841)
        dofs -- option to compensate the rad energy density using dofs during RD era
        Neff -- effective number of neutrino species (default is 3)

    Returns:
        factor -- ratio of a''/a to conformal Hubble rate H times a_*/a_0
    """

    g0, g0s, T0, H0, rho_rad0, OmR0 = values_0(h0=h0, neut=True, Neff=Neff, ret_rad=True)
    OmM0 = 1 - OmL0 - OmR0
    Om_rat_dof = Omega_rad_dof(a, Neff=Neff, dir0=dir0)
    factor = .5*OmM0/OmR0/Om_rat_dof

    return factor

def norm_variables_cut(eta_n, HH_n, a_n, Omega, Omega_mat, eta_n_0, dir0='',
                       T=Tref, OmM0=OmM0_ref, h0=h0_ref):

    """
    Function that cuts the normalized variables between the initial time eta/eta_* = 1
    to present-time.

    Arguments:
        eta_n   -- normalized conformal time eta/eta_*
        HH_n    -- normalized conformal Hubble rate H/H_*
        a_n     -- normalized scale factor a/a_*
        Omega   -- ratio of total energy to present-time critical energy denstiy
        Om_mat  -- matter energy density (normalized)
        eta_n_0 -- normalized conformal present time
        Hs      -- Hubble rate at the initial time
        ast     -- scale factor at the initial time
        dir0    -- directory where the file of dof is stored
                   ('/cosmology/' directory by default)
        T       -- temperature scale at the initial time in energy units
                   (default is 100 GeV)
        OmM0    -- present-time content of matter (default is 0.3159)
        h0      -- present-time value of the Hubble rate H0 = h0 x 100 km/s/Mpc
                   (default is 67.32 km/s/Mpc based on CMB observations)

    Returns: variables given as arrays from the initial time until present time
        eta_nn       -- normalized conformal time
        HH_nn        -- normalized conformal Hubble rate
        a_nn         -- normalized scale factor
        Omega_nn     -- ratio of total energy to present-time critical energy denstiy
        Omega_mat_nn -- matter energy density (normalized)
        app_nn       -- second time derivative of the scale factor
        w_nn         -- equation of state p/rho
    """

    H0 = h0*H0_ref

    # relativistic and adiabatic dofs
    g   = thermal_g(T=T, s=0, dir0=dir0)
    gS  = thermal_g(T=T, s=1, dir0=dir0)
    # scale factor and Hubble rate
    ast = as_a0_rat(T=T, g=gS)
    Hs  = Hs_val(T=T, g=g)

    # indices
    inds   = np.where(eta_n > 1)[0]
    inds2  = np.where(eta_n[inds] < eta_n_0)[0]

    eta_nn = cut_var(eta_n, 1, eta_n_0, inds, inds2)
    HH_n0  = np.interp(1, eta_n, HH_n.value)
    HH_nn  = cut_var(HH_n.value, HH_n0, H0/Hs/ast, inds, inds2)
    a_nn   = cut_var(a_n, 1, 1/ast, inds, inds2)
    Omega_nn = cut_var(Omega, (Hs/H0)**2, 1, inds, inds2)
    Omega_mat_nn = cut_var(Omega_mat, OmM0*ast**(-3), OmM0, inds, inds2)
    Omega_rad_nn = Omega_nn - Omega_mat_nn - (1 - OmM0)
    w_nn   = (1/3*Omega_rad_nn - (1 - OmM0))/Omega_nn
    app_nn = .5*HH_nn**2*(1 - 3*w_nn)

    return eta_nn, HH_nn, a_nn, Omega_nn, Omega_mat_nn, app_nn, w_nn

def cut_var(x, x0, xf, inds, inds2):

    """
    Function that cuts the variable x using the indices inds and inds2
    and adding an initial value x0 and final value xf.

    Used in norm_variables_cut function.
    """

    y = x[inds][inds2]
    y = np.append(x0, y)
    y = np.append(y, xf)

    return y
