"""
GW_analytical.py is a Python routine that contains analytical
calculations and useful mathematical functions.

Author: Alberto Roper Pol
Created: 01/12/2021
Updated: 31/08/2024

Other contributors: Antonino Midiri, Madeline Salome

Main references are:

RPCNS22 - A. Roper Pol, C. Caprini, A. Neronov, D. Semikoz, "The gravitational wave
signal from primordial magnetic fields in the Pulsar Timing Array frequency band,"
Phys. Rev. D 105, 123502 (2022), arXiv:2201.05630

RPNCBS23 - A. Roper Pol, A. Neronov, C. Caprini, T. Boyer, D. Semikoz, "LISA and γ-ray telescopes as
multi-messenger probes of a first-order cosmological phase transition," arXiv:2307.10744 (2023)

RPPC23 - A. Roper Pol, S. Procacci, C. Caprini, "Characterization of the gravitational wave spectrum
from sound waves within the sound shell model," arXiv:2308.12943

RPMC24 - A. Roper Pol, A. Midiri, C. Caprini, "Gravitational wave spectrum from decaying MHD
turbulence in the early-universe: the constant-in-time model," in preparation
"""

import numpy as np
import matplotlib.pyplot as plt
import plot_sets

import os
HOME = os.getcwd()

### Reference values
cs2 = 1/3      # speed of sound

### Reference slopes
a_ref = 4      # Batchelor spectrum k^4
b_ref = 5/3    # Kolmogorov spectrum k^(-5/3)
alp_ref = 2    # reference smoothness of broken power-law transition

############### ANALYTICAL FUNCTIONS USED FOR A SMOOTHED BROKEN POWER LAW ###############

def smoothed_bPL(k, A=1, a=a_ref, b=b_ref, kpeak=1., alp=alp_ref, norm=True,
                 Omega=False, alpha2=False, piecewise=False):

    """
    Function that returns the value of the smoothed broken power law (bPL) model
    for a spectrum of the form:

        zeta(K) = A x (b + abs(a))^(1/alp) K^a/[ b + c K^(alp(a + b)) ]^(1/alp),

    where K = k/kpeak, c = 1 if a = 0 or c = abs(a) otherwise. This spectrum is defined
    such that kpeak is the correct position of the peak and its maximum amplitude is given
    by A.

    If norm is set to False, then the non-normalized spectrum is used:

        zeta (K) = A x K^a/(1 + K^(alp(a + b)))^(1/alp)

    The function is only correctly defined when b > 0 and a + b >= 0

    Introduced in RPCNS22, equations 6 and 8
    Main reference is RPMC24

    Arguments:

        k -- array of wave numbers
        A -- amplitude of the spectrum
        a -- slope of the spectrum at low wave numbers, k^a
        b -- slope of the spectrum at high wave numbers, k^(-b)
        alp -- smoothness of the transition from one power law to the other
        norm -- option to normalize the spectrum such that its peak is located at
                kpeak and its maximum value is A
        kpeak -- spectral peak, i.e., position of the break from k^a to k^(-b)
        Omega -- option to use the integrated energy density as the input A
        alpha2 -- option to use the alternative convention, such that the spectrum
                  takes the form: zeta(K) ~ K^a/( b + c K^alpha )^((a + b)/alpha)
        piecewise -- option to return a piecewise broken power law:
                     zeta(K) = K^a for K < 1, and K^(-b) for K > 1
                    corresponding to the alpha -> infinity limit

    Returns:
        spectrum array
    """

    if b < max(0, -a):
        print('b has to be larger than 0 and -a')
        return 0*k**0

    c = abs(a)
    if a == 0: c = 1
    if alpha2: alp = alp/(a + b)

    K = k/kpeak
    spec = A*K**a
    if piecewise:
        spec[np.where(K > 1)] = A*K[np.where(K > 1)]**(-b)
    else:
        alp2 = alp*(a + b)
        if norm:
            m = (b + abs(a))**(1/alp)
            spec = m*spec/(b + c*K**alp2)**(1/alp)

        else: spec = spec/(1 + K**alp2)**(1/alp)

    if Omega: spec = spec/kpeak/calA(a=a, b=b, alp=alp, norm=norm,
                                     alpha2=alpha2, piecewise=piecewise)

    return spec

def complete_beta(a, b):

    '''
    Function that computes the complete beta function, only converges for
    positive arguments.

    B(a, b; x \to \infty) = \int_0^x u^(a - 1) (1 - u)^(b - 1) du

    Arguments:
        a, b -- arguments a, b of the complete beta function

    Returns:
        B -- value of the complete beta function
    '''

    import math as m

    if a > 0 and b > 0: B = m.gamma(a)*m.gamma(b)/m.gamma(a + b)
    else:
        print('arguments of beta function need to be positive')
        B = 0

    return B

def calIab_n_alpha(a=a_ref, b=b_ref, alp=alp_ref, n=0, norm=True):

    '''
    Function that computes the normalization factor that enters in the
    calculation of Iabn

    Arguments:
        a, b -- slopes of the smoothed_bPL function
        alp -- smoothness parameter of the smoothed_bPL function
        n -- n-moment of the integral
        norm -- option to normalize the spectrum such that its peak is located at
                kpeak and its maximum value is 1

    Returns:
        calI -- normalization parameter that appears in the integral
    '''

    alp2 = 1/alp/(a + b)
    a_beta = (a + n + 1)*alp2

    c = abs(a)
    if a == 0: c = 1

    calI = alp2
    if norm: calI = calI*((b + abs(a))/b)**(1/alp)/(c/b)**a_beta

    return calI

def Iabn(a=a_ref, b=b_ref, alp=alp_ref, n=0, norm=True, alpha2=False,
         piecewise=False):

    '''
    Function that computes the moment n of the smoothed dbPL spectra:

    \int K^n zeta(K) dK

    Reference is RPMC24, appendix A

    Arguments:

        a -- slope of the spectrum at low wave numbers, k^a
        b -- slope of the spectrum at high wave numbers, k^(-b)
        alp -- smoothness of the transition from one power law to the other
        n -- moment of the integration

    Returns: value of the n-th moment
    '''

    if a + n + 1 <= 0:

        print('a + n has to be larger than -1 for the integral',
              'to converge')
        return 0

    if b - n - 1 <= 0:

        print('b + n has to be larger than 1 for the integral',
              'to converge')
        return 0

    if piecewise:

        return (a + b)/(a + n + 1)/(b - n - 1)

    if alpha2: alp = alp/(a + b)
    alp2 = 1/alp/(a + b)
    a_beta = (a + n + 1)*alp2
    b_beta = (b - n - 1)*alp2
    calI = calIab_n_alpha(a=a, b=b, alp=alp, n=n, norm=norm)
    comp_beta = complete_beta(a_beta, b_beta)

    return comp_beta*calI

def calA(a=a_ref, b=b_ref, alp=alp_ref, norm=True, alpha2=False,
         piecewise=False):

#     '''
#     Function that computes the parameter AA = Iab,0 that relates the
#     peak and the integrated values of the smoothed_bPL spectrum
#
#     References are RPCNS22, equation 8, and RPMC24, appendix A
#
#     Arguments:
#
#         a -- slope of the spectrum at low wave numbers, k^a
#         b -- slope of the spectrum at high wave numbers, k^(-b)
#         alp -- smoothness of the transition from one power law to the other
#     '''

    return Iabn(a=a, b=b, alp=alp, n=0, norm=norm, alpha2=alpha2,
                piecewise=piecewise)

def calB(a=a_ref, b=b_ref, alp=alp_ref, n=1, norm=True, alpha2=False,
         piecewise=False):

    '''
    Function that computes the parameter {\cal B} = Iab;-1/Iab;0 that relates the
    peak and the integral scale, {\cal B} = xi kpeak
    '''

    Im1 = Iabn(a=a, b=b, alp=alp, n=-n, norm=norm, alpha2=alpha2, piecewise=piecewise)
    I0 = Iabn(a=a, b=b, alp=alp, n=0, norm=norm, alpha2=alpha2, piecewise=piecewise)
    BB = (Im1/I0)**n

    return BB

def calC(a=a_ref, b=b_ref, alp=alp_ref, tp='vort', norm=True, alpha2=False,
         piecewise=False, proj=True, q=.5):

    '''
    Function that computes the parameter {\cal C} that allows to
    compute the TT-projected stress spectrum by taking the convolution of the
    smoothed bPL spectra over \kk and \kk - \pp.

    It gives the spectrum of the stress of Gaussian vortical non-helical fields
    as

    P_\Pi (0) = 2 \pi^2 EM*^2 {\cal C} / k*

    References are RPCNS22, equation 22, for vortical and RPPC23, equation 46,
    for compressional fields. Detailed reference is RPMC24, appendix A

    Arguments:

        a -- slope of the spectrum at low wave numbers, k^a
        b -- slope of the spectrum at high wave numbers, k^(-b)
        alp -- smoothness of the transition from one power law to the other
        tp -- type of sourcing field: 'vort' or 'comp' available
    '''

    if proj:
        if tp == 'vort': pref = 28/15
        elif tp == 'comp': pref = 32/15
        elif tp == 'mix': pref = 16/5*q*(1 - q)
        elif tp == 'hel': pref = -4/3
    else:
        if tp == 'vort': pref = 6
        if tp == 'comp': pref = 8
        if tp == 'mix': pref = 8*q*(1 - q)
        if tp == 'hel': pref = -2

    if tp not in ['vort', 'comp', 'mix', 'hel']:
        print('tp has to be vortical (vort), compressional (comp),',
              'mixed (mix) or helical (hel)')
        pref = 0.

    return pref*Iabn(a=a*2, b=b*2, alp=alp/2, n=-2, norm=norm,
                     alpha2=alpha2, piecewise=piecewise)

############### ANALYTICAL TEMPLATE USED FOR A DOUBLE SMOOTHED BROKEN POWER LAW ###############

def smoothed_double_bPL(k, kpeak1, kpeak2, A=1, a=a_ref, b=1, c=b_ref, alp1=alp_ref, alp2=alp_ref, kref=1.):

    """
    Function that returns the value of the smoothed double broken power law (double_bPL) model
    for a spectrum of the form:

        zeta(K) = K^a/(1 + (K/K1)^[(a - b)*alp1])^(1/alp1)/(1 + (K/K2)^[(c + b)*alp2])^(1/alp2)

    where K = k/kref, K1 and K2 are the two position peaks, a is the low-k slope, c is the intermediate
    slope, and -c is the high-k slope. alp1 and alp2 are the smoothness parameters for each spectral
    transition.

    Reference is RPPC23, equation 50. Also used in RPNCBS23, equation 7

    Arguments:

        k -- array of wave numbers
        kpeak1, kpeak2 -- peak positions
        A -- amplitude of the spectrum
        a -- slope of the spectrum at low wave numbers, k^a
        b -- slope of the spectrum at intermediate wave numbers, k^b
        c -- slope of the spectrum at high wave numbers, k^(-c)
        alp1, alp2 -- smoothness of the transitions from one power law to the other
        kref -- reference wave number used to normalize the spectrum (default is 1)

    Returns:
        spectrum array
    """

    K = k/kref
    K1 = k/kref
    K2 = k/kref

    spec1 = (1 + (K/K1)**((a - b)*alp1))**(1/alp1)
    spec2 = (1 + (K/K2)**((c + b)*alp2))**(1/alp2)
    spec = A*K^a/spec1/spec2

    return spec

############################### NOT PUBLIC ON GITHUB ##############################

def Iabn_inc(x, alp=2, a=4, b=5/3, n=0):

    import scipy.special as spe

    xstar = a/b*x**(alp*(a + b))
    xprime = xstar/(1 + xstar)
    a_beta = (a + n + 1)/alp/(a + b)
    b_beta = (b - n - 1)/alp/(a + b)
    comp_beta = complete_beta(a_beta, b_beta)

    beta = spe.betainc(a_beta, b_beta, xprime)*comp_beta

    return beta

def corr_int_low(k, a=4):

    """
    Function that computes the integral of K^a from 0 to kmin, to be included
    in numerical integration of more complicated spectra that have this asymptotic
    behavior when K -> 0.
    """

    # a has to be larger than -1 for the integral to converge
    if a > -1: return k**(a + 1)/(a + 1)
    else:
        print('a has to be larger than -1')
        return 0

################# OBSOLETE FUNCTIONS (TO BE DELETED) #################
def get_D1_D2(a, b, alp, D1, D2, fact):

    D1 = 0.
    D2 = 1.
    fact = 1.

    if a*b > 0:
        D1 = a/b
        D2 = a/b
    elif a*b == 0:
        if a == 0 and b == 0:
            fact = fact*(1 + D2)**(1/alp)
        elif b == 0:
            fact = fact*D2**(1/alp)
    else:
        fact = fact*(1 + D2)**(1/alp)

    return D1, D2, fact

def fit_smoothed_bPL(x, A=1, a=4, b=5/3, alp=2, xc=1, Omega=False,
                     lfactors=True):

    """
    Function that returns the value of the smoothed broken power law (PL) model
    for the magnetic spectrum, which has the form
        y = A(1 + D)^(1/alp)*(k/kp)^a/(1 + D(k/kp)^(alp * (a + b)))^(1/alp),
    where D = a/b

    Arguments:
        x -- values of x
        A -- amplitude of the spectrum at the peak
        a -- slope of the smoothed broken PL at low wave numbers, k^a
        b -- slope of the smoothed broken PL at high wave numbers, k^(-b)
        alp -- smoothness of the transition from one power law to the other
        xc -- spectral peak, i.e., position of the break from k^a to k^(-b)
        Omega -- option to use the integrated energy desnity as the input A

    Returns:
        y -- value of the spectrum
    """

    if Omega: A = A/xc/A_alpha(alp=alp, a=a, b=b)

    if a + b > 0:
        alp = alp
    else: alp = -alp

    D1 = 0.
    D2 = 1.
    fact = 1.
    if lfactors:
        D1, D2, fact = get_D1_D2(a, b, alp, D1, D2, fact)
        fact = fact*(1 + D1)**(1/alp)

    y = A*fact*(x/xc)**a
    y = y/(1 + D2*(x/xc)**((a + b)*alp))**(1/alp)

    return y

def value_Pi0_sbPL(alp=2):

    """
    Function that computes the value of the Pi spectrum at k = 0 analytically
    using the smoothed broken power law model defined in fit_smoothed_bPL()
    with the slopes a = 4 and b = 5/3 and the amplitude A = 1 at the spectral
    peak.

    Arguments:
        alpha -- smoothing parameter

    Returns:
        C -- amplitude of the Pi spectrum at k = 0
    """

    import math as m

    C = 17**(2/alp)*5**(-13/17/alp - 1)*3**(-21/17/alp - 1)* \
            2**(-42/17/alp + 2)
    C *= m.gamma(1 + 21/17/alp)
    C *= m.gamma(13/17/alp)
    C *= 1/m.gamma(2/alp)

    return C

#### parameter {\cal A} that relates the peak and the integrated
#### values (only valid for a = 4 and b = 5/3)
#### OBSOLETE FUNCTION (use A_alpha instead)

def A_alp(alp=2):

    import math as m

    A = (1 + 12/5)**(1/alp)*2**(-30/17/alp - 1)* \
            3**(-15/17/alp + 1)*5**(15/17/alp)
    A *= m.gamma(1 + 2/17/alp)*m.gamma(15/17/alp)/m.gamma(1/alp)

    return A

#### Function that uses the analytical expression of the integral
#### Iab_n (alpha) = \int dK K^(a+n) (1 + (a/b)K^(alpha(a + b)))^(-1/alpha)

## obsolete (to be deleted)
def Iab_n_alpha(alp=2, a=4, b=5/3, n=0):

    import math as m

    D = a/b
    Iab = D**(-(a + 1 + n)/alp/(a + b))/alp/(a + b)/m.gamma(1/alp)
    Iab *= m.gamma((b - 1 - n)/alp/(a + b))*m.gamma((a + 1 + n)/alp/(a + b))

    return Iab

#### parameter {\cal A} that relates the peak and the integrated
#### values of the smoothed double BPL

## obsolete (to be deleted), new function is calA
def A_alpha(alp=2, a=4, b=5/3):

    D = a/b
    A = (1 + D)**(1/alp)*Iab_n_alpha(alp=alp, a=a, b=b, n=0)

    return A

#### parameter {\cal B} that relates the wave number of the integral
#### scale and the peak wave number

## obsolete (to be deleted), new function is calB
def B_alpha(alp=2, a=4, b=5/3):

    B = Iab_n_alpha(alp=alp, a=a, b=b, n=-1)
    B = B/Iab_n_alpha(alp=alp, a=a, b=b, n=0)

    return B

#### Function that computes parameters A2 and B2 that allow to map
#### between smoothed bouble BPL models.

## obsolete (to be deleted)
def AB2_alpha(alp=2, a=4, b=5/3):

    import math as m

    D = a/b
    alp2 = 1/alp/(a + b)
    A2 = m.gamma(alp2*(a + b))/alp2/m.gamma(alp2*(b - 1))/m.gamma(alp2*(a + 1))
    B2 = m.gamma(alp2*(b - 1))*m.gamma(alp2*(a + 1))
    B2 *= 1/m.gamma(alp2*b)/m.gamma(alp2*a)

    return A2, B2

## obsolete (to be deleted), new function is calC
def C_alpha(alp=2, a=4, b=5/3):

    D = a/b
    C = 28/15*(1 + D)**(2/alp)*Iab_n_alpha(a=2*a, b=2*b, alp=alp/2, n=-2)

    return C

def Cp_alpha(alp=2, a=4, b=5/3):

    D = a/b
    A = A_alpha(alp=alp, a=a, b=b)

    return 8/3*((1 + D)/D)**(1/alp)*A

def Cp_alpha_2(alp=2, a=4, b=5/3):

    D = a/b
    Cp = 10/7/Iab_n_alpha(a=2*a, b=2*b, alp=alp/2, n=-2)
    Cp *= Iab_n_alpha(a=a, b=b, alp=alp, n=0)*D**(-1/alp)

    return Cp

def K_Cp(alp=2, a=4, b=5/3):

    Cp = Cp_alpha_2(alp=alp, a=a, b=b)

    return Cp**(1/(b + 2))

def alpha2(alp=2, a=4, b=5/3):

    return 1./alp/(a + b)

def Omega_additional_sbpl(kpeak=1., Epeak=1., k0=1, kf=100,
                          a=4, b=5/3, alp=2.):

    """
    Function that computes the integrated spectrum that is out of the available
    wave numbers for a simulation.

    It uses the smoothed broken power law model and assumes that the asymptotic
    values of the spectrum at low and high wave numbers are:
    1. Epeak * (1 + a/b)^(1/alp) * (k/kpeak)^a at k < k0
    2. Epeak * (1 + b/a)^(1/alp) * (k/kpeak)^-b at k > kf
    """

    D = a/b
    corr0 = (1 + D)**(1/alp)*(k0/kpeak)**(a + 1)/(a + 1)
    corr1 = (1 + 1/D)**(1/alp)*(kf/kpeak)**(1 - b)/(b - 1)

    return (corr0 + corr1)*kpeak*Epeak

###### Functions that compute the spectrum of the stress by taking the
###### two-point correlation function of the TT-projected stresses \Pi_{ij}
###### by splitting the four-point correlators of the velocity or magnetic
###### field using Wick's theorem (for Gaussian fields), since T_ij = u_i u_j

## stress spectrum for nonhelical compressional fields

## obsolete (to be moved to GW_models)
def ET_correlator_compr(k, EK, Np=3000, Nk=60, plot=False,
                        extend=False, largek=3, smallk=-3):

    p = np.logspace(np.log10(k[0]), np.log10(k[-1]), Np)
    kp = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)
    if extend:
        Nk = int(Nk/6)
        kp = np.logspace(smallk, np.log10(k[0]), Nk)
        kp = np.append(kp, np.logspace(np.log10(k[0]), np.log10(k[-1]),
                       4*Nk))
        kp = np.append(kp, np.logspace(np.log10(k[-1]), largek, Nk))

    Nz = 500
    z = np.linspace(-1, 1, Nz)
    kij, pij, zij = np.meshgrid(kp, p, z, indexing='ij')
    ptilde2 = pij**2 + kij**2 - 2*kij*pij*zij
    ptilde = np.sqrt(ptilde2)

    EK_p = np.interp(p, k, EK)
    if plot:
        plt.plot(p, EK_p)
        plt.xscale('log')
        plt.yscale('log')

    EK_ptilde = np.interp(ptilde, k, EK)
    ptilde[np.where(ptilde == 0)] = 1e-50

    Pi_1 = np.trapz(EK_ptilde/ptilde**4*(1 - zij**2)**2, z, axis=2)
    kij, EK_pij = np.meshgrid(kp, EK_p, indexing='ij')
    kij, pij = np.meshgrid(kp, p, indexing='ij')
    Pi_2 = np.trapz(Pi_1*pij**2*EK_pij, p, axis=1)

    return kp, np.pi**2*Pi_2

## stress spectrum for nonhelical vortical fields

## obsolete (to be moved to GW_models)
def ET_correlator_vort(k, EK, Np=3000, Nk=60, plot=False,
                        extend=False, largek=3, smallk=-3):

    p = np.logspace(np.log10(k[0]), np.log10(k[-1]), Np)
    kp = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)
    if extend:
        Nk = int(Nk/6)
        kp = np.logspace(smallk, np.log10(k[0]), Nk)
        kp = np.append(kp, np.logspace(np.log10(k[0]), np.log10(k[-1]),
                       4*Nk))
        kp = np.append(kp, np.logspace(np.log10(k[-1]), largek, Nk))

    Nz = 500
    z = np.linspace(-1, 1, Nz)
    kij, pij, zij = np.meshgrid(kp, p, z, indexing='ij')
    ptilde2 = pij**2 + kij**2 - 2*kij*pij*zij
    ptilde = np.sqrt(ptilde2)

    EK_p = np.interp(p, k, EK)
    if plot:
        plt.plot(p, EK_p)
        plt.xscale('log')
        plt.yscale('log')

    EK_ptilde = np.interp(ptilde, k, EK)
    ptilde[np.where(ptilde == 0)] = 1e-20

    Pi0 = EK_ptilde/ptilde**2*(1 + zij**2)
    Pi_1 = np.trapz(Pi0*(2 - pij**2/ptilde**2*(1 - zij**2)), z, axis=2)
    kij, EK_pij = np.meshgrid(kp, EK_p, indexing='ij')
    kij, pij = np.meshgrid(kp, p, indexing='ij')
    Pi_2 = np.trapz(Pi_1*EK_pij, p, axis=1)

    return kp, np.pi**2*Pi_2

## additional stress spectrum contribution from helical vortical fields

## obsolete (to be moved to GW_models)
def ET_correlator_vort_hel(k, EK, Np=3000, Nk=60, plot=False, eps=1,
                           extend=False, largek=3, smallk=-3):

    p = np.logspace(np.log10(k[0]), np.log10(k[-1]), Np)
    kp = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)
    if extend:
        Nk = int(Nk/6)
        kp = np.logspace(smallk, np.log10(k[0]), Nk)
        kp = np.append(kp, np.logspace(np.log10(k[0]), np.log10(k[-1]),
                       4*Nk))
        kp = np.append(kp, np.logspace(np.log10(k[-1]), largek, Nk))

    Nz = 500
    z = np.linspace(-1, 1, Nz)
    kij, pij, zij = np.meshgrid(kp, p, z, indexing='ij')
    ptilde2 = pij**2 + kij**2 - 2*kij*pij*zij
    ptilde = np.sqrt(ptilde2)

    HK_p = 2*eps*np.interp(p, k, EK)/p
    if plot:
        plt.plot(p, HK_p)
        plt.xscale('log')
        plt.yscale('log')

    HK_ptilde = 2*eps*np.interp(ptilde, k, EK)/ptilde
    ptilde[np.where(ptilde == 0)] = 1e-20

    Pi0 = HK_ptilde/ptilde**2*(kij*zij - pij*zij**2)
    Pi_1 = np.trapz(Pi0, z, axis=2)
    kij, HK_pij = np.meshgrid(kp, HK_p, indexing='ij')
    kij, pij = np.meshgrid(kp, p, indexing='ij')
    p_pi = np.trapz(Pi_1*HK_pij*pij, p, axis=1)

    # prefactor to compensate for amplitude such that Pi_2 (0) = 1

    return kp, p_pi

##### Functions to compute the GW spectrum under the stationary
##### assumption for the UETC

def incoherent_time_integral(t, k, p, ptilde, cs=1, m=1, n=1, tini=1):

    import scipy.special as spe

    pp = n*k + cs*(m*ptilde + p)
    si_t, ci_t = spe.sici(pp*t)
    si_tini, ci_tini = spe.sici(pp*tini)

    # correct the values when the argument is zero to avoid infinities
    # in the cosine integral by giving them their logarithmic values
    # that appear when we take the differences between two times.

    inds_pp0 = np.where(pp == 0)
    ci_t[inds_pp0] = np.log(t)
    ci_tini[inds_pp0] = np.log(tini)

    return si_t, ci_t, si_tini, ci_tini

def incoherent_time_integral_flat(t, k, p, ptilde, cs=1, m=1, n=1, tini=1):

    pp = n*k + cs*(m*ptilde + p)

    return 2*(1 - np.cos(pp*(t - tini)))/pp**2

def effective_ET_correlator_stat(k, EK, tfin, Np=3000, Nk=60, plot=False, flat=False,
                                 extend=False, largek=3, smallk=-3, tini=1, cs2=.333):

    """
    Function that computes the projected stress spectrum Pi(k) from the
    magnetic spectrum under the assumption of Gaussianity.

    Arguments:
        k -- array of wave numbers
        mag -- array of magnetic spectrum values
        Np -- number of discretizations in the wave number p to be numerically
              integrated
        Nk -- number of discretizations of k to be used for the computation of
              the final spectrum
        plot -- option to plot the interpolated magnetic spectrum for debugging
                purposes (default False)
        extend -- option to extend the array of wave numbers of the resulting
                  Pi spectrum compared to that of the given magnetic spectrum
                  (default False)

    Returns:
        PiM -- spectrum of the magnetic stress
        kp -- final array of wave numbers
    """

    import scipy.special as spe

    cs = np.sqrt(cs2)

    p = np.logspace(np.log10(k[0]), np.log10(k[-1]), Np)
    kp = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)
    if extend:
        Nk = int(Nk/6)
        kp = np.logspace(smallk, np.log10(k[0]), Nk)
        kp = np.append(kp, np.logspace(np.log10(k[0]), np.log10(k[-1]),
                       4*Nk))
        kp = np.append(kp, np.logspace(np.log10(k[-1]), largek, Nk))

    Nz = 500
    z = np.linspace(-1, 1, Nz)
    kij, pij, zij = np.meshgrid(kp, p, z, indexing='ij')
    ptilde2 = pij**2 + kij**2 - 2*kij*pij*zij
    ptilde = np.sqrt(ptilde2)

    EK_p = np.interp(p, k, EK)
    if plot:
        plt.plot(p, EK_p)
        plt.xscale('log')
        plt.yscale('log')

    EK_ptilde = np.interp(ptilde, k, EK)
    ptilde[np.where(ptilde == 0)] = 1e-50

    GG_p = kij**0 - 1

    for m in [1, -1]:
        for n in [1, -1]:
            if flat:
                GG_p += incoherent_time_integral_flat(tfin, kij, pij, ptilde,
                                                      m=m, n=n, tini=tini, cs=cs)
            else:
                Si_A, Ci_A, Si_A_tini, Ci_A_tini = \
                        incoherent_time_integral(tfin, kij, pij, ptilde,
                                                 m=m, n=n, tini=tini, cs=cs)
                GG_p += (Ci_A - Ci_A_tini)**2 + (Si_A - Si_A_tini)**2

    Pi_1 = .25*np.trapz(EK_ptilde/ptilde**4*(1 - zij**2)**2*GG_p, z, axis=2)
    kij, EK_pij = np.meshgrid(kp, EK_p, indexing='ij')
    kij, pij = np.meshgrid(kp, p, indexing='ij')
    Pi_2 = np.trapz(Pi_1*pij**2*EK_pij, p, axis=1)

    return kp, np.pi**2*Pi_2

## function to compute GW spectrum time growth using Hindmarsh & Hijazi approximation
## of delta(k - p - \tilde p)

def stationary_OmGW(k, EK, Np=3000, Nk=60, plot=False, cs2=1/3,
                    extend=False, largek=3, smallk=-3):

    cs = np.sqrt(cs2)
    #p = np.logspace(np.log10(k[0]), np.log10(k[-1]), Np)
    kp = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)
    if extend:
        Nk = int(Nk/6)
        kp = np.logspace(smallk, np.log10(k[0]), Nk)
        kp = np.append(kp, np.logspace(np.log10(k[0]), np.log10(k[-1]),
                       4*Nk))
        kp = np.append(kp, np.logspace(np.log10(k[-1]), largek, Nk))
        print(kp)

#    pij = np.zeros(len(k), len(p))
    p_inf = kp*(1 - cs)/2/cs
    p_sup = kp*(1 + cs)/2/cs

    PiM = np.zeros(len(kp))

    for i in range(0, len(kp)):
        p = np.logspace(np.log10(p_inf[i]), np.log10(p_sup[i]), Np)
        ptilde = kp[i]/cs - p
        z = -kp[i]*(1 - cs2)/2/p/cs2 + 1/cs

        EK_p = np.interp(p, k, EK)
        EK_ptilde = np.interp(ptilde, k, EK)

        Pi1 = (1 - z**2)**2*p/ptilde**3*EK_p*EK_ptilde

        PiM[i] = .25*np.pi**2*np.trapz(Pi1, p)

    return kp, PiM

### computes the value at k -> 0 of the effective stress spectrum
### due to the stationarity of the UETC for sound waves production

def Epi_effective_0_stat(k, EK, tfin, Np=3000, Nk=60, plot=False,
                         extend=False, largek=3, smallk=-3,
                         tini=1, cs2=1/3):

    import scipy.special as spe

    p = np.logspace(np.log10(k[0]), np.log10(k[-1]), Np)
    EK_p = np.interp(p, k, EK)

    cs = np.sqrt(cs2)

    si_t, ci_t = spe.sici(2*cs*p*tfin)
    si_tini, ci_tini = spe.sici(2*cs*p*tini)

    GG_p = p**0*np.log(tfin/tini)**2
    GG_p += (ci_t - ci_tini)**2
    GG_p += (si_t - si_tini)**2

    Pi1 = np.trapz(GG_p*EK_p**2/p**2, p)

    return .5*Pi1*16/15*np.pi**2

#### old function of stress spectrum for vortical nonhelical fields
#### it integrates over \phi instead of \cos \phi = z

def compute_Pi_from_numerical(k, mag, Np=3000, Nk=60, plot=False,
                              extend=False, largek=3, smallk=-3):

    """
    Function that computes the projected stress spectrum Pi(k) from the
    magnetic spectrum under the assumption of Gaussianity.

    Arguments:
        k -- array of wave numbers
        mag -- array of magnetic spectrum values
        Np -- number of discretizations in the wave number p to be numerically
              integrated
        Nk -- number of discretizations of k to be used for the computation of
              the final spectrum
        plot -- option to plot the interpolated magnetic spectrum for debugging
                purposes (default False)
        extend -- option to extend the array of wave numbers of the resulting
                  Pi spectrum compared to that of the given magnetic spectrum
                  (default False)

    Returns:
        PiM -- spectrum of the magnetic stress
        kp -- final array of wave numbers
    """

    p = np.logspace(np.log10(k[0]), np.log10(k[-1]), Np)
    kp = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)
    if extend:
        Nk = int(Nk/6)
        kp = np.logspace(-smallk, np.log10(k[0]), Nk)
        kp = np.append(kp, np.logspace(np.log10(k[0]), np.log10(k[-1]),
                       4*Nk))
        kp = np.append(kp, np.logspace(np.log10(k[-1]), largek, Nk))

    Nphi = 300
    phi = np.linspace(0, np.pi, Nphi)
    kij, pij, phij = np.meshgrid(kp, p, phi, indexing='ij')
    PB_p = np.interp(p, k, mag)
    if plot:
        plt.plot(p, PB_p)
        plt.xscale('log')
        plt.yscale('log')

    kmp_mod_2 = pij**2 + kij**2 - 2*pij*kij*np.cos(phij)
    kmp_mod = np.sqrt(kmp_mod_2)
    kmp_mod_2[np.where(kmp_mod_2==0)] = 1e-20
    PB_kmp = np.interp(kmp_mod, k, mag)/kmp_mod_2
    Pi0 = 1 + np.cos(phij)**2
    Pi1 = 1 + (kij - pij*np.cos(phij))**2/kmp_mod_2
    Piph_B = Pi0*Pi1*PB_kmp*np.sin(phij)
    Pi_imB = np.trapz(Piph_B, phi, axis=2)
    Pi_p_BB = Pi_imB*PB_p
    PiM = .5*np.trapz(Pi_p_BB, p, axis=1)

    return kp, PiM

#### old function of helical contribution to the stress spectrum
#### for vortical fields
#### it integrates over \phi instead of \cos \phi = z

def compute_Pihel_from_numerical(k, mag, Np=3000, Nk=60, plot=False,
                                 extend=False, largek=3, smallk=-3):

    """
    Function that computes the projected stress spectrum Pi(k) from the
    magnetic helicity spectrum under the assumption of Gaussianity.

    Arguments:
        k -- array of wave numbers
        mag -- array of magnetic spectrum values, it should be given in the
               form (1/2) k H_M (k)
        Np -- number of discretizations in the wave number p to be numerically
              integrated
        Nk -- number of discretizations of k to be used for the computation of
              the final spectrum
        plot -- option to plot the interpolated magnetic spectrum for debugging
                purposes (default False)
        extend -- option to extend the array of wave numbers of the resulting
                  Pi spectrum compared to that of the given magnetic spectrum
                  (default False)

    Returns:
        PiMhel -- spectrum of the magnetic stress
        kp -- final array of wave numbers
    """

    p = np.logspace(np.log10(k[0]), np.log10(k[-1]), Np)
    kp = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)
    if extend:
        Nk = int(Nk/6)
        kp = np.logspace(-smallk, np.log10(k[0]), Nk)
        kp = np.append(kp, np.logspace(np.log10(k[0]), np.log10(k[-1]),
                       4*Nk))
        kp = np.append(kp, np.logspace(np.log10(k[-1]), largek, Nk))

    Nphi = 300
    phi = np.linspace(0, np.pi, Nphi)
    kij, pij, phij = np.meshgrid(kp, p, phi, indexing='ij')
    PB_p = np.interp(p, k, mag)
    if plot:
        plt.plot(p, PB_p)
        plt.xscale('log')
        plt.yscale('log')

    kmp_mod_2 = pij**2 + kij**2 - 2*pij*kij*np.cos(phij)
    kmp_mod = np.sqrt(kmp_mod_2)
    kmp_mod_2[np.where(kmp_mod_2==0)] = 1e-20
    kmp_mod[np.where(kmp_mod==0)] = 1e-20
    PB_kmp = np.interp(kmp_mod, k, mag)/kmp_mod_2
    Pi0 = 4*np.cos(phij)
    Pi1 = (pij*np.cos(phij) - kij)/kmp_mod
    Piph_B = Pi0*Pi1*PB_kmp*np.sin(phij)
    Pi_imB = np.trapz(Piph_B, phi, axis=2)
    Pi_p_BB = Pi_imB*PB_p
    PiM = .5*np.trapz(Pi_p_BB, p, axis=1)

    return kp, PiM

def compute_Piref(ks=[], alp=2, save=False, str_alp='0'):

    """
    Function that computes the spectrum Pi using a magnetic spectrum
    with amplitude and peak values of 1 and stores it as Pi_ref_alpha''.csv

    Arguments:
        ks -- array of wave numbers to define the magnetic spectrum
              (default is 1000 points logarithmically distributed
              from 10^{-3} to 10^3)
        alp -- alpha (smoothing parameter of the magnetic field spectrum)
               value used for the computation
        save -- option to save the resulting Pi in a file

    Returns:
        kf -- array of wave numbers for the spectrum Pi
        Pi -- values of Pi compensated by the factor C = Pi(k = 0)
    """

    import pandas as pd

    if len(ks) == 0: ks = np.logspace(-3, 3, 1000)
    EM_test = fit_smoothed_bPL(ks, alp=alp)
    kfPi, Pi = compute_Pi_from_numerical(ks, EM_test, Np=1500,
                                         Nk=300, plot=True)
    C = value_Pi0_sbPL(alp=alp)
    Pi /= C

    if save:
        df = pd.DataFrame({'k': kfPi, 'Pi':Pi})
        fl = dir0 + 'analytical/Pi_ref_alpha%.1f.csv'%alp
        if str_alp != '0': fl = dir0 + \
                '/analytical/Pi_ref_alpha%s.csv'%str_alp
        df.to_csv(fl)
        print('Pi_ref saved in %s'%fl)

    return kfPi, Pi

def read_Pi(dir='', str_alp='2'):

    """
    Function that reads the spectrum Pi from a file of the type
    Pi_ref_alpha'str_alpha'.csv previously generated with compute_Piref.

    Arguments:
        str_alp -- string that defines the name of the specific file to be
                   read
        dir -- directory where 'analytical' directory is located with the
               files in it (default uses dir0, which corresponds to the same
               directory as where the routine GW_analytical.py is stored)

    Returns:
        kf -- array of wave numbers for the spectrum Pi
        Pi -- values of Pi compensated by the factor C = Pi(k = 0)
    """

    import pandas as pd

    if dir == '': dir = dir0
    file = 'Pi_ref_alpha' + str_alp
    df = pd.read_csv(dir + '/analytical/' + file + '.csv')
    kf_Pi = np.array(df['k'])
    Pi_ref = np.array(df['Pi'])
    kf1 = np.logspace(3.001, 9, 1000)
    kf0 = np.logspace(-9, -2.999, 1000)
    Pi_ref1 = Pi_ref[-1]*(kf1/kf_Pi[-1])**(-11/3)
    Pi_ref0 = Pi_ref[0]*kf0**0
    kf_Pi = np.append(kf0, kf_Pi)
    kf_Pi = np.append(kf_Pi, kf1)
    Pi_ref = np.append(Pi_ref0, Pi_ref)
    Pi_ref = np.append(Pi_ref, Pi_ref1)

    return kf_Pi, Pi_ref

def shift_Pi(k, Pi, kpeak):

    """
    Function that returns Pi by shifting the position of its peak, i.e.,
    shifts a function in the x-axis.

    Arguments:
        k -- original wave number array
        Pi -- original Pi values
        kpeak -- position of the peak k (or value of x used for shifting)

    Returns:
        Pinew -- new Pi values after the shift
    """

    ks = k/kpeak
    Pinew = 10**np.interp(np.log10(ks), np.log10(k), np.log10(Pi))
    return Pinew

def plot_Pi_max(dir0='', ymin=1e-3, ymax=1e1, xmin=1e-1, xmax=20,
                str_alp='2', plot=True, txt=True):

    """
    Function that plots the reference Pi (i.e., for amplitude and position of
    the peak being 1) and shows the maximum values and positions of k*Pi,
    k^2*Pi, and k^3*Pi.

    Arguments:
        dir0 -- directory where the Pi_ref files are stored
                (dir0/analytical/Pi_ref_''.csv)
        ymin, ymax -- minimum and maximum y limits of the plot
        xmin, xmax -- minimum and maximum x limits of the plot
        str_alp -- string that indicates the name of the Pi_ref file to be
                   read (default is str_alp = '2', indicating a smoothing
                   parameter alpha = 2)
        plot -- option to plot the resulting Pi function (default True)
        txt -- option to print out the text with the results (default True)

    Returns:
        max_ks, Pi_max_ks -- k and Pi that correspond to maximum k*Pi
        max_ks2, Pi_max_ks2 -- k and Pi that correspond to maximum k^2*Pi
        max_ks3, Pi_max_ks3 -- k and Pi that correspond to maximum k^3*Pi
    """

    import spectra as sp

    kf_Pi, Pi_ref = read_Pi(str_alp=str_alp)
    max_ks, Pi_max_ks = sp.max_E_kf(kf_Pi, Pi_ref, exp=1)
    max_ks2, Pi_max_ks2 = sp.max_E_kf(kf_Pi, Pi_ref, exp=2)
    max_ks3, Pi_max_ks3 = sp.max_E_kf(kf_Pi, Pi_ref, exp=3)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(kf_Pi, Pi_ref, color='black')
        plt.vlines(1, ymin, ymax, color='black', ls='dashed', lw=.6)
        plt.hlines(1, xmin, xmax, color='black', ls='dashed', lw=.6)

        plt.vlines(max_ks, ymin, ymax, color='black', ls='dashed', lw=.6)
        plt.vlines(max_ks2, ymin, ymax, color='black', ls='dashed', lw=.6)
        plt.vlines(max_ks3, ymin, ymax, color='black', ls='dashed', lw=.6)

        plt.hlines(Pi_max_ks, xmin, xmax, color='black', ls='dashed', lw=.6)
        plt.hlines(Pi_max_ks2, xmin, xmax, color='black', ls='dashed', lw=.6)
        plt.hlines(Pi_max_ks3, xmin, xmax, color='black', ls='dashed', lw=.6)

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xscale('log')
        plt.yscale('log')
        plot_sets.axes_lines()
        plt.yticks(np.logspace(-3, 1, 5))

    if txt:
        print('Maximum of k*Pi corresponds to Pi = ', Pi_max_ks,
              ' at k = ', max_ks)
        print('Maximum of k^2*Pi corresponds to Pi = ', Pi_max_ks2,
              ' at k = ', max_ks2)
        print('Maximum of k^3*Pi corresponds to Pi = ', Pi_max_ks3,
              ' at k = ', max_ks3)

    return max_ks, Pi_max_ks, max_ks2, Pi_max_ks2, max_ks3, Pi_max_ks3

##### FUNCTIONS THAT ARE USED FOR THE CONSTANT-IN-TIME MODEL IN VORTICAL
##### TURBULENCE

def function_D(k, t, tfin=1e4, tini=1):

    """
    Function that computes the value of the function D(k, t) used in the
    analytical calculations of the GW energy density spectrum when assuming
    a constant sourcing stress spectrum, i.e., Pi(k, t1, t2) = Pi(k)

    Arguments:
        k -- array of wave numbers
        t -- array of times
        tini -- initial time of the turbulence sourcing (default 1)
        tfin -- final time of the turbulence sourcing

    Returns:
        D -- function D(k, t)
    """

    import scipy.special as spe

    tij, kij = np.meshgrid(t, k, indexing='ij')
    cost = np.cos(kij*tij)
    sint = np.sin(kij*tij)
    tij[np.where(tij>tfin)] = tfin
    si_t, ci_t = spe.sici(kij*tij)
    si_tini, ci_tini = spe.sici(kij*tini)
    aux1 = cost*(ci_t - ci_tini)
    aux2 = sint*(si_t - si_tini)
    D = aux1 + aux2

    return D

def function_D2_av(k, tfin=1e4, tini=1):

    """
    Function that computes the value of the function D(k, t) used in the
    analytical calculations of the GW energy density spectrum when assuming
    a constant sourcing stress spectrum, i.e., Pi(k, t1, t2) = Pi(k).
    It takes the average over oscillations considering very large times, i.e.,
    by shifting to present time.

    Arguments:
        k -- array of wave numbers
        t -- array of times
        tini -- initial time of the turbulence sourcing (default 1)
        tfin -- final time of the turbulence sourcing

    Returns:
        D2_av -- average value of the square of the function, D^2_av (k)
    """

    import scipy.special as spe

    si_t, ci_t = spe.sici(k*tfin)
    si_tini, ci_tini = spe.sici(k*tini)
    aux1 = (ci_t - ci_tini)**2
    aux2 = (si_t - si_tini)**2
    D2_av = aux1 + aux2

    return D2_av

def factor_FF(tini=1, tfin=1e4):

    FF = (1 + tini/tfin)**2
    return FF

def function_D2_env(k, tini=1, tfin=1e4, FF=0, A=1):

    # envelope at low wave numbers
    f1 = np.log(tfin)**2
    # envelope at high wave numbers
    f2 = A*np.log(tini + 1/k)**2
    # overshooting factor that appears for short tfin
    # (if FF is given as an input parameter, then not computed)
    if FF == 0: FF = factor_FF(tini=tini, tfin=tfin)

    # envelope
    f = np.minimum(f1, FF*f2)

    # position of the break
    diff = f1 - FF*f2
    ind = np.where(diff > 0)[0][0]
    k_br = k[ind]

    return f, k_br

def model_EM(k, E, ks=[], indt=0):

    """
    Function that fits the numerical results of the magnetic spectrum EM(k, t)
    over times to a smoothed broken power law (alpha = 2) using
    'fit_smoothed_bPL' and returns the fitting parameters.

    Arguments:
        k -- array of wave numbers of the original spectra
        E -- 2d array of spectral values (first index is time and second is
             wave number)
        ks -- new array of wave numbers in which to define the modelled spectra
              (default is to use same as k)
        indt -- allows to limit the times in which the modelling is performed,
                from time 0 to indt (default 0, i.e, only at initial time)

    Returns:
        mod_EMs -- modelled magnetic spectra using the numerical wave numbers k
        err_M -- error of the fitting model
        mod_EMs_ks -- modelled magnetic spectra using the new wave numbers ks
        As_EM -- amplitudes of the fitting model
        alps_EM -- smoothing parameter of the fitting model
        xcs_EM -- position of the peak of the fitting model
    """

    import scipy.optimize as opt

    if len(ks) == 0: ks = k

    mod_EMs_ks = np.zeros((indt + 1, len(ks)))
    mod_EMs = np.zeros((indt + 1, len(k)))
    As_EM = np.zeros(indt + 1)
    as_EM = np.zeros(indt + 1)
    bs_EM = np.zeros(indt + 1)
    alps_EM = np.zeros(indt + 1)
    xcs_EM = np.zeros(indt + 1)

    def fit_test(x, A, b, alp, xc):
        y = fit_smoothed_bPL(x, A=A, b=b, alp=alp, xc=xc)
        return y

    for i in range(0, indt + 1):

        bound = ((-np.inf, 0, -np.inf, 0),
                 (np.inf, np.inf, np.inf, 30))

        popt, pcov = opt.curve_fit(fit_test, k, E[i, :],
                                   p0=(1e-5, 5/3, 2, 10), bounds=bound)
        As_EM[i] = popt[0]
        as_EM[i] = 4
        bs_EM[i] = popt[1]
        alps_EM[i] = popt[2]
        xcs_EM[i] = popt[3]
        mod_EMs_ks[i, :] = fit_test(ks, popt[0], popt[1], popt[2], popt[3])
        mod_EMs[i, :] = fit_test(k, popt[0], popt[1], popt[2], popt[3])

    err_M = np.sqrt(np.array(np.trapz((mod_EMs - E[:indt + 1, :])**2, k,
                                       axis=1)/\
                                       np.array(np.trapz(E[:indt + 1, :]**2,
                                                         k, axis=1)),
                                                dtype='float'))

    return mod_EMs, err_M, mod_EMs_ks, As_EM, alps_EM, xcs_EM

def factM(k1, k2, E1, E2):

    """
    Function that computes the ratio between the amplitude of two magnetic
    spectra at initial time.

    Arguments:
        k1, k2 -- wave number arrays of runs 1 and 2
        E1, E2 -- spectral values arrays of runs 1 and 2

    Returns:
        factM -- ratio of amplitudes A2/A1
    """

    _ = model_EM(k1, E1)
    A1 = _[3]
    _ = model_EM(k2, E2)
    A2 = _[3]
    return A2/A1

def OmGW_from_Pi0(Pi0, k, t, tfin=1e4, tini=1):

    D = function_D(k, t, tfin=tfin, tini=tini)
    tij, kij = np.meshgrid(t, k, indexing='ij')
    tij, Pi0 = np.meshgrid(t, Pi0, indexing='ij')
    OmGW = 3*kij**3*D**2*Pi0
    EGW = 3*kij**2*D**2*Pi0

    return EGW, OmGW

def OmGW_from_Pi0_av(Pi0, k, tfin=1e4, tini=1):

    D2 = function_D2_av(k, tfin=tfin, tini=tini)
    OmGW = 3*k**3*D2*Pi0
    EGW = 3*k**2*D2*Pi0

    return EGW, OmGW

def OmGW_from_Pi0_env(Pi0, k, tini=1, tfin=1e4, FF=0, A=1):

    D2, k_br = function_D2_env(k, tini=tini, tfin=tfin, FF=FF, A=A)
    OmGW = 3*k**3*D2*Pi0
    EGW = 3*k**2*D2*Pi0

    return EGW, OmGW, k_br

def OmGW_from_OmM_kM_env(OmM, kM, k, tini=1, tfin=1e4, FF=0, A=1, multi=False):

    EM = OmM/A_alp()/kM
    if multi:
        NN = np.shape(OmM)
        OmGW = np.zeros((len(k), NN[0], NN[1]))
        EGW = np.zeros((len(k), NN[0], NN[1]))
        k_br = np.zeros((NN[0], NN[1]))
        kf_Pi, Pi_ref = read_Pi(str_alp='2')
        Pi0 = value_Pi0_sbPL()
        for i in range(0, NN[0]):
            Pi_ref_i = shift_Pi(kf_Pi, Pi_ref, kM[i, 0])
            for j in range(0, NN[1]):
                PiM = Pi0*Pi_ref_i*EM[i, j]**2/kM[i, j]
                PiM_k = 10**np.interp(np.log10(k), np.log10(kf_Pi),
                                      np.log10(PiM))
                EGW[:, i, j], OmGW[:, i, j], k_br[i, j] = \
                        OmGW_from_Pi0_env(PiM_k, k, tini=tini,
                                          tfin=tfin[i, j], FF=FF, A=A)
    else:
        kf_Pi, Pi = compute_PiM(EM, kM)
        PiM = 10**np.interp(np.log10(k), np.log10(kf_Pi),
                            np.log10(Pi))
        EGW, OmGW, k_br = OmGW_from_Pi0_env(PiM, k, tini=tini, tfin=tfin, FF=FF,
                                            A=A)

    return EGW, OmGW, k_br

def OmGW_from_OmM_kM_env_tfin_fit(OmM, kM, k, tini=1, FF=0, A=1, multi=False):

    if multi: kM_ij, OmM_ij = np.meshgrid(kM, OmM, indexing='ij')
    else:
        kM_ij = kM
        OmM_ij = OmM
    vA = np.sqrt(1.5*OmM_ij)
    dte = 1/kM_ij/vA
    dtfin = dtfin_dte(dte)
    EGW, OmGW, k_br = OmGW_from_OmM_kM_env(OmM_ij, kM_ij, k, tini=tini,
                                           tfin=tini + dtfin, FF=FF, A=A,
                                           multi=multi)

    return EGW, OmGW, k_br

def compute_PiM(EM, kp, dir='', alpha=2, alp_str='2'):

    import spectra as sp

    # read the reference Pi
    kf_Pi, Pi_ref = read_Pi(str_alp=alp_str, dir=dir)
    # shift values of Pi by the factor kp
    Pi = shift_Pi(kf_Pi, Pi_ref, kp)
    # amplitude of Pi at k = 0
    Pi0 = value_Pi0_sbPL(alp=alpha)
    Pi = Pi0*Pi*EM**2/kp

    return kf_Pi, Pi

def OmGW_from_Pi_coh(t, k, Pi, tini=1., NNt=100):

    Nt = len(t)
    Nk = len(k)
    OmGW = np.zeros((Nt, Nk))
    for j in range(0, Nt):
        tint = np.logspace(np.log10(tini), np.log10(t[j]), NNt)
        tij_int, kij = np.meshgrid(tint, k, indexing='ij')
        Pi_interp = np.zeros((NNt, Nk))
        for i in range(0, Nk):
            Pi_interp[:, i] = np.interp(tint, t, Pi[:, i])
        Pisqr = np.sqrt(Pi_interp)
        func_aux = Pisqr/tij_int*np.cos(kij*((t[j] - tij_int)))
        func = np.trapz(func_aux, tint, axis=0)
        OmGW[j, :] = func**2
    tij, kij = np.meshgrid(t, k, indexing='ij')
    EGW = 3*kij**2*OmGW
    OmGW = 3*kij**3*OmGW
    return EGW, OmGW

def dte_Om_k(OmM=.1, ks=2*np.pi, multi=False):

    """
    Function that returns the eddy turnover time as a function of the
    spectral peak and energy density (assumes radiation-dominated era to
    compute the Alfven speed).
    """

    if multi == True:
        ks, OmM = np.meshgrid(ks, OmM, indexing='ij')

    vA = np.sqrt(3/2*OmM)

    return 1/ks/vA

def dtfin_dte(dte):

    """
    Function that uses the empirical fit from numerical simulations to give
    a value of \delta \tfin as a function of the eddy turnover time.

    From A. Roper Pol et al., "The gravitational wave signal from primordial
    magnetic fields in the Pulsar Timing Array frequency band,"
    https://arxiv.org/abs/2201.05630 (2022).
    """
    dtfin = 0.184 + 1.937*dte
    return dtfin

def Af_dte(dte):

    """
    Function that uses the empirical fit from numerical simulations to give
    a value of the numerical ratio at the peak as a function of the eddy
    turnover time.

    From A. Roper Pol et al., "The gravitational wave signal from primordial
    magnetic fields in the Pulsar Timing Array frequency band,"
    https://arxiv.org/abs/2201.05630 (2022).
    """

    Af = 1.317 - .097*dte
    return Af

def slope_log_OmGW_model(k):

    """
    Function that returns the slope of the k^3 ln^2 (1 + 1/k) that appears
    in the analytical description of the GW spectra, see function_D2_env()
    and OmGW_from_Pi0().

    From A. Roper Pol et al., "The gravitational wave signal from primordial
    magnetic fields in the Pulsar Timing Array frequency band,"
    https://arxiv.org/abs/2201.05630 (2022).
    """

    beta = 3 - 2/(1 + k)/abs(np.log(1 + 1/k))

    return beta

def broken_PL_log(k, A=1., a=2, c=11/3, b=0, alp=2., kp=1.,
                  kH=1., kbreak=.5):

    """
    Function that returns the value of the smoothed broken power law (PL) model
    multiplied by a logarithmic branch, based on the model derived in
    Roper Pol et al., "The gravitational wave signal from primordial
    magnetic fields in the Pulsar Timing Array frequency band,"
    https://arxiv.org/abs/2201.05630 (2022).

    The GW spectrum in this model has the form

        y = A * (1 + D)^(1/alp) * Tau_GW * (k/kp)^a
            /(1 + D * (k/kp)^(alp * (b + c)))^(1/alp),

    where D = b/c and Tau_GW is

    Tau_GW = log(1 + kH/kbreak)^(a-b) if k < kbreak
             log(1 + kH/k)^(a-b) if k >= kbreak

    The resulting spectrum has slopes
       - k^a at k < kbreak
       - k^b at kbreak <= k < kpeak
       - k^(-c) at k >= kpeak

    Arguments:
        x -- values of x
        A -- amplitude of the spectrum at the peak
        a -- slope of the smoothed broken PL at low wave numbers, k^a
        b -- slope of the smoothed broken PL at high wave numbers, k^(-b)
        alp -- smoothness of the transition from one power law to the other
        xc -- spectral peak, i.e., position of the break from k^a to k^(-b)
        Omega -- option to use the integrated energy desnity as the input A

    Returns:
        y -- value of the spectrum
    """

    D1 = 0.
    D2 = 1.
    fact = 1.

    if b + c < 0:
        alp = -alp

    # when c = 0, we are interested in the value at the plateau at large k
    # otherwise, we want to adapt at the value around k*
    if c != 0:
        if b!= 0:
            if b*c > 0:
                D1 = b/c
            else:
                D1 = 1.
            D2 = D1
        fact = fact*np.log(1 + kH/kp)**(-(a - b))
    else:
        if b == 0:
            fact = fact*(1 + D2)**(1/alp)
        fact = fact*(kH/kp)**(-(a - b))
    # peak at kbreak is larger when a > = 0 and b < 0
    if a >= 0 and b < 0:
        fact = (1 + D1)**(-1/alp)*(kbreak/kp)**(-a)
        fact = fact*(1 + D2*(kbreak/kp)**(alp*(b + c)))**(-1/alp)
        if b + c == 0: fact = fact*(1 + D2)**(2/alp)
        fact = fact*np.log(1 + kH/kbreak)**(-(a - b))

    Tau_GW = np.zeros(len(k))
    Tau_GW[k >= kbreak] = np.log(1 + kH/k[k >= kbreak])**(a - b)
    Tau_GW[k < kbreak] = np.log(1 + kH/kbreak)**(a - b)*k[k < kbreak]**0

    y = A*Tau_GW*fact*(1 + D1)**(1/alp)*(k/kp)**a/ \
            (1 + D2*(k/kp)**((b + c)*alp))**(1/alp)

    return y

def double_brokenpl(k, A=1., a=2, b=11/3, c=2, alp=2., kp=1., alp2=2.,
                    kbreak=.5):

# a -> b,
    if b + c < 0:
        alp = -alp
    if a - b < 0:
        alp2 = -alp2

    D1 = 0.
    D2 = 1.
    D3 = 1.
    fact = 1.

    if c != 0:
        if b!= 0:
            if b*c > 0:
                D1 = b/c
            else:
                D1 = 1.
            D2 = D1
        fact = fact*(1 + D3*(kp/kbreak)**(-alp2*(a-b)))**(1/alp2)
    else:
        if b == 0:
            fact = fact*(1 + D2)**(1/alp)
            if a == 0:
                fact = fact*(1+D1)**(1/alp)*(1+D3)**(1/alp2)
        #fact = fact*(kH/kp)**(-(a - b))

    # peak at kbreak is larger when a > = 0 and b < 0
    if a >= 0 and b < 0:
        fact=1.
        if b*a < 0 and c!=0:
            D1 = b/c
            D2 = D1
        if b*c < 0:
            D2 = 1.
        if a != 0:
            D3 = -b/a
            fact = fact*(1 + D3)**(1/alp2)
        fact = fact*(1+D1)**(-1/alp)*(kbreak/kp)**(-b)
        fact = fact*(1 + D2*(kbreak/kp)**(alp*(b + c)))**(1/alp)

    y = A*fact*(1 + D1)**(1/alp)*(k/kp)**b/(1 + D2*(k/kp)**(alp*(b + c)))**(1/alp)
    y = y/(1 + D3*(k/kbreak)**(-alp2*(a - b)))**(1/alp2)
    #else:
    #    y = y*(1 + (k/kbreak)**(-alp2*(a - c)))**(1/alp2)

    return y

def spec_GW_from_hg0(k, t, GWs0, GWh0, GWm0, tini=1):

    dt = t - tini

    GWs = k**2*GWh0*np.sin(k*dt)**2 + GWs0*np.cos(k*dt)**2
    GWs -= k*GWm0*np.sin(2*k*dt)
    GWh = GWh0*np.cos(k*dt)**2 + GWs0/k**2*np.sin(k*dt)**2
    GWh += GWm0/k*np.sin(2*k*dt)
    GWm = -.5*k*GWh0*np.sin(2*k*dt) + GWm0*np.cos(2*k*dt)
    GWm += .5*GWs0/k*np.sin(2*k*dt)

    return GWs, GWh, GWm
