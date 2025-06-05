"""
GW_analytical.py is a Python routine that contains analytical
calculations and useful mathematical functions.

Adapted from the original GW_analytical in cosmoGW
(https://github.com/AlbertoRoper/cosmoGW),
created in Dec. 2021

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/development/src/cosmoGW/GW_analytical.py

Author: Alberto Roper Pol
Created: 01/12/2021
Updated: 31/08/2024
Updated: 04/06/2025 (release cosmoGW 1.0: https://pypi.org/project/cosmoGW)

Other contributors: Antonino Midiri, Madeline Salomé

Main references are:

RoperPol:2022iel - A. Roper Pol, C. Caprini, A. Neronov, D. Semikoz,
"The gravitational wave signal from primordial magnetic fields in the
Pulsar Timing Array frequency band," Phys. Rev. D 105, 123502 (2022),
arXiv:2201.05630

RoperPol:2023bqa - A. Roper Pol, A. Neronov, C. Caprini, T. Boyer,
D. Semikoz, "LISA and γ-ray telescopes as multi-messenger probes of a
first-order cosmological phase transition," arXiv:2307.10744 (2023)

RoperPol:2023dzg - A. Roper Pol, S. Procacci, C. Caprini,
"Characterization of the gravitational wave spectrum from sound waves within
the sound shell model," Phys. Rev. D 109, 063531 (2024), arXiv:2308.12943.

RoperPol:2025b - A. Roper Pol, A. Midiri, M. Salomé, C. Caprini,
"Modeling the gravitational wave spectrum from slowly decaying sources in the
early Universe: constant-in-time and coherent-decay models," in preparation
"""

import numpy as np
import matplotlib.pyplot as plt
import cosmoGW.plot_sets as plot_sets

# reference values
a_ref   = 4          # Batchelor spectrum k^4
b_ref   = 5/3        # Kolmogorov spectrum k^(-5/3)
alp_ref = 2          # reference smoothness of broken power-law transition

####### ANALYTICAL FUNCTIONS USED FOR A SMOOTHED BROKEN POWER LAW #######

def smoothed_bPL(k, A=1., a=a_ref, b=b_ref, kpeak=1., alp=alp_ref, norm=True,
                 Omega=False, alpha2=False, piecewise=False):

    """
    Function that returns the value of the smoothed broken power law (bPL) model
    for a spectrum of the form:

        zeta(K) = A x (b + abs(a))^(1/alp) K^a/[ b + c K^(alp(a + b)) ]^(1/alp),

    where K = k/kpeak, c = 1 if a = 0 or c = abs(a) otherwise.

    This spectrum is defined such that kpeak is the correct position of
    the peak and its maximum amplitude is given by A.

    If norm is set to False, then the non-normalized spectrum is used:

        zeta (K) = A x K^a/(1 + K^(alp(a + b)))^(1/alp)

    The function is only correctly defined when b > 0 and a + b >= 0

    Introduced in RoperPol:2022iel, see equation 6
    Main reference is RoperPol:2025b

    Arguments:

        k -- array of wave numbers
        A -- amplitude of the spectrum
        a -- slope of the spectrum at low wave numbers, k^a
        b -- slope of the spectrum at high wave numbers, k^(-b)
        kpeak -- spectral peak, i.e., position of the break from k^a to k^(-b)
        alp -- smoothness of the transition from one power law to the other
        norm -- option to normalize the spectrum such that its peak is located at
                kpeak and its maximum value is A
        Omega -- option to use the integrated energy density as the input A
        alpha2 -- option to use the alternative convention, such that the spectrum
                  takes the form: zeta(K) ~ K^a/( b + c K^alpha )^((a + b)/alpha)
        piecewise -- option to return a piecewise broken power law:
                     zeta(K) = K^a for K < 1, and K^(-b) for K > 1
                    corresponding to the alpha -> infinity limit

    Returns:
        spec -- spectrum array
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

    B(a, b; x -> infinity) = int_0^x u^(a - 1) (1 - u)^(b - 1) du

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

    Reference: appendix A of RoperPol:2025b
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
    Function that computes the moment n of the smoothed bPL spectra,
    defined in smoothed_bPL for A = 1, kpeak = 1:

    int K^n zeta(K) dK

    Reference: appendix A of RoperPol:2025b

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

    '''
    Function that computes the parameter calA = Iab,0 that relates the
    peak and the integrated values of the smoothed_bPL spectrum

    References are RoperPol:2022iel, equation 8, and RoperPol:2025b, appendix A

    Same arguments as Iabn function with n = 0
    '''

    return Iabn(a=a, b=b, alp=alp, n=0, norm=norm, alpha2=alpha2,
                piecewise=piecewise)

def calB(a=a_ref, b=b_ref, alp=alp_ref, n=1, norm=True, alpha2=False,
         piecewise=False):

    '''
    Function that computes the parameter calB = Iab;-1/Iab;0 that relates the
    peak and the integral scale, cal B = xi kpeak

    Same arguments as Iabn function with n = -1 (Im1) and n = 0 (I0)
    Returns calB = (Im1/Im0)^n
    '''

    Im1 = Iabn(a=a, b=b, alp=alp, n=-n, norm=norm, alpha2=alpha2,
               piecewise=piecewise)
    I0 = Iabn(a=a, b=b, alp=alp, n=0, norm=norm, alpha2=alpha2,
              piecewise=piecewise)
    calB = (Im1/I0)**n

    return calB

def calC(a=a_ref, b=b_ref, alp=alp_ref, tp='vort', norm=True, alpha2=False,
         piecewise=False, proj=True, q=1.):

    '''
    Function that computes the parameter calC that allows to
    compute the TT-projected stress spectrum by taking the convolution of the
    smoothed bPL spectra over k and tilde p = |k - p|.

    It gives the spectrum of the stress of Gaussian vortical non-helical fields
    as

    P_Pi (0) = 2 pi^2 EM*^2 calC / k*

    References are:
    a) RoperPol:2022iel, equation 22, for vortical
    b) RoperPol:2023dzg, equation 46, for compressional
    c) RoperPol:2025b (appendix A) for generic fields

    Arguments:

        a -- slope of the spectrum at low wave numbers, k^a
        b -- slope of the spectrum at high wave numbers, k^(-b)
        alp -- smoothness of the transition from one power law to the other
        tp -- type of sourcing field: 'vort' or 'comp' available
    '''

    if proj:
        if tp   == 'vort': pref = 28/15
        elif tp == 'comp': pref = 32/15
        elif tp == 'mix':  pref = 16/5*q*(1 - q)
        elif tp == 'hel':  pref = -4/3
    else:
        if tp   == 'vort': pref = 6
        if tp   == 'comp': pref = 8
        if tp   == 'mix':  pref = 8*q*(1 - q)
        if tp   == 'hel':  pref = -2

    if tp not in ['vort', 'comp', 'mix', 'hel']:
        print('tp has to be vortical (vort), compressional (comp),',
              'mixed (mix) or helical (hel)')
        pref = 0.

    return pref*Iabn(a=a*2, b=b*2, alp=alp/2, n=-2, norm=norm,
                     alpha2=alpha2, piecewise=piecewise)

###### ANALYTICAL TEMPLATE USED FOR A DOUBLE SMOOTHED BROKEN POWER LAW ######

def smoothed_double_bPL(k, kpeak1, kpeak2, A=1., a=a_ref, b=1,
                        c=b_ref, alp1=alp_ref, alp2=alp_ref, kref=1.):

    """
    Function that returns the value of the smoothed double broken power
    law (double_bPL) model with a spectrum of the form:

        zeta(K) = A K^a/(1 + (K/K1)^[(a - b)*alp1])^(1/alp1)
                  x (1 + (K/K2)^[(c + b)*alp2])^(-1/alp2)

    where K = k/kref, K1 and K2 are the two position peaks,
    a is the low-k slope, b is the intermediate slope,
    and -c is the high-k slope.
    alp1 and alp2 are the smoothness parameters for each spectral transition.

    Reference is RoperPol:2023dzg, equation 50
    Also used in RoperPol:2023bqa, equation 7

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

    K  = k/kref
    K1 = kpeak1/kref
    K2 = kpeak2/kref

    spec1 = (1 + (K/K1)**((a - b)*alp1))**(1/alp1)
    spec2 = (1 + (K/K2)**((c + b)*alp2))**(1/alp2)
    spec  = A*K^a/spec1/spec2

    return spec
