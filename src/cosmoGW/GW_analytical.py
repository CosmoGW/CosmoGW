"""
GW_analytical.py is a Python routine that contains analytical
calculations and useful mathematical functions.

Adapted from the original GW_analytical in cosmoGW
(https://github.com/AlbertoRoper/cosmoGW),
created in Dec. 2021

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/GW_analytical.py

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

Caprini:2024gyk  - A. Roper Pol, I. Stomberg, C. Caprini, R. Jinno,
T. Konstandin, H. Rubira, "Gravitational waves from first-order
phase transitions: from weak to strong," JHEP, arxiv:2409.03651

Caprini:2024hue  - E. Madge, C. Caprini, R. Jinno, M. Lewicki,
M. Merchand, G. Nardini, M. Pieroni, A. Roper Pol, V. Vaskonen,
"Gravitational waves from first-order phase transitions in LISA:
reconstruction pipeline and physics interpretation,"
JCAP 10, 020 (2024), arxiv:2403.03723

RoperPol:2025b   - A. Roper Pol, A. Midiri, M. Salomé, C. Caprini,
"Modeling the gravitational wave spectrum from slowly decaying sources in the
early Universe: constant-in-time and coherent-decay models," in preparation
"""

import numpy as np
import matplotlib.pyplot as plt

### Reference slopes
a_ref   = 5    # Batchelor spectrum k^5
b_ref   = 2/3  # Kolmogorov spectrum k^(-2/3)
alp_ref = 2    # reference smoothness of broken power-law transition

####### ANALYTICAL FUNCTIONS USED FOR A SMOOTHED BROKEN POWER LAW #######

'''
    RoperPol:2022iel and RoperPol:2025b consider a spectral function
    defined such that the average squared field corresponds to

    <v^2> ~ 2 E* k* int zeta(K) dK,     in RoperPol:2022iel
    <v^2> ~ 2 E*    int zeta(K) dlnK,   in RoperPol:2025b

    The first convention can be chosen in the following functions if
    dlogK is set to False, while the second one is assumed when dlogK
    is True
'''

def check_slopes(a, b, dlogk=True):

    '''
    Function that checks the slopes used in power spectra to ensure
    that the integral over k or log k is converging.

    Arguments:
        a -- slope of the spectrum at low wave numbers, k^a
        b -- slope of the spectrum at high wave numbers, k^(-b)
        dlogk -- option to consider the spectral function per unit of k
                 or unit of logarithm of k (see comment above)
    '''

    conv = True
    if (dlogk):
        if b < 0 or a < 0:
            print('a and b have to be larger than 0')
            conv = False
    else:
        if b < 1 or a < -1:
            print('b has to be larger than 1 and a larger than -1')
            conv = False

    return conv

def smoothed_bPL(k, A=1., a=a_ref, b=b_ref, kpeak=1., alp=alp_ref, norm=True,
                 Omega=False, alpha2=False, piecewise=False, dlogk=True):

    """
    Function that returns the value of the smoothed broken power law (bPL) model
    for a spectrum of the form:

        zeta(K) = A x (b + abs(a))^(1/alp) K^a/[ b + c K^(alp(a + b)) ]^(1/alp),

    where K = k/kpeak, c = 1 if a = 0 or c = abs(a) otherwise.

    This spectrum is defined such that kpeak is the correct position of
    the peak and its maximum amplitude is given by A when norm is True.

    If norm is set to False, then the non-normalized spectrum is used:

        zeta (K) = A x K^a/(1 + K^(alp(a + b)))^(1/alp)

    Introduced in RoperPol:2022iel, see equation 6
    Main reference is RoperPol:2025b

    Arguments:

        k      -- array of wave numbers
        A      -- amplitude of the spectrum
        a      -- slope of the spectrum at low wave numbers, k^a
        b      -- slope of the spectrum at high wave numbers, k^(-b)
        kpeak  -- spectral peak, i.e., position of the break from k^a to k^(-b)
        alp    -- smoothness of the transition from one power law to the other
        norm   -- option to normalize the spectrum such that its peak is located at
                kpeak and its maximum value is A
        Omega  -- option to use the integrated energy density as the input A
        alpha2 -- option to use the alternative convention, such that the spectrum
                  takes the form: zeta(K) ~ K^a/( b + c K^alp )^((a + b)/alp)
        piecewise -- option to return a piecewise broken power law:
                     zeta(K) = K^a for K < 1, and K^(-b) for K > 1
                     corresponding to the alpha -> infinity limit
        dlogk  -- option to consider the spectral function per unit of k
                 or unit of logarithm of k (see comment above)

    Returns:
        spec -- spectrum array
    """

    conv = check_slopes(a, b, dlogk=dlogk)
    if not conv: return 0*k**0

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
                                     alpha2=alpha2, piecewise=piecewise,
                                     dlogk=dlogk)

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
        alp  -- smoothness parameter of the smoothed_bPL function
        n    -- n-moment of the integral
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

        a   -- slope of the spectrum at low wave numbers, k^a
        b   -- slope of the spectrum at high wave numbers, k^(-b)
        alp -- smoothness of the transition from one power law to the other
        n   -- moment of the integration

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
         piecewise=False, dlogk=True):

    '''
    Function that computes the parameter calA = Iab,0 that relates the
    peak and the integrated values of the smoothed_bPL spectrum

    References are RoperPol:2022iel, equation 8 (for dlogk False)
    and RoperPol:2025b, appendix A (for dlogk True)

    Same arguments as Iabn function with n = 0 or -1
    '''

    nn = 0
    if dlogk: nn = -1

    return Iabn(a=a, b=b, alp=alp, n=nn, norm=norm, alpha2=alpha2,
                piecewise=piecewise)

def calB(a=a_ref, b=b_ref, alp=alp_ref, n=1, norm=True, alpha2=False,
         piecewise=False, dlogk=True):

    '''
    Function that computes the parameter calB = Iab;-1/Iab;0 that relates the
    peak and the integral scale, cal B = xi kpeak

    Same arguments as Iabn function with n = -1 (Im1) and n = 0 (I0)
    Returns calB = (Im1/Im0)^n

    Main reference is RoperPol:2025b, appendix A (for dlogk True)
    '''

    nn = 0
    if dlogk: nn = -1

    Im1 = Iabn(a=a, b=b, alp=alp, n=-n + nn, norm=norm, alpha2=alpha2,
               piecewise=piecewise)
    I0 = Iabn(a=a, b=b, alp=alp, n=nn, norm=norm, alpha2=alpha2,
              piecewise=piecewise)
    calB = (Im1/I0)**n

    return calB

def zetam(a=a_ref, b=b_ref, alp=alp_ref, m=-1, norm=True, alpha2=False,
          piecewise=False):

    '''
    Function that computes the amplitude at the peak of the function

    K^m zeta(K)

    It allows to relate spectral functions using different normalizations,
    e.g., m = -1 allows to relate zeta used in RoperPol:2022iel with
    zeta used in RoperPol:2025b

    Main reference is RoperPol:2025b, appendix A

    Arguments:
        a      -- slope of the spectrum at low wave numbers, k^a
        b      -- slope of the spectrum at high wave numbers, k^(-b)
        kpeak  -- spectral peak, i.e., position of the break from k^a to k^(-b)
        alp    -- smoothness of the transition from one power law to the other
        m      -- power of K^m used to compute new function K^m zeta(K)
        norm   -- option to normalize the spectrum such that its peak is located at
                        kpeak and its maximum value is A
        alpha2 -- option to use the alternative convention, such that the spectrum
                          takes the form: zeta(K) ~ K^a/( b + c K^alp )^((a + b)/alp)
        piecewise -- option to return a piecewise broken power law:
                             zeta(K) = K^a for K < 1, and K^(-b) for K > 1
                             corresponding to the alpha -> infinity limit

    Returns:
        zetam -- amplitude of new function K^m zeta(K)
    '''

    if piecewise: return 1.

    if alpha2: alp2 = alp
    else: alp2 = alp*(a + b)
    zetam = ((a + m)/a)**((a + m)/alp2)*((b - m)/b)**((b - m)/alp2)

    return zetam

def Kstarm(a=a_ref, b=b_ref, alp=alp_ref, m=-1, norm=True, alpha2=False,
           piecewise=False):

    '''
    Function that computes the location of the peak of the function K^m zeta(K)

    Arguments are same as zetam
    Returns:
        Kstarm -- position of the peak of new function K^m zeta(K)
    '''

    if piecewise: return 1.

    if alpha2: alp2 = alp
    else: alp2 = alp*(a + b)
    Kstarm = (b/a*(a + m)/(b - m))**(1./alp2)

    return Kstarm

def calC(a=a_ref, b=b_ref, alp=alp_ref, tp='vort', norm=True, alpha2=False,
         piecewise=False, dlogk=True):

    '''
    Function that computes the parameter calC that allows to
    compute the TT-projected stress spectrum by taking the convolution of the
    smoothed bPL spectra over k and tilde p = |k - p|.

    It gives the value of the spectrum at K -> 0 limit as a prefactor
    pref that depends on the type of source and an integral over

    zeta^2/P^4     when dlogk is True (see appendix B of RoperPol:2025b)
    zeta^2/P^2     when dlogk is False (eq. 22 of RoperPol:2022iel for
                     vortical fields and eq. 46 of RoperPol:2023dzg
                     for compressional fields

    The spectrum of the stresses is then normalized such that
    (see appendix B of RoperPol:2025b)

    EPi = K^3 Epeak^2 calC zetaPi

    For the antisymmetric spectrum (helical) of the stresses
    the normalization becomes

    HPi = K^4 Hpeak Epeak calC zetaPi

    Arguments:

        a -- slope of the spectrum at low wave numbers, k^a
        b -- slope of the spectrum at high wave numbers, k^(-b)
        alp -- smoothness of the transition from one power law to the other
        tp -- type of sourcing field: 'vort', 'comp', 'hel' or 'mix' available
        norm -- option to normalize the spectrum such that its peak is located at
                        kpeak and its maximum value is A
        alpha2 -- option to use the alternative convention, such that the spectrum
                          takes the form: zeta(K) ~ K^a/( b + c K^alp )^((a + b)/alp)
        piecewise -- option to return a piecewise broken power law:
                             zeta(K) = K^a for K < 1, and K^(-b) for K > 1
                             corresponding to the alpha -> infinity limit
    '''

    pref = 1.

    if tp   == 'vort': pref = 28/15
    elif tp == 'comp': pref = 32/15
    elif tp == 'mix':  pref = 16/5
    elif tp == 'hel':  pref = 1/3

    if tp not in ['vort', 'comp', 'mix', 'hel', 'none']:
        print('tp has to be vortical (vort), compressional (comp),',
              'mixed (mix) or helical (hel)')
        print('returning pref = 1')

    nn = 0
    if dlogk: nn = 2

    return pref*Iabn(a=a*2, b=b*2, alp=alp/2, n=-2-nn, norm=norm,
                     alpha2=alpha2, piecewise=piecewise)

###### ANALYTICAL TEMPLATE USED FOR A DOUBLE SMOOTHED BROKEN POWER LAW ######

def smoothed_double_bPL(k, kpeak1, kpeak2, A=1., a=a_ref, b=1,
                        c=b_ref, alp1=alp_ref, alp2=alp_ref, kref=1.,
                        alpha2=False):

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

    The same broken power law with a slightly different form is used in
    Caprini:2024gyk, Caprini:2024hue and can be used setting alpha2 = True

        zeta(K) = A K^a/(1 + (K/K1)^alp1)^((a - b)/alp1)
                           x (1 + (K/K2)^alp2)^(-(b + c)/alp2)

    Arguments:

        k -- array of wave numbers
        kpeak1, kpeak2 -- peak positions
        A -- amplitude of the spectrum
        a -- slope of the spectrum at low wave numbers, k^a
        b -- slope of the spectrum at intermediate wave numbers, k^b
        c -- slope of the spectrum at high wave numbers, k^(-c)
        alp1, alp2 -- smoothness of the transitions from one power law to the other
        kref   -- reference wave number used to normalize the spectrum (default is 1)
        alpha2 -- option to use different normalization

    Returns:
        spectrum array
    """

    K  = k/kref
    K1 = kpeak1/kref
    K2 = kpeak2/kref

    if not alpha2:
        spec1 = (1 + (K/K1)**((a - b)*alp1))**(1/alp1)
        spec2 = (1 + (K/K2)**((c + b)*alp2))**(1/alp2)
    else:
        spec1 = (1 + (K/K1)**alp1)**((a - b)/alp1)
        spec2 = (1 + (K/K2)**alp2)**((c + b)/alp2)
    spec = A*K**a/spec1/spec2

    return spec
