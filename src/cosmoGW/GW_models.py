"""
GW_models.py is a Python routine that contains analytical and
semi-analytical models of cosmological GW backgrounds.

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/development/src/cosmoGW/GW_models.py

Author: Alberto Roper Pol
Created: 29/08/2024
Updated: 04/06/2025 (release cosmoGW 1.0: https://pypi.org/project/cosmoGW)

Other contributors: Antonino Midiri, Simona Procacci, Madeline Salomé

Main references are:

RoperPol:2022iel - A. Roper Pol, C. Caprini, A. Neronov, D. Semikoz,
"The gravitational wave signal from primordial magnetic fields in the
Pulsar Timing Array frequency band," Phys. Rev. D 105, 123502 (2022),
arXiv:2201.05630

RoperPol:2023dzg - A. Roper Pol, S. Procacci, C. Caprini,
"Characterization of the gravitational wave spectrum from sound waves within
the sound shell model," Phys. Rev. D 109, 063531 (2024), arXiv:2308.12943.

Hindmarsh:2019phv - M. Hindmarsh, M. Hijazi, "Gravitational waves from first order
cosmological phase transitions in the Sound Shell Model,"
JCAP 12 (2019) 062, arXiv:1909.10040

RoperPol:2025b - A. Roper Pol, A. Midiri, M. Salomé, C. Caprini,
"Modeling the gravitational wave spectrum from slowly decaying sources in the
early Universe: constant-in-time and coherent-decay models," in preparation
"""

import numpy as np
import matplotlib.pyplot as plt
import cosmoGW.plot_sets as plot_sets
import cosmoGW.hydro_bubbles as hb

# reference values
cs2_ref    = 1/3    # speed of sound squared
Nk_ref     = 1000   # reference number of wave number discretization
Nkconv_ref = 1000   # reference number of wave number discretization
                    # for convolution calculations
Np_ref     = 3000   # reference number of wave number discretization
                    # for convolution calculations
NTT_ref    = 5000   # reference number of lifetimes discretization

def EPi_correlators_ptilde(kp, p, k, EK, eps=1., eps_spec=False, Nptilde=100,
                           quiet=True, proj=True, q=.5, q_spec=False, tp='all'):

    '''
    Routine to compute the spectrum of the projected (anisotropic) stresses
    from the two-point correlator of the source (e.g. kinetic spectrum) under
    the assumption of Gaussianity.

    It computes the vortical and compressional components of the stress.

    It returns EPi/k^2

    References are:
    a) RoperPol:2022iel, equations 11 and 20, for vortical
    b) RoperPol:2023dzg, equation 45, for compressional
    c) RoperPol:2025b for generic fields
    '''

    kij, pij = np.meshgrid(kp, p, indexing='ij')
    K_pP = (kij + pij)
    K_mP = abs(kij - pij)
    K_mP[np.where(K_mP == 0)] = 1e-30

    ptilde = np.zeros((Nptilde, len(kp), len(p)))
    kij = np.zeros((Nptilde, len(kp), len(p)))
    pij = np.zeros((Nptilde, len(kp), len(p)))

    for i in range(0, len(kp)):
        for j in range(0, len(p)):
            ptilde[:, i, j] = np.logspace(np.log10(K_mP[i, j]),
                                          np.log10(K_pP[i, j]), Nptilde)
            kij[:, i, :] = kp[i]
            pij[:, :, j] = p[j]

    ptilde[np.where(ptilde == 0)] = 1e-50
    z = (pij**2 + kij**2 - ptilde**2)/(2*kij*pij)

    EK_ptilde = np.interp(ptilde, k, EK)
    if eps_spec: HK_ptilde = np.interp(ptilde, k, 2*eps*EK)/ptilde
    else: HK_ptilde = EK_ptilde*2*eps/ptilde

    kij, pij = np.meshgrid(kp, p, indexing='ij')
    EK_p = np.interp(pij, k, EK)
    if eps_spec: HK_p = np.interp(pij, k, 2*EK*eps)/pij
    else: HK_p = 2*EK_p*eps/pij

    if tp == 'vort' or tp == 'all':

        if proj:
            JJ_vort = .5*np.trapz(EK_ptilde/ptilde**3*(1 + z**2)* \
                      (2*ptilde**2 + pij**2*(z**2 - 1))/kij/pij,
                                  ptilde, axis=0)
        else:
            JJ_vort = .5*np.trapz(EK_ptilde/ptilde**3*(6*ptilde**2 \
                      + kij**2*(z**2 - 1))/kij/pij,
                                  ptilde, axis=0)

        zetaPi_vort = np.trapz(EK_p*JJ_vort, p, axis=1)

    if tp == 'comp' or tp == 'all':

        if proj:
            JJ_comp = 2*np.trapz(EK_ptilde/ptilde**3*(1 - z**2)**2* \
                      pij/kij, ptilde, axis=0)
        else:
            JJ_comp = 2*np.trapz(EK_ptilde/ptilde**3*(2*ptilde**2 + \
                      kij**2*(z**2 - 1))/kij/pij,
                                 ptilde, axis=0)
        zetaPi_comp = np.trapz(EK_p*JJ_comp, p, axis=1)

    if tp == 'all':
        return zetaPi_vort, zetaPi_comp

    elif tp == 'vort':
        return zetaPi_vort
    elif tp == 'comp':
        return zetaPi_comp

############### SOUND-SHELL MODEL FOR SOUND WAVES IN PTs ###############

'''
Kinetic spectra computed for the sound-shell model from f' and l functions.
f' and l functions need to be previously computed from the self-similar
fluid perturbations induced by expanding bubbles (see hydro_bubbles.py)
'''

def compute_kin_spec_ssm(z, vws, fp, l, sp='sum', type_n='exp', cs2=cs2_ref,
                         corr=True, min_qbeta=-4, max_qbeta=5, Nqbeta=Nk_ref,
                         min_TT=-1, max_TT=3, NTT=NTT_ref, corr=False,
                         dens=True, normbeta=True):

    '''
    Function that computes the kinetic power spectral density assuming
    exponential or simultaneous nucleation.

    Reference: Equations (32)--(37) of RoperPol:2023dzg

    Arguments:
        z -- array of values of z
        vws -- array of wall velocities
        fp -- function f'(z) computed from the hydro_bubble module using fp_z
        l -- function lambda(z) computed from the hydro_bubble module using fp_z
              (using lz = True)
        sp -- type of function computed for the kinetic spectrum description
        type_n -- type of nucleation hystory (default is exponential 'exp',
                  another option is simultaneous 'sym')
        cs2 -- square of the speed of sound (default 1/3)
        corr -- option to 'correct' Rstar beta with max(vw, cs) (default is False)
        dens -- option to return power spectral density (if True, default), or kinetic
                spectrum (if False)
        normbeta -- normalization of k with beta (default is True).
                    If normbeta is False, k is normalized with Rstar

    Returns:
        qbeta -- wave number, normalized with 1/beta or Rstar
        P     -- power spectral density (power spectrum if dens is False)
    '''

    if sp == 'sum':    A2 = .25*(cs2*l**2 + fp**2)
    if sp == 'only_f': A2 = .5*fp**2
    if sp == 'diff':   A2 = .25*(fp**2 - cs2*l**2)
    if sp == 'cross':  A2 = -.5*fp*np.sqrt(cs2)*l

    qbeta = np.logspace(min_qbeta, max_qbeta, Nqbeta)
    TT = np.logspace(min_TT, max_TT, NTT)
    q_ij, TT_ij = np.meshgrid(qbeta, TT, indexing='ij')
    Pv = np.zeros((len(vws), len(qbeta)))

    funcT = np.zeros((len(vws), len(qbeta), len(TT)))
    for i in range(0, len(vws)):
        if type_n == 'exp':
            funcT[i, :, :] = np.exp(-TT_ij)*TT_ij**6*np.interp(TT_ij*q_ij, z, A2[i, :])
        if type_n == 'sim':
            funcT[i, :, :] = .5*np.exp(-TT_ij**3/6)*TT_ij**8*np.interp(TT_ij*q_ij, z, A2[i, :])
        Pv[i, :] = np.trapz(funcT[i, :, :], TT, axis=1)

    if dens == False:
        Rstar_beta = hb.Rstar_beta(vw=vws[i], cs2=cs2, corr=corr)
        for i in range(0, len(vws)):
            pref = qbeta[i, :]**2/Rstar_beta[i]**4/(2*np.pi**2)
            Pv[i, :] *= pref

    if normbeta == False:
        if not dens: Rstar_beta = hb.Rstar_beta(vw=vws[i], cs2=cs2, corr=corr)
        for i in range(0, len(vws)):
            qbeta[i, :] *= Rstar_beta[i]

    return qbeta, Pv

def OmGW_ssm_HH19(k, EK, Np=Np_ref, Nk=Nkconv_ref, plot=False, cs2=cs2_ref):

    '''
    Function to compute GW spectrum using the approximation
    introduced in the first sound-shell model analysis of Hindmarsh:2019phv
    under the delta assumption for the 'stationary' term
    (see appendix B of RoperPol:2023dzg for details).

    The resulting GW spectrum is

     Omega_GW (k) = (3pi)/(8cs) x (k/kst)^2 x (K/KK)^2 x TGW x Omm(k)

    Reference: Appendix B of RoperPol:2023dzg; see eq.(B3)
    '''

    cs = np.sqrt(cs2)
    kp = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)

    p_inf = kp*(1 - cs)/2/cs
    p_sup = kp*(1 + cs)/2/cs

    Omm = np.zeros(len(kp))
    for i in range(0, len(kp)):

        p = np.logspace(np.log10(p_inf[i]), np.log10(p_sup[i]), Np)
        ptilde = kp[i]/cs - p
        z = -kp[i]*(1 - cs2)/2/p/cs2 + 1/cs

        EK_p = np.interp(p, k, EK)
        EK_ptilde = np.interp(ptilde, k, EK)

        Omm1 = (1 - z**2)**2*p/ptilde**3*EK_p*EK_ptilde
        Omm[i] = np.trapz(Omm1, p)

    return kp, Omm

def effective_ET_correlator_stat(k, EK, tfin, Np=Np_ref, Nk=Nkconv_ref,
                                 plot=False, expansion=True, kstar=1.,
                                 extend=False, largek=3, smallk=-3, tini=1,
                                 cs2=cs2_ref, terms='all',
                                 inds_m=[], inds_n=[], corr_Delta_0=False):

    """
    Function that computes the normalized GW spectrum zeta_GW(k)
    from the velocity field spectrum for purely compressional anisotropic stresses,
    assuming Gaussianity, and under the assumption of stationary UETC (e.g.,
    sound waves under the sound-shell model).

    august/2024 (alberto): slightly modified from previous version now to
    compute the combination Delta0 C zeta_GW

    Reference: RoperPol:2023dzg, eq. 90

    Arguments:
        k -- array of wave numbers
        EK -- array of values of the kinetic spectrum
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
        Omm -- GW spectrum normalized (zeta_GW)
        kp -- final array of wave numbers
    """

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

    Delta_mn = kij**0 - 1

    if terms == 'all':
        inds_m = [-1, 1]
        inds_n = [-1, 1]
        Delta_mn = np.zeros((4, len(kp), len(p), len(z)))
    tot_inds = 0
    l = 0
    for m in inds_m:
        for n in inds_n:
            Delta_mn[l, :, :, :] = \
                    compute_Delta_mn(tfin, kij*kstar, pij*kstar, ptilde*kstar,
                                     cs2=cs2, m=m, n=n, tini=tini, expansion=expansion)
            l += 1

    if l != 0: Delta_mn = Delta_mn/(l + 1)

    Omm = np.zeros((l + 1, len(kp)))
    for i in range(0, l):
        Pi_1 = np.trapz(EK_ptilde/ptilde**4*(1 - zij**2)**2*Delta_mn[i, :, :, :], z, axis=2)
        kij, EK_pij = np.meshgrid(kp, EK_p, indexing='ij')
        kij, pij = np.meshgrid(kp, p, indexing='ij')
        Omm[i, :] = np.trapz(Pi_1*pij**2*EK_pij, p, axis=1)

    return kp, Omm

def compute_Delta_mn(t, k, p, ptilde, cs2=cs2_ref, m=1, n=1, tini=1.,
                     expansion=True):

    '''
    Function that computes the integrated Green's functions and the stationary
    UETC 4 Delta_mn used in the sound shell model for the computation of the GW
    spectrum.

    Reference: RoperPol:2023dzg, eqs.56-59

    Arguments:
        t -- time
        k -- wave number k
        p -- wave number p to be integrated over
        ptilde -- second wave number tilde p to be integrated over
        expansion -- option to include the effect of the expansion of the Universe
                     (default True)
        cs2 -- square of the speed of sound (default is 1/3)
    '''

    cs = np.sqrt(cs2)
    pp = n*k + cs*(m*ptilde + p)
    pp[np.where(pp == 0)] = 1e-50

    if expansion:

        import scipy.special as spe
        si_t, ci_t = spe.sici(pp*t)
        si_tini, ci_tini = spe.sici(pp*tini)

        # compute Delta Ci^2 and Delta Si^2
        DCi = ci_t - ci_tini
        DSi = si_t - si_tini

        Delta_mn = DCi**2 + DSi**2

    else:

        Delta_mn = 2*(1 - np.cos(pp*(t - tini)))/pp**2

    return Delta_mn

############ Computation of the vorticity production in a radiation-dominated universe ############

def Evort_conv(kp, p, k, E, Nz=100, ampl=False, cs=1.):

    '''
    Calculation of the spectrum of the vorticity rate of growth.

    Input Ps is the power spectrum per logarithmic interval of k
    '''

    z = np.linspace(-1, 1, Nz)
    kij, pij, zij = np.meshgrid(kp, p, z, indexing='ij')

    ptilde = np.sqrt(kij**2 + pij**2 - 2*pij*kij*z)
    # avoid division by zero
    ptilde[np.where(ptilde < 1e-30)] = 1e-30

    E_ptilde = np.interp(ptilde, k, E)

    II = E_ptilde*(1 - zij**2)*(2*pij*zij - kij)/ptilde**4*pij**2
    I = np.trapz(II, z, axis=2)

    kij, pij = np.meshgrid(kp, p, indexing='ij')
    E_p = np.interp(pij, k, E)

    I = kp**4*np.trapz(I*E_p, p)*cs**4

    if ampl:

        CC = 4/3/np.pi*np.trapz(E**2, k)

        return I, CC

    else: return I
