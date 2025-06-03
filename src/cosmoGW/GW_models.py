"""
GW_models.py is a Python routine that contains analytical and semi-analytical
models and templates of cosmological GW backgrounds.

Author: Alberto Roper Pol
Created: 29/08/2024

Other contributors: Antonino Midiri, Simona Procacci, Madeline Salome

Main references are:

RPCNS22 - A. Roper Pol, C. Caprini, A. Neronov, D. Semikoz, "The gravitational wave
signal from primordial magnetic fields in the Pulsar Timing Array frequency band,"
Phys. Rev. D 105, 123502 (2022), arXiv:2201.05630

RPPC23 - A. Roper Pol, S. Procacci, C. Caprini, "Characterization of the gravitational wave spectrum
from sound waves within the sound shell model," Phys.Rev.D 109 (2024) 6, 063531, arXiv:2308.12943

HH19 - M. Hindmarsh, M. Hijazi, "Gravitational waves from first order
cosmological phase transitions in the Sound Shell Model,"
JCAP 12 (2019) 062, arXiv:1909.10040
"""

import numpy as np
import matplotlib.pyplot as plt
import plot_sets

import os
HOME = os.getcwd()

### Reference values
cs2 = 1/3      # speed of sound

#################### SOUND-SHELL MODEL FOR SOUND WAVES IN PTs ####################

####### Kinetic spectra computed for the sound-shell model from f' and l functions
####### f' and l functions need to be previously computed from the self-similar
####### fluid perturbations induced by expanding bubbles (see hydro_bubbles.py)

def compute_kin_spec_dens(z, vws, fp, l, sp='sum', type_n='exp', cs2=cs2, min_qbeta=-4,
                          max_qbeta=5, Nqbeta=1000, min_TT=-1, max_TT=3, NTT=5000):

    '''
    Function that computes the kinetic power spectral density assuming exponential or simultaneous
    nucleation.

    Arguments:
        z -- array of values of z
        vws -- array of wall velocities
        fp -- function f'(z) computed from the sound-shell model using compute_fs
        l -- function lambda(z) computed from the sound-shell model using compute_fs
        sp -- type of function computed for the kinetic spectrum description
        type_n -- type of nucleation hystory (default is exponential 'exp',
                  another option is simultaneous 'sym')
        dens_spec -- option to return power spectral density (if True, default), or kinetic
                     spectrum (if False)
    '''

    if sp == 'sum': A2 = .25*(cs2*l**2 + fp**2)

    q_beta = np.logspace(min_qbeta, max_qbeta, Nqbeta)
    TT = np.logspace(min_TT, max_TT, NTT)
    q_ij, TT_ij = np.meshgrid(q_beta, TT, indexing='ij')
    Pv = np.zeros((len(vws), len(q_beta)))

    funcT = np.zeros((len(vws), len(q_beta), len(TT)))
    for i in range(0, len(vws)):
        if type_n == 'exp':
            funcT[i, :, :] = np.exp(-TT_ij)*TT_ij**6*np.interp(TT_ij*q_ij, z, A2[i, :])
        if type_n == 'sim':
            funcT[i, :, :] = .5*np.exp(-TT_ij**3/6)*TT_ij**8*np.interp(TT_ij*q_ij, z, A2[i, :])
        Pv[i, :] = np.trapz(funcT[i, :, :], TT, axis=1)

    return q_beta, Pv

def compute_kin_spec(vws, q_beta, Pv, corr=True, cs2=cs2):

    '''
    Function that computes the kinetic spectrum as a function of k Rast from
    the power spectral density as a function of q/beta
    '''

    EK = np.zeros((len(vws), len(q_beta)))
    cs = np.sqrt(cs2)
    if corr: Rstar_beta = (8*np.pi)**(1/3)*np.maximum(vws, cs)
    else: Rstar_beta = (8*np.pi)**(1/3)*vws
    kks = np.zeros((len(vws), len(q_beta)))

    for i in range(0, len(vws)):
        kks[i, :] = q_beta*Rstar_beta[i]
        pref = kks[i, :]**2/Rstar_beta[i]**6/(2*np.pi**2)
        EK[i, :] = pref*Pv[i, :]

    return kks, EK

##
## function to compute GW spectrum time growth using the approximation
## introduced in the first sound-shell model analysis of HH19
##
## The resulting GW spectrum is
##
##  Omega_GW (k) = (3pi)/(8cs) x (k/kst)^2 x (K/KK)^2 x TGW x Omm(k)
##

def OmGW_ssm_HH19(k, EK, Np=3000, Nk=60, plot=False, cs2=cs2):

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

######################################### NOT PUBLIC ON GITHUB #########################################

# moved from GW_analytical
def ET_correlator_compr(k, EK, Np=3000, Nk=60):

    '''
    Function to compute the spectrum of the anisotropic (TT) stresses ~ u_i u_j (TT) of a
    fully compressional field u_i, such that curl(u) = 0, under
    the assumption that the statistical distribution of the field u_i is Gaussian.

    When the input are the field spectrum (e.g. velocity spectrum) and the wave number k,
    the output can be used to compute the spectrum of the anisotropic stresses using:

    EPi(k) = k^2 x Pi2

    If the input are the normalized kinetic spectrum zeta(K), where K = k/k*,
    then the kinetic spectrum can be found using:

    k EPi(k) = K^3 x OmM^2/AA^2 x Pi2

    where AA is the geometric factor from the spectral shape and OmM^2 is the
    integrated .5 <u^2>
    '''

    p = np.logspace(np.log10(k[0]), np.log10(k[-1]), Np)
    kp = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)

    Nz = 500
    z = np.linspace(-1, 1, Nz)
    kij, pij, zij = np.meshgrid(kp, p, z, indexing='ij')
    ptilde2 = pij**2 + kij**2 - 2*kij*pij*zij
    ptilde = np.sqrt(ptilde2)

    EK_p = np.interp(p, k, EK)
    EK_ptilde = np.interp(ptilde, k, EK)
    ptilde[np.where(ptilde == 0)] = 1e-50

    Pi_1 = np.trapz(EK_ptilde/ptilde**4*(1 - zij**2)**2, z, axis=2)
    kij, EK_pij = np.meshgrid(kp, EK_p, indexing='ij')
    kij, pij = np.meshgrid(kp, p, indexing='ij')
    Pi2 = 2*np.trapz(Pi_1*pij**2*EK_pij, p, axis=1)

    return kp, Pi2

def EPi_correlators_ptilde(kp, p, k, EK, eps=1., eps_spec=False, Nptilde=100,
                           quiet=True, proj=True, q=.5, q_spec=False, tp='all'):

    '''
    Routine to compute the spectrum of the projected (anisotropic) stresses
    integrating over ptilde instead of z (see ET_correlator).

    It computes the vortical, compressional, mixed, helical, and polarization
    components of the stress.

    It returns EPi/k^2
    '''

    kij, pij = np.meshgrid(kp, p, indexing='ij')
    K_pP = (kij + pij)
    K_mP = abs(kij - pij)
    K_mP[np.where(K_mP == 0)] = 1e-30

    ptilde = np.zeros((Nptilde, len(kp), len(p)))
    kij = np.zeros((Nptilde, len(kp), len(p)))
    pij = np.zeros((Nptilde, len(kp), len(p)))

    for i in range(0, len(kp)):
        #if not quiet: print('computed ', i + 1, '/', len(kp))
        for j in range(0, len(p)):
            ptilde[:, i, j] = np.logspace(np.log10(K_mP[i, j]), np.log10(K_pP[i, j]), Nptilde)
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
            JJ_vort = .5*np.trapz(EK_ptilde/ptilde**3*(1 + z**2)*(2*ptilde**2 + pij**2*(z**2 - 1))/kij/pij,
                              ptilde, axis=0)
        else:
            JJ_vort = .5*np.trapz(EK_ptilde/ptilde**3*(6*ptilde**2 + kij**2*(z**2 - 1))/kij/pij,
                              ptilde, axis=0)
        zetaPi_vort = np.trapz(EK_p*JJ_vort, p, axis=1)

    if tp == 'comp' or tp == 'all':

        if proj:
            JJ_comp = 2*np.trapz(EK_ptilde/ptilde**3*(1 - z**2)**2*pij/kij, ptilde, axis=0)
        else:
            JJ_comp = 2*np.trapz(EK_ptilde/ptilde**3*(2*ptilde**2 + kij**2*(z**2 - 1))/kij/pij,
                                 ptilde, axis=0)
        zetaPi_comp = np.trapz(EK_p*JJ_comp, p, axis=1)

    if tp == 'hel' or tp == 'all':

        if proj:
            JJ_hel = .5*np.trapz(HK_ptilde/ptilde*z*(kij - pij*z)/kij, ptilde, axis=0)
        else:
            JJ_hel = .25*np.trapz(HK_ptilde/ptilde*(kij*z - pij)/kij, ptilde, axis=0)
        zetaPi_hel = np.trapz(HK_p*JJ_hel, p, axis=1)

    if tp == 'pol_comp' or tp=='all':

        proj = True

        JJ_polcomp1 = np.trapz(HK_ptilde/ptilde*(kij - pij*z)*(1 - z**2)/kij/pij,
                              ptilde, axis=0)
        JJ_polcomp2 = np.trapz(EK_ptilde/ptilde**3*z*(1 - z**2)/kij*pij**2,
                              ptilde, axis=0)
        zetaPi_polcomp = np.trapz(EK_p*JJ_polcomp1 + HK_p*JJ_polcomp2, p, axis=1)

    if tp == 'pol_vort' or tp=='all':

        proj = True

        JJ_polvort1 = .5*np.trapz(HK_ptilde/ptilde*(kij - pij*z)*(1 + z**2)/kij/pij,
                              ptilde, axis=0)
        JJ_polvort2 = .5*np.trapz(EK_ptilde/ptilde**3*z*(2*ptilde**2 - pij**2*(1 - z**2))/kij,
                              ptilde, axis=0)

        zetaPi_polvort = np.trapz(EK_p*JJ_polvort1 + HK_p*JJ_polvort2, p, axis=1)

    if tp == 'mix' or tp=='all':

        if q_spec:
            EK_ptilde = np.interp(ptilde, k, EK*(1 - q))
            EK_ptildecomp = np.interp(ptilde, k, EK*q)
            EK_p = np.interp(pij, k, EK*(1 - q))
            EK_pcomp = np.interp(pij, k, EK*q)
        else:
            EK_ptildecomp = EK_ptilde*q
            EK_ptilde = EK_ptilde*(1 - q)
            EK_pcomp = EK_p*q
            EK_p = EK_p*(1 - q)

        if proj:
            aux_mix1 = (1 - z**2)*(2*ptilde**2 - pij**2*(1 - z**2))
            aux_mix2 = (1 - z**4)*pij**2
            JJ_mix1 = np.trapz(EK_ptilde*aux_mix1/ptilde**3/kij/pij, ptilde, axis=0)
            JJ_mix2 = np.trapz(EK_ptildecomp*aux_mix2/ptilde**3/kij/pij, ptilde, axis=0)
        else:
            aux_mix = 2*ptilde**2 + kij**2*(1 - z**2)
            JJ_mix1 = np.trapz(EK_ptilde*aux_mix/ptilde**3/kij/pij, ptilde, axis=0)
            JJ_mix2 = np.trapz(EK_ptildecomp*aux_mix/ptilde**3/kij/pij, ptilde, axis=0)
        zetaPi_mix = np.trapz(EK_p*JJ_mix2 + EK_pcomp*JJ_mix1, p, axis=1)

    if tp == 'all':
        return zetaPi_vort, zetaPi_comp, zetaPi_mix, zetaPi_hel, zetaPi_polcomp, zetaPi_polvort

    if tp == 'vort':
        return zetaPi_vort
    if tp == 'comp':
        return zetaPi_comp
    if tp == 'mix':
        return zetaPi_mix
    if tp == 'hel':
        return zetaPi_hel
    if tp == 'pol_vort':
        return zetaPi_polvort
    if tp == 'pol_comp':
        return zetaPi_polcomp

def ET_correlator(k, EK, Np=3000, Nk=60, proj=True, tp='comp', norm=True, kp=[],
                  p=[], extend=False, largek=3, smallk=-3, EK2=[], eps=1, Nz=500):

    '''
    Function to compute the spectrum of the stresses Tij ~ u_i u_j of a
    field u_i under the assumption that the statistical distribution of the
    field u_i is Gaussian.

    When proj is True, it is used to compute the anisotropic stresses Pi_ij ~ u_i u_j (TT),
    obtained from projecting Tij with the TT projector \Lambda_ijlm

    When norm is False, the output is ET(k) or EPi(k), when norm is True, the
    anisotropic stresses spectrum can be computed from the output C zetaPi = Pi2 as

    k EPi(k) = K^3 x CC x (OmM/AA)^2 x Pi2

    where AA is the geometric factor from the spectral shape, OmM^2 is the
    integrated .5 <u^2>, and CC is the parameter to normalize zetaPi -> 1 when
    K -> 0

    Options are tp = 'vort' for fully vortical fields, 'comp' for fully
    compressional fields, 'mix' for fields with both compressional (EK) and
    vortical (EK2), 'hel' for the contribution from the helical component
    of the spectrum, 'pol' for the computation of the projected antisymmetric
    spectrum of Pi_ij
    '''

    if len(p) == 0: p = np.logspace(np.log10(k[0]), np.log10(k[-1]), Np)

    if len(kp) == 0:
        kp = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)

        if extend:
            Nk = int(Nk/6)
            kp = np.logspace(smallk, np.log10(k[0]), Nk)
            kp = np.append(kp, np.logspace(np.log10(k[0]), np.log10(k[-1]),
                           4*Nk))
            kp = np.append(kp, np.logspace(np.log10(k[-1]), largek, Nk))

    z = np.linspace(-1, 1, Nz)
    kij, pij, zij = np.meshgrid(kp, p, z, indexing='ij')
    ptilde2 = pij**2 + kij**2 - 2*kij*pij*zij
    ptilde = np.sqrt(ptilde2)

    EK_p = np.interp(p, k, EK)
    EK_ptilde = np.interp(ptilde, k, EK)
    ptilde[np.where(ptilde == 0)] = 1e-50
    ptilde2[np.where(ptilde == 0)] = 1e-50
    if tp == 'hel' or tp == 'pol':
        HK_p = 2*np.interp(p, k, EK*eps)/p
        HK_ptilde = 2*np.interp(ptilde, k, EK*eps)/ptilde

    if tp == 'vort':
        if proj:
            Pi0 = EK_ptilde/ptilde2**2*(1 + zij**2)
            Pi1 = np.trapz(Pi0*(2*ptilde2 - pij**2*(1 - zij**2)), z, axis=2)
        else:
            Pi1 = np.trapz(EK_ptilde/ptilde2**2*(6*ptilde2 - kij**2*(1 - zij**2)), z, axis=2)
    elif tp == 'comp':
        if proj: Pi1 = np.trapz(EK_ptilde/ptilde2**2*(1 - zij**2)**2, z, axis=2)
        else: Pi1 = np.trapz(EK_ptilde/ptilde2**2*(2*ptilde2 - kij**2*(1 - zij**2)), z, axis=2)
    elif tp == 'mix':
        # check if second spectrum is given, assuming EK is the compressional component
        # and EK2 is the vortical one
        if len(EK2) == 0: EK2 = EK
        EK_p2 = np.interp(p, k, EK2)
        EK_ptilde2 = np.interp(ptilde, k, EK2)

        if proj:
            Pi1 = np.trapz(EK_ptilde/ptilde2**2*(1 - zij**4), z, axis=2)
            Pi0_2 = EK_ptilde2/ptilde2*(1 - zij**2)
            Pi1_2 = np.trapz(Pi0_2*(2 - pij**2/ptilde2*(1 - zij**2)), z, axis=2)
        else:
            Pi1 = np.trapz(EK_ptilde/ptilde2**2*(2*ptilde2 + kij**2*(1 - zij**2)), z, axis=2)
            Pi1_2 = np.trapz(EK_ptilde2/ptilde2**2*(2*ptilde2 + kij**2*(1 - zij**2)), z, axis=2)

    elif tp == 'hel':
        if proj: Pi1 = np.trapz(HK_ptilde/ptilde2*zij*(kij - pij*zij), z, axis=2)
        else: Pi1 = np.trapz(HK_ptilde/ptilde2*(kij*zij - pij), z, axis=2)

    elif tp == 'pol':
        Pi1 = np.trapz(HK_ptilde/ptilde2*(1 + zij**2)*(kij - pij*zij), z, axis=2)

    kij, EK_pij = np.meshgrid(kp, EK_p, indexing='ij')
    kij, pij = np.meshgrid(kp, p, indexing='ij')

    if tp == 'vort': Pi2 = .5*np.trapz(Pi1*EK_pij, p, axis=1)
    elif tp == 'comp':
        if proj: Pi2 = 2*np.trapz(Pi1*pij**2*EK_pij, p, axis=1)
        else: Pi2 = 2*np.trapz(Pi1*EK_pij, p, axis=1)
    elif tp == 'mix':
        kij, EK_pij2 = np.meshgrid(kp, EK_p2, indexing='ij')
        if proj:
            Pi2 = np.trapz(Pi1*EK_pij2*pij**2, p, axis=1)
            Pi2 += np.trapz(Pi1_2*EK_pij, p, axis=1)
        else:
            Pi2 = np.trapz(Pi1*EK_pij2, p, axis=1)
            Pi2 += np.trapz(Pi1_2*EK_pij, p, axis=1)

    elif tp == 'hel':
        kij, HK_pij = np.meshgrid(kp, HK_p, indexing='ij')
        Pi2 = .5*np.trapz(Pi1*HK_pij*pij, p, axis=1)

    elif tp == 'pol':
        Pi2 = np.trapz(Pi1*EK_pij, p, axis=1)

    if not norm:
        if tp == 'vort': EPi = .5*kp**2*Pi2
        elif tp == 'comp': EPi = 2*kp**2*Pi2
        elif tp == 'mix': EPi = kp**2*Pi2
        elif tp == 'hel': Epi = .5*kp**2*Pi2
        elif tp == 'pol': EPi = kp**2*Pi2

    else: EPi = Pi2

    return kp, EPi

def calC(k, zeta, tp='comp'):

    '''
    Function that computes the integral of zeta^2(K)/K^2, corresponding to the value of
    the anisotropic stresses in the K->0 limit, from the normalized kinetic spectrum

    Reference: RPPC23, eq. 46 (note that we use here a definition of calC = C/2 that of
    RPPC23), such that

    k x EPi = (k/kst)^3 x (OmK/KK)^2 x C x zetaPi

    where OmK = .5 <v^2>, being v_i the field such that Pi_ij = v_l v_m Lambda_ijlm

    Same definition is used in GW_analytical (see calC, which refers to calC computed
    for a smoothed broken power law
    '''

    if tp == 'vort': pref = 28/15
    elif tp == 'comp': pref = 32/15
    elif tp == 'mix': pref = 16/5
    elif tp == 'hel': pref = -4/3

    return np.trapz(zeta**2/k**2, k)*pref

#################### SOUND-SHELL MODEL FOR SOUND WAVES IN PTs ####################

####### Kinetic spectra computed for the sound-shell model from f' and l functions
####### f' and l functions need to be previously computed from the self-similar
####### fluid perturbations induced by expanding bubbles (see hydro_bubbles.py)

# moved from sound_shell.py
def compute_kin_spec_dens(z, vws, fp, l, sp='sum', type_n='exp', cs2=cs2, min_qbeta=-4,
                          max_qbeta=5, Nqbeta=1000, min_TT=-1, max_TT=3, NTT=5000):

    '''
    Function that computes the kinetic power spectral density assuming exponential or simultaneous
    nucleation.

    Arguments:
        z -- array of values of z
        vws -- array of wall velocities
        fp -- function f'(z) computed from the sound-shell model using compute_fs
        l -- function lambda(z) computed from the sound-shell model using compute_fs
        sp -- type of function computed for the kinetic spectrum description
        type_n -- type of nucleation hystory (default is exponential 'exp',
                  another option is simultaneous 'sym')
        dens_spec -- option to return power spectral density (if True, default), or kinetic
                     spectrum (if False)
    '''

    if sp == 'sum': A2 = .25*(cs2*l**2 + fp**2)
    if sp == 'only_f': A2 = .5*fp**2               # assumes equal contributions from fp and cs l
    if sp == 'diff': A2 = .25*(fp**2 - cs2*l**2)
    if sp == 'cross': A2 = -.5*fp*np.sqrt(cs2)*l

    q_beta = np.logspace(min_qbeta, max_qbeta, Nqbeta)
    TT = np.logspace(min_TT, max_TT, NTT)
    q_ij, TT_ij = np.meshgrid(q_beta, TT, indexing='ij')
    Pv = np.zeros((len(vws), len(q_beta)))

    funcT = np.zeros((len(vws), len(q_beta), len(TT)))
    for i in range(0, len(vws)):
        if type_n == 'exp':
            funcT[i, :, :] = np.exp(-TT_ij)*TT_ij**6*np.interp(TT_ij*q_ij, z, A2[i, :])
        if type_n == 'sim':
            funcT[i, :, :] = .5*np.exp(-TT_ij**3/6)*TT_ij**8*np.interp(TT_ij*q_ij, z, A2[i, :])
        Pv[i, :] = np.trapz(funcT[i, :, :], TT, axis=1)

    return q_beta, Pv

# moved from sound_shell.py
# def compute_kin_spec(vws, q_beta, Pv, corr=True, cs2=cs2):

#     EK = []
#     cs = np.sqrt(cs2)
#     if corr: Rstar_beta = (8*np.pi)**(1/3)*np.maximum(vws, cs)
#     else: Rstar_beta = (8*np.pi)**(1/3)*vws
#     kks = np.zeros((len(vws), len(q_beta)))

#     for i in range(0, len(vws)):
#         kks[i, :] = q_beta*Rstar_beta[i]
#         pref = kks[i, :]**2/Rstar_beta[i]**6/(2*np.pi**2)
#         EK_aux = pref*Pv[i, :]
#         EK_aux2 = kin_sp(kks[i, :], EK_aux, cs2=cs2)
#         EK = np.append(EK, EK_aux2)

#     return kks, EK

# moved from sound_shell.py
class kin_sp():

    def __init__(self, k, EK, cs2=cs2):
        self.EK = EK
        self.k = k
        ind_max = np.argmax(EK)
        self.max_EK = EK[ind_max]
        self.kmax = k[ind_max]
        self.zeta = EK/self.max_EK
        self.KK = np.trapz(self.zeta, k)
        self.CC = 16/15*np.trapz(self.zeta**2/k**2, k)
        self.CC2 = 16/15*np.trapz(self.zeta**2/k**4, k)
        self.DD = 16/15*np.trapz(self.zeta*k**2, k)/self.CC
        self.OmK = self.KK*self.max_EK
        self.amplGW = 3*(1 + cs2)*self.OmK**2/self.KK**2*self.CC

# # moved from GW_analytical
# def ET_correlator_compr(k, EK, Np=3000, Nk=60, plot=False,
#                         extend=False, largek=3, smallk=-3):

#     '''
#     Function to compute the spectrum of the anisotropic (TT) stresses ~ u_i u_j (TT) under
#     the assumption that the statistical distribution of the field u_i is Gaussian
#     '''

#     p = np.logspace(np.log10(k[0]), np.log10(k[-1]), Np)
#     kp = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)
#     if extend:
#         Nk = int(Nk/6)
#         kp = np.logspace(smallk, np.log10(k[0]), Nk)
#         kp = np.append(kp, np.logspace(np.log10(k[0]), np.log10(k[-1]),
#                        4*Nk))
#         kp = np.append(kp, np.logspace(np.log10(k[-1]), largek, Nk))

#     Nz = 500
#     z = np.linspace(-1, 1, Nz)
#     kij, pij, zij = np.meshgrid(kp, p, z, indexing='ij')
#     ptilde2 = pij**2 + kij**2 - 2*kij*pij*zij
#     ptilde = np.sqrt(ptilde2)

#     EK_p = np.interp(p, k, EK)
#     if plot:
#         plt.plot(p, EK_p)
#         plt.xscale('log')
#         plt.yscale('log')

#     EK_ptilde = np.interp(ptilde, k, EK)
#     ptilde[np.where(ptilde == 0)] = 1e-50

#     Pi_1 = np.trapz(EK_ptilde/ptilde**4*(1 - zij**2)**2, z, axis=2)
#     kij, EK_pij = np.meshgrid(kp, EK_p, indexing='ij')
#     kij, pij = np.meshgrid(kp, p, indexing='ij')
#     Pi_2 = np.trapz(Pi_1*pij**2*EK_pij, p, axis=1)

#     return kp, np.pi**2*Pi_2

##
## function to compute GW spectrum time growth using the approximation
## introduced in the first sound-shell model analysis of HH19
##
## The resulting GW spectrum is
##
##  Omega_GW (k) = (3pi)/(8cs) x (k/kst)^2 x (K/KK)^2 x TGW x Omm(k)
##

# moved from GW_analytical.py
# def OmGW_ssm_HH19(k, EK, Np=3000, Nk=60, plot=False, cs2=cs2,
#                     extend=False, largek=3, smallk=-3):

#     cs = np.sqrt(cs2)
#     kp = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)
#     if extend:
#         Nk = int(Nk/6)
#         kp = np.logspace(smallk, np.log10(k[0]), Nk)
#         kp = np.append(kp, np.logspace(np.log10(k[0]), np.log10(k[-1]),
#                        4*Nk))
#         kp = np.append(kp, np.logspace(np.log10(k[-1]), largek, Nk))

#     p_inf = kp*(1 - cs)/2/cs
#     p_sup = kp*(1 + cs)/2/cs

#     Omm = np.zeros(len(kp))
#     for i in range(0, len(kp)):

#         p = np.logspace(np.log10(p_inf[i]), np.log10(p_sup[i]), Np)
#         ptilde = kp[i]/cs - p
#         z = -kp[i]*(1 - cs2)/2/p/cs2 + 1/cs

#         EK_p = np.interp(p, k, EK)
#         EK_ptilde = np.interp(ptilde, k, EK)

#         Omm1 = (1 - z**2)**2*p/ptilde**3*EK_p*EK_ptilde
#         Omm[i] = np.trapz(Omm1, p)

#     return kp, Omm

##
## function to compute GW spectrum using the full sound-shell model,
## as described in RPPC23
##
## The resulting GW spectrum is
##
##  Omega_GW (k) = 3/4 x (k/kst)^3 x (K/KK)^2 x TGW x Omm(k)
##

# moved from GW_analytical
def effective_ET_correlator_stat(k, EK, tfin, Np=3000, Nk=60, plot=False, expansion=True, kstar=1.,
                                 extend=False, largek=3, smallk=-3, tini=1, cs2=cs2, terms='all',
                                 inds_m=[], inds_n=[], corr_Delta_0=False):

    """
    Function that computes the normalized GW spectrum zeta_GW(k)
    from the velocity field spectrum for purely compressional anisotropic stresses,
    assuming Gaussianity, and under the assumption of stationary UETC (e.g.,
    sound waves under the sound-shell model).

    29/08/24 (alberto): slightly modified from previous version now to
    compute the combination Delta0 C zeta_GW

    Reference: RPPC23, eq. 90

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
            Delta_mn[l, :, :, :] = compute_Delta_mn(tfin, kij*kstar, pij*kstar, ptilde*kstar,
                                                    cs2=cs2, m=m, n=n, tini=tini, expansion=expansion)
            l += 1

    if l != 0: Delta_mn = Delta_mn/(l + 1)

    Omm = np.zeros((l + 1, len(kp)))
    for i in range(0, l):
        Pi_1 = np.trapz(EK_ptilde/ptilde**4*(1 - zij**2)**2*Delta_mn[i, :, :, :], z, axis=2)
        kij, EK_pij = np.meshgrid(kp, EK_p, indexing='ij')
        kij, pij = np.meshgrid(kp, p, indexing='ij')
        Omm[i, :] = np.trapz(Pi_1*pij**2*EK_pij, p, axis=1)

    # compute value at k -> 0 to normalize
    # CC = 16/15*np.trapz(EK**2/k**2, k)

    # option to divide result by \tilde \Delta 0 to obtain zeta_GW as
    # defined in reference work
    # if corr_Delta_0:
    #    Delta0 = compute_Delta0(tfin, k, kstar, EK, expansion=expansion, cs2=cs2, tini=tini)
    #    CC = CC*Delta0[0]

    return kp, Omm

def compute_Delta_mn(t, k, p, ptilde, cs2=cs2, m=1, n=1, tini=1, expansion=True):

    '''
    Function that computes the integrated Green's functions and the stationary
    UETC 4 Delta_mn used in the sound shell model for the computation of the GW
    spectrum.

    Reference: RPPC23, eqs.56-59

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
