"""
GW_models.py is a Python routine that contains analytical and
semi-analytical models of cosmological GW backgrounds.

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/GW_models.py

Author: Alberto Roper Pol
Created: 29/08/2024
Updated: 04/06/2025 (release cosmoGW 1.0: https://pypi.org/project/cosmoGW)

Other contributors: Antonino Midiri, Simona Procacci, Madeline Salomé

Main references are:

RoperPol:2022iel  - A. Roper Pol, C. Caprini, A. Neronov, D. Semikoz,
"The gravitational wave signal from primordial magnetic fields in the
Pulsar Timing Array frequency band," Phys. Rev. D 105, 123502 (2022),
arXiv:2201.05630

RoperPol:2023bqa  - A. Roper Pol, A. Neronov, C. Caprini, T. Boyer,
D. Semikoz, "LISA and γ-ray telescopes as multi-messenger probes of a
first-order cosmological phase transition," arXiv:2307.10744 (2023)

RoperPol:2023dzg  - A. Roper Pol, S. Procacci, C. Caprini,
"Characterization of the gravitational wave spectrum from sound waves within
the sound shell model," Phys. Rev. D 109, 063531 (2024), arXiv:2308.12943

Hindmarsh:2019phv - M. Hindmarsh, M. Hijazi, "Gravitational waves from first order
cosmological phase transitions in the Sound Shell Model,"
JCAP 12 (2019) 062, arXiv:1909.10040

Caprini:2024gyk   - A. Roper Pol, I. Stomberg, C. Caprini, R. Jinno,
T. Konstandin, H. Rubira, "Gravitational waves from first-order
phase transitions: from weak to strong," JHEP, arxiv:2409.03651

RoperPol:2025b    - A. Roper Pol, A. Midiri, M. Salomé, C. Caprini,
"Modeling the gravitational wave spectrum from slowly decaying sources in the
early Universe: constant-in-time and coherent-decay models," in preparation

RoperPol:2025a    - A. Roper Pol, S. Procacci, A. S. Midiri,
C. Caprini, "Irrotational fluid perturbations from first-order phase
transitions," in preparation
"""

import numpy as np
import matplotlib.pyplot as plt
import cosmoGW.hydro_bubbles as hb
import cosmoGW.GW_analytical as an

# reference values
Oms_ref    = 0.1         # reference source amplitude
                         # (fraction to radiation energy)
lf_ref     = 0.01        # reference length scale of the source
                         # (normalized by the Hubble radius)
beta_ref   = 100         # reference value of the nucleation rate beta/H_ast
N_turb     = 2           # ratio between the effective time duration of
                         # the source and the eddy turnover time,
                         # based on the simulations of RoperPol:2022iel,
                         # used in RoperPol:2023bqa
Nk_ref     = 1000        # reference number of wave number discretization
Nkconv_ref = 1000        # reference number of wave number discretization
                         # for convolution calculations
Np_ref     = 3000        # reference number of wave number discretization
                         # for convolution calculations
NTT_ref    = 5000        # reference number of lifetimes discretization
dt0_ref    = 11          # dt0 = 11 is a numerical parameter of the fit
                         # provided in Caprini:2024gyk
tini_ref   = 1.          # reference initial time of GW production, normalized
                         # with the Hubble rate
tfin_ref   = 1e4         # reference final time of GW production in cit model

'''
    RoperPol:2022iel/RoperPol:2023dzg and RoperPol:2025b/RoperPol:2025b
    consider spectral functions defined such that the average squared
    field corresponds to

    <v^2> ~ 2 E* k* int zeta(K) dK,     in RoperPol:2022iel
    <v^2> ~ 2 E*    int zeta(K) dlnK,   in RoperPol:2025b

    The first convention can be chosen in the following functions if
    dlogK is set to False, while the second one is assumed when dlogK
    is True
'''

################ COMPUTING THE SPECTRUM OF THE STRESSES  ###############
######################## FOR DIFFERENT SOURCES #########################

def Integ(p, tildep, z, k=0., tp='vort', hel=False):

    '''
    Integrand of the integral over p and z (or tilde p) that is
    used to compute the anisotropic stresses in EPi_correlators
    '''

    Integ = 0
    if hel:
        if tp == 'vort': Integ = 1./p/tildep**4*(1 + z**2)*(k - p*z)
        if tp == 'comp': Integ = 2./p/tildep**4*(1 - z**2)*(k - p*z)
    else:
        if tp == 'vort': Integ = .5/p/tildep**3*(1 + z**2)* \
                                     (2 - p**2/tildep**2*(1 - z**2))
        if tp == 'comp': Integ = 2.*p/tildep**5*(1 - z**2)**2
        if tp == 'mix':  Integ = 2.*p/tildep**5*(1 - z**4)
        if tp == 'hel':  Integ = .5/p/tildep**4*z*(k - p*z)

    return Integ

def EPi_correlators(k, a=an.a_ref, b=an.b_ref, alp=an.alp_ref,
                    tp='all', zeta=False, hel=False, norm=True,
                    dlogk=True, model='dbpl', kk=[], EK_p=[]):

    '''
    Routine to compute the spectrum of the projected (anisotropic) or
    unprojected stresses from the two-point correlator of the source
    (e.g. velocity, magnetic or scalar fields) under the assumption that
    the source is Gaussian.

    It computes the vortical, compressional, helical and mixed components
    of the stresses, as well as the compressional and vortical components
    of the helical stresses.

    Main reference is RoperPol:2025a, see also eqs 11 and 20 of
    RoperPol:2022iel for the vortical component and eq. 45 of
    RoperPol:2023dzg for the compressional component

    Arguments:
        k     -- array of wave numbers
        a     -- slope of the spectrum at low wave numbers, k^a
        b     -- slope of the spectrum at high wave numbers, k^(-b)
        alp   -- smoothness of the transition from one power law to the other
        tp    -- type of sourcing field: 'vort', 'comp', 'hel' or 'mix' available
        zeta  -- option to integrate the convolution over z in (-1, 1)
                default option integrates over ptilde in (|p - k|, p + k)
        hel   -- option to compute the helical stresses
        norm  -- option to normalize the spectrum such that its peak is located at
                 kpeak and its maximum value is A
        model -- select the model to be used for the spectrum of the source,
                 current options are 'dbpl' for the smoothed double broken power
                 law model defined in GW_analytical module or 'input' to give
                 a numerical input
        kk, EK_p -- if input is chosen, then numerical array of wave numbers and
                    spectrum of the source need to be provided

    Returns
        pi -- spectrum of the anisotropic stresses of the chosen component
              or of all of them if tp = 'all' is chosen
    '''

    from scipy import integrate

    # define zeta_P times zeta_Ptilde function in funcs based on
    # the inpute model

    if model == 'dbpl':

        A    = (a + b)**(1/alp)
        alp2 = alp*(a + b)
        if norm: c = a; d = b
        else: c = 1.; d = 1.; A = 1.

        # functions following a smoothed broken power law
        def funcs(p, tildep):
            zeta_P      = A*p**a/(d + c*p**alp2)**(1/alp)
            zeta_Ptilde = A*tildep**a/(d + c*tildep**alp2)**(1/alp)
            return zeta_P*zeta_Ptilde

    elif model == 'input':

        if len(kk) == 0 or len(EK_p) == 0:
            print('For using input model provide kk and EK')
            return 0.

        # functions interpolate the input numerical data
        def funcs(p, tildep):
            zeta_P      = np.interp(p, kk, EK_p)
            zeta_Ptilde = np.interp(p, kk, EK_p)
            return zeta_P*zeta_Ptilde

    else:
        print('A model needs to be selected: dpbl or input')
        return 0.

    ## compute all components (vort, comp, mix, hel)

    if tp == 'all':
        if hel: tps = ['vort', 'comp']
        else:   tps = ['vort', 'comp', 'mix', 'hel']
        pis = np.zeros((len(tps), len(k)))
        for j in range(0, len(tps)):
            # integrate over p and z
            if zeta:

                def f(p, z, kp):
                    tildep = np.sqrt(p**2 + kp**2 - 2*p*kp*z)
                    II     = funcs(p, tildep)*Integ(p, tildep, z, k=kp,
                                                    tp=tps[j], hel=hel)
                    if not dlogk: II *= p*tildep
                    return II

                for i in range(0, len(k)):
                    kp = k[i]
                    pis[j, i], _ = \
                        integrate.nquad(f, [[0, np.inf], [-1., 1.]], args=(kp,))

            # integrate over p and ptilde
            else:

                for i in range(0, len(k)):
                    kp = k[i]

                    def f(p, tildep):
                        z  = (p**2 + kp**2 - tildep**2)/2/p/kp
                        II = funcs(p, tildep)*Integ(p, tildep, z, k=kp,
                                   tp=tps[j], hel=hel)*tildep/p/kp
                        if not dlogk: II *= p*tildep
                        return II

                    def bounds_p():       return [0, np.inf]
                    def bounds_tildep(p): return [abs(kp - p), kp + p]

                    pis[j, i], _ = integrate.nquad(f, [bounds_tildep, bounds_p])

        return pis

    ## compute the chosen component in tp
    else:

        pi = np.zeros(len(k))

        # integrate over p and z
        if zeta:

            def f(p, z, kp):
                tildep = np.sqrt(p**2 + kp**2 - 2*p*kp*z)
                II     = funcs(p, tildep)*Integ(p, tildep, z, k=kp,
                               tp=tp, hel=hel)
                if not dlogk: II *= p*tildep
                return II

            for i in range(0, len(k)):
                kp = k[i]
                pi[i], _ = \
                    integrate.nquad(f, [[0, np.inf], [-1., 1.]], args=(kp,))

        # integrate over p and ptilde
        else:
            for i in range(0, len(k)):
                kp = k[i]
                def f(p, tildep):
                    z  = (p**2 + kp**2 - tildep**2)/2/p/kp
                    II = funcs(p, tildep)*Integ(p, tildep, z, k=kp,
                               tp=tp, hel=hel)*tildep/p/kp
                    if not dlogk: II *= p*tildep
                    return II

                def bounds_p(): return [0, np.inf]
                def bounds_tildep(p): return [abs(kp - p), kp + p]

                pi[i], _ = integrate.nquad(f, [bounds_tildep, bounds_p])

        return pi

###########  FUNCTIONS FOR THE CONSTANT-IN-TIME MODEL ###########

def Delta_cit(t, k, tini=tini_ref, tfin=tfin_ref, expansion=True):

    """
    Function that computes the value of the function Delta(k, t) used in the
    analytical calculations of the GW spectrum when assuming
    a constant sourcing stress spectrum, i.e., Pi(k, t1, t2) = Pi(k)

    Arguments:
        k         -- array of wave numbers
        t         -- array of times
        tini      -- initial time of the turbulence sourcing (default 1)
        tfin      -- final time of the turbulence sourcing (default 1e4)
        expansion -- option to include the expansion of the Universe
                     during radiation domination (default is True)

    Returns:
        Delta     -- function Delta(k, t)
    """

    import scipy.special as spe

    mult_t = isinstance(t, (list, tuple, np.ndarray))
    mult_k = isinstance(k,  (list, tuple, np.ndarray))

    tij, kij = np.meshgrid(t, k, indexing='ij')
    cost     = np.cos(kij*tij)
    sint     = np.sin(kij*tij)
    tij[np.where(tij>tfin)] = tfin
    if expansion:
        si_t, ci_t       = spe.sici(kij*tij)
        si_tini, ci_tini = spe.sici(kij*tini)
    else:
        ci_t    =  np.sin(kij*tij)/kij
        ci_tini =  np.sin(kij*tini)/kij
        si_t    = -np.cos(kij*tij)/kij
        si_tini = -np.cos(kij*tini)/kij

    aux1 = cost*(ci_t - ci_tini)
    aux2 = sint*(si_t - si_tini)
    D    = aux1 + aux2

    if   not mult_t and not mult_k: D = D[0, 0]
    elif not mult_t: D = D[0, :]
    elif not mult_k: D = D[:, 0]

    return D

def Delta2_cit_aver(k, tini=tini_ref, tfin=tfin_ref, expansion=True):

    """
    Function that computes the value of the function D(k, t) used in the
    analytical calculations of the GW energy density spectrum when assuming
    a constant sourcing stress spectrum, i.e., Pi(k, t1, t2) = Pi(k)

    Arguments:
        k         -- array of wave numbers
        tini      -- initial time of the turbulence sourcing (default 1)
        tfin      -- final time of the turbulence sourcing
        expansion -- option to include the expansion of the Universe
                     during radiation domination (default is True)

    Returns:
        D2 -- function D^2(k, t) averaged over t0
    """

    import scipy.special as spe

    if expansion:
        si_t, ci_t       = spe.sici(k*tfin)
        si_tini, ci_tini = spe.sici(k*tini)
    else:
        ci_t    =  np.sin(k*tfin)/k
        ci_tini =  np.sin(k*tini)/k
        si_t    = -np.cos(k*tfin)/k
        si_tini = -np.cos(k*tini)/k
    aux1 = (ci_t - ci_tini)
    aux2 = (si_t - si_tini)
    D2   = .5*(aux1**2 + aux2**2)

    return D2

def TGW_func(s, Oms=Oms_ref, lf=lf_ref, N=N_turb, cs2=hb.cs2_ref,
             expansion=True, tdecay='eddy', tp='magnetic'):

    """
    Function that computes the logarithmic term obtained as
    the envelope of the GW template in the constant-in-time (cit)
    assumption for the unequal time correlator of the turbulent
    stresses.

    It is obtained from integrating the Green's function in the
    constant-in-time model, validated in RoperPol:2022iel
    and described for other sources in RoperPol:2025b

    The function TGW_func determines two regimes, frequencies below
    and above an inverse duration,

        fbr = 1/dtfin,

    as used in RoperPol:2023bqa, EPTA:2023xxk, Caprini:2024hue.
    However, the original reference RoperPol:2022iel used

        kbr = fbr/(2pi) = 1/dtfinm,

    such that dtfinm = dtfin/(2pi), and relates dtfin to
    the decay time (considered the eddy turnover time) as

        dtfinm = N tdecay = N/(vA k*) = N R*/(2pi vA)

    Hence, we will consider

        fbr   = 1/dtfin = 1/(2pi dtfinm) = vA/(N R*)
        dtfin = N R*/vA

    Main references are:

    - RoperPol:2022iel, equation 24
    - RoperPol:2023bqa, equation 15
    - EPTA:2023xxk,     equation 21
    - Caprini:2024hue,  equation 2.17
    - RoperPol:2025b

    Arguments:
        s      -- array of frequencies, normalized by the
                 characteristic scale, s = f R*
        Oms    -- energy density of the source (i.e., 1/2 vrms^2)
        lf     -- characteristic scale of the turbulence as a
                  fraction of the Hubble radius, R* H*
        N      -- relation between the decay time and the effective
                  source duration
        cs2    -- square of the speed of sound (default is 1/3)
        expansion -- option to include the expansion of the Universe
                     during radiation domination (default is True)
        tdecay -- determines the finite duration in the cit model
                  (default is to use eddy turnover time)
        tp     -- type of source (default is 'magnetic', other options are
                  'kinetic' or 'max') used to compute the characteristic
                  velocity in the eddy turnover time

    Returns:
        TGW   -- logarithmic function that characterizes the spectral shape
    """

    mult_Oms = isinstance(Oms, (list, tuple, np.ndarray))
    mult_lf  = isinstance(lf,  (list, tuple, np.ndarray))

    # if not mult_Oms: Oms = [Oms]
    # if not mult_lf:  lf  = [lf]

    s, Oms, lf = np.meshgrid(s, Oms, lf, indexing='ij')

    # characteristic velocity (for example, Alfven velocity or vrms)
    # see eq. 12 of RoperPol:2023bqa
    if   tp == 'kinetic':  vA = np.sqrt(Oms)
    elif tp == 'magnetic': vA = np.sqrt(2*Oms/(1 + cs2))
    else:                  vA = np.sqrt(max(1, 2/(1 + cs2))*Oms)
    # decay time in units of R*
    if   tdecay == 'eddy': tdec = 1./vA

    # effective duration of the source dtfin/R* is N units of
    # the decay time (see comment above for different choices
    # in the literature)
    dtfin = N*tdec

    TGW = np.zeros_like(dtfin)
    if expansion:
        inds = np.where(s <  1/dtfin)
        TGW[inds] = (np.log(1 + dtfin*lf/2/np.pi)**2)[inds]
        inds = np.where(s >= 1/dtfin)
        TGW[inds] = (np.log(1 + lf/2/np.pi/s)**2)[inds]
    else:
        inds = np.where(s <  1/dtfin)
        TGW[inds] = (dtfin*lf/2/np.pi)[inds]**2
        inds = np.where(s >= 1/dtfin)
        TGW[inds] = (lf/2/np.pi/s)[inds]**2


    if not mult_Oms and not mult_lf: TGW = TGW[:, 0, 0]
    elif not mult_Oms:               TGW = TGW[:, 0, :]
    elif not mult_lf:                TGW = TGW[:, :, 0]

    return TGW

################ SOUND-SHELL MODEL FOR SOUND WAVES IN PTs ################

'''
Kinetic spectra computed for the sound-shell model from f' and l functions.
f' and l functions need to be previously computed from the self-similar
fluid perturbations induced by expanding bubbles using hydro_bubbles module
'''

def compute_kin_spec_ssm(z, vws, fp, l=[], sp='sum', type_n='exp', cs2=hb.cs2_ref,
                         min_qbeta=-4, max_qbeta=5, Nqbeta=Nk_ref,
                         min_TT=-1, max_TT=3, NTT=NTT_ref, corr=False,
                         dens=True, normbeta=True):

    '''
    Function that computes the kinetic power spectral density assuming
    exponential or simultaneous nucleation.

    Reference: Equations (32)--(37) of RoperPol:2023dzg
    Main reference is RoperPol:2025a

    Arguments:
        z      -- array of values of z
        vws    -- array of wall velocities
        fp     -- function f'(z) computed from the hydro_bubble module using fp_z
        l      -- function lambda(z) computed from the hydro_bubble module using fp_z
                  (using lz = True)
        sp     -- type of function computed for the kinetic spectrum description
        type_n -- type of nucleation hystory (default is exponential 'exp',
                  another option is simultaneous 'sym')
        cs2    -- square of the speed of sound (default 1/3)
        corr   -- option to 'correct' Rstar beta with max(vw, cs) (default is False)
        dens   -- option to return power spectral density (if True, default), or kinetic
                  spectrum (if False)
        normbeta -- normalization of k with beta (default is True).
                    If normbeta is False, k is normalized with Rstar

    Returns:
        qbeta -- wave number, normalized with 1/beta or Rstar
        P     -- power spectral density (power spectrum if dens is False)
    '''

    if sp == 'sum':    A2 = .25*(cs2*l**2 + fp**2)
    if sp == 'only_f': A2 = .5*fp**2
    if sp == 'only_l': A2 = .5*cs2*l**2
    if sp == 'diff':   A2 = .25*(fp**2 - cs2*l**2)
    if sp == 'cross':  A2 = -.5*fp*np.sqrt(cs2)*l

    qbeta = np.logspace(min_qbeta, max_qbeta, Nqbeta)
    TT    = np.logspace(min_TT, max_TT, NTT)
    Pv    = np.zeros((len(vws), len(qbeta)))

    q_ij, TT_ij = np.meshgrid(qbeta, TT, indexing='ij')
    if type_n == 'exp': nu_T = np.exp(-TT_ij)
    if type_n == 'sim': nu_T = .5*np.exp(-TT_ij**3/6)*TT_ij**2

    funcT = np.zeros((len(vws), len(qbeta), len(TT)))

    for i in range(0, len(vws)):
        funcT[i, :, :]   = nu_T*TT_ij**6*np.interp(TT_ij*q_ij, z, A2[i, :])
        try:    Pv[i, :] = np.trapezoid(funcT[i, :, :], TT, axis=1)
        except: Pv[i, :] = np.trapz(funcT[i, :, :], TT, axis=1)

    if dens == False:
        Rstar_beta   = hb.Rstar_beta(vw=vws[i], cs2=cs2, corr=corr)
        for i in range(0, len(vws)):
            pref     = qbeta[i, :]**2/Rstar_beta[i]**4/(2*np.pi**2)
            Pv[i, :] *= pref

    if normbeta == False:
        if not dens: Rstar_beta = hb.Rstar_beta(vw=vws[i], cs2=cs2, corr=corr)
        for i in range(0, len(vws)):
            qbeta[i, :] *= Rstar_beta[i]

    return qbeta, Pv

def OmGW_ssm_HH19(k, EK, Np=Np_ref, Nk=Nkconv_ref, plot=False,
                  cs2=hb.cs2_ref):

    '''
    Function to compute GW spectrum using the approximation
    introduced in the first sound-shell model analysis of Hindmarsh:2019phv
    under the delta assumption for the 'stationary' term
    (see appendix B of RoperPol:2023dzg for details).

    The resulting GW spectrum is

     Omega_GW (k) = (3pi)/(8cs) x Gamma2 x (k/kst)^2 x (Oms/cal A)^2 x TGW x Omm(k)

     where Oms = <v^2> = vrms^2 and cal A = int zeta(K) d ln k
     (using the normalization of RoperPol:2025a for zeta)

    Reference: Appendix B of RoperPol:2023dzg; see eq.(B3)
    '''

    cs    = np.sqrt(cs2)
    kp    = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)

    p_inf = kp*(1 - cs)/2/cs
    p_sup = kp*(1 + cs)/2/cs

    Omm = np.zeros(len(kp))
    for i in range(0, len(kp)):

        p      = np.logspace(np.log10(p_inf[i]), np.log10(p_sup[i]), Np)
        ptilde = kp[i]/cs - p
        z      = -kp[i]*(1 - cs2)/2/p/cs2 + 1/cs

        EK_p      = np.interp(p, k, EK)
        EK_ptilde = np.interp(ptilde, k, EK)

        Omm1   = (1 - z**2)**2*p/ptilde**3*EK_p*EK_ptilde
        try:    Omm[i] = np.trapezoid(Omm1, p)
        except: Omm[i] = np.trapz(Omm1, p)

    return kp, Omm

def effective_ET_correlator_stat(k, EK, tfin, Np=Np_ref, Nk=Nkconv_ref,
                                 plot=False, expansion=True, kstar=1.,
                                 extend=False, largek=3, smallk=-3, tini=tini_ref,
                                 cs2=hb.cs2_ref, terms='all',
                                 inds_m=[], inds_n=[], corr_Delta_0=False):

    """
    Function that computes the normalized GW spectrum zeta_GW(k)
    from the velocity field spectrum for purely compressional anisotropic stresses,
    assuming Gaussianity, and under the assumption of stationary UETC (e.g.,
    sound waves under the sound-shell model).

    Reference: RoperPol:2023dzg, eq. 93

    Arguments:
        k      -- array of wave numbers
        EK     -- array of values of the kinetic spectrum
        Np     -- number of discretizations in the wave number p to be numerically
              integrated
        Nk     -- number of discretizations of k to be used for the computation of
              the final spectrum
        plot   -- option to plot the interpolated magnetic spectrum for debugging
                purposes (default False)
        extend -- option to extend the array of wave numbers of the resulting
                  Pi spectrum compared to that of the given magnetic spectrum
                  (default False)

    Returns:
        Omm -- GW spectrum normalized (zeta_GW)
        kp  -- final array of wave numbers
    """

    p  = np.logspace(np.log10(k[0]), np.log10(k[-1]), Np)
    kp = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)
    if extend:
        Nk = int(Nk/6)
        kp = np.logspace(smallk, np.log10(k[0]), Nk)
        kp = np.append(kp, np.logspace(np.log10(k[0]), np.log10(k[-1]),
                       4*Nk))
        kp = np.append(kp, np.logspace(np.log10(k[-1]), largek, Nk))

    Nz = 500
    z  = np.linspace(-1, 1, Nz)
    kij, pij, zij = np.meshgrid(kp, p, z, indexing='ij')
    ptilde2 = pij**2 + kij**2 - 2*kij*pij*zij
    ptilde  = np.sqrt(ptilde2)

    EK_p = np.interp(p, k, EK)
    if plot:
        plt.plot(p, EK_p)
        plt.xscale('log')
        plt.yscale('log')

    EK_ptilde = np.interp(ptilde, k, EK)
    ptilde[np.where(ptilde == 0)] = 1e-50

    Delta_mn = kij**0 - 1

    if terms == 'all':
        inds_m   = [-1, 1]
        inds_n   = [-1, 1]
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
        try:    Pi_1 = np.trapezoid(EK_ptilde/ptilde**4*(1 - zij**2)**2*Delta_mn[i, :, :, :],
                                    z, axis=2)
        except: Pi_1 = np.trapz(EK_ptilde/ptilde**4*(1 - zij**2)**2*Delta_mn[i, :, :, :],
                                z, axis=2)
        kij, EK_pij = np.meshgrid(kp, EK_p, indexing='ij')
        kij, pij    = np.meshgrid(kp, p, indexing='ij')
        try:    Omm[i, :]   = np.trapezoid(Pi_1*pij**2*EK_pij, p, axis=1)
        except: Omm[i, :]   = np.trapz(Pi_1*pij**2*EK_pij, p, axis=1)

    return kp, Omm

def compute_Delta_mn(t, k, p, ptilde, cs2=hb.cs2_ref, m=1, n=1, tini=1.,
                     expansion=True):

    '''
    Function that computes the integrated Green's functions and the stationary
    UETC 4 Delta_mn used in the sound shell model for the computation of the GW
    spectrum.

    Reference: RoperPol:2023dzg, eqs.56-59

    Arguments:
        t      -- time
        k      -- wave number k
        p      -- wave number p to be integrated over
        ptilde -- second wave number tilde p to be integrated over
        cs2    -- square of the speed of sound (default is 1/3)
        m, n   -- indices of the Deltamn function
        tini   -- initial time of GW production in units of 1/Hubble
        expansion -- option to include the effect of the expansion of the Universe
                     during radiation domination (default True)
    '''

    cs = np.sqrt(cs2)
    pp = n*k + cs*(m*ptilde + p)
    pp[np.where(pp == 0)] = 1e-50

    if expansion:

        import scipy.special as spe
        si_t, ci_t       = spe.sici(pp*t)
        si_tini, ci_tini = spe.sici(pp*tini)

        # compute Delta Ci^2 and Delta Si^2
        DCi = ci_t - ci_tini
        DSi = si_t - si_tini

        Delta_mn = DCi**2 + DSi**2

    else:

        Delta_mn = 2*(1 - np.cos(pp*(t - tini)))/pp**2

    return Delta_mn

########### FUNCTIONS FOR THE LOCALLY STATIONARY UETC ###########

def K2int(dtfin, K0=1., dt0=dt0_ref, b=0., expansion=False, beta=beta_ref):

    '''
    Function that returns the integrated kinetic energy density
    squared for a power law

      K(dt) = K0 (dt/dt0)^(-b),

    where dt and dt0 represent time intervals, K0 is the amplitude
    at the end of the phase transition, and dt0 = 11 is a numerical
    parameter of the fit.

    Based on the results of Higgsless simulations of Caprini:2024gyk,
    see equation 2.27.

    It determines the amplitude of GWs in the locally stationary
    UETC describing the decay of compressional sources, see equation
    2.33

    Arguments:
        dtfin -- duration of the sourcing (in units of 1/beta or 1/H* for
                 expansion = False or True)
        K0    -- amplitude of K at the end of the phase transition
        dt0   -- numerical parameter used for the fits (= 11)
        b     -- power law exponent of the decay of K in time
        expansion -- option to consider flat Minkowski space-time
                     (if expansion is False, default) or a radiation-
                     dominated expanding Universe
        beta  -- nucleation rate normalized to the Hubble time,
                 beta/H*, relevant if expansion is chosen

    Returns:
        K2int -- integrated squared kinetic energy density
    '''

    K2int = K0**2*dt0

    # computation in Minkowski space-time
    if not expansion:
        if b == 0.5:
            K2int *= np.log(1 + dtfin/dt0)
        else:
            K2int *= ((1 + dtfin/dt0)**(1. - 2*b) - 1)/(1. - 2.*b)

    # computation in expanding Universe during radiation-domination
    else:

        import scipy.special as spe

        if beta < dt0:
            print('The value of beta in K2int cannot be smaller than',
                  'dt0 = %.1f for the chosen model'%dt0)
            print('Choose a larger value of beta to include expansion')
            return 0

        dt0   =  dt0/beta
        K2int *= 1./beta/(dt0 - 1.)**2

        if b == 0.5:
            K2int *= (dt0 - 1.)/(1. + dtfin) + np.log((1 + dtfin/dt0)/(1 + dtfin))

        else:
            K2int *= 1./(1. - 2*b)
            A      = spe.hyp2f1(2, 1 - 2*b, 2. - 2*b, (dt0 + dtfin)/(dt0 - 1))
            B      = spe.hyp2f1(2, 1 - 2*b, 2. - 2*b, dt0/(dt0 - 1.))
            K2int *= (1 + dtfin/dt0)**(1. - 2*b)*A - B

    return K2int
