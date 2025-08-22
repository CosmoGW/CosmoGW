r"""
GW_models.py is a Python routine that contains analytical and
semi-analytical models of cosmological GW backgrounds.

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/GW_models.py

.. note::
   For full documentation, visit `Read the Docs
   <https://cosmogw-manual.readthedocs.io/en/latest/GW_models.html>`_.

To use it, first install `cosmoGW <https://pypi.org/project/cosmoGW>`_::

    pip install cosmoGW

Author
------
- **Alberto Roper Pol**
  (`alberto.roperpol@unige.ch <mailto:alberto.roperpol@unige.ch>`_)

Dates
-----
- Created: **29/08/2024**
- Updated: **21/08/2025**
  (release **cosmoGW 1.0**: https://pypi.org/project/cosmoGW)

Contributors
------------
- **Antonino Midiri**, **Simona Procacci**, **Madeline Salomé**,
  **Isak Stomberg**

References
----------
- [**RoperPol:2022iel**]: A. Roper Pol, C. Caprini, A. Neronov, D. Semikoz,
  "*The gravitational wave signal from primordial magnetic fields in the
  Pulsar Timing Array frequency band,*" Phys. Rev. D **105**, 123502 (2022),
  `arXiv:2201.05630 <https://arxiv.org/abs/2201.05630>`_.

- [**RoperPol:2023bqa**]: A. Roper Pol, A. Neronov, C. Caprini, T. Boyer,
  D. Semikoz, "*LISA and γ-ray telescopes as multi-messenger probes of a
  first-order cosmological phase transition,*" \
  `arXiv:2307.10744 <https://arxiv.org/abs/2307.10744>`_ (2023).

- [**RoperPol:2023dzg**]: A. Roper Pol, S. Procacci, C. Caprini,
  "*Characterization of the gravitational wave spectrum from sound waves within
  the sound shell model,*" Phys. Rev. D **109**, 063531 (2024),
  `arXiv:2308.12943 <https://arxiv.org/abs/2308.12943>`_.

- [**Hindmarsh:2019phv**]: M. Hindmarsh, M. Hijazi, "*Gravitational waves
  from first order cosmological phase transitions in the Sound Shell Model,*"
  JCAP **12** (2019), 062,
  `arXiv:1909.10040 <https://arxiv.org/abs/1909.10040>`_.

- [**Caprini:2024gyk**]: A. Roper Pol, I. Stomberg, C. Caprini, R. Jinno,
  T. Konstandin, H. Rubira, "*Gravitational waves from first-order
  phase transitions: from weak to strong,*" JHEP **07** (2025), 217,
  `arxiv:2409.03651 <https://arxiv.org/abs/2409.03651>`_.

- [**RoperPol:2025b**]: A. Roper Pol, A. Midiri, M. Salomé, C. Caprini,
  "*Modeling the gravitational wave spectrum from slowly decaying sources in the
  early Universe: constant-in-time and coherent-decay models,*" in preparation

- [**RoperPol:2025a**]: A. Roper Pol, S. Procacci, A. S. Midiri,
  C. Caprini, "*Irrotational fluid perturbations from first-order phase
  transitions,*" in preparation

Comments
--------
RoperPol:2022iel/RoperPol:2023dzg and RoperPol:2025b/RoperPol:2025b
consider spectral functions defined such that the average squared
field corresponds to

.. math::
    \langle v^2 \rangle \propto A k_\ast \int \zeta(K) dK,  \quad
    {\rm \ in \ RoperPol:2022iel},

    \langle v^2 \rangle \propto A \int \zeta(K) d \ln K,  \quad
    {\rm \ in \ RoperPol:2025b}.

The first convention can be chosen in the following functions if
dlogK is set to False, while the second one is assumed when dlogK
is True
"""

import numpy as np
from scipy import special, integrate
import matplotlib.pyplot as plt
from cosmoGW import hydro_bubbles
from cosmoGW.utils import (
    a_ref, b_ref, alp_ref, tini_ref, tfin_ref, cs2_ref, N_turb, lf_ref, Oms_ref,
    beta_ref, dt0_ref, Np_ref, Nk_ref, Nkconv_ref, NTT_ref,
    safe_trapezoid, reshape_output
)


def _Integ(p, tildep, z, k=0.0, tp="vort", hel=False):

    r"""
    Integrand for the anisotropic stress integral in
    :func:`EPi_correlators`.

    Parameters
    ----------
    p : float
        Wave number array p.
    tildep : float
        Second wave number :math:`\tilde p = |p - k|`.
    z : float
        Cosine of angle between p and k.
    k : float, optional
        Output wave number (default 0).
    tp : str, optional
        Type of sourcing field ('vort', 'comp', 'mix', 'hel').
    hel : bool, optional
        Compute helical stresses.

    Returns
    -------
    Integ : float
        Value of the integrand.

    References
    ----------
    See EPi_correlators.
    """

    Integ = 0
    if hel:
        if tp == "vort":
            Integ = 1.0 / p / tildep**4 * (1 + z**2) * (k - p * z)
        if tp == "comp":
            Integ = 2.0 / p / tildep**4 * (1 - z**2) * (k - p * z)
    else:
        if tp == "vort":
            Integ = (
                0.5 / p / tildep**3 * (1 + z**2)
                * (2 - p**2 / tildep**2 * (1 - z**2))
            )
        if tp == "comp":
            Integ = 2.0 * p / tildep**5 * (1 - z**2) ** 2
        if tp == "mix":
            Integ = 2.0 * p / tildep**5 * (1 - z**4)
        if tp == "hel":
            Integ = 0.5 / p / tildep**4 * z * (k - p * z)

    return Integ


def _get_funcs(model, a, b, alp, norm, kk, EK_p):
    '''
    Define the zeta_P times zeta_Ptilde function in funcs based on
    the input model, used in :func:`EPi_correlators`.
    '''
    # functions following a smoothed broken power law
    if model == "dbpl":
        A = (a + b) ** (1 / alp) if norm else 1.0
        alp2 = alp * (a + b)
        c, d = (a, b) if norm else (1.0, 1.0)

        def funcs(p, tildep):
            zeta_P = A * p**a / (d + c * p**alp2) ** (1 / alp)
            zeta_Ptilde = A * tildep**a / (d + c * tildep**alp2) ** (1 / alp)
            return zeta_P * zeta_Ptilde
        return funcs
    # functions interpolate the input numerical data
    elif model == "input":
        if len(kk) == 0 or len(EK_p) == 0:
            raise ValueError('For input model, provide kk and EK_p')

        def funcs(p, tildep):
            zeta_P = np.interp(p, kk, EK_p)
            zeta_Ptilde = np.interp(tildep, kk, EK_p)
            return zeta_P * zeta_Ptilde
        return funcs
    else:
        raise ValueError("Model must be 'dbpl' or 'input'")


def _integrate_over_zeta(k, tps, funcs, dlogk, hel):
    '''
    Integrate zeta_P times zeta_Ptilde over z, used in
    :func:`EPi_correlators`.
    '''
    pis = np.zeros((len(tps), len(k)))
    for j, tp in enumerate(tps):
        def f(p, z, kp):
            tildep = np.sqrt(p**2 + kp**2 - 2 * p * kp * z)
            II = funcs(p, tildep) * _Integ(p, tildep, z, k=kp, tp=tp, hel=hel)
            if not dlogk:
                II *= p * tildep
            return II
        for i, kp in enumerate(k):
            pis[j, i], _ = integrate.nquad(
                f, [[0, np.inf], [-1.0, 1.0]], args=(kp,)
            )
    return pis


def _integrate_over_ptilde(k, tps, funcs, dlogk, hel):
    '''
    Integrate zeta_P times zeta_Ptilde over ptilde, used in
    :func:`EPi_correlators`.
    '''
    pis = np.zeros((len(tps), len(k)))
    for j, tp in enumerate(tps):
        for i, kp in enumerate(k):
            def f(p, tildep):
                z = (p**2 + kp**2 - tildep**2) / (2 * p * kp)
                II = (
                    funcs(p, tildep)
                    * _Integ(p, tildep, z, k=kp, tp=tp, hel=hel)
                    * tildep / p / kp
                )
                if not dlogk:
                    II *= p * tildep
                return II

            def bounds_p():
                return [0, np.inf]

            def bounds_tildep(p):
                return [abs(kp - p), kp + p]
            pis[j, i], _ = integrate.nquad(f, [bounds_tildep, bounds_p])
    return pis


def EPi_correlators(k, a=a_ref, b=b_ref, alp=alp_ref, tp="all", zeta=False,
                    hel=False, norm=True, dlogk=True, model="dbpl", kk=None,
                    EK_p=None):
    r"""
    Compute spectrum of projected or unprojected stresses from the
    two-point correlator of the source (Gaussian assumption).

    Computes vortical, compressional, helical, and mixed components.

    References
    ----------
    RoperPol:2025a, RoperPol:2022iel (eqs 11 and 20) for vortical fields,
    and RoperPol:2023dzg for compressional ones.

    Parameters
    ----------
    k : array_like
        Wave numbers.
    a : float
        Low-k slope, :math:`k^a` (default 5)
    b : float
        High-k slope, :math:`k^{-b}` (default 2/3).
    alp : float
        Smoothness of transition, :math:`\alpha` (default 2).
    tp : str
        Type of sourcing field ('vort', 'comp', 'hel' or 'mix' available)
    zeta : bool
        Integrate convolution over :math:`z \in (-1, 1)`.
    hel : bool
        Compute helical stresses.
    norm : bool
        Normalize spectrum to peak at kpeak with amplitude A.
    dlogk : bool
        Use spectrum per unit log(k).
    model : str
        Model for source spectrum ('dbpl' or 'input').
    kk : array_like, optional
        Wave numbers for input model.
    EK_p : array_like, optional
        Spectrum for input model.

    Returns
    -------
    pi : ndarray
        Spectrum of anisotropic stresses.
    """

    if kk is None:
        kk = []
    if EK_p is None:
        EK_p = []

    funcs = _get_funcs(model, a, b, alp, norm, kk, EK_p)

    if tp == "all":
        tps = ['vort', 'comp'] if hel else ['vort', 'comp', 'mix', 'hel']
        # integrate over p and z
        if zeta:
            return _integrate_over_zeta(k, tps, funcs, dlogk, hel)
        else:
            return _integrate_over_ptilde(k, tps, funcs, dlogk, hel)
    else:
        if tp in ['vort', 'comp', 'mix', 'hel']:
            tps = np.array([tp])
        else:
            raise ValueError("tp must be 'vort', 'comp', 'mix', 'hel' or 'all'")
        if zeta:
            return _integrate_over_zeta(k, tps, funcs, dlogk, hel)[0, :]
        else:
            return _integrate_over_ptilde(k, tps, funcs, dlogk, hel)[0, :]


#  FUNCTIONS FOR THE CONSTANT-IN-TIME MODEL
def Delta_cit(t, k, tini=tini_ref, tfin=tfin_ref, expansion=True):
    """
    Compute Delta(k, t) used in the analytical calculations of
    the GW spectrum when assuming a constant sourcing stress spectrum,
    i.e., Pi(k, t1, t2) = Pi(k)

    Parameters
    ----------
    t : array_like
        Times.
    k : array_like
        Wave numbers.
    tini : float, optional
        Initial time of  sourcing (default 1).
    tfin : float, optional
        Final time of sourcing (default 1e4).
    expansion : bool, optional
        Include expansion of Universe (default is True).

    Returns
    -------
    D : ndarray
        Delta function values.

    References
    ----------
    RoperPol:2022iel, RoperPol:2023dzg
    """

    mult_t = isinstance(t, (list, tuple, np.ndarray))
    mult_k = isinstance(k, (list, tuple, np.ndarray))
    tij, kij = np.meshgrid(t, k, indexing="ij")
    cost = np.cos(kij * tij)
    sint = np.sin(kij * tij)
    tij[np.where(tij > tfin)] = tfin
    if expansion:
        si_t, ci_t = special.sici(kij * tij)
        si_tini, ci_tini = special.sici(kij * tini)
    else:
        ci_t = np.sin(kij * tij) / kij
        ci_tini = np.sin(kij * tini) / kij
        si_t = -np.cos(kij * tij) / kij
        si_tini = -np.cos(kij * tini) / kij

    aux1 = cost * (ci_t - ci_tini)
    aux2 = sint * (si_t - si_tini)
    D = aux1 + aux2
    D = reshape_output(D, mult_t, mult_k)

    return D


def Delta2_cit_aver(k, tini=tini_ref, tfin=tfin_ref, expansion=True):

    """
    Compute D^2(k, t) averaged over t0 for constant-in-time GW source.

    Parameters
    ----------
    k : array_like
        Wave numbers.
    tini : float, optional
        Initial time (default 1).
    tfin : float, optional
        Final time.
    expansion : bool, optional
        Include expansion of Universe (default is True).

    Returns
    -------
    D2 : ndarray
        D^2(k, t) averaged over t0.

    References
    ----------
    RoperPol:2022iel, RoperPol:2023dzg
    """

    if expansion:
        si_t, ci_t = special.sici(k * tfin)
        si_tini, ci_tini = special.sici(k * tini)
    else:
        ci_t = np.sin(k * tfin) / k
        ci_tini = np.sin(k * tini) / k
        si_t = -np.cos(k * tfin) / k
        si_tini = -np.cos(k * tini) / k
    aux1 = ci_t - ci_tini
    aux2 = si_t - si_tini
    D2 = 0.5 * (aux1**2 + aux2**2)

    return D2


def TGW_func(s, Oms=Oms_ref, lf=lf_ref, N=N_turb,
             cs2=cs2_ref, expansion=True, tdecay="eddy",
             tp="magnetic"):

    r"""
    Compute the logarithmic envelope of the GW template in the
    constant-in-time model for the unequal time correlator,
    :math:`{\cal T}_{\rm GW}`.

    It is obtained from integrating the Green's function in the
    constant-in-time model, validated in RoperPol:2022iel
    and described for other sources in RoperPol:2025b

    The function TGW_func determines two regimes, frequencies below
    and above an inverse duration,

    .. math ::
        f_{\rm br} = 1/\delta t_{\rm fin},

    as used in RoperPol:2023bqa, EPTA:2023xxk, Caprini:2024hue.
    However, the original reference RoperPol:2022iel used

    .. math ::
        k_{\rm br} = f_{\rm br}/(2\pi) = 1/\delta t_{\rm fin},

    such that :math:`\delta t_{\rm fin} = \delta t_{\rm fin}/(2\pi)`,
    and relates :math:`\delta t_{\rm fin}` to
    the decay time (considered the eddy turnover time) as

    .. math ::
        \delta t_{\rm finm} = N \delta t_{\rm decay} = N/(v_A k_*) =
        N R_*/(2\pi v_A)

    Hence, we will consider

    .. math ::
        f_{\rm br} = 1/\delta t_{\rm fin} = \frac{1}{2\pi \delta t_{\rm fin}} =
        \frac{v_{\rm A}}{N R_\ast}
        \delta t_{\rm fin}.

    Main references are:

    - RoperPol:2022iel, equation 24
    - RoperPol:2023bqa, equation 15
    - EPTA:2023xxk,     equation 21
    - Caprini:2024hue,  equation 2.17
    - RoperPol:2025b

    Parameters
    ----------
    s : array_like
        Frequencies normalized by characteristic scale.
    Oms : float, optional
        Source energy density.
    lf : float, optional
        Characteristic scale fraction.
    N : int, optional
        Source duration/eddy turnover ratio.
    cs2 : float, optional
        Speed of sound squared.
    expansion : bool, optional
        Include expansion of Universe.
    tdecay : str, optional
        Decay time model ('eddy').
    tp : str, optional
        Source type ('magnetic', 'kinetic', 'max').

    Returns
    -------
    TGW : ndarray
        Logarithmic function for spectral shape.

    References
    ----------
    RoperPol:2022iel (eq. 24), RoperPol:2023bqa (eq. 15),
    Caprini:2024hue (eq. 2.17), EPTA:2023xxk (eq. 21).
    """

    mult_Oms = isinstance(Oms, (list, tuple, np.ndarray))
    mult_lf = isinstance(lf, (list, tuple, np.ndarray))
    s, Oms, lf = np.meshgrid(s, Oms, lf, indexing="ij")

    # characteristic velocity (for example, Alfven velocity or vrms)
    # see eq. 12 of RoperPol:2023bqa
    if tp == "kinetic":
        vA = np.sqrt(Oms)
    elif tp == "magnetic":
        vA = np.sqrt(2 * Oms / (1 + cs2))
    else:
        vA = np.sqrt(max(1, 2 / (1 + cs2)) * Oms)
    # decay time in units of R*
    if tdecay == "eddy":
        tdec = 1.0 / vA

    # effective duration of the source dtfin/R* is N units of
    # the decay time (see comment above for different choices
    # in the literature)
    dtfin = N * tdec

    TGW = np.zeros_like(dtfin)
    if expansion:
        inds = np.where(s < 1 / dtfin)
        TGW[inds] = (np.log(1 + dtfin * lf / 2 / np.pi) ** 2)[inds]
        inds = np.where(s >= 1 / dtfin)
        TGW[inds] = (np.log(1 + lf / 2 / np.pi / s) ** 2)[inds]
    else:
        inds = np.where(s < 1 / dtfin)
        TGW[inds] = (dtfin * lf / 2 / np.pi)[inds] ** 2
        inds = np.where(s >= 1 / dtfin)
        TGW[inds] = (lf / 2 / np.pi / s)[inds] ** 2

    TGW = reshape_output(TGW, mult_Oms, mult_lf)

    return TGW


def _get_A2(fp, l, cs2, sp):
    '''
    Compute A2 coefficient for kinetic spectrum in
    :func:`compute_kin_spec_ssm`.
    '''
    fp = np.array(fp)
    l = np.array(l)
    if sp == "sum":
        return 0.25 * (cs2 * l ** 2 + fp ** 2)
    if sp == "only_f":
        return 0.5 * fp ** 2
    if sp == "only_l":
        return 0.5 * cs2 * l ** 2
    if sp == "diff":
        return 0.25 * (fp ** 2 - cs2 * l ** 2)
    if sp == "cross":
        return -0.5 * fp * np.sqrt(cs2) * l
    raise ValueError("Unknown sp value")


def _get_nu_T(TT_ij, type_n):
    '''
    Function to compute the lifetime distribution nu_T.
    '''
    if type_n == "exp":
        return np.exp(-TT_ij)
    if type_n == "sim":
        return 0.5 * np.exp(-(TT_ij**3) / 6) * TT_ij**2
    raise ValueError("Unknown type_n value")


def _convert_to_spec(Pv, Rstar_beta, qbeta, vws):
    '''
    Function to convert power spectral density to spectrum
    '''
    for i in range(len(vws)):
        pref = qbeta**2 / Rstar_beta[i] ** 4 / (2 * np.pi**2)
        Pv[i, :] *= pref
    return Pv


def _normalize_qbeta(qbeta, vws, Rstar_beta):
    _, qbeta_mesh = np.meshgrid(vws, qbeta, indexing="ij")
    for i in range(len(vws)):
        qbeta_mesh[i, :] *= Rstar_beta[i]
    return qbeta_mesh


def compute_kin_spec_ssm(z, vws, fp, l=None, sp="sum", type_n="exp",
                         cs2=cs2_ref, min_qbeta=-4, max_qbeta=5,
                         Nqbeta=Nk_ref, min_TT=-1, max_TT=3, NTT=NTT_ref,
                         corr=False, dens=True, normbeta=True):
    """
    Compute kinetic power spectral density for the sound-shell model.

    Kinetic spectra computed for the sound-shell model from :math:`f' (z)` and
    :math:`l(z)` functions.
    :math:`f'` and :math:`l` functions need to be previously computed from
    the self-similar fluid perturbations induced by expanding bubbles using
    hydro_bubbles module.

    Parameters
    ----------
    z : array_like
        Values of z.
    vws : array_like
        Wall velocities.
    fp : array_like
        :math:`f'(z)` from hydro_bubbles.
    l : array_like, optional
        :math:`l(z)` from hydro_bubbles (if lz is True)
    sp : str, optional
        Type of function for kinetic spectrum.
    type_n : str, optional
        Nucleation history ('exp', 'sim').
    cs2 : float, optional
        Speed of sound squared (default 1/3).
    min_qbeta, max_qbeta, Nqbeta : int, optional
        Wave number range and discretization.
    min_TT, max_TT, NTT : int, optional
        Lifetime range and discretization.
    corr : bool, optional
        Correct Rstar beta with max(vw, cs) (default False)
    dens : bool, optional
        Return power spectral density (True) or kinetic spectrum (False).
    normbeta : bool, optional
        Normalize k with beta (True) or Rstar (False).

    Returns
    -------
    qbeta : ndarray
        Wave number normalized.
    Pv : ndarray
        Power spectral density or kinetic spectrum if
        dens is True or False.

    References
    ----------
    Equations (32)-(37) of RoperPol:2023dzg, RoperPol:2025a
    """

    if l is None:
        l = []

    A2 = _get_A2(fp, l, cs2, sp)
    qbeta = np.logspace(min_qbeta, max_qbeta, Nqbeta)
    TT = np.logspace(min_TT, max_TT, NTT)
    Pv = np.zeros((len(vws), len(qbeta)))
    q_ij, TT_ij = np.meshgrid(qbeta, TT, indexing="ij")
    nu_T = _get_nu_T(TT_ij, type_n)
    funcT = np.zeros((len(vws), len(qbeta), len(TT)))

    for i in range(0, len(vws)):
        funcT[i, :, :] = nu_T * TT_ij**6 * np.interp(TT_ij * q_ij, z, A2[i, :])
        Pv[i, :] = safe_trapezoid(funcT[i, :, :], TT, axis=1)

    if not dens:
        Rstar_beta = hydro_bubbles.Rstar_beta(vws=vws, cs2=cs2, corr=corr)
        Pv = _convert_to_spec(Pv, Rstar_beta, qbeta, vws)

    if not normbeta:
        qbeta = _normalize_qbeta(qbeta, vws, Rstar_beta)

    return qbeta, Pv


def OmGW_ssm_HH19(k, EK, Np=Np_ref, Nk=Nkconv_ref, plot=False, cs2=cs2_ref):

    r"""
    Compute normalized GW spectrum :math:`\Omega(k)`
    using Hindmarsh:2019phv sound-shell model
    under the delta assumption for the 'stationary' term
    (see appendix B of RoperPol:2023dzg for details).

    The resulting GW spectrum is

    .. math::
        \Omega_{\rm GW} (k) = \frac{3 \pi}{8 c_{\rm s}}
        \times \Gamma^2 \times (k/k_\ast)^2 \times (\Omega_\ast/{\cal A})^2
        \times T_{\rm GW} \times \Omega (k).

    where :math:`\Omega_\ast = \langle v^2 \rangle =
    v_{\rm rms}^2` and :math:`\cal A = \int \zeta(K) d \ln k`
    (using the normalization of RoperPol:2025a for :math:`\zeta`).

    Parameters
    ----------
    k : array_like
        Wave numbers.
    EK : array_like
        Kinetic spectrum.
    Np : int, optional
        Number of p discretizations.
    Nk : int, optional
        Number of k discretizations.
    plot : bool, optional
        Plot interpolated spectrum.
    cs2 : float, optional
        Speed of sound squared.

    Returns
    -------
    kp : ndarray
        Wave numbers.
    Omm : ndarray
        GW spectrum.

    References
    ----------
    Hindmarsh:2019phv, RoperPol:2023dzg (appendix B, eq. B3)
    """

    cs = np.sqrt(cs2)
    kp = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)

    p_inf = kp * (1 - cs) / 2 / cs
    p_sup = kp * (1 + cs) / 2 / cs

    Omm = np.zeros(len(kp))
    for i in range(0, len(kp)):

        p = np.logspace(np.log10(p_inf[i]), np.log10(p_sup[i]), Np)
        ptilde = kp[i] / cs - p
        z = -kp[i] * (1 - cs2) / 2 / p / cs2 + 1 / cs

        EK_p = np.interp(p, k, EK)
        EK_ptilde = np.interp(ptilde, k, EK)

        Omm1 = (1 - z**2) ** 2 * p / ptilde**3 * EK_p * EK_ptilde
        Omm[i] = safe_trapezoid(Omm1, p)

    return kp, Omm


def effective_ET_correlator_stat(k, EK, tfin, Np=Np_ref, Nk=Nkconv_ref,
                                 plot=False, expansion=True, kstar=1.0,
                                 extend=False, largek=3, smallk=-3,
                                 tini=tini_ref, cs2=cs2_ref,
                                 terms="all", inds_m=None, inds_n=None):

    r"""
    Compute normalized GW spectrum :math:`\zeta_{\rm GW}(k)`
    from velocity field spectrum for compressional anisotropic stresses,
    assuming stationary UETC
    (e.g., sound waves under the sound-shell model).

    Parameters
    ----------
    k : array_like
        Wave numbers.
    EK : array_like
        Kinetic spectrum.
    tfin : float
        Final time.
    Np : int, optional
        Number of p discretizations.
    Nk : int, optional
        Number of k discretizations.
    plot : bool, optional
        Plot interpolated spectrum.
    expansion : bool, optional
        Include expansion of Universe.
    kstar : float, optional
        Peak position of the kinetic spectrum.
    extend : bool, optional
        Extend wave number array.
    largek, smallk : int, optional
        Range for extended k (in log10).
    tini : float, optional
        Initial time of GW production.
    cs2 : float, optional
        Speed of sound squared (default 1/3).
    terms : str, optional
        Terms to compute ('all').
    inds_m, inds_n : list, optional
        Indices for Delta_mn.
    corr_Delta_0 : bool, optional
        Correct Delta_0.

    Returns
    -------
    kp : ndarray
        Wave numbers.
    Omm : ndarray
        Normalized GW spectrum, :math:`\zeta_{\rm GW}(k)`.

    References
    ----------
    RoperPol:2023dzg, eq. 93
    """

    if inds_m is None:
        inds_m = []
    if inds_n is None:
        inds_n = []
    p = np.logspace(np.log10(k[0]), np.log10(k[-1]), Np)
    kp = np.logspace(np.log10(k[0]), np.log10(k[-1]), Nk)
    if extend:
        Nk = int(Nk / 6)
        kp = np.logspace(smallk, np.log10(k[0]), Nk)
        kp = np.append(
            kp, np.logspace(np.log10(k[0]), np.log10(k[-1]), 4 * Nk)
        )
        kp = np.append(
            kp, np.logspace(np.log10(k[-1]), largek, Nk)
        )

    Nz = 500
    z = np.linspace(-1, 1, Nz)
    kij, pij, zij = np.meshgrid(kp, p, z, indexing="ij")
    ptilde2 = pij**2 + kij**2 - 2 * kij * pij * zij
    ptilde = np.sqrt(ptilde2)

    EK_p = np.interp(p, k, EK)
    if plot:
        plt.plot(p, EK_p)
        plt.xscale("log")
        plt.yscale("log")

    EK_ptilde = np.interp(ptilde, k, EK)
    ptilde[np.where(ptilde == 0)] = 1e-50

    Delta_mn = kij**0 - 1

    if terms == "all":
        inds_m = [-1, 1]
        inds_n = [-1, 1]
        Delta_mn = np.zeros((4, len(kp), len(p), len(z)))

    l = 0
    for m in inds_m:
        for n in inds_n:
            Delta_mn[l, :, :, :] = compute_Delta_mn(
                tfin, kij * kstar, pij * kstar, ptilde * kstar,
                cs2=cs2, m=m, n=n, tini=tini, expansion=expansion,
            )
            l += 1

    if l != 0:
        Delta_mn = Delta_mn / (l + 1)

    Omm = np.zeros((l + 1, len(kp)))
    for i in range(0, l):
        Pi_1 = safe_trapezoid(
            EK_ptilde / ptilde**4 * (1 - zij**2) ** 2 * Delta_mn[i, :, :, :],
            z, axis=2
        )
        kij, EK_pij = np.meshgrid(kp, EK_p, indexing="ij")
        kij, pij = np.meshgrid(kp, p, indexing="ij")
        Omm[i, :] = safe_trapezoid(Pi_1 * pij**2 * EK_pij, p, axis=1)

    return kp, Omm


def compute_Delta_mn(t, k, p, ptilde, cs2=cs2_ref,
                     m=1, n=1, tini=1.0, expansion=True):

    r"""
    Compute integrated Green's functions and stationary UETC Delta_mn
    for the sound shell model.

    Parameters
    ----------
    t : float
        Time.
    k : float
        Wave number k.
    p : float
        Wave number p.
    ptilde : float
        Second wave number :math:`\tilde{p} = |k - p|`.
    cs2 : float, optional
        Speed of sound squared (default 1/3).
    m, n : int, optional
        Indices for Delta_mn.
    tini : float, optional
        Initial time.
    expansion : bool, optional
        Include expansion of Universe.

    Returns
    -------
    Delta_mn : ndarray
        Integrated Green's function.

    References
    ----------
    RoperPol:2023dzg, eqs.56-59
    """

    cs = np.sqrt(cs2)
    pp = n * k + cs * (m * ptilde + p)
    pp[np.where(pp == 0)] = 1e-50

    if expansion:
        si_t, ci_t = special.sici(pp * t)
        si_tini, ci_tini = special.sici(pp * tini)
        # compute Delta Ci^2 and Delta Si^2
        DCi = ci_t - ci_tini
        DSi = si_t - si_tini
        Delta_mn = DCi**2 + DSi**2

    else:
        Delta_mn = 2 * (1 - np.cos(pp * (t - tini))) / pp**2

    return Delta_mn


# FUNCTIONS FOR THE LOCALLY STATIONARY UETC
def K2int(dtfin, K0=1.0, dt0=dt0_ref, b=0.0, expansion=False, beta=beta_ref):

    r"""
    Compute integrated kinetic energy density squared for a power law,

    .. math ::
        K(\Delta t) = K_0 \left(\frac{\Delta t}{\Delta t_0}\right)^{-b},

    where :math:`\Delta t` and :math:`\Delta t_0` represent time intervals,
    :math:`K_0` is the amplitude at the end of the phase transition,
    and :math:`\Delta t_0 = 11` is a numerical parameter of the fit.

    Based on the results of Higgsless simulations of Caprini:2024gyk,
    see equation 2.27.

    It determines the amplitude of GWs in the locally stationary
    UETC describing the decay of compressional sources, see equation
    2.33,

    .. math ::
        K_{\rm int}^2 = \int K^2 (t) d t.

    Parameters
    ----------
    dtfin : float
        Duration of sourcing (in units of :math:`1/\beta` or :math:`1/H_\ast`
        if expansion is False or True).
    K0 : float, optional
        Amplitude :math:`K_0` at end of phase transition.
    dt0 : float, optional
        Numerical parameter :math:`\Delta t_0` for fit.
    b : float, optional
        Power law exponent.
    expansion : bool, optional
        Consider expanding Universe.
    beta : float, optional
        Nucleation rate normalized to Hubble time, :math:`\beta/H_\ast`.

    Returns
    -------
    K2int : float
        Integrated squared kinetic energy density.

    References
    ----------
    Caprini:2024gyk
    """

    K2int = K0**2 * dt0

    # computation in Minkowski space-time
    if not expansion:
        if b == 0.5:
            K2int *= np.log(1 + dtfin / dt0)
        else:
            K2int *= ((1 + dtfin / dt0) ** (1.0 - 2 * b) - 1) / (1.0 - 2.0 * b)

    # computation in expanding Universe during radiation-domination
    else:

        if beta < dt0:
            print(
                "The value of beta in K2int cannot be smaller than",
                "dt0 = %.1f for the chosen model" % dt0,
            )
            print("Choose a larger value of beta to include expansion")
            return 0

        dt0 = dt0 / beta
        K2int *= 1.0 / beta / (dt0 - 1.0) ** 2

        if b == 0.5:
            K2int *= (dt0 - 1.0) / (1.0 + dtfin) + np.log(
                (1 + dtfin / dt0) / (1 + dtfin)
            )

        else:
            K2int *= 1.0 / (1.0 - 2 * b)
            A = special.hyp2f1(
                2, 1 - 2 * b, 2.0 - 2 * b, (dt0 + dtfin) / (dt0 - 1)
            )
            B = special.hyp2f1(2, 1 - 2 * b, 2.0 - 2 * b, dt0 / (dt0 - 1.0))
            K2int *= (1 + dtfin / dt0) ** (1.0 - 2 * b) * A - B

    return K2int
