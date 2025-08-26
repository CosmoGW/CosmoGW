r"""
GW_templates.py is a Python routine that contains analytical and semi-analytical
templates of cosmological GW backgrounds, usually based on spectral fits,
either from GW models (see GW_models) or from numerical simulations

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/GW_templates.py

.. note::
   For full documentation, visit `Read the Docs
   <https://cosmogw-manual.readthedocs.io/en/latest/GW_templates.html>`_.

To use it, first install `cosmoGW <https://pypi.org/project/cosmoGW>`_::

    pip install cosmoGW

Author
------
- **Alberto Roper Pol**
  (`alberto.roperpol@unige.ch <mailto:alberto.roperpol@unige.ch>`_)

Dates
-----
- Created: **01/12/2022**
- Updated: **31/08/2024**
- Updated: **21/08/2025**
  (release **cosmoGW 1.0**: https://pypi.org/project/cosmoGW)

References
----------
- [**Espinosa:2010hh**]: J. R. Espinosa, T. Konstandin, J. M. No, G. Servant,
  "*Energy Budget of Cosmological First-order Phase Transitions,*"
  JCAP **06** (2010), 028, `arXiv:1004.4187 <https://arxiv.org/abs/1004.4187>`_.

- [**Hindmarsh:2017gnf**]: M. Hindmarsh, S. J. Huber, K. Rummukainen,
  D. J. Weir, "*Shape of the acoustic gravitational wave
  power spectrum from a first order phase transition,*"
  Phys.Rev.D **96** (2017) 10, 103520, Phys.Rev.D **101** (2020) 8,
  089902 (erratum), `arXiv:1704.05871 <https://arxiv.org/abs/1704.05871>`_.

- [**Hindmarsh:2019phv**]: M. Hindmarsh, M. Hijazi, "*Gravitational waves from
  first order cosmological phase transitions in the Sound Shell Model,*"
  JCAP **12** (2019) 062,
  `arXiv:1909.10040 <https://arxiv.org/abs/1909.10040>`_.

- [**Caprini:2019egz**]: [LISA CosWG], "*Detecting gravitational
  waves from cosmological phase transitions with LISA: an update,*"
  JCAP **03** (2020) 024,
  `arXiv:1910.13125 <https://arxiv.org/abs/1910.13125>`_.

- [**Hindmarsh:2020hop**]: M. Hindmarsh, M. Lueben, J. Lumma,
  M. Pauly, "*Phase transitions in the early universe,*"
  SciPost Phys. Lect. Notes **24** (2021), 1,
  `arXiv:2008.09136 <https://arxiv.org/abs/2008.09136>`_.

- [**Jinno:2022mie**]: R. Jinno, T. Konstandin, H. Rubira, I. Stomberg,
  "*Higgsless simulations of cosmological phase transitions and
  gravitational waves,*" JCAP **02**, 011 (2023),
  `arXiv:2209.04369 <https://arxiv.org/abs/2209.04369>`_.

- [**RoperPol:2022iel**]: A. Roper Pol, C. Caprini, A. Neronov,
  D. Semikoz, "*The gravitational wave signal from primordial
  magnetic fields in the Pulsar Timing Array frequency band,*"
  Phys. Rev. D **105**, 123502 (2022),
  `arXiv:2201.05630 <https://arxiv.org/abs/2201.05630>`_.

- [**RoperPol:2023bqa**]: A. Roper Pol, A. Neronov, C. Caprini, T. Boyer,
  D. Semikoz, "*LISA and γ-ray telescopes as multi-messenger probes of a
  first-order cosmological phase transition,*"
  `arXiv:2307.10744 <https://arxiv.org/abs/2307.10744>`_ (2023)

- [**EPTA:2023xxk**]: [EPTA and InPTA Collaborations], "*The second data
  release from the European Pulsar Timing Array - IV. Implications
  for massive black holes, dark matter, and the early Universe,*"
  Astron. Astrophys. **685**, A94 (2024),
  `arXiv:2306.16227 <https://arxiv.org/abs/2306.16227>`_

- [**RoperPol:2023dzg**]: A. Roper Pol, S. Procacci, C. Caprini,
  "*Characterization of the gravitational wave spectrum from sound waves within
  the sound shell model,*" Phys. Rev. D **109**, 063531 (2024),
  `arXiv:2308.12943 <https://arxiv.org/abs/2308.12943>`_.

- [**Caprini:2024gyk**]: A. Roper Pol, I. Stomberg, C. Caprini, R. Jinno,
  T. Konstandin, H. Rubira, "*Gravitational waves from first-order
  phase transitions: from weak to strong,*" JHEP **07** (2025) 217,
  `arXiv:2409.03651 <https://arxiv.org/abs/2409.03651>`_.

- [**Caprini:2024hue**]: E. Madge, C. Caprini, R. Jinno, M. Lewicki,
  M. Merchand, G. Nardini, M. Pieroni, A. Roper Pol, V. Vaskonen,
  "*Gravitational waves from first-order phase transitions in LISA:
  reconstruction pipeline and physics interpretation,*"
  JCAP **10** (2024) 020,
  `arXiv:2403.03723 <https://arxiv.org/abs/2403.03723>`_.

- [**RoperPol:2025b**]: A. Roper Pol, A. Midiri, M. Salomé, C. Caprini,
  "*Modeling the gravitational wave spectrum from slowly decaying sources in the
  early Universe: constant-in-time and coherent-decay models,*" in preparation

- [**RoperPol:2025a**]: A. Roper Pol, S. Procacci, A. S. Midiri,
  C. Caprini, "*Irrotational fluid perturbations from first-order phase
  transitions,*" in preparation

Comments
--------
Reference values for turbulence template are
based on RoperPol:2023bqa and RoperPol:2022iel.
Template used in Caprini:2024hue for LISA and in EPTA:2023xxk

Note that the values used for a_turb, b_turb, bPi_vort, fPi
are found assuming that the spectrum of the source is defined
such that

.. math ::
    \langle v^2 \rangle \propto 2 E_\ast \int \zeta(K) d \ln K,
    \quad {\rm as \  in \ RoperPol:2025b},

which is a different normalization than that in previous papers,
where zeta is defined such that

.. math ::
    \langle v^2 \rangle \propto 2 E_\ast k_\ast
    \int \zeta(K) dK, \quad {\rm in \ RoperPol:2022iel}.

Hence, the values are considered for the former :math:`\zeta`
(this choice yields different coefficients).
However, the final result is not affected by this choice.
See RoperPol:2025b for details
"""

import numpy as np
import pandas as pd

from cosmoGW import GW_back, GW_analytical, GW_models, hydro_bubbles
from cosmoGW import COSMOGW_HOME
from cosmoGW.utils import (
    OmGW_sw_ref, Oms_ref, lf_ref, beta_ref, cs2_ref, gref, Tref, Neff_ref,
    a_sw_ref, b_sw_ref, c_sw_ref, zpeak1_LISA_old, alp1_ssm, alp2_ssm,
    alp1_sw_ref, alp2_sw_ref, alp1_LISA, alp2_LISA, peak1_LISA, peak2_LISA,
    alp1_HL, alp2_HL, peak1_HL, peak2_HL_weak, peak2_HL_interm, peak2_HL_str,
    dt0_ref, bs_HL_eff, bs_k1HL, a_ref, b_ref, alp_turb,
    alpPi, fPi, bPi_vort, N_turb, tdecay_ref,
    safe_trapezoid, reshape_output
)


# SOUND WAVES
# values from Higgsless simulations
def _data_warning(boxsize=bs_HL_eff):

    r"""
    Print a warning about the validity range of interpolated numerical data.

    This function informs the user that the values being used are interpolated
    from numerical simulations with a given box size (:math:`L/v_w`).
    It also warns that only certain values of :math:`\alpha` (0.0046, 0.05, 0.5)
    and wall velocities (:math:`v_w` from 0.32 to 0.8) are available
    in the simulations.
    Values outside this range should be used with caution.

    Parameters
    ----------
    boxsize : int, optional
        The box size (:math:`L/v_w`) used in the numerical simulations
        (default 20).

    Returns
    -------
    None
    """

    print('You are using values that are interpolated from numerical',
          'data with L/vw = ', boxsize)
    print('Take into account that only alpha = 0.0046, 0.05',
          ' and 0.5 are found in simulations for vws from 0.32 to 0.8')
    print('Values out of this range should be taken with care')


def _fill_grid(df2, val_vws, val_alphas, value):
    '''
    Function to read variable values from the DataFrame and fill a grid
    used in :func:`interpolate_HL_vals`.
    '''
    Omegas = np.full((len(val_vws), len(val_alphas)), -1e30)
    for i, alpha in enumerate(val_alphas):
        for j, vw in enumerate(val_vws):
            Om = np.array(df2[value][(df2.v_wall == vw) & (df2.alpha == alpha)])
            if len(Om) > 0:
                Omegas[j, i] = Om[0]
                if value == 'curly_K_0_512':
                    Omegas[j, i] *= (1 + alpha) / alpha
    return Omegas


def _interpolate_vws(Omegas, vws, val_vws):
    '''
    Interpolate Omega values to arbitrary wall velocities
    used in :func:`interpolate_HL_vals`.
    '''
    Omss = np.zeros((len(vws), Omegas.shape[1]))
    for i in range(Omegas.shape[1]):
        inds = np.where(Omegas[:, i] > -1e30)[0]
        Omss[:, i] = np.interp(vws, val_vws[inds], Omegas[inds, i])
        inds2 = np.where(Omegas[:, i] == -1e30)[0]
        Omegas[inds2, i] = np.interp(
            val_vws[inds2], val_vws[inds], Omegas[inds, i]
        )
    return Omss, Omegas


def _interpolate_alphas(Omss, alphas, val_alphas, vws, value):
    '''
    Interpolate Omega values to arbitrary alpha values
    used in :func:`interpolate_HL_vals`.
    '''
    mult_alpha = isinstance(alphas, (list, tuple, np.ndarray))
    if mult_alpha:
        Omsss = np.zeros((len(vws), len(alphas)))
        for i in range(len(vws)):
            Omsss[i, :] = np.interp(
                np.log10(alphas), np.log10(val_alphas), Omss[i, :]
            )
        if value == 'curly_K_0_512':
            _, alpsij = np.meshgrid(vws, alphas, indexing='ij')
            Omsss *= alpsij / (1 + alpsij)
    else:
        Omsss = np.zeros(len(vws))
        for i in range(len(vws)):
            Omsss[i] = np.interp(
                np.log10(alphas), np.log10(val_alphas), Omss[i, :]
            )
        if value == 'curly_K_0_512':
            Omsss *= alphas / (1 + alphas)
    return Omsss


def interpolate_HL_vals(df, vws, alphas, value='Omega_tilde_int_extrap',
                        boxsize=bs_k1HL, numerical=False, quiet=False):

    r"""
    Interpolate Higgsless simulation results to arbitrary wall velocities
    (:math:`v_w`) and alpha (:math:`\alpha`).

    This function uses numerical results from Caprini:2024gyk to interpolate
    quantities (such as GW efficiency or spectral parameters) for different
    values of wall velocity and phase transition strength.
    The data is read from::

       resources/higgsless/parameters_fit_sims.csv

    If numerical is True, the function also returns the full grid of
    simulation values and the available :math:`\alpha` and
    :math:`v_w` values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the simulation results.
    vws : array_like
        Array of wall velocities for interpolation.
    alphas : array_like
        Array of alpha values (phase transition strength) for interpolation.
    value : str, optional
        Name of the column in the DataFrame to interpolate (default:
        'Omega_tilde_int_extrap').
    boxsize : int, optional
        Box size (L/vw) used in the simulation (default: 40).
    numerical : bool, optional
        If True, also return the full grid of simulation values and available
        alpha and v_wall values.
    quiet : bool, optional
        If False, print warnings about the validity range of the interpolation.

    Returns
    -------
    Omsss : ndarray
        Interpolated values for the requested quantity.
    Omegas : ndarray, optional
        Full grid of simulation values (only if numerical=True).
    val_alphas : ndarray, optional
        Available alpha values in the simulation (only if numerical=True).
    val_vws : ndarray, optional
        Available wall velocity values in the simulation
        (only if numerical=True).

    References
    ----------
    Caprini:2024gyk
    """

    columns = df.box_size == boxsize
    df2 = df[columns]
    val_alphas = np.unique(df2['alpha'])
    val_vws = np.unique(df2['v_wall'])

    # construct Omegas from numerical data
    Omegas = _fill_grid(df2, val_vws, val_alphas, value)
    # interpolate for all values of vws for the 3 values of alpha
    Omss, Omegas = _interpolate_vws(Omegas, vws, val_vws)
    if value == 'curly_K_0_512':
        _, alpsij = np.meshgrid(val_vws, val_alphas, indexing='ij')
        Omegas *= alpsij / (1 + alpsij)
    Omsss = _interpolate_alphas(Omss, alphas, val_alphas, vws, value)
    mult_alpha = isinstance(alphas, (list, tuple, np.ndarray))
    mult_vws = isinstance(vws, (list, tuple, np.ndarray))
    Omsss = reshape_output(Omsss, mult_a=mult_vws, mult_b=mult_alpha)

    if not quiet:
        _data_warning(boxsize=boxsize)
        if not numerical:
            print(
                'To see numerical values call interpolate_HL_vals function',
                ' setting numerical to True'
            )

    # if numerical is chosen, also return numerical values
    if numerical:
        return Omsss, Omegas, val_alphas, val_vws
    else:
        return Omsss


def _get_higgsless(vws, alphas, numerical, bs_HL, quiet):
    '''
    Get Higgsless model parameters for given wall velocities
    and alpha values, used in :func:`ampl_GWB_sw`.
    '''
    val_str = 'Omega_tilde_int_extrap'
    if len(vws) == 0 or len(alphas) == 0:
        print('Provide values of vws and alphas to use Higgsless model',
              'in ampl_GWB_sw')
        return 0, None, None, None
    dirr = COSMOGW_HOME + 'resources/higgsless/parameters_fit_sims.csv'
    df = pd.read_csv(dirr)
    if numerical:
        return interpolate_HL_vals(
            df, vws, alphas, value=val_str, boxsize=bs_HL,
            numerical=numerical, quiet=quiet
        )
    else:
        return interpolate_HL_vals(
            df, vws, alphas, value=val_str, boxsize=bs_HL, quiet=quiet
        ), None, None, None


# TEMPLATE FOR SOUND WAVES
def ampl_GWB_sw(model='fixed_value', OmGW_sw=OmGW_sw_ref, vws=None,
                alphas=None, numerical=False, bs_HL=bs_HL_eff, quiet=False):

    r"""
    Compute the GW efficiency parameter for sound waves.

    This function returns the normalized amplitude (efficiency) of the GW
    spectrum from sound waves, based on either a fixed reference value or
    interpolation from Higgsless simulation results.

    - For `model='fixed_value'`, the amplitude is set to a constant value
      (default: 1e-2), based on Hindmarsh:2019phv and Hindmarsh:2017gnf.
    - For `model='higgsless'`, the amplitude is interpolated from simulation
      data (Caprini:2024gyk) for given wall velocities and alpha values.
      The simulation data is available for :math:`\alpha` = 0.0046,
      0.05, 0.5 and wall velocities in the range 0.32 to 0.8.

    Parameters
    ----------
    model : str, optional
        Model for GW efficiency ('fixed_value' or 'higgsless').
    OmGW_sw : float, optional
        Reference value for GW efficiency, :math:`\tilde \Omega_{\rm GW}`
        (default: 1e-2).
    vws : array_like, optional
        Array of wall velocities for interpolation (required for 'higgsless').
    alphas : array_like, optional
        Array of alpha values for interpolation (required for 'higgsless').
    numerical : bool, optional
        If True and model is 'higgsless', also return the full grid of
        simulation values and available alpha and v_wall values.
    bs_HL : int, optional
        Box size (L/vw) used in the simulation (default: 20).
    quiet : bool, optional
        If False, print warnings about the validity range of the interpolation.

    Returns
    -------
    Omegas : ndarray
        GW efficiency parameter for the requested values.
    Omnum : ndarray, optional
        Full grid of simulation values (only if numerical=True and
        model='higgsless').
    val_alphas : ndarray, optional
        Available alpha values in the simulation (only if numerical=True
        and model='higgsless').
    val_vws : ndarray, optional
        Available wall velocity values in the simulation
        (only if numerical=True and model='higgsless').

    References
    ----------
    Hindmarsh:2019phv, Hindmarsh:2017gnf, Caprini:2024gyk
    """

    if vws is None:
        vws = []
    if alphas is None:
        alphas = []

    mult_alp = isinstance(alphas, (list, tuple, np.ndarray))
    mult_vw = isinstance(vws, (list, tuple, np.ndarray))

    if not mult_alp:
        alphas = [alphas]
    if not mult_vw:
        vws = [vws]

    if model == 'fixed_value':
        Omegas = np.full((len(vws), len(alphas)), OmGW_sw)
        Omnum = val_alphas = val_vws = None
    elif model == 'higgsless':
        result = _get_higgsless(vws, alphas, numerical, bs_HL, quiet)
        if numerical:
            Omegas, Omnum, val_alphas, val_vws = result
        else:
            Omegas, Omnum, val_alphas, val_vws = result[0], None, None, None
    else:
        print('Choose an available model for ampl_GWB_sw for sound waves')
        print('Available models are fixed_value and higgsless')
        return 0

    Omegas = reshape_output(Omegas, mult_a=mult_vw, mult_b=mult_alp)

    if numerical and model == 'higgsless':
        return Omegas, Omnum, val_alphas, val_vws
    else:
        return Omegas


def pref_GWB_sw(Oms=Oms_ref, lf=lf_ref, alpha=0, model='sound_waves',
                Nshock=1., b=0., expansion=True, beta=beta_ref, cs2=cs2_ref):

    r"""
    Compute the prefactor for the GW spectrum from sound waves.

    This function calculates the dependence of the GW spectrum amplitude
    on the kinetic energy density, :math:`\Omega_\ast`
    and the mean bubble size :math:`l_\ast = R_\ast H_\ast`,
    for different physical models of the source duration and decay.

    .. math::
        \Omega_\ast = v_{\rm rms}^2 = \frac{\langle w v^2 \rangle}
        {\langle w \rangle},

    related to :math:`K = \kappa \alpha/(1 + \alpha)`
    used in Caprini:2024gyk as

    .. math::
        \Omega_\ast = \frac{K}{\Gamma} =
        \frac{\kappa \alpha}{(1 + c_s^2)},
        \quad {\rm \ where \ }
        \Gamma = \frac{\langle w \rangle}{\langle \rho \rangle} =
        \frac{1 + c_{\rm s}^2}{1 + \alpha}

    is the adiabatic index.

    Note that RoperPol:2023dzg uses :math:`\Omega_K = {1 \over 2} v_{\rm rms}^2`
    so an extra factor of 2 appears in its relation to :math:`K`.

    - For `model='sound_waves'`, the prefactor is computed using the
      stationary sound wave assumption, including the effect of Universe
      expansion via the suppression factor Upsilon
      (Caprini:2024gyk eqs. 2.16 and 2.30).
      It corresponds to the linear growth with the source duration,
      which is set to

      .. math::
        t_{\rm dur} = N_{\rm shock} t_{\rm shock} =
        N_{\rm shock} R_\ast/\sqrt{\Omega_\ast},

      assuming sound waves do not decay and their UETC is stationary:

      .. math::

           \Omega_{\rm GW} \sim K^2 R_\ast t_{\rm dur} =
           K^2 R_\ast N_{\rm shock} t_{\rm shock}.

      When the Universe expansion is included, :math:`t_{\rm dur}`
      is substituted by the suppression factor :math:`\Upsilon`
      describing the growth with the source duration.
      For a radiation-dominated Universe (see RoperPol:2023dzg):

      .. math::

           \Upsilon = \frac{t_{\rm dur}}{1 + t_{\rm dur}}

      This choice includes Universe expansion and assumes sourcing
      occurs during the radiation-dominated era

      .. math::
           \Omega_{\rm GW} \sim K^2 R_\ast \Upsilon(t_{\rm dur})
           = K^2 R_\ast \frac{t_{\rm dur}}{1 + t_{\rm dur}}.

      When :math:`N_{\rm shock} = 1, t_{\rm dur} = t_{\rm shock} =
      R_\ast/\sqrt{\Omega_{\rm s}}` and it becomes equation 3
      of RoperPol:2023bqa, based on Hindmarsh:2020hop, equation 8.24.

    - For `model='decay'`, the prefactor includes the decay of the source
      and uses a power-law decay for the kinetic energy density.
      It uses Caprini:2024gyk, eq. 5.6, assuming
      a locally stationary UETC.
      For Minkowski space-time:

      .. math ::

           \Omega_{\rm GW} \sim K^2_{\rm int} R_\ast, \quad
           {\rm \ where \ } K^2_{\rm int} = \int K^2 dt,

      while for a radiation-dominated expanding Universe:

      .. math ::
           \Omega_{\rm GW} \sim K^2_{\rm int, exp} R_\ast, \quad
           {\rm \ where \ } K^2_{\rm int, exp} = \int \frac{K^2}{t^2} dt.

    Parameters
    ----------
    Oms : float or array_like, optional
        Kinetic energy density, defined as :math:`v_{\rm rms}^2`,
        such that :math:`\Omega_\ast = K/\Gamma = (1 + c_s^2) \kappa \alpha`.
    lf : float or array_like, optional
        Mean bubble size as a fraction of the Hubble radius.
    alpha : float, optional
        Ratio of vacuum to radiation energy densities.
    model : str, optional
        Model for the prefactor ('sound_waves' or 'decay').
    Nshock : float, optional
        Duration time in units of the shock time.
    b : float, optional
        Power law decay exponent for the kinetic energy source.
        Default 0 recovers stationary sound waves.
    expansion : bool, optional
        If True, include Universe expansion (radiation-dominated era).
    beta : float, optional
        Rate of nucleation of the phase transition.
    cs2 : float, optional
        Square of the speed of sound (default is 1/3 for radiation domination).

    Returns
    -------
    pref : float or ndarray
        Prefactor for the GW spectrum, consisting of :math:`K^2 R_\ast`
        or :math:`K^2_{\rm int, exp} R_\ast`, depending on the chosen model.

    References
    ----------
    Caprini:2024gyk, RoperPol:2023bqa, RoperPol:2023dzg
    """

    if len(np.shape(alpha)) == 0:
        if alpha == 0:
            print(
                'you need to give the value of alpha as input',
                'for a correct result in pref_GWB_sw'
            )
            alpha = np.zeros_like(Oms)

    Gamma = (1. + cs2) / (1. + alpha)
    K = Gamma * Oms

    pref = K ** 2 * lf
    tdur = Nshock * lf / np.sqrt(Oms)

    if model == 'sound_waves':
        if expansion:
            pref *= 1. / (1. + 1. / tdur)
        else:
            pref *= tdur

    elif model == 'decay':
        K2int = GW_models.K2int(
            tdur, K0=K, b=b, dt0=dt0_ref,
            expansion=expansion, beta=beta
        )
        pref = K2int * lf

    return pref


def Sf_shape_sw(s, model='sw_LISA', Dw=1., a_sw=a_sw_ref, b_sw=b_sw_ref,
                c_sw=c_sw_ref, alp1_sw=0, alp2_sw=0, strength='weak',
                interpolate_HL=False, bs_k1HL=bs_k1HL, bs_k2HL=bs_HL_eff,
                vws=None, alphas=None, quiet=False,
                interpolate_HL_n3=False, corrRs=True, cs2=cs2_ref):

    r"""
    Compute the GW spectral shape generated by sound waves based on
    different templates.

    Several models are available, including analytic fits and
    interpolation from Higgsless simulations.

    Parameters
    ----------
    s : array_like
        Normalized wave number, divided by the mean bubble size,
        :math:`s = f R_\ast`.
    model : str, optional
        Model for the sound-wave template
        ('sw_SSM', 'sw_HL', 'sw_LISA', 'sw_LISAold', 'sw_HLnew').
    Dw : float or array_like, optional
        Ratio between peak frequencies :math:`\Delta_w`,
        determined by the fluid shell thickness.
    a_sw, b_sw, c_sw : float, optional
        Slopes for sound wave template (default 3, 1, 3).
    alp1_sw, alp2_sw : float, optional
        Transition smoothness parameters for sound wave template.
    strength : str, optional
        Phase transition strength ('weak', 'interm', 'strong'),
        used to determine peak2 in sw_HLnew.
    interpolate_HL : bool, optional
        Use numerical data from Caprini:2024gyk for slope and peak positions.
    bs_k1HL, bs_k2HL : int, optional
        Box size :math:`L/v_w` of Higgsless simulations for peak interpolation.
    vws, alphas : array_like, optional
        Wall velocities and alpha values for parameter estimation.
    quiet : bool, optional
        Suppress warnings about interpolation range.
    interpolate_HL_n3 : bool, optional
        Use Higgsless simulations to estimate
        value of high-frequency slope c_sw.
    corrRs : bool, optional
        Correct Rstar beta with max(vw, cs).
    cs2 : float, optional
        Square of the speed of sound (default 1/3).

    Returns
    -------
    S : ndarray
        Spectral shape of the GW spectrum as a function of :math:`s = f R_\ast`.
        If interpolate_HL is True, returns an array of shape (s, alphas, vws).

    References
    ----------
    RoperPol:2023bqa, Hindmarsh:2019phv, Jinno:2022mie, Caprini:2024gyk
    """
    if vws is None:
        vws = []
    if alphas is None:
        alphas = []

    # Prepare sound-shell thickness for the models where it is
    # required
    mult_Dw, s, Dw = _prepare_Dw(s, Dw, model, strength, interpolate_HL)

    if model == 'sw_LISAold':
        return _shape_sw_LISAold(s)
    elif model == 'sw_SSM':
        return _shape_sw_SSM(s, Dw, a_sw, b_sw, c_sw, alp1_sw, alp2_sw, mult_Dw)
    elif model == 'sw_HL':
        return _shape_sw_HL(s, Dw, a_sw, b_sw, c_sw, alp1_sw, alp2_sw, mult_Dw)
    elif model == 'sw_LISA':
        return _shape_sw_LISA(s, Dw, a_sw, b_sw, c_sw, alp1_sw, alp2_sw)
    elif model == 'sw_HLnew':
        return _shape_sw_HLnew(
            s, Dw, a_sw, b_sw, c_sw, alp1_sw, alp2_sw, strength,
            interpolate_HL, bs_k1HL, bs_k2HL, vws, alphas, quiet,
            interpolate_HL_n3, corrRs, cs2
        )
    else:
        print(
            'Choose an available model in Sf_shape_sw for the sound wave '
            'spectral shape.'
        )
        print('Available models are sw_LISAold, sw_SSM, sw_HL, sw_LISA, '
              'sw_HLnew')
        return 0


def _prepare_Dw(s, Dw, model, strength, interpolate_HL):
    '''
    Prepare sound-shell thickness for the models where it is required,
    called from :func:`Sf_shape_sw`.
    '''
    mult_Dw = isinstance(Dw, (list, tuple, np.ndarray))
    Dw_2d = False
    if model == 'sw_LISA':
        Dw_2d = True
    if model == 'sw_HLnew' and strength == 'weak' and not interpolate_HL:
        Dw_2d = True
    if Dw_2d:
        NDw = len(np.shape(Dw))
        if NDw == 1:
            s, Dw = np.meshgrid(s, Dw, indexing='ij')
        elif NDw == 2:
            s0 = np.zeros((len(s), np.shape(Dw)[0], np.shape(Dw)[1]))
            Dw0 = np.zeros((len(s), np.shape(Dw)[0], np.shape(Dw)[1]))
            for i in range(np.shape(Dw)[0]):
                for j in range(np.shape(Dw)[1]):
                    s0[:, i, j] = s
                    Dw0[:, i, j] = Dw[i, j]
            s = s0
            Dw = Dw0
    return mult_Dw, s, Dw


def _shape_sw_LISAold(s):
    '''
    Shape function for the LISA old model, called from
    :func:`Sf_shape_sw`.

    Reference for sound waves based on simulations of Hindmarsh:2017gnf
    is Caprini:2019egz (equation 30) with only one peak.
    '''
    peak1 = 2 * np.pi / zpeak1_LISA_old
    s = peak1 * s
    return s ** 3 * (7.0 / (4 + 3 * s ** 2)) ** (7 / 2)


def _shape_sw_SSM(s, Dw, a_sw, b_sw, c_sw, alp1_sw, alp2_sw, mult_Dw):
    '''
    Shape function for the Sound Shell Model (sw_SSM), called from
    :func:`Sf_shape_sw`.

    Reference for sound waves based on Sound Shell Model (sw_SSM) is
    RoperPol:2023bqa, equation 6, based on the results presented in
    Hindmarsh:2019phv, equation 5.7.
    Uses Dw = |vw - cs|/max(vw, cs)
    '''

    s, Dw = np.meshgrid(s, Dw, indexing='ij')
    # different slope at large frequencies to adapt the result
    # at intermediate ones
    if c_sw == 3:
        c_sw = 4
    # takes a different slope at small frequencies
    if a_sw == 3:
        a_sw = 9
    m = (9 * Dw ** 4 + 1) / (Dw ** 4 + 1)
    # amplitude such that S = 1 at s = 1/Dw
    A = Dw ** 9 * (1 + Dw ** (-4)) ** 2 * (5 / (5 - m)) ** (5 / 2)
    # peak positions
    peak1 = 1.
    peak2 = np.sqrt((5 - m) / m) / Dw
    if alp1_sw == 0:
        alp1_sw = alp1_ssm
    if alp2_sw == 0:
        alp2_sw = alp2_ssm
    S = A * GW_analytical.smoothed_double_bPL(
        s, peak1, peak2, A=1., a=a_sw, b=b_sw, c=c_sw,
        alp1=alp1_sw, alp2=alp2_sw, alpha2=True
    )
    if not mult_Dw:
        S = S[:, 0]
    return S


def _shape_sw_HL(s, Dw, a_sw, b_sw, c_sw, alp1_sw, alp2_sw, mult_Dw):
    '''
    Shape function for the Higgsless (sw_HL) model, called from
    :func:`Sf_shape_sw`.

    Reference for sound waves based on Higgsless (sw_HL) simulations is
    RoperPol:2023bqa, equation 6, based on the results presented in
    Hindmarsh:2019phv, equation 5.7.
    Uses Dw = |vw - cs|/max(vw, cs)
    '''

    s, Dw = np.meshgrid(s, Dw, indexing='ij')
    # amplitude such that S = 1 at s = 1/Dw
    A = 16 * (1 + Dw ** (-3)) ** (2 / 3) * Dw ** 3
    # peak positions
    peak1 = 1.
    peak2 = np.sqrt(3) / Dw
    if alp1_sw == 0:
        alp1_sw = alp1_sw_ref
    if alp2_sw == 0:
        alp2_sw = alp2_sw_ref
    S = A*GW_analytical.smoothed_double_bPL(
        s, peak1, peak2, A=1., a=a_sw, b=b_sw,
        c=c_sw, alp1=alp1_sw, alp2=alp2_sw
    )
    if not mult_Dw:
        S = S[:, 0]
    return S


def _shape_sw_LISA(s, Dw, a_sw, b_sw, c_sw, alp1_sw, alp2_sw):

    '''
    Shape function for the LISA model, called from
    :func:`Sf_shape_sw`.

    Reference for sound waves based on Higgsless simulations is
    Caprini:2024hue (equation 2.8), based on the results presented in
    Jinno:2022mie, see updated results and discussion in Caprini:2024gyk.
    Uses Dw = xi_shell/max(vw, cs)
    '''
    # smoothness parameters
    if alp1_sw == 0:
        alp1_sw = alp1_LISA
    if alp2_sw == 0:
        alp2_sw = alp2_LISA
    # peak positions
    peak1 = peak1_LISA
    peak2 = peak2_LISA / Dw
    return GW_analytical.smoothed_double_bPL(
        s, peak1, peak2, A=1., a=a_sw, b=b_sw,
        c=c_sw, alp1=alp1_sw, alp2=alp2_sw, alpha2=True
    )


def _shape_sw_HLnew(s, Dw, a_sw, b_sw, c_sw, alp1_sw, alp2_sw, strength,
                    interpolate_HL, bs_k1HL, bs_k2HL, vws, alphas, quiet,
                    interpolate_HL_n3, corrRs, cs2):
    '''
    Shape function for the Higgsless (sw_HLnew) model, called from
    :func:`Sf_shape_sw`.

    Reference for sound waves based on updated HL results (sw_HLnew)
    Caprini:2024gyk.
    Uses Dw = xi_shell/max(vw, cs).
    '''

    # smoothness parameters
    if alp1_sw == 0:
        alp1_sw = alp1_HL
    if alp2_sw == 0:
        alp2_sw = alp2_HL
    if not interpolate_HL:
        # peak positions
        peak1, peak2 = _get_peaks_HLnew(strength, Dw)
        return _compute_shape_HLnew(
            s, peak1, peak2, a_sw, b_sw, c_sw, alp1_sw, alp2_sw
        )
    else:
        if len(vws) == 0 or len(alphas) == 0:
            print('To use interpolate_HL in Sf_shape_sw, '
                  'give values of vws and alphas')
            return 0
        # take values from higgsless dataset
        peaks1, peaks2, s, c_sw = _interpolate_peaks_and_shape_HLnew(
            s, vws, alphas, bs_k1HL, bs_k2HL, quiet,
            corrRs, cs2, interpolate_HL_n3
        )
        return _compute_shape_HLnew(
            s, peaks1, peaks2, a_sw, b_sw, c_sw, alp1_sw, alp2_sw
        )


def _get_peaks_HLnew(strength, Dw):
    peak1 = peak1_HL
    if strength == 'weak':
        peak2 = peak2_HL_weak / Dw
    elif strength == 'interm':
        peak2 = peak2_HL_interm
    else:
        peak2 = peak2_HL_str
    return peak1, peak2


def _compute_shape_HLnew(s, peak1, peak2, a_sw, b_sw, c_sw,
                         alp1_sw, alp2_sw):
    return GW_analytical.smoothed_double_bPL(
        s, peak1, peak2, A=1., a=a_sw, b=b_sw,
        c=c_sw, alp1=alp1_sw, alp2=alp2_sw, alpha2=True
    )


def _interpolate_peaks_and_shape_HLnew(s, vws, alphas, bs_k1HL, bs_k2HL,
                                       quiet, corrRs, cs2,
                                       interpolate_HL_n3):
    dirr = COSMOGW_HOME + 'resources/higgsless/parameters_fit_sims.csv'
    df = pd.read_csv(dirr)
    val_str = 'k1'
    peaks1 = interpolate_HL_vals(
        df, vws, alphas, quiet=True, value=val_str, boxsize=bs_k1HL
    )
    val_str = 'k2'
    peaks2 = interpolate_HL_vals(
        df, vws, alphas, quiet=True, value=val_str, boxsize=bs_k2HL
    )
    s0 = np.zeros((len(s), len(vws), len(alphas)))
    peaks10 = np.zeros((len(s), len(vws), len(alphas)))
    peaks20 = np.zeros((len(s), len(vws), len(alphas)))
    Rstar_beta = hydro_bubbles.Rstar_beta(
        vws=vws, corr=corrRs, cs2=cs2
    ) / 2 / np.pi
    for i in range(len(vws)):
        for j in range(len(alphas)):
            peaks10[:, i, j] = peaks1[i, j] * Rstar_beta[i]
            peaks20[:, i, j] = peaks2[i, j] * Rstar_beta[i]
            s0[:, i, j] = s
    peaks1 = peaks10
    peaks2 = peaks20
    s = s0
    if interpolate_HL_n3:
        val_str = 'n3'
        c_sw = -interpolate_HL_vals(
            df, vws, alphas, quiet=True, value=val_str, boxsize=bs_k2HL
        )
    else:
        c_sw = None
    if not quiet:
        _data_warning(boxsize=f'{bs_k1HL} and {bs_k2HL}')
    return peaks1, peaks2, s, c_sw


def OmGW_spec_sw(
    s, alphas, betas, vws=1., cs2=cs2_ref, quiet=True,
    a_sw=a_sw_ref, b_sw=b_sw_ref, c_sw=c_sw_ref, alp1_sw=0, alp2_sw=0,
    corrRs=True, expansion=True, Nsh=1., model_efficiency='fixed_value',
    OmGW_tilde=OmGW_sw_ref, bs_HL_eff=bs_HL_eff, model_K0='Espinosa',
    bs_k1HL=bs_k1HL, model_decay='sound_waves', interpolate_HL_decay=True, b=0,
    model_shape='sw_LISA', strength='weak', interpolate_HL_shape=False,
    interpolate_HL_n3=False, redshift=False, gstar=gref, gS=0,
    T=Tref, h0=1., Neff=Neff_ref
):

    r"""
    Compute the GW spectrum for sound waves, normalized to radiation
    energy density.

    The GW spectrum is calculated as:

    .. math ::
        \Omega_{\rm GW} (f) = 3 \times {\rm ampl\_GWB} \times
        {\rm pref\_GWB} \times S(f)

    where ampl_GWB is the efficiency, pref_GWB is the amplitude prefactor,
    and :math:`S(f)` is the normalized spectral shape.

    The GW spectrum as an observable at present time is then computed using

    .. math ::

        \Omega_{\rm GW}^0 (f) = \Omega_{\rm GW} (f) \times F_{\rm GW, 0},

    where :math:`F_{\rm GW, 0}` is the redshift from the time of
    generation to present time, computed in GW_back.py that depends
    on the degrees of freedom at the time of generation.

    Parameters
    ----------
    s : array_like
        Normalized wave number, divided by the mean bubble size Rstar,
        :math:`s = f R_\ast`.
    alphas : array_like
        Strength of the phase transition, :math:`\alpha`.
    betas : array_like
        Rate of nucleation of the phase transition, :math:`\beta/H_\ast`.
    vws : array_like, optional
        Array of wall velocities, :math:`v_w`.
    cs2 : float, optional
        Square of the speed of sound (default 1/3).
    quiet : bool, optional
        Suppress debugging output.
    a_sw, b_sw, c_sw : float, optional
        Slopes of the sound wave template.
    alp1_sw, alp2_sw : float, optional
        Transition parameters of the sound wave template.
    corrRs : bool, optional
        Correct Rstar beta with max(vw, cs).
    expansion : bool, optional
        Include Universe expansion.
    Nsh : float, optional
        Number of shock formation times for source duration.
    model_efficiency : str, optional
        Model for GW efficiency ('fixed_value' or 'higgsless').
    OmGW_tilde : float, optional
        Value of GW efficiency, :math:`\tilde \Omega_{\rm GW}`
        for fixed_value model.
    bs_HL_eff : int, optional
        Box size for Higgsless simulation interpolation.
    model_K0 : str, optional
        Model for kinetic energy ratio K0 ('Espinosa' or 'higgsless').
    bs_k1HL : int, optional
        Box size for Higgsless simulation interpolation for k1.
    model_decay : str, optional
        Model for GW amplitude prefactor ('sound_waves' or 'decay').
    interpolate_HL_decay : bool, optional
        Use Higgsless simulations to interpolate decay law.
    b : float, optional
        Decay law exponent for kinetic energy density.
    model_shape : str, optional
        Model for spectral shape.
    strength : str, optional
        Strength of the phase transition for sw_HLnew.
    interpolate_HL_shape : bool, optional
        Use Higgsless simulations to interpolate k1 and k2.
    interpolate_HL_n3 : bool, optional
        Use Higgsless simulations to interpolate c_sw.
    redshift : bool, optional
        Redshift the GW spectrum and frequencies to present time.
    gstar, gS : int, optional
        Degrees of freedom for redshift calculation.
    T : astropy.units.Quantity
        Temperature scale for redshift calculation.
    h0 : float, optional
        Hubble rate at present time.
    Neff : int, optional
        Effective number of neutrino species.

    Returns
    -------
    freqs : ndarray
        Array of frequencies (normalized or in Hz).
    OmGW : ndarray
        GW spectrum normalized to radiation energy density
        (or present critical density).

    References
    ----------
    RoperPol:2023bqa, Caprini:2024gyk, Hindmarsh:2019phv
    """

    cs = np.sqrt(cs2)
    alphas, betas, vws, mult_alpha, mult_beta, mult_vws = \
        _prepare_inputs(alphas, betas, vws)

    # Computing ampl_GWB
    if model_efficiency == 'higgsless' and not quiet:
        print('Computing the OmGW efficiency')
        _data_warning(boxsize=bs_HL_eff)
    ampl = ampl_GWB_sw(
        model=model_efficiency, OmGW_sw=OmGW_tilde,
        vws=vws, alphas=alphas, bs_HL=bs_HL_eff, quiet=True
    )

    # Computing pref_GWB
    if not quiet:
        print('Computing the kinetic energy density using the model',
              model_K0)
        if model_K0 == 'higgsless':
            _data_warning(boxsize=bs_HL_eff)

    # Kinetic energy density K = rho_kin/rho_total = kappa alpha/(1 + alpha),
    # where kappa is the efficiency in converting vacuum to kinetic energy.
    # Oms_sw = v_f^2 = kappa alpha/(1 + cs2)
    _, Oms_sw = _compute_kinetic_energy(
        model_K0, vws, alphas, cs2, bs_HL_eff=bs_HL_eff, quiet=quiet
    )

    # Decay rate
    interpol_b = False
    if interpolate_HL_decay and model_decay == 'decay':
        b, interpol_b = _compute_decay_b(vws, alphas, bs_HL_eff, quiet)

    # prefactor GWB of sound waves
    pref = _compute_prefactor(
        vws, alphas, betas, Oms_sw, model_decay=model_decay, Nsh=Nsh, b=b,
        expansion=expansion, cs2=cs2, corrRs=corrRs, interpol_b=interpol_b
    )

    # Computing the spectral shape and sound-shell thickness
    S = _compute_spectral_shape(
        s, vws, alphas, cs, a_sw, b_sw, c_sw, alp1_sw, alp2_sw,
        model_shape, strength, interpolate_HL_shape, bs_k1HL, bs_HL_eff,
        interpolate_HL_n3, corrRs, cs2, quiet
    )

    freqs, OmGW = _assemble_spectrum(
        s, vws, alphas, betas, ampl, pref, S, cs2, corrRs
    )

    freqs = reshape_output(freqs, mult_a=mult_vws, mult_b=mult_beta, skip=1)
    OmGW = reshape_output(
        OmGW, mult_a=mult_vws, mult_b=mult_alpha,
        mult_c=mult_beta, skip=1
    )

    if redshift:
        freqs, OmGW = GW_back.shift_OmGW_today(
            freqs, OmGW, g=gstar, gS=gS, T=T, h0=h0, kk=False, Neff=Neff
        )

    return freqs, OmGW


def _prepare_inputs(alphas, betas, vws):
    '''
    Prepare the inputs for the GW calculation called
    from :func:`OmGW_spec_sw` and :func:`OmGW_spec_turb_alphabeta`.
    '''
    mult_alpha = isinstance(alphas, (list, tuple, np.ndarray))
    mult_beta = isinstance(betas, (list, tuple, np.ndarray))
    mult_vws = isinstance(vws, (list, tuple, np.ndarray))
    if not mult_alpha:
        alphas = np.array([alphas])
    if not mult_vws:
        vws = np.array([vws])
    if not mult_beta:
        betas = np.array([betas])
    return alphas, betas, vws, mult_alpha, mult_beta, mult_vws


def _compute_kinetic_energy(model_K0, vws, alphas, cs2,
                            bs_HL_eff=bs_HL_eff, quiet=True):
    '''
    Compute the kinetic energy density for the given model and parameters,
    called from :func:`OmGW_spec_sw` and :func:`OmGW_spec_turb_alphabeta`.
    '''
    # compute kappa, K and Oms following the bag equation of
    # state as in Espinosa:2010hh
    if model_K0 == 'Espinosa':
        kap = hydro_bubbles.kappas_Esp(vws, alphas, cs2=cs2)
        K = kap * alphas / (1 + alphas)
        Oms_sw = kap * alphas / (1 + cs2)
    # compute K and Oms directly from the numerical results of the
    # Higgsless simulations of Caprini:2024gyk and interpolate to
    # values of alpha and vws
    elif model_K0 == 'higgsless':
        dirr = COSMOGW_HOME + 'resources/higgsless/parameters_fit_sims.csv'
        df = pd.read_csv(dirr)
        val_str = 'curly_K_0_512'
        K = interpolate_HL_vals(
            df, vws, alphas, quiet=quiet, value=val_str, boxsize=bs_HL_eff
        )
        kap = K * (1 + alphas) / alphas
        Oms_sw = kap * alphas / (1 + cs2)
    else:
        print('Choose an available model for K0 in OmGW_spec_sw')
        print('Available models are Espinosa and higgsless')
        return 0, 0
    return K, Oms_sw


def _compute_decay_b(vws, alphas, bs_HL_eff, quiet):
    '''
    Compute the decay parameter b when interpolate_HL_decay is True
    and model_decay is 'decay', called from :func:`OmGW_spec_sw`.
    '''
    dirr = COSMOGW_HOME + 'resources/higgsless/parameters_fit_sims.csv'
    df = pd.read_csv(dirr)
    return interpolate_HL_vals(
        df, vws, alphas, quiet=quiet, value='b', boxsize=bs_HL_eff
    ), True


def _compute_prefactor(vws, alphas, betas, Oms_sw, model_decay='sound_waves',
                       Nsh=1., b=0., expansion=True, cs2=cs2_ref,
                       corrRs=True, interpol_b=False):
    '''
    Compute the prefactor for the gravitational wave background,
    called from :func:`OmGW_spec_sw`.
    '''
    pref = np.zeros((len(vws), len(alphas), len(betas)))
    for i in range(len(vws)):
        # Fluid length scale R_star x beta
        lf = hydro_bubbles.Rstar_beta(vws[i], cs2=cs2, corr=corrRs) / betas
        for j, alpha in enumerate(alphas):
            for l in range(len(betas)):
                b_ij = b[i, j] if interpol_b else b
                pref[i, j, l] = pref_GWB_sw(
                    Oms=Oms_sw[i, j], lf=lf[l], alpha=alpha,
                    model=model_decay, Nshock=Nsh, b=b_ij,
                    expansion=expansion, beta=betas[l], cs2=cs2
                )
    return pref


def _compute_spectral_shape(
    s, vws, alphas, cs, a_sw, b_sw, c_sw, alp1_sw, alp2_sw,
    model_shape, strength, interpolate_HL_shape, bs_k1HL, bs_HL_eff,
    interpolate_HL_n3, corrRs, cs2, quiet
):

    '''
    Compute the spectral shape for the given model and parameters,
    called from :func:`OmGW_spec_sw`.
    '''

    if not quiet:
        print('Computing spectral shape using model ', model_shape)

    if model_shape == ['sw_LISAold']:
        return _spectral_shape_LISAold(s, vws, alphas)

    if model_shape in ['sw_HL', 'sw_SSM']:
        return _spectral_shape_HL_SSM(
            s, vws, alphas, cs, a_sw, b_sw, c_sw, alp1_sw, alp2_sw,
            model_shape
        )

    if model_shape in ['sw_LISA', 'sw_HLnew']:
        return _spectral_shape_LISA_HLnew(
            s, vws, alphas, cs, a_sw, b_sw, c_sw, alp1_sw, alp2_sw,
            strength, interpolate_HL_shape, bs_k1HL, bs_HL_eff,
            interpolate_HL_n3, corrRs, cs2, quiet, model_shape
        )

    print('Choose an available model for model_shape in OmGW_spec_sw')
    print('Available models are sw_LISA, sw_HL, sw_HLnew, sw_SSM, sw_LISAold')
    return 0


def _spectral_shape_LISAold(s, vws, alphas):
    S = Sf_shape_sw(s, model=['sw_LISAold'])
    mu = safe_trapezoid(S, np.log(s))
    S, _, _ = np.meshgrid(S, vws, alphas, indexing='ij')
    return S / mu


def _spectral_shape_HL_SSM(s, vws, alphas, cs, a_sw, b_sw, c_sw,
                           alp1_sw, alp2_sw, model_shape):
    Dw = abs(vws - cs) / vws
    S = Sf_shape_sw(
        s, model=model_shape, Dw=Dw, a_sw=a_sw, b_sw=b_sw, c_sw=c_sw,
        alp1_sw=alp1_sw, alp2_sw=alp2_sw
    )
    mu = safe_trapezoid(S, np.log(s), axis=0)
    S = S / mu
    S0 = np.zeros((len(s), len(vws), len(alphas)))
    for i in range(len(alphas)):
        S0[:, :, i] = S
    return S0


def _spectral_shape_LISA_HLnew(
    s, vws, alphas, cs, a_sw, b_sw, c_sw, alp1_sw, alp2_sw,
    strength, interpolate_HL_shape, bs_k1HL, bs_HL_eff,
    interpolate_HL_n3, corrRs, cs2, quiet, model_shape
):
    Dw_2d = True
    if model_shape == 'sw_HLnew':
        if strength != 'weak' or interpolate_HL_shape:
            Dw_2d = False
    if Dw_2d:
        if not quiet:
            print('Computing sound-shell thickness')
        _, _, _, _, _, _, xi_shocks, _ = \
            hydro_bubbles.compute_profiles_vws_multalp(
                    alphas, vws=vws
            )
        Dw = np.zeros((len(vws), len(alphas)))
        for i in range(len(alphas)):
            vw_max = np.maximum(vws, cs)
            vw_min = np.minimum(vws, cs)
            Dw[:, i] = (xi_shocks[:, i] - vw_min) / vw_max
    else:
        Dw = 0.
    S = Sf_shape_sw(
        s, model=model_shape, Dw=Dw, a_sw=a_sw, b_sw=b_sw, c_sw=c_sw,
        alp1_sw=alp1_sw, alp2_sw=alp2_sw, strength=strength,
        interpolate_HL=interpolate_HL_shape,
        bs_k1HL=bs_k1HL, bs_k2HL=bs_HL_eff, vws=vws, alphas=alphas,
        quiet=quiet, interpolate_HL_n3=interpolate_HL_n3,
        corrRs=corrRs, cs2=cs2
    )
    mu = safe_trapezoid(S, np.log(s), axis=0)
    if not interpolate_HL_shape:
        if strength != 'weak':
            S, _, _ = np.meshgrid(S, vws, alphas, indexing='ij')
    return S / mu


def _assemble_spectrum(s, vws, alphas, betas, ampl, pref, S, cs2, corrRs):
    OmGW = np.zeros((len(s), len(vws), len(alphas), len(betas)))
    freqs = np.zeros((len(s), len(vws), len(betas)))
    for i, vw in enumerate(vws):
        lf = hydro_bubbles.Rstar_beta(vw, cs2=cs2, corr=corrRs) / betas
        for l in range(len(betas)):
            freqs[:, i, l] = s / lf[l]
            for j in range(len(alphas)):
                OmGW[:, i, j, l] = 3 * ampl[i, j] * pref[i, j, l] * S[:, i, j]
    return freqs, OmGW


# TURBULENCE
# fit for the anisotropic stresses
def pPi_fit(s, b=b_ref, alpPi=alpPi, fPi=fPi, bPi=bPi_vort):

    r"""
    Fit the spectrum of the anisotropic stresses for turbulence.

    The spectrum can be computed numerically for a Gaussian source using
    EPi_correlators in GW_models. Default values are valid for a purely
    vortical velocity or magnetic field following a von Kárman spectrum,
    as indicated in RoperPol:2023bqa, equation 17.

    The fit is:

    .. math::

        p_\Pi = \left[1 + \left(\frac{f}{f_\Pi}\right)^{\alpha_\Pi}
        \right]^{-\frac{(b + b_\Pi)}{\alpha_\Pi}}.

    Parameters
    ----------
    s : array_like
        Array of frequencies, normalized by the characteristic scale,
        :math:`s = f R_\ast`.
    b : float, optional
        High-frequency slope :math:`f^{-b}`.
    alpPi : float, optional
        Smoothness parameter of the fit, :math:`\alpha_\Pi`.
    fPi : float, optional
        Position of the fit break, :math:`f_\Pi`.
    bPi : float, optional
        Extra power law decay of the spectrum of the stresses
        compared to b, :math:`b_\Pi`.

    Returns
    -------
    Pi : ndarray
        Spectrum of the anisotropic stresses, :math:`p_\Pi(f)`.
    fGW : float
        Frequency where :math:`s p_\Pi(s)`,
        which determines amplitude of GW spectrum
        for MHD turbulence, is maximum, :math:`f_{\rm GW}`.
    pimax : float
        Maximum value of :math:`p_\Pi`, i.e., at :math:`s = f_{\rm GW}`.

    References
    ----------
    RoperPol:2023bqa, RoperPol:2025b
    """

    Pi = GW_analytical.smoothed_bPL(
        s, a=0, b=b + bPi, kpeak=fPi, alp=alpPi,
        norm=False, alpha2=True, dlogk=False
    )
    pimax = ((b + bPi) / (b + bPi - 1)) ** (-(b + bPi) / alpPi)
    fGW = fPi / (b + bPi - 1) ** (1 / alpPi)
    return Pi, fGW, pimax


def ampl_GWB_turb(a_turb=a_ref, b_turb=b_ref, alp=alp_turb):

    """
    Compute the GW efficiency parameter for turbulence.

    Parameters
    ----------
    a_turb : float, optional
        Low-frequency slope of the turbulent source spectrum.
    b_turb : float, optional
        High-frequency slope of the turbulent source spectrum.
    alp : float, optional
        Smoothness parameter.

    Returns
    -------
    ampl : float
        GW efficiency parameter for turbulence.

    References
    ----------
    RoperPol:2023bqa, equation 9, based on the template of
    RoperPol:2022iel, section 3 D.

    See footnote 3 of RoperPol:2023bqa
    for clarification (extra factor 1/2 for oscillation average).
    """

    A = GW_analytical.calA(
        a=a_turb - 1, b=b_turb + 1, alp=alp, dlogk=False
    )
    C = GW_analytical.calC(
        a=a_turb - 1, b=b_turb + 1, alp=alp, tp='vort', dlogk=False
    )
    ampl = 0.5 * C / A ** 2
    return ampl


def pref_GWB_turb(Oms=Oms_ref, lf=lf_ref):

    """
    Compute the prefactor for the GW spectrum from turbulence.

    The GW spectrum depends on the turbulence length scale (lf) and
    the fraction of turbulent to radiation energy density (Oms).

    Parameters
    ----------
    Oms : float or array_like, optional
        Energy density of the source (i.e., 1/2 vrms^2).
    lf : float or array_like, optional
        Mean-size of the bubbles, as a fraction of the Hubble radius.

    Returns
    -------
    pref : float or ndarray
        Prefactor for the GW spectrum from turbulence.

    References
    ----------
    RoperPol:2023bqa, equation 9, based on RoperPol:2022iel,
    section II D.

    Also used in EPTA:2023xxk, Caprini:2024hue.
    """

    mult_Oms = isinstance(Oms, (list, tuple, np.ndarray))
    mult_lfs = isinstance(lf, (list, tuple, np.ndarray))
    Oms, lf = np.meshgrid(Oms, lf, indexing='ij')
    pref = (Oms * lf) ** 2
    pref = reshape_output(pref, mult_a=mult_Oms, mult_b=mult_lfs)
    return pref


def Sf_shape_turb(s, Oms=Oms_ref, lf=lf_ref,
                  N=N_turb, cs2=cs2_ref,
                  expansion=True, tdecay=tdecay_ref, tp='magnetic',
                  b_turb=b_ref, alpPi=alpPi, fPi=fPi, bPi=bPi_vort):

    r"""
    Compute the spectral shape for GWs generated by MHD turbulence.

    Parameters
    ----------
    s : array_like
        Normalized wave number, divided by the mean bubble size,
        :math:`s = f R_\ast`.
    Oms : float or array_like, optional
        Energy density of the source, :math:`\Omega_\ast = {1 \over 2}
        v_{\rm rms}^2)`.
    lf : float or array_like, optional
        Characteristic scale of the turbulence as a fraction of the
        Hubble radius, :math:`l_\ast H_\ast`.
    N : int, optional
        Ratio between decay time and effective source duration.
    cs2 : float, optional
        Square of the speed of sound (default 1/3).
    expansion : bool, optional
        Include Universe expansion.
    tdecay : str, optional
        Determines finite duration in the cit model.
    tp : str, optional
        Type of source ('magnetic', 'kinetic', or 'max').
    b_turb : float, optional
        Slope of the velocity/magnetic field spectrum in the UV.
    alpPi, fPi, bPi : float, optional
        Parameters for the anisotropic stress fit.

    Returns
    -------
    S : ndarray
        Spectral shape for turbulence.

    References
    ----------
    RoperPol:2023bqa, equation 9, based on RoperPol:2022iel,
    section II D. Also used in EPTA:2023xxk, Caprini:2024hue.

    Also used in EPTA:2023xxk, Caprini:2024hue.

    See further details in RoperPol:2025b.
    """

    mult_Oms = isinstance(Oms, (list, tuple, np.ndarray))
    mult_lf = isinstance(lf, (list, tuple, np.ndarray))
    if not mult_Oms:
        Oms = [Oms]
    if not mult_lf:
        lf = [lf]
    TGW = GW_models.TGW_func(
        s, Oms=Oms, lf=lf, N=N, cs2=cs2, expansion=expansion,
        tdecay=tdecay, tp=tp
    )
    Pi, _, _ = pPi_fit(s, b=b_turb, alpPi=alpPi, fPi=fPi, bPi=bPi)
    s3Pi = s ** 3 * Pi
    s3Pi, Oms, lf = np.meshgrid(s3Pi, Oms, lf, indexing='ij')
    S = s3Pi / lf ** 2 * TGW
    S = reshape_output(S, mult_a=mult_Oms, mult_b=mult_lf, skip=1)
    return S


def OmGW_spec_turb(s, Oms, lfs, N=N_turb, cs2=cs2_ref, a_turb=a_ref,
                   b_turb=b_ref, alp=alp_turb, expansion=True,
                   tdecay=tdecay_ref, tp='magnetic', alpPi=alpPi, fPi=fPi,
                   bPi=bPi_vort, redshift=False, gstar=gref, gS=0, T=Tref,
                   h0=1., Neff=Neff_ref):

    r'''
    Compute the GW spectrum for turbulence, normalized to radiation
    energy density.
    The general shape of the GW spectrum is based on that of reference

    .. math ::
        \Omega_{\rm GW} (f) = 3 * ampl_GWB * pref_GWB * S(f),

    where:

    - ampl_GWB is the efficiency of GW production by the specific source,
    - pref_GWB is the dependence of the GW amplitude on the source
      parameters (e.g. length scale and strength of the source),
    - S(f) is a normalized spectral shape, such that
      :math:`\int S(f) d \ln f = 1`.

    The GW spectrum as an observable at present time is then computed using

    .. math ::

        \Omega_{\rm GW0} (f) = \Omega_{\rm GW} x F_{\rm GW, 0},

    where :math:`F_{\rm GW, 0}` is the redshift from the time of generation
    to present time, computed in GW_back.py that depends on the degrees
    of freedom at the time of generation.

    Parameters
    ----------
    s : array_like
        Normalized wave number, divided by the mean bubble size :math:`R_\ast`,
        :math:`s = f R_\ast`.
    Oms : float or array_like, optional
        Energy density of the source, :math:`\Omega_\ast = {1 \over 2}
        v_{\rm rms}^2)`.
    lfs : float or array_like, optional
          Characteristic scale of the turbulence as a fraction of the
          Hubble radius, :math:`l_\ast H_\ast`.
    N : int, optional
        Ratio between decay time and effective source duration.
    cs2 : float, optional
        Square of the speed of sound (default 1/3).
    quiet : bool, optional
        Suppress debugging output.
    a_turb, b_turb, alp : float, optional
        Slopes and smoothness of the turbulent source spectrum.
    expansion : bool, optional
        Include Universe expansion.
    tdecay : str, optional
        Determines finite duration in the cit model.
    tp : str, optional
        Type of source ('magnetic', 'kinetic', or 'max').
    alpPi, fPi, bPi : float, optional
        Parameters for the anisotropic stress fit.
    redshift : bool, optional
        Redshift the GW spectrum and frequencies to present time.
    gstar, gS : int, optional
        Degrees of freedom for redshift calculation.
    T : astropy.units.Quantity
        Temperature scale for redshift calculation.
    h0 : float, optional
        Hubble rate at present time.
    Neff : int, optional
        Effective number of neutrino species.

    Returns
    -------
    freqs : ndarray
        Array of frequencies (normalized or in Hz).
    OmGW : ndarray
        GW spectrum normalized to radiation energy density
        (or present critical density).

    Reference
    ---------
    RoperPol:2023bqa, equations 3 and 9.
    '''

    mult_Oms = isinstance(Oms, (list, tuple, np.ndarray))
    mult_lfs = isinstance(lfs, (list, tuple, np.ndarray))
    if not mult_Oms:
        Oms = np.array([Oms])
    if not mult_lfs:
        lfs = np.array([lfs])

    # Computing ampl_GWB
    ampl = ampl_GWB_turb(a_turb=a_turb, b_turb=b_turb, alp=alp)

    # Computing pref_GWB
    pref = pref_GWB_turb(Oms=Oms, lf=lfs)

    # Computing the spectral shape
    S = Sf_shape_turb(
        s, Oms=Oms, lf=lfs, N=N, cs2=cs2, expansion=expansion,
        tdecay=tdecay, tp=tp, b_turb=b_turb, alpPi=alpPi,
        fPi=fPi, bPi=bPi
    )

    OmGW = np.zeros((len(s), len(Oms), len(lfs)))
    freqs = np.zeros((len(s), len(lfs)))

    for l in range(len(lfs)):
        freqs[:, l] = s / lfs[l]
        for j in range(len(Oms)):
            OmGW[:, j, l] = 3 * ampl * pref[j, l] * S[:, j, l]

    if redshift:
        freqs, OmGW = GW_back.shift_OmGW_today(
            freqs, OmGW, g=gstar, gS=gS, T=T, h0=h0, kk=False, Neff=Neff
        )
    freqs = reshape_output(freqs, mult_a=mult_lfs, skip=1)
    OmGW = reshape_output(OmGW, mult_a=mult_Oms, mult_b=mult_lfs, skip=1)

    return freqs, OmGW


def OmGW_spec_turb_alphabeta(
    s, alphas, betas, vws=1., eps_turb=1., model_K0='Espinosa',
    bs_HL_eff=bs_HL_eff, N=N_turb, cs2=cs2_ref,
    corrRs=True, quiet=True, a_turb=a_ref, b_turb=b_ref, alp=alp_turb,
    expansion=True, tdecay=tdecay_ref, tp='both', alpPi=alpPi,
    fPi=fPi, bPi=bPi_vort, redshift=False, gstar=gref, gS=0,
    T=Tref, h0=1., Neff=Neff_ref
):

    r"""
    Compute the GW spectrum for turbulence using :math:`\alpha` and
    :math:`\beta` (i.e., the parameters of the
    phase transition), following the description of RoperPol:2023bqa.

    It is assumed that turbulence has two contributions (from velocity
    and magnetic fields), which are in equipartition,

    .. math ::
        \Omega_\ast = \Omega_v + \Omega_B \rightarrow
        \Omega_v = \Omega_B = 0.5 \Omega_\ast.

    It takes the kinetic energy density from the PT,
    :math:`K = \rho_{\rm kin}/\rho_{\rm total}`, and assumes that the
    turbulence energy density is a fraction
    :math:`\epsilon_{\rm turb}` of it,

    .. math ::
        \Omega_\ast = \epsilon_{\rm turb} K.

    The duration of the GW production :math:`\delta t_{\rm fin}`
    is assumed to be

    .. math ::
        \delta t_{\rm fin} = N_{\rm turb} \frac{l_\ast}{u_\ast},

    where :math:`u_\ast` is a characteristic velocity,

    .. math ::
        u_\ast = \sqrt{\max(\Omega_v, \frac{2}{1 + c_s^2} \Omega_B)}.

    Parameters
    ----------
    s : array_like
        Normalized wave number, divided by the mean bubble size
        :math:`R_\ast`, :math:`s = f R_\ast`.
    alphas : array_like
        Strength of the phase transition :math:`\alpha`.
    betas : array_like
        Rate of nucleation of the phase transition
        :math:`\beta/H_\ast`.
    vws : array_like, optional
        Array of wall velocities.
    eps_turb : float, optional
        Fraction of kinetic energy converted to turbulence.
    model_K0 : str, optional
        Model for kinetic energy ratio K0 ('Espinosa' or 'higgsless').
    bs_HL_eff : int, optional
        Box size for Higgsless simulation interpolation.
    N : int, optional
        Ratio between decay time and effective source duration.
    cs2 : float, optional
        Square of the speed of sound.
    corrRs : bool, optional
        Correct Rstar beta with max(vw, cs).
    quiet : bool, optional
        Suppress debugging output.
    a_turb, b_turb, alp : float, optional
        Slopes and smoothness of the turbulent source spectrum.
    expansion : bool, optional
        Include Universe expansion.
    tdecay : str, optional
        Determines finite duration in the cit model.
    tp : str, optional
        Type of source ('both', 'magnetic', 'kinetic').
    alpPi, fPi, bPi : float, optional
        Parameters for the anisotropic stress fit.
    redshift : bool, optional
        Redshift the GW spectrum and frequencies to present time.
    gstar, gS : int, optional
        Degrees of freedom for redshift calculation.
    T : astropy.units.Quantity
        Temperature scale for redshift calculation.
    h0 : float, optional
        Hubble rate at present time.
    Neff : int, optional
        Effective number of neutrino species.

    Returns
    -------
    freqs : ndarray
        Array of frequencies (normalized or in Hz).
    OmGW : ndarray
        GW spectrum normalized to radiation energy density
        (or present critical density).
    """

    alphas, betas, vws, mult_alpha, mult_beta, mult_vws = \
        _prepare_inputs(alphas, betas, vws)

    K, _ = _compute_kinetic_energy(
        model_K0, vws, alphas, cs2,
        bs_HL_eff=bs_HL_eff, quiet=quiet
    )

    # amplitude used to determine the duration of the source,
    # assuming equipartition
    Oms = 0.5 * K * eps_turb
    # length scale
    lf = hydro_bubbles.Rstar_beta(vws=vws, corr=corrRs, cs2=cs2)

    OmGW = np.zeros((len(s), len(vws), len(alphas), len(betas)))
    freqs = np.zeros((len(s), len(vws), len(betas)))

    for i in range(len(vws)):
        for l, beta in enumerate(betas):
            lf_ij = lf[i] / beta
            for j in range(len(alphas)):
                freqs[:, i, l], OmGW[:, i, j, l] = OmGW_spec_turb(
                    s, Oms[i, j], lf_ij, N=N, cs2=cs2,
                    a_turb=a_turb, b_turb=b_turb, alp=alp,
                    expansion=expansion, tdecay=tdecay, tp=tp,
                    alpPi=alpPi, fPi=fPi, bPi=bPi, redshift=False
                )

    freqs = reshape_output(freqs, mult_a=mult_vws, mult_b=mult_beta, skip=1)
    OmGW = reshape_output(
        OmGW, mult_a=mult_vws, mult_b=mult_alpha,
        mult_c=mult_beta, skip=1
    )

    if redshift:
        freqs, OmGW = GW_back.shift_OmGW_today(
            freqs, OmGW, g=gstar, gS=gS, T=T, h0=h0, kk=False, Neff=Neff
        )

    # multiply by 4 to take into account contribution from both velocity and
    # magnetic fields in the amplitude OmGW ~ Oms^2 = (2 * OmB)^2
    return freqs, 4.0 * OmGW
