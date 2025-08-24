r"""
analysis.py is a Python routine that contains functions used in the analyses
of potential detection of GW signals by space-based GW detectors
(LISA or Taiji).

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/analysis.py

.. note::
   For full documentation, visit `Read the Docs
   <https://cosmogw-manual.readthedocs.io/en/latest/analysis.html>`_.

To use it, first install `cosmoGW <https://pypi.org/project/cosmoGW>`_::

    pip install cosmoGW

Author
------
- **Alberto Roper Pol**
  (`alberto.roperpol@unige.ch <mailto:alberto.roperpol@unige.ch>`_)

Dates
-----
- Created: **01/12/2021**
- Updated: **21/08/2025**
  (release **cosmoGW 1.0**: https://pypi.org/project/cosmoGW)

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
"""

from cosmoGW.utils import (
    cs2_ref, a_sw_ref, b_sw_ref, c_sw_ref, OmGW_sw_ref,
    bs_HL_eff, bs_k1HL, N_turb, a_ref, b_ref, alp_turb, tdecay_ref,
    alpPi, fPi, bPi_vort, Neff_ref, T_PLS,
    check_temperature_MeV, reshape_output
)

import numpy as np
import astropy.units as u
from cosmoGW import GW_templates, GW_back, interferometry, cosmology


def analysis_LISA_alphabeta(
    s, alphas, betas, vws, TTs, eps_turb=1., turb=True,
    cs2=cs2_ref, quiet=True, a_sw=a_sw_ref, b_sw=b_sw_ref, c_sw=c_sw_ref,
    alp1_sw=0, alp2_sw=0, corrRs=True, expansion=True, Nsh=1.,
    model_efficiency='fixed_value', OmGW_tilde=OmGW_sw_ref,
    bs_HL_eff=bs_HL_eff, model_K0='Espinosa', bs_k1HL=bs_k1HL,
    model_decay='sound_waves', interpolate_HL_decay=True, b=0,
    model_shape='sw_LISA', strength='weak', interpolate_HL_shape=False,
    interpolate_HL_n3=False, N_turb=N_turb, a_turb=a_ref, b_turb=b_ref,
    alp_turb=alp_turb, tdecay=tdecay_ref, tp='both', alpPi_turb=alpPi,
    fPi_turb=fPi, bPi_turb=bPi_vort, h0=1., Neff=Neff_ref, interf='LISA',
    Tobs=T_PLS
):
    r"""
    Compute the signal-to-noise ratio (SNR) for gravitational wave backgrounds
    from sound waves and turbulence in a first-order cosmological phase
    transition, for a range of phase transition and detector parameters,
    comparing the sensitivity of LISA with 4 years of observations
    (and no foregrounds).

    This function calculates the GW spectrum for sound waves and turbulence,
    redshifts it to today, and computes the SNR for a given detector
    sensitivity.
    It supports vectorized input for all physical parameters, automatically
    adapting the output shape to match scalar or array inputs.

    Parameters
    ----------
    s : array_like
        Array of normalized wave numbers, :math:`s = f R_\ast`.
    alphas : float or array_like
        Strength of the phase transition :math:`\alpha` (latent heat fraction).
    betas : float or array_like
        Nucleation rate of the phase transition :math:`\beta/H_\ast`.
    vws : float or array_like
        Bubble wall velocity :math:`v_w`.
    TTs : float, array_like, or astropy.units.Quantity
        Temperature(s) of the phase transition (can be MeV, GeV, etc).
    eps_turb : float or array_like, optional
        Fraction of kinetic energy converted to turbulence (default: 1).
    turb : bool, optional
        If True, include turbulence contribution (default: True).
    cs2 : float, optional
        Square of the speed of sound (default: 1/3).
    quiet : bool, optional
        Suppress verbose output (default: True).
    a_sw, b_sw, c_sw : float, optional
        Spectral slopes for spectral shape of GWs from sound waves
        (default 3, 1, 3).
    alp1_sw, alp2_sw : float, optional
        Additional shape parameters for sound waves
        (default are the reference values for model_shape=`sw_LISA`, see
        GW_templates.OmGW_spec_sw).
    corrRs : bool, optional
        Correct Rstar beta with :math:`\max(v_w, c_s)` (default: True).
    expansion : bool, optional
        Include Universe expansion (default: True).
    Nsh : float, optional
        Duration of the sound waves sourcing in number of shock
        formation times (default: 1).
    model_efficiency : str, optional
        Model for GW production efficiency (default: 'fixed_value',
        other option is 'higgsless').
    OmGW_tilde : float, optional
        Reference amplitude for GW spectrum (default: 1e-2).
    bs_HL_eff, bs_k1HL : int, optional
        Box size parameters for Higgsless simulation interpolation
        for UV and IR quantities (default: 20, 40), used if
        model_efficiency='higgsless', model_decay='decay' with
        interpolate_HL_decay True, or model_shape='sw_HL_new
        with interpolate_HL_shape True.
    model_K0 : str, optional
        Model for kinetic energy ratio ('Espinosa' or 'higgsless').
    model_decay : str, optional
        Model for decay of the source ('sound_waves', 'decay').
    interpolate_HL_decay : bool, optional
        Interpolate decay parameter from HL simulations if model_decay='decay'
        (default: True).
    b : float, optional
        Decay parameter (default: 0).
    model_shape : str, optional
        Spectral shape model for sound waves (default: 'sw_LISA').
    strength : str, optional
        Strength regime for HL shape interpolation (default: 'weak', used when
        model_shape='sw_HL_new' and interpolate_HL_shape False).
    interpolate_HL_shape : bool, optional
        Interpolate HL shape (default: False, used when
        model_shape='sw_HL_new').
    interpolate_HL_n3 : bool, optional
        Interpolate HL n3 slope parameter (default: False, used when
        model_shape='sw_HL_new').
    N_turb : int, optional
        Ratio between decay time and effective source duration for turbulence
        production of GWs.
    a_turb, b_turb, alp_turb : float, optional
        Spectral fit parameters (slopes and smoothness) for turbulence
        (default 5, 2/3, 2).
    tdecay : str, optional
        Determines finite duration in the turbulence model
        (default 'eddy').
    tp : str, optional
        Type of turbulence source ('both', 'magnetic', 'kinetic').
    alpPi_turb, fPi_turb, bPi_turb : float, optional
        Parameters for the anisotropic stress fit for turbulence.
    h0 : float, optional
        Hubble rate at present time (default 1), such that 
        :math:`H_0 = 100` km/s/Mpc.
    Neff : int, optional
        Effective number of neutrino species (default: 3).
    interf : str, optional
        Detector/interferometer name (default: 'LISA').
    Tobs : float, optional
        Observation time in years (default: 4 years).

    Returns
    -------
    SNR : ndarray
        Signal-to-noise ratio for each parameter combination. The output shape
        adapts to the input: axes corresponding to scalar inputs are squeezed.
        For example, if all inputs are arrays, shape is
        (len(eps_turb), len(TTs), len(vws), len(alphas), len(betas)).
        If some inputs are scalars, those axes are removed.

    References
    ----------
    RoperPol:2022iel, RoperPol:2023bqa
    """

    mult_alpha = isinstance(alphas, (list, tuple, np.ndarray))
    mult_beta = isinstance(betas, (list, tuple, np.ndarray))
    mult_vws = isinstance(vws, (list, tuple, np.ndarray))

    if isinstance(TTs, u.Quantity):
        TTs = check_temperature_MeV(
            TTs, func='analysis.analysis_LISA_alphabeta'
        )
        TTs = TTs.value
    mult_TTs = isinstance(TTs, (list, tuple, np.ndarray))

    if not mult_alpha:
        alphas = np.array([alphas])
    if not mult_vws:
        vws = np.array([vws])
    if not mult_beta:
        betas = np.array([betas])
    if not mult_TTs:
        TTs = np.array([TTs])

    g = cosmology.thermal_g(T=TTs*u.MeV, s=0, file=True, Neff=Neff)
    gS = cosmology.thermal_g(T=TTs*u.MeV, s=1, file=True, Neff=Neff)

    if not turb:
        eps_turb = [0]

    mult_epss = isinstance(eps_turb, (list, tuple, np.ndarray))
    if not mult_epss:
        eps_turb = np.array([eps_turb])
    shape = (len(eps_turb), len(TTs), len(vws), len(alphas), len(betas))
    SNR = np.zeros(shape)

    f_LISA, OmLISA, _ = interferometry.read_sens(interf=interf)
    freqs, OmGWs_sw = GW_templates.OmGW_spec_sw(
        s, alphas, betas, vws=vws, cs2=cs2, quiet=quiet, a_sw=a_sw, b_sw=b_sw,
        c_sw=c_sw, alp1_sw=alp1_sw, alp2_sw=alp2_sw, corrRs=corrRs,
        expansion=expansion, Nsh=Nsh, model_efficiency=model_efficiency,
        OmGW_tilde=OmGW_tilde, bs_HL_eff=bs_HL_eff, model_K0=model_K0,
        bs_k1HL=bs_k1HL, model_decay=model_decay,
        interpolate_HL_decay=interpolate_HL_decay, b=b, model_shape=model_shape,
        strength=strength, interpolate_HL_shape=interpolate_HL_shape,
        interpolate_HL_n3=interpolate_HL_n3, redshift=False
    )
    OmGWs_turb = np.zeros(
        (len(eps_turb), len(s), len(vws), len(alphas), len(betas))
    )

    for p in range(0, len(eps_turb)):

        if eps_turb[p] > 0:

            print('Computing spectra for eps_turb = %.1f' % eps_turb[p])
            # Spectrum from MHD turbulence based on the constant-in-time model
            # (RoperPol:2023bqa, Jinno:2022mie)
            _, OmGWs_turb[p] = \
                GW_templates.OmGW_spec_turb_alphabeta(
                    s, alphas, betas, vws=vws, eps_turb=eps_turb[p],
                    model_K0=model_K0, bs_HL_eff=bs_HL_eff,
                    N=N_turb, cs2=cs2, corrRs=corrRs, quiet=quiet,
                    a_turb=a_turb, b_turb=b_turb, alp=alp_turb,
                    expansion=expansion, tdecay=tdecay, tp=tp,
                    alpPi=alpPi_turb, fPi=fPi_turb, bPi=bPi_turb,
                    redshift=False
                )

    for i in range(0, len(TTs)):

        print('Redshifting spectra for T = %.e MeV' % TTs[i])
        freqs_red, OmGWs_sw_red = \
            GW_back.shift_OmGW_today(
                freqs, OmGWs_sw, g=g[i], gS=gS[i], T=TTs[i]*u.MeV,
                kk=False, h0=h0, Neff=Neff
            )

        for p in range(0, len(eps_turb)):
            print(
                'Redshifting spectra for eps_turb = %.1f and computing SNR'
                % eps_turb[p]
            )
            _, OmGWs_turb_red = \
                GW_back.shift_OmGW_today(
                    freqs, OmGWs_turb[p], g=g[i], gS=gS[i],
                    kk=False, T=TTs[i]*u.MeV, h0=h0, Neff=Neff
                )
            OmGWs_sum = OmGWs_sw_red + OmGWs_turb_red
            for m in range(0, len(vws)):
                for l in range(0, len(betas)):
                    for j in range(0, len(alphas)):
                        SNR[p, i, m, j, l] = interferometry.SNR(
                            freqs_red[:, m, l], OmGWs_sum[:, m, j, l],
                            f_LISA, OmLISA, T=Tobs
                        )

    SNR = reshape_output(SNR, mult_a=mult_epss, mult_b=mult_TTs,
                         mult_c=mult_vws, mult_d=mult_alpha, mult_e=mult_beta)

    return SNR
