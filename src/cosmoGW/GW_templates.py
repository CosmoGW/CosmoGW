"""
GW_templates.py is a Python routine that contains analytical and semi-analytical
templates of cosmological GW backgrounds, usually based on spectral fits,
either from GW models (see GW_models) or from numerical simulations

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/GW_templates.py

Author: Alberto Roper Pol
Created: 01/12/2022
Updated: 31/08/2024
Updated: 04/06/2025 (release cosmoGW 1.0: https://pypi.org/project/cosmoGW)

Main references are:

Espinosa:2010hh - J. R. Espinosa, T. Konstandin, J. M. No, G. Servant,
"Energy Budget of Cosmological First-order Phase Transitions,"
JCAP 06 (2010) 028, arXiv:1004.4187

Hindmarsh:2017gnf - M. Hindmarsh, S. J. Huber, K. Rummukainen,
D. J. Weir, "Shape of the acoustic gravitational wave
power spectrum from a first order phase transition,"
Phys.Rev.D 96 (2017) 10, 103520, Phys.Rev.D 101 (2020) 8,
089902 (erratum), arXiv:1704.05871

Hindmarsh:2019phv - M. Hindmarsh, M. Hijazi, "Gravitational waves from
first order cosmological phase transitions in the Sound Shell Model,"
JCAP 12 (2019) 062, arXiv:1909.10040

Caprini:2019egz   - [LISA CosWG], "Detecting gravitational
waves from cosmological phase transitions with LISA: an update,"
JCAP 03 (2020) 024, arXiv:1910.13125

Hindmarsh:2020hop - M. Hindmarsh, M. Lueben, J. Lumma,
M. Pauly, "Phase transitions in the early universe,"
SciPost Phys. Lect. Notes 24 (2021), 1, arXiv:2008.09136

Jinno:2022mie     - R. Jinno, T. Konstandin, H. Rubira, I. Stomberg,
"Higgsless simulations of cosmological phase transitions and
gravitational waves," JCAP 02, 011 (2023), arxiv:2209.04369

RoperPol:2022iel  - A. Roper Pol, C. Caprini, A. Neronov,
D. Semikoz, "The gravitational wave signal from primordial
magnetic fields in the Pulsar Timing Array frequency band,"
Phys. Rev. D 105, 123502 (2022), arXiv:2201.05630

RoperPol:2023bqa  - A. Roper Pol, A. Neronov, C. Caprini, T. Boyer,
D. Semikoz, "LISA and γ-ray telescopes as multi-messenger probes of a
first-order cosmological phase transition," arXiv:2307.10744 (2023)

EPTA:2023xxk      - EPTA and InPTA Collaborations, "The second data
release from the European Pulsar Timing Array - IV. Implications
for massive black holes, dark matter, and the early Universe,"
Astron. Astrophys. 685, A94 (2024), arxiv:2306.16227

RoperPol:2023dzg  - A. Roper Pol, S. Procacci, C. Caprini,
"Characterization of the gravitational wave spectrum from sound waves within
the sound shell model," Phys. Rev. D 109, 063531 (2024), arXiv:2308.12943.

Caprini:2024gyk   - A. Roper Pol, I. Stomberg, C. Caprini, R. Jinno,
T. Konstandin, H. Rubira, "Gravitational waves from first-order
phase transitions: from weak to strong," JHEP, arxiv:2409.03651

Caprini:2024hue   - E. Madge, C. Caprini, R. Jinno, M. Lewicki,
M. Merchand, G. Nardini, M. Pieroni, A. Roper Pol, V. Vaskonen,
"Gravitational waves from first-order phase transitions in LISA:
reconstruction pipeline and physics interpretation,"
JCAP 10, 020 (2024), arxiv:2403.03723

RoperPol:2025b    - A. Roper Pol, A. Midiri, M. Salomé, C. Caprini,
"Modeling the gravitational wave spectrum from slowly decaying sources in the
early Universe: constant-in-time and coherent-decay models," in preparation

RoperPol:2025a    - A. Roper Pol, S. Procacci, A. S. Midiri,
C. Caprini, "Irrotational fluid perturbations from first-order phase
transitions," in preparation
"""

import numpy  as np
import pandas as pd
import matplotlib.pyplot     as plt
import cosmoGW.GW_analytical as an
import cosmoGW.GW_models     as mod
import cosmoGW.hydro_bubbles as hb
import importlib

# reference values
cs2_ref    = hb.cs2_ref     # speed of sound squared
Oms_ref    = mod.Oms_ref    # reference source amplitude
                            # (fraction to radiation energy density)
lf_ref     = mod.lf_ref     # reference length scale of the source
                            # (normalized by the Hubble radius)

'''
Reference values for turbulence template
(based on RoperPol:2023bqa and RoperPol:2022iel)
Template used in Caprini:2024hue for LISA and in EPTA:2023xxk

Note that the values used for a_turb, b_turb, bPi_vort, fPi
are found assuming that the spectrum of the source is defined
such that

<v^2> ~ 2 E*    int zeta(K) dlnK,   as in RoperPol:2025b

which is a different normalization than that in previous papers,
where zeta is defined such that

<v^2> ~ 2 E* k* int zeta(K) dK,     in RoperPol:2022iel

Hence, the values are considered for the former zeta (this choice
yields different coefficients. However, the final result is not
affected by this choice. See RoperPol:2025b for details
'''

a_turb   = an.a_ref    # Batchelor spectrum k^5
b_turb   = an.b_ref    # Kolmogorov spectrum k^(-2/3)
alp_turb = 6/17        # von Karman smoothness parameter, see RoperPol:2023bqa
bPi_vort = 3           # spectral slope found in the anisotropic stresses
                       # in the K to infinity limit
alpPi    = 2.15        # smoothness parameter for the anisotropic
                       # stresses obtained for a von Karman spectrum
fPi      = 1.25        # break frequency of the anisotropic stresses
                       # obtained for a von Karman spectrum
N_turb   = mod.N_turb  # ratio between the effective time duration of
                       # the source and the eddy turnover time,
                       # based on the simulations of RoperPol:2022iel,
                       # used in RoperPol:2023bqa

### Reference values for sound waves templates

OmGW_sw_ref = 1e-2    # normalized amplitude, based on the simulations of
                      # Hindmarsh:2017gnf
a_sw_ref    = 3       # low frequency slope f^3 found for GWs in
                      # the HL simulations, see Jinno:2022mie and Caprini:2024gyk,
                      # and the sound-shell model in RoperPol:2023dzg
b_sw_ref    = 1       # intermediate frequency slope f found for GWs in
                      # the HL simulations, see Jinno:2022mie and Caprini:2024gyk
c_sw_ref    = 3       # high frequency slope f^(-3) found for GWs in the
                      # HL simulations, see Jinno:2022mie and Caprini:2024gyk

# first and second peak smoothness parameters
alp1_sw_ref = 1.5; alp2_sw_ref = 0.5  # used in RoperPol:2023bqa
alp1_HL     = 3.6; alp2_HL     = 2.4  # found in Caprini:2024gyk
alp1_LISA   = 2.;  alp2_LISA   = 4.   # used in Caprini:2024hue

####### GWB TEMPLATES FOR SOUND WAVES AND TURBULENCE #######

'''
Main reference: RoperPol:2023bqa

Turbulence template is based on the constant-in-time model
developed in RoperPol:2022iel, extended details in RoperPol:2025b

Sound waves template is based on the sound-shell model presented
in Hindmarsh:2019phv, revised and extended in RoperPol:2023dzg
and RoperPol:2025a

Double broken power law for sound waves uses the fits presented in
RoperPol:2023bqa, adapted from the numerical results of Jinno:2022mie
and Caprini:2024gyk

Templates have been used in RoperPol:2023bqa, Caprini:2024hue,
EPTA:2023xxk

The Higgsless template for decaying compressional sources
is developed and used in Caprini:2024gyk
'''

#################### GW SPECTRUM FOR SOUND WAVES AND TURBULENCE ####################

# def OmGW_spec(ss, alpha, beta, xiw=1., tp='turb', cs2=cs2_ref,
#               multi_ab=False, multi_xi=False, OmGW_tilde=OmGW_sw_ref,
#               a_sw=a_sw, b_sw=b_sw, c_sw=c_sw, alp1_sw=alp1_sw, alp2_sw=alp2_sw,
#               a_turb=a_turb, b_turb=b_turb, alp_turb=alp_turb, alpPi=alpPi, fPi=fPi,
#               bPi=bPi_vort, eps_turb=1., ref='f', corrRs=True,
#               model_shape_sw=''):

#     '''
#     Function that computes the GW spectrum (normalized to radiation
#     energy density within RD era) for sound waves and turbulence.

#     It takes the form:

#         OmGW = 3 * ampl_GWB * pref_GWB * Sf_shape,

#     see ampl_GWB, pref_GWB, and Sf_shape functions for details and references.

#     Arguments:
#         ss -- normalized wave number, divided by the mean bubbles size Rstar, s = f R*
#         alpha -- strength of the phase transition
#         beta -- rate of nucleation of the phase transition
#         xiw -- wall velocity
#         tp -- type of GW source (options are sw_SSM for the sound shell model of Hindmarsh:2019phv,
#                'sw_HL' for the fit based on the Higgsless simulations of Jinno:2022mie, and
#                'turb' for MHD turbulence)
#         cs2 -- square of the speed of sound (default is 1/3 for radiation domination)
#         multi_ab -- option to provide an array of values of alpha and beta as input
#         multi_xi -- option to provide an array of values of xiw as input
#         Omgwtilde_sw -- efficiency of GW production from sound waves (default value is 1e-2,
#                         based on numerical simulations)
#         a_sw, b_sw, c_sw -- slopes for sound wave template, used when tp = 'sw_HL'
#         alp1_sw, alp2_sw -- transition parameters for sound wave template, used when tp = 'sw_HL'

#         a_turb, b_turb, alp_turb -- slopes and smoothness of the turbulent source spectrum
#                                     (either magnetic or kinetic), default values are for a
#                                     von Karman spectrum
#         alpPi, fPi -- parameters of the fit of the spectral anisotropic stresses for turbulence
#         eps_turb -- fraction of energy density converted from sound waves into turbulence
#     '''

#     cs = np.sqrt(cs2)

#     ###### input parameters
#     #
#     # kappa (efficiency in converting vacuum to kinetic energy)
#     # computed using the bag equation of state taken from
#     # Espinosa:2010hh
#     kap = np.zeros((len(alpha), len(xiw)))
#     if isinstance(alpha, (list, tuple, np.ndarray)):
#         for i in range(0, len(alpha)):
#             kap[i] = hb.kappas_Esp(xiw, alpha[i], cs2=cs2)
#     #
#     # K = rho_kin/rho_total = kappa alpha/(1 + alpha)
#     #
#     if isinstance(xiw, (list, tuple, np.ndarray)):
#         alpha, _ = np.meshgrid(alpha, xiw, indexing='ij')
#         beta, vw = np.meshgrid(beta,  xiw, indexing='ij')
#     K = kap*alpha/(1 + alpha)

#     # Oms = rho_source/rho_total = K eps_turb
#     Oms = K*eps_turb

#     if corrRs:
#         lf = (8*np.pi)**(1/3)*np.maximum(xiw, cs)/beta
#     else:
#         lf = (8*np.pi)**(1/3)*xiw/beta

#     Dw = abs(xiw - cs)/xiw

#     # amplitude factors
#     if multi_ab and multi_xi:
#         Oms, lf, xiw_ij = np.meshgrid(alpha, beta, xiw, indexing='ij')
#         for i in range(0, len(beta)): Oms_ij[:, i, :] = Oms
#         for i in range(0, len(alpha)): lf_ij[i, :, :] = lf
#     elif multi_ab and not multi_xi:
#         Oms, lf = np.meshgrid(Oms, lf, indexing='ij')

#     preff = pref_GWB(Oms=Oms, lf=lf, tp=tp, b_turb=b_turb, alpPi=alpPi, fPi=fPi)

#     ampl  = ampl_GWB(tp=tp, cs2=cs2, Omgwtilde_sw=OmGW_sw_ref, a_turb=a_turb,
#                     b_turb=b_turb, alp=alp_turb, alpPi=alpPi, fPi=fPi)

#     # spectral shape for sound waves templates
#     if tp == 'sw':
#         if multi_xi:
#             OmGW_aux = np.zeros((len(ss), len(xiw)))
#             for i in range(0, len(xiw)):
#                 S = Sf_shape(ss, tp=tp, Dw=Dw[i], a_sw=a_sw, b_sw=b_sw, c_sw=c_sw,
#                              alp1_sw=alp1_sw, alp2_sw=alp2_sw)
#                 mu = np.trapz(S, np.log(ss))
#                 OmGW_aux[:, i] = 3*S*ampl/mu

#             if multi_ab:
#                 OmGW = np.zeros((len(ss), len(alpha), len(beta), len(xiw)))
#                 for i in range(0, len(xiw)):
#                     for j in range(0, len(ss)):
#                         OmGW[j, :, :, i] = OmGW_aux[j, i]*preff[:, :, i]
#             else: OmGW = OmGW_aux*preff[i]

#         else:
#             S = Sf_shape(ss, tp=tp, Dw=Dw, a_sw=a_sw, b_sw=b_sw, c_sw=c_sw,
#                          alp1_sw=alp1_sw, alp2_sw=alp2_sw)
#             mu = np.trapz(S, np.log(ss))
#             OmGW_aux = 3*S*ampl/mu
#             if multi_ab:
#                 OmGW = np.zeros((len(ss), len(alpha), len(beta)))
#                 for j in range(0, len(ss)):
#                     OmGW[j, :, :] = OmGW_aux[j]*preff
#             else: OmGW = OmGW_aux*preff

#     # spectral shape for turbulence templates
#     if tp2 == 'turb':
#         if multi_xi:
#             if multi_ab:
#                 OmGW = np.zeros((len(ss), len(xiw), len(alpha), len(beta)))
#                 for i in range(0, len(xiw)):
#                     OmGW[:, i, :, :] = Sf_shape(ss, tp=tp, b_turb=b_turb, N=N_turb,
#                                                 Oms=Oms[:, :, i], lf=lf[:, :, i],
#                                                 alpPi=alpPi, fPi=fPi, ref=ref, cs2=cs2,
#                                                 multi=multi_ab)
#                     for j in range(0, len(ss)):
#                         OmGW[j, i, :, :] = 3*OmGW[j, i, :, :]*ampl*preff[i, :, :]

#             else:
#                 OmGW = np.zeros((len(ss), len(xiw)))
#                 for i in range(0, len(xiw)):
#                     OmGW[:, i] = Sf_shape(ss, tp=tp, b_turb=b_turb, N=N_turb, Oms=Oms[i],
#                                           lf=lf[i], alpPi=alpPi, fPi=fPi,
#                                           ref=ref, cs2=cs2, multi=multi_ab)
#                     OmGW[:, i] = 3*OmGW[:, i]*ampl*preff[i]

#         else:
#             S = Sf_shape(ss, tp=tp, b_turb=b_turb, N=N_turb, Oms=Oms, lf=lf,
#                          alpPi=alpPi, fPi=fPi, ref=ref, cs2=cs2, multi=multi_ab)
#             OmGW = 3*OmGW*ampl*preff

#     return OmGW

############# fit for the anisotropic stresses #############

def pPi_fit(s, b=b_turb, alpPi=alpPi, fPi=fPi, bPi=bPi_vort):

    """
    Function that computes the fit of the spectrum of the
    anisotropic stresses.

    The spectrum can be computed numerically for a Gaussian
    source using EPi_correlators in GW_models module.

    Default values are valid for a purely vortical velocity or
    magnetic field following a von Kárman spectrum, as indicated
    in RoperPol:2023bqa, equation 17.

    Using different values of alpPi, fPi, bPi can be generalized
    to other sources, see RoperPol:2025b

    It assumes that the anisotropic stresses can
    be expressed with the following fit:

    p_Pi = (1 + (f/fPi)^alpPi)^(-(b + bPi)/alpPi)

    Arguments:
        s     -- array of frequencies, normalized by the characteristic scale,
                 s = f R*
        b     -- high-f slope f^(-b)
        alpPi -- smoothness parameter of the fit
        fPi   -- position of the fit break
        bPi   -- extra power law decay of the spectrum of the
                 stresses compared to b

    Returns:
        Pi -- array of the anisotropic stresses spectrum
        fGW -- maximum value of the function s * Pi that determines
               the amplitude of the GW spectrum for MHD turbulence
        pimax -- maximum value of Pi when s = fGW
    """

    #Pi    = (1 + (s/fPi)**alpPi)**(-(b + bPi)/alpPi)
    Pi    = an.smoothed_bPL(s, a=0, b=b + bPi, kpeak=fPi,
                            alp=alpPi, norm=False, alpha2=True,
                            dlogk=False)
    pimax = ((b + bPi)/(b + bPi - 1))**(-(b + bPi)/alpPi)
    fGW   = fPi/(b + bPi - 1)**(1/alpPi)

    return Pi, fGW, pimax

############# values from Higgsless simulations #############

def data_warning(boxsize=20):

    print('You are using values that are interpolated from numerical',
          'data with L/vw = ', boxsize)
    print('Take into account that only alpha = 0.0046, 0.05',
          ' and 0.5 are found in simulations for vws from 0.32 to 0.8')
    print('Values out of this range should be taken with care')

def interpolate_HL_vals(df, vws, alphas, value='Omega_tilde_int_extrap',
                        boxsize=40, numerical=False, quiet=False):

    '''
    Function that uses the numerical results from Caprini:2024gyk to
    interpolate them to different values of wall velocity and alpha

    The data is stored under
      src/CosmoGW/resources/higgsless/parameters_fit_sims.csv
    '''

    mult_alpha = isinstance(alphas, (list, tuple, np.ndarray))

    columns    = df.box_size == boxsize
    df2        = df[columns]
    val_alphas = np.unique(df2['alpha'])
    val_vws    = np.unique(df2['v_wall'])
    Omegas     = np.zeros((len(val_vws), len(val_alphas))) - 1e30
    for i in range(0, len(val_alphas)):
        for j in range(0, len(val_vws)):
            Om = np.array(df2[value][(df2.v_wall == val_vws[j])* \
                          (df2.alpha == val_alphas[i])])
            if (len(Om) > 0):
                Omegas[j, i] = Om
                # for curly_K0 we interpolate kappa0 = curly_K0/alpha*(1 + alpha)
                if value == 'curly_K_0_512':
                    Omegas[j, i] *= (1 + val_alphas[i])/val_alphas[i]

    # interpolate for all values of vws for the 3 values of
    # alpha
    Omss = np.zeros((len(vws), len(val_alphas)))
    for i in range(0, len(val_alphas)):
        inds  = np.where(Omegas[:, i] > -1e30)[0]
        Omss[:, i] = np.interp(vws, val_vws[inds], Omegas[inds, i])
        inds2 = np.where(Omegas[:, i] == -1e30)[0]
        Omegas[inds2, i] = np.interp(val_vws[inds2], val_vws[inds],
                                     Omegas[inds, i])

    # interpolate for all values of alpha
    if mult_alpha:
        Omsss = np.zeros((len(vws), len(alphas)))
        for i in range(0, len(vws)):
            Omsss[i, :] = np.interp(np.log10(alphas), np.log10(val_alphas),
                                    Omss[i, :])

        # for curly_K0 we interpolate kappa0 = curly_K0/alpha*(1 + alpha)
        if value == 'curly_K_0_512':
            _, alpsij  = np.meshgrid(val_vws, val_alphas, indexing='ij')
            Omegas    *= alpsij/(1 + alpsij)
            _, alpsij  = np.meshgrid(vws, alphas, indexing='ij')
            Omsss     *= alpsij/(1 + alpsij)
    else:
        Omsss = np.zeros(len(vws))
        for i in range(0, len(vws)):
            Omsss[i] = np.interp(np.log10(alphas), np.log10(val_alphas),
                                 Omss[i, :])
        # for curly_K0 we interpolate kappa0 = curly_K0/alpha*(1 + alpha)
        if value == 'curly_K_0_512':
            Omsss  *= alphas/(1 + alphas)
            Omegas *= val_alphas/(1 + val_alphas)

    if not quiet:
        data_warning(boxsize=boxsize)
        if not numerical:
            print('To see numerical values call interpolate_HL_vals function setting',
                  ' numerical to True')

    # if numerical is chosen, also return numerical values
    if numerical: return Omsss, Omegas, val_alphas, val_vws
    else:         return Omsss

###################### TEMPLATE FOR SOUND WAVES ######################

def ampl_GWB_sw(model='fixed_value', OmGW_sw=OmGW_sw_ref, vws=[],
                alphas=[], numerical=False, bs_HL=20, quiet=False):

    '''
    Reference for sound waves is RoperPol:2023bqa, equation 3.

    Value of Omgwtilde = 1e-2 is based on Hindmarsh:2019phv, Hindmarsh:2017gnf
    and used for model = 'fixed_value'

    Values of Omgwtilde from the simulation results of Caprini:2024gyk
    are used for model = 'higgsless' by interpolating from the numerical
    data, which is available for alpha = 0.0046, 0.05 and 0.5 and vws in
    0.32 to 0.8
    '''

    mult_alp = isinstance(alphas, (list, tuple, np.ndarray))
    mult_vw  = isinstance(vws,    (list, tuple, np.ndarray))

    if not mult_alp: alphas = [alphas]
    if not mult_vw:  vws    = [vws]

    if   model == 'fixed_value': Omegas = np.full((len(vws), len(alphas)), OmGW_sw)

    elif model == 'higgsless':

        val_str  = 'Omega_tilde_int_extrap'

        try:
            if len(vws) == 0 or len(alphas) == 0:
                print('Provide values of vws and alphas to use Higgsless model',
                      ' in ampl_GWB_sw')
                return 0

        except:
            tst = True

        # take values from higgsless dataset
        dirr   = importlib.resources.open_binary('cosmoGW',
                        'resources/higgsless/parameters_fit_sims.csv')

        df     = pd.read_csv(dirr)

        if numerical:
            Omegas, Omnum, val_alphas, val_vws = \
                        interpolate_HL_vals(df, vws, alphas,
                                            value=val_str, boxsize=bs_HL,
                                            numerical=numerical, quiet=quiet)

        else: Omegas = interpolate_HL_vals(df, vws, alphas,
                            value=val_str, boxsize=bs_HL, quiet=quiet)

    else:
        print('Choose an available model for ampl_GWB_sw for sound waves')
        print('Available models are fixed_value and higgsless')

    if   not mult_alp and not mult_vw: Omegas = Omegas[0, 0]
    elif not mult_alp: Omegas = Omegas[:, 0]
    elif not mult_vw:  Omegas = Omegas[0, :]

    if numerical and model == 'higgsless': return Omegas, Omnum, val_alphas, val_vws
    else:                                  return Omegas



def pref_GWB_sw(Oms=Oms_ref, lf=lf_ref, alpha=0, model='sound_waves',
                Nshock=1., b=0., expansion=True, beta=100, cs2=cs2_ref):

    '''
    Dependence of the GW spectrum from sound waves on the mean
    size of the bubbles lf = R* H_* and the kinetic energy density

        Oms = vrms^2 = <w v^2>/<w>,

    related to K = kappa alpha/(1 + alpha) used in Caprini:2024gyk as

        Oms   = K/Gamma = kappa alpha/(1 + cs2),     where
        Gamma = <w>/<rho> = (1 + cs2)/(1 + alpha)

    is the adiabatic index Gamma ~ (1 + cs2)/(1 + alpha)

    Note that RoperPol:2023dzg uses OmK = .5 vrms^2 so an extra factor of
    2 appears in its relation to K.

    We consider different models:

    - Model 'sound_waves' uses Caprini:2024gyk eqs. 2.16 and 2.30
        It corresponds to the linear growth with the source duration,
        which is set to

           tdur = Nshock tshock = Nshock R*/sqrt(Oms),

        assuming sound waves do not decay and their UETC is stationary:

           OmGW ~ K^2 R* tdur = K^2 R* Nshock tshock

        When the Universe expansion is included, tdur is substituted
        by the suppresion factor Upsilon describing the growth
        with the source duration. For a radiation-dominated Universe
        (see RoperPol:2023dzg):

           Upsilon = tdur/(1 + tdur)

        This choice includes Universe expansion and assumes sourcing
        occurs during the radiation-dominated era

           OmGW ~ K^2 R* Upsilon(tdur) = K^2 R* tdur/(1 + tdur)

       When Nshock = 1, tdur = tshock = R*/sqrt(Oms) and it becomes equation 3
       of RoperPol:2023bqa, based on Hindmarsh:2020hop, equation 8.24

    - Model 'decaying' uses Caprini:2024gyk, eq. 5.6
        It includes the decay of the source assuming
        a locally stationary UETC. For Minkowski space-time:

           OmGW ~ K2int R*,       K2int     = int K^2 dt,

        while for a radiation-dominated expanding Universe:

            OmGW ~ K2int_exp R*,  K2int_exp = int K^2/t^2 dt.

        We will assume a power law decay K(t > t0) = K0 (dt/dt0)^(-b)
        based on the numerical findings of Caprini:2024gyk,
        see K2int function in GW_models module

    Arguments:
        Oms    -- kinetic energy density, defined as vrms^2, such that
                  Oms = K/Gamma = (1 + cs2) kappa alpha
        lf     -- mean-size of the bubbles, given as a fraction
                  of the Hubble radius, input must be one value
        alpha  -- ratio of vacuum to radiation energy densities
        model  -- chooses the model (sound_waves or decaying)
        Nshock -- describes the duration time in units of the
                  shock time
        b      -- power law decay in time of the kinetic energy
                  source (default 0 recovers stationary sound waves)
        expansion -- option to consider flat Minkowski space-time
                     (if expansion is False, default) or a radiation-
                     dominated expanding Universe
        beta   -- rate of nucleation of the phase transition, input must
                  be one value
        cs2    -- square of the speed of sound (default is 1/3 for
                  radiation domination)


    Returns:
        pref -- prefactor in the GW spectrum, consisting on
                K2int_exp x R*
    '''

    if len(np.shape(alpha)) == 0:
        if alpha == 0:
            print('you need to give the value of alpha as input',
                          'for a correct result in pref_GWB_sw')
            alpha = np.zeros_like(Oms)

    Gamma = (1. + cs2)/(1. + alpha)
    K     = Gamma*Oms

    pref  = K**2*lf
    tdur  = Nshock*lf/np.sqrt(Oms)

    if   model == 'sound_waves':
        if expansion: pref *= 1./(1. + 1./tdur)
        else:         pref *= tdur

    elif model == 'decay':

        K2int = mod.K2int(tdur, K0=K, b=b, expansion=expansion, beta=beta)
        pref  = K2int*lf

    return pref

def Sf_shape_sw(s, model='sw_LISA', Dw=1., a_sw=a_sw_ref, b_sw=b_sw_ref, c_sw=c_sw_ref,
                alp1_sw=0, alp2_sw=0, strength='weak', interpolate_HL=False,
                bsk1_HL=40, bsk2_HL=20, vws=[], alphas=[], quiet=False):

    """
    Function that computes the GW spectral shape generated by sound waves
    based on different templates.

    Arguments:
        s     -- normalized wave number, divided by the mean bubbles
                 size, s = f R*
        model -- model for the sound-wave template (options are 'sw_SSM',
                 'sw_HL', 'sw_LISA', 'sw_LISAold', and 'sw_HLnew')
        Dw -- ratio between peak frequencies, determined by the shell thickness,
              note that different models use slightly different conventions for Dw
        a_sw, b_sw, c_sw -- slopes for sound wave template (default is 3, 1, 3)
        alp1_sw, alp2_sw -- transition parameters for sound wave template
                            (default values for each model are listed above)
        strength -- phase transition strength, used to determine peak2 in sw_HLnew
                    (unless interpolate_HL is True)
        interpolate_HL -- option to use numerical data from Caprini:2024gyk to
                          estimate slope c_sw and peak1 and peak2 positions
        bsk1_HL, bsk2_HL -- box size of Higgsless simulations used to estimate the
                            the peaks (default is 20 for k2 and 40 for k1)
        vws, alphas    -- array of wall velocities and alphas for which the
                          parameters are to be estimated
        quiet          -- option to output a warning about the interpolation
                          method and its range of validity

    Returns:
        S -- spectral shape of the GW spectrum (still to be normalized)
             as a function of s = f R*.
             If interpolate_HL is True, it returns an array of size
             (s, alphas, vws)
    """

    mult_Dw = isinstance(Dw, (list, tuple, np.ndarray))

    if model == 'sw_LISAold':

        # Reference for sound waves based on simulations of Hindmarsh:2017gnf
        # is Caprini:2019egz (equation 30) with only one peak

        # peak positions
        peak1 = 10/2/np.pi
        s     = peak1*s

        S = s**3*(7/(4 + 3*s**2))**(7/2)

    if model == 'sw_SSM':

        # Reference for sound waves based on Sound Shell Model (sw_SSM) is
        # RoperPol:2023bqa, equation 6, based on the results presented in
        # Hindmarsh:2019phv, equation 5.7
        # Uses Dw = |vw - cs|/max(vw, cs)

        if not mult_Dw: Dw = [Dw]
        s, Dw = np.meshgrid(s, Dw, indexing='ij')

        s2 = s*Dw
        m  = (9*Dw**4 + 1)/(Dw**4 + 1)
        M1 = ((Dw**4 + 1)/(Dw**4 + s2**4))**2
        M2 = (5/(5 - m + m*s2**2))**(5/2)
        S  =  M1*M2*s2**9

        if not mult_Dw: S = S[:, 0]

    if model == 'sw_HL':

        # Reference for sound waves based on Higgsless (sw_HL) simulations is
        # RoperPol:2023bqa, equation 7, based on the results presented in
        # Jinno:2022mie
        # Uses Dw = |vw - cs|/max(vw, cs)

        if not mult_Dw: Dw = [Dw]
        s, Dw = np.meshgrid(s, Dw, indexing='ij')

        # amplitude such that S = 1 at s = 1/Dw
        A = 16*(1 + Dw**(-3))**(2/3)*Dw**3*9

        # peak positions
        peak1 = 1.
        peak2 = np.sqrt(3)/Dw

        if alp1_sw == 0: alp1_sw=alp1_sw_ref
        if alp2_sw == 0: alp2_sw=alp2_sw_ref

        S = A*an.smoothed_double_bPL(s, peak1, peak2, A=1., a=a_sw, b=b_sw,
                                     c=c_sw, alp1=alp1_sw, alp2=alp2_sw)

        if not mult_Dw: S = S[:, 0]

    if model == 'sw_LISA':

        # Reference for sound waves based on Higgsless simulations is
        # Caprini:2024hue (equation 2.8), based on the results presented in
        # Jinno:2022mie, see updated results and discussion in Caprini:2024gyk.
        # Uses Dw = xi_shell/max(vw, cs)

        # smoothness parameters
        if alp1_sw == 0: alp1_sw = alp1_LISA
        if alp2_sw == 0: alp2_sw = alp2_LISA

        NDw = len(np.shape(Dw))
        if NDW == 1: s, Dw = np.meshgrid(s, Dw, indexing='ij')
        if NDw == 2:
            s0  = np.zeros((len(s), np.shape(Dw)[0], np.shape(Dw)[1]))
            Dw0 = np.zeros((len(s), np.shape(Dw)[0], np.shape(Dw)[1]))
            for i in range(0, np.shape(Dw)[0]):
                for j in range(0, np.shape(Dw)[1]):
                    s0[:,  i, j] = s
                    Dw0[:, i, j] = Dw[i, j]
            s  = s0
            Dw = Dw0

         # peak positions
        peak1 = 0.2
        peak2 = 0.5/Dw

        S = an.smoothed_double_bPL(s, peak1, peak2, A=1., a=a_sw, b=b_sw,
                                   c=c_sw, alp1=alp1_sw, alp2=alp2_sw, alpha2=True)

    if model == 'sw_HLnew':

        # Reference for sound waves based on updated HL results (sw_HLnew)
        # is Caprini:2024gyk
        # Uses Dw = xi_shell/max(vw, cs)

        # smoothness parameters
        if alp1_sw == 0: alp1_sw = alp1_HL
        if alp2_sw == 0: alp2_sw = alp2_HL

        NDw = len(np.shape(Dw))
        if NDW == 1: s, Dw = np.meshgrid(s, Dw, indexing='ij')
        if NDw == 2:
            s0  = np.zeros((len(s), np.shape(Dw)[0], np.shape(Dw)[1]))
            Dw0 = np.zeros((len(s), np.shape(Dw)[0], np.shape(Dw)[1]))
            for i in range(0, np.shape(Dw)[0]):
                for j in range(0, np.shape(Dw)[1]):
                    s0[:,  i, j] = s
                    Dw0[:, i, j] = Dw[i, j]
            s  = s0
            Dw = Dw0

        # peak positions
        peak1 = 0.4
        if   strength == 'interm': peak2 = 1.
        elif strength == 'strong': peak2 = 0.5
        else:                      peak2 = 0.5/Dw

        S = an.smoothed_double_bPL(s, peak1, peak2, A=1., a=a_sw, b=b_sw,
                                   c=c_sw, alp1=alp1_sw, alp2=alp2_sw, alpha2=True)

        if interpolate_HL:

            if len(vws) or len(alphas) == 0:
                print('To use interpolate_HL in Sf_shape_sw',
                      ' give values of vws and alphas')

            # take values from higgsless dataset
            dirr     = importlib.resources.open_binary('cosmoGW',
                            'resources/higgsless/parameters_fit_sims.csv')
            df       = pd.read_csv(dirr)

            val_str  = 'k1'
            peaks1   = interpolate_HL_vals(df, vws, alphas, quiet=True,
                                   value=val_str, boxsize=bs_k1HL)/(2*np.pi)
            val_str  = 'k2'
            peaks2   = interpolate_HL_vals(df, vws, alphas, quiet=True,
                                   value=val_str, boxsize=bs_k2HL)/(2*np.pi)
            val_str  = 'n3'
            c_sw     = - interpolate_HL_vals(df, vws, alphas, quiet=True,
                                   value=val_str, boxsize=bs_k2HL)

            if not quiet: data_warning(boxsize=bs_HL)

            S = an.smoothed_double_bPL(s, peaks1,
                                       peaks2, A=1., a=a_sw, b=b_sw,
                                          c=c_sw, alp1=alp1, alp2=alp2)

    return S

def OmGW_spec_sw(ss, alphas, betas, vws=1., cs2=cs2_ref, quiet=False, a_sw=a_sw_ref,
                 b_sw=b_sw_ref, c_sw=c_sw_ref, alp1_sw=0, alp2_sw=0, corrRs=True,
                 expansion=False, Nsh=1.,
                 model_efficiency='fixed_value', OmGW_tilde=OmGW_sw_ref,
                 bs_HL_eff=20, model_K0='Espinosa', bs_HL_peak1=40,
                 model_decay='sound_waves', interpolate_HL_decay=True, b=0,
                 model_shape='sw_LISA', interpolate_HL_shape=False):

    '''
    Function that computes the GW spectrum (normalized to radiation
    energy density within RD era) for sound waves and turbulence.

    The general shape of the GW spectrum is based on that of reference
    RoperPol:2023bqa, equations 3 and 9:

        OmGW (f) = 3 * ampl_GWB * pref_GWB * S(f),

    where:

    - ampl_GWB is the efficiency of GW production by the specific source,
    - pref_GWB is the dependence of the GW amplitude on the source
      parameters (e.g. length scale and strength of the source),
    - S(f) is a normalized spectral shape, such that int S(f) d ln f = 1

    The GW spectrum as an observable at present time is then computed using

        OmGW0 (f) = OmGW x  FGW0,

    where FGW0 is the redshift from the time of generation to present
    time, computed in cosmoGW.py that depends on the degrees of freedom at
    the time of generation.

    Arguments:
        ss     -- normalized wave number, divided by the mean bubbles size Rstar, s = f R*
        alphas -- strength of the phase transition
        betas  -- rate of nucleation of the phase transition
        vws    -- wall velocity

        cs2 -- square of the speed of sound (default is 1/3 for radiation domination)
        quiet --
        multi_ab -- option to provide an array of values of alpha and beta as input
        multi_xi -- option to provide an array of values of xiw as input
        a_sw, b_sw, c_sw -- slopes for sound wave template, used when tp = 'sw_HL'
        alp1_sw, alp2_sw -- transition parameters for sound wave template, used when tp = 'sw_HL'
        corr_Rs
        model_efficiency
        OmGW_tilde -- efficiency of GW production from sound waves (default value is 1e-2,
                        based on numerical simulations)
        bs_HL_eff
        model_K0


        a_turb, b_turb, alp_turb -- slopes and smoothness of the turbulent source spectrum
                                    (either magnetic or kinetic), default values are for a
                                    von Karman spectrum
        alpPi, fPi -- parameters of the fit of the spectral anisotropic stresses for turbulence
        eps_turb -- fraction of energy density converted from sound waves into turbulence
    '''

    cs         = np.sqrt(cs2)
    mult_alpha = isinstance(alpha, (list, tuple, np.ndarray))
    mult_beta  = isinstance(beta,  (list, tuple, np.ndarray))
    mult_vws   = isinstance(vws,   (list, tuple, np.ndarray))

    #### Computing ampl_GWB

    if model_efficiency == 'higgsless' and not quiet:
        print('Computing the OmGW efficiency')
        data_warning(boxsize=bs_HL_eff)

    ampl = ampl_GWB(model=model_efficiency, OmGW_sw=OmGW_tilde,
                    vws=vws, alphas=alphas, bs_HL=bs_HL_eff, quiet=True)

    #### Computing pref_GWB

    # Kinetic energy density

    if not quiet:
        print('Computing the kinetic energy density using the model',
              model_K0)
        if model_K0 == 'higgsless': data_warning(boxsize=bs_HL_eff)

    if model_K0 == 'Espinosa':

        # compute kappa, K and Oms following the bag equation of
        # state as in Espinosa:2010hh

        # kappa: efficiency in converting vacuum to kinetic energy
        # K = rho_kin/rho_total = kappa alpha/(1 + alpha)
        # Oms_sw = v_f^2 = kappa alpha/(1 + cs2)
        kap    = hb.kappas_Esp(vws, alphas, cs2=cs2)
        Oms_sw = kap*alphas/(1 + cs2)

    elif model_K0 == 'higgsless':

        # compute K and Oms directly from the numerical results of the
        # Higgsless simulations of Caprini:2024gyk and interpolate to
        # values of alpha and vws

        dirr    = importlib.resources.open_binary('cosmoGW',
                       'resources/higgsless/parameters_fit_sims.csv')
        df      = pd.read_csv(dirr)
        val_str = 'curly_K_0_512'
        K       = interpolate_HL_vals(df, vws, alphas, quiet=quiet,
                                      value=val_str, boxsize=bs_HL_eff)
        kap     = K*(1 + alphas)/alphas
        Oms_sw  = kap*alphas/(1 + cs2)

    else:
        print('Choose an available model for K0 in OmGW_spec_sw')
        print('Available models are Espinosa and higgsless')

    # Decay rate

    if interpolate_HL_decay and model_decay == 'decay':

        dirr    = importlib.resources.open_binary('cosmoGW',
                    'resources/higgsless/parameters_fit_sims.csv')
        df      = pd.read_csv(dirr)

        val_str = 'b'
        b       = tmp.interpolate_HL_vals(df, vws, alphas, quiet=quiet,
                                          value=val_str, boxsize=bs_HL_eff)

    # Fluid length scale

    if not mult_alpha: alphas = [alphas]
    if not mult_vws:   vws    = [vws]
    if not mult_beta:  betas  = [betas]

    lf = np.zeros((len(vws), len(alphas), len(betas)))
    for i in range(0, len(vws)):
        for j in range(0, len(alphas)):
            lf[i, j, :] = hb.Rstar_beta(vws[i], cs2=cs2, corr=corrRs)/betas

    # prefactor GWB of sound waves

    pref = np.zeros_like(lf)
    for i in range(0, len(betas)):
        pref = pref_GWB_sw(Oms=Oms_sw, lf=lf[:, :, i], alpha=alphas,
                           model_decay='sound_waves', Nshock=Nsh, b=b,
                           expansion=expansion, beta=betas[i], cs2=cs2)

    #### Computing spectral shape

    # sound-shell thickness and spectral shape

    if model == ['sw_LISAold']:

        S  = Sf_shape_sw(ss, model=model)

    elif model in ['sw_HL',   'sw_SSM']:

        Dw = abs(vws - cs)/vws
        S  = Sf_shape_sw(ss, model=model, Dw=Dw, a_sw=a_sw, b_sw=b_sw, c_sw=c_sw,
                         alp1_sw=alp1_sw, alp2_sw=alp2_sw)

    elif model in ['sw_LISA', 'sw_HLnew']:

        print('Computing sound-shell thickness')
        xis, vvs, wws, alphas_n, conv, shocks, xi_shocks, wms = \
                        hb.compute_profiles_vws_multalp(alphas, vws=vws,
                                alphan=True, quiet=True, eff=False)

        Dw = np.zeros((len(vws), len(alphas)))
        for i in range(0, len(alphas)):
            Dw[:, i] = (xi_shocks[:, i] - np.minimum(vws, cs))/np.maximum(vws, cs)

        S  = Sf_shape_sw(ss, model=model, Dw=Dw, a_sw=a_sw, b_sw=b_sw, c_sw=c_sw,
                alp1_sw=alp1_sw, alp2_sw=alp2_sw, strength=strength_shape,
                interpolate_HL=interpolate_HL_shape,
                bsk1_HL=bs_HL_peak1, bsk2_HL=bs_HL_eff, vws=vws,
                alphas=alphas, quiet=quiet)

    else:
        print('Choose an available model for model_shape in OmGW_spec_sw')
        print('Available models are sw_LISA, sw_HL, sw_HLnew, sw_SSM, sw_LISAold')

    # amplitude factors
    if multi_ab and multi_xi:
        Oms, lf, xiw_ij = np.meshgrid(alpha, beta, xiw, indexing='ij')
        for i in range(0, len(beta)): Oms_ij[:, i, :] = Oms
        for i in range(0, len(alpha)): lf_ij[i, :, :] = lf
    elif multi_ab and not multi_xi:
        Oms, lf = np.meshgrid(Oms, lf, indexing='ij')

    preff = pref_GWB(Oms=Oms, lf=lf, tp=tp, b_turb=b_turb, alpPi=alpPi, fPi=fPi)

    # spectral shape for sound waves templates
    if tp == 'sw':
        if multi_xi:
            OmGW_aux = np.zeros((len(ss), len(xiw)))
            for i in range(0, len(xiw)):
                S = Sf_shape(ss, tp=tp, Dw=Dw[i], a_sw=a_sw, b_sw=b_sw, c_sw=c_sw,
                             alp1_sw=alp1_sw, alp2_sw=alp2_sw)
                mu = np.trapz(S, np.log(ss))
                OmGW_aux[:, i] = 3*S*ampl/mu

            if multi_ab:
                OmGW = np.zeros((len(ss), len(alpha), len(beta), len(xiw)))
                for i in range(0, len(xiw)):
                    for j in range(0, len(ss)):
                        OmGW[j, :, :, i] = OmGW_aux[j, i]*preff[:, :, i]
            else: OmGW = OmGW_aux*preff[i]

        else:
            S = Sf_shape(ss, tp=tp, Dw=Dw, a_sw=a_sw, b_sw=b_sw, c_sw=c_sw,
                         alp1_sw=alp1_sw, alp2_sw=alp2_sw)
            mu = np.trapz(S, np.log(ss))
            OmGW_aux = 3*S*ampl/mu
            if multi_ab:
                OmGW = np.zeros((len(ss), len(alpha), len(beta)))
                for j in range(0, len(ss)):
                    OmGW[j, :, :] = OmGW_aux[j]*preff
            else: OmGW = OmGW_aux*preff

    # Oms = rho_source/rho_total = K eps_turb
    Oms_turb = K*eps_turb

    # spectral shape for turbulence templates
    if tp2 == 'turb':
        if multi_xi:
            if multi_ab:
                OmGW = np.zeros((len(ss), len(xiw), len(alpha), len(beta)))
                for i in range(0, len(xiw)):
                    OmGW[:, i, :, :] = Sf_shape(ss, tp=tp, b_turb=b_turb, N=N_turb,
                                                Oms=Oms[:, :, i], lf=lf[:, :, i],
                                                alpPi=alpPi, fPi=fPi, ref=ref, cs2=cs2,
                                                multi=multi_ab)
                    for j in range(0, len(ss)):
                        OmGW[j, i, :, :] = 3*OmGW[j, i, :, :]*ampl*preff[i, :, :]

            else:
                OmGW = np.zeros((len(ss), len(xiw)))
                for i in range(0, len(xiw)):
                    OmGW[:, i] = Sf_shape(ss, tp=tp, b_turb=b_turb, N=N_turb, Oms=Oms[i],
                                          lf=lf[i], alpPi=alpPi, fPi=fPi,
                                          ref=ref, cs2=cs2, multi=multi_ab)
                    OmGW[:, i] = 3*OmGW[:, i]*ampl*preff[i]

        else:
            S = Sf_shape(ss, tp=tp, b_turb=b_turb, N=N_turb, Oms=Oms, lf=lf,
                         alpPi=alpPi, fPi=fPi, ref=ref, cs2=cs2, multi=multi_ab)
            OmGW = 3*OmGW*ampl*preff

    return OmGW

################# TEMPLATE FOR MHD TURBULENCE #################

def ampl_GWB_turb(a_turb=a_turb, b_turb=b_turb, alp=alp_turb):

    """
    Reference for turbulence is RoperPol:2023bqa, equation 9,
    based on the template of RoperPol:2022iel, section 3 D.

    See footnote 3 of RoperPol:2023bqa for clarification
    (extra factor 1/2 has been added to take into account average over
    oscillations that were ignored in RoperPol:2022iel).

    Arguments:
        a_turb, b_turb, alp_turb -- slopes and smoothness of the turbulent
                     source spectrum (either magnetic or kinetic), default
                     values are for a von Karman spectrum
    """

    A    = an.calA(a=a_turb, b=b_turb, alp=alp_turb)
    C    = an.calC(a=a_turb, b=b_turb, alp=alp_turb, tp='vort')
    ampl = .5*C/A**2

    return ampl

def pref_GWB_turb(Oms=Oms_ref, lf=lf_ref, b_turb=b_turb,
                  alpPi=alpPi, fPi=fPi, bPi=bPi_vort):

    '''
    Dependence of the GW spectrum from turbulence on the mean
    size of the bubbles lf = R* H_* and the kinetic energy density Oms.

    Reference is RoperPol:2023bqa, equation 9, based on RoperPol:2022iel,
    section II D.

    Arguments:
        Oms -- energy density of the source (i.e., 1/2 vrms^2)
        lf  -- mean-size of the bubbles, given as a fraction of the Hubble radius
    '''

    pref = (Oms*lf)**2

    return pref

def Sf_shape_turb(s, b_turb=b_turb, N=N_turb, Oms=.1, lf=1., alpPi=alpPi, fPi=fPi,
                  bPi=bPi_vort,
                  ref='f', cs2=cs2_ref, multi=False):

    """
    Function that computes the spectral shape derived for GWs generated by
    MHD turbulence.

    Reference for vortical (MHD) turbulence is RoperPol:2023bqa,
    equation 9, based on the analytical model presented in RoperPol:2022iel,
    section II D.

    Also used in EPTA:2023xxk, Caprini:2024hue
    See further details in RoperPol:2025b

    Arguments:
        s      -- normalized wave number, divided by the mean bubbles
                  size, s = f R*
        b_turb -- slope of the velocity/magnetic field spectrum in the UV
        N      -- relation between the decay time and the effective
                  source duration
        Oms    -- energy density of the source (i.e., 1/2 vrms^2)
        lf     -- characteristic scale of the turbulence as a
                 fraction of the Hubble radius, R* H*
        alpPi, fPi, bPi_vort  -- parameters of the pPi_fit
        fPi    --
        alp1_sw, alp2_sw -- transition parameters for sound wave template, used when tp = 'sw_HL'

    Returns:
        spec -- spectral shape, normalized such that S = 1 at its peak
    """

    TGW      = mod.TGW_func(s, N=N, Oms=Oms, lf=lf, cs2=cs2, multi=multi)
    Pi, _, _ = pPi_fit(s, b=b_turb, alpPi=alpPi, fPi=fPi, bPi=bPi)
    s3Pi = s**3*Pi
    BB   = 1/lf**2
    if multi: s3Pi, BB, _ = np.meshgrid(s3Pi, BB, Oms, indexing='ij')

    S = s3Pi*BB*TGW

    return S
