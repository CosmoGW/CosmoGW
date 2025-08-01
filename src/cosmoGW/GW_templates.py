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
import cosmoGW.cosmoGW       as cGW
import cosmoGW.cosmology     as co
import cosmoGW.GW_analytical as an
import cosmoGW.GW_models     as mod
import cosmoGW.hydro_bubbles as hb

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
fPi      = 2.2         # break frequency of the anisotropic stresses
                       # obtained for a von Karman spectrum (using dlogk = False)
                       # as considered in RoperPol:2023bqa
tdecay_ref = 'eddy'    # decaying time used in the constant-in-time model
                       # (default is eddy turnover time)

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
alp1_ssm    = 4;   alp2_ssm    = 2.   # used in Hindmarsh:2019phv
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

############################ SOUND WAVES ############################

################# values from Higgsless simulations #################

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
            if len(np.shape(vws)) == 1 or len(np.shape(alphas)) == 1:
                if len(vws) == 0 or len(alphas) == 0:
                    print('Provide values of vws and alphas to use Higgsless model',
                          ' in ampl_GWB_sw')
                    return 0

        except:
            tst = True

        # take values from higgsless dataset
        dirr = cGW.HOME + 'resources/higgsless/parameters_fit_sims.csv'
        df   = pd.read_csv(dirr)

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
        return 0

    if   not mult_alp and not mult_vw: Omegas = Omegas[0, 0]
    elif not mult_alp: Omegas = Omegas[:, 0]
    elif not mult_vw:  Omegas = Omegas[0, :]

    if numerical and model == 'higgsless': return Omegas, Omnum, val_alphas, val_vws
    else:                                  return Omegas

def pref_GWB_sw(Oms=mod.Oms_ref, lf=mod.lf_ref, alpha=0, model='sound_waves',
                Nshock=1., b=0., expansion=True, beta=mod.beta_ref,
                cs2=hb.cs2_ref):

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

        K2int = mod.K2int(tdur, K0=K, b=b, dt0=mod.dt0_ref, expansion=expansion, beta=beta)
        pref  = K2int*lf

    return pref

def Sf_shape_sw(s, model='sw_LISA', Dw=1., a_sw=a_sw_ref, b_sw=b_sw_ref, c_sw=c_sw_ref,
                alp1_sw=0, alp2_sw=0, strength='weak', interpolate_HL=False,
                bs_k1HL=40, bs_k2HL=20, vws=[], alphas=[], quiet=False,
                interpolate_HL_n3=False, corrRs=True, cs2=hb.cs2_ref):

    """
    Function that computes the GW spectral shape generated by sound waves
    based on different templates.

    Arguments:
        s     -- normalized wave number, divided by the mean bubbles
                 size, s = f R*
        model -- model for the sound-wave template (options are 'sw_SSM',
                 'sw_HL', 'sw_LISA', 'sw_LISAold', and 'sw_HLnew')
        Dw    -- ratio between peak frequencies, determined by the shell thickness,
                 note that different models use slightly different conventions for Dw
        a_sw, b_sw, c_sw -- slopes for sound wave template (default is 3, 1, 3)
        alp1_sw, alp2_sw -- transition parameters for sound wave template
                            (default values for each model are listed above)
        strength -- phase transition strength, used to determine peak2 in sw_HLnew
                    (unless interpolate_HL is True)
        interpolate_HL   -- option to use numerical data from Caprini:2024gyk to
                            estimate slope c_sw and peak1 and peak2 positions
        bsk1_HL, bsk2_HL -- box size of Higgsless simulations used to estimate the
                            the peaks (default is 20 for k2 and 40 for k1)
        vws, alphas      -- array of wall velocities and alphas for which the
                            parameters are to be estimated
        quiet            -- option to output a warning about the interpolation
                            method and its range of validity

    Returns:
        S -- spectral shape of the GW spectrum (still to be normalized)
             as a function of s = f R*.
             If interpolate_HL is True, it returns an array of size
             (s, alphas, vws)
    """

    mult_Dw = isinstance(Dw, (list, tuple, np.ndarray))

    # in some models, Dw can be a 0, 1, or 2d input (vw, alpha) dependence
    Dw_2d = False
    if model == 'sw_LISA': Dw_2d = True
    if model == 'sw_HLnew':
        if strength == 'weak' and not interpolate_HL:
            Dw_2d = True

    if Dw_2d:

            # sound-shell thickness Dw needs to be used as an input
            NDw = len(np.shape(Dw))
            if NDw == 1: s, Dw = np.meshgrid(s, Dw, indexing='ij')
            if NDw == 2:
                s0  = np.zeros((len(s), np.shape(Dw)[0], np.shape(Dw)[1]))
                Dw0 = np.zeros((len(s), np.shape(Dw)[0], np.shape(Dw)[1]))
                for i in range(0, np.shape(Dw)[0]):
                    for j in range(0, np.shape(Dw)[1]):
                        s0[:,  i, j] = s
                        Dw0[:, i, j] = Dw[i, j]
                s   = s0
                Dw  = Dw0

    if model == 'sw_LISAold':

        # Reference for sound waves based on simulations of Hindmarsh:2017gnf
        # is Caprini:2019egz (equation 30) with only one peak

        # peak positions
        peak1 = 2*np.pi/10
        s     = peak1*s

        S = s**3*(7/(4 + 3*s**2))**(7/2)

    elif model == 'sw_SSM':

        # Reference for sound waves based on Sound Shell Model (sw_SSM) is
        # RoperPol:2023bqa, equation 6, based on the results presented in
        # Hindmarsh:2019phv, equation 5.7
        # Uses Dw = |vw - cs|/max(vw, cs)

        # if not mult_Dw: Dw = [Dw]
        s, Dw = np.meshgrid(s, Dw, indexing='ij')

        # uses a different slope at large frequencies to adapt the result
        # at intermediate ones
        if c_sw == 3: c_sw = 4
        # takes a different slope at small frequencies
        if a_sw == 3: a_sw = 9

        # amplitude such that S = 1 at s = 1/Dw
        m = (9*Dw**4 + 1)/(Dw**4 + 1)
        A = Dw**9*(1 + Dw**(-4))**2*(5/(5 - m))**(5/2)

        # peak positions
        peak1 = 1.
        peak2 = np.sqrt((5 - m)/m)/Dw

        if alp1_sw == 0: alp1_sw=alp1_ssm
        if alp2_sw == 0: alp2_sw=alp2_ssm

        S = A*an.smoothed_double_bPL(s, peak1, peak2, A=1., a=a_sw, b=b_sw,
                                     c=c_sw, alp1=alp1_sw, alp2=alp2_sw,
                                     alpha2=True)

        if not mult_Dw: S = S[:, 0]

    elif model == 'sw_HL':

        # Reference for sound waves based on Higgsless (sw_HL) simulations is
        # RoperPol:2023bqa, equation 7, based on the results presented in
        # Jinno:2022mie
        # Uses Dw = |vw - cs|/max(vw, cs)

        # if not mult_Dw: Dw = [Dw]
        s, Dw = np.meshgrid(s, Dw, indexing='ij')

        # amplitude such that S = 1 at s = 1/Dw
        A = 16*(1 + Dw**(-3))**(2/3)*Dw**3

        # peak positions
        peak1 = 1.
        peak2 = np.sqrt(3)/Dw

        if alp1_sw == 0: alp1_sw=alp1_sw_ref
        if alp2_sw == 0: alp2_sw=alp2_sw_ref

        S = A*an.smoothed_double_bPL(s, peak1, peak2, A=1., a=a_sw, b=b_sw,
                                     c=c_sw, alp1=alp1_sw, alp2=alp2_sw)

        if not mult_Dw: S = S[:, 0]

    elif model == 'sw_LISA':

        # Reference for sound waves based on Higgsless simulations is
        # Caprini:2024hue (equation 2.8), based on the results presented in
        # Jinno:2022mie, see updated results and discussion in Caprini:2024gyk.
        # Uses Dw = xi_shell/max(vw, cs)

        # smoothness parameters
        if alp1_sw == 0: alp1_sw = alp1_LISA
        if alp2_sw == 0: alp2_sw = alp2_LISA

        # peak positions
        peak1 = 0.2
        peak2 = 0.5/Dw

        S = an.smoothed_double_bPL(s, peak1, peak2, A=1., a=a_sw, b=b_sw,
                                   c=c_sw, alp1=alp1_sw, alp2=alp2_sw, alpha2=True)

    elif model == 'sw_HLnew':

        # Reference for sound waves based on updated HL results (sw_HLnew)
        # is Caprini:2024gyk
        # Uses Dw = xi_shell/max(vw, cs)

        # smoothness parameters
        if alp1_sw == 0: alp1_sw = alp1_HL
        if alp2_sw == 0: alp2_sw = alp2_HL

        if not interpolate_HL:

            # peak positions
            peak1 = 0.4
            if   strength == 'weak':   peak2 = 0.5/Dw
            elif strength == 'interm': peak2 = 1.
            else:                      peak2 = 0.5

            S = an.smoothed_double_bPL(s, peak1, peak2, A=1., a=a_sw, b=b_sw,
                                       c=c_sw, alp1=alp1_sw, alp2=alp2_sw, alpha2=True)

        else:

            if len(vws) == 0 or len(alphas) == 0:
                print('To use interpolate_HL in Sf_shape_sw',
                      ' give values of vws and alphas')
                return 0

            # take values from higgsless dataset
            dirr     = cGW.HOME + 'resources/higgsless/parameters_fit_sims.csv'
            df       = pd.read_csv(dirr)

            val_str  = 'k1'
            peaks1   = interpolate_HL_vals(df, vws, alphas, quiet=True,
                                   value=val_str, boxsize=bs_k1HL)
            val_str  = 'k2'
            peaks2   = interpolate_HL_vals(df, vws, alphas, quiet=True,
                                   value=val_str, boxsize=bs_k2HL)

            s0      = np.zeros((len(s), len(vws), len(alphas)))
            peaks10 = np.zeros((len(s), len(vws), len(alphas)))
            peaks20 = np.zeros((len(s), len(vws), len(alphas)))

            Rstar_beta = hb.Rstar_beta(vws=vws, corr=corrRs, cs2=cs2)/2/np.pi
            for i in range(0, len(vws)):
                for j in range(0, len(alphas)):
                    peaks10[:, i, j] = peaks1[i, j]*Rstar_beta[i]
                    peaks20[:, i, j] = peaks2[i, j]*Rstar_beta[i]
                    s0[:, i, j]      = s

            peaks1 = peaks10
            peaks2 = peaks20
            s      = s0

            if interpolate_HL_n3:
                val_str  = 'n3'
                c_sw     = - interpolate_HL_vals(df, vws, alphas, quiet=True,
                                       value=val_str, boxsize=bs_k2HL)

            if not quiet: data_warning(boxsize='%i and %i'%(bs_k1HL,bs_k2HL))

            S = an.smoothed_double_bPL(s, peaks1, peaks2, A=1., a=a_sw, b=b_sw,
                                       c=c_sw, alp1=alp1_sw, alp2=alp2_sw, alpha2=True)

    else:
        print('Choose an available model in Sf_shape_sw for the sound wave',
              'spectral shape.')
        print('Available models are sw_LISAold, sw_SSM, sw_HL, sw_LISA, sw_HLnew')
        return 0

    return S

def OmGW_spec_sw(s, alphas, betas, vws=1., cs2=hb.cs2_ref, quiet=True, a_sw=a_sw_ref,
                 b_sw=b_sw_ref, c_sw=c_sw_ref, alp1_sw=0, alp2_sw=0, corrRs=True,
                 expansion=True, Nsh=1., model_efficiency='fixed_value',
                 OmGW_tilde=OmGW_sw_ref, bs_HL_eff=20, model_K0='Espinosa', bs_k1HL=40,
                 model_decay='sound_waves', interpolate_HL_decay=True, b=0,
                 model_shape='sw_LISA', strength='weak', interpolate_HL_shape=False,
                 interpolate_HL_n3=False, redshift=False, gstar=co.gref, gS=0, T=co.Tref,
                 h0=1., Neff=co.Neff_ref):

    '''
    Function that computes the GW spectrum (normalized to radiation
    energy density within RD era) for sound waves.

    The general shape of the GW spectrum is based on that of reference
    RoperPol:2023bqa, equations 3 and 9:

        OmGW (f) = 3 * ampl_GWB * pref_GWB * S(f),

    where:

    - ampl_GWB is the efficiency of GW production by the specific source,
    - pref_GWB is the dependence of the GW amplitude on the source
      parameters (e.g. length scale and strength of the source),
    - S(f) is a normalized spectral shape, such that int S(f) d ln f = 1

    The GW spectrum as an observable at present time is then computed using

        OmGW0 (f) = OmGW x FGW0,

    where FGW0 is the redshift from the time of generation to present
    time, computed in cosmoGW.py that depends on the degrees of freedom at
    the time of generation.

    Arguments:
        s                    -- normalized wave number, divided by the mean bubbles
                                size Rstar, s = f R*
        alphas               -- strength of the phase transition
        betas                -- rate of nucleation of the phase transition
        vws                  -- array of wall velocities
        cs2                  -- square of the speed of sound
                                (default is 1/3 for radiation domination)
        quiet                -- option to avoid printing debugging info (default is True)
        a_sw, b_sw, c_sw     -- slopes of the sound wave template
                                (takes 3, 1, 3 by default)
        alp1_sw, alp2_sw     -- transition parameters of the sound wave template
                                (takes the template values by default of each model_shape)
        corrRs               -- option to correct the ratio R* beta with max(vw, cs)
        expansion            -- option to include the Universe expansion
                                (in radiation domination)
        Nsh                  -- number of shock formation times to determine the
                                source duration (default is 1)
        model_efficiency     -- model to compute Omega tilde (default is 'fixed_value',
                                other option available is 'higgsless')
        OmGW_tilde           -- value of Omega tilde used when
                                model_efficiency='fixed_value' (default is 1e-2)
        bs_HL_eff            -- box size of the Higgsless simulations used to interpolate
                                the values of Omega tilde when model_efficiency='higgsless'
                                (default if L/vw = 20)
        model_K0             -- model to compute the kinetic energy ratio K0
                                (default is 'Espinosa' to use the the single-bubble value,
                                other option is 'higgsless')
        bs_k1HL              -- box size of the Higgsless simulations used to interpolate
                                the values of k1 when model_shape='higgsless'
                                (default if L/vw = 20)
        model_decay          -- model to compute the prefactor of the GW amplitude
                                (default is 'sound_waves' to consider stationary sourcing,
                                other option is 'decay')
        interpolate_HL_decay -- option to use Higgsless simulations to interpolate the
                                values of b (default is False)
        b                    -- decay law in time of the kinetic energy density
                                (default is 0)
        model_shape          -- model used to compute the spectral shape (options
                                are 'sw_LISAold', 'sw_SSM', 'sw_HL', 'sw_LISA',
                                'sw_HLnew')
        strength             -- strength of the phase transition to use average
                                values of k1 and k2 when model_shape = 'sw_HLnew'
                                is used (default is 'weak', other options are
                                'interm' and 'strong')
        interpolate_HL_shape -- option to use Higgsless simulations to interpolate the
                                values of k1 and k2 (default is False)
        interpolate_HL_n3    -- option to use Higgsless simulations to interpolate the
                                values of c_sw (default is False)
        redshift             -- option to redshift the GW spectrum and frequencies to
                                present time (default is False)
        gstar                -- number of degrees of freedom (default is 100) when
                                redshift = True
        gS                   -- number of adiabatic degrees of freedom (default is gstar)
                                when redshift = True
        T                    -- temperature scale (default is 100 GeV) when redshift = True
        h0                   -- value of the Hubble rate at present time (100 h0 Mpc/km/s)
                                (default is one)
        Neff                 -- effective number of neutrino species (default is 3)

    Returns:
        freq                 -- array of frequencies (normalized with Rstar if redshift
                                is False and in Hz otherwise)
        OmGW                 -- GW spectrum normalized to the radiation energy density
                                (or to the present critical density if redshift is True)
    '''

    cs         = np.sqrt(cs2)
    mult_alpha = isinstance(alphas, (list, tuple, np.ndarray))
    mult_beta  = isinstance(betas,  (list, tuple, np.ndarray))
    mult_vws   = isinstance(vws,    (list, tuple, np.ndarray))

    if not mult_alpha: alphas = np.array([alphas])
    if not mult_vws:   vws    = np.array([vws])
    if not mult_beta:  betas  = np.array([betas])

    #### Computing ampl_GWB

    if model_efficiency == 'higgsless' and not quiet:
        print('Computing the OmGW efficiency')
        data_warning(boxsize=bs_HL_eff)

    ampl = ampl_GWB_sw(model=model_efficiency, OmGW_sw=OmGW_tilde,
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
        K      = kap*alphas/(1 + alphas)
        Oms_sw = kap*alphas/(1 + cs2)

    elif model_K0 == 'higgsless':

        # compute K and Oms directly from the numerical results of the
        # Higgsless simulations of Caprini:2024gyk and interpolate to
        # values of alpha and vws

        dirr    = cGW.HOME + 'resources/higgsless/parameters_fit_sims.csv'
        df      = pd.read_csv(dirr)
        val_str = 'curly_K_0_512'
        K       = interpolate_HL_vals(df, vws, alphas, quiet=quiet,
                                      value=val_str, boxsize=bs_HL_eff)
        kap     = K*(1 + alphas)/alphas
        Oms_sw  = kap*alphas/(1 + cs2)

    else:
        print('Choose an available model for K0 in OmGW_spec_sw')
        print('Available models are Espinosa and higgsless')
        return 0

    # Decay rate

    interpol_b = False
    if interpolate_HL_decay and model_decay == 'decay':

        dirr    = cGW.HOME + 'resources/higgsless/parameters_fit_sims.csv'
        df      = pd.read_csv(dirr)

        val_str = 'b'
        b       = interpolate_HL_vals(df, vws, alphas, quiet=quiet,
                                          value=val_str, boxsize=bs_HL_eff)
        interpol_b = True

    # prefactor GWB of sound waves

    pref = np.zeros((len(vws), len(alphas), len(betas)))
    for i in range(0, len(vws)):
        # Fluid length scale R_star x beta
        lf = hb.Rstar_beta(vws[i], cs2=cs2, corr=corrRs)/betas
        for j in range(0, len(alphas)):
            for l in range(0, len(betas)):
                if interpol_b: b_ij = b[i, j]
                else:          b_ij = b
                pref[i, j, l] = pref_GWB_sw(Oms=Oms_sw[i, j], lf=lf[l], alpha=alphas[j],
                                    model=model_decay, Nshock=Nsh, b=b_ij,
                                    expansion=expansion, beta=betas[l], cs2=cs2)

    #### Computing the spectral shape

    # sound-shell thickness and spectral shape

    if not quiet: print('Computing spectral shape using model ',
                        model_shape)

    if model_shape == ['sw_LISAold']:

        S  = Sf_shape_sw(s,  model=model_shape)
        try:    mu = np.trapezoid(S, np.log(s))
        except: mu = np.trapz(S, np.log(s))
        # normalized spectral shape
        S  = S/mu
        S, _, _ = np.meshgrid(S, vws, alphas, indexing='ij')

    elif model_shape in ['sw_HL', 'sw_SSM']:

        Dw = abs(vws - cs)/vws
        S  = Sf_shape_sw(s, model=model_shape, Dw=Dw, a_sw=a_sw, b_sw=b_sw, c_sw=c_sw,
                         alp1_sw=alp1_sw, alp2_sw=alp2_sw)
        try:    mu = np.trapezoid(S, np.log(s), axis=0)
        except: mu = np.trapz(S, np.log(s), axis=0)
        S  = S/mu
        S0 = np.zeros((len(s), len(vws), len(alphas)))
        for i in range(0, len(alphas)):
            S0[:, :, i] = S
        S  = S0

    elif model_shape in ['sw_LISA', 'sw_HLnew']:

        # in some models, Dw is required to be computed for the array
        # of vws and alphas
        Dw_2d = True
        if model_shape == 'sw_HLnew':
            if strength != 'weak' or interpolate_HL_shape: Dw_2d = False

        if Dw_2d:
            if not quiet: print('Computing sound-shell thickness')
            _, _, _, _, _, _, xi_shocks, _ = \
                            hb.compute_profiles_vws_multalp(alphas, vws=vws)

            Dw = np.zeros((len(vws), len(alphas)))
            for i in range(0, len(alphas)):
                Dw[:, i] = (xi_shocks[:, i] - np.minimum(vws, cs))/np.maximum(vws, cs)

        else: Dw = 0.

        S = Sf_shape_sw(s, model=model_shape, Dw=Dw, a_sw=a_sw, b_sw=b_sw, c_sw=c_sw,
                alp1_sw=alp1_sw, alp2_sw=alp2_sw, strength=strength,
                interpolate_HL=interpolate_HL_shape,
                bs_k1HL=bs_k1HL, bs_k2HL=bs_HL_eff, vws=vws, alphas=alphas, quiet=quiet,
                interpolate_HL_n3=interpolate_HL_n3, corrRs=corrRs, cs2=cs2)

        try:    mu = np.trapezoid(S, np.log(s), axis=0)
        except: mu = np.trapz(S, np.log(s), axis=0)
        S  = S/mu

        if not interpolate_HL_shape:
            if strength != 'weak':
                S, _, _ = np.meshgrid(S, vws, alphas, indexing='ij')

    else:
        print('Choose an available model for model_shape in OmGW_spec_sw')
        print('Available models are sw_LISA, sw_HL, sw_HLnew, sw_SSM, sw_LISAold')
        return 0

    OmGW  = np.zeros((len(s), len(vws), len(alphas), len(betas)))
    freqs = np.zeros((len(s), len(vws), len(betas)))
    for i in range(0, len(vws)):
        lf = hb.Rstar_beta(vws[i], cs2=cs2, corr=corrRs)/betas
        for l in range(0, len(betas)):
            # express freqs as f/H_ast (instead of f/R_ast)
            freqs[:, i, l] = s/lf[l]
            for j in range(0, len(alphas)):
                OmGW[:, i, j, l] = 3*ampl[i, j]*pref[i, j, l]*S[:, i, j]

    if redshift:

        freqs, OmGW = cGW.shift_OmGW_today(freqs, OmGW, g=gstar, gS=gS,
                                           T=T, h0=h0,  kk=False, Neff=Neff)

    if not mult_vws:
        if not mult_beta:
            freqs = freqs[:, 0, 0]
            if not mult_alpha: OmGW = OmGW[:, 0, 0, 0]
            else:              OmGW = OmGW[:, 0, :, 0]
        else:
            freqs = freqs[:, 0, :]
            if not mult_alpha: OmGW = OmGW[:, 0, 0, :]
            else:              OmGW = OmGW[:, 0, :, :]
    else:
        if not mult_beta:
            freqs = freqs[:, :, 0]
            if not mult_alpha: OmGW = OmGW[:, :, 0, 0]
            else:              OmGW = OmGW[:, :, :, 0]
        else:
            if not mult_alpha: OmGW = OmGW[:, :, 0, :]

    return freqs, OmGW

############################ TURBULENCE ############################

################# fit for the anisotropic stresses #################

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
        Pi    -- array of the anisotropic stresses spectrum
        fGW   -- maximum value of the function s * Pi that determines
                 the amplitude of the GW spectrum for MHD turbulence
        pimax -- maximum value of Pi when s = fGW
    """

    Pi    = an.smoothed_bPL(s, a=0, b=b + bPi, kpeak=fPi,
                            alp=alpPi, norm=False, alpha2=True,
                            dlogk=False)
    pimax = ((b + bPi)/(b + bPi - 1))**(-(b + bPi)/alpPi)
    fGW   = fPi/(b + bPi - 1)**(1/alpPi)

    return Pi, fGW, pimax

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

    A    = an.calA(a=a_turb - 1, b=b_turb + 1, alp=alp_turb, dlogk=False)
    C    = an.calC(a=a_turb - 1, b=b_turb + 1, alp=alp_turb, tp='vort',
                   dlogk=False)
    ampl = .5*C/A**2

    return ampl

def pref_GWB_turb(Oms=mod.Oms_ref, lf=mod.lf_ref):

    '''
    Dependence of the GW spectrum from turbulence on the turbulence length
    scale, defined as lf = 2pi/kf, where kf is the position of the spectral peak,
    and the fraction of turbulent to radiation energy density Oms.

    Reference is RoperPol:2023bqa, equation 9, based on RoperPol:2022iel,
    section II D.

    Also used in EPTA:2023xxk, Caprini:2024hue
    See further details in RoperPol:2025b

    Arguments:
        Oms -- energy density of the source (i.e., 1/2 vrms^2)
        lf  -- mean-size of the bubbles, given as a fraction of the Hubble radius
    '''

    mult_Oms = isinstance(Oms, (list, tuple, np.ndarray))
    mult_lf  = isinstance(lf,  (list, tuple, np.ndarray))

    # if not mult_Oms: Oms = [Oms]
    # if not mult_lf:  lf  = [lf]
    Oms, lf = np.meshgrid(Oms, lf, indexing='ij')
    pref    = (Oms*lf)**2

    if   not mult_Oms and not mult_lf: pref = pref[0, 0]
    elif not mult_Oms: pref = pref[0, :]
    elif not mult_lf:  pref = pref[:, 0]

    return pref

def Sf_shape_turb(s, Oms=mod.Oms_ref, lf=mod.lf_ref, N=mod.N_turb, cs2=hb.cs2_ref,
                  expansion=True, tdecay=tdecay_ref, tp='magnetic', b_turb=b_turb,
                  alpPi=alpPi, fPi=fPi, bPi=bPi_vort):

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
        Oms    -- energy density of the source (i.e., 1/2 vrms^2)
        lf     -- characteristic scale of the turbulence as a
                  fraction of the Hubble radius, R* H*
        N      -- relation between the decay time and the effective
                  source duration
        cs2    -- square of the speed of sound (default is 1/3)
        tdecay -- determines the finite duration in the cit model
                  (default is to use eddy turnover time)
        tp     -- type of source (default is 'magnetic', other options are
                  'kinetic' or 'max') used to compute the characteristic
                  velocity in the eddy turnover time
        b_turb -- slope of the velocity/magnetic field spectrum in the UV
        alpPi, fPi, bPi -- parameters of the pPi_fit

    Returns:
        S      -- spectral shape
    """

    mult_Oms = isinstance(Oms, (list, tuple, np.ndarray))
    mult_lf  = isinstance(lf,  (list, tuple, np.ndarray))

    if not mult_Oms: Oms = [Oms]
    if not mult_lf:  lf  = [lf]

    TGW = mod.TGW_func(s, Oms=Oms, lf=lf, N=N, cs2=cs2, expansion=expansion,
                       tdecay=tdecay, tp=tp)

    Pi, _, _ = pPi_fit(s, b=b_turb, alpPi=alpPi, fPi=fPi, bPi=bPi)
    s3Pi  = s**3*Pi
    s3Pi, Oms, lf = np.meshgrid(s3Pi, Oms, lf, indexing='ij')
    S = s3Pi/lf**2*TGW

    if   not mult_Oms and not mult_lf: S = S[:, 0, 0]
    elif not mult_Oms: S = S[:, 0, :]
    elif not mult_lf:  S = S[:, :, 0]

    return S

def OmGW_spec_turb(s, Oms, lfs, N=mod.N_turb, cs2=hb.cs2_ref, quiet=True, a_turb=a_turb,
                   b_turb=b_turb, alp=alp_turb, expansion=True, tdecay=tdecay_ref,
                   tp='magnetic', alpPi=alpPi, fPi=fPi, bPi=bPi_vort, redshift=False,
                   gstar=co.gref, gS=0, T=co.Tref, h0=1., Neff=co.Neff_ref):

    '''
    Function that computes the GW spectrum (normalized to radiation
    energy density within RD era) for turbulence.

    The general shape of the GW spectrum is based on that of reference
    RoperPol:2023bqa, equations 3 and 9:

        OmGW (f) = 3 * ampl_GWB * pref_GWB * S(f),

    where:

    - ampl_GWB is the efficiency of GW production by the specific source,
    - pref_GWB is the dependence of the GW amplitude on the source
      parameters (e.g. length scale and strength of the source),
    - S(f) is a normalized spectral shape, such that int S(f) d ln f = 1

    The GW spectrum as an observable at present time is then computed using

        OmGW0 (f) = OmGW x FGW0,

    where FGW0 is the redshift from the time of generation to present
    time, computed in cosmoGW.py that depends on the degrees of freedom at
    the time of generation.

    Arguments:
        s               -- normalized wave number, divided by the mean bubbles
                           size Rstar, s = f R*
        Oms             -- energy density of the source (i.e., 1/2 vrms^2)
        lf              -- characteristic scale of the turbulence as a
                           fraction of the Hubble radius, R* H*
        N               -- relation between the decay time and the effective
                           source duration
        cs2             -- square of the speed of sound
                           (default is 1/3 for radiation domination)
        quiet           -- option to avoid printing debugging info (default is True)
        a_turb, b_turb, -- slopes and smoothness of the turbulent
        alp_turb           source spectrum (either magnetic or kinetic), default
                           values are for a von Karman spectrum
        expansion       -- option to include the Universe expansion
                           (in radiation domination)
        tdecay          -- determines the finite duration in the cit model
                           (default is to use eddy turnover time)
        tp              -- type of source (default is 'magnetic', other options are
                           'kinetic' or 'max') used to compute the characteristic
                           velocity in the eddy turnover time
        alpPi, fPi, bPi -- parameters of the pPi_fit
        redshift        -- option to redshift the GW spectrum and frequencies to
                           present time (default is False)
        gstar           -- number of degrees of freedom (default is 100) when
                           redshift = True
        gS              -- number of adiabatic degrees of freedom (default is gstar)
                           when redshift = True
        T               -- temperature scale (default is 100 GeV) when redshift = True
        h0              -- value of the Hubble rate at present time (100 h0 Mpc/km/s)
                           (default is one)
        Neff            -- effective number of neutrino species (default is 3)

    Returns:
        freq            -- array of frequencies (normalized with Rstar if redshift
                           is False and in Hz otherwise)
        OmGW            -- GW spectrum normalized to the radiation energy density
                           (or to the present critical density if redshift is True)
    '''

    mult_Oms = isinstance(Oms, (list, tuple, np.ndarray))
    mult_lfs = isinstance(lfs,  (list, tuple, np.ndarray))

    if not mult_Oms: Oms = np.array([Oms])
    if not mult_lfs: lfs = np.array([lfs])

    #### Computing ampl_GWB

    ampl = ampl_GWB_turb(a_turb=a_turb, b_turb=b_turb, alp=alp_turb)

    #### Computing pref_GWB

    pref = pref_GWB_turb(Oms=Oms, lf=lfs)

    #### Computing the spectral shape

    S = Sf_shape_turb(s, Oms=Oms, lf=lfs, N=N, cs2=cs2, expansion=expansion,
                      tdecay=tdecay, tp=tp, b_turb=b_turb, alpPi=alpPi,
                      fPi=fPi, bPi=bPi)

    OmGW  = np.zeros((len(s), len(Oms), len(lfs)))
    freqs = np.zeros((len(s), len(lfs)))

    for l in range(0, len(lfs)):
        # express freqs as f/H_ast (instead of f/R_ast)
        freqs[:, l] = s/lfs[l]
        for j in range(0, len(Oms)):
            OmGW[:, j, l] = 3*ampl*pref[j, l]*S[:, j, l]

    if redshift:
        freqs, OmGW = cGW.shift_OmGW_today(freqs, OmGW, g=gstar, gS=gS,
                                           T=T, h0=h0,  kk=False, Neff=Neff)

    if not mult_lfs:
        freqs = freqs[:, 0]
        if not mult_Oms: OmGW = OmGW[:, 0, 0]
        else:            OmGW = OmGW[:, :, 0]
    else:
        if not mult_Oms: OmGW = OmGW[:, 0, :]

    return freqs, OmGW

def OmGW_spec_turb_alphabeta(s, alphas, betas, vws=1., eps_turb=1., model_K0='Espinosa',
                             bs_HL_eff = 20, N=mod.N_turb, cs2=hb.cs2_ref, corrRs=True,
                             quiet=True, a_turb=a_turb, b_turb=b_turb, alp=alp_turb,
                             expansion=True, tdecay=tdecay_ref, tp='both', alpPi=alpPi,
                             fPi=fPi, bPi=bPi_vort, redshift=False, gstar=co.gref, gS=0,
                             T=co.Tref, h0=1., Neff=co.Neff_ref):

    '''
    Function that computes the GW spectrum (normalized to radiation
    energy density within RD era) for turbulence using OmGW_spec_turb and
    considering alpha, beta (i.e., the parameters of the phase transition)
    as inputs, following the description of RoperPol:2023bqa.

    It is assumed that turbulence has two contributions (from velocity
    and magnetic fields), which are in equipartition,

        Oms = Om_v + Om_B --> Om_v = Om_B = .5 Oms

    It takes the kinetic energy density from the PT, K = rho_kin/rho_total
    and assumes that the turbulence energy density is a fraction eps_turb of it,

        Oms = eps_turb * K

    The duration of the GW production dtfin is assumed to be

        dtfin = N_turb * lf / u*,

    where u* is a characteristic velocity,

        u* = sqrt(max(Omv, 2/(1 + cs2) OmB))

    Arguments are the same as the function OmGW_spec_turb and additional ones are:
        alphas    -- strength of the phase transition
        betas     -- rate of nucleation of the phase transition
        eps_turb  -- fraction of kinetic energy converted to turbulence
        vws       -- array of wall velocities
        corrRs    -- option to correct the ratio R* beta with max(vw, cs)
        model_K0  -- model to compute the kinetic energy ratio K0
                     (default is 'Espinosa' to use the the single-bubble value,
                     other option is 'higgsless')
        bs_HL_eff -- box size of the Higgsless simulations used to interpolate
                     the values of Omega tilde when model_efficiency='higgsless'
                     (default if L/vw = 20)

    Returns:
        freq      -- array of frequencies (normalized with Rstar if redshift
                     is False and in Hz otherwise)
        OmGW      -- GW spectrum normalized to the radiation energy density
                     (or to the present critical density if redshift is True)
    '''

    mult_alpha = isinstance(alphas, (list, tuple, np.ndarray))
    mult_beta  = isinstance(betas,  (list, tuple, np.ndarray))
    mult_vws   = isinstance(vws,    (list, tuple, np.ndarray))

    if not mult_alpha: alphas = np.array([alphas])
    if not mult_vws:   vws    = np.array([vws])
    if not mult_beta:  betas  = np.array([betas])

    if model_K0 == 'Espinosa':

        # compute kappa, K and Oms following the bag equation of
        # state as in Espinosa:2010hh

        # kappa: efficiency in converting vacuum to kinetic energy
        # K = rho_kin/rho_total = kappa alpha/(1 + alpha)
        # Oms_sw = v_f^2 = kappa alpha/(1 + cs2)
        kap    = hb.kappas_Esp(vws, alphas, cs2=cs2)
        K      = kap*alphas/(1 + alphas)

    elif model_K0 == 'higgsless':

        # compute K and Oms directly from the numerical results of the
        # Higgsless simulations of Caprini:2024gyk and interpolate to
        # values of alpha and vws

        dirr    = cGW.HOME + 'resources/higgsless/parameters_fit_sims.csv'
        df      = pd.read_csv(dirr)
        val_str = 'curly_K_0_512'
        K       = interpolate_HL_vals(df, vws, alphas, quiet=quiet,
                                      value=val_str, boxsize=bs_HL_eff)

    # amplitude used to determine the duration of the source,
    # assuming equipartition
    Oms = .5*K*eps_turb
    # length scale
    lf  = hb.Rstar_beta(vws=vws, corr=corrRs)

    OmGW  = np.zeros((len(s), len(vws), len(alphas), len(betas)))
    freqs = np.zeros((len(s), len(vws), len(betas)))
    if redshift:
        import astropy.units as u
        freqs *= u.Hz

    for i in range(0, len(vws)):
        for l in range(0, len(betas)):
            lf_ij = lf[i]/betas[l]
            for j in range(0, len(alphas)):
                freqs[:, i, l], OmGW[:, i, j, l] = \
                    OmGW_spec_turb(s, Oms[i, j], lf_ij, N=N, cs2=cs2, quiet=quiet,
                                   a_turb=a_turb, b_turb=b_turb, alp=alp_turb,
                                   expansion=expansion, tdecay=tdecay, tp=tp,
                                   alpPi=alpPi, fPi=fPi, bPi=bPi, redshift=redshift,
                                   gstar=gstar, gS=gS, T=T, h0=h0, Neff=Neff)

    if not mult_vws:
        if not mult_beta:
            freqs = freqs[:, 0, 0]
            if not mult_alpha: OmGW = OmGW[:, 0, 0, 0]
            else:              OmGW = OmGW[:, 0, :, 0]
        else:
            freqs = freqs[:, 0, :]
            if not mult_alpha: OmGW = OmGW[:, 0, 0, :]
            else:              OmGW = OmGW[:, 0, :, :]
    else:
        if not mult_beta:
            freqs = freqs[:, :, 0]
            if not mult_alpha: OmGW = OmGW[:, :, 0, 0]
            else:              OmGW = OmGW[:, :, :, 0]
        else:
            if not mult_alpha: OmGW = OmGW[:, :, 0, :]

    # multiply by 4 to take into account contribution from both velocity and
    # magnetic fields in the amplitude OmGW ~ Oms^2 = (2 * OmB)^2
    return freqs, 4.*OmGW