"""
GW_templates.py is a Python routine that contains analytical and semi-analytical
templates of cosmological GW backgrounds, usually based on spectral fits,
either from GW models (see GW_models) or from numerical simulations

Author: Alberto Roper Pol
Created: 01/12/2022
Updated: 31/08/2024

Main references are:

RPCNS22 - A. Roper Pol, C. Caprini, A. Neronov, D. Semikoz, "The gravitational wave
signal from primordial magnetic fields in the Pulsar Timing Array frequency band,"
Phys. Rev. D 105, 123502 (2022), arXiv:2201.05630

RPNCBS23 - A. Roper Pol, A. Neronov, C. Caprini, T. Boyer, D. Semikoz, "LISA and γ-ray telescopes as
multi-messenger probes of a first-order cosmological phase transition," arXiv:2307.10744 (2023)

RPPC23 - A. Roper Pol, S. Procacci, C. Caprini, "Characterization of the gravitational wave spectrum
from sound waves within the sound shell model," arXiv:2308.12943

HH19 - M. Hindmarsh, M. Hijazi, "Gravitational waves from first order
cosmological phase transitions in the Sound Shell Model,"
JCAP 12 (2019) 062, arXiv:1909.10040

HLLP21 - M. Hindmarsh, M. Lueben, J. Lumma, M. Pauly, "Phase transitions in the early universe,"
SciPost Phys. Lect. Notes 24 (2021), 1, arXiv:2008.09136

HHRW17 - M. Hindmarsh, S. J. Huber, K. Rummukainen, D. J. Weir, "Shape of the acoustic gravitational wave
power spectrum from a first order phase transition," Phys.Rev.D 96 (2017) 10, 103520, Phys.Rev.D 101 (2020) 8,
089902 (erratum), arXiv:1704.05871

JKRS23 - R. Jinno, T. Konstandin, H. Rubira, I. Stomberg, "Higgsless simulations of cosmological phase
transitions and gravitational waves," JCAP 02, 011 (2023), arxiv:2209.04369
"""

import numpy as np
import matplotlib.pyplot as plt
import plot_sets

import os
HOME = os.getcwd()

### Reference values
cs2 = 1/3      # speed of sound

### Reference values for turbulence template (based on RPNCBS23 and RPCNS22)
a_turb = 4         # Batchelor spectrum k^4
b_turb = 5/3       # Kolmogorov spectrum k^(-5/3)
alp_turb = 6/17    # von Karman smoothness parameter, see RPNCBS23
alpPi = 2.15       # smoothness parameter for the anisotropic stresses obtained for a von Karman spectrum
fPi = 2.2          # break frequency of the anisotropic stresses obtained for a von Karman spectrum
N_turb = 2         # ratio between the effective time duration of the source and the eddy turnover time,
                   # based on the simulations of RPCNS22, used in RPNCBS23
    
### Reference values for sound waves template
Omgwtilde_sw = 1e-2   # normalized amplitude, based on the simulations of HHRW17
a_sw = 3              # low frequency slope f^3 found for GWs in the HL simulations, see JKRS23
b_sw = 1              # intermediate frequency slope f found for GWs in the HL simulations, see JKRS23
c_sw = 3              # high frequency slope f^(-3) found for GWs in the HL simulations, see JKRS23
alp1_sw = 1.5         # first peak smoothness fit for sound waves found in RPNCBS23
alp2_sw = 0.5         # second peak smoothness fit for sound waves found in RPNCBS23

######################### GWB TEMPLATES FOR SOUND WAVES AND TURBULENCE #########################

'''
Main reference: RPNCBS23

Turbulence template is based in the constant-in-time model developed in RPCNS22
Sound waves template is based in the sound-shell model presented in HH19, and a double
broken power law fit presented in RPNCBS23 adapted from the numerical results of JKRS23
'''

####
#### The general shape of the GW spectrum is based on that of reference RPNCBS23, equations 3 and 9:
####
#### OmGW (f) = 3 * ampl_GWB * pref_GWB * FGW0 * S(f),
####
#### where ampl_GWB is the efficiency of GW production by the specific source (sound waves or turbulence),
#### pref_GWB is the dependence of the GW amplitude on the characteristic size and energy density,
#### FGW0 is the redshift from the time of generation to present time, and S(f) is a normalized spectral
#### function such that its value at the peak is one.
####

####################### specific functions for turbulence templates #######################

#### function obtained from integrating the Green's function in the constant-in-time model
#### used for vortical turbulence

def TGW_func(s, N=N_turb, Oms=.1, lf=1, cs2=cs2, multi=False):
    
    """
    Function that computes the logarithmic function obtained as the envelope of the GW template
    in the constant-in-time assumption for the unequal time correlator of the turbulent
    stresses.
    
    Reference is RPNCBS23, equation 15, based on RPCNS22, equation 24.
    
    Arguments:
        s -- array of frequencies, normalized by the characteristic scale, s = f R*
        N -- relation between eddy turnover time and effective source duration
        Oms -- energy density of the source
        lf -- characteristic scale of the turbulence as a fraction of the Hubble radius, R* H*
        cs2 -- square of the speed of sound (default is 1/3 for radiation domination)
        multi -- option to use arrays for Oms and lf if multi is set to True
        """

    ## characteristic velocity (for example, Alfven velocity)
    vA = np.sqrt(2*Oms/(1 + cs2))
    ## effective duration of the source (divided by R_*)
    dtfin = N/vA
    
    if multi:
    
        s_ij, lf_ij, Oms_ij = np.meshgrid(s, lf, Oms, indexing='ij')
        TGW1 = np.log(1 + lf_ij/2/np.pi/s_ij)**2

        lf_ij, dtfin_ij = np.meshgrid(lf, dtfin, indexing='ij')
        TGW0 = np.log(1 + dtfin_ij*lf_ij/2/np.pi)**2

        TGW = np.zeros((len(s), len(lf), len(Oms)))
        for i in range(0, len(dtfin)):
            TGW[s < 1/dtfin[i], :, i] = TGW0[:, i]
            TGW[s >= 1/dtfin[i], :, i] = TGW1[s >= 1/dtfin[i], :, i]
    else:
        
        TGW = np.zeros(len(s))
        TGW[s < 1/dtfin] = np.log(1 + lf*dtfin/2/np.pi)**2*s[(s < 1/dtfin)]**0
        TGW[s >= 1/dtfin] = np.log(1 + lf/2/np.pi/s[np.where(s >= 1/dtfin)])**2
    
    return TGW

##### fit of the anisotropic stresses based on a broken-power law

def pPi_fit(s, b=b_turb, alpPi=alpPi, fPi=fPi):
    
    """
    Function that computes the fit for the anisotropic stress spectrum that is valid
    for a von Karman velocity or magnetic spectrum.

    Reference is RPNCBS23, equation 17. It assumes that the anisotropic stresses in turbulence can
    be expressed with the following fit:

    p_Pi = (1 + (f/fPi)^alpPi)^(-(b + 2)/alpPi)

    Arguments:
        s -- array of frequencies, normalized by the characteristic scale, s = f R*
        b -- high-k slope k^(-b)
        alpPi -- smoothness parameter of the fit
        fPi -- position of the fit break

    Returns:
        Pi -- array of the anisotropic stresses spectrum
        fGW -- maximum value of the function s * Pi that determines the amplitude of the
               GW spectrum for MHD turbulence
        pimax -- maximum value of Pi when s = fGW
    """

    Pi = (1 + (s/fPi)**alpPi)**(-(b + 2)/alpPi)
    pimax = ((b + 2)/(b + 1))**(-(b + 2)/alpPi)
    fGW = fPi/(b + 1)**(1/alpPi)

    return Pi, fGW, pimax

######################### general template for sound waves and turbulence ##########################

def ampl_GWB(xiw=1., cs2=cs2, Omgwtilde_sw=Omgwtilde_sw, a_turb=a_turb,
             b_turb=b_turb, alp=alp_turb, alpPi=alpPi, fPi=fPi, tp='sw'):
    
    """
    Reference for sound waves is RPNCBS23, equation 3. Value of Omgwtilde ~ 1e-2 is based on HH19, HHRW17.
    Reference for turbulence is RPNCBS23, equation 9, based on the template of RPCNS22, section 3 D.

    See footnote 3 of RPNCBS23 for clarification (extra factor 1/2 has been added to take into account average over
    oscillations that were ignored in RPCNS22).

    Arguments:
        xiw -- wall velocity
        cs2 -- square of the speed of sound (default is 1/3 for radiation domination)
        Omgwtilde_sw -- efficiency of GW production from sound waves (default value is 1e-2,
                        based on numerical simulations)
        comp_mu -- option to compute mu for SSM and correct amplitude (default is False)
        a_turb, b_turb, alp_turb -- slopes and smoothness of the turbulent source spectrum
                                    (either magnetic or kinetic), default values are for a
                                    von Karman spectrum
        alpPi, fPi -- parameters of the fit of the spectral anisotropic stresses for turbulence
    """

    if tp == 'sw':

        ampl = Omgwtile

    if tp == 'turb':

        A = an.calA(a=a_turb, b=b_turb, alp=alp_turb)
        C = an.calC(a=a_turb, b=b_turb, alp=alp_turb, tp='vort')

        # use fit pPi_fit for anisotropic stresses (valid for von Karman
        # spectrum)
        _, fGW, pimax = pPi_fit(1, b=b_turb, alpPi=alpPi, fPi=fPi)
        ampl = .5*C/A**2*fGW**3*pimax

    return ampl
    
def pref_GWB(Oms=.1, lf=1., tp='sw', b_turb=b_turb, alpPi=alpPi, fPi=fPi):
    
    '''
    Dependence of the GW spectrum from sound waves and turbulence on the mean
    size of the bubbles lf = R* H_* and the kinetic energy density Oms.
    
    Reference for sound waves is RPNCBS23, equation 3, based on HLLP21, equation 8.24.
    Reference for turbulence is RPNCBS23, equation 9, based on RPCNS22, section II D.
    
    Arguments:
        Oms -- kinetic energy density
        lf -- mean-size of the bubbles, given as a fraction of the Hubble radius
        tp -- type of background ('sw' correspond to sound waves and 'turb' to vortical
              turbulence)
        b_turb -- high frequency slope of the turbulence spectrum (default is 5/3)
        alpPi, fPi -- parameters of the fit of the spectral anisotropic stresses for turbulence
    '''
    
    pref = (Oms*lf)**2
    if tp == 'sw': pref = pref/(np.sqrt(Oms) + lf)
    if tp == 'turb':
        # use fit pPi_fit for anisotropic stresses (valid for von Karman
        # spectrum)
        _, fGW, pimax = pPi_fit(1, b=b_turb, alpPi=alpPi, fPi=fPi)
        pref = pref/lf**2*np.log(1 + lf/2/np.pi/fGW)**2

    return pref

def Sf_shape(s, tp='turb', Dw=1, a_sw=a_sw, b_sw=b_sw, c_sw=c_sw, alp1_sw=alp1_sw,
             alp2_sw=alp2_sw, b_turb=b_turb, N=N_turb, Oms=.1, lf=1., alpPi=alpPi, fPi=fPi,
             ref='f', cs2=cs2, multi=False):

    """
    Function that computes the spectral shape derived for GWs generated by sound waves
    or MHD turbulence.
    
    Reference for sound waves based on Sound Shell Model (SSM) is RPNCBS23, equation 6,
    based on the results presented in HH19, equation 5.7.
    
    Reference for sound waves based on Higgsless (HL) simulations is RPNCBS23, equation 7,
    based on the results presented in JKRS23.
    
    Reference for vortical (MHD) turbulence is RPNCBS23, equation 9, based on the analytical
    model presented in RPCNS22, section II D.
    
    Arguments:
         s -- normalized wave number, divided by the mean bubbles size Rstar, s = f R*
         tp -- type of GW source (options are sw_SSM for the sound shell model of HH19,
               'sw_HL' for the fit based on the Higgsless simulations of JKRS23, and
               'turb' for MHD turbulence)
        Dw -- ratio between peak frequencies, determined by the shell thickness
        a_sw, b_sw, c_sw -- slopes for sound wave template, used when tp = 'sw_HL'
        alp1_sw, alp2_sw -- transition parameters for sound wave template, used when tp = 'sw_HL'
    
    Returns:
        spec -- spectral shape, normalized such that S = 1 at its peak
    """
    
    if tp == 'sw_SSM':

        s2 = s*Dw
        m = (9*Dw**4 + 1)/(Dw**4 + 1)
        M1 = ((Dw**4 + 1)/(Dw**4 + s2**4))**2
        M2 = (5/(5 - m + m*s2**2))**(5/2)
        S =  M1*M2*s2**9
        
    if tp == 'sw_HL':

        A = 16*(1 + Dw**(-3))**(2/3)*Dw**3
        S = an.smoothed_double_bPL(s, 1, np.sqrt(3)/Dw, A=A, a=a_sw, b=b_sw, c=c_sw, alp1=alp1_sw,
                                   alp2=alp2_sw, kref=1.)

    if tp == 'turb':

        TGW = TGW_func(s, N=N, Oms=Oms, lf=lf, cs2=cs2, multi=multi)
        Pi, fGW, pimax = pPi_fit(s, b=b_turb, alpPi=alpPi, fPi=fPi)
        BB = 1/fGW**3/pimax/np.log(1 + lf/2/np.pi/fGW)**2
        s3Pi = s**3*Pi

        if multi:
            s3Pi, BB, _ = np.meshgrid(s3Pi, BB, Oms, indexing='ij')

        S = s3Pi*BB*TGW

    return S

def OmGW_spec(ss, alpha, beta, xiw=1., tp='turb', cs2=cs2, multi_ab=False, multi_xi=False,
              Omgwtilde_sw=Omgwtilde_sw, a_sw=a_sw, b_sw=b_sw, c_sw=c_sw, alp1_sw=alp1_sw, alp2_sw=alp2_sw,
              a_turb=a_turb, b_turb=b_turb, alp_turb=alp_turb, alpPi=alpPi, fPi=fPi,
              eps_turb=1., ref='f', corrRs=True):
    
    '''
    Function that computes the GW spectrum (normalized to radiation energy density within RD
    era) for sound waves and turbulence.

    It takes the form:

        OmGW = 3 * ampl_GWB * pref_GWB * Sf_shape,

    see ampl_GWB, pref_GWB, and Sf_shape functions for details and references.
    
    Arguments:
        ss -- normalized wave number, divided by the mean bubbles size Rstar, s = f R*
        alpha -- strength of the phase transition
        beta -- rate of nucleation of the phase transition
        xiw -- wall velocity
        tp -- type of GW source (options are sw_SSM for the sound shell model of HH19,
               'sw_HL' for the fit based on the Higgsless simulations of JKRS23, and
               'turb' for MHD turbulence)
        cs2 -- square of the speed of sound (default is 1/3 for radiation domination)
        multi_ab -- option to provide an array of values of alpha and beta as input
        multi_xi -- option to provide an array of values of xiw as input
        Omgwtilde_sw -- efficiency of GW production from sound waves (default value is 1e-2,
                        based on numerical simulations)
        a_sw, b_sw, c_sw -- slopes for sound wave template, used when tp = 'sw_HL'
        alp1_sw, alp2_sw -- transition parameters for sound wave template, used when tp = 'sw_HL'
        
        a_turb, b_turb, alp_turb -- slopes and smoothness of the turbulent source spectrum
                                    (either magnetic or kinetic), default values are for a
                                    von Karman spectrum
        alpPi, fPi -- parameters of the fit of the spectral anisotropic stresses for turbulence
        eps_turb -- fraction of energy density converted from sound waves into turbulence
    '''
    
    dir0 = os.getcwd()
    os.chdir(HOME)
    import hydro_bubbles as hb
    os.chdir(dir0)
    
    cs = np.sqrt(cs2)
    
    # input parameters
    kap = hb.kappas_Esp(xiw, alpha, cs2=cs2)
    if isinstance(xiw, (list, tuple, np.ndarray)):
        alpha, _ = np.meshgrid(alpha, xiw, indexing='ij')
        beta, vw = np.meshgrid(beta, xiw, indexing='ij')
    K = kap*alpha/(1 + alpha)

    Oms = K*eps_turb
    
    if corrRs:
        lf = (8*np.pi)**(1/3)*np.maximum(xiw, cs)/beta
    else:
        lf = (8*np.pi)**(1/3)*xiw/beta

    Dw = abs(xiw - cs)/xiw
    
    # amplitude factors
    if multi_ab and multi_xi:
        Oms, lf, xiw_ij = np.meshgrid(alpha, beta, xiw, indexing='ij')
        for i in range(0, len(beta)): Oms_ij[:, i, :] = Oms
        for i in range(0, len(alpha)): lf_ij[i, :, :] = lf
    elif multi_ab and not multi_xi:
        Oms, lf = np.meshgrid(Oms, lf, indexing='ij')
            
    preff = pref_GWB(Oms=Oms, lf=lf, tp=tp, b_turb=b_turb, alpPi=alpPi, fPi=fPi)
    
    ampl = ampl_GWB(tp=tp, cs2=cs2, Omgwtilde_sw=Omgwtilde_sw, a_turb=a_turb,
                    b_turb=b_turb, alp=alp_turb, alpPi=alpPi, fPi=fPi)

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

############################### NOT PUBLIC ON GITHUB ##########################

################################ Sound wave templates ################################

### obsolete
def Sf_sw_SSM(s, Dw=1):
    
    """
    Function that computes the spectral shape derived for GWs generated by sound waves
    according to the Sound Shell Model (SSM) found in HH19, equation 5.7.
    
    Used in reference RPNCBS23, equation 6.
    
    Arguments:
         s -- normalized wave number, divided by the mean bubbles size Rstar, s = f R*
         Dw -- ratio between peak frequencies, determined by the shell thickness
    
    Returns:
        spec -- spectral shape, normalized such that S = 1 at its peak
    """
    
    s2 = s*Dw
    m = (9*Dw**4 + 1)/(Dw**4 + 1)
    M1 = ((Dw**4 + 1)/(Dw**4 + s2**4))**2
    M2 = (5/(5 - m + m*s2**2))**(5/2)
    
    S =  M1*M2*s2**9
    
    return S

### obsolete
def Sf_sw_hl(s, Dw=1, a1=1.5, a2=0.5, n1=3, n2=1, n3=-3):
    
    """
    Function that computes the fit spectral shape generated by sound waves
    based on the Higgsless simulation results from JKRS23. It uses a smoothed
    double broken power template (see GW_analytical for details).
    
    Used in reference RPNCBS23, equation 7.
    
    Arguments:
         s -- normalized wave number, divided by the mean bubbles size Rstar, s = f R*
         Dw -- ratio between peak frequencies, determined by the shell thickness
    
    Returns:
        spec -- spectral shape, normalized such that S = 1 at its peak
    """
    
    ### to be tested if this is correct!
    s2 = s*Dw
    S = s**n1/(1 + s**(a1*(n1 - n2)))**(1/a1)/(-n3 + n2*f2**(a2*(n2 - n3)))**((1)/a2)
    
    #A = 16*(1 + Dw**(-3))**(2/3)*Dw**3
    #S = an.smoothed_double_bPL(s, 1, np.sqrt(3)/Dw, A=A, a=3, b=1, c=3, alp1=a1, alp2=a2, kref=1.)
    
    return S

#### obsolete (to be checked!)
def Sf_sw_hl2(f, Dw=1, a1=2, a2=4, n1=3, n2=1, n3=-3):
    
    """
    Function that computes the fit spectral shape generated by sound waves
    based on the Higgsless simulation results from R. Jinno, T. Konstandin,
    H. Rubira, I. Stomberg, "Higgsless simulations of cosmological phase
    transitions and gravitational waves," https://arxiv.org/abs/2209.04369
    from LISA CosWG template
    
    Dw refers to the ratio between peak frequencies, determined by
    the shell thickness.
    
    s is the normalized wave number by the mean size of nucleated
    bubbles.
    """
    
    f2 = f*Dw
    S = f**n1*(1 + f**a1)**((-n1+n2)/a1)*(1 + f2**a2)**((-n2 + n3)/a2)
    
    return S

## obsolete
def pref_sw(Oms, lf):
    
    '''
    Dependence of the GW spectrum from sound waves on the mean size of the bubbles lf = R* H_*
    and the kinetic energy density Oms.
    
    Reference is RPNCBS23, equation 3, based on HLLP21, equation 8.24.
    
    Arguments:
        Oms -- kinetic energy density
        lf -- mean-size of the bubbles, given as a fraction of the Hubble radius
    '''
    
    return (Oms*lf)**2/(np.sqrt(Oms) + lf)

## obsolete
def AA_soundwaves(xiw, cs2=cs2, Omgwtilde=1e-2):
    
    """
    Function that computes the amplitude of the sound wave template A = Omegatilde/mu_b,
    according to the Sound Shell Model of HH19, equation 5.8.
    
    Reference is RPNCBS23, equation 3. Value of Omgwtilde is based on HH19, HHRW17.
    
    Arguments:
        xiw -- wall velocity
        cs2 -- square of the speed of sound (default is 1/3 for radiation domination)
        Omgwtilde -- efficiency of GW production from sound waves (default value is 1e-2,
                     based on numerical simulations)
    """
    
    Dw = Delta_w(xiw=xiw, cs2=cs2)
    Dws, mu_b, _ = mu_vs_rb()
    mu = np.interp(Dw, Dws, mu_b)
    
    return Omgwtilde/mu

def OmGW_sws(ss, alpha, beta, xiw=1., cs2=cs2, tilOmGW=Omgwtilde_sw, alp1_sw=alp1_sw, alp2_sw=alp2_sw,
             multi_ab=False):
    
    '''
    Function that computes the GW spectrum (normalized to radiation energy density within RD
    era) for the sound shell model and the Higgsless fit templates for sound waves.
    '''
    
    # input parameters
    OmK = Oms_alpha(alpha, xiw=xiw, cs2=cs2, eps=1., multi=multi_ab)
    lf = beta_Rs(beta, xiw=xiw, cs2=cs2, multi=multi_ab)
    Dw = Delta_w(xiw=xiw)
    
    # spectral shapes
    #S_sw_SSM = Sf_sw_SSM(ss, Dw=Dw)
    S_sw_SSM = Sf_shape(ss, tp='sw_SSM', Dw=Dw)
    mu_SSM = np.trapz(S_sw_SSM, np.log(ss), axis=0)
    
    S_sw_hl = Sf_sw_hl(ss, Dw=Dw, a2=a2_hl, a1=a1_hl)
    mu_hl = np.trapz(S_sw_hl, np.log(ss), axis=0)
    
    if multi_ab:
        ss, _, _ = np.meshgrid(ss, lf, OmK, indexing='ij')
        S_sw_SSM, _, _ = np.meshgrid(S_sw_SSM, lf, OmK, indexing='ij')
        S_sw_hl, lf, OmK = np.meshgrid(S_sw_hl, lf, OmK, indexing='ij')

    # amplitude factors
    preff_sw = pref_sw(OmK, lf)
    # GW spectrum
    OmGW_sw_SSM = preff_sw*3*tilOmGW*S_sw_SSM/mu_SSM
    OmGW_sw_hl = preff_sw*3*tilOmGW*S_sw_hl/mu_hl
    
    return OmGW_sw_SSM, OmGW_sw_hl

################################ MHD turbulence template ################################

### obsolete (to be deleted)
def pref_MHDturb(Oms, lf):
    
    '''
    Prefix that appears in the GW spectrum for MHD turbulence due to the characteristic
    size and turbulent energy density.
    It gives an effective efficiency of the GW production
    '''
    
    return Oms**2*lf**2

### obsolete (to be deleted)
def AA_MHDturb(alpha=6/17, a=4, b=5/3, alpPi=2.15, fPi=2.2):
    
    """
    Function that computes the amplitude of the MHD turbulence
    template (1/8pi^2) A/C^2 based on the model developed in
    A. Roper Pol et al., "The gravitational wave signal from primordial
    magnetic fields in the Pulsar Timing Array frequency band,"
    https://arxiv.org/abs/2201.05630 (2022).
    
    The factor 1/2 has been added to take into account average over
    oscillations that were ignored in the original work.
    
    The factor 1/(4pi^2) has been added to define a normalized
    spectral shape function, such that its value at the peak is 1.
    """
    
    A = an.A_alpha(alp=alpha, a=a, b=b)
    C = an.C_alpha(alp=alpha, a=a, b=b)
    pimax = (11/8)**(-11/3/alpPi)
    fGW = fPi*(3/8)**(1/alpPi)
    
    return C/A**2/(8*np.pi**2)*pimax*fGW
    
# obsolete
def Sf_MHDturb(f, N=2, Oms=.1, lf=1, cs2=cs2, equip=False, ref='f', multi=False):
    
    if ref=='k': lf = 2*np.pi/lf
    
    Pi, fGW, pimax = pPi_fit(f)
    BB = 4*np.pi**2/pimax/fGW/lf**2
    
    TGW = TGW_func(f, N=N, Oms=Oms, lf=lf, cs2=cs2, equip=equip, multi=multi)
    f3Pi = f**3*Pi
    
    if multi:
        f3Pi, BB, _ = np.meshgrid(f3Pi, BB, Oms, indexing='ij')

    Sf = BB*f3Pi*TGW
    
    return Sf

# obsolete
def eddy_turnover(Om, k, s='mag'):
    
    """
    Function that computes the eddy turnover time.
    """
    
    if s == 'mag': v = mf.Alfven_velocity(Om)
    if s == 'kin': v = mf.Alfven_velocity(Om, w=0)
        
    te = 1/k/v
    
    return te

def OmGW_MHDturb(ss, alpha, beta, xiw=1., cs2=cs2, eps=1., a_turb=4, b_turb=5/3,
                 alpha_turb=6/17, alpPi=2.15, fPi=2.2, N_turb=2, equip=True, multi_ab=False):
    
    '''
    Function that computes the GW spectrum (normalized to radiation energy density within RD
    era) for the MHD turbulence template.
    '''
    
    # input parameters
    Oms = Oms_alpha(alpha, xiw=xiw, cs2=cs2, eps=eps, multi=multi_ab)
    lf = beta_Rs(beta, xiw=xiw, cs2=cs2, multi=multi_ab)
    Dw = Delta_w(xiw=xiw)
    # spectral shape
    S_mhd = Sf_MHDturb(ss, N=N_turb, Oms=Oms, lf=lf, cs2=cs2, equip=equip, multi=multi_ab)
    
    if multi_ab:
        ss, lf, Oms = np.meshgrid(ss, lf, Oms, indexing='ij')
    # amplitude factor
    preff_mhd = pref_MHDturb(Oms, lf)
    
    # GW spectrum
    AA_mhd = AA_MHDturb(alpha=alpha_turb, a=a_turb, b=b_turb,
                        alpPi=alpPi, fPi=fPi)
    OmGW_mhd = preff_mhd*3*AA_mhd*S_mhd
    
    return OmGW_mhd
    
def OmGW_sw_MHD(ss, alpha, beta, vw=1., w=cs2, eps=1., tilOmGW=1e-2, a_turb=4, b_turb=5/3,
                a2_hl=.6, a1_hl=1.5, alpha_turb=6/17, alpPi=2.15, fPi=2.2, N_turb=2, equip=True):
    
    '''
    Function that computes the GW spectrum (normalized to radiation energy density within RD
    era) for the MHD turbulence template and for the sound shell model and the Higgsless fit
    templates for sound waves.
    '''
    
    # input parameters
    OmK = Oms_alpha(alpha, xiw=vw, w=w, eps=1.)
    Oms = OmK*eps
    lf = beta_Rs(beta, vw=vw, w=w)
    Dw = Delta_w(vw=vw)
    
    # amplitude factors
    preff_mhd = pref_MHDturb(Oms, lf)
    preff_sw = pref_sw(Oms, lf)
    
    # spectral shapes
    S_mhd = Sf_MHDturb(ss, N=N_turb, Oms=Oms, lf=lf, w=w, equip=True)
    S_sw_SSM = Sf_sw_SSM(ss, rb=Dw)
    mu_SSM = np.trapz(S_sw_SSM, np.log(ss))
    S_sw_hl = Sf_sw_hl(ss, rb=Dw, a2=a2_hl, a1=a1_hl)
    mu_hl = np.trapz(S_sw_hl, np.log(ss))
    
    # GW spectrum
    AA_mhd = AA_MHDturb(alpha=alpha_turb, a=a_turb, b=b_turb,
                        alpPi=alpPi, fPi=fPi)
    OmGW_mhd = preff_mhd*3*AA_mhd*S_mhd
    OmGW_sw_SSM = preff_sw*3*tilOmGW*S_sw_SSM/mu_SSM
    OmGW_sw_hl = preff_sw*3*tilOmGW*S_sw_hl/mu_hl
    
    return OmGW_mhd, OmGW_sw_SSM, OmGW_sw_hl