"""
interferometry.py is a Python routine that computes the response and
sensitivity functions of interferometer space-based GW detectors, e.g., LISA and Taiji,
to the detection of SGWB (see tutorial on interferometry/interferometry.ipynb).

Adapted from the original interferometry in GW_turbulence
(https://github.com/AlbertoRoper/GW_turbulence),
created in May 2022

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/interferometry.py

Author:  Alberto Roper Pol
Created: 01/05/2022
Updated: 01/11/2023 (preparation of cosmoGW code, included tutorial)
Updated: 18/07/2025 (release cosmoGW 1.0: https://pypi.org/project/cosmoGW)

Main references are:

Caprini:2019pxz  - C. Caprini, D. Figueroa, R. Flauger, G. Nardini, M. Peloso,
M. Pieroni, A. Ricciardone, G. Tassinato, "Reconstructing the spectral shape
of a stochastic gravitational wave background with LISA," JCAP 11 (2019), 017,
arXiv:1906.09244

Schmitz:2020syl  - K. Schmitz "New Sensitivity Curves for Gravitational-Wave Signals from
Cosmological Phase Transitions," JHEP 01, 097 (2021), arXiv:2002.04615.

Orlando:2020oko  - G. Orlando, M. Pieroni, A. Ricciardone, "Measuring Parity Violation
in the Stochastic Gravitational Wave Background with the LISA-Taiji network,"
JCAP 03, 069 (2021), arXiv:2011.07059.

RoperPol:2021xnd - A. Roper Pol, S. Mandal, A. Brandenburg, T. Kahniashvili,
"Polarization of gravitational waves from helical MHD turbulent sources,"
JCAP 04 (2022), 019, arXiv:2107.05356.

RoperPol:2022iel - A. Roper Pol, C. Caprini, A. Neronov,
D. Semikoz, "The gravitational wave signal from primordial
magnetic fields in the Pulsar Timing Array frequency band,"
Phys. Rev. D 105, 123502 (2022), arXiv:2201.05630
"""

import astropy.constants as const
import astropy.units     as u
import numpy  as np
import pandas as pd
import cosmoGW.cosmology as co

from cosmoGW.cosmoGW import HOME
dir0 = HOME + 'resources/detectors_sensitivity/'

# Reference values for LISA and Taiji interferometers
L_LISA   = 2.5e6*u.km
P_LISA   = 15
A_LISA   = 3
L_Taiji  = 3e6*u.km
P_Taiji  = 8
A_Taiji  = 3
SNR_PLS  = 10
T_PLS    = 4
v_dipole = 1.23e-3

# Reference f and beta arrays used in the calculations
f_ref    = np.logspace(-4, 0, 5000)*u.Hz
beta_ref = np.linspace(-20, 20, 3000)

################## SENSITIVITIES AND NOISE POWER SPECTRAL DENSITY ##################

################### READING FUNCTIONS FROM FILES ON SENSITIVITY ###################

def read_response_LISA_Taiji(dir0=dir0, TDI=True, interf='LISA'):

    """
    Function that reads the response functions of LISA and Taiji from the files
    'dir0/LISA_response_f.csv' and 'dir0/Taiji_response_f.csv',
    generated and saved in the routine 'compute_response_LISA_Taiji'.

    TDI chanels are defined following RoperPol:2021xnd, appendix B.

    Arguments:
        dir0 -- directory where to save the results (default is 'detector_sensitivity')
        TDI  -- option to read the response functions for TDI chanels, instead of XYZ
                chanels (default True)
    """

    if TDI:
        dirr = dir0 + interf + '_response_f_TDI.csv'
        df   = pd.read_csv(dirr)
        fs   = np.array(df['frequency'])
        fs   = fs*u.Hz
        MAs  = np.array(df['MA'])
        MTs  = np.array(df['MT'])
        DAEs = np.array(df['DAE'])

    else:
        dirr = dir0 + interf + '_response_f_X.csv'
        df   = pd.read_csv(dirr)
        fs   = np.array(df['frequency'])
        fs   = fs*u.Hz
        MAs  = np.array(df['MX'])
        MTs  = np.array(df['MXY'])
        DAEs = np.array(df['DXY'])

    # Note that for interferometry channels we have MA -> MX, MT -> MXY, DAE -> DXY

    return fs, MAs, MTs, DAEs

def read_sens(dir0=dir0, SNR=SNR_PLS, T=T_PLS, interf='LISA', Xi=False,
              TDI=True, chan='A'):

    """
    Function that reads the sensitivity Omega (expressed as a GW energy
    density spectrum), and the power law sensitivity (PLS)
    of the interferometer chosen (options are 'LISA', 'Taiji', and the
    LISA-Taiji network 'comb').

    For LISA and Taiji the sensitivity can be given for each chanel X, Y,
    Z or on the TDI chanels A=E, T (A chanel is the default option, and
    the relevant for Omega sensitivity).

    Arguments:
        dir0   -- directory where the sensitivity files are stored
                  (default 'detector_sensitivity')
        SNR    -- signal-to-noise ratio (SNR) of the resulting PLS (default 10)
        T      -- duration of the mission (in years) of the resulting PLS
                  (default 4)
        interf -- option to chose the interferometer (default 'LISA', other
                  options are 'Taiji', 'comb', i.e., LISA + Taiji, and
                  'muAres')
        Xi     -- option to return the helical sensitivity and PLS (default False),
                  available when 'LISA' or 'Taiji' interf are chosen
        TDI    -- option to compute sensitivity in TDI chanels A and T
                  (default True)
        chan   -- specific chanel (default 'A', other options are 'X', 'Y', 'Z' if no TDI,
                  and 'T' if TDI), only available for LISA and Taiji

    Returns:
        fs           -- array of frequencies
        interf_Om    -- sensitivity of the interferometer 'interf'
                        expressed as a GW energy density spectrum
                        (can be given for different chanels, see above)
        interf_OmPLS -- PLS of the interferometer 'interf'
        interf_Xi    -- helical sensitivity of the interferometer 'interf'
        interf_XiPLS -- helical PLS of the interferometer 'interf'
    """

    fact = SNR/np.sqrt(T)

    if interf == 'LISA':

        fs, LISA_Om    = read_csv('LISA_Omega',    dir0=dir0)
        fs, LISA_OmPLS = read_csv('LISA_OmegaPLS', dir0=dir0)
        LISA_OmPLS    *= fact
        if Xi:
            fs, LISA_Xi    = read_csv('LISA_Xi',    dir0=dir0, b='Xi')
            fs, LISA_XiPLS = read_csv('LISA_XiPLS', dir0=dir0, b='Xi')
            LISA_XiPLS *= fact
            return fs, LISA_Om, LISA_OmPLS, LISA_Xi, LISA_XiPLS
        else: return fs, LISA_Om, LISA_OmPLS

    if interf == 'Taiji':

        fs, Taiji_Om    = read_csv('Taiji_Omega',    dir0=dir0)
        fs, Taiji_OmPLS = read_csv('Taiji_OmegaPLS', dir0=dir0)
        Taiji_OmPLS    *= fact
        if Xi:
            fs, Taiji_Xi    = read_csv('Taiji_Xi',    dir0=dir0, b='Xi')
            fs, Taiji_XiPLS = read_csv('Taiji_XiPLS', dir0=dir0, b='Xi')
            Taiji_XiPLS    *= fact
            return fs, Taiji_Om, Taiji_OmPLS, Taiji_Xi, Taiji_XiPLS
        else: return fs, Taiji_Om, Taiji_OmPLS

    if interf == 'comb':

        fs, LISA_Taiji_Xi    = read_csv('LISA_Taiji_Xi',    dir0=dir0, b='Xi')
        fs, LISA_Taiji_XiPLS = read_csv('LISA_Taiji_XiPLS', dir0=dir0, b='Xi')
        LISA_Taiji_XiPLS    *= fact
        return fs, LISA_Taiji_Xi, LISA_Taiji_XiPLS

    if interf == 'muAres':

        fs, muAres_Om    = read_csv('muAres_Omega',    dir0=dir0)
        fs, muAres_OmPLS = read_csv('muAres_OmegaPLS', dir0=dir0)
        muAres_OmPLS    *= fact
        return fs, muAres_Om, muAres_OmPLS

def read_csv(file, dir0=dir0, a='f', b='Omega'):

    """
    Function that reads a csv file with two arrays and returns them.

    Arguments:
        dir0 -- directory that contains the file (default is 'detector_sensitivity')
        file -- name of the csv file
        a -- identifier in pandas dataframe of first array (default 'f')
        b -- identifier in pandas dataframe of second array (default 'Omega')

    Returns:
        x -- first array of the file
        y -- second array of the file
    """

    dirr = dir0 + file + '.csv'
    df   = pd.read_csv(dirr)
    x    = np.array(df[a])
    y    = np.array(df[b])

    return x, y

def read_detector_PLIS_Schmitz(dir0=dir0 + '/power-law-integrated_sensitivities/',
                               det='BBO', SNR=SNR_PLS, T=T_PLS):

    """
    Read power law integrated senstivities from reference Schmitz:2020syl.

    Arguments:
        dir0 -- directory where the PLS are stored (default is
                'detector_sensitivity/power-law-integrated_sensitivities')
        det  -- GW detector (check available detectors in default directory)
        SNR  -- signal-to-noise ratio (SNR) of the resulting PLS (default 10)
        T    -- duration of the mission (in years) of the resulting PLS
                (default 4)
    """

    frac  = SNR/np.sqrt(T)
    dirr  = dir0 + 'plis_' + det + '.dat'
    BBO   = pd.read_csv(dirrr, header=14, delimiter='\t',
                        names=['f', 'Omega (log)', 'hc (log)', 'Sh (log)'])
    f     = 10**np.array(BBO['f'])
    Omega = 10**np.array(BBO['Omega (log)'])

    return f, Omega*frac

def read_MAC(dir0=dir0 + '/LISA_Taiji/', M='MAC', V='V'):

    """
    Function that reads the V response functions of the cross-correlated
    channels of the LISA-Taiji network.

    Reference is RoperPol:2021xnd, see figure 18. The data is taken from  Orlando:2020oko, see figure 2.

    Argument:
        dir0 -- directory where to save the results (default is
                'detector_sensitivity/LISA_Taiji/')
        M -- string of the channels to be read (options are default 'MAC',
             'MAD', 'MEC', 'MED')
        V -- Stokes parameter. It can be changed to read the 'I' response
             function (default 'V')
    """

    dirr = dir0 + M + '_' + V + '.csv'
    df   = pd.read_csv(dirr)
    f    = np.array(df['f'])
    MAC  = np.array(df['M'])
    inds = np.argsort(f)
    f    = f[inds]
    MAC  = MAC[inds]
    f    = f*u.Hz

    return f, MAC

def read_all_MAC(V='V'):

    """
    Function that reads all relevant TDI cross-correlated response functions between LISA
    and Taiji (AC, AD, EC, ED channels) using read_MAC.

    Reference is RoperPol:2021xnd, see figure 18. The data is taken from  Orlando:2020oko, see figure 2.

    Argument:
         V -- Stokes parameter. It can be changed to read the 'I' response
             function (default 'V')
    """

    f_AC, M_AC = read_MAC(M='MAC', V=V)
    f_AD, M_AD = read_MAC(M='MAD', V=V)
    f_EC, M_EC = read_MAC(M='MEC', V=V)
    f_ED, M_ED = read_MAC(M='MED', V=V)

    min_f = np.max([np.min(f_AC.value), np.min(f_AD.value),
                    np.min(f_EC.value), np.min(f_ED.value)])
    max_f = np.min([np.max(f_AC.value), np.max(f_AD.value),
                    np.max(f_EC.value), np.max(f_ED.value)])

    fs   = np.logspace(np.log10(min_f), np.log10(max_f), 1000)*u.Hz
    M_AC = np.interp(fs, f_AC, M_AC)*2
    M_AD = np.interp(fs, f_AD, M_AD)*2
    M_EC = np.interp(fs, f_EC, M_EC)*2
    M_ED = np.interp(fs, f_ED, M_ED)*2

    return fs, M_AC, M_AD, M_EC, M_ED

####### NOISE POWER SPECTRAL DENSITY FUNCTIONS FOR SPACE-BASED INTERFEROMETERS

def Poms_f(f=f_ref, P=P_LISA, L=L_LISA):

    """
    Function that computes the power spectral density (PSD) of the optical
    metrology system (oms) noise.

    Reference is RoperPol:2021xnd, equation B.24.

    Arguments:
        f -- frequency array (should be in units of Hz)
        P -- noise parameter (default 15 for LISA)
             value for Taiji is P = 8
        L -- length of the interferometer arm (default is L = 2.5e6 km for LISA)

    Returns:
        Poms -- oms PSD noise
    """

    f_mHz = f.to(u.mHz)
    L_pm  = L.to(u.pm)
    Poms  = P**2/L_pm.value**2*(1 + (2/f_mHz.value)**4)/u.Hz

    return Poms

def Pacc_f(f=f_ref, A=A_LISA, L=L_LISA):

    """
    Function that computes the power spectral density (PSD) of the mass
    acceleration noise.

    Reference is RoperPol:2021xnd, equation B.25.

    Arguments:
        f -- frequency array (should be in units of Hz)
        A -- noise parameter (default 3 for LISA)
             value for Taiji is A = 3
        L -- length of the interferometer arm (default is L = 2.5e6 km for LISA)

    Returns:
        Pacc -- mass acceleration PSD noise
    """

    f_mHz = f.to(u.mHz)
    L_fm  = L.to(u.fm)
    c     = const.c
    Loc   = L/c
    Loc   = Loc.to(u.s)

    fsinv = (c/2/np.pi/f/L)
    fsinv = fsinv.to(1)

    Pacc  = A**2*Loc.value**4/L_fm.value**2*(1 + (.4/f_mHz.value)**2)*(1 + \
            (f_mHz.value/8)**4)*fsinv.value**4/u.Hz

    return Pacc

def Pn_f(f=f_ref, P=P_LISA, A=A_LISA, L=L_LISA, TDI=True):

    """
    Function that computes the noise power spectral density (PSD) of an
    interferometer channel X, Pn(f), and of the cross-correlation of two
    different interferometer channels XY, Pn_cross(f). It gives the A and
    T PSD of the TDI chanels if TDI is True (default)

    Reference is RoperPol:2021xnd, equations B.23 and B.26.

    Arguments:
        f -- frequency array (should be in units of Hz)
        P -- noise parameter (default 15 for LISA)
        A -- noise parameter (default 3 for LISA)
        L -- length of the interferometer arm (default is L = 2.5e6 km for LISA,
             otherwise should be in units of km)

    Returns:
        Pn (noise PSD) and Pn_cross (cross-correlation noise PSD) if TDI is False,
            or PnA and PnT (noise PSD of the TDI channels A and T) if TDI is True
    """

    Poms = Poms_f(f=f, P=P, L=L)
    Pacc = Pacc_f(f=f, A=A, L=L)
    c    = const.c
    f0   = c/2/np.pi/L
    f_f0 = f.to(u.Hz)/f0.to(u.Hz)

    Pn       = Poms + (3 + np.cos(2*f_f0.value))*Pacc
    Pn_cross = -.5*np.cos(f_f0.value)*(Poms + 4*Pacc)

    if TDI:
        PnA = 2*(Pn - Pn_cross)/3
        PnT = (Pn + 2*Pn_cross)/3
        return PnA, PnT

    else:
        return Pn, Pn_cross

############################ INTERFEROMETRY CALCULATIONS ############################

##### ANALYTICAL FIT FOR LISA SENSITIVITY

def R_f(f=f_ref, L=L_LISA):

    """
    Function that computes the analytical fit of the response function.

    Reference is RoperPol:2021xnd, equation B.15.

    Arguments:
        f -- frequency array (should be in units of Hz)
        L -- length of the interferometer arm (default is L = 2.5e6 km for LISA,
             otherwise should be in units of km)

    Returns:
        Rf -- response function (using analytical fit)
    """

    c    = const.c
    f0   = c/2/np.pi/L
    f_f0 = f.to(u.Hz)/f0.to(u.Hz)
    Rf   = .3/(1 + .6*f_f0.value**2)

    return Rf

def Sn_f_analytical(f=f_ref, P=P_LISA, A=A_LISA, L=L_LISA):

    """
    Function that computes the strain sensitivity using the analytical fit
    for an interferometer channel X.

    Arguments:
        f  -- frequency array (should be in units of Hz)
        P  -- noise parameter (default 15 for LISA)
        A  -- noise parameter (default 3 for LISA)
        L  -- length of the interferometer arm (default is L = 2.5e6 km for LISA,
              otherwise should be in units of km)

    Returns:
        Rf -- response function (using analytical fit)
    """

    Pn = Pn_f(f=f, P=P, A=A, L=L)
    Rf = R_f(f=f,  L=L)

    return Pn/Rf

######################## NUMERICAL COMPUTATION OF RESPONSE FUNCTIONS ########################

def delta(a, b):

    """
    Function that returns the Kronecker delta delta(a, b).
    Returns 1 if a = b and 0 otherwise.
    """

    if a==b: return 1
    else:    return 0

def compute_interferometry(f=f_ref, L=L_LISA, TDI=True, order=1, comp_all=False,
                           comp_all_rel=True):

    """
    Function that computes numerically the monopole (or dipole if order = 2)
    response functions of an interferometer channel of a space-based GW detector
    using the interferometer channels (or TDI channels if TDI is True).

    Reference is RoperPol:2021xnd, appendix B (in particular, equations B.13 and B.16).

    Arguments:
        f            -- frequency array (should be in units of Hz)
        L            -- length of the interferometer arm
                        (default is L = 2.5e6 km for LISA, otherwise should be
                        in units of km)
        TDI          -- option to compute sensitivity in TDI chanels A and T
                        (default True), note that dipole is only computed for TDI.
        order        -- moment of the response function (default 1 corresponds
                        to monopole response functions, order = 2 corresponds
                        to dipole response function).
        comp_all     -- computes all response functions (monopole and dipole,
                        X and A channels), allows to compute everything faster
                        than rerunning the code for each case (default False)
        comp_all_rel -- computes only (and all) relevant response functions
                        (i.e., the ones that are not identically zero or equal
                        to each other due to the geometric symmetries
                        of the interferometer)

    Returns:
        MXY -- if order = 1, returns monopole response functions where
                X, Y can be X, Y, Z (not TDI), A, E, T (TDI)
        DXY -- if order = 2, returns dipole response functions where X, Y
               can be X, Y, Z (not TDI), A, E, T (TDI)
    """

    if comp_all_rel: comp_all = True

    c = const.c

    # integration over sky directions (theta, phi)
    theta = np.linspace(0, np.pi, 50)
    phi   = np.linspace(0, 2*np.pi, 50)

    # array of wave numbers
    k  = 2*np.pi*f/c
    kL = L*k
    kL = kL.to(1)

    kLij, th, ph = np.meshgrid(kL, theta, phi, indexing='ij')

    kx1 = 0
    kx2 = np.cos(th)
    kx3 = .5*(np.sqrt(3)*np.cos(ph)*np.sin(th) + np.cos(th))

    kU1 = kx2 - kx1
    kU2 = kx3 - kx2
    kU3 = kx1 - kx3

    # detector transfer functions (eq. B.3)
    TkU1   = np.exp(-1j*kLij*(1+kU1)/2)*np.sinc(kLij.value*(1 - kU1)/2/np.pi)
    TkU1  += np.exp(1j*kLij*(1-kU1)/2)*np.sinc(kLij.value*(1 + kU1)/2/np.pi)
    TkU2   = np.exp(-1j*kLij*(1+kU2)/2)*np.sinc(kLij.value*(1 - kU2)/2/np.pi)
    TkU2  += np.exp(1j*kLij*(1-kU2)/2)*np.sinc(kLij.value*(1 + kU2)/2/np.pi)
    TkU3   = np.exp(-1j*kLij*(1+kU3)/2)*np.sinc(kLij.value*(1 - kU3)/2/np.pi)
    TkU3  += np.exp(1j*kLij*(1-kU3)/2)*np.sinc(kLij.value*(1 + kU3)/2/np.pi)
    TkmU1  = np.exp(-1j*kLij*(1-kU1)/2)*np.sinc(kLij.value*(1 + kU1)/2/np.pi)
    TkmU1 += np.exp(1j*kLij*(1+kU1)/2)*np.sinc(kLij.value*(1 - kU1)/2/np.pi)
    TkmU2  = np.exp(-1j*kLij*(1-kU2)/2)*np.sinc(kLij.value*(1 + kU2)/2/np.pi)
    TkmU2 += np.exp(1j*kLij*(1+kU2)/2)*np.sinc(kLij.value*(1 - kU2)/2/np.pi)
    TkmU3  = np.exp(-1j*kLij*(1-kU3)/2)*np.sinc(kLij.value*(1 + kU3)/2/np.pi)
    TkmU3 += np.exp(1j*kLij*(1+kU3)/2)*np.sinc(kLij.value*(1 - kU3)/2/np.pi)

    U1 = np.array([0, 0, 1])
    U2 = .5*np.array([np.sqrt(3), 0, -1])
    U3 = -.5*np.array([np.sqrt(3), 0, 1])

    c = np.matrix([[2, -1, -1], [0, -np.sqrt(3), np.sqrt(3)],
                  [1,1,1]])/3

    # interferometer response functions (eqs. B.1 and B.9)
    if TDI or comp_all:
        QA = np.zeros((3, 3, len(kL), len(theta), len(phi)))*1j
        QE = np.zeros((3, 3, len(kL), len(theta), len(phi)))*1j
        QT = np.zeros((3, 3, len(kL), len(theta), len(phi)))*1j
    if not TDI or comp_all:
        QX = np.zeros((3, 3, len(kL), len(theta), len(phi)))*1j
        QY = np.zeros((3, 3, len(kL), len(theta), len(phi)))*1j
        if not comp_all_rel:
            QZ = np.zeros((3, 3, len(kL), len(theta), len(phi)))*1j
    for i in range(0, 3):
        for j in range(0,3):
            Q1 = .25*np.exp(-1j*kLij*kx1)*(TkU1*U1[i]*U1[j] - TkmU3*U3[i]*U3[j])
            Q2 = .25*np.exp(-1j*kLij*kx2)*(TkU2*U2[i]*U2[j] - TkmU1*U1[i]*U1[j])
            Q3 = .25*np.exp(-1j*kLij*kx3)*(TkU3*U3[i]*U3[j] - TkmU2*U2[i]*U2[j])
            if TDI or comp_all:
                QA[i,j,:,:,:] = Q1*c[0,0] + Q2*c[0,1] + Q3*c[0,2]
                QE[i,j,:,:,:] = Q1*c[1,0] + Q2*c[1,1] + Q3*c[1,2]
                QT[i,j,:,:,:] = Q1*c[2,0] + Q2*c[2,1] + Q3*c[2,2]
            if not TDI or comp_all:
                QX[i,j,:,:,:] = Q1
                QY[i,j,:,:,:] = Q2
                if not comp_all_rel: QZ[i,j,:,:,:] = Q3

    k1 = np.cos(ph)*np.sin(th)
    k2 = np.sin(ph)*np.sin(th)
    k3 = np.cos(th)

    # polarization tensors (eq. B.14)
    e1ab = np.zeros((3, 3, len(kL), len(theta), len(phi)))*1j
    for i in range(0, 3):
        if i==0:   ki=k1
        elif i==1: ki=k2
        else:      ki=k3
        for j in range(0, 3):
            if j==0:   kj=k1
            elif j==1: kj=k2
            else:      kj=k3
            e1ab[i,j,:,:,:] = delta(i,j) - ki*kj
            if i==0:
                if j==1:   e1ab[i,j,:,:,:] += -1j*k3
                elif j==2: e1ab[i,j,:,:,:] += 1j*k2
            elif i==1:
                if j==0:   e1ab[i,j,:,:,:] += 1j*k3
                elif j==2: e1ab[i,j,:,:,:] += -1j*k1
            else:
                if j==0:   e1ab[i,j,:,:,:] += -1j*k2
                elif j==1: e1ab[i,j,:,:,:] += 1j*k1

    if TDI or comp_all:
        if order == 1 and not comp_all:
            print('computing TDI monopole response functions')
        if order == 2 and not comp_all:
            print('computing TDI dipole response functions')
        if comp_all:
            print('computing TDI monopole and dipole response functions')
        FAA = np.zeros((len(kL), len(theta), len(phi)))*1j
        FAE = np.zeros((len(kL), len(theta), len(phi)))*1j
        FTT = np.zeros((len(kL), len(theta), len(phi)))*1j
        if not comp_all_rel:
            FEE = np.zeros((len(kL), len(theta), len(phi)))*1j
            FAT = np.zeros((len(kL), len(theta), len(phi)))*1j
            FET = np.zeros((len(kL), len(theta), len(phi)))*1j
    if not TDI or comp_all:
        if order == 1 and not comp_all:
            print('computing interferometer monopole response functions')
        if order == 2 and not comp_all:
            print('computing interferometer dipole response functions')
        if comp_all:
            print('computing interferometer monopole and dipole response functions')
        FXX = np.zeros((len(kL), len(theta), len(phi)))*1j
        FXY = np.zeros((len(kL), len(theta), len(phi)))*1j
        if not comp_all_rel:
            FYY = np.zeros((len(kL), len(theta), len(phi)))*1j
            FZZ = np.zeros((len(kL), len(theta), len(phi)))*1j
            FXZ = np.zeros((len(kL), len(theta), len(phi)))*1j
            FYZ = np.zeros((len(kL), len(theta), len(phi)))*1j

    for a in range(0, 3):
        for b in range(0, 3):
            for c in range(0, 3):
                for d in range(0, 3):
                    eabcd = .25*e1ab[a,c,:,:,:]*e1ab[b,d,:,:,:]
                    if TDI or comp_all:
                        FAA += eabcd*QA[a,b,:,:,:]*np.conjugate(QA[c,d,:,:,:])
                        FAE += eabcd*QA[a,b,:,:,:]*np.conjugate(QE[c,d,:,:,:])
                        FTT += eabcd*QT[a,b,:,:,:]*np.conjugate(QT[c,d,:,:,:])
                        if not comp_all_rel:
                            FEE += eabcd*QE[a,b,:,:,:]*np.conjugate(QE[c,d,:,:,:])
                            FAT += eabcd*QA[a,b,:,:,:]*np.conjugate(QT[c,d,:,:,:])
                            FET += eabcd*QE[a,b,:,:,:]*np.conjugate(QT[c,d,:,:,:])
                    if not TDI or comp_all:
                        FXX += eabcd*QX[a,b,:,:,:]*np.conjugate(QX[c,d,:,:,:])
                        FXY += eabcd*QX[a,b,:,:,:]*np.conjugate(QY[c,d,:,:,:])
                        if not comp_all_rel:
                            FYY += eabcd*QY[a,b,:,:,:]*np.conjugate(QY[c,d,:,:,:])
                            FZZ += eabcd*QZ[a,b,:,:,:]*np.conjugate(QZ[c,d,:,:,:])
                            FXZ += eabcd*QX[a,b,:,:,:]*np.conjugate(QZ[c,d,:,:,:])
                            FYZ += eabcd*QY[a,b,:,:,:]*np.conjugate(QZ[c,d,:,:,:])

    # Monopole (eq. B.13) and dipole (eq. B.16) response functions of LISA for TDI channels
    if TDI or comp_all:

        if order == 1 or comp_all:
            MAA1 = np.trapz(FAA*np.sin(th), th, axis=1)
            MAA  = np.trapz(MAA1, phi, axis=1)/np.pi
            MTT1 = np.trapz(FTT*np.sin(th), th, axis=1)
            MTT  = np.trapz(MTT1, phi, axis=1)/np.pi
            if not comp_all_rel:
                MEE1 = np.trapz(FEE*np.sin(th), th, axis=1)
                MEE  = np.trapz(MEE1, phi, axis=1)/np.pi
                MAE1 = np.trapz(FAE*np.sin(th), th, axis=1)
                MAE  = np.trapz(MAE1, phi, axis=1)/np.pi
                MAT1 = np.trapz(FAT*np.sin(th), th, axis=1)
                MAT  = np.trapz(MAT1, phi, axis=1)/np.pi
                MET1 = np.trapz(FET*np.sin(th), th, axis=1)
                MET  = np.trapz(MET1, phi, axis=1)/np.pi

            if not comp_all:
                return MAA, MEE, MTT, MAE, MAT, MET

        if order == 2 or comp_all:
            DAE1 = 1j*np.trapz(FAE*np.sin(th)**2*np.sin(ph), th, axis=1)
            DAE = np.trapz(DAE1, phi, axis=1)/np.pi
            if not comp_all_rel:
                DAA1 = 1j*np.trapz(FAA*np.sin(th)**2*np.sin(ph), th, axis=1)
                DAA  = np.trapz(DAA1, phi, axis=1)/np.pi
                DEE1 = 1j*np.trapz(FEE*np.sin(th)**2*np.sin(ph), th, axis=1)
                DEE  = np.trapz(DEE1, phi, axis=1)/np.pi
                DTT1 = 1j*np.trapz(FTT*np.sin(th)**2*np.sin(ph), th, axis=1)
                DTT  = np.trapz(DTT1, phi, axis=1)/np.pi
                DAT1 = 1j*np.trapz(FAT*np.sin(th)**2*np.sin(ph), th, axis=1)
                DAT  = np.trapz(DAT1, phi, axis=1)/np.pi
                DET1 = 1j*np.trapz(FET*np.sin(th)**2*np.sin(ph), th, axis=1)
                DET  = np.trapz(DET1, phi, axis=1)/np.pi

            if not comp_all:
                return DAA, DEE, DTT, DAE, DAT, DET

    if not TDI or comp_all:

        if order == 1 or comp_all:
            MXX1 = np.trapz(FXX*np.sin(th), th, axis=1)
            MXX  = np.trapz(MXX1, phi, axis=1)/np.pi
            MXY1 = np.trapz(FXY*np.sin(th), th, axis=1)
            MXY  = np.trapz(MXY1, phi, axis=1)/np.pi
            if not comp_all_rel:
                MYY1 = np.trapz(FYY*np.sin(th), th, axis=1)
                MYY  = np.trapz(MYY1, phi, axis=1)/np.pi
                MZZ1 = np.trapz(FZZ*np.sin(th), th, axis=1)
                MZZ  = np.trapz(MZZ1, phi, axis=1)/np.pi
                MXZ1 = np.trapz(FXZ*np.sin(th), th, axis=1)
                MXZ  = np.trapz(MXZ1, phi, axis=1)/np.pi
                MYZ1 = np.trapz(FYZ*np.sin(th), th, axis=1)
                MYZ  = np.trapz(MYZ1, phi, axis=1)/np.pi

            if not comp_all:
                return MXX, MYY, MZZ, MXY, MXZ, MYZ

        if order == 2 or comp_all:
            DXY1 = 1j*np.trapz(FXY*np.sin(th)**2*np.sin(ph), th, axis=1)
            DXY = np.trapz(DXY1, phi, axis=1)/np.pi
            if not comp_all_rel:
                DXX1 = 1j*np.trapz(FXX*np.sin(th)**2*np.sin(ph), th, axis=1)
                DXX  = np.trapz(DXX1, phi, axis=1)/np.pi
                DYY1 = 1j*np.trapz(FYY*np.sin(th)**2*np.sin(ph), th, axis=1)
                DYY  = np.trapz(DYY1, phi, axis=1)/np.pi
                DZZ1 = 1j*np.trapz(FZZ*np.sin(th)**2*np.sin(ph), th, axis=1)
                DZZ  = np.trapz(DZZ1, phi, axis=1)/np.pi
                DXZ1 = 1j*np.trapz(FXZ*np.sin(th)**2*np.sin(ph), th, axis=1)
                DXZ  = np.trapz(DXZ1, phi, axis=1)/np.pi
                DYZ1 = 1j*np.trapz(FYZ*np.sin(th)**2*np.sin(ph), th, axis=1)
                DYZ  = np.trapz(DYZ1, phi, axis=1)/np.pi

            if not comp_all:
                return DXX, DYY, DZZ, DXY, DXZ, DYZ

    if comp_all:
        if comp_all_rel:
            return MAA, MTT, DAE, MXX, MXY, DXY
        else:
            return MAA, MEE, MTT, MAE, MAT, MET, DAA, DEE, DTT, DAE, DAT, DET, \
                   MXX, MYY, MZZ, MXY, MXZ, MYZ, DXX, DYY, DZZ, DXY, DXZ, DYZ

def refine_M(f, M, A=.3, exp=0):

    """
    Function that refines the response function by appending
    a A*f^exp in the low-frequency regime.

    Arguments:
        f   -- frequency array (should be in units of Hz)
        M   -- response function to be refined at low frequencies
        A   -- amplitude of the response function at low frequencies
               (default 0.3 for the LISA monopole response function)
        exp -- exponent of the response function at low frequencies (default 0
               for the LISA monopole response function)

    Returns:
        fs -- refined array of frequencies
        Ms -- refined response function
    """

    ff0 = np.logspace(-6, np.log10(f[0].value), 1000)*u.Hz
    fs  = np.append(ff0, f)
    Ms  = np.append(A*ff0.value**exp, np.real(M))
    return fs, Ms

def compute_response_LISA_Taiji(f=f_ref, dir0=dir0, save=True, ret=False):

    """
    Function that computes LISA and Taiji's monopole and dipole response functions
    using the 'compute_interferometry' routine. It only computes the relevant
    response functions (see the tutorial 'response_functions.ipynb' for details).

    Arguments:
        dir0 -- directory where to save the results (default is 'detector_sensitivity')
        save -- option to save the results as output files (default True) with name
                'LISA_response_f' and 'Taiji_response_f'
        ret  -- option to return the results from the function

    Returns:
        MAA -- monopole response function of the TDI channel A
        MTT -- monopole response function of the TDI channel T (Sagnac channel)
        DAE -- dipole response function of the TDI correlation of the
               channels A and E
        MXX -- monopole response function of the interferometer channel X
        MXY -- monopole response function of the correlation of interferometer
               channels X and Y
        DXY -- dipole response functions of the correlation of the interferometer
               channels X and Y
    """

    f = np.logspace(-4, 0, 5000)*u.Hz

    # LISA response functions
    print('Calculating LISA response functions')
    MAA, MTT, DAE, MXX, MXY, DXY = compute_interferometry(f=f, L=L_LISA, comp_all_rel=True)
    # Taiji response functions
    print('Calculating Taiji response functions')
    MAA_Tai, MTT_Tai, DAE_Tai, MXX_Tai, MXY_Tai, DXY_Tai = \
            compute_interferometry(f=f, L=L_Taiji, comp_all_rel=True)

    # refine response functions at low frequencies (from known results)
    fs, MAs      = refine_M(f, MAA)
    fs, MAs_Tai  = refine_M(f, MAA_Tai)
    fs, MTs      = refine_M(f, MTT, A=1.709840e6, exp=6)
    fs, MTs_Tai  = refine_M(f, MTT_Tai, A=5.105546e6, exp=6)
    fs, DAEs     = refine_M(f, DAE, A=.2)
    fs, DAEs_Tai = refine_M(f, DAE_Tai, A=.2)
    fs, MXs      = refine_M(f, MXX, A=MXX[0])
    fs, MXs_Tai  = refine_M(f, MXX_Tai, A=MXX_Tai[0])
    fs, MXYs     = refine_M(f, MXY, A=MXY[0])
    fs, MXYs_Tai = refine_M(f, MXY_Tai, A=MXY_Tai[0])
    fs, DXYs     = refine_M(f, DXY, A=DXY[0])
    fs, DXYs_Tai = refine_M(f, DXY_Tai, A=DXY_Tai[0])

    # Write response functions in csv files
    if save:
        df = pd.DataFrame({'frequency': fs, 'MX': np.real(MXs), 'MXY': np.real(MXYs),
                           'DXY': np.real(DXYs)})
        df.to_csv(dir0 + 'LISA_response_f_X.csv')
        df = pd.DataFrame({'frequency': fs, 'MA': MAs, 'MT': MTs,
                           'DAE': DAEs})
        df.to_csv(dir0 + 'LISA_response_f_TDI.csv')
        print('saved response functions of channels X, Y of LISA in ', dir0 + 'LISA_response_f_X.csv')
        print('saved response functions of TDI channels of LISA in ', dir0 + 'LISA_response_f_TDI.csv')
        df_Tai = pd.DataFrame({'frequency': fs, 'MX': np.real(MXs_Tai), 'MXY': np.real(MXYs_Tai),
                           'DXY': np.real(DXYs_Tai)})
        df_Tai.to_csv(dir0 + 'Taiji_response_f_X.csv')
        df_Tai = pd.DataFrame({'frequency': fs, 'MA': MAs_Tai, 'MT': MTs_Tai,
                           'DAE': DAEs_Tai})
        df_Tai.to_csv(dir0 + 'Taiji_response_f_TDI.csv')
        print('saved response functions of channels X, Y of Taiji in ', dir0 + 'Taiji_response_f_X.csv')
        print('saved response functions of TDI channels of Taiji in ', dir0 + 'Taiji_response_f_TDI.csv')

    if ret:
        return fs, MAs, MTs, DAEs, MXs, MXYs, DXYs, \
               MAs_Tai, MTs_Tai, DAEs_Tai, MXs_Tai, MXYs_Tai, DXYs_Tai

############################## SENSITIVITIES AND SNR ##############################

def Sn_f(fs=f_ref, interf='LISA', TDI=True, M='MED', Xi=False):

    """
    Function that computes the strain sensitivity using the analytical fit
    for an interferometer channel X.

    Arguments:
        interf -- option to chose the interferometer (default 'LISA', other
                  option availables are 'Taiji' and 'comb', referring to cross-correlated
                  channels of LISA and Taiji)
        TDI    -- option to read the response functions for TDI chanels, instead of XYZ
                  chanels (default True)
        M      -- selection of cross-correlated channels (only when interf='comb', default 'MED')
        V      -- selection of Stokes parameter (default intensity I)
        Xi     -- option to compute strain sensitivity to polarized GW backgrounds from the
                  anisotropies induced due to the Solar System proper motion (default False)

    Returns:
        fs  -- frequency array
        SnA -- strain sensitivity in the channel A (if TDI), X (if not TDI) or if Xi is True,
               in the cross-correlated channels A and E (if TDI) or X and Y (if not TDI)
        SnT -- strain sensitivity in the Sagnac channel T (if TDI), or in the cross-correlated
               X and Y channels (if not TDI), only returns SnT if Xi is False
    """

    # read LISA and Taiji TDI response functions
    if interf != 'comb':
        fs, MAs, MTs, DAEs = read_response_LISA_Taiji(TDI=TDI, interf=interf)
    else:
        if not Xi:
            f_ED_I, M_ED_I = read_MAC(M=M, V='I') # read correlated ED response
            if M == 'MED': f_ED_I, M_ED_I = refine_M(f_ED_I, M_ED_I, A=0.028277196782809974)
            M_ED_I *= 2 # to get monopole response function
            fs = np.logspace(np.log10(f_ED_I[0].value), np.log10(f_ED_I[-1].value), 1000)*u.Hz
            MAs = np.interp(fs, f_ED_I, M_ED_I)
        else:
            fs, M_AC, M_AD, M_EC, M_ED = read_all_MAC(V='V')
            if M == 'MAC': MAs = abs(M_AC)
            if M == 'MAD': MAs = abs(M_AD)
            if M == 'MEC': MAs = abs(M_EC)
            if M == 'MED': MAs = abs(M_ED)
        MTs = MAs**0

    ## power spectral density of the noise
    if interf == 'LISA':
        PnA, PnT = Pn_f(f=fs, TDI=TDI, P=P_LISA, A=A_LISA, L=L_LISA)
    if interf == 'Taiji':
        PnA, PnT = Pn_f(f=fs, TDI=TDI, P=P_Taiji, A=A_Taiji, L=L_Taiji)
    if interf == 'comb':
        PnA, PnT = Pn_f(f=fs, TDI=TDI, P=P_LISA, A=A_LISA, L=L_LISA)
        PnC, PnS = Pn_f(f=fs, TDI=TDI, P=P_Taiji, A=A_Taiji, L=L_Taiji)
        PnA = np.sqrt(PnA*PnC)
        PnT = np.sqrt(PnT*PnS)

    # if Xi is True, it only returns the strain sensitivity to cross-correlated
    # channels A and E
    if Xi == True:
        if interf != 'comb': SnA = PnA/v/abs(DAEs)
        else:                SnA = PnA/MAs
        return fs, SnA

    # if not TDI, then PnA -> PnX, PnT -> Pncross, MAs -> MXs, MTs -> MXYs
    else: return fs, PnA/MAs, PnT/MTs

def Oms(f, S, h0=1., comb=False, S2=[], S3=[], S4=[], Xi=False):

    """
    Function that returns the sensitivity Sh(f) in terms of the GW energy density Om(f)

    Reference for Omega is RoperPol:2021xnd, equation B.18 (seems like it might have
    a typo, need to investigate this!). Final PLS sensitivites are again correct
    for a single chanel, since the factor of 2 in the SNR compensates for the
    1/2 factor here.

    Reference of Xi is RoperPol:2021xnd, equation B.21.

    Reference for combined sensitivities is RoperPol:2021xnd, equations B.37 (GW energy density,
    combining LISA and Taiji TDI channels A and C), and B.41 (polarization, combining
    4 cross-correlation between LISA and Taiji TDI channels AE, AD, CE, CD).

    Strain sensitivities, Omega sensitivities, and OmGW PLS agree with those of reference
    Caprini:2019pxz, see equation 2.14.

    Arguments:
        f    -- frequency array (should be in units of Hz)
        S    -- strain sensitivity function
        h0   -- parameterizes the uncertainties (Hubble tension) on the value
                of the Hubble rate (default 1)
        Xi   -- option to compute strain sensitivity to polarized GW backgrounds from the
                anisotropies induced due to the Solar System proper motion (default False)
        comb -- option to combine two sensitivities (e.g. LISA-Taiji sensitivities)
        S2   -- if comb is True, then second sensitivity to be combined
        S3, S4 -- if comb and Xi are both True, then it combines 4 sensitivites by cross-correlating
                  each of the 4 combinations of channels between LISA and Taiji (for example)

    Returns:
        Omega -- GW energy density sensitivity
    """

    H0 = co.H0_ref*h0           # reference value of H0 from cosmology (100 km/s/Mpc)
    A = 4*np.pi**2/3/H0**2
    if Xi: A /= 2
    Omega = S*A*f**3

    # if we are combining more than one channel for the sensitivity, we average them
    # through the harmonic mean
    if comb:
        Omega2 = S2*A*f**3
        Omega  = 1/np.sqrt(1/Omega**2 + 1/Omega2**2)
        if Xi:
            Omega3 = S3*A*f**3
            Omega4 = S4*A*f**3
            Omega  = 1/np.sqrt(1/Omega**2 + 1/Omega2**2 + \
                     1/Omega3**2 + 1/Omega4**2)

    return Omega

def compute_Oms_LISA_Taiji(interf='LISA', TDI=True, h0=1.):

    """
    Function that reads the response functions for LISA and/or Taiji, computes the strain
    sensitivities and from those, the sensitivity to the GW energy density spectrum Omega_s

    Arguments:
         interf -- option to chose the interferometer (default 'LISA', other
                   option availables are 'Taiji' and 'comb', referring to cross-correlated
                   channels of LISA and Taiji)
         TDI    -- option to read the response functions for TDI chanels, instead of XYZ
                   chanels (default True)
    """

    # read LISA and Taiji strain sensitivities Sn_f(f)
    fs, SnA, SnT = Sn_f(interf=interf, TDI=TDI)

    # Sn is the sensitivity of the channel A (if TDI) or X (if not TDI) for LISA or
    # Taiji (depending on what is interf)
    OmSA = Oms(fs, SnA, h0=h0, comb=False, Xi=False)
    OmST = Oms(fs, SnT, h0=h0, comb=False, Xi=False)

    return fs, OmSA, OmST

def OmPLS(Oms, f=f_ref, beta=beta_ref, SNR=1, T=1, Xi=0):

    """
    Function that computes the power law sensitivity (PLS).
    Reference is RoperPol:2021xnd, appendix B (equation B.31).

    Arguments:
        Oms  -- GW energy density sensitivity
        f    -- frequency array (should be in units of Hz, default is f_ref)
        beta -- array of slopes (default is beta_ref from -20 to 20)
        SNR  -- threshold signal-to-noise ratio (SNR) to compute the PLS
                (default 1)
        T    -- duration of the observation (in units of year, default 1)
        Xi   -- allows to compute PLS for polarization signals using the dipole
                response function setting Xi = 0.25 (default 0)

    Returns:
        Omega -- GW energy density power law sensitivity (PLS)
    """

    Cbeta = np.zeros(len(beta))
    T     = (T*u.yr).to(u.s)
    for i in range(0, len(beta)):
        aux = f.value**(2*beta[i])
        aa  = abs(1 - Xi*beta[i])
        Cbeta[i] = SNR/aa/np.sqrt(np.trapz(aux/Oms**2, f.value)*T.value)

    funcs = np.zeros((len(f), len(beta)))
    for i in range(0, len(beta)): funcs[:,i] = f.value**beta[i]*Cbeta[i]
    Omega = np.zeros(len(f))
    for j in range(0, len(f)): Omega[j] = np.max(funcs[j,:])

    return Omega

def SNR(f, OmGW, fs, Oms, T=1.):

    """
    Function that computes the signal-to-noise ratio (SNR) of a GW signal as

    SNR = sqrt(T int (OmGW/Oms)^2 df)

    Reference is RoperPol:2021xnd, appendix B (equation B.30).

    Arguments:
        f    -- frequency array of the GW signal
        OmGW -- GW energy density spectrum of the GW signal
        fs   -- frequency array of the GW detector sensitivity
        Oms  -- GW energy density sensitivity of the GW detector
        T    -- duration of observations in years (default 1)

    Returns:
        SNR -- SNR of the GW signal
    """

    T = T*u.yr
    T = T.to(u.s)
    T = T.value
    f = f.to(u.Hz)
    f = f.value
    OmGW = np.interp(fs, f, OmGW)
    OmGW[np.where(fs < f[0])] = 0
    OmGW[np.where(fs > f[-1])] = 0
    integ = np.trapz((OmGW/Oms)**2, fs)
    SNR   = np.sqrt(T*integ)

    return SNR
