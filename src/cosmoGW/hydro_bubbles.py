"""
hydro_bubbles.py is a Python routine that contains functions
to study the 1D hydrodynamic solutions of expanding bubbles
produced from first-order phase transitions.

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/hydro_bubbles.py

Author: Alberto Roper Pol
Created: 01/02/2023
Updated: 03/06/2025 (release cosmoGW 1.0: https://pypi.org/project/cosmoGW)

Other contributors: Antonino Midiri, Simona Procacci

Main references are:

Appendix A of RoperPol:2025a - A. Roper Pol, S. Procacci, A. S. Midiri,
C. Caprini, "Irrotational fluid perturbations from first-order phase
transitions," in preparation

Other relevant references are:

Espinosa:2010hh - J. R. Espinosa, T. Konstandin, J. M. No, G. Servant,
"Energy Budget of Cosmological First-order Phase Transitions,"
JCAP 06 (2010) 028, arXiv:1004.4187

Hindmarsh:2016lnk - M. Hindmarsh, "Sound shell model for acoustic gravitational
wave production at a first-order phase transition in the early Universe,"
Phys. Rev. Lett. 120 (2018) 7, 071301, arXiv:1608.04735

Hindmarsh:2019phv - M. Hindmarsh, M. Hijazi, "Gravitational waves from first order
cosmological phase transitions in the Sound Shell Model,"
JCAP 12 (2019) 062, arXiv:1909.10040

RoperPol:2023dzg - A. Roper Pol, S. Procacci, C. Caprini,
"Characterization of the gravitational wave spectrum from sound waves within
the sound shell model," Phys. Rev. D 109, 063531 (2024), arXiv:2308.12943.
"""

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

# reference values
cs2_ref   = 1/3        # speed of sound squared
Nxi_ref   = 10000      # reference discretization in xi
Nxi2_ref  = 10         # reference discretization in xi out
                       # of the profiles
Nvws_ref  = 20         # reference discretization in vwall
tol_ref   = 1e-5       # reference tolerance on shooting algorithm
it_ref    = 30         # reference number of iterations

# reference values for a deflagration
vw_def    = 0.5
alpha_def = 0.263
# reference values for a hybrid
vw_hyb    = 0.7
alpha_hyb = 0.052
# reference values for a detonation
vw_det    = 0.77
alpha_det = 0.091

# reference set of colors
cols_ref = ['black', 'darkblue', 'blue', 'darkgreen', 'green',
            'purple', 'darkred', 'red', 'darkorange', 'orange', 'violet']

def Chapman_Jouget(alp):

    '''
    Chapman-Jouget is the wall velocity at which the relative speed
    behind the wall becomes that of the spped of sound.
    It corresponds to the limiting case separating detonations
    and supersonic deflagrations.

    Reference: Eq. 39 of Espinosa:2010hh
    Note that Eq. B19 of Hindmarsh:2019phv has a typo

    Arguments:
        alp -- alpha at the + side of the wall (alpha_pl)
    Returns:
        vcJ -- Chapman-Jouget speed
    '''

    return 1./np.sqrt(3)/(1 + alp)*(1 + np.sqrt(alp*(2 + 3*alp)))

def type_nucleation(vw, alp, cs2=cs2_ref):

    '''
    Function that determines the type of bubble solution:
    a) subsonic deflagrations ('def'),
    b) supersonic deflagrations ('hyb'), or
    c) detonations ('det').

    Arguments:
        vw -- bubble wall speed
        alp -- strenght of the phase transition at the + side
                 of the wall (alpha_pl)
        cs2 -- square of the speed of sound (default 1/3)

    Returns:
        ty -- type of solution ('def', 'hyb', 'det')
    '''

    mult_vw  = isinstance(vw,  (list, tuple, np.ndarray))
    mult_alp = isinstance(alp, (list, tuple, np.ndarray))
    cs       = np.sqrt(cs2)

    if not mult_vw:  vw  = np.array([vw])
    if not mult_alp: alp = np.array([alp])
    v_cj = Chapman_Jouget(alp)

    ty = np.full((len(vw), len(alp)), 'hyb')
    vw, v_cj = np.meshgrid(vw, v_cj, indexing='ij')
    ty[vw < cs]   = 'def'
    ty[vw > v_cj] = 'det'

    if   not mult_alp and not mult_vw: ty = ty[0, 0]
    elif not mult_alp: ty = ty[:, 0]
    elif not mult_vw:  ty = ty[0, :]

    return ty

######## 1D HYDRO SOLUTIONS UNDER SPHERICAL SYMMETRY ########

'''
The solutions are based in the 1D hydrodynamic descriptions
of Espinosa:2010hh and Hindmarsh:2019phv.

The details accompanying the code are provided in RoperPol:2023dzg
and RoperPol:2025a (appendix A)
'''

def Lor_mu(v, vw):

    '''
    Lorentz transform of the velocity v in the reference frame of the wall
    (with speed vw)

    mu(vw, v) = (vw - v)/(1 - vw*v)

    Reference: Eq. B12 of Hindmarsh:2019phv
    '''

    return (vw - v)/(1 - v*vw)

### hydrodynamic shocks with no vacuum energy

def v_shock(xi):

    '''
    Function that computes the shock velocity at the - side of the
    shock. Computed from the v_+ v_- = 1/3 condition.
    '''

    vsh = (3*xi**2 - 1)/2/xi

    return vsh

def w_shock(xi):

    '''
    Function that computes the ratio of enthalpies w-/w+ across the
    shock. Computed from the v_+ v_- = 1/3 condition.
    '''

    wsh = (9*xi**2 - 1)/3/(1 - xi**2)

    return wsh

#### differential equation for the velocity radial profile

def xi_o_v(xi, v, cs2=cs2_ref):

    '''
    Function that characterizes the 1d hydro equation under radial
    symmetry.

    It returns the value of dxi/dv used to solve the
    equation in compute_xi_from_v using RK4.

    Reference: Eq. 27 of Espinosa:2010hh

    Arguments:
        xi  -- self-similar r/t
        v   -- 1d velocity profile
        cs2 -- square of the speed of sound (default 1/3)

    Returns:
        f   -- value of f(xi, v) = dxi/dv
    '''

    gamma2 = 1/(1 - v**2)
    mu = Lor_mu(v, xi)
    f = xi*gamma2*(1 - xi*v)*(mu**2/cs2 - 1)/2/v

    return f

def compute_xi_from_v(v, xi0, cs2=cs2_ref, shock=False):

    '''
    Function that computes the solution xi (v) using a 4th-order
    Runge-Kutta scheme. Since dv/dxi has a singularity, it is necessary
    to compute xi(v) and then invert each of the solutions to have the full
    dynamical solution. However, for the physical solution, computing v (xi)
    is more practical.

    Arguments:
        v     -- velocity array
        xi0   -- position where boundary condition is known (it has to correspond
                 to the first value in the velocity array)
        cs2   -- square of the speed of sound (default 1/3)
        shock -- option to stop the integration when a shock is found
                 (as happens in deflagrations)

    Returns:
        xi    -- self-similar array
        sh    -- boolean determining if a shock is formed
        indsh -- index determining the position of the shock
                 (-1 if no shock)
    '''

    xi = np.zeros(len(v)) + xi0
    sh = False
    indsh = -1

    for i in range(0, len(v) - 1):
        dv = v[i+1] - v[i]
        k1 = xi_o_v(xi[i], v[i])
        k2 = xi_o_v(xi[i] + dv*k1/2, .5*(v[i+1] + v[i]))
        k3 = xi_o_v(xi[i] + dv*k2/2, .5*(v[i+1] + v[i]))
        k4 = xi_o_v(xi[i] + dv*k3, v[i + 1])
        xi_new = xi[i] + 1/6*(k1 + 2*k2 + 2*k3 + k4)*dv
        if shock:
            xi_sh = xi_new
            v_sh = v_shock(xi_sh)
            if v[i + 1] < v_sh:
                xi[i + 1:] = xi_sh
                sh = True
                indsh = i
                break
        xi[i + 1] = xi_new
        if xi_new > 1: xi[i + 1] = 1

    return xi, sh, indsh

def compute_int_w(xi, v, cs2=cs2_ref):

    '''
    Function that computes the integrand for the integration of dw/dxi
    (enthalpy) equation, as a function of the solution v(xi).

    Reference: Eq. (29) of Espinosa:2010hh

     Arguments:
        xi  -- self-similar r/t
        v   -- 1d velocity profile
        cs2 -- square of the speed of sound (default 1/3)

    '''

    return (1. + 1./cs2)/(1. - v**2)*Lor_mu(v, xi)

def compute_w(v, xi, cs2=cs2_ref):

    '''
    Function that computes the enthalpy from the solution of the
    velocity profile.

     Arguments:
        xi  -- self-similar r/t
        v   -- 1d velocity profile
        cs2 -- square of the speed of sound (default 1/3)
    '''

    w = np.zeros(len(v)) + 1
    ss = 0
    for i in range(0, len(v) - 1):
        ff_ip = compute_int_w(xi[i + 1], v[i + 1], cs2=cs2)
        ff_i = compute_int_w(xi[i], v[i], cs2=cs2)
        ss += .5*(v[i + 1] - v[i])*(ff_ip + ff_i)
        w[i + 1] = np.exp(ss)

    return w

######### SOLVE FOR THE DIFFERENT TYPE OF BOUNDARY CONDITIONS #########

############# MATCHING CONDITIONS ACROSS DISCONTINUITIES ##############

def vp_tilde_from_vm_tilde(vw, alpha, plus=True, sg='plus'):

    '''
    Function that computes the + (symmetric phase) or - (broken phase)
    velocity, defined in the wall reference frame, across the wall,
    as a function of the value of the velocity at the opposite side of
    the wall (via the matching conditions).

    Reference: Eqs. B6 and B7 of Hindmarsh:2019phv

    Arguments:
        vw    -- velocity imposed at one of the sides of the wall.
                 It usually is the wall velocity but becomes cs for
                 hybrids
        alpha -- phase transition strength at the symmetric phase
                 (can be different than that at the nucleation temperature)
        plus  -- option to consider positive (True) or negative (False)
                 branch of the equation found from the matching conditions
                 (default is True)
        sg    -- option to compute v+ from v- = vw if sg == 'plus' or to compute
                 v- from v+ = vw if sg == 'minus' (default is 'plus')

    Returns:
        vp_vm -- v+ or v- from the value at the other side of the bubble wall
    '''

    if sg == 'plus':

        a1 = 1/3/vw + vw
        a2 = np.sqrt((1/3/vw - vw)**2 + 4*alpha**2 + 8/3*alpha)
        aa = .5/(1 + alpha)

    else:

        a1 = (1 + alpha)*vw + (1 - 3*alpha)/3/vw
        a2 = np.sqrt(((1 + alpha)*vw + (1 - 3*alpha)/3/vw)**2 - 4/3)
        aa = 1/2

    if plus: a = a1 + a2
    else: a = a1 - a2

    vp_vm = aa*a

    return vp_vm

def vplus_vminus(alpha, vw=1., ty='det', cs2=cs2_ref):

    '''
    Function that returns v_+ and v_- (in the wall frame)
    for the different type of solutions
    (deflagrations, detonations and hybrids).

    This allows to give the boundary conditions corresponding to each
    of the solutions.

    Reference: Appendix A of RoperPol:2025a

    Arguments:
        alpha -- value of alpha at the + side of the wall
        vw    -- wall velocity
        cs2   -- speed of sound squared (default is 1/3)
        ty    -- type of solution

    Returns:
        vplus, vminus -- + and - velocities expressed in the wall
                            reference of frame
    '''

    cs = np.sqrt(cs2)

    if not isinstance(ty, (list, tuple, np.ndarray)):

        if ty == 'det':
            vplus = vw
            vminus = vp_tilde_from_vm_tilde(vw, alpha, plus=True, sg='minus')
        if ty == 'def':
            vminus = vw
            vplus = vp_tilde_from_vm_tilde(vw, alpha, plus=False, sg='plus')
        if ty == 'hyb':
            vminus = cs
            vplus = vp_tilde_from_vm_tilde(cs, alpha, plus=False, sg='plus')

    else:

        vplus    = np.zeros(len(vw))
        vminus   = np.zeros(len(vw))
        inds_det = np.where(ty == 'det')
        inds_def = np.where(ty == 'def')
        inds_hyb = np.where(ty == 'hyb')
        vplus[inds_det]  = vw[inds_det]
        vminus[inds_det] = vp_tilde_from_vm_tilde(vw, alpha,
                                    plus=True, sg='minus')[inds_det]
        vplus[inds_def]  = vp_tilde_from_vm_tilde(vw, alpha,
                                    plus=False, sg='plus')[inds_def]
        vminus[inds_def] = vw[inds_def]
        vplus[inds_hyb]  = vp_tilde_from_vm_tilde(cs, alpha,
                                    plus=False, sg='plus')
        vminus[inds_hyb] = cs

    return vplus, vminus

######## function that computes the detonation part of the solutions ########
def det_sol(v0, xi0, cs2=cs2_ref, Nxi=Nxi_ref, zero_v=-4):

    '''
    Function that computes a detonation solution with boundary condition
    v0 at xi0

    Arguments:
        v0     -- value of v0 at the boundary
        xi0    -- position of the boundary
        cs2    -- speed of sound squared (default is 1/3)
        Nxi    -- number of discretization points in xi
        zero_v -- reference zero velocity (default 1e-4)

    Returns:
        xis    -- array of xi
        vs     -- array of 1d velocities
        ws     -- array of 1d enthalpies
    '''

    # compute solution from initial condition v = v0 at xi = vw
    # until v reduces to 4 orders of magntiude below value v minus
    cs = np.sqrt(cs2)

    # included option to initialize with multiple vws
    if not isinstance(xi0, (list, tuple, np.ndarray)):

        vs = np.logspace(np.log10(v0), np.log10(v0) + zero_v, Nxi)
        xis, sh, indsh = compute_xi_from_v(vs, xi0, cs2=cs2, shock=False)
        xi_sh = xis[indsh]
        ws    = compute_w(vs, xis, cs2=cs2)

        inds_sort = np.argsort(xis)
        xis = xis[inds_sort]
        vs  = vs[inds_sort]
        ws  = ws[inds_sort]

    else:

        xis = np.zeros((len(xi0), Nxi))
        vs  = np.zeros((len(xi0), Nxi))
        ws  = np.zeros((len(xi0), Nxi))

        for i in range(0, len(xi0)):

            vs[i, :]  = np.logspace(np.log10(v0), np.log10(v0) + zero_v, Nxi)
            xis[i, :], _, _ = compute_xi_from_v(vs[i, :], xi0[i], cs2=cs2,
                                                shock=False)
            ws[i, :]  = compute_w(vs[i, :], xis[i, :], cs2=cs2)

            inds_sort = np.argsort(xis[i, :])
            xis[i, :] = xis[i, inds_sort]
            vs[i, :]  = vs[i, inds_sort]
            ws[i, :]  = ws[i, inds_sort]

    return xis, vs, ws

####### function that computes the deflagration part of the solutions #######
def def_sol(v0, xi0, cs2=cs2_ref, Nxi=Nxi_ref, shock=True, zero_v=-4):

    '''
    Function that computes a deflagration solution with boundary condition
    v0 at xi0

    Arguments:
        v0     -- value of v0 at the boundary
        xi0    -- position of the boundary
        cs2    -- speed of sound squared (default is 1/3)
        Nxi    -- number of discretization points in xi
        shock  -- possibility to stop the calculation once a shock is formed
                 (default is True)
        zero_v -- reference zero velocity (default 1e-4)

    Returns:
        xis    -- array of xi
        vs     -- array of 1d velocities
        ws     -- array of 1d enthalpies
        xi_sh, sh -- position of the shock and boolean which becomes
                        True if a shock forms
    '''

    vs = np.logspace(np.log10(v0), np.log10(v0) + zero_v, Nxi)
    cs = np.sqrt(cs2)
    xi_sh = cs
    sh    = False
    v_sh  = 0

    if shock:
        xiss, sh, indsh = compute_xi_from_v(vs, xi0, cs2=cs2, shock=True)
        xi_sh = xiss[indsh]
        if not sh: xi_sh = cs
        v_sh  = v_shock(xi_sh)
        xis   = np.linspace(xi0, xi_sh, Nxi + 1)
        xis   = xis[:Nxi-1]
        vs    = np.interp(xis, xiss, vs)
        vs    = np.append(vs, v_sh)
        xis   = np.append(xis, xi_sh)

    else:
        xis, sh, indsh = compute_xi_from_v(vs, xi0, cs2=cs2, shock=False)
        xi_sh = xis[indsh]

    ws   = compute_w(vs, xis, cs2=cs2)
    w_sh = w_shock(xi_sh)
    ws   = ws*w_sh/ws[-1]

    return xis, vs, ws, xi_sh, sh

def compute_def(vw=vw_def, alpha=alpha_def, cs2=cs2_ref, Nxi=Nxi_ref,
                shock=True):

    '''
    Function that computes the solutions for a subsonic deflagration
    1d profile given vw and alpha, using def_sol

    Arguments:
        vw    -- wall velocity (default is 0.5)
        alpha -- strength of the phase transition (default is 0.263)
        cs2   -- speed of sound squared (default is 1/3)
        Nxi   -- number of discretization points in xi
        shock -- possibility to stop the calculation once a shock is formed
                 (default is True)

    Returns:
        xis   -- array of xi
        vs    -- array of 1d velocities
        ws    -- array of 1d enthalpies
        xi_sh, sh -- position of the shock and boolean which becomes
                        True if a shock forms
        w_pl, w_m -- plus and minus values of the enthalpies
                        across the bubble
    '''

    ## relative velocity at + is computed from \tilde v- = \xi_w
    vrels, _ = vplus_vminus(alpha, vw=vw, ty='def')
    # Lorentz boosted v plus
    vpl = Lor_mu(vrels, vw)

    xis, vs, ws, xi_sh, sh = def_sol(vpl, vw, cs2=cs2, Nxi=Nxi, shock=shock)

    # values at both sides of the bubble wall
    w_pl = ws[0]
    w_m = w_pl*vrels/(1 - vrels**2)/vw*(1 - vw**2)

    return xis, vs, ws, xi_sh, sh, w_pl, w_m

def compute_hyb(vw=vw_hyb, alpha=alpha_hyb, cs2=cs2_ref,Nxi=Nxi_ref,
                shock=True):

    '''
    Function that computes the solutions for a supersonic deflagration
    1d profile given vw and alpha, using det_sol and def_sol

    Arguments:
        vw    -- wall velocity (default is 0.7)
        alpha -- strength of the phase transition (default is 0.052)
        cs2   -- speed of sound squared (default is 1/3)
        Nxi   -- number of discretization points in xi
        shock -- possibility to stop the calculation once a shock is formed
                 (default is True)

    Returns:
        xis   -- array of xi
        vs    -- array of 1d velocities
        ws    -- array of 1d enthalpies
        xi_sh, sh -- position of the shock and boolean which becomes
                        True if a shock forms
        w_pl, w_m -- plus and minus values of the enthalpies
    '''

    cs = np.sqrt(cs2)

    ## relative velocity at + is computed from \tilde v- = cs
    vrels, _ = vplus_vminus(alpha, cs2=cs2, ty='hyb')
    vpl = Lor_mu(vrels, vw)
    vm  = Lor_mu(cs, vw)

    # compute deflagration solution
    xis, vs, ws, xi_sh, sh = def_sol(vpl, vw, cs2=cs2,
                                     Nxi=int(Nxi/2), shock=shock)

    # compute detonation solution
    xis2, vs2, ws2 = det_sol(vm, vw, cs2=cs2, Nxi=int(Nxi/2))
    # ratio of w+ over w- across the bubble wall
    w_pl = ws[0]
    w_m  = w_pl*vrels/(1 - vrels**2)*(1 - cs2)/cs
    ws2 *= w_m

    xis = np.append(xis2, xis)
    vs  = np.append(vs2, vs)
    ws  = np.append(ws2, ws)

    return xis, vs, ws, xi_sh, sh, w_pl, w_m

def compute_det(vw=vw_det, alpha=alpha_det, cs2=cs2_ref, Nxi=Nxi_ref):

    '''
    Function that computes the solutions for a detonation 1d profile
    given vw and alpha, using det_sol

    Arguments:
        vw    -- wall velocity (default is 0.77)
        alpha -- strength of the phase transition (default is 0.091)
        cs2   -- speed of sound squared (default is 1/3)
        Nxi   -- number of discretization points in xi

    Returns:
        xis   -- array of xi
        vs    -- array of 1d velocities
        ws    -- array of 1d enthalpies
        w_pl, w_m -- plus and minus values of the enthalpies
                        across the bubble
    '''

    ## relative velocity at - is computed from \tilde v+ = \xi_w
    _, vrels = vplus_vminus(alpha, vw=vw, ty='det')
    # Lorentz boosted v minus
    vm   = Lor_mu(vrels, vw)
    w_m  = vw/(1 - vw**2)/vrels*(1 - vrels**2)
    w_pl = 1

    xis, vs, ws = det_sol(vm, vw, cs2=cs2, Nxi=Nxi)
    ws *= w_m

    # no shock is formed in detonations, so xi_sh is set to vw
    # and sh to False
    xi_sh = vw
    sh    = False

    return xis, vs, ws, xi_sh, sh, w_pl, w_m

def compute_alphan(vw=vw_def, alpha_obj=alpha_def, tol=tol_ref, cs2=cs2_ref,
                   quiet=False, max_it=it_ref, meth=1, Nxi=Nxi_ref, ty='def'):

    '''
    Function that computes the value of \alpha_+ corresponding to alpha.
    It requires to compute the 1d profile of w and then iteratively look for
    the value of alpha_+ that gives the correct alpha.

    Arguments:
        vw        -- wall velocity
        alpha_obj -- value of alpha defined at the nucleation temperature that
                        wants to be reached
        tol       -- relative tolerance to consider convergence (default is 1e-4)
        cs2       -- speed of sound squared (default is 1/3)
        quiet     -- option to avoid printing some debug options (default is False)
        max_it    -- maximum number of allowed iterations (default is 30)
        meth      -- method on the Newton-Raphson update (2 options)
        Nxi       -- number of points in xi discretization (default is 1000)
        ty        -- type of solution (options are def or hyb)

    Returns:
        xis0, vvs0, wws0 -- arrays of xi, velocity and enthalpy of the converged
                               solutions
        xi_sh, sh, w_pl, w_m -- shock position, boolean if shock has formed,
                                    plus and minus enthalpies
        alpha_n   -- converged (or not) alpha at nucleation
        alp_plus  -- value of alpha_+ leading to alpha_obj
        conv      -- boolean determining if the algorithm has converged
    '''

    # first guess
    alp_plus = alpha_obj

    j = 0
    conv = False

    while not conv and j < max_it:

        j += 1

        if ty == 'hyb':
            xis0, vvs0, wws0, xi_sh, sh, w_pl, w_m = \
                compute_hyb(vw=vw, alpha=alp_plus, cs2=cs2, Nxi=Nxi, shock=True)

        if ty == 'def':
            xis0, vvs0, wws0, xi_sh, sh, w_pl, w_m = \
                compute_def(vw=vw, alpha=alp_plus, cs2=cs2, Nxi=Nxi, shock=True)

        alpha_n = alp_plus*w_pl

        if abs(alpha_n - alpha_obj)/alpha_obj < tol: conv = True

        else:

            if meth==1: alp_plus = alpha_obj/w_pl
            if meth==2: alp_plus += (alpha_obj - alpha_n)/w_pl

        if not quiet:
            print('iteration', j, 'alpha', alpha_n)
            print('iteration', j, 'new guess', alp_plus)

    if not quiet:
        print(j, 'iterations for vw=', vw,' and alpha= ',alpha_obj)
        print('alpha:', alpha_n, ', alpha_+:', alp_plus)

    return xis0, vvs0, wws0, xi_sh, sh, w_pl, w_m, alpha_n, alp_plus, conv

def compute_profiles_vws_multalp(alphas, vws=[0], cs2=cs2_ref, Nvws=Nvws_ref, Nxi=Nxi_ref,
                                 Nxi2=Nxi2_ref, alphan=True, quiet=True, tol=tol_ref,
                                 max_it=it_ref, lam=False, eff=False):

    '''
    Function that computes the velocity and enthalpy profiles for an array
    of alpha using compute_profiles_vws function

    Arguments:
        alphas -- nucleation T alphas
        vws    -- range of wall velocities
        cs2    -- square of the speed of sound (default is 1/3)
        Nvws   -- number of wall velocities (if vws is not given)
        Nxi    -- number of discretization points in xi where the profiles are
                  computed
        Nxi2   -- number of discretization points in xi out of the solution
                  of the 1d profiles
        alphan -- option to identify if the input alpha is at the
                  nucleation temperature (if True) or alpha+ (if False),
                  default is True
        quiet  -- option to avoid printing debugging information (default is True)
        max_it -- maximum number of iterations to find alpha_+
        lam    -- option to compute energy perturbations lambda instead
                  of enthalpy (default is False, so enthalpy)
        tol    -- tolerance of the relative error to consider convergence
                  of alpha_+ has been reached
        eff    -- option to compute the efficiency factors kappa and
                  omega (default is False)

    Returns:
        xis       -- array of xi positions
        vvs       -- array of velocities
        wws       -- array of enthalpies (energy density perturbations
                     if lam is True)
        alphas_n  -- array of nucleation alphas (if input is alpha+, when
                     alphan is False, otherwise it returns array of values
                     of alpha+)
        conv      -- array of booleans determining if the alphan calculation
                     has converged
        shocks    -- array of booleans determining if a shock is produced
                     in the fluid profile
        xi_shocks -- positions of the shocks (if shock is not produced,
                     it returns xi_front)
        wms       -- enthalpy values at - side of the wall
        kappas    -- ratio of kinetic to vacuum energy density
                     (only returned if eff is True)
        omegas    -- ratio of energy density perturbations to vacuum
                     energy density (only returned if eff is True)
    '''

    if len(np.shape(vws)) == 1:
        if len(vws) == 0: vws = np.linspace(0.1, .99, Nvws)
    xis = np.linspace(0, 1, Nxi + Nxi2)

    if isinstance(alphas, (list, tuple, np.ndarray)):

        vvs       = np.zeros((len(vws), len(alphas), len(xis)))
        wws       = np.zeros((len(vws), len(alphas), len(xis))) + 1.
        alphas_n  = np.zeros((len(vws), len(alphas)))
        conv      = np.zeros((len(vws), len(alphas))) + 1.
        shocks    = np.zeros((len(vws), len(alphas)))
        xi_shocks = np.zeros((len(vws), len(alphas)))
        wms       = np.zeros((len(vws), len(alphas)))

    if eff:

        if isinstance(alphas, (list, tuple, np.ndarray)):

            kappas = np.zeros((len(vws), len(alphas)))
            omegas = np.zeros((len(vws), len(alphas)))

            for i in range(0, len(alphas)):

                if not quiet:
                    print('Computing alpha = %'%alphas[i], ' out of',
                          '%i'%(len(alphas) + 1))

                xis, vvs[:, i, :], wws[:, i, :], alphas_n[:, i], \
                    conv[:, i], shocks[:, i], xi_shocks[:, i], wms[:, i], \
                    kappas[:, i], omegas[:, i] = \
                        compute_profiles_vws(alphas[i], vws=vws, cs2=cs2,
                            Nxi=Nxi, Nxi2=Nxi2, plot=False, alphan=alphan,
                            quiet=True, tol=tol, max_it=max_it, lam=lam,
                            eff=eff)

        else:

            xis, vvs, wws, alphas_n, conv, shocks, xi_shocks, wms, \
                    kappas, omegas = \
                        compute_profiles_vws(alphas, vws=vws, cs2=cs2,
                            Nxi=Nxi, Nxi2=Nxi2, plot=False, alphan=alphan,
                            quiet=True, tol=tol, max_it=max_it, lam=lam,
                            eff=eff)

        return xis, vvs, wws, alphas_n, conv, shocks, xi_shocks, wms, kappas, omegas

    else:

        if isinstance(alphas, (list, tuple, np.ndarray)):

            for i in range(0, len(alphas)):

                if not quiet:
                    print('Computing alpha = %f'%alphas[i], ' out of',
                          '%i'%(len(alphas) + 1))

                xis, vvs[:, i, :], wws[:, i, :], alphas_n[:, i], \
                    conv[:, i], shocks[:, i], xi_shocks[:, i], wms[:, i] = \
                        compute_profiles_vws(alphas[i], vws=vws, cs2=cs2,
                            Nxi=Nxi, Nxi2=Nxi2, plot=False, alphan=alphan,
                            quiet=True, tol=tol, max_it=max_it, lam=lam,
                            eff=eff)

        else:

            xis, vvs, wws, alphas_n, conv, shocks, xi_shocks, wms = \
                compute_profiles_vws(alphas, vws=vws, cs2=cs2,
                                     Nxi=Nxi, Nxi2=Nxi2, plot=False, alphan=alphan,
                                     quiet=True, tol=tol, max_it=max_it, lam=lam,
                                     eff=eff)

        return xis, vvs, wws, alphas_n, conv, shocks, xi_shocks, wms

def compute_profiles_vws(alpha, vws=[], cs2=cs2_ref, Nvws=Nvws_ref, Nxi=Nxi_ref,
                         Nxi2=Nxi2_ref, plot=False, plot_v='v', cols=[],
                         alphan=True, quiet=True, tol=tol_ref, max_it=it_ref,
                         ls='solid', alp=1., lam=False, meth=1, legs=False,
                         fs_lg=14, st_lg=2, eff=False, save=False, dec_vw=1,
                         ress='results/1d_profiles', strs_vws=[], str_alp=[]):

    '''
    Function that computes the velocity and enthalpy profiles for a given
    alpha (at nucleation T, not alpha+) and a range of wall velocities.

    Arguments:
        alpha  -- nucleation T alpha
        vws    -- range of wall velocities
        cs2    -- square of the speed of sound (default is 1/3)
        Nvws   -- number of wall velocities (if vws is not given)
        Nxi    -- number of discretization points in xi where the profiles are
                  computed
        Nxi2   -- number of discretization points in xi out of the solution
                  of the 1d profiles
        plot   -- option to plot the resulting 1d profiles
        plot_v -- choice of plotting ('v' for velocity, 'w' for
                  enthalpy, 'both' for both)
        cols   -- array with colors (cols_ref by default)
        alphan -- option to identify if the input alpha is at the
                  nucleation temperature (if True) or alpha+ (if False),
                  default is True
        quiet  -- option to avoid printing debugging information (default is True)
        max_it -- maximum number of iterations to find alpha_+
        ls     -- line styles
        alp    -- opacity of the plots
        lam    -- option to compute energy perturbations lambda instead
                  of enthalpy (default is False, so enthalpy)
        tol    -- tolerance of the relative error to consider convergence
                  of alpha_+ has been reached
        legs   -- legends to be included (default is False)
        save   -- option to save in a file the results of the 1d
                  profiles (default is False)
        ress   -- directory where to save the files (default is
                  'results/1d_profiles')
        eff    -- option to compute the efficiency factors kappa and
                  omega (default is False)

    Returns:
        xis       -- array of xi positions
        vvs       -- array of velocities
        wws       -- array of enthalpies (energy density perturbations
                     if lam is True)
        alphas_n  -- array of nucleation alphas (if input is alpha+, when
                     alphan is False, otherwise it returns array of values
                     of alpha+)
        conv      -- array of booleans determining if the alphan calculation
                     has converged
        shocks    -- array of booleans determining if a shock is produced
                     in the fluid profile
        xi_shocks -- positions of the shocks (if shock is not produced,
                     it returns xi_front)
        wms       -- enthalpy values at - side of the wall
        kappas    -- ratio of kinetic to vacuum energy density
                     (only returned if eff is True)
        omegas    -- ratio of energy density perturbations to vacuum
                     energy density (only returned if eff is True)
    '''

    if len(vws) == 0: vws = np.linspace(0.1, .99, Nvws)
    vCJ       = Chapman_Jouget(alp=alpha)
    cs        = np.sqrt(cs2)
    xis       = np.linspace(0, 1, Nxi + Nxi2)
    vvs       = np.zeros((len(vws), len(xis)))
    wws       = np.zeros((len(vws), len(xis))) + 1.
    alphas_n  = np.zeros(len(vws))
    conv      = np.zeros(len(vws)) + 1.
    kappas    = np.zeros(len(vws))
    omegas    = np.zeros(len(vws))
    shocks    = np.zeros(len(vws))
    xi_shocks = np.zeros(len(vws))
    wms       = np.zeros(len(vws))

    if plot_v == 'both' and plot:
        plt.figure(1)
        plt.figure(2)

    if len(cols) == 0: cols = cols_ref

    for i in range(0, len(vws)):

        # determine type of solution
        ty = type_nucleation(vws[i], alpha, cs2=cs2)

        if ty == 'def':

            ## iteratively compute the real alpha_+ leading to alpha
            if alphan:
                xis0, vvs0, wws0, xi_sh, sh, w_pl, w_m, alpha_n, \
                    alp_plus, conv[i] = compute_alphan(vw=vws[i], alpha_obj=alpha,
                                            tol=tol, cs2=cs2, meth=meth,
                                            quiet=quiet, max_it=max_it, Nxi=Nxi,
                                            ty='def')
                alphas_n[i] = alp_plus

            ## otherwise, alpha given is assumed to be alpha_+
            else:
                xis0, vvs0, wws0, xi_sh, sh, w_pl, w_m = \
                        compute_def(vw=vws[i], alpha=alpha, cs2=cs2,
                                    Nxi=Nxi, shock=True)
                alphas_n[i] = alpha*w_pl

            inds = np.where((xis >= vws[i])*(xis <= xi_sh))[0]
            inds2 = np.where(xis < vws[i])[0]
            wws[i, inds2] = w_m

        elif ty == 'hyb':

            ## iteratively compute the real alpha_+ leading to alpha
            if alphan:
                xis0, vvs0, wws0, xi_sh, sh, w_pl, w_m, alpha_n, \
                    alp_plus, conv[i] = compute_alphan(vw=vws[i], alpha_obj=alpha,
                                            tol=tol, cs2=cs2, ty='hyb', meth=meth,
                                            quiet=quiet, max_it=max_it, Nxi=Nxi)
                alphas_n[i] = alp_plus

            ## otherwise, alpha given is assumed to be alpha_+
            else:
                xis0, vvs0, wws0, xi_sh, sh, w_pl, w_m = \
                        compute_hyb(vw=vws[i], alpha=alpha, cs2=cs2,
                                    Nxi=Nxi, shock=True)
                alphas_n[i] = alpha*w_pl

            inds = np.where((xis >= cs)*(xis <= xi_sh))[0]
            if xi_sh == vws[i]: inds = \
                    np.append(inds, np.where((xis <= xi_sh))[-1] + 1)
            inds2 = np.where(xis < cs)[0]
            wws[i, inds2] = wws0[0]

        else:

            xis0, vvs0, wws0, xi_sh, sh, w_pl, w_m = \
                    compute_det(vw=vws[i], alpha=alpha, cs2=cs2,
                                Nxi=Nxi)

            inds  = np.where((xis >= cs)*(xis <= vws[i]))[0]
            inds2 = np.where(xis < cs)[0]
            wws[i, inds2] = wws0[0]
            alphas_n[i]   = alpha

        vvs[i, inds] = np.interp(xis[inds], xis0, vvs0)
        wws[i, inds] = np.interp(xis[inds], xis0, wws0)
        shocks[i]    = sh
        xi_shocks[i] = xi_sh
        wms[i]       = w_m

        # compute efficiency of energy density production
        if eff:
            kappas[i], omegas[i] = kappas_from_prof(vws[i], alpha, xis,
                                                    wws[i, :], vvs[i, :])


        # compute mean energy density from enthalpy if lam is True
        if lam:
            if alphan: alp_lam = alpha
            else: alp_lam      = alphas_n[i]
            wws[i, :]          = w_to_lam(xis, wws[i, :], vws[i], alp_lam)

        if plot:

            j = i%11
            if st_lg == 1: str_lg = r'$\xi_w=%.1f$'%vws[i]
            if st_lg == 2: str_lg = r'$\xi_w=%.2f$'%vws[i]
            if plot_v=='v': plt.plot(xis, vvs[i, :], color=cols[j], ls=ls, alpha=alp,
                                     label=str_lg)
            if plot_v=='w': plt.plot(xis, wws[i, :], color=cols[j], ls=ls, alpha=alp,
                                     label=str_lg)
            if plot_v=='both':
                plt.figure(1)
                plt.plot(xis, vvs[i, :], color=cols[j], ls=ls, alpha=alp, label=str_lg)
                plt.figure(2)
                plt.plot(xis, wws[i, :], color=cols[j], ls=ls, alpha=alp, label=str_lg)

        if save and alphan:

            # option to save the results on files (one file for each alpha
            # and wall velocity)
            # save is only possible when the input alpha is that at the
            # nucleation temperature (alphan = True)
            df = pd.DataFrame({'alpha': alpha*xis**0, 'xi_w': vws[i]*xis**0,
                               'xi': xis, 'v': vvs[i, :], 'w': wws[i, :],
                               'alpha_pl': alphas_n[i]*xis**0, 'shock': shocks[i]*xis**0,
                               'xi_sh': xi_shocks[i]*xis**0, 'wm': wms[i]*xis**0})
            # save file
            if len(str_alp) == 0:
                str_alp = '%s'%(alpha)
                str_alp = '0' + str_alp
                str_alp = str_alp[2:]
            if len(strs_vws) == 0:
                str_vws = '%s'%(np.round(vws[i], decimals=dec_vw))
                str_vws = str_vws[2:]
            else: str_vws = strs_vws[i]
            file_dir = ress + '/alpha_%s_vw_%s.csv'%(str_alp, str_vws)
            try:
                df.to_csv(file_dir)
                print('results of 1d profile saved in ', file_dir)
            except:
                print('create directory results/1d_profiles to save',
                      ' the 1d profiles')

    if plot:

        if plot_v == 'v' or plot_v == 'both':
            if plot_v == 'both': plt.figure(1)
            plt.ylim(-.05, 1.05)
            plt.ylabel(r'$ v_{\rm ip} (\xi)$')
        if plot_v == 'w' or plot_v == 'both':
            if plot_v == 'both': plt.figure(2)
            plt.ylim(0, 5)
            if lam: plt.ylabel(r'$ \lambda_{\rm ip} (\xi)$')
            else: plt.ylabel(r'$ w(\xi)$')
        l = [1]
        if plot_v == 'both': l = [1, 2]
        for j in l:
            plt.figure(j)
            plt.xlim(0, 1)
            plt.vlines(cs, -5, 30, color='black', ls='dashed', lw=1)
            plt.vlines(vCJ, -5, 30, color='black', ls='dashed', lw=1)
            plt.xlabel(r'$\xi$')
            if legs: plt.legend(fontsize=fs_lg)

    if eff:
        return xis, vvs, wws, alphas_n, conv, shocks, xi_shocks, wms, kappas, omegas
    else:
        return xis, vvs, wws, alphas_n, conv, shocks, xi_shocks, wms

################### COMPUTING EFFICIENCIES FROM 1D PROFILES ###################

def kappas_from_prof(vw, alpha, xis, ws, vs):

    '''
    Function that computes the kinetic energy density efficiency kappa
    and thermal factor omega from the 1d profiles.
    '''

    try:
        kappa = 4/vw**3/alpha*np.trapezoid(xis**2*ws/(1 - vs**2)*vs**2, xis)
        omega = 3/vw**3/alpha*np.trapezoid(xis**2*(ws - 1), xis)
    except:
        kappa = 4/vw**3/alpha*np.trapz(xis**2*ws/(1 - vs**2)*vs**2, xis)
        omega = 3/vw**3/alpha*np.trapz(xis**2*(ws - 1), xis)

    return kappa, omega

def kappas_Esp(vw, alp, cs2=cs2_ref):

    """"
    Function that computes the efficiency in converting vacuum to
    kinetic energy density for detonations, deflagrations and hybrids.

    Uses the semiempirical fits from Espinosa:2010hh, appendix A,
    following the bag equation of state.

    Numerical values can be computed from the 1d profiles using
    compute_profiles_vws function with eff = True

    Arguments:
        vw    -- value or array of wall velocities
        alp   -- strength of the phase transition at the nucleation temperature
        cs2   -- square of the speed of sound (default is 1/3)

    Returns:
        kappa -- ratio of kinetic to vacuum energy density, computed
                 in the bag equation of state
    """

    cs   = np.sqrt(cs2)

    mult_alp = isinstance(alp, (list, tuple, np.ndarray))
    mult_vw  = isinstance(vw,  (list, tuple, np.ndarray))

    if not mult_vw:  vw  = np.array([vw])
    if not mult_alp: alp = np.array([alp])
    v_cj = Chapman_Jouget(alp)

    ty       = type_nucleation(vw, alp, cs2=cs2)
    _,  v_cj = np.meshgrid(vw, v_cj, indexing='ij')
    vw, alp  = np.meshgrid(vw, alp,  indexing='ij')
    kappa    = np.zeros_like(vw)

    # kappa at vw << cs
    kapA = vw**(6/5)*6.9*alp/(1.36 - 0.037*np.sqrt(alp) + alp)
    # kappa at vw = cs
    kapB = alp**(2/5)/(0.017 + (0.997 + alp)**(2/5))
    # kappa at vw = cJ (Chapman-Jouget)
    kapC = np.sqrt(alp)/(0.135 + np.sqrt(0.98 + alp))
    # kappa at vw -> 1
    kapD = alp/(0.73 + 0.083*np.sqrt(alp) + alp)

    # deflagrations
    den       = (cs**(11/5) - vw**(11/5))*kapB + vw*cs**(6/5)*kapA
    kappa_def = cs**(11/5)*kapA*kapB/den
    kappa[ty == 'def'] = kappa_def[ty == 'def']

    # detonations
    kappa_det = (v_cj - 1)**3*(v_cj/vw)**(5/2)*kapC*kapD
    den       = ((v_cj - 1)**3 - (vw - 1)**3)*v_cj**(5/2)*kapC + (vw - 1)**3*kapD
    kappa_det = kappa_det/den
    kappa[ty == 'det'] = kappa_det[ty == 'det']

    # hybrids
    ddk        = -.9*np.log(np.sqrt(alp)/(1 + np.sqrt(alp)))
    kappa_hyb  = kapB + (vw - cs)*ddk
    kappa_hyb += ((vw - cs)/(v_cj - cs))**3*(kapC - kapB - (v_cj - cs)*ddk)
    kappa[ty == 'hyb'] = kappa_hyb[ty == 'hyb']

    if   not mult_alp and not mult_vw: kappa = kappa[0, 0]
    elif not mult_alp: kappa = kappa[:, 0]
    elif not mult_vw:  kappa = kappa[0, :]

    return kappa

######################### COMPUTING DIAGNOSTIC PROFILES #########################

def w_to_lam(xis, ws, vw, alphan):

    '''
    Function that computes the energy density perturbations
    given the 1d profile of enthalpy using the bag equation
    of state.

    Reference is Appendix A of RoperPol:2025a

    Arguments:
        xis    -- array of xi
        ws     -- array of 1d enthalpies
        vw     -- wall velocity
        alphan -- value of the nucleation alpha

    Returns:
        lam    -- array of energy density perturbations
    '''

    lam        = 3/4*(ws - 1)
    inds       = np.where(xis < vw)[0]
    lam[inds] -= 3/4*alphan

    return lam

########## COMPUTING FUNCTIONS RELEVANT FOR VELOCITY SPECTRAL DENSITY ##########

'''
Main reference is: RoperPol:2025a
Other used references are: Hindmarsh:2016lnk, Hindmarsh:2019phv, RoperPol:2023dzg
'''

#### f' and l functions

def fp_z(xi, vs, z, lz=False, ls=[], multi=True, quiet=False):

    '''
    Function that computes the functions f'(z) and l(z)
    that describe the Fourier transform of the velocity field
    and the energy density perturbations for each expanding
    bubble.

    These functions provide the initial conditions used to
    compute the kinetic spectrum in the sound-wave regime
    according to the Sound-Shell Model.

    Arguments:
        xi    -- array of xi
        vs    -- array of 1d velocities
        z     -- array of z = k (t - t_n) where f' and l functions
                 are to be computed
        lz    -- option to compute l(z) (default is False)
        ls    -- array of energy density perturbations (used if lz
                 is True)
        multi -- option to use an array of wall velocities
        quiet -- option to avoid information printing
                 (default is False)
    '''

    xi_ij, z_ij = np.meshgrid(xi[1:], z, indexing='ij')
    zxi_ij      = z_ij*xi_ij
    j1_z        = np.sin(zxi_ij)/zxi_ij**2 - np.cos(zxi_ij)/zxi_ij
    # avoid division by zero
    j1_z[np.where(zxi_ij == 0)] = 0

    if lz:
        if ls == []:
            print('if lz is chosen you need to provide a l(xi) profile')
            lz = False
        j0_z = np.sin(zxi_ij)/zxi_ij
        j0_z[np.where(zxi_ij == 0)] = 1

    if multi:
        Nvws = np.shape(vs)[0]
        fpzs = np.zeros((Nvws, len(z)))
        if lz: lzs = np.zeros((Nvws, len(z)))

        for i in range(0, Nvws):
            v_ij, z_ij = np.meshgrid(vs[i, 1:], z, indexing='ij')
            try:    fpzs[i, :] = -4*np.pi*np.trapezoid(j1_z*xi_ij**2*v_ij, xi[1:], axis=0)
            except: fpzs[i, :] = -4*np.pi*np.trapz(j1_z*xi_ij**2*v_ij, xi[1:], axis=0)
            if lz:
                l_ij, z_ij = np.meshgrid(ls[i, 1:], z, indexing='ij')
                try:    lzs[i, :]  = 4*np.pi*np.trapezoid(j0_z*xi_ij**2*l_ij, xi[1:], axis=0)
                except: lzs[i, :]  = 4*np.pi*np.trapz(j0_z*xi_ij**2*l_ij, xi[1:], axis=0)

            if not quiet: print('vw ', i + 1, '/', Nvws, ' computed')

    else:

        v_ij, z_ij = np.meshgrid(vs[1:], z, indexing='ij')
        try:    fpzs = -4*np.pi*np.trapezoid(j1_z*xi_ij**2*v_ij, xi[1:], axis=0)
        except: fpzs = -4*np.pi*np.trapz(j1_z*xi_ij**2*v_ij, xi[1:], axis=0)
        if lz:
            l_ij, z_ij = np.meshgrid(ls[1:], z, indexing='ij')
            try:    lzs = 4*np.pi*np.trapezoid(j0_z*xi_ij**2*l_ij, xi[1:], axis=0)
            except: lzs = 4*np.pi*np.trapz(j0_z*xi_ij**2*l_ij, xi[1:], axis=0)

    if lz:
        return fpzs, lzs
    else:
        return fpzs

def Rstar_beta(vws=1., cs2=cs2_ref, corr=True):

    """
    Function that computes the ratio of the mean-bubble separation Rstar
    to the inverse nucleation rate parameter beta.
    """

    Rbeta = (8*np.pi)**(1/3)*vws
    if corr:
        cs = np.sqrt(cs2)
        Rbeta = Rbeta*np.maximum(1., cs/vws)

    return Rbeta
