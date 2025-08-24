"""
hydro_bubbles.py is a Python routine that contains functions
to study the 1D hydrodynamic solutions of expanding bubbles
produced from first-order phase transitions.

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/hydro_bubbles.py

.. note::
   For full documentation, visit `Read the Docs
   <https://cosmogw-manual.readthedocs.io/en/latest/hydro_bubbles.html>`_

To use it, first install `cosmoGW <https://pypi.org/project/cosmoGW>`_::

    pip install cosmoGW

Author
------
- **Alberto Roper Pol**
  (`alberto.roperpol@unige.ch <mailto:alberto.roperpol@unige.ch>`_)

Dates
-----
- Created: **01/02/2023**
- Updated: **21/08/2025**
  (release **cosmoGW 1.0**: https://pypi.org/project/cosmoGW)

Contributors
------------
- **Antonino Midiri**, **Simona Procacci**

References
----------
- [**RoperPol:2025a**]: A. Roper Pol, S. Procacci, A. S. Midiri,
  C. Caprini, "*Irrotational fluid perturbations from first-order phase
  transitions,*" in preparation

- [**Espinosa:2010hh**]: J. R. Espinosa, T. Konstandin, J. M. No, G. Servant,
  "*Energy Budget of Cosmological First-order Phase Transitions,*"
  JCAP **06** (2010) 028, `arXiv:1004.4187 <https://arxiv.org/abs/1004.4187>`_.

- [**Hindmarsh:2016lnk**]: M. Hindmarsh, "*Sound shell model for acoustic
  gravitational wave production at a first-order phase transition in
  the early Universe,*" Phys. Rev. Lett. **120** (2018) 7, 071301,
  `arXiv:1608.04735 <https://arxiv.org/abs/1608.04735>`_.

- [**Hindmarsh:2019phv**]: M. Hindmarsh, M. Hijazi, "*Gravitational waves from
  first order cosmological phase transitions in the Sound Shell Model,*"
  JCAP **12** (2019) 062,
  `arXiv:1909.10040 <https://arxiv.org/abs/1909.10040>`_.

- [**RoperPol:2023dzg**]: A. Roper Pol, S. Procacci, C. Caprini,
  "*Characterization of the gravitational wave spectrum from sound waves within
  the sound shell model,*" Phys. Rev. D **109**, 063531 (2024),
  `arXiv:2308.12943 <https://arxiv.org/abs/2308.12943>`_.

Comments
--------
The solutions are based in the 1D hydrodynamic descriptions
of Espinosa:2010hh and Hindmarsh:2019phv.

The details accompanying the code are provided in RoperPol:2023dzg
and RoperPol:2025a (appendix A).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cosmoGW.utils import (
    cs2_ref, Nxi_ref, vw_def, alpha_def, vw_hyb, alpha_hyb, vw_det, alpha_det,
    tol_ref, it_ref, Nvws_ref, Nxi2_ref, cols_ref,
    safe_trapezoid, reshape_output
)


def Chapman_Jouget(alp):

    r"""
    Compute the Chapman-Jouget wall velocity.

    The Chapman-Jouget velocity is the wall velocity at which the relative speed
    behind the wall becomes that of the speed of sound. It separates detonations
    and supersonic deflagrations.

    .. math::
        v_{\rm CJ} = \frac{1}{\sqrt{3}(1 + \alpha)}
        \left[1 + \sqrt{\alpha(2 + 3\alpha)}\right].

    Reference: Eq. 39 of Espinosa:2010hh

    Parameters
    ----------
    alp : float
        Alpha at the + side of the wall (:math:`\alpha_{+}`).

    Returns
    -------
    vcJ : float
        Chapman-Jouget speed :math:`v_{\rm CJ}`.

    References
    ----------
    Espinosa:2010hh (eq. 39) and Hindmarsh:2019phv (eq. B19).
    Note that Eq. B19 of Hindmarsh:2019phv has a typo.
    """
    return 1.0 / np.sqrt(3) / (1 + alp) * (1 + np.sqrt(alp * (2 + 3 * alp)))


def type_nucleation(vw, alp, cs2=cs2_ref):

    """
    Determine the type of bubble solution.

    Returns 'def' for subsonic deflagrations, 'hyb' for
    supersonic deflagrations, or 'det' for detonations.

    Parameters
    ----------
    vw : float or array_like
        Bubble wall speed.
    alp : float or array_like
        Strength of the phase transition at the + side of the wall
        (:math:`\\alpha_{+}`).
    cs2 : float, optional
        Square of the speed of sound (default 1/3).

    Returns
    -------
    ty : str or ndarray
        Type of solution ('def', 'hyb', 'det').
    """

    mult_vw = isinstance(vw, (list, tuple, np.ndarray))
    mult_alp = isinstance(alp, (list, tuple, np.ndarray))
    cs = np.sqrt(cs2)

    if not mult_vw:
        vw = np.array([vw])
    if not mult_alp:
        alp = np.array([alp])
    v_cj = Chapman_Jouget(alp)

    ty = np.full((len(vw), len(alp)), 'hyb')
    vw, v_cj = np.meshgrid(vw, v_cj, indexing='ij')
    ty[vw < cs] = 'def'
    ty[vw > v_cj] = 'det'

    ty = reshape_output(ty, mult_a=mult_vw, mult_b=mult_alp)

    return ty


# 1D HYDRO SOLUTIONS UNDER SPHERICAL SYMMETRY
def _Lor_mu(v, vw):

    r"""
    Lorentz transform of the velocity v in the reference frame of the wall.

    .. math::
        \mu(v_w, v) = \frac{v_w - v}{1 - v_w v}

    Parameters
    ----------
    v : float or array_like
        Velocity.
    vw : float
        Wall velocity.

    Returns
    -------
    mu : float or ndarray
        Lorentz transformed velocity.

    Reference
    ---------
    Eq. B12 of Hindmarsh:2019phv
    """

    return (vw - v) / (1 - v * vw)


# hydrodynamic shocks with no vacuum energy
def v_shock(xi):

    """
    Compute the shock velocity at the - side of the shock.

    Computed from the :math:`v_+ v_- = 1/3` condition.

    Parameters
    ----------
    xi : float or array_like
        Self-similar variable.

    Returns
    -------
    vsh : float or ndarray
        Shock velocity.
    """

    return (3 * xi ** 2 - 1) / (2 * xi)


def w_shock(xi):

    """
    Compute the ratio of enthalpies w-/w+ across the shock.

    Computed from the :math:`v_+ v_- = 1/3` condition.

    Parameters
    ----------
    xi : float or array_like
        Self-similar variable.

    Returns
    -------
    wsh : float or ndarray
        Ratio of enthalpies across the shock.
    """

    return (9 * xi ** 2 - 1) / (3 * (1 - xi ** 2))


# differential equation for the velocity radial profile
def _xi_o_v(xi, v, cs2=cs2_ref):

    r"""
    Characterize the 1D hydro equation under radial symmetry.

    Returns the value of :math:`d\xi/dv` used to solve the equation in
    :func:`compute_xi_from_v` using RK4.

    Parameters
    ----------
    xi : float
        Self-similar r/t.
    v : float
        1D velocity profile.
    cs2 : float, optional
        Square of the speed of sound (default 1/3).

    Returns
    -------
    f : float
        Value of :math:`f(\xi, v) = d\xi/dv`.

    Reference
    ---------
    Eq. 27 of Espinosa:2010hh
    """

    gamma2 = 1 / (1 - v ** 2)
    mu = _Lor_mu(v, xi)
    f = xi * gamma2 * (1 - xi * v) * (mu ** 2 / cs2 - 1) / (2 * v)
    return f


def compute_xi_from_v(v, xi0, cs2=cs2_ref, shock=False):

    r"""
    Compute the solution :math:`\xi(v)` using a 4th-order Runge-Kutta scheme.

    Since :math:`dv/d\xi` has a singularity, it is necessary to compute
    :math:`\xi(v)` and then invert each of the solutions to have the full
    dynamical solution. For the physical solution, computing :math:`v(\xi)`
    is more practical.

    Parameters
    ----------
    v : array_like
        Velocity array.
    xi0 : float
        Position where boundary condition is known
        (first value in velocity array).
    cs2 : float, optional
        Square of the speed of sound (default 1/3).
    shock : bool, optional
        Option to stop the integration when a shock is found
        (as in deflagrations).

    Returns
    -------
    xi : ndarray
        Self-similar array.
    sh : bool
        Boolean determining if a shock is formed.
    indsh : int
        Index determining the position of the shock (-1 if no shock).
    """

    xi = np.zeros(len(v)) + xi0
    sh = False
    indsh = -1

    for i in range(0, len(v) - 1):
        dv = v[i + 1] - v[i]
        k1 = _xi_o_v(xi[i], v[i])
        k2 = _xi_o_v(xi[i] + dv * k1 / 2, 0.5 * (v[i + 1] + v[i]))
        k3 = _xi_o_v(xi[i] + dv * k2 / 2, 0.5 * (v[i + 1] + v[i]))
        k4 = _xi_o_v(xi[i] + dv * k3, v[i + 1])
        xi_new = xi[i] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * dv
        if shock:
            xi_sh = xi_new
            v_sh = v_shock(xi_sh)
            if v[i + 1] < v_sh:
                xi[i + 1:] = xi_sh
                sh = True
                indsh = i
                break
        xi[i + 1] = xi_new
        if xi_new > 1:
            xi[i + 1] = 1
    return xi, sh, indsh


def _integrand_w(xi, v, cs2=cs2_ref):

    r"""
    Compute the integrand for the integration of :math:`dw/d\xi`
    (enthalpy) equation, as a function of the solution :math:`v(\xi)`.


    Parameters
    ----------
    xi : float
        Self-similar r/t.
    v : float
        1D velocity profile.
    cs2 : float, optional
        Square of the speed of sound (default 1/3).

    Returns
    -------
    float
        Integrand value.

    Reference
    ---------
    Eq. (29) of Espinosa:2010hh
    """

    return (1.0 + 1.0 / cs2) / (1.0 - v ** 2) * _Lor_mu(v, xi)


def compute_w(v, xi, cs2=cs2_ref):

    """
    Compute the enthalpy from the solution of the velocity profile.

    Parameters
    ----------
    xi : array_like
        Self-similar r/t.
    v : array_like
        1D velocity profile.
    cs2 : float, optional
        Square of the speed of sound (default 1/3).

    Returns
    -------
    w : ndarray
        Enthalpy profile.
    """

    w = np.zeros(len(v)) + 1
    ss = 0
    for i in range(0, len(v) - 1):
        ff_ip = _integrand_w(xi[i + 1], v[i + 1], cs2=cs2)
        ff_i = _integrand_w(xi[i], v[i], cs2=cs2)
        ss += 0.5 * (v[i + 1] - v[i]) * (ff_ip + ff_i)
        w[i + 1] = np.exp(ss)

    return w


# SOLVE FOR THE DIFFERENT TYPE OF BOUNDARY CONDITIONS
# MATCHING CONDITIONS ACROSS DISCONTINUITIES
def _vp_tilde_from_vm_tilde(vw, alpha, plus=True, sg='plus'):

    """
    Compute the velocity on one side of the bubble wall from the velocity on
    the other side.

    This function computes either the symmetric phase (+) or broken
    phase (-) velocity, defined in the wall reference frame, across the wall,
    as a function of the velocity at the opposite side of the wall,
    using the matching conditions.

    Parameters
    ----------
    vw : float
        Velocity imposed at one side of the wall (usually the wall velocity,
        but can be cs for hybrids).
    alpha : float
        Phase transition strength at the symmetric phase
        (can differ from nucleation temperature value).
    plus : bool, optional
        If True, use the positive branch of the matching condition
        (default: True).
    sg : str, optional
        If 'plus', compute v+ from v- = vw; if 'minus', compute v- from v+ = vw
        (default: 'plus').

    Returns
    -------
    vp_vm : float
        Computed v+ or v- from the value at the other side of the bubble wall.

    Reference
    ---------
    Eqs. B6 and B7 of Hindmarsh:2019phv
    """

    if sg == 'plus':
        a1 = 1 / 3 / vw + vw
        a2 = np.sqrt((1 / 3 / vw - vw) ** 2 + 4 * alpha ** 2 + 8 / 3 * alpha)
        aa = 0.5 / (1 + alpha)
    else:
        a1 = (1 + alpha) * vw + (1 - 3 * alpha) / 3 / vw
        a2 = np.sqrt(((1 + alpha) * vw + (1 - 3 * alpha) / 3 / vw) ** 2 - 4 / 3)
        aa = 0.5

    a = a1 + a2 if plus else a1 - a2
    vp_vm = aa * a
    return vp_vm


def vplus_vminus(alpha, vw=1.0, ty='det', cs2=cs2_ref):

    """
    Compute :math:`v_+` and :math:`v_-` (in the wall frame)
    for different types of solutions.

    This function returns the boundary velocities for
    deflagrations, detonations, and hybrids, as required for
    hydrodynamic matching.

    Parameters
    ----------
    alpha : float
        Value of alpha at the + side of the wall.
    vw : float, optional
        Wall velocity.
    ty : str, optional
        Type of solution ('def', 'det', 'hyb').
    cs2 : float, optional
        Speed of sound squared (default is 1/3).

    Returns
    -------
    vplus : float or ndarray
        Velocity on the + side of the wall (in wall frame), :math:`v_+`.
    vminus : float or ndarray
        Velocity on the - side of the wall (in wall frame), :math:`v_-`.
    """
    cs = np.sqrt(cs2)
    if not isinstance(ty, (list, tuple, np.ndarray)):
        if ty == 'det':
            vplus = vw
            vminus = _vp_tilde_from_vm_tilde(vw, alpha, plus=True, sg='minus')
        elif ty == 'def':
            vminus = vw
            vplus = _vp_tilde_from_vm_tilde(vw, alpha, plus=False, sg='plus')
        elif ty == 'hyb':
            vminus = cs
            vplus = _vp_tilde_from_vm_tilde(cs, alpha, plus=False, sg='plus')
    else:
        vplus = np.zeros(len(vw))
        vminus = np.zeros(len(vw))
        inds_det = np.where(ty == 'det')
        inds_def = np.where(ty == 'def')
        inds_hyb = np.where(ty == 'hyb')
        vplus[inds_det] = vw[inds_det]
        vminus[inds_det] = _vp_tilde_from_vm_tilde(
            vw, alpha, plus=True, sg='minus'
        )[inds_det]
        vplus[inds_def] = _vp_tilde_from_vm_tilde(
            vw, alpha, plus=False, sg='plus'
        )[inds_def]
        vminus[inds_def] = vw[inds_def]
        vplus[inds_hyb] = _vp_tilde_from_vm_tilde(
            cs, alpha, plus=False, sg='plus'
        )
        vminus[inds_hyb] = cs
    return vplus, vminus


# function that computes the detonation part of the solutions
def _det_sol(v0, xi0, cs2=cs2_ref, Nxi=Nxi_ref, zero_v=-4):

    """
    Compute a detonation solution with boundary condition v0 at xi0.

    The solution is computed until the velocity drops by four
    orders of magnitude.

    Parameters
    ----------
    v0 : float
        Value of velocity at the boundary.
    xi0 : float
        Position of the boundary.
    cs2 : float, optional
        Speed of sound squared (default is 1/3).
    Nxi : int, optional
        Number of discretization points in xi.
    zero_v : float, optional
        Logarithmic decrement for velocity (default -4).

    Returns
    -------
    xis : ndarray
        Array of xi.
    vs : ndarray
        Array of 1D velocities.
    ws : ndarray
        Array of 1D enthalpies.
    """

    # included option to initialize with multiple vws
    if not isinstance(xi0, (list, tuple, np.ndarray)):
        vs = np.logspace(np.log10(v0), np.log10(v0) + zero_v, int(Nxi))
        xis, _, _ = compute_xi_from_v(vs, xi0, cs2=cs2, shock=False)
        ws = compute_w(vs, xis, cs2=cs2)
        inds_sort = np.argsort(xis)
        xis = xis[inds_sort]
        vs = vs[inds_sort]
        ws = ws[inds_sort]

    else:
        xis = np.zeros((len(xi0), int(Nxi)))
        vs = np.zeros((len(xi0), int(Nxi)))
        ws = np.zeros((len(xi0), int(Nxi)))
        for i in range(len(xi0)):
            vs[i, :] = np.logspace(
                np.log10(v0), np.log10(v0) + zero_v, int(Nxi)
            )
            xis[i, :], _, _ = compute_xi_from_v(
                vs[i, :], xi0[i], cs2=cs2, shock=False
            )
            ws[i, :] = compute_w(vs[i, :], xis[i, :], cs2=cs2)
            inds_sort = np.argsort(xis[i, :])
            xis[i, :] = xis[i, inds_sort]
            vs[i, :] = vs[i, inds_sort]
            ws[i, :] = ws[i, inds_sort]

    return xis, vs, ws


# function that computes the deflagration part of the solutions #######
def _def_sol(v0, xi0, cs2=cs2_ref, Nxi=Nxi_ref, shock=True, zero_v=-4):

    """
    Compute a deflagration solution with boundary condition v0 at xi0.

    The solution is computed until a shock is formed (if shock=True).

    Parameters
    ----------
    v0 : float
        Value of velocity at the boundary.
    xi0 : float
        Position of the boundary.
    cs2 : float, optional
        Speed of sound squared (default is 1/3).
    Nxi : int, optional
        Number of discretization points in xi.
    shock : bool, optional
        Stop calculation once a shock is formed (default True).
    zero_v : float, optional
        Logarithmic decrement for velocity (default -4).

    Returns
    -------
    xis : ndarray
        Array of xi.
    vs : ndarray
        Array of 1D velocities.
    ws : ndarray
        Array of 1D enthalpies.
    xi_sh : float
        Position of the shock.
    sh : bool
        Boolean, True if a shock forms.
    """

    vs = np.logspace(np.log10(v0), np.log10(v0) + zero_v, int(Nxi))
    cs = np.sqrt(cs2)
    xi_sh = cs
    sh = False
    v_sh = 0

    if shock:
        xiss, sh, indsh = compute_xi_from_v(vs, xi0, cs2=cs2, shock=True)
        xi_sh = xiss[indsh]
        if not sh:
            xi_sh = cs
        v_sh = v_shock(xi_sh)
        xis = np.linspace(xi0, xi_sh, int(Nxi) + 1)
        xis = xis[:int(Nxi) - 1]
        vs = np.interp(xis, xiss, vs)
        vs = np.append(vs, v_sh)
        xis = np.append(xis, xi_sh)

    else:
        xis, sh, indsh = compute_xi_from_v(vs, xi0, cs2=cs2, shock=False)
        xi_sh = xis[indsh]

    ws = compute_w(vs, xis, cs2=cs2)
    w_sh = w_shock(xi_sh)
    ws = ws * w_sh / ws[-1]

    return xis, vs, ws, xi_sh, sh


def compute_def(vw=vw_def, alpha=alpha_def, cs2=cs2_ref, Nxi=Nxi_ref,
                shock=True):

    """
    Compute the solutions for a subsonic deflagration 1D profile given
    vw and alpha.

    Uses :func:`_def_sol` to compute the velocity and enthalpy profiles.

    Parameters
    ----------
    vw : float, optional
        Wall velocity (default is 0.5).
    alpha : float, optional
        Strength of the phase transition (default is 0.263).
    cs2 : float, optional
        Speed of sound squared (default is 1/3).
    Nxi : int, optional
        Number of discretization points in xi.
    shock : bool, optional
        Stop calculation once a shock is formed (default is True).

    Returns
    -------
    xis : ndarray
        Array of xi.
    vs : ndarray
        Array of 1D velocities.
    ws : ndarray
        Array of 1D enthalpies.
    xi_sh : float
        Position of the shock.
    sh : bool
        Boolean, True if a shock forms.
    w_pl : float
        Plus value of the enthalpy across the bubble wall.
    w_m : float
        Minus value of the enthalpy across the bubble wall.
    """

    # relative velocity at + is computed from \tilde v- = \xi_w
    vrels, _ = vplus_vminus(alpha, vw=vw, ty='def')
    # Lorentz boosted v plus
    vpl = _Lor_mu(vrels, vw)

    xis, vs, ws, xi_sh, sh = _def_sol(vpl, vw, cs2=cs2, Nxi=Nxi, shock=shock)

    # values at both sides of the bubble wall
    w_pl = ws[0]
    w_m = w_pl * vrels / (1 - vrels ** 2) / vw * (1 - vw ** 2)

    return xis, vs, ws, xi_sh, sh, w_pl, w_m


def compute_hyb(vw=vw_hyb, alpha=alpha_hyb, cs2=cs2_ref, Nxi=Nxi_ref,
                shock=True):

    """
    Compute the solutions for a supersonic deflagration 1D profile given
    vw and alpha.

    Uses :func:`_det_sol` and :func:`_def_sol` to compute velocity and
    enthalpy profiles.

    Parameters
    ----------
    vw : float, optional
        Wall velocity (default is 0.7).
    alpha : float, optional
        Strength of the phase transition (default is 0.052).
    cs2 : float, optional
        Speed of sound squared (default is 1/3).
    Nxi : int, optional
        Number of discretization points in xi.
    shock : bool, optional
        Stop calculation once a shock is formed (default is True).

    Returns
    -------
    xis : ndarray
        Array of xi.
    vs : ndarray
        Array of 1D velocities.
    ws : ndarray
        Array of 1D enthalpies.
    xi_sh : float
        Position of the shock.
    sh : bool
        Boolean, True if a shock forms.
    w_pl : float
        Plus value of the enthalpy across the bubble wall.
    w_m : float
        Minus value of the enthalpy across the bubble wall.
    """

    cs = np.sqrt(cs2)

    # relative velocity at + is computed from \tilde v- = cs
    vrels, _ = vplus_vminus(alpha, cs2=cs2, ty='hyb')
    vpl = _Lor_mu(vrels, vw)
    vm = _Lor_mu(cs, vw)

    # compute deflagration solution
    xis, vs, ws, xi_sh, sh = _def_sol(
       vpl, vw, cs2=cs2, Nxi=int(Nxi / 2), shock=shock
    )

    # compute detonation solution
    xis2, vs2, ws2 = _det_sol(vm, vw, cs2=cs2, Nxi=int(Nxi / 2))
    # ratio of w+ over w- across the bubble wall
    w_pl = ws[0]
    w_m = w_pl * vrels / (1 - vrels ** 2) * (1 - cs2) / cs
    ws2 *= w_m

    xis = np.append(xis2, xis)
    vs = np.append(vs2, vs)
    ws = np.append(ws2, ws)

    return xis, vs, ws, xi_sh, sh, w_pl, w_m


def compute_det(vw=vw_det, alpha=alpha_det, cs2=cs2_ref, Nxi=Nxi_ref):

    """
    Compute the solutions for a detonation 1D profile given vw and alpha.

    Uses :func:`_det_sol` to compute velocity and enthalpy profiles.

    Parameters
    ----------
    vw : float, optional
        Wall velocity (default is 0.77).
    alpha : float, optional
        Strength of the phase transition (default is 0.091).
    cs2 : float, optional
        Speed of sound squared (default is 1/3).
    Nxi : int, optional
        Number of discretization points in xi.

    Returns
    -------
    xis : ndarray
        Array of xi.
    vs : ndarray
        Array of 1D velocities.
    ws : ndarray
        Array of 1D enthalpies.
    xi_sh : float
        Position of the shock (set to vw for detonations).
    sh : bool
        Boolean, always False for detonations.
    w_pl : float
        Plus value of the enthalpy across the bubble wall.
    w_m : float
        Minus value of the enthalpy across the bubble wall.
    """

    # relative velocity at - is computed from \tilde v+ = \xi_w
    _, vrels = vplus_vminus(alpha, vw=vw, ty='det')

    # Lorentz boosted v minus
    vm = _Lor_mu(vrels, vw)
    w_m = vw / (1 - vw ** 2) / vrels * (1 - vrels ** 2)
    w_pl = 1

    xis, vs, ws = _det_sol(vm, vw, cs2=cs2, Nxi=Nxi)
    ws *= w_m

    # no shock is formed in detonations, so xi_sh is set to vw
    # and sh to False
    xi_sh = vw
    sh = False

    return xis, vs, ws, xi_sh, sh, w_pl, w_m


def compute_alphan(vw=vw_def, alpha_obj=alpha_def, tol=tol_ref, cs2=cs2_ref,
                   quiet=False, max_it=it_ref, Nxi=Nxi_ref, ty='def'):

    r"""
    Iteratively compute the value of :math:`\alpha_+` corresponding to
    a target :math:`\alpha`.

    Uses the 1D profile of w and Newton-Raphson update to find
    :math:`\alpha_+` that gives the correct :math:`\alpha`.

    Parameters
    ----------
    vw : float
        Wall velocity.
    alpha_obj : float
        Target value of alpha at nucleation temperature.
    tol : float, optional
        Relative tolerance for convergence (default is 1e-5).
    cs2 : float, optional
        Speed of sound squared (default is 1/3).
    quiet : bool, optional
        If True, suppress debug output.
    max_it : int, optional
        Maximum number of iterations (default is 30).
    Nxi : int, optional
        Number of discretization points in xi.
    ty : str, optional
        Type of solution ('def' or 'hyb').

    Returns
    -------
    xis0 : ndarray
        Array of xi.
    vvs0 : ndarray
        Array of velocities.
    wws0 : ndarray
        Array of enthalpies.
    xi_sh : float
        Position of the shock.
    sh : bool
        Boolean, True if a shock forms.
    w_pl : float
        Plus value of the enthalpy.
    w_m : float
        Minus value of the enthalpy.
    alpha_n : float
        Converged alpha at nucleation.
    alp_plus : float
        Value of alpha_+ leading to alpha_obj.
    conv : bool
        True if the algorithm has converged.
    """

    alp_plus = alpha_obj
    j = 0
    conv = False
    while not conv and j < max_it:
        j += 1
        if ty == 'hyb':
            xis0, vvs0, wws0, xi_sh, sh, w_pl, w_m = compute_hyb(
                vw=vw, alpha=alp_plus, cs2=cs2, Nxi=Nxi, shock=True)
        elif ty == 'def':
            xis0, vvs0, wws0, xi_sh, sh, w_pl, w_m = compute_def(
                vw=vw, alpha=alp_plus, cs2=cs2, Nxi=Nxi, shock=True)
        alpha_n = alp_plus * w_pl
        if abs(alpha_n - alpha_obj) / alpha_obj < tol:
            conv = True
        else:
            alp_plus = alpha_obj / w_pl
        if not quiet:
            print('iteration', j, 'alpha', alpha_n)
            print('iteration', j, 'new guess', alp_plus)
    if not quiet:
        print(j, 'iterations for vw=', vw, ' and alpha= ', alpha_obj)
        print('alpha:', alpha_n, ', alpha_+:', alp_plus)

    return xis0, vvs0, wws0, xi_sh, sh, w_pl, w_m, alpha_n, alp_plus, conv


def compute_profiles_vws_multalp(
    alphas, vws=None, cs2=cs2_ref, Nvws=Nvws_ref, Nxi=Nxi_ref, Nxi2=Nxi2_ref,
    alphan=True, quiet=True, tol=tol_ref, max_it=it_ref, lam=False, eff=False
):
    """
    Compute velocity and enthalpy profiles for an array of alpha and
    wall velocities.

    This function wraps :func:`compute_profiles_vws` for multiple alpha values
    and wall velocities, optionally computing efficiency factors.

    Parameters
    ----------
    alphas : array_like
        Array of nucleation temperature alpha values.
    vws : array_like, optional
        Array of wall velocities. If None, uses a default linspace.
    cs2 : float, optional
        Square of the speed of sound (default is 1/3).
    Nvws : int, optional
        Number of wall velocities if vws is not provided.
    Nxi : int, optional
        Number of discretization points in xi.
    Nxi2 : int, optional
        Number of discretization points in xi out of the profiles.
    alphan : bool, optional
        If True, input alpha is at nucleation temperature; if False,
        input is alpha+.
    quiet : bool, optional
        If True, suppress debug output.
    tol : float, optional
        Tolerance for convergence of alpha+ (default is 1e-5).
    max_it : int, optional
        Maximum number of iterations for alpha+ convergence.
    lam : bool, optional
        If True, compute energy perturbations lambda instead of enthalpy.
    eff : bool, optional
        If True, also compute efficiency factors kappa and omega.

    Returns
    -------
    xis : ndarray
        Array of xi positions.
    vvs : ndarray
        Array of velocities.
    wws : ndarray
        Array of enthalpies (or energy density perturbations if lam is True).
    alphas_n : ndarray
        Array of nucleation alphas (if input is alpha+, returns alpha+).
    conv : ndarray
        Array of booleans for alpha+ convergence.
    shocks : ndarray
        Array of booleans for shock formation.
    xi_shocks : ndarray
        Positions of shocks (or xi_front if no shock).
    wms : ndarray
        Enthalpy values at - side of the wall.
    kappas : ndarray, optional
        Ratio of kinetic to vacuum energy density (if eff is True).
    omegas : ndarray, optional
        Ratio of energy density perturbations to vacuum energy density
        (if eff is True).
    """
    if vws is None or len(np.shape(vws)) == 1 and len(vws) == 0:
        vws = np.linspace(0.1, 0.99, Nvws)
    xis = np.linspace(0, 1, int(Nxi) + int(Nxi2))

    n_vws = len(vws)
    n_alphas = (
        len(alphas)
        if isinstance(alphas, (list, tuple, np.ndarray))
        else 1
    )

    vvs = np.zeros((n_vws, n_alphas, len(xis)))
    wws = np.ones((n_vws, n_alphas, len(xis)))
    alphas_n = np.zeros((n_vws, n_alphas))
    conv = np.ones((n_vws, n_alphas))
    shocks = np.zeros((n_vws, n_alphas))
    xi_shocks = np.zeros((n_vws, n_alphas))
    wms = np.zeros((n_vws, n_alphas))

    if eff:
        kappas = np.zeros((n_vws, n_alphas))
        omegas = np.zeros((n_vws, n_alphas))

    for i in range(n_alphas):
        if not quiet:
            print(
                'Computing alpha = %s out of %d' %
                (alphas[i] if n_alphas > 1 else alphas, n_alphas)
            )
        results = compute_profiles_vws(
            alphas[i] if n_alphas > 1 else alphas,
            vws=vws, cs2=cs2, Nxi=Nxi, Nxi2=Nxi2, plot=False,
            alphan=alphan, quiet=True, tol=tol, max_it=max_it, lam=lam, eff=eff
        )
        if eff:
            (
                xis, vvs[:, i, :], wws[:, i, :], alphas_n[:, i], conv[:, i],
                shocks[:, i], xi_shocks[:, i], wms[:, i], kappas[:, i],
                omegas[:, i]
            ) = results
        else:
            (xis, vvs[:, i, :], wws[:, i, :], alphas_n[:, i], conv[:, i],
             shocks[:, i], xi_shocks[:, i], wms[:, i]) = results

    if eff:
        return (
            xis, vvs, wws, alphas_n, conv, shocks, xi_shocks, wms,
            kappas, omegas
        )
    else:
        return xis, vvs, wws, alphas_n, conv, shocks, xi_shocks, wms


def compute_profiles_vws(
    alpha, vws=None, cs2=cs2_ref, Nvws=Nvws_ref, Nxi=Nxi_ref, Nxi2=Nxi2_ref,
    plot=False, plot_v='v', cols=None, alphan=True, quiet=True, tol=tol_ref,
    max_it=it_ref, ls='solid', alp=1.0, lam=False, legs=False,
    fs_lg=14, st_lg=2, eff=False, save=False, dec_vw=1,
    ress='results/1d_profiles', strs_vws=None, str_alp=None
):

    """
    Compute velocity and enthalpy profiles for a given alpha and a range
    of wall velocities.

    This function solves the hydrodynamic equations for bubble expansion
    and returns the velocity and enthalpy (or energy density perturbation)
    profiles for each wall velocity.

    Parameters
    ----------
    alpha : float
        Nucleation temperature alpha.
    vws : array_like, optional
        Array of wall velocities. If None, uses a default linspace.
    cs2 : float, optional
        Square of the speed of sound (default is 1/3).
    Nvws : int, optional
        Number of wall velocities if vws is not provided.
    Nxi : int, optional
        Number of discretization points in xi.
    Nxi2 : int, optional
        Number of discretization points in xi out of the profiles.
    plot : bool, optional
        If True, plot the resulting profiles.
    plot_v : str, optional
        What to plot: 'v', 'w', or 'both' (default 'v').
    cols : list, optional
        List of colors to use for plotting.
    alphan : bool, optional
        If True, input alpha is at nucleation temperature; if False,
        input is alpha+.
    quiet : bool, optional
        If True, suppress debug output.
    tol : float, optional
        Tolerance for convergence of alpha+ (default is 1e-5).
    max_it : int, optional
        Maximum number of iterations for alpha+ convergence.
    ls : str, optional
        Line style for plots.
    alp : float, optional
        Opacity for plots.
    lam : bool, optional
        If True, compute energy perturbations lambda instead of enthalpy.
    legs : bool, optional
        Whether to show legend in plots.
    fs_lg : int, optional
        Font size for legend.
    st_lg : int, optional
        Legend style.
    eff : bool, optional
        If True, also compute efficiency factors kappa and omega.
    save : bool, optional
        If True, save results to CSV files.
    dec_vw : int, optional
        Decimal precision for wall velocity in filenames.
    ress : str, optional
        Directory to save results.
    strs_vws : list, optional
        Custom strings for wall velocities in filenames.
    str_alp : list, optional
        Custom strings for alpha in filenames.

    Returns
    -------
    xis : ndarray
        Array of xi positions.
    vvs : ndarray
        Array of velocities.
    wws : ndarray
        Array of enthalpies (or energy density perturbations if lam is True).
    alphas_n : ndarray
        Array of nucleation alphas (if input is alpha+, returns alpha+).
    conv : ndarray
        Array of booleans for alpha+ convergence.
    shocks : ndarray
        Array of booleans for shock formation.
    xi_shocks : ndarray
        Positions of shocks (or xi_front if no shock).
    wms : ndarray
        Enthalpy values at - side of the wall.
    kappas : ndarray, optional
        Ratio of kinetic to vacuum energy density (if eff is True).
    omegas : ndarray, optional
        Ratio of energy density perturbations to vacuum energy density
        (if eff is True).
    """

    if vws is None or len(vws) == 0:
        vws = np.linspace(0.1, 0.99, Nvws)
    vCJ = Chapman_Jouget(alp=alpha)
    cs = np.sqrt(cs2)
    xis = np.linspace(0, 1, int(Nxi + Nxi2))
    vvs = np.zeros((len(vws), len(xis)))
    wws = np.ones((len(vws), len(xis)))
    alphas_n = np.zeros(len(vws))
    conv = np.ones(len(vws))
    kappas = np.zeros(len(vws))
    omegas = np.zeros(len(vws))
    shocks = np.zeros(len(vws))
    xi_shocks = np.zeros(len(vws))
    wms = np.zeros(len(vws))

    if plot_v == 'both' and plot:
        plt.figure(1)
        plt.figure(2)
    if cols is None or len(cols) == 0:
        cols = cols_ref

    for i, vw in enumerate(vws):
        ty = type_nucleation(vw, alpha, cs2=cs2)
        result = _compute_profiles_block(
            ty, vw, alpha, alphan, tol, cs2, quiet, max_it, Nxi
        )
        # Unpack mandatory values, handle optional ones
        xis0, vvs0, wws0, xi_sh, sh, w_pl, w_m, *rest = result
        alpha_plus = rest[0] if len(rest) > 0 else None
        conv_i = rest[1] if len(rest) > 1 else None

        _assign_results(
            i, ty, xis, vw, xi_sh, cs, vvs, wws, w_m, vvs0, wws0, xis0, sh,
            xi_shocks, shocks, wms, alpha_plus, conv_i, alphas_n, conv
        )

    # compute efficiency of energy density production
    if eff:
        kappas[i], omegas[i] = kappas_from_prof(
            vw, alpha, xis, wws[i, :], vvs[i, :]
        )

    # compute mean energy density from enthalpy if lam is True
    if lam:
        alp_lam = alpha if alphan else alphas_n[i]
        wws[i, :] = w_to_lam(xis, wws[i, :], vw, alp_lam)

    if plot:
        _plot_profiles(i, vw, xis, vvs, wws, cols, ls, alp, st_lg, plot_v,
                       legs, fs_lg, cs, vCJ, lam)

    if save and alphan:
        _save_profiles(i, alpha, xis, vvs, wws, alphas_n, shocks, xi_shocks,
                       wms, str_alp, strs_vws, dec_vw, ress, vw)

    if eff:
        return (
            xis, vvs, wws, alphas_n, conv, shocks, xi_shocks, wms,
            kappas, omegas
        )
    else:
        return xis, vvs, wws, alphas_n, conv, shocks, xi_shocks, wms


def _compute_profiles_block(ty, vw, alpha, alphan, tol, cs2,
                            quiet, max_it, Nxi):
    '''
    Compute the profiles for a given type of nucleation, called
    from compute_profiles_vws.
    '''
    if ty == 'def':
        # iteratively compute the real alpha_+ leading to alpha
        if alphan:
            return compute_alphan(
                vw=vw, alpha_obj=alpha, tol=tol, cs2=cs2, quiet=quiet,
                max_it=max_it, Nxi=Nxi, ty='def'
            )
        return compute_def(vw=vw, alpha=alpha, cs2=cs2, Nxi=Nxi, shock=True)
    if ty == 'hyb':
        if alphan:
            return compute_alphan(
                vw=vw, alpha_obj=alpha, tol=tol, cs2=cs2, quiet=quiet,
                max_it=max_it, Nxi=Nxi, ty='hyb'
            )
        return compute_hyb(vw=vw, alpha=alpha, cs2=cs2, Nxi=Nxi, shock=True)
    return compute_det(vw=vw, alpha=alpha, cs2=cs2, Nxi=Nxi)


def _assign_results(
        i, ty, xis, vw, xi_sh, cs, vvs, wws, w_m, vvs0, wws0, xis0, sh,
        xi_shocks, shocks, wms, alpha_plus, conv_i, alphas_n, conv
):
    '''
    Assign results to arrays from 1d profiles, called from
    compute_profiles_vws.
    '''

    if ty == 'def':
        inds = np.where((xis >= vw) & (xis <= xi_sh))[0]
        inds2 = np.where(xis < vw)[0]
    elif ty == 'hyb':
        inds = np.where((xis >= cs) & (xis <= xi_sh))[0]
        if xi_sh == vw:
            inds = np.append(inds, np.where((xis <= xi_sh))[-1] + 1)
        inds2 = np.where(xis < cs)[0]
    else:
        inds = np.where((xis >= cs) & (xis <= vw))[0]
        inds2 = np.where(xis < cs)[0]

    # Interpolate and assign velocity and enthalpy profiles
    vvs[i, inds] = np.interp(xis[inds], xis0, vvs0)
    wws[i, inds] = np.interp(xis[inds], xis0, wws0)
    wws[i, inds2] = w_m

    # Assign shock and wall values
    xi_shocks[i] = xi_sh
    shocks[i] = sh
    wms[i] = w_m

    # Assign alpha_plus and convergence if available
    if alpha_plus is not None:
        alphas_n[i] = alpha_plus
    if conv_i is not None:
        conv[i] = conv_i


def _plot_profiles(i, vw, xis, vvs, wws, cols, ls, alp, st_lg, plot_v,
                   legs, fs_lg, cs, vCJ, lam):

    '''
    Plot profiles for a given type of nucleation, called from
    compute_profiles_vws.
    '''
    j = i % len(cols)
    str_lg = r'$\xi_w=%.2f$' % vw if st_lg == 2 else r'$\xi_w=%.1f$' % vw
    if plot_v == 'v':
        plt.plot(xis, vvs[i, :], color=cols[j], ls=ls, alpha=alp, label=str_lg)
    if plot_v == 'w':
        plt.plot(xis, wws[i, :], color=cols[j], ls=ls, alpha=alp, label=str_lg)
    if plot_v == 'both':
        plt.figure(1)
        plt.plot(xis, vvs[i, :], color=cols[j], ls=ls, alpha=alp, label=str_lg)
        plt.figure(2)
        plt.plot(xis, wws[i, :], color=cols[j], ls=ls, alpha=alp, label=str_lg)
    # Axis labels and legends
    if plot_v == 'v' or plot_v == 'both':
        if plot_v == 'both':
            plt.figure(1)
        plt.ylim(-.05, 1.05)
        plt.ylabel(r'$ v_{\rm ip} (\xi)$')
    if plot_v == 'w' or plot_v == 'both':
        if plot_v == 'both':
            plt.figure(2)
        plt.ylim(0, 5)
        plt.ylabel(r'$ \lambda_{\rm ip} (\xi)$' if lam else r'$ w(\xi)$')
    l = [1] if plot_v != 'both' else [1, 2]
    for fig_num in l:
        plt.figure(fig_num)
        plt.xlim(0, 1)
        plt.vlines(cs, -5, 30, color='black', ls='dashed', lw=1)
        plt.vlines(vCJ, -5, 30, color='black', ls='dashed', lw=1)
        plt.xlabel(r'$\xi$')
        if legs:
            plt.legend(fontsize=fs_lg)


def _save_profiles(i, alpha, xis, vvs, wws, alphas_n, shocks, xi_shocks, wms,
                   str_alp, strs_vws, dec_vw, ress, vw):

    '''
    Save the profiles to a CSV file, called from compute_profiles_vws.
    '''

    df = pd.DataFrame({'alpha': alpha * xis ** 0, 'xi_w': vw * xis ** 0,
                       'xi': xis, 'v': vvs[i, :], 'w': wws[i, :],
                       'alpha_pl': alphas_n[i] * xis ** 0,
                       'shock': shocks[i] * xis ** 0,
                       'xi_sh': xi_shocks[i] * xis ** 0,
                       'wm': wms[i] * xis ** 0})
    if str_alp is None:
        str_alp_val = '%s' % alpha
        str_alp_val = '0' + str_alp_val
        str_alp_val = str_alp_val[2:]
    else:
        str_alp_val = str_alp[i]
    if strs_vws is None:
        str_vws_val = '%s' % np.round(vw, decimals=dec_vw)
        str_vws_val = str_vws_val[2:]
    else:
        str_vws_val = strs_vws[i]
    file_dir = f'{ress}/alpha_{str_alp_val}_vw_{str_vws_val}.csv'
    try:
        df.to_csv(file_dir)
        print('results of 1d profile saved in ', file_dir)
    except Exception:
        print('create directory results/1d_profiles to save the 1d profiles')


# COMPUTING EFFICIENCIES FROM 1D PROFILES
def kappas_from_prof(vw, alpha, xis, ws, vs):

    r"""
    Compute the kinetic energy density efficiency kappa and thermal
    factor omega from 1D profiles.

    .. math::
        \kappa = \frac{4}{v_w^3 \alpha} \int \xi^2 \frac{w}{1-v^2} v^2 d\xi
        \omega = \frac{3}{v_w^3 \alpha} \int \xi^2 (w-1) d\xi

    Parameters
    ----------
    vw : float
        Wall velocity.
    alpha : float
        Strength of the phase transition.
    xis : ndarray
        Array of xi positions.
    ws : ndarray
        Array of enthalpy profiles.
    vs : ndarray
        Array of velocity profiles.

    Returns
    -------
    kappa : float
        Ratio of kinetic to vacuum energy density.
    omega : float
        Ratio of energy density perturbations to vacuum energy density.
    """

    integrand_kappa = xis ** 2 * ws / (1 - vs ** 2) * vs ** 2
    integrand_omega = xis ** 2 * (ws - 1)
    kappa = 4. / vw ** 3. / alpha * safe_trapezoid(integrand_kappa, xis)
    omega = 3. / vw ** 3. / alpha * safe_trapezoid(integrand_omega, xis)

    return kappa, omega


def kappas_Esp(vw, alp, cs2=cs2_ref):

    """
    Compute the efficiency in converting vacuum to kinetic energy density.

    Uses semiempirical fits from Espinosa:2010hh, appendix A,
    following the bag equation of state.

    Numerical values can be computed from the 1d profiles using
    :func:`compute_profiles_vws` setting eff to True.

    Parameters
    ----------
    vw : float or array_like
        Wall velocity.
    alp : float or array_like
        Strength of the phase transition at nucleation temperature.
    cs2 : float, optional
        Square of the speed of sound (default is 1/3).

    Returns
    -------
    kappa : float
        Ratio of kinetic to vacuum energy density, computed
        in the bag equation of state.
    """

    cs = np.sqrt(cs2)
    mult_alp = isinstance(alp, (list, tuple, np.ndarray))
    mult_vw = isinstance(vw, (list, tuple, np.ndarray))

    if not mult_vw:
        vw = np.array([vw])
    if not mult_alp:
        alp = np.array([alp])
    v_cj = Chapman_Jouget(alp)

    ty = type_nucleation(vw, alp, cs2=cs2)
    _, v_cj = np.meshgrid(vw, v_cj, indexing='ij')
    vw, alp = np.meshgrid(vw, alp, indexing='ij')
    kappa = np.zeros_like(vw)

    # kappa at vw << cs
    kapA = vw ** (6 / 5) * 6.9 * alp / (1.36 - 0.037 * np.sqrt(alp) + alp)
    # kappa at vw = cs
    kapB = alp ** (2 / 5) / (0.017 + (0.997 + alp) ** (2 / 5))
    # kappa at vw = cJ (Chapman-Jouget)
    kapC = np.sqrt(alp) / (0.135 + np.sqrt(0.98 + alp))
    # kappa at vw -> 1
    kapD = alp / (0.73 + 0.083 * np.sqrt(alp) + alp)

    # deflagrations
    den = (cs ** (11 / 5) - vw ** (11 / 5)) * kapB + vw * cs ** (6 / 5) * kapA
    kappa_def = cs ** (11 / 5) * kapA * kapB / den
    kappa[ty == 'def'] = kappa_def[ty == 'def']

    # detonations
    kappa_det = (v_cj - 1) ** 3 * (v_cj / vw) ** (5 / 2) * kapC * kapD
    den = (
        ((v_cj - 1) ** 3 - (vw - 1) ** 3) * v_cj ** (5 / 2) * kapC
        + (vw - 1) ** 3 * kapD
    )
    kappa_det = kappa_det / den
    kappa[ty == 'det'] = kappa_det[ty == 'det']

    # hybrids
    ddk = -0.9 * np.log(np.sqrt(alp) / (1 + np.sqrt(alp)))
    kappa_hyb = kapB + (vw - cs) * ddk
    kappa_hyb += ((vw - cs) / (v_cj - cs)) ** 3 * (
        kapC - kapB - (v_cj - cs) * ddk
    )
    kappa[ty == 'hyb'] = kappa_hyb[ty == 'hyb']

    kappa = reshape_output(kappa, mult_a=mult_vw, mult_b=mult_alp)

    return kappa


# COMPUTING DIAGNOSTIC PROFILES
def w_to_lam(xis, ws, vw, alphan):

    r"""
    Compute the energy density perturbations from the enthalpy profile
    using the bag equation of state.

    .. math::
        \lambda(\xi) = \frac{3}{4} (w(\xi) - 1) - \frac{3}{4} \alpha_n
        \text{ for } \xi < v_w

    Parameters
    ----------
    xis : ndarray
        Array of xi positions.
    ws : ndarray
        Array of enthalpy profiles.
    vw : float
        Wall velocity.
    alphan : float
        Value of nucleation alpha.

    Returns
    -------
    lam : ndarray
        Array of energy density perturbations.

    Reference
    ---------
    Appendix A of RoperPol:2025a.
    """

    lam = 3 / 4 * (ws - 1)
    inds = np.where(xis < vw)[0]
    lam[inds] -= 3 / 4 * alphan
    return lam


# COMPUTING FUNCTIONS RELEVANT FOR VELOCITY SPECTRAL DENSITY
# f' and l functions
def fp_z(xi, vs, z, lz=False, ls=None, multi=True, quiet=False):

    """
    Compute the functions :math:`f'(z)` and :math:`l(z)`
    for the Fourier transform of the velocity field.

    These functions provide the initial conditions used to
    compute the kinetic spectrum in the sound-wave regime
    according to the Sound-Shell Model.

    Parameters
    ----------
    xi : ndarray
        Array of xi positions.
    vs : ndarray
        Array of velocity profiles.
    z : ndarray
        Array of z = k (t - t_n) where f' and l functions are to be computed.
    lz : bool, optional
        If True, compute l(z) (default False).
    ls : ndarray, optional
        Array of energy density perturbations (used if lz is True).
    multi : bool, optional
        If True, use an array of wall velocities.
    quiet : bool, optional
        If True, suppress debug output.

    Returns
    -------
    fpzs : ndarray
        Array of f'(z) values.
    lzs : ndarray, optional
        Array of l(z) values (if lz is True).
    """

    if ls is None:
        ls = []
    xi_ij, z_ij = np.meshgrid(xi[1:], z, indexing='ij')
    zxi_ij = z_ij * xi_ij
    j1_z = np.sin(zxi_ij) / zxi_ij ** 2 - np.cos(zxi_ij) / zxi_ij
    j1_z[np.where(zxi_ij == 0)] = 0

    if lz:
        if len(ls) == 0:
            print('if lz is chosen you need to provide a l(xi) profile')
            lz = False
        j0_z = np.sin(zxi_ij) / zxi_ij
        j0_z[np.where(zxi_ij == 0)] = 1

    if multi:
        fpzs, lzs = _fp_z_multi(
            xi, vs, z, lz, ls, xi_ij, j1_z, j0_z if lz else None, quiet
        )
    else:
        fpzs, lzs = _fp_z_single(
            xi, vs, z, lz, ls, xi_ij, j1_z, j0_z if lz else None
        )

    if lz:
        return fpzs, lzs
    return fpzs


def _fp_z_multi(xi, vs, z, lz, ls, xi_ij, j1_z, j0_z, quiet):
    Nvws = np.shape(vs)[0]
    fpzs = np.zeros((Nvws, len(z)))
    lzs = np.zeros((Nvws, len(z))) if lz else None
    for i in range(Nvws):
        v_ij, _ = np.meshgrid(vs[i, 1:], z, indexing='ij')
        integrand_fp = j1_z * xi_ij ** 2 * v_ij
        fpzs[i, :] = -4 * np.pi * safe_trapezoid(integrand_fp, xi[1:], axis=0)
        if lz:
            l_ij, _ = np.meshgrid(ls[i, 1:], z, indexing='ij')
            integrand_l = j0_z * xi_ij ** 2 * l_ij
            lzs[i, :] = 4 * np.pi * safe_trapezoid(integrand_l, xi[1:], axis=0)
        if not quiet:
            print('vw ', i + 1, '/', Nvws, ' computed')
    return fpzs, lzs


def _fp_z_single(xi, vs, z, lz, ls, xi_ij, j1_z, j0_z):
    v_ij, _ = np.meshgrid(vs[1:], z, indexing='ij')
    integrand_fp = j1_z * xi_ij ** 2 * v_ij
    fpzs = -4 * np.pi * safe_trapezoid(integrand_fp, xi[1:], axis=0)
    lzs = None
    if lz:
        l_ij, _ = np.meshgrid(ls[1:], z, indexing='ij')
        integrand_l = j0_z * xi_ij ** 2 * l_ij
        lzs = 4 * np.pi * safe_trapezoid(integrand_l, xi[1:], axis=0)
    return fpzs, lzs


def Rstar_beta(vws=1., cs2=cs2_ref, corr=True):

    r"""
    Compute the ratio of the mean bubble separation :math:`R_\ast`
    to the inverse nucleation rate parameter :math:`\beta`.

    .. math::
        R_\ast \beta = (8\pi)^{1/3} v_w

    If `corr` is True, applies a correction for the speed of sound:

    .. math::
        R_\ast \beta \to R_\ast \beta \times \max(1, c_s/v_w)

    Parameters
    ----------
    vws : float or array_like
        Bubble wall velocity :math:`v_w`.
    cs2 : float, optional
        Square of the speed of sound :math:`c_s^2` (default is 1/3).
    corr : bool, optional
        If True, apply correction for the speed of sound (default is True).

    Returns
    -------
    Rbeta : float or ndarray
        Ratio :math:`R_* \beta`.
    """

    Rbeta = (8 * np.pi) ** (1 / 3) * vws
    if corr:
        cs = np.sqrt(cs2)
        Rbeta = Rbeta * np.maximum(1.0, cs / vws)

    return Rbeta
