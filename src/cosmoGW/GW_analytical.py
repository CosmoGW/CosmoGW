"""
GW_analytical.py is a Python routine that contains analytical
calculations and useful mathematical functions.

Adapted from the original GW_analytical in cosmoGW
(https://github.com/AlbertoRoper/cosmoGW),
created in Dec. 2021

Currently part of the cosmoGW code:

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/GW_analytical.py

.. note::
   For full documentation, visit `Read the Docs
   <https://cosmogw-manual.readthedocs.io/en/latest/GW_analytical.html>`_

To use it, first install `cosmoGW <https://pypi.org/project/cosmoGW>`_::

    pip install cosmoGW

Author
------
- **Alberto Roper Pol**
  (`alberto.roperpol@unige.ch <mailto:alberto.roperpol@unige.ch>`_)

Dates
-----
- Created: **01/12/2021**
- Updated: **31/08/2024**
- Updated: **21/08/2025**
  (release **cosmoGW 1.0**: https://pypi.org/project/cosmoGW)

Contributors
------------
- **Antonino Midiri**, **Madeline Salomé**

References
----------
- [**RoperPol:2022iel**]: A. Roper Pol, C. Caprini, A. Neronov, D. Semikoz,
  "*The gravitational wave signal from primordial magnetic fields in the
  Pulsar Timing Array frequency band*," Phys. Rev. D **105**, 123502 (2022),
  `arXiv:2201.05630 <https://arxiv.org/abs/2201.05630>`_

- [**RoperPol:2023bqa**]: A. Roper Pol, A. Neronov, C. Caprini, T. Boyer,
  D. Semikoz, "*LISA and γ-ray telescopes as multi-messenger probes of a
  first-order cosmological phase transition,*"
  `arXiv:2307.10744 <https://arxiv.org/abs/2307.10744>`_ (2023)

- [**RoperPol:2023dzg**]: A. Roper Pol, S. Procacci, C. Caprini,
  *"Characterization of the gravitational wave spectrum from sound waves within
  the sound shell model,*" Phys. Rev. D **109**, 063531 (2024),
  `arXiv:2308.12943 <https://arxiv.org/abs/2308.12943>`_

- [**Caprini:2024gyk**]: A. Roper Pol, I. Stomberg, C. Caprini, R. Jinno,
  T. Konstandin, H. Rubira, "*Gravitational waves from first-order
  phase transitions: from weak to strong,*" JHEP **07** (2025) 217,
  `arXiv:2409.03651 <https://arxiv.org/abs/2409.03651>`_.

- [**Caprini:2024hue**]: E. Madge, C. Caprini, R. Jinno, M. Lewicki,
  M. Merchand, G. Nardini, M. Pieroni, A. Roper Pol, V. Vaskonen,
  *"Gravitational waves from first-order phase transitions in LISA:
  reconstruction pipeline and physics interpretation,*"
  JCAP **10** (2024), 020,
  `arXiv:2403.03723 <https://arxiv.org/abs/2403.03723>`_.

- [**RoperPol:2025b**]: A. Roper Pol, A. Midiri, M. Salomé, C. Caprini,
  "Modeling the gravitational wave spectrum from slowly decaying sources in the
  early Universe: constant-in-time and coherent-decay models," in preparation
"""

import numpy as np
from cosmoGW.utils import a_ref, b_ref, alp_ref, complete_beta


def _check_slopes(a, b, dlogk=True):
    r"""
        Check the slopes used in power spectra to ensure
        that the integral over k or log k is converging.

        Parameters
        ----------
        a : float
            Slope of the spectrum at low wave numbers, :math:`k^a`
        b : float
            Slope of the spectrum at high wave numbers, :math:`k^{-b}`
        dlogk : bool, optional
            If True, consider the spectral function per unit of :math:`\log(k)`.
            If False, per unit of :math:`k`.

        Returns
        -------
        conv : bool
            True if the slopes lead to a convergent integral, False otherwise.
        """
    conv = True
    if dlogk:
        if b < 0 or a < 0:
            print('a and b have to be larger than 0')
            conv = False
    else:
        if b < 1 or a < -1:
            print('b has to be larger than 1 and a larger than -1')
            conv = False

    return conv


def smoothed_bPL(k, A=1., a=a_ref, b=b_ref, kpeak=1., alp=alp_ref, norm=True,
                 Omega=False, alpha2=False, piecewise=False, dlogk=True):
    r"""
    Return the value of the smoothed broken power law (bPL) model.

    The spectrum is of the form:

    .. math::

        \zeta(K) = A \cdot (b + \left| a \right|)^{1/\alpha}
        \frac{K^a}{\left[ b + c \cdot K^{\alpha(a + b)} \right]^{1/\alpha}}

    where :math:`K = k/k_\ast', :math:`c = 1` if :math:`a = 0` or
    :math:`c = \text{abs}(a)` otherwise.

    If norm is False:

    .. math::

         \zeta(K) = A \cdot \frac{K^a}{
             \left( 1 + K^{\alpha(a + b)} \right)^{1/\alpha}}

    RoperPol:2022iel and RoperPol:2025b consider a spectral function
    :math:`\zeta(K)`
    defined such that the average squared field corresponds to

    .. math::

        \langle v^2 \rangle \propto \Omega_v = A k_\ast \int \zeta(K) dK,
        \qquad
        {\rm in  \ RoperPol:2022iel}

        \langle v^2 \rangle \propto \Omega_v = A \int \zeta(K) d \ln K,   \qquad
        {\rm in  \ RoperPol:2025b}

    The first convention can be chosen in the following functions if
    dlogk is set to False, while the second one is assumed when dlogk
    is True.

    An alternative parameterization is used when alpha2 is True:

    .. math::

        \zeta(K) = A \cdot (b + \left| a \right|)^{(a + b)/\alpha}
        \frac{K^a}{\left[ b + c \cdot K^{\alpha} \right]^{(a + b)/\alpha}}

    Parameters
    ----------
    k : array_like
        Array of wave numbers.
    A : float, optional
        Amplitude of the spectrum (default 1).
    a : float, optional
        Slope at low wave numbers (default 5).
    b : float, optional
        Slope at high wave numbers (default 2/3).
    kpeak : float, optional
        Spectral peak position :math:`k_\ast` (default 1).
    alp : float, optional
        Smoothness of the transition :math:`\alpha` (default 2).
    norm : bool, optional
        Normalize spectrum to peak at :math:`k_\ast` with
        amplitude A (default True).
    Omega : bool, optional
        Use integrated energy density :math:`\Omega_v` as input A.
    alpha2 : bool, optional
        Use alternative convention for spectrum.
    piecewise : bool, optional
        Return piecewise broken power law.
        Corresponds to :math:`\alpha \to \infty`.
    dlogk : bool, optional
        Use spectrum defined per unit :math:`\log(k)` or per unit k.

    Returns
    -------
    spec : ndarray
        Spectrum array.
    """
    conv = _check_slopes(a, b, dlogk=dlogk)
    if not conv:
        return 0*k**0

    c = abs(a)
    if a == 0:
        c = 1
    if alpha2:
        alp = alp/(a + b)

    K = k/kpeak
    spec = A*K**a
    if piecewise:
        spec[np.where(K > 1)] = A*K[np.where(K > 1)]**(-b)
    else:
        alp2 = alp*(a + b)
        if norm:
            m = (b + abs(a))**(1/alp)
            spec = m*spec/(b + c*K**alp2)**(1/alp)
        else:
            spec = spec/(1 + K**alp2)**(1/alp)

    if Omega:
        spec = spec/kpeak/calA(a=a, b=b, alp=alp, norm=norm,
                               alpha2=alpha2, piecewise=piecewise,
                               dlogk=dlogk)

    return spec


def calIab_n_alpha(a=a_ref, b=b_ref, alp=alp_ref, n=0, norm=True):

    r"""
    Compute the normalization factor used in the calculation
    of :math:`I_{ab; n}` (see Iabn function),

    .. math ::

        {\cal I}_{ab; n} (\alpha) = \frac{1}{\alpha (a + b)}
        \left(\frac{a + b}{b} \right)^{1/\alpha}
        \left(\frac{a}{b}\right)^{-\frac{a + 1 + n}{\alpha (a + b)}}

    Parameters
    ----------
    a, b : float
        Slopes of the smoothed_bPL function (default 5 and 2/3).
    alp : float
        Smoothness parameter :math:`\alpha` (default 2).
    n : int
        n-moment of the integral (default 0).
    norm : bool
        Normalize the spectrum (default True).

    Returns
    -------
    calI : float
        Normalization parameter for the integral,
        :math:`\cal I_{ab; n} (\alpha)`.

    Reference
    ---------
    Appendix A of RoperPol:2025b
    """
    alp2 = 1/alp/(a + b)
    a_beta = (a + n + 1)*alp2

    c = abs(a)
    if a == 0:
        c = 1

    calI = alp2
    if norm:
        calI = calI*((b + abs(a))/b)**(1/alp)/(c/b)**a_beta

    return calI


def Iabn(a=a_ref, b=b_ref, alp=alp_ref, n=0, norm=True, alpha2=False,
         piecewise=False):

    r"""
    Compute the n-th moment of the smoothed bPL spectra.

    .. math::

        I_{ab; n} (\alpha) = \int K^n \zeta(K) dK =
        {\cal I}_{ab; n} (\alpha) \frac{\Gamma
        \bigl[\frac{a + n + 1}{\alpha (a + b)}\bigr]
        \Gamma\bigl[\frac{b - n - 1}{\alpha (a + b)}\bigr]}
        {\Gamma\bigl[\frac{1}{\alpha}\bigr]}

    Parameters
    ----------
    a, b : float
        Slopes of the spectrum (default 5 and 2/3).
    alp : float
        Smoothness of the transition :math:`\alpha` (default 2).
    n : int
        Moment of the integration (default 0).
    norm : bool
        Normalize the spectrum (default True).
    alpha2 : bool
        Use alternative convention (default False).
    piecewise : bool
        Use piecewise broken power law (default False).

    Returns
    -------
    moment_value : float
        Value of the n-th moment.
    """
    if a + n + 1 <= 0:
        print('a + n has to be larger than -1 for the integral to converge')
        return 0

    if b - n - 1 <= 0:
        print('b + n has to be larger than 1 for the integral to converge')
        return 0

    if piecewise:
        return (a + b)/(a + n + 1)/(b - n - 1)

    if alpha2:
        alp = alp/(a + b)
    alp2 = 1/alp/(a + b)
    a_beta = (a + n + 1)*alp2
    b_beta = (b - n - 1)*alp2
    moment_value = calIab_n_alpha(a=a, b=b, alp=alp, n=n, norm=norm)
    moment_value *= complete_beta(a_beta, b_beta)

    return moment_value


def calA(a=a_ref, b=b_ref, alp=alp_ref, norm=True, alpha2=False,
         piecewise=False, dlogk=True):

    r"""
    Compute the parameter :math:`{\cal A} = I_{ab;0}` that relates the
    peak and the integrated values of the smoothed bPL spectrum.

    References
    ----------
    RoperPol:2022iel, equation 8 (for dlogk False).

    RoperPol:2025b, appendix A (for dlogk True).

    Parameters
    ----------
    Same as Iabn function with n = -1 or 0 (if dlogk is True or False).

    dlogk : bool
        Use logarithmic spacing (default True).

    Returns
    -------
    calA : float
        Value of the parameter :math:`{\cal A}`.
    """
    nn = 0
    if dlogk:
        nn = -1
    calA = Iabn(a=a, b=b, alp=alp, n=nn, norm=norm, alpha2=alpha2,
                piecewise=piecewise)

    return calA


def calB(a=a_ref, b=b_ref, alp=alp_ref, n=1, norm=True, alpha2=False,
         piecewise=False, dlogk=True):

    r"""
    Compute the parameter

    .. math::
        {\cal B} = \left(\frac{I_{ab; -1 - n}}{I_{ab; -1}}\right)^{1/n}

    that relates the peak and the integral scale (when :math:`n = 1`),
    :math:`{\cal B} = \xi\, k_{\text{peak}}`.
    When dlogk is False, the definition changes slightly, :math:`-1 \to 0`.

    Parameters
    ----------
    Same as Iabn function.

    n : int
        Moment of the integration (default 1).
    dlogk : bool
        Use logarithmic spacing (default True).

    Returns
    -------
    calB : float
        Value of the parameter :math:`{\cal B}`.

    Reference
    ---------
    RoperPol:2025b, appendix A (for dlogk True)
    """
    nn = 0
    if dlogk:
        nn = -1

    Im1 = Iabn(a=a, b=b, alp=alp, n=-n + nn, norm=norm, alpha2=alpha2,
               piecewise=piecewise)
    I0 = Iabn(a=a, b=b, alp=alp, n=nn, norm=norm, alpha2=alpha2,
              piecewise=piecewise)
    calB = (Im1/I0)**n

    return calB


def zetam(a=a_ref, b=b_ref, alp=alp_ref, m=-1, norm=True, alpha2=False,
          piecewise=False):

    r"""
    Compute the amplitude at the peak of the function

    .. math::
        \zeta_m (K) = K^m \zeta(K),

    where :math:`\zeta(K)` is the original smoothed double broken power
    law (dbP).

    Parameters
    ----------
    a, b : float
        Slopes of the spectrum (default 5 and 2/3).
    alp : float
        Smoothness of the transition :math:`\alpha` (default 2).
    m : int
        Power of :math:`K^m` (default -1).
    norm : bool
        Normalize the spectrum (default True).
    alpha2 : bool
        Use alternative convention (default False).
    piecewise : bool
        Use piecewise broken power law (default False).

    Returns
    -------
    zetam : float
        Amplitude of new function :math:`K^m \zeta(K)`.
    """
    if piecewise:
        return 1.

    if alpha2:
        alp2 = alp
    else:
        alp2 = alp*(a + b)
    zetam = ((a + m)/a)**((a + m)/alp2)*((b - m)/b)**((b - m)/alp2)

    return zetam


def Kstarm(a=a_ref, b=b_ref, alp=alp_ref, m=-1, norm=True, alpha2=False,
           piecewise=False):

    r"""
    Compute the location of the peak of the function

    .. math::
        \zeta_m (K) = K^m \zeta(K),

    where :math:`\zeta(K)` is the original smoothed double broken power
    law (dbP).

    Parameters
    ----------
    Same as zetam.

    Returns
    -------
    Kstarm : float
        Position of the peak of new function :math:`K^m \zeta(K)`.
    """
    if piecewise:
        return 1.

    if alpha2:
        alp2 = alp
    else:
        alp2 = alp*(a + b)
    Kstarm = (b/a*(a + m)/(b - m))**(1./alp2)

    return Kstarm


def calC(a=a_ref, b=b_ref, alp=alp_ref, tp='vort', norm=True, alpha2=False,
         piecewise=False, dlogk=True):

    r"""
    Compute the parameter :math:`\cal C` for the TT-projected stress spectrum.

    It gives the value of the stresses spectrum at
    :math:`K \to 0` limit as a prefactor
    pref that depends on the type of source and an integral over

    .. math::

        \int \frac{\zeta^2}{P^4} d\log k,
        \qquad {\rm when \ dlogk \ is \ True
        \ (see \ appendix \ B \ of \ RoperPol:2025b)}

        \int \frac{\zeta^2}{P^2} d K, \qquad {\rm when \ dlogk \ is \
        False \ (RoperPol:2022iel \ for \ vortical \ and }

        \qquad \qquad {\rm
        eq. \ 46 \ of \ RoperPol:2023dzg \
        for \ compressional \ fields)}

    The spectrum of the stresses is then normalized such that
    (see appendix B of RoperPol:2025b):

    .. math::

        E_\Pi (K) = K^3 E_\ast^2 \, {\cal C} \, \zeta_\Pi (K).

    Parameters
    ----------
    a, b : float
        Slopes of the spectrum (default 5 and 2/3).
    alp : float
        Smoothness of the transition :math:`\alpha` (default 2).
    tp : str
        Type of sourcing field: 'vort', 'comp', 'hel', or 'mix'.
    norm : bool
        Normalize the spectrum (default True).
    alpha2 : bool
        Use alternative convention (default False).
    piecewise : bool
        Use piecewise broken power law (default False).
    dlogk : bool
        Use per unit log(k) or per unit k (default True).

    Returns
    -------
    calC : float
        Value of the parameter :math:`\cal C`.
    """
    pref = 1.

    if tp == 'vort':
        pref = 28/15
    elif tp == 'comp':
        pref = 32/15
    elif tp == 'mix':
        pref = 16/5
    elif tp == 'hel':
        pref = 1/3

    if tp not in ['vort', 'comp', 'mix', 'hel', 'none']:
        print('tp has to be vortical (vort), compressional (comp),',
              'mixed (mix) or helical (hel)')
        print('returning pref = 1')

    nn = 0
    if dlogk:
        nn = 2

    calC = pref*Iabn(a=a*2, b=b*2, alp=alp/2, n=-2-nn, norm=norm,
                     alpha2=alpha2, piecewise=piecewise)

    return calC


def smoothed_double_bPL(k, kpeak1, kpeak2, A=1., a=a_ref, b=1,
                        c=b_ref, alp1=alp_ref, alp2=alp_ref, kref=1.,
                        alpha2=False):

    r"""
    Return the value of the smoothed double broken power law (double bPL) model.

    The spectrum is of the form:

    .. math::

        \zeta(K) = A K^a \times
        \bigl(1 + (K/K1)^{(b - a) \alpha_1}\bigr)^{1/\alpha_1}
        \times \bigl(1 + (K/K2)^{(c + b) \alpha_2}\bigr)^{-1/\alpha_2},

    where :math:`K = k/k_\ast`, K1 and K2 are the two position peaks,
    a is the low-k slope, b is the intermediate slope,
    and -c is the high-k slope.
    :math:`\alpha_1` and :math:`\alpha_2` are the smoothness parameters
    for each spectral transition.

    The same broken power law with a slightly different form is used in
    Caprini:2024gyk, Caprini:2024hue and can be used setting
    alpha2 = True:

    .. math::

        \zeta(K) = A K^a \times
        \bigl(1 + (K/K1)^{\alpha_1}\bigr)^{(b - a)/\alpha_1}
        \times \bigl(1 + (K/K2)^{\alpha_2}\bigr)^{-(c + b)/\alpha_2},

    Parameters
    ----------
    k : array_like
        Array of wave numbers.
    kpeak1, kpeak2 : float
        Peak positions.
    A : float, optional
        Amplitude of the spectrum.
    a : float, optional
        Slope at low wave numbers (default 5).
    b : float, optional
        Slope at intermediate wave numbers (default 2/3).
    c : float, optional
        Slope at high wave numbers (default 2/3).
    alp1, alp2 : float, optional
        Smoothness of the transitions.
    kref : float, optional
        Reference wave number used to normalize the spectrum.
    alpha2 : bool, optional
        Use different normalization.

    Returns
    -------
    spec : ndarray
        Spectrum array.

    References
    ----------
    RoperPol:2023dzg, equation 50
    RoperPol:2023bqa, equation 7
    """
    K = k/kref
    K1 = kpeak1/kref
    K2 = kpeak2/kref

    if not alpha2:
        spec1 = (1 + (K/K1)**((a - b)*alp1))**(1/alp1)
        spec2 = (1 + (K/K2)**((c + b)*alp2))**(1/alp2)
    else:
        spec1 = (1 + (K/K1)**alp1)**((a - b)/alp1)
        spec2 = (1 + (K/K2)**alp2)**((c + b)/alp2)
    spec = A*K**a/spec1/spec2

    return spec
