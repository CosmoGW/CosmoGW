utils
-----------------------

utils.py contains reference values and utility functions for array
reshaping and other common operations used throughout the cosmoGW package.

https://github.com/cosmoGW/cosmoGW/
https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/utils.py

.. note::
   For full documentation, visit `Read the Docs
   <https://cosmogw-manual.readthedocs.io/en/latest/utils.html>`_.

These functions are designed to be imported and reused in multiple modules
(e.g., GW_models.py, GW_templates.py, hydro_bubbles.py, etc.) to keep code DRY
and maintainable.

Example usage::

    from cosmoGW.utils import reshape_output

Author
------
- **Alberto Roper Pol**
  (`alberto.roperpol@unige.ch <mailto:alberto.roperpol@unige.ch>`_)

Dates
-----
- Created: **21/08/2025**
  (release **cosmoGW 1.0**: https://pypi.org/project/cosmoGW)

References
----------
Used in cosmoGW scientific routines for output reshaping
and general utilities.

.. _utils_reference_values:

Reference Values in ``cosmoGW.utils``
=====================================

The following reference values are defined in ``cosmoGW.utils``
and are used throughout the cosmoGW codebase for cosmological and
gravitational wave calculations.
These constants provide standard parameters for
phase transitions, cosmology, turbulence, and source modeling.

.. list-table:: Reference Values
   :header-rows: 1

   * - Name
     - Value
     - Description
   * - Tref
     - 100 GeV
     - Electroweak phase transition temperature
   * - gref
     - 100
     - Effective degrees of freedom (EWPT)
   * - Neff_ref
     - 3
     - Reference effective neutrino number
   * - T0K
     - 2.72548 K
     - CMB temperature
   * - H0_ref
     - 100 km / s / Mpc (converted to Hz)
     - Hubble constant
   * - OmL0_ref
     - 0.6841
     - Cosmological constant density parameter
   * - OmM0_ref
     - 0.316
     - Matter density parameter
   * - h0_ref
     - 0.6732
     - Reduced Hubble constant
   * - a_ref
     - 5
     - Batchelor spectrum exponent
   * - b_ref
     - 2/3
     - Kolmogorov spectrum exponent
   * - alp_ref
     - 2
     - Smoothness of broken power-law transition
   * - alp_turb
     - 6/17
     - von Karman smoothness parameter
   * - bPi_vort
     - 3
     - Spectral slope in anisotropic stresses
   * - alpPi
     - 2.15
     - Smoothness for anisotropic fit
   * - fPi
     - 2.2
     - Break frequency of anisotropic stresses
   * - tdecay_ref
     - 'eddy'
     - Decaying time in constant-in-time model
   * - OmGW_sw_ref
     - 1e-2
     - Normalized amplitude (sound waves)
   * - a_sw_ref
     - 3
     - Low frequency slope (:math:`f^3``)
   * - b_sw_ref
     - 1
     - Intermediate frequency slope (f)
   * - c_sw_ref
     - 3
     - High frequency slope (:math:`f^{-3}`)
   * - alp1_sw_ref
     - 1.5
     - Used in RoperPol:2023bqa
   * - alp2_sw_ref
     - 0.5
     - Used in RoperPol:2023bqa
   * - alp1_ssm
     - 4
     - Used in Hindmarsh:2019phv
   * - alp2_ssm
     - 2.0
     - Used in Hindmarsh:2019phv
   * - alp1_HL
     - 3.6
     - Found in Caprini:2024gyk
   * - alp2_HL
     - 2.4
     - Found in Caprini:2024gyk
   * - alp1_LISA
     - 2.0
     - Used in Caprini:2024hue
   * - alp2_LISA
     - 4.0
     - Used in Caprini:2024hue
   * - Oms_ref
     - 0.1
     - Source amplitude (fraction to radiation energy)
   * - lf_ref
     - 0.01
     - Source length scale (normalized by Hubble radius)
   * - beta_ref
     - 100
     - Nucleation rate beta/H_ast
   * - N_turb
     - 2
     - Source duration/eddy turnover time ratio
   * - Nk_ref
     - 1000
     - Wave number discretization
   * - Nkconv_ref
     - 1000
     - Wave number discretization for convolution
   * - Np_ref
     - 3000
     - Wave number discretization for convolution
   * - NTT_ref
     - 5000
     - Lifetimes discretization
   * - dt0_ref
     - 11
     - Numerical parameter for fit (Caprini:2024gyk)
   * - tini_ref
     - 1.0
     - Initial time of GW production (normalized)
   * - tfin_ref
     - 1e4
     - Final time of GW production in cit model
   * - cs2_ref
     - 1/3.
     - Speed of sound squared
   * - Nxi_ref
     - 10000
     - Discretization in xi
   * - Nxi2_ref
     - 10
     - Discretization in xi out of profiles
   * - Nvws_ref
     - 20
     - Discretization in vwall
   * - tol_ref
     - 1e-5
     - Tolerance on shooting algorithm
   * - it_ref
     - 30
     - Number of iterations
   * - vw_def
     - 0.5
     - Default wall velocity
   * - alpha_def
     - 0.263
     - Default alpha parameter
   * - vw_hyb
     - 0.7
     - Hybrid wall velocity
   * - alpha_hyb
     - 0.052
     - Hybrid alpha parameter

.. note::
   These values are available in ``cosmoGW.utils`` and can be
   imported for use in your own calculations and models.


.. automodule:: cosmoGW.utils
   :members:
   :show-inheritance:
   :undoc-members: