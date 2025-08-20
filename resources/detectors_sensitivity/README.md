This directory contains numerical values of the sensitivities of
different detectors to gravitational wave (GW) signals.

- The sensitivities (given in ratio of GW energy density to present time critical energy density,
Omega_GW) and power law sensitivities (PLS) for LISA and Taiji are computed using the routine
[interferometry.py](https://github.com/AlbertoRoper/GW_turbulence/blob/master/interferometry.py),
following the methodology of the appendix B in A. Roper Pol, S. Mandal, A. Brandenburg, T. Kahniashvili,
"Polarization of gravitational waves from helical MHD turbulent sources," JCAP 04 (2022), 019,
[arXiv:2107.05356](https://arxiv.org/abs/2107.05356).

  - 'LISA_Omega.csv' 
  - 'LISA_OmegaPLS.csv'
  - 'LISA_Xi.csv'
  - 'LISA_Xi_PLS.csv'
  - 'Taiji_Omega.csv'
  - 'Taiji_Omega_PLS.csv'
  - 'Taiji_Xi.csv'
  - 'Taiji_Xi_PLS.csv'
  - 'LISA_Taiji_Xi.csv'
  - 'LISA_Taiji_Xi_PLS.csv'

- The monopole and dipole response functions of LISA and Taiji are computed using the routine
[interferometry.py](https://github.com/AlbertoRoper/GW_turbulence/blob/master/interferometry.py),
following the methodology of the appendix B in A. Roper Pol, S. Mandal, A. Brandenburg, T. Kahniashvili,
"Polarization of gravitational waves from helical MHD turbulent sources," JCAP 04 (2022), 019,
[arXiv:2107.05356](https://arxiv.org/abs/2107.05356).

  - 'LISA_response_f.csv'
  - 'Taiji_response_f.csv'

- The monopole response functions (I and V Stokes parameters) obtained by cross-correlating channels of the LISA-Taiji
network are stored in the directory
['LISA_Taiji'](https://github.com/AlbertoRoper/GW_turbulence/tree/master/detector_sensitivity/LISA_Taiji),
from G. Orlando, M. Pieroni, A. Ricciardone, "Measuring Parity Violation
in the Stochastic Gravitational Wave Background with the LISA-Taiji network,"
JCAP 03, 069 (2021), [arXiv:2011.07059](https://arxiv.org/abs/2011.07059);
see Figure 2.

  - 'MAC_V.csv'
  - 'MAD_V.csv'
  - 'MEC_V.csv'
  - 'MED_I.csv'
  - 'MED_V.csv'

- The power law integrated sensitivities (PLS) of LISA, BBO, DECIGO, CE, ET, EPTA, SKA, IPTA, NANOGrav, PPTA,
HL, HLV, HLVK, HLVO2 in the directory
['power-law-integrated_sensitivities'](https://github.com/AlbertoRoper/GW_turbulence/tree/master/detector_sensitivity/power-law-integrated_sensitivities),
from K. Schmitz, "New Sensitivity Curves for Gravitational-Wave Signals from Cosmological Phase
Transitions," JHEP 01, 097 (2021), [arXiv:2002.04615](https://arxiv.org/abs/2002.04615);
available in [Zenodo](https://doi.org/10.5281/zenodo.3689582).

- The power law sensitivities (PLS) of atomic interferometers AION and AEDGE are from the references
L. Badurina et al., "AION: An Atom Interferometer Observatory and Network," JCAP 05 (2020), 011,
[arXiv:1911.11755](https://arxiv.org/abs/1911.11755), and \[AEDGE Collaboration\],
"AEDGE: Atomic Experiment for Dark Matter and Gravity Exploration in Space," EPJ Quant.Technol. 7,
6 (2020), [arXiv:1908.00802](https://arxiv.org/abs/1908.00802), respectively.

  - 'AEDGE.csv'
  - 'AION.csv'

- The power law sensitivities obtained considering binary resonances are taken from D. Blas,
A. Jenkins, "Bridging the μHz Gap in the Gravitational-Wave Landscape with Binary Resonances,"
Phys. Rev. Lett. 128, 10, 101103 (2022), [arXiv:2107.04601](https://arxiv.org/abs/2107.04601).

  - 'binaries_LLR_2038.csv' (from Lunar Laser Ranging forecast for 2038)
  - 'binaries MSPs_2038.csv' (from milisecond pulars forcast for 2038)
  - 'binaries SLR_2038.csv' (from Satellite Laser Ranging forecast for 2038)
  
- The power law sensitivities with SNR of 10 for DECIGO and SKA and using astrometry methods with
Gaia and Theia catalogues is taken from J. Garcia-Bellido, H. Murayama, G. White,
"Exploring the early Universe with Gaia and Theia," JCAP 12 (2021), 023,
[arXiv:2104.04778](https://arxiv.org/abs/2104.04778).

  - 'DECIGO_PLS_SNR10.csv'
  - 'SKA_PLS_SNR10.csv'
  - 'Gaia_PLS_SNR10.csv'
  - 'Theia_PLS_SNR10.csv'
  
- The sensitivity of the Voyage 2050 concept μAres is taken from A. Sesana et al.,
"Unveiling the Gravitational Universe at μ-Hz Frequencies", Exper. Astron. 51, 3, 1333 (2021),
[arXiv:1908.11391](https://arxiv.org/abs/1908.11391), and the PLS are calculated using the routine
[interferometry.py](https://github.com/AlbertoRoper/GW_turbulence/blob/master/interferometry.py),
following the methodology of the appendix B in A. Roper Pol, S. Mandal, A. Brandenburg, T. Kahniashvili,
"Polarization of gravitational waves from helical MHD turbulent sources," JCAP 04 (2022), 019,
[arXiv:2107.05356](https://arxiv.org/abs/2107.05356).

  - 'muAres_sens.csv'
  - 'muAres_Omega.csv'
  - 'muAres_OmegaPLS.csv'

- LISA PLS from C. Caprini et al., "Reconstructing the spectral shape of a stochastic gravitational
wave background with LISA," JCAP 11 (2019), 017, [arXiv:1906.09244](https://arxiv.org/abs/1906.09244).

  - 'OmegaPLS_Caprinietal19.csv'
  
- LISA PLS to polarization from J. Ellis, M. Fairbairn, M. Lewicki, V. Vaskonen, A. Wickens,
"Detecting circular polarisation in the stochastic gravitational-wave background from a first-order
cosmological phase transition," JCAP 10 (2020), 032, [arXiv:2005.05278](https://arxiv.org/abs/2005.05278).

  - 'XiPLS_Ellisetal19.csv'
