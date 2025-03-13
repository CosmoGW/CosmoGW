# CosmoGW v 1.1.0

CosmoGW is a python library that can be installed using pip from the
project stored in [pypi](https://pypi.org/project/cosmoGW)

pip install cosmoGW

cosmoGW contains functions for the study of cosmological gravitational wave (GW)
backgrounds from different sources in the early Universe, focusing on the
production of GWs from sound waves and MHD turbulence from cosmological phase
transitions.

It corresponds to an extension of the original [*GW_turbulence*](https://github.com/AlbertoRoper/GW_turbulence))
code previously used for calculations of GWs produced by MHD turbulence.

* Author: Alberto Roper Pol (alberto.roperpol@unige.ch)
* Created: 01/12/2021 (GW_turbulence)
* Updated: 13/03/2025 (release cosmoGW 1.1.0)  

The project includes state-of-the-art GW models developed by the community, postprocessing calculations,
numerical computations, plotting routines, and detector sensitivities.

If you use any of the cosmoGW results, please cite this [repository](https://github.com/MHDcosmoGW/cosmoGW) and the
relevant reference/s listed in the routines, corresponding to the original published work.

I would also love to hear about your interest for this project and your work, so consider reaching out to me for
any issues related to the code, questions or discussion: alberto.roperpol@unige.ch.

Some of the routines use results from large-scale numerical simulations conducted with the open-source
[Pencil Code](https://github.com/pencil-code); see

[Pencil Code Collaboration], *The Pencil Code, a modular MPI code for partial differential equations and particles:
multipurpose and multiuser-maintained,* J. Open Source Softw. **6**, 2807 (2021), 
[arXiv:2009.08231](https://arxiv.org/abs/2009.08231), [DOI:10.21105/joss.02807](https://joss.theoj.org/papers/10.21105/joss.02807).

In particular, if you use any of the results that involve Pencil Code simulations, please cite the
[Pencil Code paper](https://joss.theoj.org/papers/10.21105/joss.02807) and the
[code](https://github.com/pencil-code), as well as the original work.

## Routines

The main routines of cosmoGW are stored under src/cosmoGW:

* [**cosmoGW.py**](src/cosmoGW/cosmoGW.py): functions relevant for cosmological stochastic gravitational wave backgrounds (SGWB).
* [**cosmology.py**](src/cosmoGW/cosmology.py): functions relevant for cosmological calculations, including a Friedmann equations
  solver (see tutorial on Friedmann equations in [cosmology.ipnyb](cosmology/cosmology.ipynb)) that can generate the solution
  files being read in some Pencil Code simulations (see tutorial [cosmology_PC.ipnyb](cosmology/cosmology_PC.ipynb)).
* [**cosmoMF.py**](src/cosmoGW/cosmoMF.py): functions relevant for cosmological magnetic fields like bounds from different experiments,
  observations or projected sensitivities, and expectations from theory, among others.
* [**GW_analytical.py**](src/cosmoGW/GW_analytical.py)
* [**GW_fopt.py**](src/cosmoGW/GW_fopt.py)
* [**hydro_bubbles.py**](src/cosmoGW/hydro_bubbles.py): functions to compute fluid perturbations induced by the expansion of
  bubbles in first-order phase transitions
* [**interferometry.py**](src/cosmoGW/interferometry.py): functions to compute the response and sensitivity functions of interferometer
  space-based GW detectors (e.g., LISA and Taiji) to the detection of SGWBs (see tutorial on LISA interferometry in
  [interferometry.ipynb](interferometry/interferometry.ipynb)) energy density and polarization, including the space-based network
  LISA-Taiji to detect polarization.
* [**modified_grav.py**](src/cosmoGW/modified_grav.py): functions relevant for GW production in the context of general theories of
  modified gravity.
* [**pta.py**](src/cosmoGW/pta.py): functions used in the analysis of observations by pulsar timing array (PTA) collaborations:
  NANOGrav, PPTA, EPTA, and IPTA.
* [**reading.py**](src/cosmoGW/reading.py): functions to read the output files of a specific set of runs (project)
  of the Pencil Code.
* [**run.py**](src/cosmoGW/run.py): contains the class **run**, used to store all the variables computed in the Pencil Code and in
  cosmoGW from the Pencil Code solitions. It includes functions to initialize and postprocess the results of a set of runs.
* [**spectra.py**](src/cosmoGW/spectra.py): contains description for specific spectral templates, postprocessing routines for
  numerical spectra, and other mathematical routines.

## Resources

Some data files are available in cosmoGW that are useful in some of the projects.
They are stored in src/cosmoGW/resources

* [**cosmology**](src/cosmoGW/resources/cosmology): includes files relevant for the cosmological evolution of the Universe and
  contains a tutorial on solving Friedmann equations.
* [**interferometry**](src/cosmoGW/resources/interferometry): includes files relevant for space-based GW interferometry calculations
  and contains a tutorial on computing the response functions, sensitivities and power law sensitivities to SGWB energy density
  and polarization.
* [**detector_sensitivity**](detector_sensitivity): includes the sensitivity of various detectors (ground-based, space-based,
  and pulsar timing arrays, among others), see the [README](detector_sensitivity/README.md) file for info and references.

## Projects

Particular projects with Jupyter notebooks are available under projects

* [**GWs_from_PTs**](projects/GWs_from_PTs): contains tutorials related to the production of GWs (self-similar profiles
  calculation for now, but more coming soon!)

## Publications

The work of the following publications can be reproduced using CosmoGW:

* *Numerical Simulations of Gravitational Waves from Early-Universe Turbulence,* A. Roper Pol, S. Mandal, A. Brandenburg,
  T. Kahaniashvili, A. Kosowsky, *Phys. Rev. D* **102** (2020) 8, 083512, [arXiv:1903.08585](https://arxiv.org/abs/1903.08585).
  Datasets produced using Pencil Code are publicly available at [doi:10.5281/zenodo.3692072](https://zenodo.org/records/3692072).
  Direct access to Pencil Code files is also available in
  [brandenb/projects/GW/](http://norlx65.nordita.org/~brandenb/projects/GW/).
  
* *Polarization of gravitational waves from helical MHD turbulent sources,* A. Roper Pol, S. Mandal, A. Brandenburg,
  T. Kahniashvili, *J. Cosmol. Astropart. Phys.* **04** (2022) 04, 019, [arXiv:2107.05356](https://arxiv.org/abs/2107.05356).
  Datasets produced using Pencil Code are publicly available at
  [doi:10.5281/zenodo.5525504](https://zenodo.org/records/5525504).
  
* *Gravitational wave signal from primordial magnetic fields in the Pulsar Timing Array frequency band,* A. Roper Pol,
  C. Caprini, A. Neronov, D. Semikoz, *Phys. Rev. D* **105** (2022) 12, 123502, [arXiv:2201.05630](https://arxiv.org/abs/2201.05630).
  Datasets produced using Pencil Code are publicly available at
  [doi:10.5281/zenodo.5782752](https://zenodo.org/records/5782752).

* *Modified propagation of gravitational waves from the early radiation era,* Y. He, A. Roper Pol, A. Brandenburg, *J. Cosmol.
  Astropart. Phys.* **06** (2023) 025, [arXiv:2212.06082](https://arxiv.org/abs/2212.06082).
  Datasets produced using Pencil Code are publicly available at [doi:10.5281/zenodo.5525504](https://zenodo.org/records/5525504).
  Direct access to Pencil Code files is also available in
  [brandenb/projects/Horndeski/](http://norlx65.nordita.org/~brandenb/projects/Horndeski/).

* *LISA and γ-ray telescopes as multi-messenger probes of a first-order cosmological phase transition,* A. Roper Pol,
  A. Neronov, C. Caprini, T. Boyer, D. Semikoz, *submitted to Astron. Astrophys.*,
  [arXiv:2307.10744](https://arxiv.org/abs/2307.10744).

* *Characterization of the gravitational wave spectrum from sound waves within the sound shell model,* A. Roper Pol,
  S. Procacci, C. Caprini, *Phys. Rev. D* **109** (2024) 6, 063531, [arXiv:2308.12943](https://arxiv.org/abs/2308.12943).

* *Gravitational waves from decaying sources in strong phase transitions,* A. Roper Pol, I. Stomberg, C. Caprini, R. Jinno,
  T. Konstandin, H. Rubira, *submitted to J. Cosmol. Astropart. Phys.*, [arXiv:2409.03651](https://arxiv.org/abs/2409.03651).
