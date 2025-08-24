[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI version](https://img.shields.io/pypi/v/cosmogw.svg)](https://pypi.org/project/cosmogw/)

# CosmoGW v1.0

**CosmoGW** is a Python package for the analysis and modeling of cosmological gravitational wave (GW) backgrounds from different sources in the early Universe, including analytical models, templates, and data analysis routines.

**v1.0 (August 2025)** focuses on the production of GWs from *sound waves* and *MHD turbulence* from
cosmological phase transitions.

Other sources of GWs will be included in future versions, contact me if you are interested
in contributing and becoming a developer.

**CosmoGW** is an extension of [*GW_turbulence*](https://github.com/AlbertoRoper/GW_turbulence), created in December 2021, used for calculations of GWs produced by MHD turbulence.

---

<img src="https://img.icons8.com/ios-filled/50/000000/conference-call.png" width="24" /> **Contact**

**Author:** Alberto Roper Pol  
**Email:** alberto.roperpol@unige.ch  
**GitHub:** [albertoroperpol](https://github.com/AlbertoRoper)  

If you use any of the cosmoGW results, please cite this [repository](https://github.com/cosmoGW/cosmoGW),
the [manual] (coming soon) and the relevant references listed in the routines.

Feel free to reach out for questions, issues, or collaboration!
I would also love to hear about your interest for this project and your work, so feel free to reach out for questions, issues, or collaboration!

---

<img src="https://img.icons8.com/ios-filled/50/000000/download.png" width="24" /> **Installation**

CosmoGW is a Python package that can be installed using pip from [PyPI](https://pypi.org/project/cosmoGW):

```sh
pip install cosmoGW
```

Or directly from GitHub (last version but not necessarily stable):

```sh
pip install git+https://github.com/cosmoGW/cosmoGW.git
```

**Requirements:**  
Python >= 3.8, numpy, scipy, matplotlib, astropy, pandas

To use the plotting routines available in the tutorials and in
[plot_sets](https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/plot_sets.py), make sure to install LaTeX for Python:

*LaTeX on Ubuntu/Debian:*
```bash
sudo apt-get update
sudo apt-get install -y cm-super dvipng texlive-latex-extra texlive-latex-recommended
```

*LaTeX on macOS (using Homebrew):*
```bash
brew install --cask mactex
```

---

<img src="https://img.icons8.com/ios-filled/50/000000/code-file.png" width="24" /> **File Structure**

The main routines of cosmoGW are stored under `src/cosmoGW`:

- [`analysis.py`](https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/analysis.py): Analysis of GWs detectability with LISA.
- [`cosmology.py`](https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/cosmology.py): Cosmological calculations.
- [`GW_analytical.py`](https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/GW_analytical.py): Analytical calculations and mathematical functions.
- [`GW_back.py`](https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/GW_back.py): Functions for cosmological GW backgrounds.
- [`GW_models.py`](https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/GW_models.py): Theoretical and numerical models for sources of GWs.
- [`GW_templates.py`](https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/GW_templates.py): Templates for GW backgrounds from different sources.
- [`hydro_bubbles.py`](https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/hydro_bubbles.py): Fluid perturbations from bubbles in first-order phase transitions.
- [`interferometry.py`](https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/interferometry.py): Response and sensitivity functions for GW detectors (LISA, Taiji).
- [`utils.py`](https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/utils.py): Utility functions and reference values for CosmoGW.
- [`tests/`](https://github.com/cosmoGW/cosmoGW/blob/main/tests/): Unit tests and example data.
- [`resources/`](https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/resources/): Data files used in the code (numerical data, sensitivity curves, ...)
- [`tutorials/`](https://github.com/cosmoGW/cosmoGW/blob/main/tutorials/): Tutorials showing how to use CosmoGW for different applications.

---

<img src="https://img.icons8.com/ios-filled/50/000000/book.png" width="24" /> **Documentation**

Comprehensive documentation is available at [Read the Docs](https://cosmogw-manual.readthedocs.io/).

- **API Reference:** Detailed documentation for all modules and functions.
- **Tutorials:** Step-by-step guides and example workflows.
- **Theory:** Background information and mathematical details.

To build the documentation locally:

```sh
cd docs
make html
```

---

<img src="https://img.icons8.com/ios-filled/50/000000/education.png" width="24" /> **Tutorials**

Tutorials and example notebooks are available in the [`tutorials/`](https://github.com/cosmoGW/cosmoGW/blob/main/tutorials/) directory:

- [**GWs_sound-waves**](https://github.com/cosmoGW/cosmoGW/blob/main/tutorials/GWs_sound-waves.ipynb): Tutorial showing different templates of GWs from sound waves and non-linear compressional fluid perturbations from first-order phase transitions.
- [**GWs_MF_from_FOPT**](https://github.com/cosmoGW/cosmoGW/blob/main/tutorials/GWs_MF_from_FOPT.ipynb): Tutorial showing the analysis of GWs from sound waves and turbulence from first-order phase transitions with LISA in connection to primordial magnetic fields.

These tutorials can be run from [*Google Colab*](https://colab.research.google.com/)
by loading the notebooks from GitHub.

When using Colab, latex needs to be installed, together with CosmoGW, for plotting routines.
For the installation, uncomment and execute the first cell of the tutorials:

```bash
pip install cosmoGW
sudo apt-get update
sudo apt-get install -y cm-super dvipng texlive-latex-extra texlive-latex-recommended
```

---

<img src="https://img.icons8.com/ios-filled/50/000000/test-passed.png" width="24" /> **Tests**

Unit tests are provided in the [`tests/`](https://github.com/cosmoGW/cosmoGW/blob/main/tests/) directory.

To run all tests (from cosmoGW directory):

```sh
pytest
```

Make sure pytest is installed and that all required test data files are present in the `tests` directory.

Further test is available using the tutorial notebooks (see [`python-package_nbs.yml`]).
To test the output plots and results from the notebooks run (e.g. from GWs_sound-waves tutorial):

```sh
pytest --nbval tutorials/GWs_sound-waves.ipynb
```

Make sure to have installed pytest, nbval and LaTeX:

*LaTeX on Ubuntu/Debian:*
```bash
pip install pytest nbval
sudo apt-get update
sudo apt-get install -y cm-super dvipng texlive-latex-extra texlive-latex-recommended
```

*LaTeX on macOS (using Homebrew):*
```bash
pip install pytest nbval
brew install --cask mactex
```

---

<img src="https://img.icons8.com/ios-filled/50/000000/database.png" width="24" /> **Resources**

Data files are available in cosmoGW that are required by some of the libraries.

They are stored in `src/cosmoGW/resources`:

- [`cosmology`](https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/resources/cosmology): Cosmological evolution files and Friedmann equation tutorial.
- [`interferometry`](https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/resources/interferometry): Space-based GW interferometry calculations and tutorials.
- [`detector_sensitivity`](https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/resources/detector_sensitivity): Sensitivity of various detectors; see the [README](https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/resources/detector_sensitivity/README.md).
- [`higgsless`](https://github.com/cosmoGW/cosmoGW/blob/main/src/cosmoGW/resources/higgsless): Data sets from Higgsless simulations of phase transitions.

---

📄 **License**

This project is licensed under the GNU General Public License v3 (GPLv3).

---

🚀 **Enjoy exploring the early Universe with gravitational waves!** 🪐