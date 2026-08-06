# Tutorial: LISA and gamma-ray telescopes as multi-messenger probes of a first-order cosmological phase transition

This page documents the workflow shown in the tutorial notebook `tutorials/GWs_MF_from_FOPT.ipynb`.

The example demonstrates how to use CosmoGW to compute the gravitational-wave background from sound waves and turbulence in a first-order phase transition and compare it with observational sensitivities.

## Overview

The tutorial illustrates:

- how to define a first-order phase-transition scenario with parameters such as $\alpha$, $\beta/H_\ast$, $v_w$, and $T_\ast$;
- how to compute the GW spectra from sound waves and turbulence with CosmoGW;
- how to compare the resulting spectra with LISA sensitivity curves and PTA constraints;
- how to generate representative plots for the resulting signal.

## Key ingredients

The example uses the following CosmoGW components:

- `cosmology` for thermal quantities and cosmological conversions;
- `GW_templates` for the sound-wave and turbulence spectra;
- `interferometry` for sensitivity curves and SNR estimates;
- `plot_sets` for plotting utilities;
- `hydro_bubbles` and `GW_models` for related phase-transition quantities.

## Minimal example

The core workflow is the following:

```python
from cosmoGW import cosmology, GW_templates, plot_sets, interferometry, hydro_bubbles, GW_back, analysis, GW_models
import numpy as np
import astropy.units as u

alpha = 2
beta = 5
vw = 0.999999999
eps_turb = 1
T = 100 * u.MeV

# relativistic and adiabatic degrees of freedom
g = cosmology.thermal_g(T=T, s=0, file=True)
gS = cosmology.thermal_g(T=T, s=1, file=True)

# frequency range, normalized by mean bubble separation
s = np.logspace(-3, 4, 1000)

# spectrum from sound waves
freqs_sw_HL, OmGW_sw_HL = GW_templates.OmGW_spec_sw(
    s,
    alpha,
    beta,
    vws=vw,
    expansion=True,
    Nsh=1.,
    model_efficiency='fixed_value',
    model_K0='Espinosa',
    model_decay='sound_waves',
    model_shape='sw_HL',
    redshift=True,
    gstar=g,
    gS=gS,
    T=T,
)

# spectrum from turbulence
freqs_turb, OmGW_turb = GW_templates.OmGW_spec_turb_alphabeta(
    s,
    alpha,
    beta,
    vws=vw,
    eps_turb=eps_turb,
    redshift=True,
    gstar=g,
    gS=gS,
    T=T,
)

# LISA sensitivity curve
f_LISA, OmLISA, LISA_OmPLS = interferometry.read_sens(SNR=10, T=4)
```

## Notes

- This tutorial is intended as a high-level example rather than a fully reproducible documentation build from the notebook.
- If you want the page to be more detailed, you can later expand it with:
  - a short explanation of each parameter,
  - a figure generated from the example,
  - a link back to the notebook source.

## Related resources

- Notebook: `tutorials/GWs_MF_from_FOPT.ipynb`
- Main documentation index: `docs/source/index.rst`
