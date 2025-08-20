
## test based on the GWs_sound-waves tutorial

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
import unittest
import pickle
import os

from cosmoGW import plot_sets, GW_models, GW_templates, hydro_bubbles
# import cosmoGW.plot_sets as plot_sets
# import cosmoGW.GW_models as mod
# import cosmoGW.GW_analytical  as an
# import cosmoGW.GW_templates   as tmp
# import cosmoGW.hydro_bubbles  as hb
# import cosmoGW.cosmology      as co
# import cosmoGW.interferometry as inte
# import cosmoGW.GW_back        as cGW
from   cosmoGW import COSMOGW_HOME

class TestUnits(unittest.TestCase):

    def test_GWs_sound_waves(self):

        # reference values
        cs2 = hydro_bubbles.cs2_ref    # default value is 1/3
        cs  = np.sqrt(cs2)

        # range of frequencies normalized with the fluid length scale s = fR*
        s   = np.logspace(-3, 3, 1000)

        alphas = np.logspace(np.log10(4.6e-3), np.log10(0.5),  5)
        vws    = np.array([0.36, 0.6, .76])
        betas  = [12, 100, 1000]

        _, _, _, _  = GW_templates.ampl_GWB_sw(model='higgsless', vws=vws,
                                      alphas=alphas, numerical=True, quiet=True,
                                      bs_HL=20)

        # temperature scale and relativistic degrees of freedom of the phase transition
        T = 100*u.GeV
        g = 100

        # GW spectrum obtained considering sound waves, taking a fixed amplitude OmGWtilde = 1e-2,
        # taking the kinetic energy fraction from the single bubble results of Espinosa:2010hh, and the
        # spectral shape of Caprini:2024hue (sw_LISA)

        # In all cases, we consider an expanding Universe with the sourcing process taking place during
        # the radiation-dominated era

        freq, OmGW = GW_templates.OmGW_spec_sw(s, alphas, betas, vws=vws, expansion=True, Nsh=1.,
                      model_efficiency='fixed_value', model_K0='Espinosa',
                      model_decay='sound_waves', model_shape='sw_LISA', redshift=True, T=T, gstar=g)


        # GW spectrum obtained considering decaying compressional motion, taking the amplitude OmGWtilde, the
        # kinetic energy fraction, and the spectral shape (sw_HLnew) parameters interpolated from the simulation results
        # of Caprini:2024gyk. As the model continues after the shock formation time, we allow the time of sourcing to
        # be larger than one shock formation time (Nsh = 100).

        freq2, OmGW2 = GW_templates.OmGW_spec_sw(s, alphas, betas, vws=vws, expansion=True, Nsh=100,
                 model_efficiency='higgsless', model_K0='higgsless',
                 model_decay='decay', interpolate_HL_decay=True, model_shape='sw_HLnew', interpolate_HL_shape=True,
                 interpolate_HL_n3=True, redshift=True, T=T, gstar=g)

        ### read test data

        f = open(COSMOGW_HOME + 'tests/GWs_sound_waves/Oms_sws1.pkl', 'rb')
        freq_tst, OmGW_tst, freq2_tst, OmGW2_tst = pickle.load(f)
        f.close()

        # check that the output is correct
        self.assertTrue(np.all((freq_tst - freq) == 0.0))
        self.assertTrue(np.all((OmGW_tst - OmGW) == 0.0))
        self.assertTrue(np.all((freq2_tst - freq2) == 0.0))
        self.assertTrue(np.all((OmGW2_tst - OmGW2) == 0.0))

    def test_read_Omega_higgsless(self):

        alphas = np.logspace(-3, 0,  30)
        vws    = np.linspace(0.1, .99, 50)
    
        OmGWtilde = GW_templates.ampl_GWB_sw(model='higgsless', vws=vws, bs_HL=20,
                            alphas=alphas, numerical=False, quiet=True)

        f = open(COSMOGW_HOME + 'tests/GWs_sound_waves/OmGWtilde_HL.pkl', 'rb')
        OmGWtilde2 = pickle.load(f)
        f.close()

        # check that the output is correct
        self.assertTrue(np.all((OmGWtilde - OmGWtilde2) == 0.0))

    def test_compute_read_K0s(self):

        cs2    = hydro_bubbles.cs2_ref
        vws    = np.linspace(0.1, .99, 1000)
        val_alphas = np.array([0.0046, 0.05, 0.5])
        alphas = np.logspace(-3, 0,  30)
        K_xi   = hydro_bubbles.kappas_Esp(vws, alphas)*alphas/(1 + alphas)
        Oms_xi = hydro_bubbles.kappas_Esp(vws, alphas)*alphas/(1 + cs2)

        dirr = COSMOGW_HOME + 'resources/higgsless/parameters_fit_sims.csv'
        df   = pd.read_csv(dirr)

        K0, K0num, _, _ = GW_templates.interpolate_HL_vals(df, vws, val_alphas, quiet=True, numerical=True,
                               value='curly_K_0_512', boxsize=20)

        f = open(COSMOGW_HOME + 'tests/GWs_sound_waves/K0_Esp_HL.pkl', 'rb')
        [K_xi_tst, Oms_xi_tst, K0_tst, K0num_tst] = pickle.load(f)
        f.close()

        # check that the output is correct
        self.assertTrue(np.all((K_xi_tst - K_xi) == 0.0))
        self.assertTrue(np.all((Oms_xi_tst - Oms_xi) == 0.0))
        self.assertTrue(np.all((K0_tst - K0) == 0.0))
        self.assertTrue(np.all((K0num_tst - K0num) == 0.0))

    def test_tdurs_Esp(self):

        cs2 = hydro_bubbles.cs2_ref

        dtfins  = np.logspace(-6, 6, 100)
        betas   = np.logspace(1, 3, 3)

        _, _, val_alphas, val_vws  = GW_templates.ampl_GWB_sw(model='higgsless', vws=[.1], alphas=[.1],
                                                     numerical=True, quiet=True)

        Oms_xi2 = hydro_bubbles.kappas_Esp(val_vws, val_alphas)*val_alphas/(1 + cs2)
        lf2     = hydro_bubbles.Rstar_beta(val_vws, corr=True)

        lf2, _ = np.meshgrid(lf2, val_alphas, indexing='ij')
        dtdur  = lf2/np.sqrt(Oms_xi2)

        f = open(COSMOGW_HOME + 'tests/GWs_sound_waves/tdur_eddy_HL.pkl', 'rb')
        dtdur_tst = pickle.load(f)
        f.close()

        # check that the output is correct
        self.assertTrue(np.all((dtdur_tst - dtdur) == 0.0))

    def test_K2ints_flat(self):

        dtfins = np.logspace(-6, 6, 100)
        cs2    = hydro_bubbles.cs2_ref

        dirr = COSMOGW_HOME + 'resources/higgsless/parameters_fit_sims.csv'
        df   = pd.read_csv(dirr)
        
        b, bnum, val_alphas, val_vws = \
            GW_templates.interpolate_HL_vals(df, [.1], [.1], quiet=True, numerical=True,
                                    value='b', boxsize=20)

        K2int_flat = np.zeros((len(dtfins), len(val_vws), len(val_alphas)))
        lf2    = hydro_bubbles.Rstar_beta(val_vws, corr=True)
        lf2, _ = np.meshgrid(lf2, val_alphas, indexing='ij')
        Oms_xi2 = hydro_bubbles.kappas_Esp(val_vws, val_alphas)*val_alphas/(1 + cs2)
        dtdur  = lf2/np.sqrt(Oms_xi2)
        K2int_flat_tdur = np.zeros((len(val_vws), len(val_alphas)))

        for i in range(0, len(val_vws)):
            for j in range(0, len(val_alphas)):
                K2int_flat[:, i, j]   = GW_models.K2int(dtfins, K0=1., b=bnum[i, j],
                                                expansion=False)
                K2int_flat_tdur[i, j] = GW_models.K2int(dtdur[i, j], K0=1.,
                                                  b=bnum[i, j], expansion=False)

        f = open(COSMOGW_HOME + 'tests/GWs_sound_waves/K2ints_flat_HL.pkl', 'rb')
        K2int_flat_tst, K2int_flat_tdur_tst = pickle.load(f)
        f.close()

        # check that the output is correct
        self.assertTrue(np.all((K2int_flat_tst - K2int_flat) == 0.0))
        self.assertTrue(np.all((K2int_flat_tdur_tst - K2int_flat_tdur) == 0.0))

    def test_K2ints_exp(self):

        dtfins = np.logspace(-6, 6, 100)
        betas  = np.logspace(1, 3, 3)
        cs2    = hydro_bubbles.cs2_ref

        dirr = COSMOGW_HOME + 'resources/higgsless/parameters_fit_sims.csv'
        df   = pd.read_csv(dirr)
        
        b, bnum, val_alphas, val_vws = \
            GW_templates.interpolate_HL_vals(df, [.1], [.1], quiet=True, numerical=True,
                                    value='b', boxsize=20)

        K2int_exp = np.zeros((len(dtfins), len(val_vws), len(val_alphas)))
        lf2    = hydro_bubbles.Rstar_beta(val_vws, corr=True)
        lf2, _ = np.meshgrid(lf2, val_alphas, indexing='ij')
        Oms_xi2 = hydro_bubbles.kappas_Esp(val_vws, val_alphas)*val_alphas/(1 + cs2)
        dtdur  = lf2/np.sqrt(Oms_xi2)
        K2int_exp_tdur = np.zeros((len(val_vws), len(val_alphas)))

        l = 1
        for i in range(0, len(val_vws)):
            for j in range(0, len(val_alphas)):
                K2int_exp[:, i, j]   = GW_models.K2int(dtfins, K0=1., dt0=GW_models.dt0_ref,
                          b=bnum[i, j], expansion=True, beta=betas[l])
                K2int_exp_tdur[i, j] = GW_models.K2int(dtdur[i, j]/betas[l],
                          dt0=GW_models.dt0_ref, K0=1., b=bnum[i, j], expansion=True,
                          beta=betas[l])

        f = open(COSMOGW_HOME + 'tests/GWs_sound_waves/K2ints_exp_HL.pkl', 'rb')
        K2int_exp_tst, K2int_exp_tdur_tst = pickle.load(f)
        f.close()

        # check that the output is correct
        self.assertTrue(np.all((K2int_exp_tst - K2int_exp) == 0.0))
        self.assertTrue(np.all((K2int_exp_tdur_tst - K2int_exp_tdur) == 0.0))

    def test_prefs_sw_HL(self):

        ### compute the GW prefactor as a function of an array of vws, alphas, betas

        cs2    = hydro_bubbles.cs2_ref
        alphas = np.logspace(np.log10(4.6e-3), np.log10(0.5),  5)
        vws    = np.array([0.36, 0.6, .76])
        Oms_xi = hydro_bubbles.kappas_Esp(vws, alphas)*alphas/(1 + cs2)

        dirr = COSMOGW_HOME + 'resources/higgsless/parameters_fit_sims.csv'
        df   = pd.read_csv(dirr)
        
        b, _, _, _ = \
            GW_templates.interpolate_HL_vals(df, vws, alphas, quiet=True, numerical=True,
                                    value='b', boxsize=20)

        betas       = np.logspace(2, 3, 2)
        pref_exp    = np.zeros((len(vws), len(alphas), len(betas)))
        pref_exp2   = np.zeros((len(vws), len(alphas), len(betas)))
        pref_sw_exp = np.zeros((len(vws), len(alphas), len(betas)))
        for i in range(0, len(vws)):
            # ratio Rstar/beta using the correction max(vw, cs)
            lf = hydro_bubbles.Rstar_beta(vws[i], corr=True)/betas
            for j in range(0, len(alphas)):
                for l in range(0, len(betas)):
                    pref_exp[i, j, l]    = \
                        GW_templates.pref_GWB_sw(Oms=Oms_xi[i, j], lf=lf[l], alpha=alphas[j], model='decay',
                                        Nshock=1, b=b[i, j], expansion=True, beta=betas[l], cs2=cs2)
                    pref_exp2[i, j, l]   = \
                        GW_templates.pref_GWB_sw(Oms=Oms_xi[i, j], lf=lf[l], alpha=alphas[j], model='decay',
                                        Nshock=50, b=b[i, j], expansion=True, beta=betas[l], cs2=cs2)
                    pref_sw_exp[i, j, l] = \
                        GW_templates.pref_GWB_sw(Oms=Oms_xi[i, j], lf=lf[l], alpha=alphas[j], model='sound_waves',
                                        Nshock=1., b=b[i, j], expansion=True, beta=betas[l], cs2=cs2)

        f = open(COSMOGW_HOME + 'tests/GWs_sound_waves/prefs_HL.pkl', 'rb')
        pref_exp_tst, pref_exp2_tst, pref_sw_exp_tst = pickle.load(f)
        f.close()

        # check that the output is correct
        self.assertTrue(np.all((pref_exp_tst    - pref_exp)    == 0.0))
        self.assertTrue(np.all((pref_exp2_tst   - pref_exp2)   == 0.0))
        self.assertTrue(np.all((pref_sw_exp_tst - pref_sw_exp) == 0.0))

    def test_shapes_sw_HL(self):

        s = np.logspace(-3, 5, 1000)

        # sw_LISAold model
        S  = GW_templates.Sf_shape_sw(s, model='sw_LISAold')
        S0 = 1./np.trapezoid(S, np.log(s))
        S_swLISAold  = S*S0

        # sw_SSM model
        Dw = np.linspace(0.1, 0.5, 5)
        S  = GW_templates.Sf_shape_sw(s, Dw=Dw, model='sw_SSM')
        S0 = 1./np.trapezoid(S, np.log(s), axis=0)
        S_swSSM  = S*S0

        # sw_HL model
        S  = GW_templates.Sf_shape_sw(s, Dw=Dw, model='sw_HL')
        S0 = 1./np.trapezoid(S, np.log(s), axis=0)
        S_swHL = S*S0

        f = open(COSMOGW_HOME + 'tests/GWs_sound_waves/shapes_sw_HL.pkl', 'rb')
        S_swLISAold_tst, S_swSSM_tst, S_swHL_tst = pickle.load(f)
        f.close()

        # check that the output is correct
        self.assertTrue(np.all((S_swLISAold_tst - S_swLISAold) == 0.0))
        self.assertTrue(np.all((S_swSSM_tst     - S_swSSM)     == 0.0))
        self.assertTrue(np.all((S_swHL_tst      - S_swHL)      == 0.0))

    def test_shapes2_sw_HL(self):

        s  = np.logspace(-3, 5, 1000)
        Dw = np.linspace(0.1, 0.5, 5)

        # spectral shape using model sw_LISA
        S  = GW_templates.Sf_shape_sw(s, Dw=Dw, model='sw_LISA')
        S0 = 1./np.trapezoid(S, np.log(s), axis=0)
        S_sw_LISA = S*S0

        # spectral shape using model sw_HLnew
        S  = GW_templates.Sf_shape_sw(s, Dw=Dw, model='sw_HLnew')
        S0 = 1./np.trapezoid(S, np.log(s), axis=0)
        S_sw_HL = S*S0

        f = open(COSMOGW_HOME + 'tests/GWs_sound_waves/shapes2_sw_HL.pkl', 'rb')
        S_sw_LISA_tst, S_sw_HL_tst = pickle.load(f)
        f.close()

        # check that the output is correct
        self.assertTrue(np.all((S_sw_LISA_tst - S_sw_LISA) == 0.0))
        self.assertTrue(np.all((S_sw_HL_tst - S_sw_HL) == 0.0))

    def test_shapes3_sw_HL(self):

        s  = np.logspace(-3, 5, 1000)

        # spectral shape using model sw_HLnew
        S  = GW_templates.Sf_shape_sw(s, model='sw_HLnew', strength='strong')
        S0 = 1./np.trapezoid(S, np.log(s), axis=0)
        S_sw_HLstr = S*S0

        S     = GW_templates.Sf_shape_sw(s, model='sw_HLnew', strength='interm')
        S0 = 1./np.trapezoid(S, np.log(s), axis=0)
        S_sw_HLint = S*S0

        f = open(COSMOGW_HOME + 'tests/GWs_sound_waves/shapes3_sw_HL.pkl', 'rb')
        S_sw_HLstr_tst, S_sw_HLint_tst = pickle.load(f)
        f.close()

        # check that the output is correct
        self.assertTrue(np.all((S_sw_HLstr_tst - S_sw_HLstr) == 0.0))
        self.assertTrue(np.all((S_sw_HLint_tst - S_sw_HLint) == 0.0))

    def test_shell_thickness_HL(self):

      cs = np.sqrt(hydro_bubbles.cs2_ref)

      vwss = np.linspace(0.1, 0.99, 5)
      val_alphas = np.array([0.0046, 0.05, 0.5])
      _, _, _, _, _, _, xi_shocks,_ = \
                 hydro_bubbles.compute_profiles_vws_multalp(val_alphas, vws=vwss)
      Dw = np.zeros((len(vwss), len(val_alphas)))
      for i in range(0, len(val_alphas)):
          Dw[:, i] = (xi_shocks[:, i] - np.minimum(vwss, cs))/np.maximum(vwss, cs)

      f      = open(COSMOGW_HOME + 'tests/GWs_sound_waves/shell_thickness_HL.pkl', 'rb')
      Dw_tst = pickle.load(f)
      f.close()

      # check that the output is correct
      self.assertTrue(np.all((Dw - Dw_tst) == 0.0))
