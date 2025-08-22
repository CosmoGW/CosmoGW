
# test based on the GWs_sound-waves tutorial

import numpy as np
import pandas as pd
import astropy.units as u
import unittest

from cosmoGW import GW_models, GW_templates, hydro_bubbles
from cosmoGW import COSMOGW_HOME
from cosmoGW.utils import safe_trapezoid, cs2_ref, dt0_ref

import os
test_dir = os.path.dirname(__file__)
file_path = os.path.join(test_dir, "GWs_sound_waves")


class TestUnits(unittest.TestCase):

    def test_GWs_sound_waves(self):

        # reference values
        s = np.logspace(-3, 3, 1000)
        alphas = np.logspace(np.log10(4.6e-3), np.log10(0.5), 5)
        vws = np.array([0.36, 0.6, 0.76])
        betas = [12, 100, 1000]

        GW_templates.ampl_GWB_sw(
            model='higgsless', vws=vws, alphas=alphas,
            numerical=True, quiet=True, bs_HL=20
        )

        # temperature scale and relativistic degrees of freedom of
        # the phase transition
        T = 100*u.GeV
        g = 100

        # GW spectrum obtained considering sound waves, taking a fixed
        # amplitude OmGWtilde = 1e-2, taking the kinetic energy fraction
        # from the single bubble results of Espinosa:2010hh, and the
        # spectral shape of Caprini:2024hue (sw_LISA)

        # In all cases, we consider an expanding Universe with the sourcing
        # process taking place during the radiation-dominated era

        freq, OmGW = GW_templates.OmGW_spec_sw(
            s, alphas, betas, vws=vws, expansion=True, Nsh=1.,
            model_efficiency='fixed_value', model_K0='Espinosa',
            model_decay='sound_waves', model_shape='sw_LISA',
            redshift=True, T=T, gstar=g
        )

        # GW spectrum obtained considering decaying compressional motion,
        # taking the amplitude OmGWtilde, the kinetic energy fraction,
        # and the spectral shape (sw_HLnew) parameters interpolated
        # from the simulation results of Caprini:2024gyk.
        # As the model continues after the shock formation time,
        # we allow the time of sourcing to
        # be larger than one shock formation time (Nsh = 100).

        freq2, OmGW2 = GW_templates.OmGW_spec_sw(
            s, alphas, betas, vws=vws, expansion=True, Nsh=100,
            model_efficiency='higgsless', model_K0='higgsless',
            model_decay='decay', interpolate_HL_decay=True,
            model_shape='sw_HLnew', interpolate_HL_shape=True,
            interpolate_HL_n3=True, redshift=True, T=T, gstar=g
        )

        # check that the output is correct
        data = np.load(file_path + '/Oms_sws1.npz')
        self.assertTrue(np.allclose(data['freq'], freq.value))
        self.assertTrue(np.allclose(data['OmGW'], OmGW))
        self.assertTrue(np.allclose(data['freq2'], freq2.value))
        self.assertTrue(np.allclose(data['OmGW2'], OmGW2))

    def test_read_Omega_higgsless(self):

        alphas = np.logspace(-3, 0, 30)
        vws = np.linspace(0.1, 0.99, 50)

        OmGWtilde = GW_templates.ampl_GWB_sw(
            model='higgsless', vws=vws, bs_HL=20,
            alphas=alphas, numerical=False, quiet=True
        )

        # check that the output is correct
        data = np.load(file_path + '/OmGWtilde_HL.npz')
        self.assertTrue(np.allclose(data['OmGWtilde2'], OmGWtilde))

    def test_compute_read_K0s(self):

        cs2 = cs2_ref
        vws = np.linspace(0.1, 0.99, 1000)
        val_alphas = np.array([0.0046, 0.05, 0.5])
        alphas = np.logspace(-3, 0, 30)
        K_xi = hydro_bubbles.kappas_Esp(vws, alphas) * alphas / (1 + alphas)
        Oms_xi = hydro_bubbles.kappas_Esp(vws, alphas) * alphas / (1 + cs2)

        dirr = COSMOGW_HOME + 'resources/higgsless/parameters_fit_sims.csv'
        df = pd.read_csv(dirr)

        K0, K0num, _, _ = GW_templates.interpolate_HL_vals(
            df, vws, val_alphas, quiet=True, numerical=True,
            value='curly_K_0_512', boxsize=20
        )

        data = np.load(file_path + '/K0_Esp_HL.npz')

        # check that the output is correct
        self.assertTrue(np.allclose(data['K_xi'], K_xi))
        self.assertTrue(np.allclose(data['Oms_xi'], Oms_xi))
        self.assertTrue(np.allclose(data['K0'], K0))
        self.assertTrue(np.allclose(data['K0num'], K0num))

    def test_tdurs_Esp(self):

        cs2 = cs2_ref
        _, _, val_alphas, val_vws = GW_templates.ampl_GWB_sw(
            model='higgsless', vws=[0.1], alphas=[0.1],
            numerical=True, quiet=True
        )

        Oms_xi2 = hydro_bubbles.kappas_Esp(
            val_vws, val_alphas
        ) * val_alphas/(1 + cs2)
        lf2 = hydro_bubbles.Rstar_beta(val_vws, corr=True)

        Oms_xi2 = hydro_bubbles.kappas_Esp(
            val_vws, val_alphas
        ) * val_alphas / (1 + cs2)
        lf2 = hydro_bubbles.Rstar_beta(val_vws, corr=True)
        lf2, _ = np.meshgrid(lf2, val_alphas, indexing='ij')
        dtdur = lf2 / np.sqrt(Oms_xi2)

        # check that the output is correct
        data = np.load(file_path + '/tdur_eddy_HL.npz')
        self.assertTrue(np.allclose(data['dtdur'], dtdur))

    def test_K2ints_flat(self):

        dtfins = np.logspace(-6, 6, 100)
        cs2 = cs2_ref

        dirr = COSMOGW_HOME + 'resources/higgsless/parameters_fit_sims.csv'
        df = pd.read_csv(dirr)

        _, bnum, val_alphas, val_vws = GW_templates.interpolate_HL_vals(
            df, [0.1], [0.1], quiet=True, numerical=True,
            value='b', boxsize=20
        )

        K2int_flat = np.zeros((len(dtfins), len(val_vws), len(val_alphas)))
        lf2 = hydro_bubbles.Rstar_beta(val_vws, corr=True)
        lf2, _ = np.meshgrid(lf2, val_alphas, indexing='ij')
        Oms_xi2 = hydro_bubbles.kappas_Esp(
            val_vws, val_alphas
        ) * val_alphas / (1 + cs2)
        dtdur = lf2 / np.sqrt(Oms_xi2)
        K2int_flat_tdur = np.zeros((len(val_vws), len(val_alphas)))

        for i in range(len(val_vws)):
            for j in range(len(val_alphas)):
                K2int_flat[:, i, j] = GW_models.K2int(
                    dtfins, K0=1., b=bnum[i, j], expansion=False
                )
                K2int_flat_tdur[i, j] = GW_models.K2int(
                    dtdur[i, j], K0=1., b=bnum[i, j], expansion=False
                )

        # check that the output is correct
        data = np.load(file_path + '/K2ints_flat_HL.npz')
        self.assertTrue(np.allclose(data['K2int_flat'], K2int_flat))
        self.assertTrue(np.allclose(data['K2int_flat_tdur'], K2int_flat_tdur))

    def test_K2ints_exp(self):

        dtfins = np.logspace(-6, 6, 100)
        betas = np.logspace(1, 3, 3)
        cs2 = cs2_ref

        dirr = COSMOGW_HOME + 'resources/higgsless/parameters_fit_sims.csv'
        df = pd.read_csv(dirr)

        _, bnum, val_alphas, val_vws = GW_templates.interpolate_HL_vals(
            df, [0.1], [0.1], quiet=True, numerical=True,
            value='b', boxsize=20
        )

        K2int_exp = np.zeros((len(dtfins), len(val_vws), len(val_alphas)))
        lf2 = hydro_bubbles.Rstar_beta(val_vws, corr=True)
        lf2, _ = np.meshgrid(lf2, val_alphas, indexing='ij')
        Oms_xi2 = hydro_bubbles.kappas_Esp(
            val_vws, val_alphas
        ) * val_alphas / (1 + cs2)
        dtdur = lf2 / np.sqrt(Oms_xi2)
        K2int_exp_tdur = np.zeros((len(val_vws), len(val_alphas)))

        l = 1
        for i in range(len(val_vws)):
            for j in range(len(val_alphas)):
                K2int_exp[:, i, j] = GW_models.K2int(
                    dtfins, K0=1., dt0=dt0_ref,
                    b=bnum[i, j], expansion=True, beta=betas[l]
                )
                K2int_exp_tdur[i, j] = GW_models.K2int(
                    dtdur[i, j] / betas[l], dt0=dt0_ref,
                    K0=1., b=bnum[i, j], expansion=True, beta=betas[l]
                )

        # check that the output is correct
        data = np.load(file_path + '/K2ints_exp_HL.npz')
        self.assertTrue(np.allclose(data['K2int_exp'], K2int_exp))
        self.assertTrue(np.allclose(data['K2int_exp_tdur'], K2int_exp_tdur))

    def test_prefs_sw_HL(self):

        # compute the GW prefactor as a function of an array of vws,
        # alphas, betas
        cs2 = cs2_ref
        alphas = np.logspace(np.log10(4.6e-3), np.log10(0.5), 5)
        vws = np.array([0.36, 0.6, 0.76])
        Oms_xi = hydro_bubbles.kappas_Esp(vws, alphas) * alphas / (1 + cs2)

        dirr = COSMOGW_HOME + 'resources/higgsless/parameters_fit_sims.csv'
        df = pd.read_csv(dirr)

        b, _, _, _ = GW_templates.interpolate_HL_vals(
            df, vws, alphas, quiet=True, numerical=True,
            value='b', boxsize=20
        )

        betas = np.logspace(2, 3, 2)
        pref_exp = np.zeros((len(vws), len(alphas), len(betas)))
        pref_exp2 = np.zeros((len(vws), len(alphas), len(betas)))
        pref_sw_exp = np.zeros((len(vws), len(alphas), len(betas)))
        for i in range(len(vws)):
            lf = hydro_bubbles.Rstar_beta(vws[i], corr=True) / betas
            for j in range(len(alphas)):
                for l in range(len(betas)):
                    pref_exp[i, j, l] = GW_templates.pref_GWB_sw(
                        Oms=Oms_xi[i, j], lf=lf[l], alpha=alphas[j],
                        model='decay', Nshock=1, b=b[i, j],
                        expansion=True, beta=betas[l], cs2=cs2
                    )
                    pref_exp2[i, j, l] = GW_templates.pref_GWB_sw(
                        Oms=Oms_xi[i, j], lf=lf[l], alpha=alphas[j],
                        model='decay', Nshock=50, b=b[i, j],
                        expansion=True, beta=betas[l], cs2=cs2
                    )
                    pref_sw_exp[i, j, l] = GW_templates.pref_GWB_sw(
                        Oms=Oms_xi[i, j], lf=lf[l], alpha=alphas[j],
                        model='sound_waves', Nshock=1., b=b[i, j],
                        expansion=True, beta=betas[l], cs2=cs2
                    )

        data = np.load(file_path + '/prefs_HL.npz')
        # check that the output is correct
        self.assertTrue(np.allclose(pref_exp, data['pref_exp']))
        self.assertTrue(np.allclose(pref_exp2, data['pref_exp2']))
        self.assertTrue(np.allclose(pref_sw_exp, data['pref_sw_exp']))

    def test_shapes_sw_HL(self):

        s = np.logspace(-3, 5, 1000)

        # sw_LISAold model
        S = GW_templates.Sf_shape_sw(s, model='sw_LISAold')
        S0 = 1. / safe_trapezoid(S, np.log(s))
        S_swLISAold = S * S0

        # sw_SSM model
        Dw = np.linspace(0.1, 0.5, 5)
        S = GW_templates.Sf_shape_sw(s, Dw=Dw, model='sw_SSM')
        S0 = 1. / safe_trapezoid(S, np.log(s), axis=0)
        S_swSSM = S * S0

        # sw_HL model
        S = GW_templates.Sf_shape_sw(s, Dw=Dw, model='sw_HL')
        S0 = 1. / safe_trapezoid(S, np.log(s), axis=0)
        S_swHL = S * S0

        data = np.load(file_path + '/shapes_sw_HL.npz')
        # check that the output is correct
        self.assertTrue(np.allclose(data['S_swLISAold'], S_swLISAold))
        self.assertTrue(np.allclose(data['S_swSSM'], S_swSSM))
        self.assertTrue(np.allclose(data['S_swHL'], S_swHL))

    def test_shapes2_sw_HL(self):

        s = np.logspace(-3, 5, 1000)
        Dw = np.linspace(0.1, 0.5, 5)

        # spectral shape using model sw_LISA
        S = GW_templates.Sf_shape_sw(s, Dw=Dw, model='sw_LISA')
        S0 = 1. / safe_trapezoid(S, np.log(s), axis=0)
        S_sw_LISA = S * S0

        # spectral shape using model sw_HLnew
        S = GW_templates.Sf_shape_sw(s, Dw=Dw, model='sw_HLnew')
        S0 = 1. / safe_trapezoid(S, np.log(s), axis=0)
        S_sw_HL = S * S0

        data = np.load(file_path + '/shapes2_sw_HL.npz')
        # check that the output is correct
        self.assertTrue(np.allclose(data['S_sw_LISA'], S_sw_LISA))
        self.assertTrue(np.allclose(data['S_sw_HL'], S_sw_HL))

    def test_shapes3_sw_HL(self):

        s = np.logspace(-3, 5, 1000)

        # spectral shape using model sw_HLnew
        S = GW_templates.Sf_shape_sw(s, model='sw_HLnew', strength='strong')
        S0 = 1. / safe_trapezoid(S, np.log(s), axis=0)
        S_sw_HLstr = S * S0

        S = GW_templates.Sf_shape_sw(s, model='sw_HLnew', strength='interm')
        S0 = 1. / safe_trapezoid(S, np.log(s), axis=0)
        S_sw_HLint = S * S0

        data = np.load(file_path + '/shapes3_sw_HL.npz')
        # check that the output is correct
        self.assertTrue(np.allclose(data['S_sw_HLstr'], S_sw_HLstr))
        self.assertTrue(np.allclose(data['S_sw_HLint'], S_sw_HLint))

    def test_shell_thickness_HL(self):

        cs = np.sqrt(cs2_ref)
        vwss = np.linspace(0.1, 0.99, 5)
        val_alphas = np.array([0.0046, 0.05, 0.5])
        _, _, _, _, _, _, xi_shocks, _ = \
            hydro_bubbles.compute_profiles_vws_multalp(
                val_alphas, vws=vwss
            )
        Dw = np.zeros((len(vwss), len(val_alphas)))
        for i in range(len(val_alphas)):
            Dw[:, i] = (
                xi_shocks[:, i] - np.minimum(vwss, cs)
            ) / np.maximum(vwss, cs)

        data = np.load(file_path + '/shell_thickness_HL.npz')
        # check that the output is correct
        self.assertTrue(np.allclose(Dw, data['Dw']))
