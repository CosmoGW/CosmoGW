# test based on the tutorial: tutorials/GWs_MF_from_FOPT.ipynb

from cosmoGW import analysis, cosmology, GW_templates, interferometry
import numpy as np
import astropy.units as u
import unittest

import os
test_dir = os.path.dirname(__file__)
file_path = os.path.join(test_dir, "sound_waves_turb_GWs")


class TestUnits(unittest.TestCase):

    def test_GWbackground_sws_turb_1p(self):
        alpha = 0.5
        beta = 10
        vw = 0.95
        eps_turb = 1
        T = 100 * u.GeV
        g = cosmology.thermal_g(T=T, s=0, file=True)
        gS = cosmology.thermal_g(T=T, s=1, file=True)
        s = np.logspace(-3, 4, 1000)
        freqs_sw, OmGW_sw = \
            GW_templates.OmGW_spec_sw(
                s, alpha, beta, vws=vw, expansion=True, Nsh=1.,
                model_efficiency='fixed_value', model_K0='Espinosa',
                model_decay='sound_waves', model_shape='sw_SSM',
                redshift=True, gstar=g, gS=gS, T=T
            )
        freqs_sw_HL, OmGW_sw_HL = \
            GW_templates.OmGW_spec_sw(
                s, alpha, beta, vws=vw, expansion=True, Nsh=1.,
                model_efficiency='fixed_value', model_K0='Espinosa',
                model_decay='sound_waves', model_shape='sw_HL',
                redshift=True, gstar=g, gS=gS, T=T
            )
        freqs_turb, OmGW_turb = \
            GW_templates.OmGW_spec_turb_alphabeta(
                s, alpha, beta, vws=vw,
                eps_turb=eps_turb, redshift=True,
                gstar=g, gS=gS, T=T
            )
        f_LISA, OmLISA, LISA_OmPLS = interferometry.read_sens(SNR=10, T=4)

        data = np.load(file_path + '/OmGWs_sound_waves_1p.npz')
        self.assertTrue(np.allclose(data['freqs'], freqs_sw.value))
        self.assertTrue(np.allclose(data['OmGW'], OmGW_sw))
        data = np.load(file_path + '/OmGWs_sound_waves_HL_1p.npz')
        self.assertTrue(np.allclose(data['freqs'], freqs_sw_HL.value))
        self.assertTrue(np.allclose(data['OmGW'], OmGW_sw_HL))
        data = np.load(file_path + '/OmGWs_turbulence_1p.npz')
        self.assertTrue(np.allclose(data['freqs'], freqs_turb.value))
        self.assertTrue(np.allclose(data['OmGW'], OmGW_turb))
        data = np.load(file_path + '/OmLISA_PLS_1p.npz')
        self.assertTrue(np.allclose(data['freqs'], f_LISA))
        self.assertTrue(np.allclose(data['OmLISA'], OmLISA))
        self.assertTrue(np.allclose(data['LISA_OmPLS'], LISA_OmPLS))

    def test_GWbackground_eps_turb_SNR(self):

        alphas = np.logspace(-3, 0,  4)
        betas = np.logspace(0, 3.2, 3)
        TTs = np.logspace(0, 6, 13)*u.GeV
        vws = np.linspace(0.4, .999, 13)
        eps_turb = np.array([1e-10, 0.1, 1])
        s = np.logspace(-3, 4, 1000)
        SNR = analysis.analysis_LISA_alphabeta(
                s, alphas, betas, vws, TTs, eps_turb=eps_turb,
                model_shape='sw_HL', turb=True
        )
        freqs_sw_HL, OmGWs_sw_HL = GW_templates.OmGW_spec_sw(
            s, alphas, betas, vws=vws, expansion=True, Nsh=1.,
            model_efficiency='fixed_value', model_K0='Espinosa',
            model_decay='sound_waves', model_shape='sw_HL', redshift=False
        )
        freqs_turb = np.zeros((len(eps_turb), len(s), len(vws), len(betas)))
        OmGWs_turb = np.zeros((
            len(eps_turb), len(s), len(vws), len(alphas), len(betas)
        ))
        for p in range(len(eps_turb)):
            freqs_turb[p], OmGWs_turb[p] = \
                GW_templates.OmGW_spec_turb_alphabeta(
                    s, alphas, betas, vws=vws, eps_turb=eps_turb[p],
                    redshift=False
                )

        data = np.load(file_path + '/GW_background_spectra_SNR.npz')
        self.assertTrue(np.allclose(data['freqs_sw'], freqs_sw_HL))
        self.assertTrue(np.allclose(data['OmGW_sw'], OmGWs_sw_HL))
        self.assertTrue(np.allclose(data['freqs_turb'], freqs_turb))
        self.assertTrue(np.allclose(data['OmGW_turb'], OmGWs_turb))
        self.assertTrue(np.allclose(data['SNR'], SNR))
