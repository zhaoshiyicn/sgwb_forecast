#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import json
import types

import numpy as np
from enterprise import constants as const
from enterprise.pulsar import Pulsar
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import gp_priors
from enterprise.signals import utils
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_bases
from enterprise_extensions.blocks import common_red_noise_block as crn_block
# from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise_extensions import hypermodel, model_orfs
from enterprise_extensions.frequentist.optimal_statistic import OptimalStatistic as OS
from enterprise_extensions.chromatic.solar_wind import solar_wind, createfourierdesignmatrix_solar_dm
from enterprise.signals.parameter import function

from crn import get_crn_model_dict

def model(psrs, noisesdict: dict, wn_const=True, add_rn=True, rn_comp=30, crn_comp=30, multi=False, crn_name='pl_hd_freegam'):

    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)
    fundamental_freq = 1 / Tspan

    max_freq = 1.0/240.0/86400.0 #1 / 240 days
    components = int(max_freq / fundamental_freq)

    selection = selections.Selection(selections.by_backend)

    if wn_const:
        efac = parameter.Constant() 
        tnequad = parameter.Constant() 
    else:
        efac = parameter.Uniform(0.01, 2)
        tnequad = parameter.Uniform(-8.5, -5)

    # red noise parameters
    log10_A = parameter.Uniform(-20, -11)
    gamma = parameter.Uniform(0, 7)
    red_cadence = 240
    rn_pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)


    ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
    eq = white_signals.TNEquadNoise(log10_tnequad=tnequad, selection=selection)

    rn = gp_signals.FourierBasisGP(spectrum=rn_pl, components=rn_comp, Tspan=Tspan)

    tm = gp_signals.MarginalizingTimingModel(use_svd=True)

    if add_rn:
        s = tm + ef + eq + rn
    else:
        s = tm + ef + eq
    if multi:
        crn_name1, crn_name2 = crn_name.split('+')
        crn1 = get_crn_model_dict(Tspan, components=crn_comp)[crn_name1]
        crn2 = get_crn_model_dict(Tspan, components=crn_comp)[crn_name2]
        s += (crn1 + crn2)
        pta = signal_base.PTA([s(p) for p in psrs])
        pta.set_default_params(noisesdict)

        return pta
    else:
        crn = get_crn_model_dict(Tspan, components=crn_comp)[crn_name]
        s += crn
        pta = signal_base.PTA([s(p) for p in psrs])
        pta.set_default_params(noisesdict)

        return pta
    # if crn_name is not None:
    #     if '-' in crn_name:
    #         crn_name1, crn_name2 = crn_name.split('-')
    #         crn1 = get_crn_model_dict(Tspan)[crn_name1]
    #         crn2 = get_crn_model_dict(Tspan)[crn_name2]
    #         s1 = s + crn1
    #         s2 = s + crn2
    #         pta1 = signal_base.PTA([s1(p) for p in psrs])
    #         pta1.set_default_params(noisesdict)
    #         pta2 = signal_base.PTA([s2(p) for p in psrs])
    #         pta2.set_default_params(noisesdict)
    #         return pta1, pta2
    #     else:
    #         crn = get_crn_model_dict(Tspan)[crn_name]
    #         s += crn
    #         pta = signal_base.PTA([s(p) for p in psrs])
    #         pta.set_default_params(noisesdict)

    #         return pta

def create_work_model(psrs, model_name: str, noisesdict: dict, wn_const=True, add_rn=True, rn_comp=30):

    flags = [s in model_name for s in ['-', '+']]

    if flags == [0, 0]:
        pta = model(psrs, noisesdict, wn_const, add_rn, rn_comp, crn_name=model_name)
        return pta

    elif flags == [1, 0]:
        crn_name1, crn_name2 = model_name.split('-')
        pta1 = model(psrs, noisesdict, wn_const, add_rn, rn_comp, crn_name=crn_name1)
        pta2 = model(psrs, noisesdict, wn_const, add_rn, rn_comp, crn_name=crn_name2)
        return pta1, pta2
    
    elif flags == [0, 1]:
        pta = model(psrs, noisesdict, wn_const, add_rn, rn_comp, multi=True, crn_name=model_name)
        return pta
    
    else:
        crn_name1, crn_name2 = model_name.split('-')
        pta1 = model(psrs, noisesdict, wn_const, add_rn, rn_comp, crn_name=crn_name1)
        pta2 = model(psrs, noisesdict, wn_const, add_rn, rn_comp, multi=True, crn_name=crn_name2)
        return pta1, pta2
