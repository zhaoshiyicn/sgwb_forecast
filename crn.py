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

from utils import omega_DW
h1 = 0.674
H0 = 2.192711267238057e-18
mpc = 3.0856e22
lc = 299792458  # speed of light
kb = 1.38064852e-23  # 玻尔兹曼常数
cg = 0.387


@signal_base.function
def zero_diag_crn(pos1, pos2):
    """
    Off-diagonal uncorrelated CRN correlation function (i.e. correlation = 0)
    Explicitly sets cross-correlation terms to 0. Auto terms are 1 (do not run with additional CURN term)
    """
    if np.all(pos1 == pos2):
        return 1e-20
    else:
        return 1e-20
    
@signal_base.function
def singlebin_orf(pos1, pos2, param):
    '''
    used for inferring the correlation for a single pair of pulsars.
    param is the correlation value for the pair, passed in as a Uniform distr.
    '''
    if np.all(pos1 == pos2):
        return 1
    else:
        return param
    
def common_red_noise_block(psd='powerlaw', prior='log-uniform',
                           Tspan=None, components=30, combine=True,
                           log10_A_val=None, gamma_val=None, delta_val=None,
                           logmin=None, logmax=None, select = None,
                           orf=None, orf_ifreq=0, leg_lmax=5,
                           name='gw', coefficients=False,
                           pshift=False, pseed=None):
    """
    MODIFIED TO TAKE `SELECT` AS A KWARG
    
    Returns common red noise model:

        1. Red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum', 'broken_powerlaw']
    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for individual pulsar.
    :param log10_A_val:
        Value of log10_A parameter for fixed amplitude analyses.
    :param gamma_val:
        Value of spectral index for power-law and turnover
        models. By default spectral index is varied of range [0,7]
    :param delta_val:
        Value of spectral index for high frequencies in broken power-law
        and turnover models. By default spectral index is varied in range [0,7].\
    :param logmin:
        Specify the lower bound of the prior on the amplitude for all psd but 'spectrum'.
        If psd=='spectrum', then this specifies the lower prior on log10_rho_gw
    :param logmax:
        Specify the lower bound of the prior on the amplitude for all psd but 'spectrum'.
        If psd=='spectrum', then this specifies the lower prior on log10_rho_gw
    :param orf:
        String representing which overlap reduction function to use.
        By default we do not use any spatial correlations. Permitted
        values are ['hd', 'dipole', 'monopole'].
    :param orf_ifreq:
        Frequency bin at which to start the Hellings & Downs function with
        numbering beginning at 0. Currently only works with freq_hd orf.
    :param leg_lmax:
        Maximum multipole of a Legendre polynomial series representation
        of the overlap reduction function [default=5]
    :param pshift:
        Option to use a random phase shift in design matrix. For testing the
        null hypothesis.
    :param pseed:
        Option to provide a seed for the random phase shift.
    :param name: Name of common red process

    """

    
    orfs = {'crn': None, 'hd': model_orfs.hd_orf(),
            'gw_monopole': model_orfs.gw_monopole_orf(),
            'gw_dipole': model_orfs.gw_dipole_orf(),
            'st': model_orfs.st_orf(),
            'gt': model_orfs.gt_orf(tau=parameter.Uniform(-1.5, 1.5)('tau')),
            'dipole': model_orfs.dipole_orf(),
            'monopole': model_orfs.monopole_orf(),
            'param_hd': model_orfs.param_hd_orf(a=parameter.Uniform(-1.5, 3.0)('gw_orf_param0'),
                                                b=parameter.Uniform(-1.0, 0.5)('gw_orf_param1'),
                                                c=parameter.Uniform(-1.0, 1.0)('gw_orf_param2')),
            'spline_orf': model_orfs.spline_orf(params=parameter.Uniform(-0.9, 0.9, size=7)('gw_orf_spline')),
            'bin_orf': model_orfs.bin_orf(params=parameter.Uniform(-1.0, 1.0, size=7)('gw_orf_bin')),
            'single_bin_orf': singlebin_orf(param=parameter.Uniform(-1.0, 1.0)('gw_orf_bin')),
            'zero_diag_crn': zero_diag_crn(),
            'zero_diag_hd': model_orfs.zero_diag_hd(),
            'zero_diag_bin_orf': model_orfs.zero_diag_bin_orf(params=parameter.Uniform(
                -1.0, 1.0, size=7)('gw_orf_bin_zero_diag')),
            'freq_hd': model_orfs.freq_hd(params=[components, orf_ifreq]),
            'legendre_orf': model_orfs.legendre_orf(params=parameter.Uniform(
                -1.0, 1.0, size=leg_lmax+1)('gw_orf_legendre')),
            'zero_diag_legendre_orf': model_orfs.zero_diag_legendre_orf(params=parameter.Uniform(
                -1.0, 1.0, size=leg_lmax+1)('gw_orf_legendre_zero_diag'))}

    # common red noise parameters
    if psd in ['powerlaw', 'turnover', 'turnover_knee', 'broken_powerlaw']:
        amp_name = '{}_log10_A'.format(name)
        if log10_A_val is not None:
            log10_Agw = parameter.Constant(log10_A_val)(amp_name)

        elif logmin is not None and logmax is not None:
            if prior == 'uniform':
                log10_Agw = parameter.LinearExp(logmin, logmax)(amp_name)
            elif prior == 'log-uniform' and gamma_val is not None:
                if np.abs(gamma_val - 4.33) < 0.1:
                    log10_Agw = parameter.Uniform(logmin, logmax)(amp_name)
                else:
                    log10_Agw = parameter.Uniform(logmin, logmax)(amp_name)
            else:
                log10_Agw = parameter.Uniform(logmin, logmax)(amp_name)

        else:
            if prior == 'uniform':
                log10_Agw = parameter.LinearExp(-18, -11)(amp_name)
            elif prior == 'log-uniform' and gamma_val is not None:
                if np.abs(gamma_val - 4.33) < 0.1:
                    log10_Agw = parameter.Uniform(-18, -14)(amp_name)
                else:
                    log10_Agw = parameter.Uniform(-18, -11)(amp_name)
            else:
                log10_Agw = parameter.Uniform(-18, -11)(amp_name)

        gam_name = '{}_gamma'.format(name)
        if gamma_val is not None:
            gamma_gw = parameter.Constant(gamma_val)(gam_name)
        else:
            gamma_gw = parameter.Uniform(0, 7)(gam_name)

        # common red noise PSD
        if psd == 'powerlaw':
            cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
        elif psd == 'broken_powerlaw':
            delta_name = '{}_delta'.format(name)
            kappa_name = '{}_kappa'.format(name)
            log10_fb_name = '{}_log10_fb'.format(name)
            kappa_gw = parameter.Uniform(0.01, 0.5)(kappa_name)
            log10_fb_gw = parameter.Uniform(-10, -7)(log10_fb_name)

            if delta_val is not None:
                delta_gw = parameter.Constant(delta_val)(delta_name)
            else:
                delta_gw = parameter.Uniform(0, 7)(delta_name)
            cpl = gp_priors.broken_powerlaw(log10_A=log10_Agw,
                                      gamma=gamma_gw,
                                      delta=delta_gw,
                                      log10_fb=log10_fb_gw,
                                      kappa=kappa_gw)
        elif psd == 'turnover':
            kappa_name = '{}_kappa'.format(name)
            lf0_name = '{}_log10_fbend'.format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lf0_gw = parameter.Uniform(-9, -7)(lf0_name)
            cpl = utils.turnover(log10_A=log10_Agw, gamma=gamma_gw,
                                 lf0=lf0_gw, kappa=kappa_gw)
        elif psd == 'turnover_knee':
            kappa_name = '{}_kappa'.format(name)
            lfb_name = '{}_log10_fbend'.format(name)
            delta_name = '{}_delta'.format(name)
            lfk_name = '{}_log10_fknee'.format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lfb_gw = parameter.Uniform(-9.3, -8)(lfb_name)
            delta_gw = parameter.Uniform(-2, 0)(delta_name)
            lfk_gw = parameter.Uniform(-8, -7)(lfk_name)
            cpl = gp_priors.turnover_knee(log10_A=log10_Agw, gamma=gamma_gw,
                                    lfb=lfb_gw, lfk=lfk_gw,
                                    kappa=kappa_gw, delta=delta_gw)

    if psd == 'spectrum':
        rho_name = '{}_log10_rho'.format(name)

        # checking if priors specified, otherwise give default values
        if logmin is None:
            logmin = -9
        if logmax is None:
            logmax = -4

        if prior == 'uniform':
            log10_rho_gw = parameter.LinearExp(logmin, logmax,
                                               size=components)(rho_name)
        elif prior == 'log-uniform':
            log10_rho_gw = parameter.Uniform(logmin, logmax, size=components)(rho_name)

        cpl = gp_priors.free_spectrum(log10_rho=log10_rho_gw)
    
    if psd == 'domian_wall':

        dw_b = parameter.Constant(0.5)('dw_b')
        dw_c = parameter.Constant(3)('dw_c')
        dw_lgalpha = parameter.Uniform(-3, -0.5)('dw_lgalpha')
        dw_lgTa = parameter.Uniform(-2.57, 10)('dw_lgTa')

        @signal_base.function
        def DW_psd(f, lg_Ta, lg_alpha, b, c):
            # https://arxiv.org/pdf/2306.16219 eq(6)

            H0 = 2.192711267238057e-18
            Ta = 10 ** lg_Ta
            alpha = 10 ** lg_alpha
            df = np.diff(np.concatenate((np.array([0]), f[::2])))

            return (H0**2.0/8.0/np.pi**4 * f**-5 * omega_DW(f, b, c, alpha, Ta) * np.repeat(df, 2))
        
        cpl = DW_psd(lg_Ta=dw_lgTa, lg_alpha=dw_lgalpha, b=dw_b, c=dw_c)


    if select == 'backend':
        # define selection by observing backend
        selection = selections.Selection(selections.by_backend)
    elif select == 'band' or select == 'band+':
        # define selection by observing band
        selection = selections.Selection(selections.by_band)
    elif isinstance(select, types.FunctionType):
        selection = selections.Selection(select)
    else:
        # define no selection
        selection = selections.Selection(selections.no_selection)
        
    if orf is None:
        crn = gp_signals.FourierBasisGP(cpl, coefficients=coefficients,
                                        components=components, Tspan=Tspan,
                                        name=name, pshift=pshift, pseed=pseed,
                                        selection=selection)
    elif orf in orfs.keys():
        if orf == 'crn':
            crn = gp_signals.FourierBasisGP(cpl, coefficients=coefficients,
                                            components=components, Tspan=Tspan,
                                            name=name, pshift=pshift, pseed=pseed,
                                            selection=selection)
        else:
            crn = gp_signals.FourierBasisCommonGP(cpl, orfs[orf],
                                                  components=components,
                                                  Tspan=Tspan,
                                                  name=name, pshift=pshift,
                                                  pseed=pseed)
    elif isinstance(orf, types.FunctionType):
        crn = gp_signals.FourierBasisCommonGP(cpl, orf,
                                              components=components,
                                              Tspan=Tspan,
                                              name=name, pshift=pshift,
                                              pseed=pseed)
    else:
        raise ValueError('ORF {} not recognized'.format(orf))

    return crn

def get_crn_model_dict(tspan, components=None):
    """
    function to return a dictionary "menu" of common red noise models set up via crn_block
    with appropriate number of commponents.
    """
    if components is None:
        fundamental_freq = 1.0/tspan
        max_freq = 1.0/240.0/86400.0 #1 / 240 days
        float_components = max_freq/fundamental_freq 
        components = int(float_components)

    crn_model_dict = {
        'domian_wall': common_red_noise_block(psd='domian_wall', components=30, orf='hd', name='dw'),
        #powerlaw, no correlation, free spectral index
        'pl_nocorr_freegam': crn_block(psd='powerlaw', prior='log-uniform', gamma_val=None,
                                       components=components, orf=None, name='gw_crn'),
        #free-spectrum no correlation
        'freespec_nocorr': crn_block(psd = 'spectrum', prior = "log-uniform", components = 40,
                                    orf = None, name = 'gw_freespec_nocorr'),
        #power law red noise, ORF sampled in angular bins
        'pl_orf_bins': crn_block(psd='powerlaw', prior='log-uniform', gamma_val=4.333,
                                 components=components, orf='bin_orf', name='gw_apl_orf_bins'),
        #power law red noise, spline-interpolated ORF bins
        'pl_orf_spline': crn_block(psd='powerlaw', prior='log-uniform', gamma_val=4.333,
                                   components=components, orf='spline_orf', name='gw_apl_orf_spline'),
        #power law red noise, free spectral index, Hellings-Downs correlations,
        'pl_hd_freegam': crn_block(psd='powerlaw', prior='log-uniform',
                                   components=30, orf='hd', name='gw_pl_hd_freegam', gamma_val = None),    
        #free spectrum red noise, Hellings-Downs correlations
        'freespec_hd': crn_block(psd = 'spectrum', prior = "log-uniform", components = 40,
                                 orf = 'hd', name = 'gw_freespec_hd')   
    }

    return crn_model_dict
