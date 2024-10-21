import numpy as np
import matplotlib.pyplot as plt
import pickle, json
from scipy.optimize import fsolve
from enterprise_extensions import deterministic as det
from enterprise.signals import utils
import scipy.constants as sc
import importlib, inspect
import healpy as hp

# This module was adapted from fakepta: https://github.com/mfalxa/fakepta/tree/main
module = importlib.import_module('spectrum')
spec = inspect.getmembers(module, inspect.isfunction)
spec_params = {}
for s_name, s_obj in spec:
    pnames = [*inspect.signature(s_obj).parameters]
    pnames.remove('f')
    spec_params[s_name] = pnames
spec = dict(spec)

class Pulsar:
    
    def __init__(self, toas, toaerr, theta, phi, residuals=None, fakepsr=False, name=None, pdist=(1., 0.2), freqs=[1400], custom_noisedict=None, custom_model=None, tm_params=None, backends='backend', ephem=None):
        
        self.is_fakepsr = fakepsr
        self.nepochs = len(toas)
        # self.toas = np.repeat(toas, len(backends))
        self.toas = toas
        self.toaerrs = toaerr * np.ones(len(self.toas))
        if residuals is None:
            self.residuals = np.zeros(len(self.toas))
        else: self.residuals = residuals
        self.Tspan = np.amax(self.toas) - np.amin(self.toas)
        if custom_model is None:
            self.custom_model = {'RN':30, 'DM':100, 'Sv':None}
        else:
            self.custom_model = custom_model
        self.signal_model = {}
        self.flags = {}
        self.flags['pta'] = ['FAKE'] * len(self.toas)
        if isinstance(backends, str):
            self.backend_flags = np.tile([backends], self.nepochs)
        elif isinstance(backends, (list, np.ndarray)) and len(backends) == len(toas):
            self.backend_flags = backends
        else: raise ValueError('Check the value of backends!')
        self.backends = np.unique(self.backend_flags)
        self.freqs = np.array(freqs)
        self.theta = theta
        self.phi = phi
        self.pos = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
        if ephem is not None:
            self.ephem = ephem
            self.planetssb = ephem.get_planet_ssb(self.toas)
            self.pos_t = np.tile(self.pos, (len(self.toas), 1))
        else:
            self.planetssb = None
            self.pos_t = None
        self.pdist = pdist
        if name is None:
            self.name = self.get_psrname()
        else:
            self.name = name
        self.init_tm_pars(tm_params)
        self.make_Mmat()
        self.fitpars = [*self.tm_pars]
        self.init_noisedict(custom_noisedict)


    def get_psrname(self):

        # RA
        h = int(24*self.phi/(2*np.pi))
        m = int((24*self.phi/(2*np.pi) - h) * 60)
        h = '0'+str(h) if len(str(h)) < 2 else str(h)
        m = '0'+str(m) if len(str(m)) < 2 else str(m)
        # DEC
        dec = round(180 * (np.pi/2 - self.theta) / np.pi, 2)
        sign = '+' if dec >= 0 else '-'
        decl, decr = str(abs(dec)).split('.')
        decl = '0'+str(decl) if len(str(decl)) < 2 else str(decl)
        decr = '0'+str(decr) if len(str(decr)) < 2 else str(decr)

        return 'J'+h+m+sign+decl+decr

    def init_noisedict(self, custom_noisedict=None):

        if custom_noisedict is None:
            custom_noisedict = {}
            noisedict = {}
            for backend in self.backends:
                noisedict[self.name+'_'+backend+'_efac'] = 1.
                noisedict[self.name+'_'+backend+'_log10_tnequad'] = -8.
                noisedict[self.name+'_'+backend+'_log10_t2equad'] = -8.
                noisedict[self.name+'_'+backend+'_log10_ecorr'] = -8.
            self.noisedict = noisedict
        
        else:
            # print("load custom noisedict..")
            keys = [*custom_noisedict]
            noisedict = {}
            for key in keys:
                if self.name in key:
                    # Note: for backend in backends
                    backend = 'bnuzh'
                    if 'efac' in key:
                        noisedict[self.name+'_'+backend+'_efac'] = custom_noisedict[key]
                    elif 'tnequad' in key:
                        noisedict[self.name+'_'+backend+'_log10_tnequad'] = custom_noisedict[key]
            self.noisedict = noisedict

        for key in [*custom_noisedict]:
            if 'red' in key:
                if 'log10_A' in key:
                    noisedict[self.name+'_red_noise_log10_A'] = custom_noisedict[key]
                if 'gamma' in key:
                    noisedict[self.name+'_red_noise_gamma'] = custom_noisedict[key]
            if 'dm' in key:
                if 'log10_A' in key:
                    noisedict[self.name+'_dm_gp_log10_A'] = custom_noisedict[key]
                if 'gamma' in key:
                    noisedict[self.name+'_dm_gp_gamma'] = custom_noisedict[key]
        # if np.any(['red' in key for key in [*custom_noisedict]]):
        #     if 'log10_A' in key:
        #         print(f"here! {key}")
        #         noisedict[self.name+'_red_noise_log10_A'] = custom_noisedict[key]
        #     if 'gamma' in key:
        #         print(f"here! {key}")
        #         noisedict[self.name+'_red_noise_gamma'] = custom_noisedict[key]
        
        # if np.any(['dm' in key for key in [*custom_noisedict]]):
        #     if 'log10_A' in key:
        #         noisedict[self.name+'_dm_gp_log10_A'] = custom_noisedict[key]
        #     if 'gamma' in key:
        #         noisedict[self.name+'_dm_gp_gamma'] = custom_noisedict[key]

    def init_tm_pars(self, timing_model):

        self.tm_pars = {}
        self.tm_pars['F0'] = (200, 1e-13)
        self.tm_pars['F1'] = (0., 1e-20)
        self.tm_pars['DM'] = (0., 5e-4)
        self.tm_pars['DM1'] = (0., 1e-4)
        self.tm_pars['DM2'] = (0., 1e-5)
        self.tm_pars['ELONG'] = (0., 1e-5)
        self.tm_pars['ELAT'] = (0., 1e-5)
        if timing_model is not None:
            self.tm_pars.update(timing_model)

    def make_Mmat(self, t0=0.):

        npar = len([*self.tm_pars]) + 1
        self.Mmat = np.zeros((len(self.toas), npar))
        self.Mmat[:, 0] = np.ones(len(self.toas))
        self.Mmat[:, 1] = -(self.toas - t0) / self.tm_pars['F0'][0]
        self.Mmat[:, 2] = -0.5 * (self.toas - t0)**2 / self.tm_pars['F0'][0]
        self.Mmat[:, 3] = 1 / self.freqs**2
        self.Mmat[:, 4] = (self.toas - t0) / self.freqs**2 / self.tm_pars['F0'][0]
        self.Mmat[:, 5] = 0.5 * (self.toas - t0)**2 / self.freqs**2 / self.tm_pars['F0'][0]
        self.Mmat[:, 6] = np.cos(2*np.pi/sc.Julian_year * (self.toas - t0))
        self.Mmat[:, 7] = np.sin(2*np.pi/sc.Julian_year * (self.toas - t0))
    
    def make_ideal(self):

        # set residuals to zero and clean signal model dict

        self.residuals = np.zeros(len(self.toas))
        for signal in [*self.signal_model]:
            self.signal_model.pop(signal)
            for key in [*self.noisedict]:
                if signal in key:
                    self.noisedict.pop(key)

    def add_white_noise(self, add_ecorr=False, randomize=False):

        if randomize:
            for key in [*self.noisedict]:
                if 'efac' in key:
                    self.noisedict[key] = np.random.uniform(0.5, 2.5)
                if 'equad' in key:
                    self.noisedict[key] = np.random.uniform(-8., -5.)
                if add_ecorr and 'ecorr' in key:
                    self.noisedict[key] = np.random.uniform(-10., -7.)
        if self.backends is None:
            toaerrs2 = self.noisedict[self.name+'_efac']**2 * self.toaerrs**2 + 10**(2*self.noisedict[self.name+'_log10_tnequad'])
        else:
            toaerrs2 = np.zeros(len(self.toaerrs))
            # print('add wn..')
            for backend in self.backends:
                mask_backend = self.backend_flags == backend
                toaerrs2[mask_backend] = self.noisedict[self.name+'_'+backend+'_efac']**2 * self.toaerrs[mask_backend]**2 + 10**(2*self.noisedict[self.name+'_'+backend+'_log10_tnequad'])
        
        if add_ecorr:
            for backend in self.backends:
                quant_idx = self.quantise_ecorr(backends=[backend])
                for q_i in quant_idx:
                    if len(q_i) < 2:
                        self.residuals[q_i] += np.random.normal(scale=toaerrs2[q_i]**0.5)
                    else:
                        white_block = np.ones((len(q_i), len(q_i))) * 10**self.noisedict[self.name+'_'+backend+'_log10_ecorr']
                        white_block = np.fill_diagonal(white_block, np.diag(white_block) + toaerrs2[q_i])
                        self.residuals[q_i] += np.random.multivariate_normal(mean=np.zeros(len(q_i)), cov=white_block)
        else:
            self.residuals += np.random.normal(scale=toaerrs2**0.5)
            # print(self.residuals)

    def add_red_noise(self, spectrum='powerlaw', f_psd=None, **kwargs):

        rn_components = self.custom_model['RN']
        if rn_components is not None:

            if f_psd is None:
                f_psd = np.arange(1, rn_components+1) / self.Tspan

            if 'red_noise' in self.signal_model:
                self.residuals -= self.reconstruct_signal(['red_noise'])

            if spectrum == 'custom':
                psd = kwargs['custom_psd']
            elif spectrum in [*spec]:
                if len(kwargs) == 0:
                    try:
                        kwargs = {pname : self.noisedict[self.name+'_red_noise_'+pname] for pname in spec_params[spectrum]}
                    except:
                        print('PSD parameters must be in noisedict or parsed as input.')
                        return
                psd = spec[spectrum](f_psd, **kwargs)
                self.update_noisedict(self.name+'_red_noise', kwargs)

                self.add_time_correlated_noise(signal='red_noise', spectrum=spectrum, idx=0., psd=psd, f_psd=f_psd)

    # def add_dw_signal(self, b, c, lgTa, lgalpha, components=30):

    #     design = utils.createfourierdesignmatrix_red

    #     toas = self.toas
    #     F, Freqs = design(toas, nmodes=components)
    #     phi = DW_psd(Freqs, lgTa, lgalpha, b, c)
    #     w = np.random.randn(len(Freqs))
    #     dw = np.dot(F, np.sqrt(phi)*w)
    #     self.residuals += dw

    def update_noisedict(self, prefix, dict_vals):

        params = {}
        for key in [*dict_vals]:
            params[prefix+'_'+key] = dict_vals[key]
        self.noisedict.update(params)
    
    def add_dm_noise(self, spectrum='powerlaw', f_psd=None, **kwargs):

        dm_components = self.custom_model['DM']
        if dm_components is not None:
            
            if f_psd is None:
                f_psd = np.arange(1, dm_components+1) / self.Tspan

            if 'dm_gp' in self.signal_model:
                self.residuals -= self.reconstruct_signal(['dm_gp'])

            if spectrum == 'custom':
                psd = kwargs['custom_psd']
            elif spectrum in [*spec]:
                if len(kwargs) == 0:
                    try:
                        kwargs = {pname : self.noisedict[self.name+'_dm_gp_'+pname] for pname in spec_params[spectrum]}
                    except:
                        print('PSD parameters must be in noisedict or parsed as input.')
                        return
                psd = spec[spectrum](f_psd, **kwargs)
                self.update_noisedict(self.name+'_dm_gp', kwargs)

            self.add_time_correlated_noise(signal='dm_gp', spectrum=spectrum, idx=2., psd=psd, f_psd=f_psd)

    def add_time_correlated_noise(self, signal='', spectrum='powerlaw', psd=None, f_psd=None, idx=0, freqf=1400, backend=None):

        # generate time correlated noise with given PSD and chromatic index

        if backend is not None:
            signal = backend + '_' + signal
            mask = self.backend_flags == backend
            if not np.any(mask):
                print(backend, 'not found in backend_flags.')
                return
        else:
            mask = np.ones(len(self.toas), dtype='bool')

        df = np.diff(np.append(0., f_psd))
        assert len(psd) == len(f_psd), '"psd" and "f_psd" must be same length. The frequencies "f_psd" correspond to the frequencies where the "psd" is evaluated.'
        psd = np.repeat(psd, 2)

        coeffs = np.random.normal(loc=0., scale=np.sqrt(psd))

        # save noise properties in signal model 
        self.signal_model[signal] = {}
        self.signal_model[signal]['spectrum'] = spectrum
        self.signal_model[signal]['f'] = f_psd
        self.signal_model[signal]['psd'] = psd[::2]
        self.signal_model[signal]['fourier'] = np.vstack((coeffs[::2] / df**0.5, coeffs[1::2] / df**0.5))
        self.signal_model[signal]['nbin'] = len(f_psd)
        self.signal_model[signal]['idx'] = idx
        
        for i in range(len(f_psd)):
            self.residuals[mask] += (freqf/self.freqs)**idx * df[i]**0.5 * coeffs[2*i] * np.cos(2*np.pi*f_psd[i]*self.toas[mask])
            self.residuals[mask] += (freqf/self.freqs)**idx * df[i]**0.5 * coeffs[2*i+1] * np.sin(2*np.pi*f_psd[i]*self.toas[mask])

    def make_time_correlated_noise_cov(self, signal='', freqf=1400):

        # returns covariance matrix of time correlated noise with given PSD and chromatic index

        if 'system_noise' in signal:
            backend = signal.split('system_noise_')[1]
        else:
            backend = None

        if backend is not None:
            signal = backend + '_' + signal
            mask = self.backend_flags == backend
            if not np.any(mask):
                print(backend, 'not found in backend_flags.')
                return
        else:
            mask = np.ones(len(self.toas), dtype='bool')

        # save noise properties in signal model
        f = self.signal_model[signal]['f']
        psd = self.signal_model[signal]['psd']
        components = self.signal_model[signal]['nbin']
        idx = self.signal_model[signal]['idx']

        df = np.diff(np.append(0, f))
        psd = np.repeat(psd * df, 2)
        basis = np.zeros((len(self.toas[mask]), 2*components))
        for i in range(components):
            basis[:, 2*i] = (freqf/self.freqs)**idx * np.cos(2*np.pi*f[i]*self.toas[mask])
            basis[:, 2*i+1] = (freqf/self.freqs)**idx * np.sin(2*np.pi*f[i]*self.toas[mask])
        cov = np.dot(basis, np.dot(np.diag(psd), basis.T))
        return cov
    
    def make_noise_covariance_matrix(self):

        # make total noise covariance matrix

        if self.backends is None:
            toaerrs = np.sqrt(self.noisedict[self.name+'_efac']**2 * self.toaerrs**2 + 10**(2*self.noisedict[self.name+'_log10_tnequad']))
        else:
            toaerrs = np.zeros(len(self.toas))
            for backend in self.backends:
                mask_backend = self.backend_flags == backend
                toaerrs[mask_backend] = np.sqrt(self.noisedict[self.name+'_'+backend+'_efac']**2 * self.toaerrs[mask_backend]**2 + 10**(2*self.noisedict[self.name+'_'+backend+'_log10_tnequad']))
        white_cov = toaerrs**2

        red_cov = np.zeros((len(self.toas), len(self.toas)))
        if self.custom_model['RN'] is not None:
            red_cov += self.make_time_correlated_noise_cov(signal='red_noise')
        if self.custom_model['DM'] is not None:
            red_cov += self.make_time_correlated_noise_cov(signal='dm_gp')
        if self.custom_model['Sv'] is not None:
            red_cov += self.make_time_correlated_noise_cov(signal='chrom_gp')
        return white_cov, red_cov
    
    def reconstruct_signal(self, signals=None, freqf=1400):

        # reconstruct time domain realisation of injected noises and signals

        if signals is None:
            signals = [*self.signal_model]
        sig = np.zeros(len(self.toas))
        for signal in signals:
            if signal == 'cgw':
                for ncgw in len(self.signal_model['cgw']):
                    sig += det.cw_delay(self.toas, self.pos, self.pdist,
                                        **self.signal_model['cgw'][str(ncgw)])
            if (signal in ['red_noise', 'dm_gp', 'chrom_gp']) or ('common' in signal):
                f = self.signal_model[signal]['f']
                idx = self.signal_model[signal]['idx']
                df = np.diff(np.append(0., f))
                c = self.signal_model[signal]['fourier']
                for c_k, f_k, df_k in zip(c.T, f, df):
                    sig += df_k * c_k[0] * (freqf/self.freqs)**idx * np.cos(2*np.pi*f_k * self.toas)
                    sig += df_k * c_k[1] * (freqf/self.freqs)**idx * np.sin(2*np.pi*f_k * self.toas)
            if 'system_noise' in signal:
                backend = signal.split('system_noise_')[1]
                mask = self.backend_flags == backend
                f = self.signal_model[signal]['f']
                df = np.diff(np.append(0., f))
                c = self.signal_model[signal]['fourier']
                for c_k, f_k, df_k in zip(c.T, f, df):
                    sig[mask] += df_k * c_k[0] * np.cos(2*np.pi*f_k * self.toas[mask])
                    sig[mask] += df_k * c_k[1] * np.sin(2*np.pi*f_k * self.toas[mask])
        return sig

def create_gw_antenna_pattern(pos, gwtheta, gwphi):

    m = np.array([np.sin(gwphi), -np.cos(gwphi), np.zeros(len(gwphi))]).T
    n = np.array([-np.cos(gwtheta) * np.cos(gwphi), -np.cos(gwtheta) * np.sin(gwphi), np.sin(gwtheta)]).T
    omhat = np.array([-np.sin(gwtheta) * np.cos(gwphi), -np.sin(gwtheta) * np.sin(gwphi), -np.cos(gwtheta)]).T

    fplus = 0.5 * (np.dot(m, pos) ** 2 - np.dot(n, pos) ** 2) / (1 + np.dot(omhat, pos))
    fcross = (np.dot(m, pos) * np.dot(n, pos)) / (1 + np.dot(omhat, pos))
    cosMu = -np.dot(omhat, pos)

    return fplus, fcross, cosMu

def hd(psrs):
    orfs = np.zeros((len(psrs), len(psrs)))
    for i in range(len(psrs)):
        for j in range(len(psrs)):
            if i == j:
                orfs[i, j] = 1.
            else:
                omc2 = (1 - np.dot(psrs[i].pos, psrs[j].pos)) / 2
                orfs[i, j] =  1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5
    return orfs

def monopole(psrs):
    npsr = len(psrs)
    return np.ones((npsr, npsr))

def dipole(psrs):
    orfs = np.zeros((len(psrs), len(psrs)))
    for i in range(len(psrs)):
        for j in range(len(psrs)):
            if i == j:
                orfs[i, j] = 1.
            else:
                omc2 = np.dot(psrs[i].pos, psrs[j].pos)
                orfs[i, j] = omc2
    return orfs

def curn(psrs):
    npsr = len(psrs)
    return np.eye(npsr)


def anisotropic(psrs, h_map):

    orfs = np.zeros((len(psrs), len(psrs)))
    npixels = len(h_map)
    pixels = hp.pix2ang(hp.npix2nside(npixels), np.arange(npixels), nest=False)
    gwtheta = pixels[0]
    gwphi = pixels[1]
    for i in range(len(psrs)):
        for j in range(len(psrs)):
            if i == j:
                k_ab = 2.
            else:
                k_ab = 1.
            fp_a, fc_a, _ = create_gw_antenna_pattern(psrs[i].pos, gwtheta, gwphi)
            fp_b, fc_b, _ = create_gw_antenna_pattern(psrs[j].pos, gwtheta, gwphi)
            orfs[i, j] = 1.5 * k_ab * np.sum((fp_a*fp_b + fc_a*fc_b) * h_map) / npixels
    return orfs

def add_common_correlated_noise(psrs, orf='hd', spectrum='powerlaw', name='gw', idx=0, components=30, freqf=1400, custom_psd=None, f_psd=None, h_map=None, **kwargs):

    if name is not None:
        signal_name = name + '_common'
    else:
        signal_name = 'common'

    Tspan = np.amax([psr.toas.max() for psr in psrs]) - np.amin([psr.toas.min() for psr in psrs])
    if f_psd is None:
        f_psd = np.arange(1, components+1) / Tspan
    df = np.diff(np.append(0., f_psd))
    if spectrum == 'custom':
        # assert f_psd is None, '"f_psd" must not be None. The frequencies "f_psd" correspond to frequencies where the "custom_psd" is evaluated.'
        assert len(custom_psd) == len(f_psd), '"custom_psd" and "f_psd" must be same length. The frequencies "f_psd" correspond to frequencies where the "custom_psd" is evaluated.'
        psd_gwb = custom_psd
    elif spectrum in [*spec]:
        psd_gwb = spec[spectrum](f_psd, **kwargs)
        for psr in psrs:
            psr.update_noisedict(signal_name, kwargs)

    # save noise properties in signal model
    for psr in psrs:
        if signal_name in [*psr.signal_model]:
            psr.residuals -= psr.reconstruct_signal(signals=[signal_name])

        psr.signal_model[signal_name] = {}
        psr.signal_model[signal_name]['orf'] = orf
        psr.signal_model[signal_name]['spectrum'] = spectrum
        psr.signal_model[signal_name]['hmap'] = h_map
        psr.signal_model[signal_name]['f'] = f_psd
        psr.signal_model[signal_name]['psd'] = psd_gwb
        psr.signal_model[signal_name]['fourier'] = np.vstack((np.zeros(components), np.zeros(components)))
        psr.signal_model[signal_name]['nbin'] = components
        psr.signal_model[signal_name]['idx'] = idx
    
    psd_gwb = np.repeat(psd_gwb, 2)
    coeffs = np.sqrt(psd_gwb)
    orf_funcs = {'hd':hd, 'monopole':monopole, 'dipole':dipole, 'curn':curn}
    if orf in [*orf_funcs]:
        orfs = orf_funcs[orf](psrs)
    elif orf == 'anisotropic':
        orfs = anisotropic(psrs, h_map)
    for i in range(components):
        orf_corr_sin = np.random.multivariate_normal(mean=np.zeros(len(psrs)), cov=orfs)
        orf_corr_cos = np.random.multivariate_normal(mean=np.zeros(len(psrs)), cov=orfs)
        for n, psr in enumerate(psrs):
            psr.signal_model[signal_name]['fourier'][0, i] = orf_corr_cos[n] * coeffs[2*i] / df[i]**0.5
            psr.signal_model[signal_name]['fourier'][1, i] = orf_corr_sin[n] * coeffs[2*i+1] / df[i]**0.5
            psr.residuals += orf_corr_cos[n] * (freqf/psr.freqs)**idx * df[i]**0.5 * coeffs[2*i] * np.cos(2*np.pi*f_psd[i]*psr.toas)
            psr.residuals += orf_corr_sin[n] * (freqf/psr.freqs)**idx * df[i]**0.5 * coeffs[2*i+1] * np.sin(2*np.pi*f_psd[i]*psr.toas)
