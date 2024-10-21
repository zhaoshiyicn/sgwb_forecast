import pickle
import glob
import json

import numpy as np
import matplotlib.pyplot as plt
import corner

from pulsar import Pulsar
gn_dict = {
           'J0437-4715': ['CASPSR_40CM', 'UWL_PDFB4_20CM', 'UWL_sbA', 'UWL_sbG'], 
           'J1017-7156': ['UWL_sbA', 'UWL_sbD'],
           'J1022+1001': ['UWL_sbE', 'UWL_sbH'],
           'J1713+0747': ['WBCORR_10CM', 'UWL_sbA', 'UWL_sbE', 'UWL_sbF'],
           'J1909-3744': ['CPSR2_50CM']
           }

def omega_DW(freq, b, c, alpha, Ta):
    # https://arxiv.org/abs/2307.03141 Eq(9)
    # set e to 0.7
    h1 = 0.674
    g_star = 10
    ee = 0.7
    a = 3
    # https://arxiv.org/abs/2307.03141 Eq(10) set g_star 10
    # unit of T is Gev
    fDW = lambda Ta: 10 ** (-9) * (g_star / 10) ** (1/6) * (100 * Ta)

    def S(x, b=b, c=c):
        # https://arxiv.org/abs/2307.03141 Eq(7)
        return ((a + b) / (b * x ** (-a / c) + a * x ** (b / c))) ** c
    
    return (10 ** (-10) * ee * (alpha / 0.01) ** 2 * 
            (10 / g_star) ** (1 / 3) * S(freq / fDW(Ta)) / h1 ** 2)

def cut_psr_dr3(psrs, cut_l, cut_r, Tmax):
    old_psrs = psrs[cut_l: cut_r]
    new_psrs = []
    # T = min([p.toas.min() for p in old_psrs]) + 365 * 86400 * nyrs
    # _, _, T = get_Tspan(psrs)
    # _, _, tmax = get_Tspan(old_psrs)
    for p in old_psrs:
        mask = np.searchsorted(p.toas, Tmax, side='right')
        p.toas = p.toas[:mask]
        p.toaerrs = p.toaerrs[:mask]
        p.residuals = p.residuals[:mask]
        p.Tspan = np.amax(p.toas) - np.amin(p.toas)
        p.flags['pta'] = ['FAKE'] * len(p.toas)
        p.backend_flags = p.backend_flags[:mask]
        p.backends = np.unique(p.backend_flags)
        p.Mmat = p.Mmat[:mask]
        new_psrs.append(p)
    
    return new_psrs

def get_Tspan(psrs, to_MJD=False, to_years=False):

    tmin = min([p.toas.min() for p in psrs])
    tmax = max([p.toas.max() for p in psrs])
    Tspan = tmax - tmin

    if to_MJD:
        return Tspan/86400, tmin/86400, tmax/86400
    elif to_years:
        return Tspan/86400/365.25, tmin/86400/365.25, tmax/86400/365.25
    else:
        return Tspan, tmin, tmax

def update_psr_30days(update_tspan, psr):

    days = 24 * 3600
    base_data = psr.toas.min()
    base_toas = np.arange(base_data / days, 63950, 30)
    base_toas += np.random.normal(0, 2, size=len(base_toas))
    base_toas *= days
    base_toaerrs = np.random.choice(psr.toaerrs, len(base_toas), replace=True)
    base_freqs = np.random.choice(psr.freqs, len(base_toas), replace=True)

    if update_tspan == 0:
        return base_toas, base_toaerrs, base_freqs

    Ts = days * 365.25 * update_tspan
    new_toas = np.array([])
    new_tspan = 0
    while new_tspan <= Ts:
        ob_span = np.random.normal(30*days, 2*days)
        if ob_span <= 0:
            ob_span = days
        new_tspan += ob_span
        new_toas = np.append(new_toas, new_tspan)
    toas = np.append(base_toas, new_toas+base_toas.max())

    # new_toaerr = np.power(10, np.random.uniform(-7., -5., size=len(new_toas)))
    new_toaerr = np.random.choice(psr.toaerrs, len(new_toas), replace=True)
    toaerr = np.append(base_toaerrs, new_toaerr)

    new_freqs = np.random.choice(psr.freqs, len(new_toas), replace=True)
    freqs = np.append(base_freqs, new_freqs)
    return toas, toaerr, freqs

def update_psr(update_tspan, psr):

    if update_tspan == 0:
        return psr.toas, psr.toaerrs, psr.freqs

    days = 24 * 3600
    Ts = days * 365.25 * update_tspan
    new_toas = np.array([])
    new_tspan = 0
    while new_tspan <= Ts:
        ob_span = np.random.normal(10*days, 1*days)
        if ob_span <= 0:
            ob_span = days
        new_tspan += ob_span
        new_toas = np.append(new_toas, new_tspan)
    toas = np.append(psr.toas, new_toas+psr.toas.max())

    # new_toaerr = np.power(10, np.random.uniform(-7., -5., size=len(new_toas)))
    new_toaerr = np.random.choice(psr.toaerrs, len(new_toas), replace=True)
    toaerr = np.append(psr.toaerrs, new_toaerr)

    new_freqs = np.random.choice(psr.freqs, len(new_toas), replace=True)
    freqs = np.append(psr.freqs, new_freqs)
    return toas, toaerr, freqs

def create_fakepsr_dr3(epsr, noisedict, uniform_toas=False, uniform_mintoas=True, rn_model=None, fakepsr=False, maxMJD=60000):

    """
    epsr: object of enterprise pulsar
    maxMJD: max MJD of fake pulsar (MJD)

    return: 
    """
    days = 86400
    if uniform_mintoas:
        minMJD = 53000
    else:
        minMJD = epsr.toas.min() / days

    toas = np.arange(minMJD, maxMJD, 30).astype(np.float64)
    if not uniform_toas:
        toas += np.random.normal(0, 2, len(toas))
    toas *= days
    toaerrs = np.random.choice(epsr.toaerrs, len(toas), replace=True)
    theta = epsr.theta
    phi = epsr.phi
    pdist = epsr.pdist

    Tspan = epsr.toas.max() - epsr.toas.min()
    if rn_model is None:
        rn_model = int(Tspan / 240 / days)
    custom_model = {'RN':rn_model, 'DM':None, 'Sv':None}

    if isinstance(noisedict, str):
        with open(noisedict,'r') as f:
            custom_noisedict = json.load(f)
    else:
        custom_noisedict = noisedict
    psr = Pulsar(toas, toaerrs, theta, phi, fakepsr=fakepsr, name=epsr.name, pdist=pdist, custom_model=custom_model, custom_noisedict=custom_noisedict, backends='bnuzh')

    return psr

def get_psrname(theta, phi):

    # RA
    h = int(24 * phi/(2 * np.pi))
    m = int((24 * phi / (2 * np.pi) - h) * 60)
    h = '0'+str(h) if len(str(h)) < 2 else str(h)
    m = '0'+str(m) if len(str(m)) < 2 else str(m)
    # DEC
    dec = round(180 * (np.pi/2 - theta) / np.pi, 2)
    sign = '+' if dec >= 0 else '-'
    decl, decr = str(abs(dec)).split('.')
    decl = '0'+str(decl) if len(str(decl)) < 2 else str(decl)
    decr = '0'+str(decr) if len(str(decr)) < 2 else str(decr)

    return 'J'+h+m+sign+decl+decr


def create_fakepsr(minMJD, maxMJD, rn_model=None, uniform_toas=False):

    days = 86400
    costheta = np.random.uniform(-1, 1)
    theta = np.arccos(costheta)
    phi = np.random.uniform(0, 2*np.pi)
    psrname = get_psrname(theta, phi)

    toas = np.arange(minMJD, maxMJD, 30).astype(np.float64)
    if not uniform_toas:
        toas += np.random.normal(0, 2, len(toas))
    toas *= days
    loc_err = np.random.uniform(-7.5, -6.5)
    toaerrs = 10 ** np.random.normal(loc_err, -0.01*loc_err, len(toas))

    dist = np.random.uniform(0.8, 1.2)
    pdist = (dist, 0.2*dist)

    Tspan = maxMJD - minMJD
    if rn_model is None:
        rn_model = int(Tspan / 240)

    custom_model = {'RN':rn_model, 'DM':None, 'Sv':None}

    custom_noisedict = {
                        f'{psrname}_efac': np.random.uniform(0.01, 2),
                        f'{psrname}_tnequad': np.random.uniform(-8., -5),
                        f'{psrname}_gamma_red': np.random.uniform(0.1, 7),
                        f'{psrname}__log10_A_red': np.random.uniform(-17, -11)
                        }
    
    psr = Pulsar(toas, toaerrs, theta, phi, fakepsr=True, name=psrname, pdist=pdist, custom_model=custom_model, custom_noisedict=custom_noisedict, backends='bnuzh')

    return psr

# def cut_psr(psrs, baseMJD, step, save=False, outdir=None):

#     days = 86400
#     yrs = 365.25

#     maxMJD = max([p.toas.max() for p in psrs]) / days
#     cutMJD = np.arange(baseMJD, maxMJD, yrs*step)
#     cutyr = int(cutMJD / yrs)

#     result = {k: [] for k in cutyr}
#     for p, m in zip(psrs, cutMJD):
#         custom_noisedict = custom_noisedict
#         cutpsr = create_fakepsr_dr3(p, custom_noisedict, maxMJD=m)
#         result[int(m/yrs)].append(cutpsr)

#     if save:
#         pass

#     return result

def cut_psr(psrs, cut_l, cut_r, tmin, nyrs=1):
    old_psrs = psrs[cut_l: cut_r]
    new_psrs = []
    # T = min([p.toas.min() for p in old_psrs]) + 365 * 86400 * nyrs
    T = tmin + 365 * 86400 * nyrs
    # _, _, tmax = get_Tspan(old_psrs)
    for p in old_psrs:
        mask = np.searchsorted(p.toas, T, side='right')
        p.toas = p.toas[:mask]
        p.toaerrs = p.toaerrs[:mask]
        p.residuals = p.residuals[:mask]
        p.Tspan = np.amax(p.toas) - np.amin(p.toas)
        p.flags['pta'] = ['FAKE'] * len(p.toas)
        p.backend_flags = p.backend_flags[:mask]
        p.backends = np.unique(p.backend_flags)
        p.Mmat = p.Mmat[:mask]
        new_psrs.append(p)
    
    return new_psrs

#########noise analysis#################
def get_3sigam(DIR_ml, DIR_3s, psrname):

    dir_js0015 = DIR_3s + f"/{psrname}_singlePsrNoise_sw_nesw0_p0015s.json"
    dir_js9985 = DIR_3s + f"/{psrname}_singlePsrNoise_sw_nesw0_p9985s.json"
    dir_jsml = DIR_ml + f"/{psrname}_singlePsrNoise_sw_nesw0_noise.json"

    data = {}
    for i, d in enumerate([dir_js0015, dir_js9985, dir_jsml]):
        with open(d, "r") as f:
            data[i] = json.load(f)

    js0015, js9985, jsml = data[0], data[1], data[2]

    keys = jsml.keys()
    result = dict()
    for k in keys:
        result[k] = (jsml[k], js0015[k], js9985[k])
    
    return result

def get_param(param_loc, param_sig, p0015, p9985, find_param=False):
    while not find_param:
        param = np.random.normal(param_loc, param_sig)
        if param >= p0015 and param <= p9985:
            find_param = True

    return param

def get_noises_params(psr, DIR_ml, DIR_3s, noise="red", nexp=1):
    psrname = psr.name
    form_dict = {
                "red": "red_noise", "dm": "dm_gp", "cho": "chrom_gp",
                "bn_h": "band_noise_high_high", "bn_m": "band_noise_mid_mid", "bn_l": "band_noise_low",
                "sw": "gp_sw", "hf": "hf_noise", "dmexp": "dmexp", "gn": "group_noise"
                }
    gn_dict = {
            'J0437-4715': ['CASPSR_40CM', 'UWL_PDFB4_20CM', 'UWL_sbA', 'UWL_sbG'], 
            'J1017-7156': ['UWL_sbA', 'UWL_sbD'],
            'J1022+1001': ['UWL_sbE', 'UWL_sbH'],
            'J1713+0747': ['WBCORR_10CM', 'UWL_sbA', 'UWL_sbE', 'UWL_sbF'],
            'J1909-3744': ['CPSR2_50CM']
            }
    noisedir = get_3sigam(DIR_ml, DIR_3s, psrname)

    if noise == "white":
        wn_params = {}
        backends = psr.backend_flags
        for b in backends:
            efac_loc, efac_0015, efac_9985 = noisedir[f'{psrname}_{b}_efac']
            sigma_efac = min(abs(efac_loc-efac_0015)/3, abs(efac_loc-efac_9985)/3)

            tnequad_loc, tnequad_0015, tnequad_9985 = noisedir[f'{psrname}_{b}_log10_tnequad']
            sigma_tnequad = min(abs(tnequad_loc-tnequad_0015)/3, abs(tnequad_loc-tnequad_9985)/3)

            efac = get_param(efac_loc, sigma_efac, efac_0015, efac_9985)
            tnequad = get_param(tnequad_loc, sigma_tnequad, tnequad_0015, tnequad_9985)
            wn_params[f'{psrname}_{b}_efac'] = efac
            wn_params[f'{psrname}_{b}_log10_tnequad'] = tnequad

        return wn_params


    elif noise == "dmexp":
        dmexp_params = []
        for i in range(nexp):
            idx_loc, idx_0015, idx_9985 = noisedir(f'{psrname}_dmexp_{i+1}_idx')
            log10_Amp_loc, log10_Amp_0015, log10_Amp_9985 = noisedir(f'{psrname}_dmexp_{i+1}_log10_Amp')
            log10_tau_loc, log10_tau_0015, log10_tau_9985 = noisedir(f'{psrname}_dmexp_{i+1}_log10_tau')
            t0_loc, t0_0015, t0_9985 = noisedir(f'{psrname}_dmexp_1_to')

            sigma_idx = min(abs(idx_loc-idx_0015)/3, abs(idx_loc-idx_9985)/3)
            sigma_Amp = min(abs(log10_Amp_loc-log10_Amp_0015)/3, abs(log10_Amp_loc-log10_Amp_9985)/3)
            sigma_tau = min(abs(log10_tau_loc-log10_tau_0015)/3, abs(log10_tau_loc-log10_tau_9985)/3)
            sigma_t0 = min(abs(t0_loc-t0_0015)/3, abs(t0_loc-t0_9985)/3)

            idx = get_param(idx_loc, sigma_idx, idx_0015, idx_9985)
            Amp = get_param(log10_Amp_loc, sigma_Amp, log10_Amp_0015, log10_A_9985)
            tau = get_param(log10_tau_loc, sigma_tau, log10_tau_0015, log10_tau_9985)
            t0 = get_param(t0_loc, sigma_t0, t0_0015, t0_9985)
            dmexp_params.append((idx, Amp, tau, t0))

        return dmexp_params

    elif noise == "gn":
        gn_params = {}
        assert psrname in gn_dict.keys(), f"{psrname} no have group noise!"
        backends = gn_dict[psrname]
        for b in backends:
            log10_A_loc, log10_A_0015, log10_A_9985 = noisedir[f'{psrname}_group_noise_{b}_log10_A']
            sigma_A = min(abs(log10_A_loc-log10_A_0015)/3, abs(log10_A_loc-log10_A_9985)/3)

            gamma_loc, gamma_0015, gamma_9985 = noisedir[f'{psrname}_group_noise_{b}_gamma']
            sigma_g = min(abs(gamma_loc-gamma_0015)/3, abs(gamma_loc-gamma_9985)/3)

            log10_A = get_param(log10_A_loc, sigma_A, log10_A_0015, log10_A_9985)
            gamma = get_param(gamma_loc, sigma_g, gamma_0015, gamma_9985)
            gn_params[f'{psrname}_group_noise_{b}_log10_A'] = log10_A
            gn_params[f'{psrname}_group_noise_{b}_gamma'] = gamma

        return gn_params

    else:
        try:
            log10_A_loc, log10_A_0015, log10_A_9985 = noisedir[f'{psrname}_{form_dict[noise]}_log10_A']
            sigma_A = min(abs(log10_A_loc-log10_A_0015)/3, abs(log10_A_loc-log10_A_9985)/3)

            gamma_loc, gamma_0015, gamma_9985 = noisedir[f'{psrname}_{form_dict[noise]}_gamma']
            sigma_g = min(abs(gamma_loc-gamma_0015)/3, abs(gamma_loc-gamma_9985)/3)
        except KeyError:
            print(f"KeyError: {psrname} not have {noise} noise!")
            return

        log10_A = get_param(log10_A_loc, sigma_A, log10_A_0015, log10_A_9985)
        gamma = get_param(gamma_loc, sigma_g, gamma_0015, gamma_9985)

        if noise == "sw":
            n_earth_loc, n_earth_0015, n_earth_9985 = noisedir["n_earth"]
            sigma_n = min(abs(n_earth_loc-n_earth_0015)/3, abs(n_earth_loc-n_earth_9985)/3)
            n_earth = get_param(n_earth_loc, sigma_n, n_earth_0015, n_earth_9985)
            return log10_A, gamma, n_earth

        else:
            return log10_A, gamma
        
def get_bestfit_value(chains, bins, burnin):

    chains = chains[burnin:].T
    ndim = len(chains) - 5
    result = {}
    for i in range(ndim):
        hist, bin_edges = np.histogram(chains[i], bins=bins)
        mid_idx = np.argmax(hist)
        mid = (bin_edges[mid_idx] + bin_edges[mid_idx + 1]) / 2
        result[i] = mid
    
    return result, ndim

def plot_noises(psrname, basedir, bins=25, burn=0.25):

    chains = np.loadtxt(basedir + f"/{psrname}_tnrn/chain_1.txt")
    burnin = int(burn * len(chains))
    # inds = [0, 5, 2, 4, 1, 3]
    inds = [0, 3, 1, 2]
    pars = np.loadtxt(basedir + f"/{psrname}_tnrn/pars.txt", dtype="str")[inds]
    bfv_dict, ndim = get_bestfit_value(chains, bins, burnin)

    fig = corner.corner(chains[burnin:, inds],
            # color=color,
            plot_density=True, plot_datapoints=True, fill_contours=False,
            labels=pars,
            # truths = truths,
            # range=range,
            bins=bins,
            label_kwargs={"fontsize": 10},
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
            smooth=1, smooth1d=1)

    axes = fig.get_axes()
    for i, axe in enumerate(axes):
        n = i / (ndim + 1)
        # print(n, i)
        if n % 1 == 0:
            # print(n)
            axe.axvline(x=bfv_dict[inds[int(n)]], color='black', linestyle='--', linewidth=0.9)
            axe.set_title(f"{bfv_dict[inds[int(n)]]:.2f}")

def save_noisefiles(psrname, basedir, outdir, bins=25, burn=0.25):

    chains = np.loadtxt(basedir + f"/{psrname}_tnrn/chain_1.txt")
    burnin = int(burn * len(chains))
    inds = [0, 3, 1, 2]
    # inds = [0, 3, 1, 2]
    pars = np.loadtxt(basedir + f"/{psrname}_tnrn/pars.txt", dtype="str")[inds]
    bfv_dict, _ = get_bestfit_value(chains, bins, burnin)
    noisefiles = {}
    for i, p in zip(inds, pars):
        noisefiles[p] = bfv_dict[i]
    
    with open(outdir + f'/noises_files/{psrname}_tnrn_maxlike_noises.json', 'w') as f:
        json.dump(noisefiles, f, indent=1)
