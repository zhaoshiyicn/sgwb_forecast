import pickle
import glob
import json

import numpy as np

from pulsar import Pulsar, add_common_correlated_noise
from utils import *

with open("/home/zhaosy/CW/PTA_forecast/work/data/ppta_dr3/32psrs_DE438.pkl", "rb") as f:
    Epsrs = pickle.load(f)

noisefiles = sorted(glob.glob("/home/zhaosy/CW/PTA_forecast/work/data/ppta_dr3/posteriori/noises/noises_files/*"))

psrs = []

# creat fake PPTA pulsar, there we set the maxMJD is 63990, the total Tspan is about 30yrs.
for p, n in zip(Epsrs, noisefiles):
    psr = create_fakepsr_dr3(p, n, maxMJD=63990)
    print(psr.name)
    psrs.append(psr)

etspan, etmin, etmax = get_Tspan(Epsrs, to_MJD=True)
tspan, tmin, tmax = get_Tspan(psrs, to_MJD=True)

cutMJD = np.arange(etmax, tmax, 365)

newpsrs = []
for m in cutMJD:
    for _ in range(3): # 3 new pulsar produced every year
        psr = create_fakepsr(m, tmax)
        newpsrs.append(psr)
        print(psr.name)

psrs += newpsrs

# add CRN in total PTA
add_common_correlated_noise(psrs,  spectrum='powerlaw', orf='hd', log10_A=-14.68, gamma=13/3)
add_common_correlated_noise(psrs, spectrum='domian_wall', orf='hd', lg_Ta=-0.72, lg_alpha=-1.39)

for p in psrs:
    p.add_white_noise()
    p.add_red_noise()

# save the fakepulsar data
with open('/home/zhaosy/CW/PTA_forecast/work/data/fake_psrs/30yr_wn_rn_68psrs.pkl', 'wb') as f:
    pickle.dump(psrs, f)

# Slice the fakepta we generated, 
# first get slices of PPTA pulsars, and set their maximum toa to be the same as the real data.
new_dr3_psrs = cut_psr_dr3(psrs, 0, 32, Tmax=etmax)
with open('/home/zhaosy/CW/PTA_forecast/work/data/fake_psrs/18yr_wn_rn_gwb_32psrs.pkl', 'wb') as f:
    pickle.dump(new_dr3_psrs, f)

# The fakepulsar was sliced in groups every three years
result = []
for i, t in zip(np.arange(32, 69, 3)[1:], cutMJD*86400):
    with open('/home/zhaosy/CW/PTA_forecast/work/data/fake_psrs/30yr_wn_rn_gwb_68psrs.pkl', 'rb') as f:
        psrs = pickle.load(f)
    newpsrs = cut_psr(psrs, 32, i, t)
    result.append(newpsrs)

# By splicing the sliced PTAs together, 
# the number of PTAs we get should be the same as the length of the cutmjd, which is 12.
final_result = [new_dr3_psrs + r for r in result]

yrs=18
for i, fr in enumerate(final_result[:-1]):
    yrs += 1
    if yrs == 30:
        break
    with open(f'/home/zhaosy/CW/PTA_forecast/work/data/fake_psrs/{yrs}yr_wn_rn_gwb_{len(fr)}psrs.pkl', 'wb') as f:
        pickle.dump(fr, f)
