import os
import yaml
import numpy as np
import heracles
import heracles.dices as dices

# Config
config_path = "./dices_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
n = config['nsims']
nside = config['nside']
lmax = config['lmax']
mode = config['mode']  # "lognormal" or "gaussian"
Njk = config['Njk']
apply_mask = config['apply_mask']
binned = config['binned']

nlbins = 10
ls = np.arange(lmax + 1)
ledges = np.logspace(np.log10(10), np.log10(lmax), nlbins + 1)
lgrid = (ledges[1:] + ledges[:-1]) / 2

ls = np.arange(lmax + 1)
path = f"../{mode}_sims/"
i_cls = {}
nu_cls = {}
pp_cls = {}
pm_cls = {}
for i in range(1, n+1):
    print(f"Loading sim {i}", end='\r')
    if os.path.exists(path+mode+f"_sim_{i}/measured_cls.fits"):
        i_cls[i] = heracles.read(path+mode+f"_sim_{i}/i_cls.fits")
        nu_cls[i] = heracles.read(path+mode+f"_sim_{i}/nu_cls.fits")
        pp_cls[i] = heracles.read(path+mode+f"_sim_{i}/pp_cls.fits")
        pm_cls[i] = heracles.read(path+mode+f"_sim_{i}/pm_cls.fits")

i_cqs = heracles.binned(i_cls, ledges)
nu_cqs = heracles.binned(nu_cls, ledges)
pp_cqs = heracles.binned(pp_cls, ledges)
pm_cqs = heracles.binned(pm_cls, ledges)
# Covariance
i_cls_cov = dices.jackknife_covariance(i_cls, nd=0)
nu_cls_cov = dices.jackknife_covariance(nu_cls, nd=0)
pp_cls_cov = dices.jackknife_covariance(pp_cls, nd=0)
pm_cls_cov = dices.jackknife_covariance(pm_cls, nd=0)
i_cqs_cov = dices.jackknife_covariance(i_cqs, nd=0)
nu_cqs_cov = dices.jackknife_covariance(nu_cqs, nd=0)
pp_cqs_cov = dices.jackknife_covariance(pp_cqs, nd=0)
pm_cqs_cov = dices.jackknife_covariance(pm_cqs, nd=0)
# Save
heracles.write(path+"cov_i_cls.fits", i_cls_cov)
heracles.write(path+"cov_nu_cls.fits", nu_cls_cov)
heracles.write(path+"cov_pp_cls.fits", pp_cls_cov)
heracles.write(path+"cov_pm_cls.fits", pm_cls_cov)
heracles.write(path+"cov_i_cqs.fits", i_cqs_cov)
heracles.write(path+"cov_nu_cqs.fits", nu_cqs_cov)
heracles.write(path+"cov_pp_cqs.fits", pp_cqs_cov)
heracles.write(path+"cov_pm_cqs.fits", pm_cqs_cov)
print("Done")
