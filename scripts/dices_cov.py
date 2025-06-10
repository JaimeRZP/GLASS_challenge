import os
import yaml
import numpy as np
import healpy as hp
import heracles
import skysegmentor
import heracles.dices as dices
from heracles.fields import Positions, Shears, Visibility, Weights
from heracles.io import read, write
from itertools import combinations

from heracles.healpy import HealpixMapper
from heracles.fields import Positions, Shears, Visibility, Weights


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

output_path = f"{mode}_dices/"
if apply_mask:
    output_path = "../masked_"+output_path
else:
    output_path = "../"+output_path

# Cls0
data_fname = output_path + "cls/cls_data_0.fits"
mask_fname = output_path + "cls/cls_mask_0.fits"
cls0 = read(data_fname)
mls0 = read(mask_fname)

# Cls1
cls1 = {}
for regions in combinations(range(1, Njk + 1), 1):
    (jk,) = regions
    data_fname = output_path + f"cls/cls_data_{jk}.fits"
    cls1[regions] = read(data_fname)

# Binning
nlbins = 10
ls = np.arange(lmax + 1)
ledges = np.logspace(np.log10(10), np.log10(lmax), nlbins + 1)
lgrid = (ledges[1:] + ledges[:-1]) / 2

if binned:
    ls = lgrid
    cls0 = heracles.binned(cls0, ledges)
    cls1 = heracles.binned(cls1, ledges)

# Delete1
data_fname = output_path + f"covs/jackknife_covariance_njk_{Njk}_binned_{binned}.fits"
if os.path.exists(data_fname):
    print(f"Jackknife covariance already exists at {data_fname}, skipping computation.")
    cov1 = read(data_fname)
else:
    print(f"Computing jackknife covariance, this may take a while...")
    cov1 = dices.jackknife_covariance(cls1)
    heracles.write(data_fname, cov1)
    print(f"Saved jackknife covariance to {data_fname}")

# Shrink
data_fname = output_path + f"covs/shrunk_jackknife_covariance_njk_{Njk}_binned_{binned}.fits"
if os.path.exists(data_fname):
    print(f"Shrunk jackknife covariance already exists at {data_fname}, skipping computation.")
    shrunk_cov1 = read(data_fname)
else:
    print(f"Computing shrunk jackknife covariance, this may take a while...")
    # Compute shrinkage factoray
    target_cov = dices.gaussian_covariance(cls0)
    if binned:
        shrinkage = 0.01 #dices.shrinkage_factor(cls1, target_cov)
    else:
        shrinkage = 0.2
    print(f"Shrinkage factor: {shrinkage}")
    shrunk_cov1 = dices.shrink(cov1, target_cov, shrinkage)
    # Save shrunk covariance
    heracles.write(data_fname, shrunk_cov1)
    print(f"Saved shrunk jackknife covariance to {data_fname}")
    data_fname = output_path + f"covs/target_covariance_binned_{binned}.fits"
    heracles.write(data_fname, target_cov)
    print(f"Saved target covariance to {data_fname}")

# Delete2
cls2 = {}
for regions in combinations(range(1, Njk + 1), 2):
    (jk1, jk2) = regions
    data_fname = output_path + f"cls/cls_data_{jk1}_{jk2}.fits"
    cls2[regions] = read(data_fname)

if binned:
    cls2 = heracles.binned(cls2, ledges)

# Debias
data_fname = output_path + f"covs/debiased_jackknife_covariance_njk_{Njk}_binned_{binned}.fits"
if os.path.exists(data_fname):
    print(f"Debiased jackknife covariance already exists at {data_fname}, skipping computation.")
    cov2 = read(data_fname)
else:
    Q = dices.jackknife.delete2_correction(cls0, cls1, cls2)
    cov2 = dices.jackknife._debias_covariance(
        cov1, Q)
    heracles.write(data_fname, cov2)
    data_fname = output_path + f"covs/debiased_jackknife_covariance_njk_{Njk}_binned_{binned}.fits"
    data_fname = output_path + f"covs/debias_correction_njk_{Njk}_binned_{binned}.fits"
    heracles.write(data_fname, Q)
    print(f"Saved debiased jackknife covariance to {data_fname}")

# DICES
data_fname = output_path + f"covs/DICES_covariance_njk_{Njk}_binned_{binned}.fits"
if os.path.exists(data_fname):
    print(f"DICES covariance already exists at {data_fname}, skipping computation.")
    dices_cov = read(data_fname)
else:
    dices_cov = dices.impose_correlation(
        cov2,
        shrunk_cov1,
    )
    heracles.write(data_fname, cov2)
    print(f"Saved DICES covariance to {data_fname}")
