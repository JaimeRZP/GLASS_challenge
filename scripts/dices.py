import os
import numpy as np
import healpy as hp
import heracles
import skysegmentor
import heracles.dices as dices
from heracles.io import read, write
from itertools import combinations

from heracles.healpy import HealpixMapper
from heracles.fields import Positions, Shears, Visibility, Weights


# Config
nside = 1024
lmax = 1500
Njk = 10
mode = "lognormal"  # "lognormal" or "gaussian"
sims_path = f"../{mode}_sims/{mode}_sim_1/"
apply_mask = False
binned = False

save = True
output_path = f"{mode}_dices/"
if apply_mask:
    output_path = "../masked_"+output_path
else:
    output_path = "../"+output_path

# Fields
POS1 = heracles.read_maps(sims_path + "POS_1.fits")
SHE1 = heracles.read_maps(sims_path + "SHE_1.fits")
POS2 = heracles.read_maps(sims_path + "POS_2.fits")
SHE2 = heracles.read_maps(sims_path + "SHE_2.fits")

# Mask
if apply_mask:
    vmap = hp.read_map("../data/vmap_rotated.fits")
else:
    vmap = POS1[('POS', 1)] / POS1[('POS', 1)]

# Data & Vis maps
data_maps = {}
vis_maps = {}
data_maps[("POS", 1)] = POS1[("POS", 1)]*vmap
data_maps[("POS", 2)] = POS2[("POS", 2)]*vmap
data_maps[("SHE", 1)] = SHE1[("SHE", 1)]*vmap
data_maps[("SHE", 2)] = SHE2[("SHE", 2)]*vmap

vis_maps[("VIS", 1)] = vmap
vis_maps[("VIS", 2)] = vmap
vis_maps[("WHT", 1)] = vmap
vis_maps[("WHT", 2)] = vmap

# JK maps
jk_maps = {}
if apply_mask:
    data_fname = f"../data/masked_jkmap_{Njk}.fits"
else:
    data_fname = f"../data/jkmap_{Njk}.fits"

if os.path.exists(data_fname):
    jkmap = hp.read_map(data_fname)
else:
    jkmap = skysegmentor.segmentmapN(vmap, Njk)
    if save:
        hp.write_map(data_fname, jkmap)

for key in list(vis_maps.keys()):
    jk_maps[key] = jkmap

# Fields
mapper = HealpixMapper(nside=nside, lmax=lmax)
fields = {
    "POS": Positions(mapper, mask="VIS"),
    "SHE": Shears(mapper, mask="WHT"),
    "VIS": Visibility(mapper),
    "WHT": Weights(mapper),
}

# Cls0
data_fname = output_path + "cls/cls_data_0.fits"
mask_fname = output_path + "cls/cls_mask_0.fits"
if os.path.exists(data_fname) & os.path.exists(mask_fname):
    cls0 = read(data_fname)
    mls0 = read(mask_fname)
else:
    # For the data
    alms = heracles.transform(fields, data_maps)
    cls0 = heracles.angular_power_spectra(alms)
    # For the visibility
    alms = heracles.transform(fields, vis_maps)
    mls0 = heracles.angular_power_spectra(alms)
    if save:
        write(data_fname, cls0)
        write(mask_fname, mls0)

# Cls1
cls1 = {}
for regions in combinations(range(1, Njk + 1), 1):
    (jk1,) = regions
    data_fname = output_path + f"cls/cls_data_{jk1}.fits"
    if os.path.exists(data_fname):
        _cls = read(data_fname)
    else:
        _cls = dices.jackknife.get_cls(data_maps, jk_maps, fields, *regions)
        _cls_mm = dices.jackknife.get_cls(vis_maps, jk_maps, fields, *regions)
        # Mask correction
        alphas = dices.mask_correction(_cls_mm, mls0)
        _cls = heracles.unmixing._PolSpice(_cls, alphas)
        # Bias correction
        _cls = dices.correct_bias(_cls, jk_maps, fields, *regions)
        # Save spectra
        data_fname = output_path + f"cls/cls_data_{jk1}.fits"
        write(data_fname, _cls)
        data_fname = output_path + f"cls/cls_mask_{jk1}.fits"
        write(data_fname, _cls_mm)
    cls1[regions] = _cls

# Binning
nlbins = 15
ls = np.arange(lmax + 1)
ledges = np.logspace(np.log10(10), np.log10(lmax), nlbins + 1)
lgrid = (ledges[1:] + ledges[:-1]) / 2

if binned:
    ls = lgrid
    cls0 = heracles.binned(cls0, ledges)
    cls1 = heracles.binned(cls1, ledges)

# Delete1
cov1 = dices.jackknife_covariance(cls1)
data_fname = output_path + f"covs/jackknife_covariance_njk_{Njk}.fits"
heracles.write(data_fname, cov1)
# Shrink
target_cov = dices.gaussian_covariance(cls0)
shrinkage = dices.shrinkage_factor(cls1, target_cov)
shrunk_cov1 = dices.shrink(cov1, target_cov, shrinkage)
data_fname = output_path + f"covs/target_covariance.fits"
heracles.write(data_fname, target_cov)
data_fname = output_path + f"covs/shrunk_jackknife_covariance_njk_{Njk}.fits"
heracles.write(data_fname, shrunk_cov1)

# Delete2
cls2 = {}
for regions in combinations(range(1, Njk + 1), 2):
    (jk1, jk2) = regions
    data_fname = output_path + f"cls/cls_data_{jk1}_{jk2}.fits"
    if os.path.exists(data_fname):
        _cls = read(data_fname)
    else:
        # Compute Cls
        _cls = dices.jackknife.get_cls(data_maps, jk_maps, fields, *regions)
        _cls_mm = dices.jackknife.get_cls(vis_maps, jk_maps, fields, *regions)
        # Mask correction
        alphas = dices.mask_correction(_cls_mm, mls0)
        _cls = heracles.unmixing._PolSpice(_cls, alphas)
        # Bias correction
        _cls = dices.correct_bias(_cls, jk_maps, fields, *regions)
        # Save spectra
        data_fname = output_path + f"cls/cls_data_{jk1}_{jk2}.fits"
        write(data_fname, _cls)
        data_fname = output_path + f"cls/cls_mask_{jk1}_{jk2}.fits"
        write(data_fname, _cls_mm)
    cls2[regions] = _cls

# Debias
cov2 = dices.debias_covariance(
    cov1,
    cls0,
    cls1,
    cls2,
)
data_fname = output_path + f"covs/debiased_jackknife_covariance_njk_{Njk}.fits"
heracles.write(data_fname, cov2)

# DICES
dices_cov = dices.impose_correlation(
    cov2,
    shrunk_cov1,
)
data_fname = output_path + f"covs/DICES_covariance_njk_{Njk}.fits"
heracles.write(data_fname, cov2)
