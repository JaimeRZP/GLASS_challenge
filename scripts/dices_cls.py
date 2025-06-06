import os
import yaml
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
config_path = "../dices_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
n = config['n_sims']
nside = config['nside']
lmax = config['lmax']
mode = config['mode']  # "lognormal" or "gaussian"
Njk = config['njk']
apply_mask = config['apply_mask']
binned = config['binned']

sims_path = f"../{mode}_sims/{mode}_sim_1/"
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
    vmap = hp.read_map("../data/vmap.fits")
    r = hp.Rotator(coord=['G','E'])
    vmap = r.rotate_map_pixel(vmap)
    vmap = np.abs(hp.ud_grade(vmap, nside))
    vmap[vmap <= 1] = 0.0
    vmap[vmap != 0] = vmap[vmap != 0] / vmap[vmap != 0]
    vmap[vmap == 0] = 2.0
    vmap[vmap == 1] = 0.0
    vmap[vmap == 2] = 1.0
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
    hp.write_map(data_fname, jkmap)

jkmap = np.abs(hp.ud_grade(jkmap, nside))
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
print("Cls1 done")

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
print("Cls2 done")