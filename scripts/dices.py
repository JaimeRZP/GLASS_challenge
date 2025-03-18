import os
import math
import numpy as np
import matplotlib.pyplot as plt
import yaml
import healpy as hp
import heracles
import skysegmentor
import heracles.dices as dices
from heracles.io import read, write

# Config
nside = 256
lmax = nside
mode = "gaussian"
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
Njk = 30
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

# Cls0
data_fname = output_path + "cls/cls_nojk.fits"
mask_fname = output_path + "cls/mls_nojk.fits"
if os.path.exists(data_fname) & os.path.exists(mask_fname):
    cls0 = read(data_fname)
    mls0 = read(mask_fname)
else:
    cls0 = dices.get_cls(data_maps, jk_maps)
    mls0 = dices.get_cls(vis_maps, jk_maps)
    if save:
        write(data_fname, cls0)
        write(mask_fname, mls0)

# Cls1
cls1 = {}
for jk in range(1, Njk + 1):
    data_fname = output_path + "cls/cls_njk_%i_jkid_%i.fits" % (
        Njk,
        jk,
    )
    if os.path.exists(data_fname):
        cls1[(jk)] = read(data_fname)
    else:
        # Compute Cls
        _cls = dices.get_cls(data_maps, jk_maps, jk=jk)
        _cls_mm = dices.get_cls(vis_maps, jk_maps, jk=jk)
        # Mask correction
        _cls = dices.correct_mask(_cls, _cls_mm, mls0)
        # Bias correction
        _cls = dices.correct_bias(_cls, jk_maps, jk=jk)
        cls1[jk] = _cls
        # Save Cls
        if save:
            write(data_fname, _cls)

# Delete1
nlbins = 10
ls = np.arange(lmax + 1)
ledges = np.logspace(np.log10(10), np.log10(lmax), nlbins + 1)
lgrid = (ledges[1:] + ledges[:-1]) / 2

if binned:
    ls = lgrid
    cls0 = heracles.binned(cls0, ledges)
    cls1 = heracles.binned(cls1, ledges)

data_fname = output_path + "covs/cov1_njk_%i_binned_%i.fits" % (
        Njk,
        binned,
    )
if os.path.exists(data_fname):
    print(f"Reading cov1 from {data_fname}")
    cov1 = read(data_fname)
else:
    cov1 = dices.get_delete1_cov(cls0, cls1)
    if save:
        write(data_fname, cov1)

# Shrinkage
data_fname = output_path + "covs/target_njk_%i_binned_%i.fits" % (
        Njk,
        binned,
    )
if os.path.exists(data_fname):
    print(f"Reading target_cov from {data_fname}")
    target_cov = read(data_fname)
else:
    target_cov = dices.get_gaussian_target(cls1)
    if save:
        write(data_fname, target_cov)

data_fname = output_path + "covs/shrunk_cov1_njk_%i_binned_%i.fits" % (
        Njk,
        binned,
    )
if os.path.exists(data_fname):
    print(f"Reading scov1 from {data_fname}")
    scov1 = read(data_fname)
else:
    W = dices.get_W(cls1)
    shrinkage = 0.2 #dices.get_shrinkage(cls0, target_cov, W)
    scov1 = dices.shrink_cov(cls0, cov1, target_cov, shrinkage)
    if save:
        write(data_fname, scov1)

# Delete2
cls2 = {}
for jk in range(1, Njk + 1):
    for jk2 in range(jk + 1, Njk + 1):
        data_fname = (
            output_path
            + "cls/cls_njk_%i_jkid2_%i_%i.fits" % (Njk, jk, jk2)
        )
        if os.path.exists(data_fname) :
            cls2[(jk, jk2)] = read(data_fname)
        else:
            # Compute Cls
            _cls = dices.get_cls(data_maps, jk_maps, jk=jk, jk2=jk2,)
            _cls_mm = dices.get_cls(vis_maps, jk_maps, jk=jk, jk2=jk2)
            # Mask correction
            _cls = dices.correct_mask(_cls, _cls_mm, mls0)
            # Bias correction
            _cls = dices.correct_bias(_cls, jk_maps, jk=jk, jk2=jk2)
            cls2[(jk, jk2)] = _cls
            # Save Cls
            if save:
                write(data_fname, _cls)

if binned:
    cls2 = heracles.binned(cls2, ledges)

data_fname = output_path + "covs/cov2_njk_%i_binned_%i.fits" % (
        Njk,
        binned,
    )
if os.path.exists(data_fname):
    print(f"Reading cov2 from {data_fname}")
    cov2 = read(data_fname)
else:
    cov2 = dices.get_delete2_cov(
        cov1,
        cls0,
        cls1,
        cls2,
    )
    if save:
        write(data_fname, cov2)

# DICES
fname = output_path + "covs/dices_njk_%i_binned_%i.fits" % (
    Njk,
    binned,
)
if os.path.exists(fname):
    dices_cov = read(fname)
else:
    dices_cov = dices.get_dices_cov(
        cls0,
        scov1,
        cov2
    )
    if save:
        write(fname, dices_cov)
