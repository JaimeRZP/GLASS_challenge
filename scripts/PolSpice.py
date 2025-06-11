import os
import yaml
import numpy as np
import healpy as hp
import heracles
import heracles.dices as dices
from heracles.io import read, write


# Config
config_path = "./sims_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
n = config['nsims']
nside = config['nside']
lmax = config['lmax']
mode = config['mode']  # "lognormal" or "gaussian"
path = f"../{mode}_sims"

for i in range(1, n+1):
    folname = f"{mode}_sim_{i}"
    print(f"Unmixing sim {i} in {folname}", end='\r')
    # Load cls
    data_cls = heracles.read(f"{path}/{folname}/cls_data_wmask.fits")
    mask_cls = heracles.read(f"{path}/{folname}/cls_mask.fits")

    # PolSpice
    nu_cls = heracles.PolSpice(data_cls, mask_cls, mode='natural', patch_hole=True)
    pp_cls = heracles.PolSpice(data_cls, mask_cls, mode='plus', patch_hole=True)
    pm_cls = heracles.PolSpice(data_cls, mask_cls, mode='minus', patch_hole=True)

    # Save cls
    output_path = f"{path}/{folname}/"
    heracles.write(output_path + "cls_data_nu.fits", nu_cls)
    heracles.write(output_path + "cls_data_pp.fits", pp_cls)
    heracles.write(output_path + "cls_data_pm.fits", pm_cls)
