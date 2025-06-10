import os
import yaml
import numpy as np
import healpy as hp
import heracles
import heracles.dices as dices
from heracles.io import read, write


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
path = f"../{mode}_sims"

n = 100
for i in range(1, n+1):
    folname = f"{mode}_sim_{i}"
    print(f"Unmixing sim {i} in {folname}", end='\r')
    # Load cls
    data_cls = heracles.read(f"{path}/{folname}/measured_cls_wmask.fits")
    mask_cls = heracles.read(f"{path}/{folname}/mask_cls.fits")

    # PolSpice
    nu_cls = heracles.PolSpice(data_cls, mask_cls, mode='natural', patch_hole=True)
    pp_cls = heracles.PolSpice(data_cls, mask_cls, mode='plus', patch_hole=True)
    pm_cls = heracles.PolSpice(data_cls, mask_cls, mode='minus', patch_hole=True)

    # Save cls
    output_path = f"{path}/{folname}/"
    heracles.write(output_path + "nu_cls.fits", nu_cls)
    heracles.write(output_path + "pp_cls.fits", pp_cls)
    heracles.write(output_path + "pm_cls.fits", pm_cls)
