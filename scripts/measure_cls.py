import yaml
import numpy as np
import healpy as hp
import heracles
import heracles.dices as dices
from heracles.fields import Positions, Shears, Visibility, Weights
from heracles import transform
from heracles.healpy import HealpixMapper


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
path = f"../{mode}_sims/"
# Fields
mapper = HealpixMapper(nside=nside, lmax=lmax)
fields = {
    "POS": Positions(mapper, mask="VIS"),
    "SHE": Shears(mapper, mask="WHT"),
    "VIS": Visibility(mapper),
    "WHT": Weights(mapper),
}
mask_mapper = HealpixMapper(nside=2 * nside, lmax=2 * lmax)
mask_fields = {
    "POS": Positions(mapper, mask="VIS"),
    "SHE": Shears(mapper, mask="WHT"),
    "VIS": Visibility(mapper),
    "WHT": Weights(mapper),
}

vmap = hp.read_map("../data/vmap.fits")
r = hp.Rotator(coord=['G','E']) 
vmap = r.rotate_map_pixel(vmap)
vmap = np.abs(hp.ud_grade(vmap, nside))
vmap[vmap <= 1] = 0.0
vmap[vmap != 0] = vmap[vmap != 0] / vmap[vmap != 0]
vmap[vmap == 0] = 2.0
vmap[vmap == 1] = 0.0
vmap[vmap == 2] = 1.0

# mask cls
vmaps = {}
vmaps[("VIS", 1)] = vmap
vmaps[("VIS", 2)] = vmap
vmaps[("WHT", 1)] = vmap
vmaps[("WHT", 2)] = vmap
mask_alms = heracles.transform(mask_fields, vmaps)
mask_cls = heracles.angular_power_spectra(mask_alms)

for i in range(1, n+1):
    print(f"Loading sim {i}", end='\r')
    data_maps = {}
    sim_path = f"{path}/{mode}_sim_{i}"
    POS1 = heracles.read_maps(f"{sim_path}/POS_1.fits")
    SHE1 = heracles.read_maps(f"{sim_path}/SHE_1.fits")
    POS2 = heracles.read_maps(f"{sim_path}/POS_2.fits")
    SHE2 = heracles.read_maps(f"{sim_path}/SHE_2.fits")

    # Full sky
    data_maps[("POS", 1)] = POS1[('POS', 1)]
    data_maps[("POS", 2)] = POS2[('POS', 2)]
    data_maps[("SHE", 1)] = SHE1[('SHE', 1)]
    data_maps[("SHE", 2)] = SHE2[('SHE', 2)]

    alms = transform(fields, data_maps)
    cls = heracles.angular_power_spectra(alms)
    heracles.write(path+mode+f"_sim_{i}/measured_cls.fits", cls)

    # Masked
    data_maps[("POS", 1)] *= vmap
    data_maps[("POS", 2)] *= vmap
    data_maps[("SHE", 1)] *= vmap
    data_maps[("SHE", 2)] *= vmap

    alms = transform(fields, data_maps)
    cls_wmask = heracles.angular_power_spectra(alms)
    heracles.write(path+mode+f"_sim_{i}/measured_cls_wmask.fits", cls_wmask)
    heracles.write(path+mode+f"_sim_{i}/mask_cls.fits", mask_cls)
print("Done")
