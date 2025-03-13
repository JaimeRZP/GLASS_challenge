import yaml
import numpy as np
import healpy as hp
import heracles
import heracles.dices as dices
from heracles.fields import Positions, Shears
from heracles import transform
from heracles.healpy import HealpixMapper

mode = "lognormal"
nside = 256
lmax = 256
mapper = HealpixMapper(nside=nside, lmax=lmax)
path = f"../{mode}_sims/"
# Fields
Nbins = 2
fields = {
    "POS": Positions(mapper, mask="VIS"),
    "SHE": Shears(mapper, mask="WHT"),
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

nlbins = 10
ls = np.arange(lmax + 1)
ledges = np.logspace(np.log10(10), np.log10(lmax), nlbins + 1)
lgrid = (ledges[1:] + ledges[:-1]) / 2

clss = {}
cqss = {}
clss_wmask = {}
cqss_wmask = {}
for i in range(1, 100+1):
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
    cqs = heracles.binned(cls, ledges)
    _cls = dices.compsep_Cls(cls)
    _cqs = dices.compsep_Cls(cqs)

    heracles.write(path+mode+f"_sim_{i}/measured_cls.fits", _cls)
    heracles.write(path+mode+f"_sim_{i}/measured_cqs.fits", _cqs)

    # Masked
    data_maps[("POS", 1)] *= vmap
    data_maps[("POS", 2)] *= vmap
    data_maps[("SHE", 1)] *= vmap
    data_maps[("SHE", 2)] *= vmap

    alms = transform(fields, data_maps)
    cls_wmask = heracles.angular_power_spectra(alms)
    cqs_wmask = heracles.binned(cls_wmask, ledges)
    _cls_wmask = dices.compsep_Cls(cls_wmask)
    _cqs_wmask = dices.compsep_Cls(cqs_wmask)

    heracles.write(path+mode+f"_sim_{i}/measured_cls_wmask.fits", _cls_wmask)
    heracles.write(path+mode+f"_sim_{i}/measured_cqs_wmask.fits", _cqs_wmask)

    for key in list(_cls.keys()):
        cl = _cls[key].__array__()
        cq = _cqs[key].__array__()
        cl_wmask = _cls_wmask[key].__array__()
        cq_wmask = _cqs_wmask[key].__array__()
        if i==1:
            clss[key] = [cl]
            cqss[key] = [cq]
            clss_wmask[key] = [cl_wmask]
            cqss_wmask[key] = [cq_wmask]
        else:
            clss[key] = clss[key]+[cl]
            cqss[key] = cqss[key]+[cq]
            clss_wmask[key] = clss_wmask[key]+[cl_wmask]
            cqss_wmask[key] = cqss_wmask[key]+[cq_wmask]

cls_m = {}
cqs_m = {}
cls_wmask_m = {}
cqs_wmask_m = {}
cls_cov = {}
cqs_cov = {}
cls_wmask_cov = {}
cqs_wmask_cov = {}
for key in list(clss.keys()):
    print(f"Measuring {key}")
    cl = np.array(clss[key])
    cq = np.array(cqss[key])
    cl_wmask = np.array(clss_wmask[key])
    cq_wmask = np.array(cqss_wmask[key])
    cls_m[key] = heracles.Result(np.mean(cl, axis=0), ell=ls)
    cqs_m[key] = heracles.Result(np.mean(cq, axis=0), ell=lgrid)
    cls_wmask_m[key] = heracles.Result(np.mean(cl_wmask, axis=0), ell=ls)
    cqs_wmask_m[key] = heracles.Result(np.mean(cq_wmask, axis=0), ell=lgrid)
    cls_cov[key] = heracles.Result(np.cov(cl.T), ell=(ls,ls))
    cqs_cov[key] = heracles.Result(np.cov(cq.T), ell=(lgrid,lgrid))
    cls_wmask_cov[key] = heracles.Result(np.cov(cl_wmask.T), ell=(ls,ls))
    cqs_wmask_cov[key] = heracles.Result(np.cov(cq_wmask.T), ell=(lgrid,lgrid))

# Save
heracles.write(path+"mean_cls.fits", cls_m)
heracles.write(path+"cov_cls.fits", cls_cov)
heracles.write(path+"mean_cqs.fits", cqs_m)
heracles.write(path+"cov_cqs.fits", cqs_cov)
heracles.write(path+"mean_cls_wmask.fits", cls_wmask_m)
heracles.write(path+"cov_cls_wmask.fits", cls_wmask_cov)
heracles.write(path+"mean_cqs_wmask.fits", cqs_wmask_m)
heracles.write(path+"cov_cqs_wmask.fits", cqs_wmask_cov)
print("Done")
