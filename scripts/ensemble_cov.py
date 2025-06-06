import os
import yaml
import numpy as np
import heracles

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
clss = {}
clss_wmask = {}
cqss = {}
cqss_wmask = {}
for i in range(1, n+1):
    print(f"Loading sim {i}", end='\r')
    if os.path.exists(path+mode+f"_sim_{i}/measured_cls.fits"):
        cls = heracles.read(path+mode+f"_sim_{i}/measured_cls.fits")
        cls_wmask = heracles.read(path+mode+f"_sim_{i}/measured_cls_wmask.fits")
        for key in list(cls.keys()):
            cl = cls[key].array
            cl_wmask = cls_wmask[key].array
            cq = heracles.binned(cls[key], ledges)
            cq_wmask = heracles.binned(cls_wmask[key], ledges)
            if i==1:
                clss[key] = [cl]
                clss_wmask[key] = [cl_wmask]
                cqss[key] = [cq]
                cqss_wmask[key] = [cq_wmask]
            else:
                clss[key] = clss[key]+[cl]
                clss_wmask[key] = clss_wmask[key]+[cl_wmask]
                cqss[key] = cqss[key]+[cq]
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
    a, b, i, j = key
    print(f"Measuring {key}")
    cl = np.array(clss[key])
    cl_wmask = np.array(clss_wmask[key])
    cq = np.array(cqss[key])
    cq_wmask = np.array(cqss_wmask[key])
    print(f"Shape of cls: {cl.shape}")
    if a == b == 'POS':
        cls_m[key] = heracles.Result(np.mean(cl, axis=0), ell=ls)
        cls_wmask_m[key] = heracles.Result(np.mean(cl_wmask, axis=0), ell=ls)
        cls_cov[key] = heracles.Result(np.cov(cl.T), ell=(ls, ls))
        cls_wmask_cov[key] = heracles.Result(np.cov(cl_wmask.T), ell=(ls, ls))
        cqs_m[key] = heracles.Result(np.mean(cq, axis=0), ell=lgrid)
        cqs_wmask_m[key] = heracles.Result(np.mean(cq_wmask, axis=0), ell=lgrid)
        cqs_cov[key] = heracles.Result(np.cov(cq.T), ell=(lgrid, lgrid))
        cqs_wmask_cov[key] = heracles.Result(np.cov(cq_wmask.T), ell=(lgrid, lgrid))
    elif a == "POS" and b == "SHE":
        cls_m[key] = heracles.Result(np.mean(cl, axis=0), ell=ls)
        cls_wmask_m[key] = heracles.Result(np.mean(cl_wmask, axis=0), ell=ls)
        cqs_m[key] = heracles.Result(np.mean(cq, axis=0), ell=lgrid)
        cqs_wmask_m[key] = heracles.Result(np.mean(cq_wmask, axis=0), ell=lgrid)
        _cov = np.zeros((2, 2, len(ls), len(ls)))
        _cov_wmask = np.zeros((2, 2, len(ls), len(ls)))
        _covq = np.zeros((2, 2, len(lgrid), len(lgrid)))
        _covq_wmask = np.zeros((2, 2, len(lgrid), len(lgrid)))
        for i in range(2):
            _cov[i, i, :, :] = np.cov(cl[:, i, :].T)
            _cov_wmask[i, i, :, :] = np.cov(cl_wmask[:, i, :].T)
            _covq[i, i, :, :] = np.cov(cq[:, i, :].T)
            _covq_wmask[i, i, :, :] = np.cov(cq_wmask[:, i, :].T)
        cls_cov[key] = heracles.Result(_cov, ell=(ls, ls))
        cls_wmask_cov[key] = heracles.Result(_cov_wmask, ell=(ls, ls))
        cqs_cov[key] = heracles.Result(_covq, ell=(lgrid, lgrid))
        cqs_wmask_cov[key] = heracles.Result(_covq_wmask, ell=(lgrid, lgrid))
    elif a == "SHE" and b == "SHE":
        cls_m[key] = heracles.Result(np.mean(cl, axis=0), ell=ls)
        cls_wmask_m[key] = heracles.Result(np.mean(cl_wmask, axis=0), ell=ls)
        cqs_m[key] = heracles.Result(np.mean(cq, axis=0), ell=lgrid)
        cqs_wmask_m[key] = heracles.Result(np.mean(cq_wmask, axis=0), ell=lgrid)
        _cov = np.zeros((2, 2, 2, 2, len(ls), len(ls)))
        _cov_wmask = np.zeros((2, 2, 2, 2, len(ls), len(ls)))
        _covq = np.zeros((2, 2, 2, 2, len(lgrid), len(lgrid)))
        _covq_wmask = np.zeros((2, 2, 2, 2, len(lgrid), len(lgrid)))
        for i in range(2):
            for j in range(2):
                _cov[i, i, j, j, :, :] = np.cov(cl[:, i, j, :].T)
                _cov_wmask[i, i, j, j, :, :] = np.cov(cl_wmask[:, i, j, :].T)
                _covq[i, i, j, j, :, :] = np.cov(cq[:, i, j, :].T)
                _covq_wmask[i, i, j, j, :, :] = np.cov(cq_wmask[:, i, j, :].T)
        cls_cov[key] = heracles.Result(_cov, ell=(ls, ls))
        cls_wmask_cov[key] = heracles.Result(_cov_wmask, ell=(ls, ls))
        cqs_cov[key] = heracles.Result(_covq, ell=(lgrid, lgrid))
        cqs_wmask_cov[key] = heracles.Result(_covq_wmask, ell=(lgrid, lgrid))

# Save
heracles.write(path+"mean_cls.fits", cls_m)
heracles.write(path+"cov_cls.fits", cls_cov)
heracles.write(path+"mean_cls_wmask.fits", cls_wmask_m)
heracles.write(path+"cov_cls_wmask.fits", cls_wmask_cov)
heracles.write(path+"mean_cqs.fits", cqs_m)
heracles.write(path+"cov_cqs.fits", cqs_cov)
heracles.write(path+"mean_cqs_wmask.fits", cqs_wmask_m)
heracles.write(path+"cov_cqs_wmask.fits", cqs_wmask_cov)
print("Done")
