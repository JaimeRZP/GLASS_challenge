import os
import numpy as np
import heracles

mode = "lognormal"
nside = 1024
lmax = 1500
ls = np.arange(lmax + 1)
path = f"../{mode}_sims/"
clss = {}
clss_wmask = {}
for i in range(1, 100+1):
    print(f"Loading sim {i}", end='\r')
    if os.path.exists(path+mode+f"_sim_{i}/measured_cls.fits"):
        cls = heracles.read(path+mode+f"_sim_{i}/measured_cls.fits")
        cls_wmask = heracles.read(path+mode+f"_sim_{i}/measured_cls_wmask.fits")
        for key in list(cls.keys()):
            cl = cls[key].array
            cl_wmask = cls_wmask[key].array
            if i==1:
                clss[key] = [cl]
                clss_wmask[key] = [cl_wmask]
            else:
                clss[key] = clss[key]+[cl]
                clss_wmask[key] = clss_wmask[key]+[cl_wmask]
cls_m = {}
cls_wmask_m = {}
cls_cov = {}
cls_wmask_cov = {}
for key in list(clss.keys()):
    a, b, i, j = key
    print(f"Measuring {key}")
    cl = np.array(clss[key])
    cl_wmask = np.array(clss_wmask[key])
    print(f"Shape of cls: {cl.shape}")
    if a == b == 'POS':
        cls_m[key] = heracles.Result(np.mean(cl, axis=0), ell=ls)
        cls_wmask_m[key] = heracles.Result(np.mean(cl_wmask, axis=0), ell=ls)
        cls_cov[key] = heracles.Result(np.cov(cl.T), ell=(ls, ls))
        cls_wmask_cov[key] = heracles.Result(np.cov(cl_wmask.T), ell=(ls, ls))
    elif a == "POS" and b == "SHE":
        cls_m[key] = heracles.Result(np.mean(cl, axis=0), ell=ls)
        cls_wmask_m[key] = heracles.Result(np.mean(cl_wmask, axis=0), ell=ls)
        _cov = np.zeros((2, 2, len(ls), len(ls)))
        _cov_wmask = np.zeros((2, 2, len(ls), len(ls)))
        for i in range(2):
            _cov[i, i, :, :] = np.cov(cl[:, i, :].T)
            _cov_wmask[i, i, :, :] = np.cov(cl_wmask[:, i, :].T)
        cls_cov[key] = heracles.Result(_cov, ell=(ls, ls))
        cls_wmask_cov[key] = heracles.Result(_cov_wmask, ell=(ls, ls))
    elif a == "SHE" and b == "SHE":
        cls_m[key] = heracles.Result(np.mean(cl, axis=0), ell=ls)
        cls_wmask_m[key] = heracles.Result(np.mean(cl_wmask, axis=0), ell=ls)
        _cov = np.zeros((2, 2, 2, 2, len(ls), len(ls)))
        _cov_wmask = np.zeros((2, 2, 2, 2, len(ls), len(ls)))
        for i in range(2):
            for j in range(2):
                _cov[i, i, j, j, :, :] = np.cov(cl[:, i, j, :].T)
                _cov_wmask[i, i, j, j, :, :] = np.cov(cl_wmask[:, i, j, :].T)
        cls_cov[key] = heracles.Result(_cov, ell=(ls, ls))
        cls_wmask_cov[key] = heracles.Result(_cov_wmask, ell=(ls, ls))

# Save
heracles.write(path+"mean_cls.fits", cls_m)
heracles.write(path+"cov_cls.fits", cls_cov)
heracles.write(path+"mean_cls_wmask.fits", cls_wmask_m)
heracles.write(path+"cov_cls_wmask.fits", cls_wmask_cov)
print("Done")
