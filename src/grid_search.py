"""

Created on 2019-05-28-16-30
Author: Stephan Rasp, raspstephan@gmail.com
"""
from L96 import *
import multiprocessing as mp
from EnKF import multi_helper
from utils import *

means = np.load('./L96TwoLevel_means.npy')
variances = np.load('./L96TwoLevel_variances.npy')
initX, initY = np.load('./initX.npy'), np.load('./initY.npy')

hs = np.arange(0.5, 1.6, 0.1)
Fs = np.arange(5, 16)
cs = np.arange(5, 16)
bs = np.arange(5, 16)

models = []
params = []
for ih, h in enumerate(hs):
    for iF, F in enumerate(Fs):
        for ic, c in enumerate(cs):
            for ib, b in enumerate(bs):
                params.append([ih, iF, ic, ib])
                models.append(
                    L96TwoLevel(noYhist=True, h=h, F=F, c=c, b=b, X_init=initX, Y_init=initY, noprog=True)
                )

pool = mp.Pool(processes=20)
results = [pool.apply_async(multi_helper, args=(l, 100)) for l in models]
models = [p.get() for p in results]
L = np.zeros((len(hs), len(Fs), len(cs), len(bs)))
for m, p in zip(models, params):
    L[p[0], p[1], p[2], p[3]] = climate_error(m, means, variances)
L = xr.DataArray(L, dims = {'h': hs, 'F': Fs, 'c': cs, 'b': bs}, coords = {'h': hs, 'F': Fs, 'c': cs, 'b': bs}, name='Loss')
L.to_netcdf('./grid_loss.nc')