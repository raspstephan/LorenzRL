"""

Created on 2019-06-11-13-38
Author: Stephan Rasp, raspstephan@gmail.com
"""

import fire
import numpy as np
from L96 import *
from EnKF import *
from utils import *
from parameterizations import *
import pickle

def run_EnKF(save_fn, mode, init_p, nens=100, mp=15, cyc_len=0.2, cycles=20, r=0.5):

    initX, initY = np.load('./data/initX.npy'), np.load('./data/initY.npy')
    init2level = np.concatenate([initX, initY])
    means = np.load('./data/L96TwoLevel_means.npy')
    variances = np.load('./data/L96TwoLevel_variances.npy')
    with open(init_p, 'rb') as f: p_det, p_ens = pickle.load(f)

    if mode == 'nwp':
        def set_state(l, x_a):
            l.set_state(x_a[:-len(l.parameterization.p)])
            l.parameterization.p = x_a[-len(l.parameterization.p):]

        def get_state(l):
            return np.concatenate([l.state, l.parameterization.p])

        def H(l):
            return l.state

        enkf = EnKF(
            l96=L96TwoLevelParam(
                noprog=True, save_dt=0.001, noYhist=True, parameterization=PolyParam(p_det)),
            nens=nens,
            obs_noise=r,
            cyc_len=cyc_len,
            mp=mp,
            get_state=get_state,
            set_state=set_state,
            H=H
        )

        init_p = InitP(p_det, p_ens)
        enkf.initialize(init2level, np.array([1] * 36 + [0.1] * 360))
        enkf.initialize_parameters(init_p, priors=-999, sigmas=-999)
        enkf.l96_tru = L96TwoLevel(noprog=True, save_dt=0.01, noYhist=True, X_init=initX,
                                   Y_init=initY)
        enkf.iterate(cycles)
        with open(save_fn, 'wb') as f: pickle.dump((
            enkf.parameter_history_det, enkf.parameter_history_ens, enkf.mse_det, enkf.mse_ens
        ), f)

    else:   # Climate
        def set_state(l, x_a):
            l.parameterization.p = x_a
            
        def H(l):
            return l.mean_stats(0)

        enkf = EnKF(
            l96=L96TwoLevelParam(noprog=True, save_dt=0.001, noYhist=True,
                                 parameterization=PolyParam(p_det)),
            nens=nens,
            obs_noise=r ** 2 * variances,
            cyc_len=cyc_len,
            mp=mp,
            get_state=lambda l: l.parameterization.p,
            set_state=set_state,
            H=H,
            y=means,
            climate=True
        )
        init_p = InitP(p_det, p_ens)
        enkf.initialize(init2level, np.array([1] * 36 + [0.1] * 360))
        enkf.initialize_parameters(init_p, priors=-999, sigmas=-999)

        enkf.iterate(cycles)
        with open(save_fn, 'wb') as f: pickle.dump((
            enkf.parameter_history_det, enkf.parameter_history_ens, enkf.climate_error
        ), f)


if __name__ == '__main__':
    fire.Fire(run_EnKF)
