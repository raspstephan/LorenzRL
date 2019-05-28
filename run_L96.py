"""

Created on 2019-05-28-15-27
Author: Stephan Rasp, raspstephan@gmail.com
"""

import fire
from L96 import *


def run_l96(save_fn=None, K=36, J=10, h=1, F=10, c=10, b=10, dt=0.001, X_init=None, Y_init=None,
            noYhist=False, save_dt=0.1, lead_time=100):

    if X_init is not None: X_init = np.load(X_init)
    if Y_init is not None: Y_init = np.load(Y_init)

    l96 = L96TwoLevel(K=K, J=J, h=h, F=F, c=c, b=b, dt=dt, X_init=X_init, Y_init=Y_init,
                      noYhist=noYhist, save_dt=save_dt)
    l96.iterate(lead_time)
    l96.history.to_netcdf(save_fn)

if __name__ == '__main__':
    fire.Fire(run_l96)