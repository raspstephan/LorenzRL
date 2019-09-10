"""

Created on 2019-05-28-16-00
Author: Stephan Rasp, raspstephan@gmail.com
"""
import numpy as np


def climate_error(l, means, variances):
    return np.sqrt((((means - l.mean_stats(0))/np.sqrt(variances))**2)).mean()

class InitP():
    def __init__(self, p, ps):
        self.p, self.ps = p, ps
        self.i = 0
    def __call__(self, l, prior, sigma):
        if np.mean(sigma) == 0:
            l.parameterization.p = self.p
        else:
            l.parameterization.p = self.ps[self.i]
            self.i +=1