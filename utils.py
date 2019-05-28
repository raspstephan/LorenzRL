"""

Created on 2019-05-28-16-00
Author: Stephan Rasp, raspstephan@gmail.com
"""
import numpy as np


def climate_error(l, means, variances):
    return np.sqrt((((means - l.mean_stats(0))/np.sqrt(variances))**2)).mean()