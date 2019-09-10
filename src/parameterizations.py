"""

Created on 2019-06-11-14-19
Author: Stephan Rasp, raspstephan@gmail.com
"""
import numpy as np

class PolyParam():
    def __init__(self, p):
        self.p = p
    def __call__(self, x):
        return np.polyval(self.p, x)