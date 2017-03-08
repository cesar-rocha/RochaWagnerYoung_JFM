"""
    Plots figure 1: energy time series, budgets,
                    wave-vorticity correlation
                    of the Lamb-Chapygin dipole solution.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

from Utils import *

plt.close('all')

pathi = "outputs/lambdipole/"
patho = "../writeup/figs/"
