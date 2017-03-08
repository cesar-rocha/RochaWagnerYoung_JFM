"""
    Plots figure 2: snapshots vorticity and near-inertial
                    kinetic energy density of the Lamb-Chapygin dipole solution.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cmocean

from Utils import *

plt.close('all')

pathi = "outputs/lambdipole/"
patho = "../writeup/figs/"

## get params
params = h5py.File(pathi+"parameters.h5","r")
Ue, ke = params['dimensional/Ue'][()], params['dimensional/ke'][()]
Te = params['dimensional/Te'][()]
Uw = params['dimensional/Uw'][()]

## setup
setup = h5py.File(pathi+"setup.h5","r")

snaps = ['000000000016667.h5', '000000001333333.h5',
            '000000002666667.h5', '000000008000000.h5']

snap = h5py.File(pathi+"snapshots/"+snaps[1])

t = snap['t'][()]/Te
q = snap['q'][:]/(Ue*ke)
phi2 = np.abs(snap['phi'])**2/Uw**2
