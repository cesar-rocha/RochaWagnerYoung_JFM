"""
    Plots figure 1: snapshots vorticity and near-inertial
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

## get grid
setup = h5py.File(pathi+"setup.h5","r")
x, y = setup['grid/x'][:]*ke/(2*np.pi), setup['grid/y'][:]*ke/(2*np.pi)
x -= x.mean()
y -= y.mean()

#files = ['000000000016667.h5', '000000001333333.h5',
#            '000000002666667.h5', '000000008000000.h5']

files = ['000000000016667.h5', '000000000900000.h5',
            '000000002666667.h5', '000000008000000.h5']


def plot_snapshot(fig, snap, panel = 1):
    """ Plot snapshot of vorticity and
            near-inertial kinetic energy density """

    t = snap['t'][()]/Te
    q = snap['q'][:]/(Ue*ke)
    phi2 = np.abs(snap['phi'])**2/Uw**2

    ax = fig.add_subplot(2,2,panel,aspect=1)
    pc = ax.contourf(x,y, phi2,cphi,vmin=cphi.min(),vmax=cphi.max(),extend='max',
                     cmap = cmocean.cm.ice_r)
    ax.contour(x,y, q, cq,colors='k')
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(xlim[0],xlim[1])

    if panel == 1:
        ax.set_xticks([])
        ax.set_yticks([-2,0,2])
        ax.set_ylabel(r"$y\times k_e/2\pi$")
        fig.subplots_adjust(wspace=.145)
    elif panel == 2:
        ax.set_xticks([])
        ax.set_yticks([])
        fig.subplots_adjust(hspace=.01)
    elif panel == 3:
        ax.set_yticks([-2,0,2])
        ax.set_xticks([-2,0,2])
        ax.set_xlabel(r"$x\times k_e/2\pi$")
        ax.set_ylabel(r"$y\times k_e/2\pi$")
    elif panel == 4:
        ax.set_yticks([])
        ax.set_xticks([-2,0,2])
        ax.set_xlabel(r"$x\times k_e/2\pi$")
    else:
        pass

    plot_fig_label(ax, xc =0.775, yc=1.03, label= r"$t \times U_e k_e$ = "+
                    str(int(round(t))), facecolor="1.0",boxstyle=None,alpha=0.)

    return pc

cphi = np.arange(0.,5.,0.1)
cq   = np.array([-1.5,-0.5,0.5,1.5])

xlim = [-3,3]
fig = plt.figure(figsize=(7.,6.5))

panel = 1
for fni in files:
    snap = h5py.File(pathi+"snapshots/"+fni)
    im = plot_snapshot(fig, snap, panel)
    panel += 1

# colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.21, 0.0275, 0.55])
fig.colorbar(im, cax=cbar_ax,label=r"Wave kinetic energy density $[|\phi|^2/U_w^2]$")
plt.savefig(patho+"fig1.png", bbox_inches='tight')
