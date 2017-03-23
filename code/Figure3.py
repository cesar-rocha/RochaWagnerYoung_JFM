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

pathi = "outputs/decaying_turbulence/coupled/"
patho = "../writeup/figs/"

## get params
params = h5py.File(pathi[:-8]+"parameters.h5","r")
Ue, ke = params['dimensional/Ue'][()], params['dimensional/ke'][()]
Te = params['dimensional/Te'][()]
Uw = params['dimensional/Uw'][()]

## get grid
setup = h5py.File(pathi+"setup.h5","r")
x, y = setup['grid/x'][:]*ke/(2*np.pi), setup['grid/y'][:]*ke/(2*np.pi)
#x -= x.mean()
#y -= y.mean()

#files = ['000000000016667.h5', '000000001333333.h5',
#            '000000002666667.h5', '000000008000000.h5']

files = ['000000001000000.h5', '000000004000000.h5', '000000000200000.h5']
files = ['000000000225000.h5', '000000000650000.h5', '000000001375000.h5']

def plot_snapshot(fig, snap, panel = 1):
    """ Plot snapshot of vorticity and
            near-inertial kinetic energy density """

    ticks = np.arange(0,xlim[1]+1,1)

    t = snap['t'][()]/Te
    q = snap['q'][:]/(Ue*ke)
    phi2 = np.abs(snap['phi'])**2/Uw**2

    ax = fig.add_subplot(3,2,panel,aspect=1)
    pc_phi = ax.contourf(x,y, phi2,cphi,vmin=cphi.min(),vmax=cphi.max(),extend='max',
                     cmap = cmocean.cm.ice_r)
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(xlim[0],xlim[1])

    ax2 = fig.add_subplot(3,2,panel+1,aspect=1)
    pc_q = ax2.contourf(x,y, q, cq, vmin=cq.min(), vmax=cq.max(),
                            cmap = cmocean.cm.curl, extend='both')
    ax2.set_xlim(xlim[0],xlim[1])
    ax2.set_ylim(xlim[0],xlim[1])

    if panel == 1:
        ax.set_xticks([])
        ax.set_yticks(ticks)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax.set_ylabel(r"$y\times k_e/2\pi$")
        fig.subplots_adjust(wspace=.165)
        fig.subplots_adjust(hspace=.165)
    elif panel == 3:
        ax.set_yticks(ticks)
        ax.set_xticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax.set_ylabel(r"$x\times k_e/2\pi$")
    elif panel == 5:
        ax.set_yticks(ticks)
        ax.set_xticks(ticks)
        ax.set_xlabel(r"$x\times k_e/2\pi$")
        ax.set_ylabel(r"$y\times k_e/2\pi$")
        ax2.set_xticks(ticks)
        ax2.set_xlabel(r"$x\times k_e/2\pi$")
        ax2.set_yticks([])
    else:
        pass

    plot_fig_label(ax, xc =0.775, yc=1.03, label= r"$t \times U_e k_e$ = "+
                    str(int(round(t))), facecolor="1.0",boxstyle=None,alpha=0.)

    return pc_phi, pc_q

cphi = np.arange(0.,5.,0.1)
cq   = np.arange(-2.5,2.6,.1)

xlim = [0,5]
fig = plt.figure(figsize=(7.7,10))

panel = 1
for fni in files:
    snap = h5py.File(pathi+"snapshots/"+fni)
    im, im2 = plot_snapshot(fig, snap, panel)
    panel += 2

# colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.15, .965, 0.275, 0.013])
fig.colorbar(im, cax=cbar_ax,label=r"Wave kinetic energy density $[|\phi|^2/U_w^2]$",
                orientation='horizontal',ticks=np.arange(0,7,1))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.5, .965, 0.275, 0.013])
fig.colorbar(im2, cax=cbar_ax,label=r"Potential vorticity $[q \times (U_e k_e)^{-1}]$",
                orientation='horizontal',ticks=np.arange(-5,5,1))

plt.savefig(patho+"fig3.png", bbox_inches='tight')
