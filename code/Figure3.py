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

#pathi = "outputs/decaying_turbulence/reference/coupled/"
pathi = "outputs/high_res/decaying_turbulence/parameter_exploration/Uw0.1/lambdaz397.5/"
patho = "../writeup/figs/"

## get params
params = h5py.File(pathi+"parameters.h5","r")
Ue, ke = params['dimensional/Ue'][()], params['dimensional/ke'][()]
Te = params['dimensional/Te'][()]
Uw = params['dimensional/Uw'][()]
m = params['dimensional/m'][()]
N0 = params['dimensional/N'][()]
f0 = params['dimensional/f0'][()]
lam2 = (N0/f0/m)**2
## get grid
setup = h5py.File(pathi+"setup.h5","r")
x, y = setup['grid/x'][:]*ke/(2*np.pi), setup['grid/y'][:]*ke/(2*np.pi)
#x -= x.mean()
#y -= y.mean()
k, l = setup['grid/k'][:], setup['grid/l'][:]
k, l = np.meshgrid(k,l)


#files = ['000000000016667.h5', '000000001333333.h5',
#            '000000002666667.h5', '000000008000000.h5']

files = ['000000001000000.h5', '000000004000000.h5', '000000020000000.h5']
files = ['000000001000000.h5', '000000008000000.h5', '000000040000000.h5']
files = ['000000000500000.h5', '000000004000000.h5', '000000010000000.h5']

def plot_snapshot(fig, snap, panel = 1, frac = 2):
    """ Plot snapshot of vorticity and
            near-inertial kinetic energy density """

    ticks = np.arange(0,xlim[1],2)



    t = snap['t'][()]/Te
    q = snap['q'][:]/(Ue*ke)
    uw, vw, ww, pw, bw = wave_fields(snap['phi'][:],f0,lam2,snap['t'][()],k,l,m)

    b = bw/(Uw*m*ke*f0*lam2)
    phi2 = np.abs(snap['phi'])**2/Uw**2

    
    nx,ny = q.shape
    nmax = nx//2

    ax2 = fig.add_subplot(3,3,panel,aspect=1)
    fig.subplots_adjust(wspace=.01)
    fig.subplots_adjust(hspace=.155)
    
    pc_q = ax2.contourf(x[:nmax,:nmax],y[:nmax,:nmax], q[:nmax,:nmax],
                            cq, vmin=cq.min(), vmax=cq.max(),
                            cmap = cmocean.cm.curl, extend='both')
    ax2.set_xlim(xlim[0],xlim[1])
    ax2.set_ylim(xlim[0],xlim[1])



    ax = fig.add_subplot(3,3,panel+3,aspect=1)
   
    pc_phi = ax.contourf(x[:nmax,:nmax],y[:nmax,:nmax], phi2[:nmax,:nmax],
                            cphi,vmin=cphi.min(),vmax=cphi.max(),extend='max',
                            cmap = cmocean.cm.ice_r)
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(xlim[0],xlim[1])


    ax3 = fig.add_subplot(3,3,panel+6,aspect=1)
    pc_b = ax3.contourf(x[:nmax,:nmax],y[:nmax,:nmax], b[:nmax,:nmax],
                            cb, vmin=cb.min(), vmax=cb.max(),
                            cmap = cmocean.cm.balance, extend='both')
    ax3.set_xlim(xlim[0],xlim[1])
    ax3.set_ylim(xlim[0],xlim[1])

    if panel == 1:
        ax.set_xticks([])
        ax2.set_xticks([])
        ax2.set_yticks(ticks)
        ax.set_yticks(ticks)
        ax3.set_yticks(ticks)
        ax3.set_xticks(ticks)
        ax2.set_ylabel(r"$y\times k_e/2\pi$")
        ax3.set_ylabel(r"$y\times k_e/2\pi$")
        ax3.set_xlabel(r"$x\times k_e/2\pi$")
        ax.set_ylabel(r"$y\times k_e/2\pi$")
        fig.subplots_adjust(wspace=.165)
        fig.subplots_adjust(hspace=.165)
    elif panel == 2 or panel == 3:
        ax.set_xticks([])
        ax.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_yticks([])
        ax3.set_xticks(ticks)
        ax3.set_xlabel(r"$x\times k_e/2\pi$")
    else:
        pass

    plot_fig_label(ax2, xc =0.7, yc=1.05, label= r"$t \times U_e k_e$ = "+
                    str(int(round(t))), facecolor="1.0",boxstyle=None,alpha=0.)

    return pc_phi, pc_q, pc_b

cphi = np.arange(0.,5.,0.1)
cq   = np.arange(-2.5,2.6,.1)
cb   = np.arange(-2.5,2.6,.1)

xlim = [0,5]
fig = plt.figure(figsize=(8.5,7.25))

panel = 1
for fni in files:
    snap = h5py.File(pathi+"snapshots/"+fni)
    imphi, imq,imb = plot_snapshot(fig, snap, panel)
    panel += 1

# colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.81, .385, 0.013,0.22])
fig.colorbar(imphi, cax=cbar_ax,label=r"Wave KE density $[|\phi|^2/U_w^2]$",
                orientation='vertical',ticks=np.arange(0,7,1))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([.81, .65, 0.013, 0.22])
fig.colorbar(imq, cax=cbar_ax,label=r"QGPV $[q/(U_e k_e)]$",
                orientation='vertical',ticks=np.arange(-5,5,1))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([.81, .1125, 0.013, 0.22])
fig.colorbar(imb, cax=cbar_ax,label=r"Wave buoyancy $[b/B]$",
                orientation='vertical',ticks=np.arange(-5,5,1))

plt.savefig(patho+"fig3.png", transparent=True,
            pad_inches=0,
            bbox_inches='tight',dpi=400)
