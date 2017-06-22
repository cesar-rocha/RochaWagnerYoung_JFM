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

pathi = "outputs/sinxsiny_wavepacket/"
pathi_passive = "outputs/sinxsiny_passive_wavepacket/"
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
k, l = setup['grid/k'][:], setup['grid/l'][:]
k, l = np.meshgrid(k,l)

diags = h5py.File(pathi+"diagnostics.h5")
diags_passive = h5py.File(pathi_passive+"diagnostics.h5")

files = [ '000000000000000.h5','000000030000000.h5']
snap_passive = h5py.File(pathi_passive+"snapshots/"+files[-1])

#ticks = np.arange(0,xlim[1],2)
cb = np.hstack([np.linspace(-6,-0.1,40),np.linspace(0.1,6,40)])

snap = h5py.File(pathi_passive+"snapshots/"+files[-1])
cstd = snap['c'][:].std()

def plot_snapshot(snap,panel):

    t = snap['t'][()]/Te
    q = snap['q'][:]/(Ue*ke)

    b = snap['phi'][:].real/cstd

    ax = fig.add_subplot(2,3,panel,aspect=1)

    ax.contour(x,y,q,np.arange(-1.5,2.,.5),colors='k')
    ax.contourf(x,y,b,cb,cmap=cmocean.cm.balance,extend='both')

    if panel==1:
        plt.title(r"initial condition")
        plot_fig_label(ax,xc=0.92, label="a")
    elif panel==2:
        plt.title(r"waves")
        plot_fig_label(ax,xc=0.92, label="b")

    ax.set_xticks([])
    ax.set_yticks([])

def plot_snapshot_passive(snap,panel):

    t = snap['t'][()]/Te
    q = snap['q'][:]/(Ue*ke)
    c = snap['c'][:]/cstd

    ax = fig.add_subplot(2,3,panel,aspect=1)

    ax.contour(x,y,q,np.arange(-1.5,2.,.5),colors='k')
    ax.contourf(x,y,c,cb,cmap=cmocean.cm.balance,extend='both')
    plt.title(r"passive scalar")
    ax.set_xticks([])
    ax.set_yticks([])
    plot_fig_label(ax,xc=0.92, label="c")

fig = plt.figure(figsize=(8.5,6.))
for i in range(2):
    snap = h5py.File(pathi+"snapshots/"+files[i])
    plot_snapshot(snap,panel=i+1)

snap = h5py.File(pathi_passive+"snapshots/"+files[-1])
plot_snapshot_passive(snap,panel=3)


ax = fig.add_subplot(2,2,3,)
ax.plot(diags['time'][:]/Te,diags['Kw'][:]/diags['Kw'][0],label='waves')
ax.plot(diags_passive['time'][:]/Te,diags_passive['C2'][:]/diags_passive['C2'][0],label='passive scalar')
plt.xlabel(r"Time $[t \times U_e k_e]$")
plt.ylabel(r"Variance [$\langle |\phi|^2\rangle]$")
plot_fig_label(ax, label="d")

fig.subplots_adjust(wspace=.3)

ax2 = fig.add_subplot(2,2,4,)
ax2.plot(diags['time'][:]/Te,diags['Pw'][:]/diags['Pw'][0],label='waves')
ax2.plot(diags_passive['time'][:]/Te,diags_passive['gradC2'][:]/diags_passive['gradC2'][0],label='passive scalar')
plt.xlabel(r"Time $[t \times U_e k_e]$")
plt.ylabel(r"Grad. variance [$\langle |\nabla\phi|^2\rangle]$")
plt.legend()
plot_fig_label(ax2, label="e")

plt.savefig(patho+"figesc.png",
            pad_inches=0,
            bbox_inches='tight',)#dpi=400)
