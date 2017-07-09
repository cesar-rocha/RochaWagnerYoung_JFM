"""
    Plots figure 1: snapshots of the Lamb-Chapygin dipole solution.
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
f0 = params['dimensional/f0'][()]
m = params['dimensional/m'][()]
N0 = params['dimensional/N'][()]
nu4w = params['dimensional/nu4'][()]
lam2  = (N0/f0/m)**2
lam  = np.sqrt(lam2)

## get grid
setup = h5py.File(pathi+"setup.h5","r")
x, y = setup['grid/x'][:]*ke/(2*np.pi), setup['grid/y'][:]*ke/(2*np.pi)
x -= x.mean()
y -= y.mean()

k, l = setup['grid/k'][:], setup['grid/l'][:]
k, l = np.meshgrid(k,l)
wv2 = k**2 + l**2
#files = ['000000000016667.h5', '000000000900000.h5',
#            '000000002666667.h5', '000000008000000.h5']

#files = ['000000000016667.h5','000000002666667.h5', '000000008000000.h5']
files = ['000000000300000.h5','000000002666667.h5', '000000005200000.h5']

snap = h5py.File(pathi+"snapshots/"+files[-1])

t = snap['t'][()]/Te
q = snap['q'][:]/(Ue*ke)
phi2 = np.abs(snap['phi'])**2/Uw**2
uw, vw, ww, pw, bw = wave_fields(snap['phi'][:],f0,lam2,snap['t'][()],k,l,m)

phih = np.fft.fft2(snap['phi'][:])
phi2h = np.fft.fft2(snap['phi'][:]**2)
phix, phiy = np.fft.ifft2(1j*k*phih), np.fft.ifft2(1j*l*phih)
J_phic_phi = np.conj(phix)*phiy - np.conj(phiy)*phix

qw1 = np.fft.ifft2(-wv2*phi2h).real/(4*f0)
qw2 = (1j*J_phic_phi).real/(2*f0)
qw = qw1+qw2
qpsi = snap['q'][:]-qw
qpsi = qpsi/(Ue*ke)
qw = qw/(Ue*ke)

cphi = np.arange(0.,4.1,0.1)
cb = np.arange(-.85,.95,0.1)
cq   = np.array([-2.5,-1.5,-0.5,0.5,1.5,2.5])
xlim = [-2,1]
ylim = [-1,2]

fig = plt.figure(figsize=(8.5,5.5))
ax = fig.add_subplot(1,2,1,aspect=1)
fig.subplots_adjust(wspace=.075)
fig.subplots_adjust(hspace=.1)
pc1 = ax.pcolormesh(x,y, phi2,vmin=cphi.min(),vmax=cphi.max(),
                 cmap = cmocean.cm.ice_r, shading='flat')
ax.contour(x,y, qpsi, cq,colors='k')
ax.set_xlim(xlim[0],xlim[1])
ax.set_ylim(ylim[0],ylim[1])

ax2 = fig.add_subplot(1,2,2,aspect=1)
pc2 = ax2.pcolormesh(x,y, bw/Uw/m/(f0*lam2)/ke,vmin=cb.min(),vmax=cb.max(),
                 cmap = cmocean.cm.balance, shading='flat')
ax2.contour(x,y, qpsi, cq,colors='k')
ax2.set_xlim(xlim[0],xlim[1])
ax2.set_ylim(ylim[0],ylim[1])

#ax.set_xticks([])
ax.set_yticks([0,1,2])
ax.set_xticks([-2,-1,0,1])
ax.set_ylabel(r"$y\times k_e/2\pi$")
ax.set_xlabel(r"$x\times k_e/2\pi$")
ax2.set_yticks([])
ax2.set_xticks([-2,-1,0,1])
ax2.set_xlabel(r"$x\times k_e/2\pi$")


#plot_fig_label(ax2, xc =0.75, yc=.05, label= r"$t \times U_e k_e$ = "+
#                str(int(round(t))), facecolor="1.0",boxstyle=None,alpha=0.)


# colorbar
#fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.18, .915, 0.28, 0.0275])
fig.colorbar(pc1, cax=cbar_ax,label=r"Wave action density $[\mathcal{A} \times 2 f_0 /U_w^2]$",
                    orientation='horizontal', ticks=[0.,2.,4.],extend='max')
cbar_ax = fig.add_axes([0.58, .915, 0.28, 0.0275])
fig.colorbar(pc2, cax=cbar_ax,label=r"Wave buoyancy $[b/B]$",
                    orientation='horizontal',ticks=[-.85,0.,.85],
                    extend='both')

fig = plt.figure(figsize=(8.5,4.5))

ax = fig.add_subplot(121,aspect=1)
fig.subplots_adjust(wspace=.075)
pc1 = ax.contourf(x,y, qpsi,cq,vmin=cq.min(),vmax=cq.max(),
                 cmap = cmocean.cm.curl, extend='both')
ax.contour(x,y, q, cq,colors='k')

plot_fig_label(ax, xc=0.955, yc=.05, label='a')


ax2 = fig.add_subplot(122,aspect=1)

pc2 = ax2.contourf(x,y, qw,cq,vmin=cq.min(),vmax=cq.max(),
                 cmap = cmocean.cm.curl, extend='both')
ax2.contour(x,y, q, cq,colors='k')
ax.set_yticks([0,1,2])
ax.set_xticks([-2,-1,0,1])
ax.set_ylabel(r"$y\times k_e/2\pi$")
ax.set_xlabel(r"$x\times k_e/2\pi$")
ax2.set_yticks([])
ax2.set_xticks([-2,-1,0,1])
ax2.set_xlabel(r"$x\times k_e/2\pi$")

ax.set_xlim(xlim[0],xlim[1])
ax.set_ylim(ylim[0],ylim[1])
ax2.set_xlim(xlim[0],xlim[1])
ax2.set_ylim(ylim[0],ylim[1])

plot_fig_label(ax2, xc=0.955, yc=.05, label='b')


plt.savefig(patho+"fig1b.png", pad_inces=0, bbox_inches='tight')
#plt.savefig(patho+"fig1.eps",dpi=200, pad_inces=0, bbox_inches='tight')
#plt.savefig(patho+"fig1.pdf",dpi=100, pad_inces=0, bbox_inches='tight')
