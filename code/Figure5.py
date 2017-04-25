"""
    Plots figure 5: snapshot different components of the potential vorticity.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cmocean

from Utils import *

plt.close('all')

pathi = "outputs/decaying_turbulence/reference/coupled/"
patho = "../writeup/figs/"

## get params
params = h5py.File(pathi+"../parameters.h5","r")
setup = h5py.File(pathi+"setup.h5","r")
Ue, ke = params['dimensional/Ue'][()], params['dimensional/ke'][()]
Te = params['dimensional/Te'][()]
Uw = params['dimensional/Uw'][()]
f0 = params['dimensional/f0'][()]
## get grid
setup = h5py.File(pathi+"setup.h5","r")
x, y = setup['grid/x'][:]*ke/(2*np.pi), setup['grid/y'][:]*ke/(2*np.pi)

# t = 20
file = '000000020000000.h5'
snap = h5py.File(pathi+"snapshots/"+file)


t = snap['t'][()]/Te

k,l = setup['grid/k'][:],setup['grid/l'][:]
k,l = np.meshgrid(k,l)
wv2 = k**2 + l**2

phi = snap['phi'][:]
phih = np.fft.fft2(phi)
phi2h = np.fft.fft2(np.abs(phi)**2)
q = snap['q'][:]

phix, phiy = np.fft.ifft2(1j*k*phih), np.fft.ifft2(1j*l*phih)
J_phic_phi = np.conj(phix)*phiy - np.conj(phiy)*phix

qw1 = np.fft.ifft2(-wv2*phi2h).real/(4*f0)
qw2 = (1j*J_phic_phi).real/(2*f0)
qw = qw1+qw2
qpsi = q-qw


q0 = Ue*ke
qw, qw1, qw2 = qw/q0, qw1/q0, qw2/q0
q, qpsi = q/q0, qpsi/q0

cq   = np.arange(-2.5,2.6,.1)
xlim = [0,5]


fig = plt.figure(figsize=(7.7,8))

plt.subplot(221, aspect=1)
plt.contourf(x,y,q,cq, vmin=cq.min(), vmax=cq.max(),
                cmap = cmocean.cm.curl, extend='both')
plt.xlim(xlim)
plt.ylim(xlim)
plt.xticks([])
plt.ylabel(r"$y\times k_e/2\pi$")

fig.subplots_adjust(wspace=.165)
fig.subplots_adjust(hspace=.05)

plt.text(2.9, 5.1, r"$q = \Delta \psi + q^w_1 + q^w_2$",)
plt.text(.25,6.8,r'$t \times U_e k_e$ ='+ str(int(round(t))))

plt.subplot(222, aspect=1)
plt.contourf(x,y,qpsi,cq, vmin=cq.min(), vmax=cq.max(),
                cmap = cmocean.cm.curl, extend='both')
plt.xlim(xlim)
plt.ylim(xlim)
plt.xticks([])
plt.yticks([])

plt.text(4.5, 5.1, r"$\Delta \psi$",)

plt.subplot(223, aspect=1)
plt.contourf(x,y,qw1,cq, vmin=cq.min(), vmax=cq.max(),
                cmap = cmocean.cm.curl, extend='both')
plt.xlim(xlim)
plt.ylim(xlim)
plt.ylabel(r"$y\times k_e/2\pi$")
plt.xlabel(r"$x\times k_e/2\pi$")

plt.text(3.4, 5.1, r"$q^w_1 = \frac{1}{4 f_0}\Delta |\phi|^2$",)

plt.subplot(224, aspect=1)
im2= plt.contourf(x,y,qw2,cq, vmin=cq.min(), vmax=cq.max(),
                cmap = cmocean.cm.curl, extend='both')
plt.xlim(xlim)
plt.ylim(xlim)
plt.yticks([])
plt.xlabel(r"$x\times k_e/2\pi$")

plt.text(3.05, 5.1, r"$q^w_2 = \frac{\mathrm{i}}{2 f_0}J(\phi^{\star},\phi)$",)

fig.subplots_adjust(top=0.9)
cbar_ax = fig.add_axes([0.41, 1., 0.275, 0.013])
fig.colorbar(im2, cax=cbar_ax,label=r"Potential vorticity $[q \times (U_e k_e)^{-1}]$",
                orientation='horizontal',ticks=np.arange(-5,5,1))
plt.savefig(patho+"fig5.png", bbox_inches='tight')
