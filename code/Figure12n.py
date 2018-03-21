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

wv2i = 1./wv2
wv2i[0] = 0

def plot_snapshot(fig, snap, panel = 1):
    """ Plot snapshot of vorticity and
            near-inertial kinetic energy density """

    t = snap['t'][()]/Te
    q = snap['q'][:]/(Ue*ke)
    phi2 = np.abs(snap['phi'])**2/Uw**2
    A =  (np.abs(snap['phi'])**2)/2/f0
    uw, vw, ww, pw, bw = wave_fields(snap['phi'][:],f0,lam2,snap['t'][()],k,l,m)

    phih = np.fft.fft2(snap['phi'][:])
    phi2h = np.fft.fft2(snap['phi'][:]**2)
    phix, phiy = np.fft.ifft2(1j*k*phih), np.fft.ifft2(1j*l*phih)
    J_phic_phi = np.conj(phix)*phiy - np.conj(phiy)*phix

    qw1 = np.fft.ifft2(-wv2*phi2h).real/(4*f0)
    qw2 = (1j*J_phic_phi).real/(2*f0)
    qw = qw1+qw2
    qpsi = snap['q'][:]-qw

    psiL = np.fft.ifft2(-np.fft.fft2(qpsi)*wv2i)
    psiE = (psiL + A)

    psiL, psiE,psiS = psiL/(Ue/ke), psiE/(Ue/ke), -A/(Ue/ke)


    qpsi = qpsi/(Ue*ke)

    ax = fig.add_subplot(3,3,panel,aspect=1)

    fig.subplots_adjust(wspace=-.0)
    fig.subplots_adjust(hspace=.1)
    #pc1 = ax.pcolormesh(x,y, phi2,vmin=cphi.min(),vmax=cphi.max(),
    #                 cmap = cmocean.cm.ice_r, shading='flat')
    pc1 = ax.contourf(x,y, psiL, cp, cmap=cmocean.cm.balance, extend='both')
    ax.contour(x,y, psiL-psiL.mean(), cp,colors='k',linewidths=.75)
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(xlim[0],xlim[1])

    plt.title(r"$t \times U_e k_e$ = "+str(int(round(t))))

    ax3 = fig.add_subplot(3,3,panel+6,aspect=1)
    #pc2 = ax2.pcolormesh(x,y, phi2,vmin=cphi.min(),vmax=cphi.max(),
    #                 cmap = cmocean.cm.ice_r, shading='flat')
    pc2 = ax3.contourf(x,y, psiS-psiS.mean(), cp,  cmap=cmocean.cm.balance,extend='both')
    ax3.contour(x,y, psiS-psiS.mean(), cp,colors='k',linewidths=.75)
    ax3.set_xlim(xlim[0],xlim[1])
    ax3.set_ylim(xlim[0],xlim[1])

    ax2 = fig.add_subplot(3,3,panel+3,aspect=1)
    #pc2 = ax2.pcolormesh(x,y, phi2,vmin=cphi.min(),vmax=cphi.max(),
    #                 cmap = cmocean.cm.ice_r, shading='flat')
    pc2 = ax2.contourf(x,y, psiE-psiE.mean(),  cp, cmap=cmocean.cm.balance,extend='both')
    ax2.contour(x,y, psiE-psiE.mean(), cp,colors='k',linewidths=.75)
    ax2.set_xlim(xlim[0],xlim[1])
    ax2.set_ylim(xlim[0],xlim[1])

    if panel == 1 or panel == 4:
        ax.set_xticks([])
        ax.set_yticks([-2,0,2])
        ax2.set_yticks([-2,0,2])
        ax3.set_yticks([-2,0,2])
        ax3.set_ylabel(r"$y\times k_e/2\pi$")
        ax.set_ylabel(r"$y\times k_e/2\pi$")
        ax2.set_xticks([])
        ax2.set_xlabel(r"")
        ax2.set_ylabel(r"$y\times k_e/2\pi$")

    elif panel == 2 or panel == 3:
        ax.set_xticks([])
        ax.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlabel('')
        ax3.set_yticks([])
        ax3.set_ylabel('')
        ax3.set_xlabel(r"$x\times k_e/2\pi$")
    else:
        pass

    if panel == 3:
        ax.text(3.5,0.,r"$\psi$",fontsize=14)
        ax2.text(3.5,0.,r"$\psi^E$",fontsize=14)
        ax3.text(3.25,0.,r"$-\mathcal{A}$",fontsize=14)

    return pc1, pc2

cphi = np.arange(0.,4.1,0.1)
cb = np.arange(-.85,.95,0.1)
cp = np.hstack([np.arange(-8,-.75,1.), np.arange(.75,9.,1.)])
#cp = np.arange(0,4.1,.1)
xlim = [-3,3]
fig = plt.figure(figsize=(8.5,8.5))

panel = 1
for fni in files:
    snap = h5py.File(pathi+"snapshots/"+fni)
    im1, im2 = plot_snapshot(fig, snap, panel)
    panel += 1

plt.savefig("figs/DipoleStreamfunction.pdf",pad_inches=0, bbox_inches='tight')


# now plot vorticity

snap = h5py.File(pathi+"snapshots/"+files[1])
t = snap['t'][()]/Te
q = snap['q'][:]/(Ue*ke)
phi2 = np.abs(snap['phi'])**2/Uw**2
A =  (np.abs(snap['phi'])**2)/2/f0
uw, vw, ww, pw, bw = wave_fields(snap['phi'][:],f0,lam2,snap['t'][()],k,l,m)

phih = np.fft.fft2(snap['phi'][:])
phi2h = np.fft.fft2(snap['phi'][:]**2)
phix, phiy = np.fft.ifft2(1j*k*phih), np.fft.ifft2(1j*l*phih)
J_phic_phi = np.conj(phix)*phiy - np.conj(phiy)*phix

qw1 = np.fft.ifft2(-wv2*phi2h).real/(4*f0)
qw2 = (1j*J_phic_phi).real/(2*f0)
qw = qw1+qw2
qpsi = snap['q'][:]-qw

psiL = np.fft.ifft2(-np.fft.fft2(qpsi)*wv2i)
psiE = (psiL + A)

zetaL = qpsi
zetaE = np.fft.ifft2(-wv2*np.fft.fft2(psiE)).real 
zetaS = np.fft.ifft2(wv2*np.fft.fft2(A)).real

psiL, psiE,psiS = psiL/(Ue/ke), psiE/(Ue/ke), -A/(Ue/ke)

zetaL, zetaE, zetaS = zetaL/(Ue*ke), zetaE/(Ue*ke), zetaS/(Ue*ke)

fig = plt.figure(figsize=(8.5,4.5))
cq = np.hstack([np.arange(-3,-.5,.5), np.arange(.5,3.5,.5)])

ax1 = fig.add_subplot(131,aspect=1)
fig.subplots_adjust(wspace=.125)

plt.contourf(x,y,zetaL,cq,cmap=cmocean.cm.curl,extend='both')
plt.contour(x,y,zetaL,cq,colors='k',linewidths=0.75)

plt.xlim(-2,2)
plt.ylim(-2,2)

ax2 = fig.add_subplot(132,aspect=1)
plt.contourf(x,y,zetaE,cq,cmap=cmocean.cm.curl,extend='both')
plt.contour(x,y,zetaE,cq,colors='k', linewidths=0.75)

plt.xlim(-2,2)
plt.ylim(-2,2)

ax3 = fig.add_subplot(133,aspect=1)
plt.contourf(x,y,zetaS,cq,cmap=cmocean.cm.curl,extend='both')
plt.contour(x,y,zetaS,cq,colors='k',linewidths=0.75)

plt.xlim(-2,2)
plt.ylim(-2,2)

ax1.set_xlabel(r"$x\times k_e/2\pi$")
ax1.set_ylabel(r"$y\times k_e/2\pi$")

ax2.set_xlabel(r"$x\times k_e/2\pi$")
ax2.set_yticks([])

ax3.set_xlabel(r"$x\times k_e/2\pi$")
ax3.set_yticks([])

ax1.text(-1.925,1.6,r"$\Delta \psi$",fontsize=14)
ax2.text(-1.925,1.6,r"$\Delta \psi^E$",fontsize=14)
ax3.text(-1.925,1.6,r"$-\Delta \mathcal{A}$",fontsize=14)

plt.savefig("figs/DipoleVorticity.pdf",pad_inches=0, bbox_inches='tight')

fig = plt.figure(figsize=(8.5,4.5))
cq = np.hstack([np.arange(-3,-.5,.5), np.arange(.5,3.5,.5)])

ax1 = fig.add_subplot(131,aspect=1)
fig.subplots_adjust(wspace=.125)

plt.contourf(x,y,psiL,cp,cmap=cmocean.cm.curl,extend='both')
plt.contour(x,y,psiL,cp,colors='k',linewidths=0.75)

plt.xlim(-2,2)
plt.ylim(-2,2)

ax2 = fig.add_subplot(132,aspect=1)
plt.contourf(x,y,psiE,cp,cmap=cmocean.cm.curl,extend='both')
plt.contour(x,y,psiE,cp,colors='k', linewidths=0.75)

plt.xlim(-2,2)
plt.ylim(-2,2)

ax3 = fig.add_subplot(133,aspect=1)
plt.contourf(x,y,psiS-psiS.mean(),cp,cmap=cmocean.cm.curl,extend='both')
plt.contour(x,y,psiS-psiS.mean(),cp,colors='k',linewidths=0.75)

plt.xlim(-2,2)
plt.ylim(-2,2)

ax1.set_xlabel(r"$x\times k_e/2\pi$")
ax1.set_ylabel(r"$y\times k_e/2\pi$")

ax2.set_xlabel(r"$x\times k_e/2\pi$")
ax2.set_yticks([])

ax3.set_xlabel(r"$x\times k_e/2\pi$")
ax3.set_yticks([])

ax1.text(1.65,1.6,r"$\psi$",fontsize=14)
ax2.text(1.55,1.6,r"$\psi^E$",fontsize=14)
ax3.text(1.3,1.6,r"$-\mathcal{A}$",fontsize=14)

plt.savefig("../writeup/figs/DipoleStreamfunction.pdf",pad_inches=0, bbox_inches='tight')
plt.savefig("../writeup/figs/fig12.tiff",pad_inches=0, bbox_inches='tight')




# colorbar
#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.2, 1.01, 0.28, 0.0275])
#fig.colorbar(im1, cax=cbar_ax,label=r"Wave action density $[\mathcal{A} \times 2 f_0 /U_w^2]$",
#                    orientation='horizontal', ticks=[0.,2.,4.],extend='max')
#cbar_ax = fig.add_axes([0.55, 1.01, 0.28, 0.0275])
#fig.colorbar(im2, cax=cbar_ax,label=r"Wave buoyancy $[b/B]$",
#                    orientation='horizontal',ticks=[-.85,0.,.85],
#                    extend='both')


#plt.savefig(patho+"fig1.png", pad_inces=0, bbox_inches='tight', dpi=300)
#plt.savefig(patho+"fig1.eps",dpi=200, pad_inces=0, bbox_inches='tight')
#plt.savefig(patho+"fig1.pdf",dpi=50, pad_inces=0, bbox_inches='tight')
