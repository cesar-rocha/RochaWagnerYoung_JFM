"""
    Figs2Movie
"""
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cmocean

from Utils import *

plt.close('all')

pathi = "outputs/sinxsiny2/"
patho = "figs2movie/"

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

k, l = setup['grid/k'][:], setup['grid/l'][:]
k, l = np.meshgrid(k,l)
wv2 = k**2 + l**2
fnz = wv2 != 0
wv2i = np.zeros_like(wv2)
wv2i[fnz] = 1./wv2[fnz]

#files = ['000000000016667.h5', '000000000900000.h5',
#            '000000002666667.h5', '000000008000000.h5']

#files = ['000000000016667.h5','000000002666667.h5', '000000008000000.h5']
files = ['000000000300000.h5','000000002666667.h5', '000000005200000.h5']
files = glob.glob(pathi[:-1]+"/snapshots/*.h5")


cb = np.arange(-5.,5.25,0.25)
cp = np.arange(-5,5,.5)
cq = np.arange(-2.5,2.75,.5)
ticks = [0,0.5,1.]

files = ['000000000000000.h5','000000020000000.h5','000000050000000.h5','000000120000000.h5']
for fni in files:
#fni = files[-1]
    snap = h5py.File(pathi+"snapshots/"+fni)
    t = snap['t'][()]/Te
    q = snap['q'][:]
    qh = np.fft.fft2(q)
    p = np.fft.ifft2(-wv2i*qh).real
    p = p*ke/Ue
    q = q/(Ue*ke)
    phi2 = np.abs(snap['phi'])**2/Uw**2
    #phi2 = np.abs(snap['c'])**2/Uw**2 # passive scalar
    #bw = snap['c'] # passive scalar
    uw, vw, ww, pw, bw = wave_fields(snap['phi'][:],f0,lam2,snap['t'][()],k,l,m)

    plt.figure()
    plt.contourf(x,y, bw/Uw/m/(f0*lam2)/ke,cb,vmin=cb.min(),vmax=cb.max(),
                         cmap = cmocean.cm.balance, shading='flat',extend='both')
    #plt.contourf(x,y, bw,cb,vmin=cb.min(),vmax=cb.max(),
    #                     cmap = cmocean.cm.balance, shading='flat',extend='both')
    plt.contour(x,y,p,cp,colors='k')
    #plt.contour(x,y,q,cq,colors='k')

    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.ylabel(r"$y\times k_e/2\pi$")
    plt.xlabel(r"$x\times k_e/2\pi$")
    tit = r"$t\times U_e k_e = %3.2f $" %(t)
    plt.title(tit)
    #plt.savefig(patho+"pb"+fni[-18:]+".png",dpi=100)
    #plt.close('all')
