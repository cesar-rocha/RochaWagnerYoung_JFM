"""
    Plots figure figure 2
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
lam = 2*np.pi/params['dimensional/m'][()]
h = f0*(lam**2)
## get grid
setup = h5py.File(pathi+"setup.h5","r")
x, y = setup['grid/x'][:]*ke/(2*np.pi), setup['grid/y'][:]*ke/(2*np.pi)
x -= x.mean()
y -= y.mean()

#files = ['000000000016667.h5', '000000001333333.h5',
#            '000000002666667.h5', '000000008000000.h5']

files = ['000000000600000.h5','000000002666667.h5']

#plot_fig_label(ax, xc =0.775, yc=1.03, label= r"$t \times U_e k_e$ = "+
#                    str(int(round(t))), facecolor="1.0",boxstyle=None,alpha=0.)

cphi = np.arange(0.,5.,0.1)
cq   = np.array([-1.5,-0.5,0.5,1.5])
xlim = [-2.5,2.5]


k,l = setup['grid/k'][:],setup['grid/l'][:]
k,l = np.meshgrid(k,l)
wv2 = k**2 + l**2

snap = h5py.File(pathi+"snapshots/"+files[0])
t = snap['t'][()]/Te
q = snap['q'][:] #/(Ue*ke)
phi = snap['phi'] #/Uw**2
phih = np.fft.fft2(phi)
phi2 = np.abs(snap['phi'])**2 #/Uw**2
phi2h = np.fft.fft2(phi2)
phix, phiy = np.fft.ifft2(1j*k*phih), np.fft.ifft2(1j*l*phih)
J_phic_phi = np.conj(phix)*phiy - np.conj(phiy)*phix

qw1 = np.fft.ifft2(-wv2*phi2h).real/(4*f0)
qw2 = (1j*J_phic_phi).real/(2*f0)
qw = qw1+qw2
qpsi = q-qw

Fscale = 1e-5
Fwx, Fwy = (0.5*h*np.conj(phi)*phix).imag/Fscale,  (0.5*h*np.conj(phi)*phiy).imag/Fscale


Fwx, Fwy = np.ma.masked_array(Fwx,np.abs(Fwx)<1.5e-3), np.ma.masked_array(Fwy,np.abs(Fwx)<1.5e-3)

dec = 10

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111,aspect=1)

plt.contour(x,y,qpsi/(Ue*ke),cq[2:],colors='k')
plt.contour(x,y,qpsi/(Ue*ke),cq[:2],colors='k')
Q = plt.quiver(x[::dec,::dec],y[::dec,::dec],Fwx[::dec,::dec],Fwy[::dec,::dec],scale=50,width=0.006)
qk = plt.quiverkey(Q, 0.2, 0.875, 0.4, r'$\mathbf{F}_w$', labelpos='E',
                   coordinates='figure')

plt.xlim(xlim)
plt.ylim(xlim)

#plt.text(.7,.95,r'$\Delta\psi > 0$',rotation=-45)
#plt.text(.725,-.7,r'$\Delta\psi < 0$',rotation=45)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.savefig(patho+"Gamma_r.pdf",
            bbox_inches='tight',
            pad_inches=0)

# colorbar
#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.21, 0.0275, 0.55])
#fig.colorbar(im, cax=cbar_ax,label=r"Wave kinetic energy density $[|\phi|^2/U_w^2]$")
#plt.savefig(patho+"fig1.png", bbox_inches='tight')

