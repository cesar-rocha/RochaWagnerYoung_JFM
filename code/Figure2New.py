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

#files = ['000000000016667.h5', '000000001333333.h5',
#            '000000002666667.h5', '000000008000000.h5']

files = ['000000000300000.h5','000000005200000.h5']
files = ['000000000300000.h5','000000002666667.h5', '000000005200000.h5']
#files = ['000000000300000.h5','000000001500000.h5', '000000005200000.h5']

#plot_fig_label(ax, xc =0.775, yc=1.03, label= r"$t \times U_e k_e$ = "+
#                    str(int(round(t))), facecolor="1.0",boxstyle=None,alpha=0.)
fig = plt.figure(figsize=(8.5,5))

for i in range(2):

    cphi = np.arange(0.,5.,0.1)
    cp   = np.array([-4,-2.,0.,2.,4])
    cq0 = np.array([-1.5,-.5,.5,1.5])
    cq = np.hstack([np.arange(-2.,0,.25), np.arange(.25,2.25,.25)])
    xlim = [-1.5,1.5]

    k,l = setup['grid/k'][:],setup['grid/l'][:]
    k,l = np.meshgrid(k,l)
    wv2 = k**2 + l**2
    wv2i = 1./wv2
    wv2i[0,0] = 0

    snap = h5py.File(pathi+"snapshots/"+files[i])
    t = snap['t'][()]/Te
    q = snap['q'][:] #/(Ue*ke)
    qh = np.fft.fft2(q)
    ph = -wv2i*qh

    pxy = np.fft.ifft2(-k*l*ph).real
    pxx, pyy = np.fft.ifft2(-k*k*ph).real, np.fft.ifft2(-l*l*ph).real
    alpha = np.sqrt(pxy**2 + 0.5*((pxx-pyy)**2))


    phi = snap['phi'] #/Uw**2
    phih = np.fft.fft2(phi)
    phi2 = np.abs(snap['phi'])**2 #/Uw**2
    phi2h = np.fft.fft2(phi2)
    phix, phiy = np.fft.ifft2(1j*k*phih), np.fft.ifft2(1j*l*phih)
    J_phic_phi = np.conj(phix)*phiy - np.conj(phiy)*phix

    gradphi2 = np.abs(phix)**2 + np.abs(phiy)**2

    a, b = -pxy, 0.5*(pxx-pyy)
    Ga =  a*( np.abs(phiy)**2 - np.abs(phix)**2 ) + 2*b*np.real(np.conj(phiy)*phix)

    qw1 = np.fft.ifft2(-wv2*phi2h).real/(4*f0)
    qw2 = (1j*J_phic_phi).real/(2*f0)
    qw = qw1+qw2
    qpsi = q-qw

    Fscale = 1e-5
    Fwx, Fwy = (0.5*h*np.conj(phi)*phix).imag/Fscale,\
                            (0.5*h*np.conj(phi)*phiy).imag/Fscale
    Fwx, Fwy = np.ma.masked_array(Fwx,np.abs(Fwx)<1.5e-3),\
                            np.ma.masked_array(Fwy,np.abs(Fwx)<1.5e-3)

    OW =  alpha**2 - qpsi**2
    uw, vw, ww, pw, bw = wave_fields(phi,f0,lam2,snap['t'][()],k,l,m)


    phix_r, phiy_r =  np.real(phix), np.real(phiy)
    # calculate unaveraged Gamma_a
    dec = 6

    p = np.fft.ifft2(ph).real

    ax = fig.add_subplot(1,2,i+1,aspect=1)
    fig.subplots_adjust(wspace=.045)

    if i == 1:
        plt.contour(x,y,p/(Ue/ke),cp[2:],colors='k')
        plt.contour(x,y,p/(Ue/ke),cp[:2],colors='k')
        cga = np.arange(-.85,.9,.05)
        pc = plt.contourf(x,y,Ga/2.5e-15,cga,vmin=-.75,vmax=.75,
                    cmap=cmocean.cm.balance,extend='both')
    else:
        plt.contour(x,y,qpsi/(Ue*ke),cq0[2:],colors='k')
        plt.contour(x,y,qpsi/(Ue*ke),cq0[:2],colors='k')
        Q = plt.quiver(x[::dec,::dec],y[::dec,::dec],Fwx[::dec,::dec],Fwy[::dec,::dec],
                        scale=50,width=0.006)
        qk = plt.quiverkey(Q, 0.175,.835, 4, r'$\mathbf{F}_w$', labelpos='E',
                       coordinates='figure')

    plt.xlim(xlim)
    plt.ylim(xlim)
    ax.set_xlabel(r"$x\times k_e/2\pi$")
    ax.set_xticks([-1.,0,1])

    if i == 0:
        ax.set_ylabel(r"$y\times k_e/2\pi$")
        ax.set_yticks([-1.,0,1])
    elif i == 1 or i == 2:
        ax.set_yticks([])
    else:
        pass

    plot_fig_label(ax, xc =0.85, yc=1.025, label= r"$t \times U_e k_e$ = "+
                    str(int(round(t))), facecolor="1.0",boxstyle=None,alpha=0.)

cbar_ax = fig.add_axes([.925, 0.2,  0.0275,0.58,])
fig.colorbar(pc, cax=cbar_ax,label=r"Advective conversion \
                        ${\nabla\phi}^T\mathbf{S}\nabla\phi^\star$",
                    orientation='vertical',ticks=[-.85,0.,.85],
                    extend='both')
plt.savefig(patho+"perhapsfig2.png", pad_inces=0, bbox_inches='tight')

# ax = fig.add_subplot(122,aspect=1)
#
#
# a = np.ma.masked_where(OW < 0.0, OW)
# plt.contour(x,y,np.fft.ifft2(ph),colors='k')
# #plt.contour(x,y,a,colors='k')
#
# plt.xlim(xlim)
# plt.ylim(xlim)

#plt.text(.7,.95,r'$\Delta\psi > 0$',rotation=-45)
#plt.text(.725,-.7,r'$\Delta\psi < 0$',rotation=45)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)

#plt.savefig(patho+"Gamma_r.png",
#            bbox_inches='tight',
#            pad_inches=0)

# colorbar
#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.21, 0.0275, 0.55])
#fig.colorbar(im, cax=cbar_ax,label=r"Wave kinetic energy density $[|\phi|^2/U_w^2]$")
#plt.savefig(patho+"fig1.png", bbox_inches='tight')
